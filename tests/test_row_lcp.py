#!/usr/bin/env python3
"""build_row_lcp 的回归测试（纯 CPU，不需要 torch / GPU）

为什么需要这个测试：
  LCP 对齐是"多轮 rollout 拼成一条训练行"的核心。如果它悄悄退化
  （比如某个 response 丢失、mask 错位、old_logprob 没对上），
  训练照跑不误，但学的是错的东西 —— 属于静默失败。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from grpo_util import _find_subseq, _lcp, build_row_lcp  # noqa: E402


# ═══════════════════════════════════════════════════════════
# 基础工具
# ═══════════════════════════════════════════════════════════
class TestLcp:
    def test_identical(self):
        assert _lcp([1, 2, 3], [1, 2, 3]) == 3

    def test_diverges(self):
        assert _lcp([1, 2, 3], [1, 9, 3]) == 1

    def test_one_is_prefix(self):
        assert _lcp([1, 2], [1, 2, 3, 4]) == 2

    def test_empty(self):
        assert _lcp([], [1, 2]) == 0


class TestFindSubseq:
    def test_found(self):
        assert _find_subseq([1, 2, 3, 4], [3, 4]) == 2

    def test_from_start(self):
        assert _find_subseq([9, 9, 3, 4], [3, 4], start=1) == 2

    def test_not_found(self):
        assert _find_subseq([1, 2, 3], [7, 8]) == -1

    def test_needle_longer_than_hay(self):
        assert _find_subseq([1], [1, 2]) == -1


# ═══════════════════════════════════════════════════════════
# 拼行
# ═══════════════════════════════════════════════════════════
def _turns():
    """模拟 3 轮：P0 与 P1 之间被 chat template 改写了 2 个 token

    P0 = [1..8] + [9,10]              （9,10 是生成提示符）
    R0 = [100,101,102]
    P1 = [1..8] + R0 + [200,201] + [9,10]     历史渲染少了 P0 的 9,10
    """
    P0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    R0 = [100, 101, 102]
    L0 = [-1.0, -2.0, -3.0]
    P1 = [1, 2, 3, 4, 5, 6, 7, 8] + R0 + [200, 201] + [9, 10]
    R1 = [110, 111]
    L1 = [-4.0, -5.0]
    P2 = [1, 2, 3, 4, 5, 6, 7, 8] + R0 + [200, 201] + R1 + [210] + [9, 10]
    R2 = [120, 121, 122]
    L2 = [-6.0, -7.0, -8.0]
    return [[P0, R0, L0], [P1, R1, L1], [P2, R2, L2]], ([R0, R1, R2], [L0, L1, L2])


class TestBuildRowLcp:
    def test_returns_three_values(self):
        turns, _ = _turns()
        out = build_row_lcp(turns)
        assert out is not None
        row, mask, olp = out
        assert len(row) == len(mask) == len(olp)

    def test_all_responses_present(self):
        """每个 response 都必须出现在最终行里（否则整轮训练信号丢失）"""
        turns, (resp, _) = _turns()
        row, _, _ = build_row_lcp(turns)
        for i, r in enumerate(resp):
            assert any(
                row[j:j + len(r)] == r for j in range(len(row) - len(r) + 1)
            ), f"response{i} 丢失"

    def test_mask_marks_generated_tokens(self):
        """mask=1 的位置必须正好是 response 覆盖的位置"""
        turns, (resp, _) = _turns()
        row, mask, _ = build_row_lcp(turns)
        for r in resp:
            for j in range(len(row) - len(r) + 1):
                if row[j:j + len(r)] == r:
                    assert all(mask[j + k] == 1 for k in range(len(r))), \
                        f"response {r} 的 mask 没标上"
                    break

    def test_old_logprobs_aligned(self):
        """old_logprobs 必须跟着 response 一起搬移，不能错位"""
        turns, (resp, lps) = _turns()
        row, mask, olp = build_row_lcp(turns)
        for r, lp in zip(resp, lps):
            for j in range(len(row) - len(r) + 1):
                if row[j:j + len(r)] == r:
                    assert olp[j:j + len(r)] == lp, f"logprob 错位: {olp[j:j+len(r)]} != {lp}"
                    break

    def test_no_mask_without_logprob(self):
        """mask=1 的位置 old_logprob 不应为 0（0 是填充值）"""
        turns, _ = _turns()
        row, mask, olp = build_row_lcp(turns)
        bad = [i for i in range(len(row)) if mask[i] == 1 and olp[i] == 0.0]
        assert not bad, f"mask=1 但 logprob=0 的位置: {bad}"

    def test_template_rewrite_is_dropped(self):
        """被 chat template 改写的那 2 个 token（P0 的 9,10）应被丢掉"""
        turns, _ = _turns()
        row, _, _ = build_row_lcp(turns)
        # row 开头应是 P0[:8] + R0，而不是 P0 + R0
        assert row[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert row[8:11] == [100, 101, 102]

    def test_empty_input(self):
        assert build_row_lcp([]) is None

    def test_missing_response(self):
        assert build_row_lcp([[ [1, 2, 3], [], []]]) is None


class TestTruncation:
    def test_respects_max_tokens(self):
        turns = []
        for i in range(50):
            P = [1, 2, 3] + list(range(1000 + i * 10, 1000 + i * 10 + 40))
            R = [9000 + i]
            turns.append([P, R, [-1.0]])
        out = build_row_lcp(turns, max_tokens=200)
        assert out is not None
        row, mask, olp = out
        assert len(row) == len(mask) == len(olp)
        # 保留头（system+题面）+ 尾部，总长不应超过 max_tokens 太多
        assert len(row) <= 200 + 3

    def test_head_is_preserved(self):
        """截断后 system/题面（P0）必须还在"""
        turns = []
        for i in range(50):
            P = [11, 22, 33] + list(range(1000 + i * 10, 1000 + i * 10 + 40))
            R = [9000 + i]
            turns.append([P, R, [-1.0]])
        row, _, _ = build_row_lcp(turns, max_tokens=120)
        assert row[:3] == [11, 22, 33]

    def test_returns_none_when_head_too_long(self):
        P0 = list(range(1000))
        out = build_row_lcp([[P0, [1, 2, 3], [-1.0, -1.0, -1.0]]], max_tokens=100)
        assert out is None


class TestLogprobPadding:
    def test_missing_logprobs_are_padded(self):
        """没记录 logprob 时不应崩，应补 0"""
        P0, R0 = [1, 2, 3], [10, 11]
        row, mask, olp = build_row_lcp([[P0, R0, []]])
        assert len(olp) == len(row)
        assert sum(mask) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
