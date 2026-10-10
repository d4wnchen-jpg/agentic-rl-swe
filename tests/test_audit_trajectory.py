#!/usr/bin/env python3
"""audit_trajectory.py 的回归测试（纯 CPU，不读真实数据）

核心是 audit_solved()：从模型自己跑的 pytest 输出里判断
「曾经修好过吗」。分类错了会直接导致错误的结论，所以逐类都测。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_trajectory import audit_solved  # noqa: E402

F2P = ["tests/test_a.py::t1", "tests/test_a.py::t2"]
TASKS = {"task1": {"FAIL_TO_PASS": F2P}}


def mk(turns, obs_texts, resolved=False, nturns=None, iid="task1"):
    """造一条 rollout：turns 个 assistant 轮 + 对应的 user 观察"""
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "fix"}]
    for i in range(turns):
        msgs.append({"role": "assistant",
                     "content": "<function=bash><parameter=command>"
                                + ("python -m pytest tests/ -rA" if i == 0 else "ls")
                                + "</parameter></function>"})
        body = obs_texts[i] if i < len(obs_texts) else f"ok {i}"
        msgs.append({"role": "user", "content": "OBSERVATION:\n" + body})
    return {"instance_id": iid, "resolved": resolved,
            "nturns": nturns if nturns is not None else turns,
            "messages": msgs}


ALL_PASS = "\n".join(f"PASSED {n}" for n in F2P)
ONE_PASS = f"PASSED {F2P[0]}"
ALL_FAIL = "\n".join(f"FAILED {n}" for n in F2P)


# ═══════════════════════════════════════════════════════════
def test_ever_solved_but_final_fail():
    """⭐ 曾经全通过、但最终判分失败 —— 这是最关键的信号"""
    r = mk(10, [ALL_PASS], resolved=False)
    out = audit_solved({"task1": r}, TASKS, "t")
    assert out["res"]["曾经全通过"] == 1
    assert out["ever_ok_resolved"][False] == 1
    assert out["ever_ok_resolved"].get(True, 0) == 0


def test_ever_solved_and_final_pass():
    r = mk(10, [ALL_PASS], resolved=True)
    out = audit_solved({"task1": r}, TASKS, "t")
    assert out["res"]["曾经全通过"] == 1
    assert out["ever_ok_resolved"][True] == 1


def test_partial_coverage_counts_as_partial():
    """只覆盖了部分 F2P（可能是 -k 过滤或输出被截断）"""
    r = mk(10, [ONE_PASS], resolved=False)
    out = audit_solved({"task1": r}, TASKS, "t")
    assert out["res"]["部分通过（覆盖不全）"] == 1
    assert out["res"]["曾经全通过"] == 0


def test_ran_test_but_never_passed():
    r = mk(10, [ALL_FAIL], resolved=False)
    out = audit_solved({"task1": r}, TASKS, "t")
    assert out["res"]["跑过测试但从没通过"] == 1


def test_ran_pytest_without_f2p_nodes():
    """跑了 pytest，但输出里一个 F2P 节点都没有（比如跑了别的文件）"""
    r = mk(10, ["PASSED tests/other.py::x"], resolved=False)
    out = audit_solved({"task1": r}, TASKS, "t")
    assert out["res"]["跑过pytest但输出无F2P"] == 1


def test_never_ran_pytest():
    """整条轨迹没有 pytest 命令"""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "f"}]
    for i in range(5):
        msgs.append({"role": "assistant",
                     "content": "<function=bash><parameter=command>ls</parameter></function>"})
        msgs.append({"role": "user", "content": "OBSERVATION:\nok"})
    r = {"instance_id": "task1", "resolved": False, "nturns": 5, "messages": msgs}
    out = audit_solved({"task1": r}, TASKS, "t")
    assert out["res"]["从没跑过测试"] == 1


def test_continue_after_solved():
    """⭐ 全通过之后还继续跑了 >3 轮"""
    r = mk(12, [ALL_PASS], resolved=False)      # 第 1 轮就全通过，总共 12 轮
    out = audit_solved({"task1": r}, TASKS, "t")
    assert len(out["cont"]) == 1
    assert out["cont"][0] == 11                 # 12 - 1


def test_no_continue_when_immediate():
    """全通过后立刻结束 —— 不算「继续跑」"""
    r = mk(2, [ALL_PASS], resolved=True)        # 第 1 轮全通过，总共 2 轮
    out = audit_solved({"task1": r}, TASKS, "t")
    assert out["res"]["曾经全通过"] == 1
    assert out["cont"] == []


def test_missing_task_definition_is_skipped():
    """rollout 的 instance_id 不在 eval.jsonl 里 → 跳过，不误分类"""
    r = mk(5, [ALL_PASS], iid="not_in_tasks")
    out = audit_solved({"not_in_tasks": r}, TASKS, "t")
    assert out["res"]["无任务定义"] == 1
    assert out["res"]["曾经全通过"] == 0


def test_xfail_counts_as_passed():
    """XFAIL 应算通过（和 evaluate() 的判据一致）"""
    obs = f"PASSED {F2P[0]}\nXFAIL {F2P[1]}"
    r = mk(5, [obs], resolved=True)
    out = audit_solved({"task1": r}, TASKS, "t")
    assert out["res"]["曾经全通过"] == 1


def test_empty_input():
    out = audit_solved({}, TASKS, "t")
    assert out["n"] == 0
    assert sum(out["res"].values()) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
