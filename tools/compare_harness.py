#!/usr/bin/env python3
"""compare_harness.py — 对比两套 harness 配置下的【行为指标】

用途：改了提示词 / 环境之后，判断模型的行为有没有按预期变化。
不需要 GPU，只读已经落盘的 rollout 明细。

    用法：
        python tools/compare_harness.py \
            --a eval_base40_detail.jsonl   --label-a "V1 旧提示词" \
            --b eval_base_v2_detail.jsonl  --label-b "V2 新提示词"

为什么看行为而不是通过率：
    113 道题的通过率噪声约 ±6pp，20 步训练的效应测不出来。
    但"提交率""写脚本的比例"这类行为分布，样本效率高得多。

关键指标：
    ⭐ 写脚本的比例   —— 新提示词要引导的行为（构造最小复现）
    ⭐ 主动提交的比例 —— 旧配置下只有 1/113（拿不到完成信号）
    ⭐ 跑 pytest 的比例
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agl_util import parse_test_statuses  # noqa: E402

ACTION = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
CMD = re.compile(r"<parameter=command>(.*?)</parameter>", re.S)

PYTEST = re.compile(r"\bpytest\b")
PY_INLINE = re.compile(r"python[0-9.]*\s+-(?:c|<<)\b|\bpython[0-9.]*\s*<<")
PY_FILE = re.compile(r"python[0-9.]*\s+[\w./-]+\.py\b")
HEREDOC = re.compile(r"(?:cat|tee)\s*>[^\n]*<<|<<\s*'?[A-Z_]{2,}'?")
EDIT_CMD = re.compile(r"<parameter=command>\s*(create|str_replace|insert)\s*</parameter>")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "instance_id" in o and "messages" in o:
            out.append(o)
    return out


def digest(rows: list[dict]) -> dict:
    """从一组 rollout 里提取行为指标"""
    n = max(len(rows), 1)
    d: dict = {
        "n": len(rows),
        "resolved": 0,
        "submitted": 0,
        "ran_pytest": 0,
        "wrote_script": 0,
        "edited": 0,
        "edits": 0,
        "pytest_cmds": 0,
        "script_cmds": 0,
        "turns": [],
        "stop": Counter(),
    }
    for r in rows:
        if r.get("resolved"):
            d["resolved"] += 1
        d["turns"].append(r.get("nturns") or 0)
        d["stop"][r.get("stop_reason") or "?"] += 1

        ran_py = wrote = edited = False
        for m in r.get("messages", []):
            if m.get("role") != "assistant":
                continue
            c = str(m.get("content", ""))
            if EDIT_CMD.search(c):
                d["edits"] += 1
                edited = True
            for mm in ACTION.finditer(c):
                name, body = mm.group(1), mm.group(2)
                if name == "submit":
                    d["submitted"] += 1
                if name != "bash":
                    continue
                cmd = " ".join((CMD.search(body).group(1) if CMD.search(body)
                                else body).split())
                if PYTEST.search(cmd):
                    d["pytest_cmds"] += 1
                    ran_py = True
                if PY_INLINE.search(cmd) or PY_FILE.search(cmd) or HEREDOC.search(cmd):
                    d["script_cmds"] += 1
                    wrote = True
        d["ran_pytest"] += ran_py
        d["wrote_script"] += wrote
        d["edited"] += edited
    return d


def pct(x: int, n: int) -> str:
    return f"{x}/{n} ({x / max(n, 1) * 100:.0f}%)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    a = ap.parse_args()

    A = digest(load(Path(a.a)))
    B = digest(load(Path(a.b)))
    if not A["n"] and not B["n"]:
        print("两个文件都读不到。")
        return 1

    la, lb = a.label_a, a.label_b
    w = max(len(la), len(lb), 12)

    print("=" * 76)
    print(f"  {la}  vs  {lb}")
    print("=" * 76)
    print(f"  {'指标':<26}{la:>{w+10}}{lb:>{w+10}}")
    print("  " + "-" * 72)

    rows = [
        ("rollout 数", lambda d: str(d["n"])),
        ("通过率（次要）", lambda d: pct(d["resolved"], d["n"])),
        ("⭐ 主动提交", lambda d: pct(d["submitted"], d["n"])),
        ("⭐ 写脚本 / 内联 python", lambda d: pct(d["wrote_script"], d["n"])),
        ("跑过 pytest", lambda d: pct(d["ran_pytest"], d["n"])),
        ("编辑过文件", lambda d: pct(d["edited"], d["n"])),
        ("平均编辑次数", lambda d: f"{d['edits'] / max(d['n'], 1):.1f}/条"),
        ("pytest 命令总数", lambda d: str(d["pytest_cmds"])),
        ("脚本类命令总数", lambda d: str(d["script_cmds"])),
        ("平均轮数", lambda d: f"{sum(d['turns']) / max(d['n'], 1):.1f}"),
    ]
    for name, f in rows:
        print(f"  {name:<26}{f(A):>{w+10}}{f(B):>{w+10}}")

    print()
    print("  结束原因分布")
    keys = sorted(set(A["stop"]) | set(B["stop"]))
    for k in keys:
        print(f"    {k:<24}{A['stop'].get(k, 0):>{w+8}}{B['stop'].get(k, 0):>{w+8}}")

    print()
    print("=" * 76)
    print("怎么读")
    print("=" * 76)
    print("  ⭐ 主动提交大幅上升   → 模型拿到了完成信号，提示词修复有效")
    print("  ⭐ 写脚本比例上升     → 新提示词的 5 步工作流被遵循了")
    print("  ▲ pytest 命令下降    → 模型不再依赖现成测试")
    print("  通过率几乎不动是正常的 —— 113 道题的噪声就有 ±6pp")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
