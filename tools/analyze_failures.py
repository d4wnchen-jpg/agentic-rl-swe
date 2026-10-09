#!/usr/bin/env python3
"""analyze_failures.py — 分析 rollout 的失败模式

回答一个问题：**瓶颈在模型还是在 harness？**

  失败主要是「轮数耗尽」/「测试跑不起来」  ->  瓶颈在 harness（提示词、轮数预算）
  失败主要是「改了代码但测试没过」          ->  瓶颈在模型能力 -> 训练才有用

用法：
    python tools/analyze_failures.py [结果文件] [轮数上限]

默认：$SWESMITH_RL_HOME/probe_par_result.jsonl，轮数上限 25（探测阶段的配置）

输入每行一个 JSON（由 probe_par.py / grpo_rollout.py 落盘）：
    {"instance_id": ..., "resolved": bool, "n_fail": int,
     "n_missing": int, "nturns": int, "messages": [...]}
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = os.environ.get("SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1]))

_CMD_RE = re.compile(
    r"<function=bash>\s*<parameter=command>(.*?)</parameter>", re.S)
_ACTION_RE = re.compile(r"<function=(\w+)>")
_EDIT_RE = re.compile(r"<parameter=path>(.*?)</parameter>", re.S)
_PYTEST_RE = re.compile(r"\bpytest\b")


# ═══════════════════════════════════════════════════════════
def load(path: Path) -> list[dict]:
    """读取 rollout 明细（跳过任务行和不完整的行）"""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        # 只保留 rollout 结果行（任务行没有 resolved 字段）
        if "resolved" in o and "instance_id" in o:
            out.append(o)
    return out


def classify(r: dict, max_turns: int) -> str:
    """按优先级判定失败模式（互斥）"""
    if r["resolved"]:
        return "成功"
    if r.get("nturns", 0) >= max_turns:
        return "轮数耗尽"
    if r.get("n_missing", 0) > 0:
        return "测试跑不起来"
    # ⭐ 区分「没修好」和「修坏了」
    fb, pb = r.get("n_f2p_bad"), r.get("n_p2p_bad")
    if fb is not None and pb is not None:
        if fb > 0 and pb > 0:
            return "没修好且弄坏了别的"
        if pb > 0:
            return "改坏了其他测试"
        if fb > 0:
            return "没修好"
    if r.get("n_fail", 0) > 0:
        return "改了但没改对"
    return "提前结束"


def trajectory(r: dict) -> dict:
    """从 messages 里提取轨迹特征（回答"为什么失败"）"""
    msgs = r.get("messages", [])
    edits, bashes, ran_pytest = [], 0, 0
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        c = str(m.get("content", ""))
        for mm in _EDIT_RE.finditer(c):
            edits.append(mm.group(1))
        bashes += len(_CMD_RE.findall(c))
        if _PYTEST_RE.search(c):
            ran_pytest += 1
    return {
        "n_edit": len(edits),
        "n_bash": bashes,
        "files_edited": len(set(edits)),
        "ran_pytest": ran_pytest,
    }


def commands_of(r: dict) -> list[str]:
    """从 messages 里抽出所有 bash 命令"""
    cmds = []
    for m in r.get("messages", []):
        if m.get("role") != "assistant":
            continue
        for mm in _CMD_RE.finditer(str(m.get("content", ""))):
            cmds.append(" ".join(mm.group(1).split()))
    return cmds


def last_actions(r: dict, n: int = 3) -> list[str]:
    """最后 n 条 assistant 消息（截断），用于人工抽样"""
    outs = [str(m.get("content", ""))
            for m in r.get("messages", []) if m.get("role") == "assistant"]
    return [" ".join(x.split())[:160] for x in outs[-n:]]


def repo_of(iid: str, row: dict | None = None) -> str:
    # 明细里若带 repo 字段，优先用它
    if row and row.get("repo"):
        return str(row["repo"]).split("/")[-1].split(".")[0]
    """从 instance_id 里取仓库名

    兼容几种已知写法：
        mistune__func_basic__07jpr8n3   -> mistune
        swesmith/mistune.abc123         -> mistune
        mistune.deadbeef.0000           -> mistune
    取不到就返回前 12 个字符（并在 ④ 里显示出来，便于发现格式不符）
    """
    s = str(iid).split("/")[-1]
    s = s.split("__")[0]          # 第一段
    s = s.split(".")[0]           # 去掉哈希后缀
    return s[:12] if s else "?"


def bar(n: int, total: int, width: int = 30) -> str:
    k = int(round(width * n / total)) if total else 0
    return "█" * k + "·" * (width - k)


# ═══════════════════════════════════════════════════════════
def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else f"{BASE}/probe_par_result.jsonl")
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    if not path.exists():
        print(f"找不到 {path}")
        return 1

    rows = load(path)
    if not rows:
        print(f"{path} 里没有可用的 rollout 行")
        return 1

    N = len(rows)
    ok = sum(1 for r in rows if r["resolved"])

    print("=" * 68)
    print(f"结果文件   {path}")
    print(f"轮数上限   {max_turns}")
    print(f"rollout    {N} 条   解决 {ok} ({ok/N*100:.1f}%)   未解决 {N-ok}")
    print("=" * 68)

    # ── ① 失败模式（互斥，按优先级）──────────────────────
    print("\n【① 失败模式】（按优先级，互斥）\n")
    cls = Counter(classify(r, max_turns) for r in rows)
    for k, v in cls.most_common():
        print(f"  {k:14} {v:5}  {v/N*100:5.1f}%  {bar(v, N)}")

    fail_total = N - ok
    print(f"\n  未解决 {fail_total} 条中的构成：")
    for k, v in cls.most_common():
        if k == "成功":
            continue
        print(f"    {k:14} {v:5}  {v/max(fail_total,1)*100:5.1f}%  {bar(v, fail_total)}")

    # ── ② 独立指标（不互斥）──────────────────────────────
    print("\n【② 独立指标】（一条 rollout 可能同时命中多项）\n")
    ind = {
        "轮数用满": sum(1 for r in rows if r.get("nturns", 0) >= max_turns),
        "有测试缺失(n_missing>0)": sum(1 for r in rows if r.get("n_missing", 0) > 0),
        "有测试失败(n_fail>0)": sum(1 for r in rows if r.get("n_fail", 0) > 0),
    }
    for k, v in ind.items():
        print(f"  {k:26} {v:5}  {v/N*100:5.1f}%")

    # ── ③ 轮数分布 ──────────────────────────────────────
    print("\n【③ 轮数分布】\n")
    for label, sel in (("成功的", [r for r in rows if r["resolved"]]),
                       ("失败的", [r for r in rows if not r["resolved"]])):
        if not sel:
            continue
        ts = sorted(r.get("nturns", 0) for r in sel)
        avg = sum(ts) / len(ts)
        hit = sum(1 for t in ts if t >= max_turns)
        print(f"  {label} {len(sel):4} 条   平均 {avg:5.1f} 轮   "
              f"中位 {ts[len(ts)//2]:3}   用满 {hit} ({hit/len(ts)*100:.0f}%)")

    # ── ④ 按仓库 ────────────────────────────────────────
    print("\n【④ 按仓库】\n")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        b = by[repo_of(r["instance_id"], r)]
        b[1] += 1
        b[0] += int(r["resolved"])
    if len(by) > 20:
        print(f"  ⚠️ 解析出 {len(by)} 个仓库名 —— 可能 instance_id 格式和预期不符")
        print(f"     最常见的几个：")
        for k, (o, t) in sorted(by.items(), key=lambda kv: -kv[1][1])[:8]:
            print(f"       {k:20} {o}/{t}")
        print()
    for k in sorted(by):
        o, t = by[k]
        print(f"  {k:14} {o:4}/{t:4}  {o/t*100:5.1f}%  {bar(o, t)}")

    # ── ⑤ 重复动作 ──────────────────────────────────────
    print("\n【⑤ 重复动作】（数同一 rollout 里重复次数最多的命令）\n")
    rep_hist = Counter()
    for r in rows:
        cs = commands_of(r)
        if not cs:
            continue
        m = max(Counter(cs).values())
        rep_hist[min(m, 8)] += 1
    tot_with_cmd = sum(rep_hist.values())
    for k in sorted(rep_hist):
        label = f"{k} 次" if k < 8 else "8+ 次"
        v = rep_hist[k]
        print(f"  最多重复 {label:6} {v:5}  {v/max(tot_with_cmd,1)*100:5.1f}%  "
              f"{bar(v, max(tot_with_cmd,1))}")

    # ── ⑥ 抽样 ──────────────────────────────────────────
    print("\n【⑥ 典型失败抽样】（每类抽 2 条，打印最后 3 个动作）\n")
    for want in ("轮数耗尽", "测试跑不起来", "改了但没改对", "提前结束"):
        picks = [r for r in rows if classify(r, max_turns) == want][:2]
        for r in picks:
            print(f"  ── [{want}] {r['instance_id']}")
            print(f"     nturns={r.get('nturns')}  n_fail={r.get('n_fail')}  "
                  f"n_missing={r.get('n_missing')}  命令数={len(commands_of(r))}")
            for a in last_actions(r, 3):
                print(f"     · {a}")
            print()

    # ── ⑦ 轨迹特征（回答"为什么失败"）──────────────────
    print("\n【⑦ 轨迹特征】（从 messages 里提取，各失败类型对比）\n")
    print(f"  {'类型':22} {'条数':>5} {'编辑次数':>8} {'改的文件':>8} {'跑过pytest':>10}")
    print("  " + "-" * 60)
    buckets = defaultdict(list)
    for r in rows:
        buckets[classify(r, max_turns)].append(r)
    for k in sorted(buckets, key=lambda x: (x == "成功", x)):
        g = buckets[k]
        tr = [trajectory(r) for r in g]
        if not tr:
            continue
        avg_e = sum(t["n_edit"] for t in tr) / len(tr)
        avg_f = sum(t["files_edited"] for t in tr) / len(tr)
        ran = sum(1 for t in tr if t["ran_pytest"])
        print(f"  {k:22} {len(g):>5} {avg_e:>8.1f} {avg_f:>8.1f} "
              f"{ran:>6}/{len(g)} ({ran/len(g)*100:>3.0f}%)")

    # 判读
    fails = [r for r in rows if not r["resolved"]]
    if fails:
        ftr = [trajectory(r) for r in fails]
        no_edit = sum(1 for t in ftr if t["n_edit"] == 0)
        no_test = sum(1 for t in ftr if t["ran_pytest"] == 0)
        print(f"\n  失败的 {len(fails)} 条中：")
        print(f"    从未编辑过任何文件      {no_edit:>4} ({no_edit/len(fails)*100:.0f}%)")
        print(f"    从未跑过 pytest 验证    {no_test:>4} ({no_test/len(fails)*100:.0f}%)")
        if no_test > len(fails) * 0.5:
            print("\n  -> 多数失败【没有自己验证】：改完就跑，不知道对不对")
            print("     提示词里应该更强调「改完必须跑测试确认」")

    # ── ⑧ 结论 ──────────────────────────────────────────
    print("=" * 68)
    print("判断")
    print("=" * 68)
    ex    = cls.get("轮数耗尽", 0)
    miss  = cls.get("测试跑不起来", 0)
    wrong = cls.get("改了但没改对", 0) + cls.get("没修好", 0)
    print(f"\n  轮数耗尽 {ex/max(fail_total,1)*100:.0f}%   "
          f"测试跑不起来 {miss/max(fail_total,1)*100:.0f}%   "
          f"改了没对 {wrong/max(fail_total,1)*100:.0f}%\n")
    if ex > fail_total * 0.4:
        print("  -> 多数失败是【轮数不够】：瓶颈在 harness")
        print("     提高 max_turns、精简提示词、给更聚焦的题面，比训练更直接")
    elif ex > fail_total * 0.2:
        print(f"  -> {ex} 条【轮数不够】：harness 有优化空间，但不是主要瓶颈")
    if miss > fail_total * 0.2:
        print("  -> 相当一部分是【测试跑不起来】：多为语法/导入错误")
        print("     提示词里强调「改完先确认能 import」，或在工具层加语法检查")
    if wrong > fail_total * 0.3:
        print("  -> 相当一部分是【改了但没改对】：这是模型能力问题")
        print("     这类失败才需要训练来改善")
    brk = cls.get("改坏了其他测试", 0) + cls.get("没修好且弄坏了别的", 0)
    if brk > fail_total * 0.15:
        print(f"  -> {brk} 条【把别的测试弄坏了】：改得太宽")
        print("     提示词里强调「只改必要的地方」，或加 diff 规模约束")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
