#!/usr/bin/env python3
"""audit_trajectory.py — 从现有 rollout 数据回答四个问题（只读，不需要 GPU）

不改任何东西，只解析已经落盘的 jsonl。

 ① 零方差组比例    —— 训练信号里有多少是有效的
 ② 分层通过率      —— 通过/失败集中在哪一类题上
 ③ 是否曾经修好    —— 区分"根本不会修"和"修好了但没保住"
 ④ 成功后是否继续  —— 停止机制有没有问题
 ⑤ reward × 轨迹   —— 什么行为能拿到分

用法：
    python tools/audit_trajectory.py [--home DIR] [--base FILE] [--s20 FILE]

默认读取 $SWESMITH_RL_HOME（或仓库根目录）下的：
    log_grpo.jsonl              每步指标（含 n_zero_var）
    eval.jsonl                  任务定义（含 FAIL_TO_PASS）
    probe_par_result.jsonl      难度探测（用于分层；没有就跳过 ②）
    eval_base40_detail.jsonl    基座明细
    eval_s20_40_detail.jsonl    step020 明细

退出码：0 正常；1 关键文件缺失
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agl_util import f2p_only_filter, parse_test_statuses  # noqa: E402

PASSED = ("PASSED", "XFAIL")
_PYTEST_RE = __import__("re").compile(r"\bpytest\b")


# ═══════════════════════════════════════════════════════════
def read_jsonl(path: Path) -> list[dict]:
    """读 jsonl，跳过非 JSON 行（任务行 / 垃圾行）"""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_rollouts(path: Path) -> dict:
    """instance_id -> rollout（只保留带 messages 的行）"""
    return {o["instance_id"]: o for o in read_jsonl(path)
            if "instance_id" in o and "messages" in o}


def disp_width(s: str) -> int:
    """显示宽度：CJK / emoji 算 2 列"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - disp_width(s))


def bar(n: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return ""
    k = int(round(width * n / total))
    return "█" * k + "·" * (width - k)


# ═══════════════════════════════════════════════════════════
def q1_zero_variance(home: Path, n_prompts: int = 4) -> None:
    print("=" * 74)
    print("① 零方差组比例 —— 有效训练信号有多少")
    print("=" * 74)
    rows = read_jsonl(home / "log_grpo.jsonl")
    if not rows:
        print("  （没有 log_grpo.jsonl，跳过）\n")
        return
    tot = zv = 0
    for o in rows:
        tot += n_prompts
        zv += o.get("n_zero_var", 0) or 0
    if tot == 0:
        print("  （log_grpo.jsonl 是空的，跳过）\n")
        return
    frac = zv / tot
    print(f"  总组数 {tot}   零方差组 {zv} ({frac*100:.0f}%)  {bar(zv, tot)}")
    print(f"  有效训练信号 {(1-frac)*100:.0f}%")
    print()
    if frac > 0.4:
        print(f"  → 信号严重不足。加训练量的收益要打对折以上，“×15” 实为 ×{15/(1-frac):.0f}")
        print("    建议先做题目池动态化（把稳定做对的题移出），再谈加训练量。")
    elif frac > 0.2:
        print(f"  → 有一定浪费。“×15” 应修正为 ×{15/(1-frac):.0f}")
    else:
        print("  → 信号利用率正常，可以按原计划加训练量。")
    print()


# ═══════════════════════════════════════════════════════════
def q2_tier(home: Path, base: dict, s20: dict) -> None:
    print("=" * 74)
    print("② 分层通过率 —— 对题来自哪一层")
    print("=" * 74)
    probe = defaultdict(list)
    for o in read_jsonl(home / "probe_par_result.jsonl"):
        if "resolved" in o and "instance_id" in o:
            probe[o["instance_id"]].append(bool(o["resolved"]))
    if not probe:
        print("  （没有 probe_par_result.jsonl，跳过）\n")
        return

    def tier(iid: str) -> str:
        rs = probe.get(iid)
        if not rs:
            return "未知"
        if all(rs):
            return "allpass"
        if not any(rs):
            return "allfail"
        return "mixed"

    print(f"  {pad('层', 11)} {pad('题数', 5)}   {pad('基座通过', 10)}   {pad('step020', 10)}")
    print("  " + "-" * 46)
    for t in ("mixed", "allpass", "allfail", "未知"):
        ids = [i for i in base if tier(i) == t]
        if not ids:
            continue
        b = sum(1 for i in ids if base[i].get("resolved"))
        ids20 = [i for i in ids if i in s20]
        s = sum(1 for i in ids20 if s20[i].get("resolved"))
        print(f"  {pad(t, 11)} {len(ids):>5}   {b:>4}/{len(ids):<5}   {s:>4}/{len(ids20):<5}")
    print()


# ═══════════════════════════════════════════════════════════
def audit_solved(rows: dict, tasks: dict, tag: str) -> dict:
    """逐条 rollout 判断「是否曾经修好」

    分类（互斥，按优先级）：
        曾经全通过              F2P 全部出现在输出里且全 PASSED
        部分通过（覆盖不全）      输出里只有部分 F2P，但那部分全 PASSED
        跑过测试但从没通过       输出里有 F2P 节点，但没全 PASSED
        跑过pytest但输出无F2P    命令里有 pytest，但输出里一个 F2P 节点都没有
        从没跑过测试            整条轨迹没出现过 pytest 命令
    """
    res = Counter()
    cont_after = []
    cover_ratio = []
    ever_ok_resolved = Counter()      # ⭐ 曾经全通过 × 最终判分

    for r in rows.values():
        f2p = set(tasks.get(r["instance_id"], {}).get("FAIL_TO_PASS") or [])
        if not f2p:
            res["无任务定义"] += 1
            continue

        ok_at = None
        seen_f2p = False
        ran_pytest = False
        full_cover_all_pass = False
        partial_all_pass = False
        turn = 0

        for m in r.get("messages", []):
            role = m.get("role")
            if role == "assistant":
                turn += 1
                if _PYTEST_RE.search(str(m.get("content", ""))):
                    ran_pytest = True
                continue
            if role != "user":
                continue
            c = str(m.get("content", ""))
            if not c.startswith("OBSERVATION:"):
                continue
            st = parse_test_statuses(c)
            hit = f2p & set(st)
            if not hit:
                continue
            seen_f2p = True
            cover_ratio.append(len(hit) / len(f2p))
            good = sum(1 for t in hit if st[t] in PASSED)
            if good == len(hit):
                if hit == f2p:
                    full_cover_all_pass = True
                    if ok_at is None:
                        ok_at = turn
                else:
                    partial_all_pass = True

        if ok_at is not None:
            res["曾经全通过"] += 1
            ever_ok_resolved[bool(r.get("resolved"))] += 1
            extra = (r.get("nturns") or turn) - ok_at
            if extra > 3:
                cont_after.append(extra)
        elif partial_all_pass:
            res["部分通过（覆盖不全）"] += 1
        elif seen_f2p:
            res["跑过测试但从没通过"] += 1
        elif ran_pytest:
            res["跑过pytest但输出无F2P"] += 1
        else:
            res["从没跑过测试"] += 1

    n = max(len(rows), 1)
    order = ["曾经全通过", "部分通过（覆盖不全）", "跑过测试但从没通过",
             "跑过pytest但输出无F2P", "从没跑过测试"]
    print(f"     {pad('分类', 24)} {pad('条数', 5)}  {pad('占比', 6)}")
    for k in order:
        v = res.get(k, 0)
        mark = "⭐ " if k == "曾经全通过" else "   "
        print(f"  {mark}{pad(k, 24)} {v:>5}  {v/n*100:>5.1f}%  {bar(v, n)}")
    if res.get("无任务定义"):
        print(f"     （{res['无任务定义']} 条找不到任务定义，已跳过）")
    if cover_ratio:
        avg = sum(cover_ratio) / len(cover_ratio)
        print(f"  F2P 节点平均覆盖率 {avg*100:.0f}%"
              f"（<100% 说明模型只用 -k 跑了部分测试，或输出被截断）")
    if ever_ok_resolved:
        ok = ever_ok_resolved.get(True, 0)
        bad = ever_ok_resolved.get(False, 0)
        print()
        print(f"  ⭐ 曾经全通过 × 最终判分：")
        print(f"       最终也通过   {ok:>4}   ← 学习有效")
        print(f"       最终却失败   {bad:>4}   ← 会修但没保住")
    if cont_after:
        print(f"  🔁 全通过后又跑了 >3 轮：{len(cont_after)} 条，"
              f"平均多跑 {sum(cont_after)/len(cont_after):.0f} 轮")

    return {"res": res, "cont": cont_after, "n": len(rows),
            "ever_ok_resolved": ever_ok_resolved}


def q3_solved(home: Path, base: dict, s20: dict) -> None:
    print("=" * 74)
    print("③ 是否曾经修好（从模型自己跑的 pytest 输出解析）")
    print("=" * 74)
    tasks = {o["instance_id"]: o for o in read_jsonl(home / "eval.jsonl")
             if "instance_id" in o}
    if not tasks:
        print("  （没有 eval.jsonl，无法拿到 FAIL_TO_PASS，跳过）\n")
        return
    out = {}
    for tag, rows in (("基座", base), ("step020", s20)):
        if not rows:
            continue
        print(f"  ── {tag}（{len(rows)} 条）")
        out[tag] = audit_solved(rows, tasks, tag)
        print()
    return out


def q3b_coverage(home: Path, base: dict, s20: dict) -> None:
    """③b 验证覆盖面 —— 模型实际跑过的节点，覆盖了判分节点的多少

    覆盖率从【观察输出】解析（模型实际跑了什么），不是从命令字符串，
    所以 `pytest tests/test_x.py` 这种没写节点名的命令也能正确计入。
    """
    print("=" * 74)
    print("③b 验证覆盖面 —— 模型跑的测试 vs 判分的 195 个节点")
    print("=" * 74)
    tasks = {o["instance_id"]: o for o in read_jsonl(home / "eval.jsonl")
             if "instance_id" in o}
    if not tasks:
        print("  （没有 eval.jsonl，跳过）\n")
        return

    for tag, rows in (("基座", base), ("step020", s20)):
        if not rows:
            continue
        f2p_c, p2p_c, all_c = [], [], []
        saw_f2p_all = 0
        f2p_pass_seen = []      # 模型自己看到：覆盖到的 F2P 里通过的比例
        for r in rows.values():
            t = tasks.get(r["instance_id"]) or {}
            f2p = t.get("FAIL_TO_PASS") or []
            p2p = t.get("PASS_TO_PASS") or []
            if not f2p:
                continue
            jf = set(f2p)
            jp = set(f2p_only_filter(f2p, p2p))
            ja = jf | jp

            seen, passed = set(), set()
            for m in r.get("messages", []):
                if m.get("role") != "user":
                    continue
                c = str(m.get("content", ""))
                if not c.startswith("OBSERVATION:"):
                    continue
                st = parse_test_statuses(c)
                seen |= set(st)
                passed |= {k for k, v in st.items() if v in PASSED}

            if len(jf):
                f2p_c.append(len(jf & seen) / len(jf))
                if jf <= seen:
                    saw_f2p_all += 1
                if jf & seen:
                    f2p_pass_seen.append(len(jf & passed) / len(jf & seen))
            if len(jp):
                p2p_c.append(len(jp & seen) / len(jp))
            if len(ja):
                all_c.append(len(ja & seen) / len(ja))

        def avg(x):
            return f"{sum(x)/len(x)*100:5.1f}%" if x else "   — "

        n = len(rows)
        print(f"  ── {tag}（{n} 条）")
        print(f"     F2P 覆盖率        {avg(f2p_c)}   ← 必须修好的那些，模型看到了多少")
        print(f"     P2P 覆盖率        {avg(p2p_c)}")
        print(f"     判分节点总覆盖率  {avg(all_c)}")
        print(f"     {pad('跑全过 F2P 的 rollout', 26)} {saw_f2p_all:>4} / {n}"
              f"  ({saw_f2p_all/max(n,1)*100:.0f}%)")
        if f2p_pass_seen:
            print(f"     {pad('模型看到的 F2P 通过率', 26)} "
                  f"{sum(f2p_pass_seen)/len(f2p_pass_seen)*100:5.1f}%"
                  f"   ← 在它跑到的范围内，通过的比例")
        print()


def q4_continue(home: Path, base: dict, s20: dict, extra_hint: int = 3) -> None:
    print("=" * 74)
    print("④ 全通过之后还跑了几轮（停止机制）")
    print("=" * 74)
    tasks = {o["instance_id"]: o for o in read_jsonl(home / "eval.jsonl")
             if "instance_id" in o}
    if not tasks:
        print("  （没有 eval.jsonl，跳过）\n")
        return
    for tag, rows in (("基座", base), ("step020", s20)):
        if not rows:
            continue
        extras = []
        for r in rows.values():
            f2p = set(tasks.get(r["instance_id"], {}).get("FAIL_TO_PASS") or [])
            if not f2p:
                continue
            ok_at = None
            turn = 0
            for m in r.get("messages", []):
                if m.get("role") == "assistant":
                    turn += 1
                    continue
                if m.get("role") != "user":
                    continue
                c = str(m.get("content", ""))
                if not c.startswith("OBSERVATION:"):
                    continue
                st = parse_test_statuses(c)
                if f2p <= set(st) and all(st[t] in PASSED for t in f2p):
                    if ok_at is None:
                        ok_at = turn
            if ok_at is not None:
                extras.append((r.get("nturns") or turn) - ok_at)
        if not extras:
            print(f"  {tag:8} 没有「全通过」的样本")
            continue
        extras.sort()
        print(f"  {tag:8} {len(extras)} 条全通过的样本，"
              f"之后平均又跑了 {sum(extras)/len(extras):.1f} 轮  "
              f"（中位 {extras[len(extras)//2]}，最大 {extras[-1]}）")
    print()


# ═══════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=os.environ.get(
        "SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1])))
    ap.add_argument("--base", default="eval_base40_detail.jsonl")
    ap.add_argument("--s20", default="eval_s20_40_detail.jsonl")
    a = ap.parse_args()

    home = Path(a.home)
    base = load_rollouts(home / a.base)
    s20 = load_rollouts(home / a.s20)

    print(f"目录 {home}")
    print(f"  基座    {len(base)} 条   ({a.base})")
    print(f"  step020 {len(s20)} 条   ({a.s20})")
    print()
    if not base and not s20:
        print("两个明细文件都读不到 —— 检查文件名和路径。")
        return 1

    q1_zero_variance(home)
    q2_tier(home, base, s20)
    q3_solved(home, base, s20)
    q3b_coverage(home, base, s20)
    q4_continue(home, base, s20)

    print("=" * 74)
    print("怎么读")
    print("=" * 74)
    print("  ③ 的「曾经全通过」占比高（>30%）→ 会修但没保住，优先修停止/验证机制")
    print("  ③ 的「曾经全通过」≈0          → 根本不会修，训练量/数据是第一优先")
    print("  ③ 的「从没跑过测试」占比高     → 模型不验证自己的修改，是提示词问题")
    print("  ④ 平均多跑 >5 轮               → 模型不停止，提示词要强化「通过后立刻 submit」")
    print()
    print("  ⑤ reward × 轨迹特征见：")
    print("     python tools/analyze_failures.py eval_base40_detail.jsonl 40")
    print("     python tools/analyze_failures.py eval_s20_40_detail.jsonl 40")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
