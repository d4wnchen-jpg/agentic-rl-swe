#!/usr/bin/env python3
"""summarize_all.py — 跨实例汇总（memo 用表）。

用法：
  python summarize_all.py g1c g2a r3a
输出 markdown 表格：
  - 每条轨迹的标签统计（harmful / beneficial / recovery pairs）
  - G1-any / G1-rec
  - 教师对照（轮次 exact/±1、符号说反、无编辑过度归因、从不回答 0）
  - 保真度（若 results/fidelity_<name>/fidelity.json 存在）
"""
from __future__ import annotations

import glob
import json
import os
import sys


def load_dir(d):
    reports, teachers = {}, {}
    for f in glob.glob(os.path.join(d, "*.json")):
        b = os.path.basename(f).split(".")[0]
        if not b.isdigit():
            continue
        if f.endswith((".traj.json", "_teacher.json", "_preds.json")):
            continue
        reports[b] = json.load(open(f))
    for f in glob.glob(os.path.join(d, "*_teacher.json")):
        b = os.path.basename(f).split("_")[0]
        if b.isdigit():
            teachers[b] = json.load(open(f))
    return reports, teachers


def ours_sets(r):
    cfs = r.get("counterfactuals") or []
    harmful = {c["turn"] for c in cfs if c.get("label") in ("causal", "causal_by_file")}
    beneficial = {c["turn"] for c in cfs if c.get("label") in ("beneficial", "beneficial_by_file")}
    recov = {x["pivot_turn"] for x in (r.get("recoveries") or [])}
    pairs = r.get("recoveries") or []
    raw_causal = sum(1 for x in pairs
                     if x.get("cf_skip_recovery_final_n_fail") is not None
                     and x.get("real_final_n_fail") is not None
                     and x["cf_skip_recovery_final_n_fail"] > x["real_final_n_fail"])
    return harmful, beneficial, recov, len(pairs), raw_causal


def main():
    names = sys.argv[1:] or ["g1c", "g2a", "r3a"]
    print("| instance | n | failed | harmful | beneficial | recov pairs | G1-any | G1-rec |")
    print("|---|---|---|---|---|---|---|---|")
    totals = dict(n=0, failed=0, harmful=0, beneficial=0, pairs=0, any=0, rec=0)
    all_rows = []
    for name in names:
        d = os.path.join("results", name)
        if not os.path.isdir(d):
            print(f"| {name} | - | - | - | - | - | - | - |")
            continue
        reports, teachers = load_dir(d)
        n = len(reports)
        failed = sum(1 for r in reports.values()
                     if (r.get("evals") or [{}])[-1].get("n_fail") not in (0, None))
        h = b = p = anyc = recc = 0
        for i, r in reports.items():
            harmful, beneficial, recov, npairs, raw = ours_sets(r)
            h += len(harmful)
            b += len(beneficial)
            p += npairs
            if harmful or beneficial:
                anyc += 1
            if recov:
                recc += 1
            all_rows.append((name, i, r, teachers.get(i, {})))
        print(f"| {name} | {n} | {failed} | {h} | {b} | {p} | "
              f"{anyc}/{n} | {recc}/{n} |")
        totals["n"] += n
        totals["failed"] += failed
        totals["harmful"] += h
        totals["beneficial"] += b
        totals["pairs"] += p
        totals["any"] += anyc
        totals["rec"] += recc

    print(f"| **合计** | {totals['n']} | {totals['failed']} | {totals['harmful']} | "
          f"{totals['beneficial']} | {totals['pairs']} | "
          f"{totals['any']}/{totals['n']} | {totals['rec']}/{totals['n']} |")

    # 教师对照（修正语义）：
    #   有 mistake 的轨迹 = 失败+harmful 或 成功+recovery
    #   无 mistake 的轨迹 = 其余；教师给出非零轮次 = 过度归因/符号错误
    print("\n**教师对照（mismatch=我们判定没有可归因错误，教师仍指认）**\n")
    print("| group | 轨迹数 | 教师 exact | ±1 | 无 mistake 轨迹 | 其中教师仍指认 |")
    print("|---|---|---|---|---|---|")
    with_m = with_m_exact = with_m_near = 0
    no_m = no_m_over = 0
    for name, i, r, t in all_rows:
        harmful, beneficial, recov, _, _ = ours_sets(r)
        evs = r.get("evals") or []
        final = evs[-1].get("n_fail") if evs else None
        failed = final not in (0, None)
        mistakes = sorted(harmful) if failed else sorted(recov)
        step = None
        if t and (t.get("teacher") or {}).get("pivot_step") is not None:
            step = t["teacher"]["pivot_step"]
        if not mistakes:
            no_m += 1
            if step not in (None, 0):
                no_m_over += 1
            continue
        with_m += 1
        if isinstance(step, int) and step in mistakes:
            with_m_exact += 1
            with_m_near += 1
        elif isinstance(step, int) and any(abs(step - o) <= 1 for o in mistakes):
            with_m_near += 1
    print(f"| 有可归因错误 | {with_m} | {with_m_exact}/{with_m} | {with_m_near}/{with_m} | - | - |")
    print(f"| 无 mistake 轨迹 | {no_m} | - | - | {no_m} | {no_m_over}/{no_m} |")

    # 保真度
    print("\n**保真度**\n")
    print("| instance | identical | diff_files |")
    print("|---|---|---|")
    for name in names:
        fd = os.path.join("results", f"fidelity_{name}")
        files = glob.glob(os.path.join(fd, "*.json"))
        if not files:
            print(f"| {name} | - | - |")
            continue
        ok = sum(1 for f in files if json.load(open(f)).get("identical"))
        print(f"| {name} | {ok}/{len(files)} | - |")


if __name__ == "__main__":
    main()
