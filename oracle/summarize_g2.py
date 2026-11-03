#!/usr/bin/env python3
"""summarize_g2.py — G2：教师事后指认 vs 我们的可执行验证标签。

口径：
  1. **可比案例**：我们有验证 pivot 的轨迹
     - 成功轨迹（final n_fail=0）→ 我们的 pivot = 可恢复配对的 pivot 轮
     - 失败且有致命 pivot（causal / causal_by_file）→ 我们的 pivot = 致命轮
     比较教师 pivot 与我们的 pivot：exact / within1 / miss
  2. **遗漏型失败**：失败但我们的方法找不到致命编辑（失败=没改对）
     教师仍给出 pivot → 记为「过度归因」（over-attribution）

用法：
  python summarize_g2.py results/g1c
"""
from __future__ import annotations

import glob
import json
import os
import sys


def load(dirpath):
    reports, teachers = {}, {}
    for f in glob.glob(os.path.join(dirpath, "*.json")):
        b = os.path.basename(f).split(".")[0]
        if not b.isdigit():
            continue
        if f.endswith(".traj.json"):
            continue
        if f.endswith("_teacher.json"):
            continue
        reports[b] = json.load(open(f))
    for f in glob.glob(os.path.join(dirpath, "*_teacher.json")):
        b = os.path.basename(f).split("_")[0]
        if b.isdigit():
            teachers[b] = json.load(open(f))
    return reports, teachers


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else "results/g1c"
    reports, teachers = load(dirpath)
    rows = []
    for i in sorted(reports, key=int):
        r, t = reports[i], teachers.get(i, {})
        evs = r.get("evals") or []
        final = evs[-1].get("n_fail") if evs else None
        fatal = sorted({c["turn"] for c in (r.get("counterfactuals") or [])
                        if c.get("label") in ("causal", "causal_by_file")})
        recov = sorted({x["pivot_turn"] for x in (r.get("recoveries") or [])})
        step = (t.get("teacher") or {}).get("pivot_step")
        if (final or 0) > 0 and fatal:
            cls, ours = "fatal_case", fatal
        elif (final or 0) == 0:
            cls, ours = "success_case", recov
        else:
            cls, ours = "omission_case", []
        if cls == "omission_case":
            match = "teacher_over" if step not in (None, 0) else "agree_none"
        elif step is None:
            match = "no_answer"
        elif step in ours:
            match = "exact"
        elif any(abs(step - o) <= 1 for o in ours):
            match = "within1"
        else:
            match = "miss"
        rows.append(dict(i=i, final=final, cls=cls, ours=ours, step=step, match=match))

    print(f"{'run':>3s} {'final':>5s} {'class':>13s} {'ours':>10s} {'teacher':>7s} {'match':>12s}")
    for x in rows:
        print(f"{x['i']:>3s} {str(x['final']):>5s} {x['cls']:>13s} {str(x['ours']):>10s} "
              f"{str(x['step']):>7s} {x['match']:>12s}")

    comp = [x for x in rows if x["cls"] != "omission_case"]
    omis = [x for x in rows if x["cls"] == "omission_case"]
    exact = sum(1 for x in comp if x["match"] == "exact")
    near = sum(1 for x in comp if x["match"] in ("exact", "within1"))
    over = sum(1 for x in omis if x["match"] == "teacher_over")
    zero = sum(1 for x in rows if x["step"] in (None, 0))

    # ── 新口径：轮次定位（对任意已验证编辑）+ 符号正确性 ──
    loc_exact = loc_near = loc_total = 0
    sign_mismatch = sign_total = 0
    noedit_teacher = noedit_total = 0
    for i in sorted(reports, key=int):
        r, t = reports[i], teachers.get(i, {})
        cfs = r.get("counterfactuals") or []
        harmful = {c["turn"] for c in cfs if c.get("label") in ("causal", "causal_by_file")}
        beneficial = {c["turn"] for c in cfs if c.get("label") in ("beneficial", "beneficial_by_file")}
        recov = {x["pivot_turn"] for x in (r.get("recoveries") or [])}
        ours = sorted(harmful | beneficial | recov)
        step = (t.get("teacher") or {}).get("pivot_step")
        if not ours:
            noedit_total += 1
            if step not in (None, 0):
                noedit_teacher += 1
            continue
        loc_total += 1
        if step in ours:
            loc_exact += 1
            loc_near += 1
        elif any(isinstance(step, int) and abs(step - o) <= 1 for o in ours):
            loc_near += 1
        # 符号：命中的轮次是否只有 beneficial 语义（教师默认说成 mistake）
        if isinstance(step, int):
            hit_harmful = step in harmful
            hit_beneficial = step in beneficial
            if hit_harmful or hit_beneficial:
                sign_total += 1
                if hit_beneficial and not hit_harmful:
                    sign_mismatch += 1

    print()
    print(f"轨迹总数: {len(rows)}")
    print(f"[轮次定位] 有验证编辑的轨迹 {loc_total}: exact {loc_exact}/{loc_total} "
          f"({loc_exact/max(1,loc_total)*100:.0f}%), ±1 {loc_near}/{loc_total} "
          f"({loc_near/max(1,loc_total)*100:.0f}%)")
    print(f"[符号正确] 教师命中轮次 {sign_total}: 把仅 beneficial 的轮说成 mistake = "
          f"{sign_mismatch}/{sign_total}")
    print(f"[无编辑可归因] {noedit_total} 条中教师仍指认 = {noedit_teacher}")
    print(f"可比案例: {len(comp)}")
    print(f"  教师 exact   : {exact}/{len(comp)} = {exact/max(1,len(comp))*100:.0f}%")
    print(f"  教师 within-1: {near}/{len(comp)} = {near/max(1,len(comp))*100:.0f}%")
    print(f"  误差 >=2 轮  : {len(comp)-near}/{len(comp)}")
    print(f"遗漏型失败（我们无致命 pivot）: {len(omis)}")
    print(f"  教师仍给 pivot（过度归因）: {over}/{len(omis)}")
    print(f"教师回答 '没有关键步'(0/空) 的比例: {zero}/{len(rows)}")


if __name__ == "__main__":
    main()
