#!/usr/bin/env python3
"""select_cases.py — 从批量结果里挑选修复实验的候选轨迹。

筛选：失败（final n_fail>0）且有验证过的有害步骤（causal / causal_by_file）。
用法：python oracle/select_cases.py results/d1a --out demo/cases_d1a.json
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-f2p-left", type=int, default=1)
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        b = os.path.basename(f).split(".")[0]
        if not b.isdigit() or f.endswith((".traj.json", "_teacher.json", "_preds.json")):
            continue
        r = json.load(open(f))
        evs = r.get("evals") or []
        final = evs[-1].get("n_fail") if evs else None
        harmful = [c["turn"] for c in (r.get("counterfactuals") or [])
                   if c.get("label") in ("causal", "causal_by_file")]
        if not harmful:
            continue
        tpath = os.path.join(args.results_dir, f"{b}_teacher.json")
        teacher = None
        if os.path.exists(tpath):
            teacher = (json.load(open(tpath)).get("teacher") or {}).get("pivot_step")
        traj_path = os.path.join(args.results_dir, f"{b}.traj.json")
        n_turns = None
        if os.path.exists(traj_path):
            t = json.load(open(traj_path))
            n_turns = sum(1 for m in t.get("messages", []) if m.get("role") == "assistant"
                          and any((tc.get("function", {}) or {}).get("name") == "bash"
                                  for tc in (m.get("tool_calls") or [])))
        rows.append({"dir": args.results_dir, "run": b,
                     "instance_id": r.get("instance_id"), "final_n_fail": final,
                     "harmful_turns": harmful, "teacher_step": teacher,
                     "n_turns": n_turns})
    print(f"候选 {len(rows)} 条（失败且有验证过的有害步骤）")
    print(f"{'run':>4s} {'final':>5s} {'harmful':>10s} {'teacher':>8s} {'turns':>6s}")
    for x in sorted(rows, key=lambda y: -((y['n_turns'] or 0))):
        print(f"{x['run']:>4s} {str(x['final_n_fail']):>5s} {str(x['harmful_turns']):>10s} "
              f"{str(x['teacher_step']):>8s} {str(x['n_turns']):>6s}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
