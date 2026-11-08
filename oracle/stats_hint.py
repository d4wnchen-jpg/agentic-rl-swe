#!/usr/bin/env python3
"""汇总 hint 实验：从最终状态继续，不同信息量/步数预算下的修复效果。

用法: python oracle/stats_hint.py results/hint_pilot
"""
import glob
import json
import os
import sys
from collections import defaultdict


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/hint_pilot"
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = os.path.basename(f)
        if b.endswith(".conv.json"):
            continue
        r = json.load(open(f))
        rows.append({
            "run": b.split("_")[0], "cond": "_".join(b.split("_")[1:]).replace(".json", ""),
            "start": r.get("start_n_fail"), "final": r.get("n_fail"),
            "f2p": r.get("f2p_pass"), "n_f2p": r.get("n_f2p"),
            "steps": r.get("steps_used"), "budget": r.get("steps_budget"),
            "tok": r.get("tokens"), "parse_fail": r.get("parse_failures"),
            "hint": r.get("hint"),
        })
    if not rows:
        print("没有结果")
        return
    print(f"{'run':>4s} {'cond':>12s} {'hint':>7s} {'start':>5s} {'final':>5s} "
          f"{'f2p':>6s} {'steps':>6s} {'tok':>8s}")
    for r in rows:
        print(f"{r['run']:>4s} {r['cond']:>12s} {str(r['hint']):>7s} {str(r['start']):>5s} "
              f"{str(r['final']):>5s} {str(r['f2p']):>3s}/{str(r['n_f2p']):<2s} "
              f"{str(r['steps']):>3s}/{str(r['budget']):<3s} {str(r['tok']):>8s}")
    print()
    g = defaultdict(list)
    for r in rows:
        g[r["cond"]].append(r)
    print(f"{'cond':>12s} {'n':>3s} {'start_avg':>9s} {'final_avg':>9s} {'resolved':>8s} "
          f"{'improved':>8s} {'worse':>6s} {'f2p_sum':>7s}")
    for cond, rs in sorted(g.items()):
        n = len(rs)
        start = sum((x["start"] or 0) for x in rs) / n
        final = sum((x["final"] or 0) for x in rs) / n
        resolved = sum(1 for x in rs if x["final"] == 0)
        improved = sum(1 for x in rs if (x["final"] or 0) < (x["start"] or 0))
        worse = sum(1 for x in rs if (x["final"] or 0) > (x["start"] or 0))
        f2p = sum((x["f2p"] or 0) for x in rs)
        print(f"{cond:>12s} {n:>3d} {start:>9.2f} {final:>9.2f} {resolved:>8d} "
              f"{improved:>8d} {worse:>6d} {f2p:>7d}")


if __name__ == "__main__":
    main()
