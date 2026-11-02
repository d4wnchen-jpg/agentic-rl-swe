#!/usr/bin/env python3
"""summarize_g1.py — 汇总多条轨迹的重放结果，算 G1 覆盖率。

用法：
  python summarize_g1.py '/workspaces/agentic-rl-swe/out_traj_*'

统计：
  - 失败轨迹数（final n_fail > 0）
  - 有 regression pivot 的失败轨迹数
  - 反事实验证为有害编辑（cf delta < 0）的失败轨迹数
  - G1a = 有 regression / 失败；G1b = 有反事实验证 / 失败
"""
from __future__ import annotations

import glob
import json
import os
import sys


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "out_traj_*"
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    rows = []
    for d in dirs:
        p = os.path.join(d, "traj_report.json")
        if not os.path.exists(p):
            continue
        try:
            r = json.load(open(p))
        except Exception:
            continue
        evals = r.get("evals") or []
        final = evals[-1] if evals else {}
        pivots = r.get("pivots") or []
        regs = [x for x in pivots if x.get("type") == "regression"]
        cfs = r.get("counterfactuals") or []
        labels = [c.get("label") for c in cfs]
        verified = [c for c in cfs if c.get("label") in ("causal", "causal_by_file")]
        causal = [c for c in cfs if c.get("label") == "causal"]
        causal_file = [c for c in cfs if c.get("label") == "causal_by_file"]
        rows.append({
            "dir": d,
            "instance": r.get("instance_id"),
            "turns": r.get("n_turns"),
            "final_n_fail": final.get("n_fail"),
            "n_regressions": len(regs),
            "regression_turns": [x["turn"] for x in regs],
            "cf_checked": len(cfs),
            "n_causal": len(causal),
            "n_causal_file": len(causal_file),
            "n_unverified": sum(1 for x in labels
                                if x in ("unverified", "transient_or_confounded",
                                         "cf_invalid_no_fallback")),
            "cf_harmful": len(verified),
            "cf_turns": [c["turn"] for c in verified],
            "labels": labels,
            "replay_s": r.get("replay_seconds"),
        })

    if not rows:
        print("没有找到 traj_report.json；检查路径模式:", pattern)
        return

    print(f"{'instance':24s} {'turns':>5s} {'fail':>5s} {'regs':>5s} "
          f"{'cf':>4s} {'causal':>6s} {'file':>5s} {'unver':>6s}")
    for x in rows:
        print(f"{str(x['instance'])[:24]:24s} {str(x['turns']):>5s} "
              f"{str(x['final_n_fail']):>5s} {x['n_regressions']:>5d} "
              f"{x['cf_checked']:>4d} {x['n_causal']:>6d} "
              f"{x['n_causal_file']:>5d} {x['n_unverified']:>6d}")

    failed = [x for x in rows if (x["final_n_fail"] or 0) > 0]
    with_reg = [x for x in failed if x["n_regressions"] > 0]
    with_cf = [x for x in failed if x["cf_harmful"] > 0]

    print()
    print(f"轨迹总数        : {len(rows)}")
    print(f"失败轨迹        : {len(failed)}")
    print(f"有 regression   : {len(with_reg)}")
    print(f"有反事实验证    : {len(with_cf)}")
    if failed:
        print(f"G1a（有 regression / 失败）   = {len(with_reg)}/{len(failed)} "
              f"= {len(with_reg)/len(failed)*100:.1f}%")
        print(f"G1b（有因果验证 / 失败）      = {len(with_cf)}/{len(failed)} "
              f"= {len(with_cf)/len(failed)*100:.1f}%")
    print()
    print("明细：")
    for x in rows:
        print(f"  {x['dir']}: final_fail={x['final_n_fail']} "
              f"reg_turns={x['regression_turns']} "
              f"verified_turns={x['cf_turns']} labels={x['labels']}")


if __name__ == "__main__":
    main()
