#!/usr/bin/env python3
"""stats_repair.py — 修复实验的配对统计（primary: F2P 修复数；secondary: resolve 率）。

用法：
  python oracle/stats_repair.py repair_results --cases demo/cases_d1a.json
结果文件命名：<out_dir>/<case_run>_<condition>.json
"""
import argparse
import glob
import json
import os
import random


def load(outdir):
    data = {}
    for f in glob.glob(os.path.join(outdir, "*.json")):
        b = os.path.basename(f)[:-5]
        parts = b.rsplit("_", 1)
        if len(parts) != 2:
            continue
        run, cond = parts
        data.setdefault(run, {})[cond] = json.load(open(f))
    return data


def boot_ci(vals, n=5000):
    if not vals:
        return (None, None)
    means = []
    for _ in range(n):
        means.append(sum(random.choice(vals) for _ in vals) / len(vals))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--cases", default="")
    args = ap.parse_args()
    data = load(args.outdir)
    conds = ["restart", "continue", "revert_random", "revert_teacher", "revert_ours"]
    print(f"cases: {len(data)}")
    print(f"{'case':>12s} " + " ".join(f"{c:>14s}" for c in conds) + "   (final n_fail / f2p_pass)")
    for run in sorted(data, key=lambda x: int(x) if x.isdigit() else x):
        cells = []
        for c in conds:
            d = data[run].get(c)
            cells.append(f"{d['n_fail']}/{d['f2p_pass']} of {d['n_f2p']}" if d else "-")
        print(f"{run:>12s} " + " ".join(f"{c:>14s}" for c in cells) + "   " +
              " ".join(f"{c:>14s}" for c in cells))

    # 配对：ours vs restart / teacher（f2p_pass 差值与 bootstrap CI）
    diffs = {c: [] for c in conds if c != "revert_ours"}
    for run, d in data.items():
        if "revert_ours" not in d:
            continue
        for c in diffs:
            if c in d:
                diffs[c].append(d["revert_ours"]["f2p_pass"] - d[c]["f2p_pass"])
    print()
    for c, vals in diffs.items():
        if not vals:
            print(f"ours - {c}: 无数据")
            continue
        lo, hi = boot_ci(vals)
        print(f"ours - {c}: mean f2p_pass diff = {sum(vals)/len(vals):+.2f} "
              f"[95% CI {lo:+.2f}, {hi:+.2f}]  (n={len(vals)})")


if __name__ == "__main__":
    main()
