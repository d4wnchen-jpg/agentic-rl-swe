#!/usr/bin/env python3
"""stats_repair.py — 修复实验配对统计（受控设计：continue / revert_*）。

文件命名：<out_dir>/<run>_<condition>.json，condition ∈
  continue, revert_random, revert_teacher, revert_ours, restart, ...

主指标：F2P 通过数（f2p_pass）的配对差值；次指标：最终失败数 n_fail、resolve。
"""
import argparse
import glob
import json
import os
import random

CONDS = ["continue", "revert_random", "revert_teacher", "revert_ours"]
OLD = {"restart", "random", "teacher", "ours"}


def parse_name(b):
    if b.endswith(".conv"):
        return None
    parts = b.split("_")
    if "revert" in parts:
        i = parts.index("revert")
        run = "_".join(parts[:i])
        cond = "revert_" + parts[i + 1] if i + 1 < len(parts) else "revert"
        return run, cond
    if len(parts) >= 2:
        cond = parts[-1]
        if cond in CONDS or cond in OLD:
            return "_".join(parts[:-1]), cond
    return None


def load(outdir):
    data = {}
    for f in glob.glob(os.path.join(outdir, "*.json")):
        b = os.path.basename(f)[:-5]
        if (b.endswith("_teacher") and "_revert_" not in b) or b.endswith("_preds"):
            continue
        p = parse_name(b)
        if not p:
            continue
        run, cond = p
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


def key(x):
    return (0, int(x)) if x.isdigit() else (1, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    args = ap.parse_args()
    data = load(args.outdir)
    runs = [r for r in data if r.isdigit()]
    print(f"cases: {len(runs)}")
    print(f"{'case':>5s}  " + "  ".join(f"{c:>15s}" for c in CONDS))
    print("              " + "  ".join(f"{'n_fail/f2p':>15s}" for _ in CONDS))
    for run in sorted(runs, key=key):
        cells = []
        for c in CONDS:
            d = data[run].get(c)
            cells.append(f"{d['n_fail']}/{d['f2p_pass']}" if d else "-")
        print(f"{run:>5s}  " + "  ".join(f"{c:>15s}" for c in cells))

    print()
    for c in CONDS[1:]:
        if c == "revert_ours":
            continue
        diffs, nd = [], 0
        for run in runs:
            a = data[run].get("revert_ours")
            b = data[run].get(c)
            if a and b:
                diffs.append(a["f2p_pass"] - b["f2p_pass"])
                nd += 1
        if not diffs:
            print(f"revert_ours - {c}: 无数据")
            continue
        lo, hi = boot_ci(diffs)
        print(f"revert_ours - {c}: mean f2p_pass diff = {sum(diffs)/len(diffs):+.2f} "
              f"[95% CI {lo:+.2f}, {hi:+.2f}]  (n={nd})")
        resolved = sum(1 for run in runs
                       if data[run].get("revert_ours") and data[run]["revert_ours"]["n_fail"] == 0)
        resolved_base = sum(1 for run in runs
                            if data[run].get(c) and data[run][c]["n_fail"] == 0)
        print(f"    resolved: revert_ours={resolved}/{len(runs)}  {c}={resolved_base}/{len(runs)}")


if __name__ == "__main__":
    main()
