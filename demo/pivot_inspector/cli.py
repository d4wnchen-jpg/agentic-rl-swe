#!/usr/bin/env python3
"""PivotInspector CLI：把一条 agent 轨迹的验证结果渲染成诊断报告。

用法：
  python -m pivot_inspector.cli --results-dir ../results/g1c --run 7 --out examples/g1c_7.html
  python -m pivot_inspector.cli --results-dir ../results/g2a --run 1 --out examples/g2a_1.html
"""
import argparse
import sys

from .report import to_html


def main():
    ap = argparse.ArgumentParser(description="PivotInspector report generator")
    ap.add_argument("--results-dir", required=True, help="results/<name> 目录（含 <run>.json / .traj.json）")
    ap.add_argument("--run", required=True, help="轨迹编号（文件名前缀）")
    ap.add_argument("--out", required=True, help="输出 HTML 路径")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    s = to_html(args.results_dir, args.run, args.out, args.title)
    print(f"instance={s['instance']} final_n_fail={s['final']} "
          f"harmful={[c['turn'] for c in s['harmful']]} "
          f"beneficial={[c['turn'] for c in s['beneficial']]} "
          f"recovery_pairs={[(r['pivot_turn'], r['recovery_turn']) for r in s['recs']]}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
