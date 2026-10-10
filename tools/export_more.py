#!/usr/bin/env python3
"""从 SWE-smith 的 parquet 里导出目标仓库的任务

输入：$SWESMITH_RL_HOME/swe_smith/data/*.parquet
输出：$SWESMITH_RL_HOME/tasks_more.jsonl

用法:
    export SWESMITH_RL_HOME=/path/to/workspace
    python tools/export_more.py

要换仓库/数量，改下面的 WANT 字典。
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

BASE = Path(os.environ.get("SWESMITH_RL_HOME", Path(__file__).resolve().parents[1]))

# 目标仓库 → 每个仓库要导出的条数
WANT = {"mistune": 200, "sqlparse": 200, "boltons": 100, "isodate": 130}
COLS = ["instance_id", "patch", "FAIL_TO_PASS", "PASS_TO_PASS", "repo", "problem_statement"]


def main():
    data_dir = BASE / "swe_smith" / "data"
    if not data_dir.is_dir():
        print(f"❌ 找不到 {data_dir}")
        print("   把 SWE-smith 的 parquet 放到这里（见 docs/DATA_FORMAT.md）")
        return 1

    out_path = BASE / "tasks_more.jsonl"
    got = defaultdict(int)
    with open(out_path, "w") as out:
        for f in sorted(data_dir.glob("*.parquet")):
            for b in pq.ParquetFile(f).iter_batches(batch_size=200, columns=COLS):
                for x in b.to_pylist():
                    n = x["repo"].split("/")[-1].split("__")[1].split(".")[0]
                    if n in WANT and got[n] < WANT[n]:
                        out.write(json.dumps(x, ensure_ascii=False) + "\n")
                        got[n] += 1
            if all(got[k] >= WANT[k] for k in WANT):
                break
    print("导出:", dict(got), "总计", sum(got.values()))

    tot = ps_ok = 0
    for l in out_path.read_text().splitlines():
        if not l.strip().startswith("{"):
            continue
        tot += 1
        if (json.loads(l).get("problem_statement") or "").strip():
            ps_ok += 1
    print(f"有题面: {ps_ok}/{tot}  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
