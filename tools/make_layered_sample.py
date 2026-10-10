#!/usr/bin/env python3
"""按仓库分层采样：每个仓库抽相同数量，避免采样偏差

输入：$SWESMITH_RL_HOME/tasks_ok.jsonl（tools/verify_tasks.py 的产出）
输出：$SWESMITH_RL_HOME/tasks_layered.jsonl

用法:
    export SWESMITH_RL_HOME=/path/to/workspace
    python tools/make_layered_sample.py [每个仓库抽几道，默认 8]
"""
import json, os, random, sys
from collections import defaultdict
from pathlib import Path

BASE = Path(os.environ.get("SWESMITH_RL_HOME", Path(__file__).resolve().parents[1]))


def repo_of(t):
    return t["repo"].split("/")[-1].split("__")[1].split(".")[0]


def main():
    src = Path(os.environ.get("LAYERED_SRC", BASE / "tasks_ok.jsonl"))
    if not src.exists():
        print(f"❌ 找不到 {src}")
        print("   先跑 tools/verify_tasks.py 产生它")
        return 1

    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip().startswith("{")]
    per = defaultdict(list)
    for t in rows:
        per[repo_of(t)].append(t)

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    random.seed(0)
    picked = []
    print(f"{'仓库':14}{'全库':>6}{'抽取':>6}")
    print("-" * 28)
    for repo, ts in sorted(per.items()):
        random.shuffle(ts)
        take = ts[:n]
        picked += take
        print(f"{repo:14}{len(ts):>6}{len(take):>6}")

    out = BASE / "tasks_layered.jsonl"
    with open(out, "w") as f:
        for t in picked:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\n分层样本: {len(picked)} 道 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
