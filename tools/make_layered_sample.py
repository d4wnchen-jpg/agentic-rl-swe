"""按仓库分层采样：每个仓库抽相同数量，避免采样偏差"""
import json, random, sys
from collections import defaultdict
import os
from pathlib import Path

BASE = os.environ.get("SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1]))

def repo_of(t):
    return t["repo"].split("/")[-1].split("__")[1].split(".")[0]

rows = [json.loads(l) for l in open(f"{BASE}/tasks_with_ps.jsonl")]
per = defaultdict(list)
for t in rows:
    per[repo_of(t)].append(t)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
random.seed(0)
picked = []
print(f"{'仓库':14}{'全库':>6}{'抽取':>6}")
print("-" * 28)
for repo, ts in sorted(per.items()):
    random.shuffle(ts)
    take = ts[:N]
    picked += take
    print(f"{repo:14}{len(ts):>6}{len(take):>6}")

with open(f"{BASE}/tasks_layered.jsonl", "w") as f:
    for t in picked:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"\n分层样本: {len(picked)} 道 → tasks_layered.jsonl")
