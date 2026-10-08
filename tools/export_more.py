import json, pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
import os
import sys

BASE = os.environ.get("SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1]))

WANT = {"mistune": 200, "sqlparse": 200, "boltons": 100, "isodate": 130}
COLS = ["instance_id","patch","FAIL_TO_PASS","PASS_TO_PASS","repo","problem_statement"]

got = defaultdict(int)
with open(f"{BASE}/tasks_more.jsonl","w") as out:
    for f in sorted(Path(f"{BASE}/swe_smith/data").glob("*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=200, columns=COLS):
            for x in b.to_pylist():
                n = x["repo"].split("/")[-1].split("__")[1].split(".")[0]
                if n in WANT and got[n] < WANT[n]:
                    out.write(json.dumps(x, ensure_ascii=False) + "\n"); got[n] += 1
        if all(got[k] >= WANT[k] for k in WANT): break
print("导出:", dict(got), "总计", sum(got.values()))

ps_ok = 0; tot = 0
for l in open(f"{BASE}/tasks_more.jsonl"):
    tot += 1
    if (json.loads(l).get("problem_statement") or "").strip(): ps_ok += 1
print(f"有题面: {ps_ok}/{tot}")
