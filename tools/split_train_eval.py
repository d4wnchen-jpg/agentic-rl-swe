#!/usr/bin/env python3
"""按难度探测结果分层切分训练/评测集

输入（都在 SWESMITH_RL_HOME 下）：
    probe_par_result.jsonl   由 tools/probe_par.py 产生（每题跑 N 次的结果）
    tasks_ok.jsonl           由 tools/verify_tasks.py 产生（验证通过的任务全集）

输出：
    train.jsonl       部分 mixed + 少量 allfail
    eval.jsonl        其余全部（held-out，含 mixed / allpass / allfail）
    val_mixed.jsonl   从 mixed 里留出的专项验证子集

用法:
    export SWESMITH_RL_HOME=/path/to/workspace
    python tools/split_train_eval.py [--val-subset 17] [--seed 42]

分层判据（按每题 N 次 rollout 的通过次数）：
    mixed    0 < 通过 < N   → 有学习信号，GRPO 的基础
    allpass  通过 = N       → 丢弃（基座已全对，优势恒为 0）
    allfail  通过 = 0       → 抽一批进训练集，防止训练集偏简单
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(os.environ.get("SWESMITH_RL_HOME", Path(__file__).resolve().parents[1]))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip().startswith("{")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=str(BASE / "probe_par_result.jsonl"),
                    help="难度探测结果（tools/probe_par.py 产生）")
    ap.add_argument("--tasks", default=str(BASE / "tasks_ok.jsonl"),
                    help="验证通过的任务全集（tools/verify_tasks.py 产生）")
    ap.add_argument("--val-subset", type=int, default=17,
                    help="从 mixed 里留出的专项验证子集大小")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    probe_path, tasks_path = Path(a.probe), Path(a.tasks)
    for p in (probe_path, tasks_path):
        if not p.exists():
            print(f"❌ 找不到 {p}")
            print("   数据准备顺序: export_more.py → verify_tasks.py → probe_par.py → 本脚本")
            return 1

    random.seed(a.seed)

    per: dict[str, list] = defaultdict(list)
    for r in load_jsonl(probe_path):
        if "instance_id" in r and "resolved" in r:
            per[r["instance_id"]].append(bool(r["resolved"]))

    mixed = [i for i, v in per.items() if 0 < sum(v) < len(v)]
    allpass = [i for i, v in per.items() if sum(v) == len(v)]
    allfail = [i for i, v in per.items() if sum(v) == 0]
    random.shuffle(mixed)

    val_subset = mixed[:a.val_subset]
    train_mixed = mixed[a.val_subset:]
    train_fail = random.sample(allfail, min(len(allfail), max(1, len(train_mixed) // 5)))

    train_ids = set(train_mixed) | set(train_fail)
    eval_ids = [i for i in per if i not in train_ids]      # 其余全部（含 val_subset）

    tasks = {t["instance_id"]: t for t in load_jsonl(tasks_path) if "instance_id" in t}

    def dump(ids, name):
        out = BASE / name
        n = 0
        with open(out, "w") as f:
            for i in ids:
                if i in tasks:
                    f.write(json.dumps(tasks[i], ensure_ascii=False) + "\n")
                    n += 1
        return n

    print(f"mixed {len(mixed)}   全对 {len(allpass)}   全错 {len(allfail)}")
    print(f"探测覆盖 {len(per)} 道，任务全集 {len(tasks)} 道")
    print()
    print(f"train     : {len(train_mixed)} mixed + {len(train_fail)} allfail"
          f" = {dump(train_mixed + train_fail, 'train.jsonl')} 道")
    print(f"eval      : {dump(eval_ids, 'eval.jsonl')} 道（held-out，不参与训练）")
    print(f"  └ mixed 专项验证: {dump(val_subset, 'val_mixed.jsonl')} 道")
    return 0


if __name__ == "__main__":
    sys.exit(main())
