#!/usr/bin/env python3
"""eval_quick.py — 评测（默认跑指定文件的前 n_limit 道，每题 1 次）"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE = os.environ.get("SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1]))


def quick_eval(llm, lora_request, eval_file=None, n_limit=None, max_turns=40, k=1,
               temperature=0.7):
    """temperature=0.7 对齐 agl 的 rollout.val_kwargs（训练用 1.0）"""
    """返回 (solve_rate, n_rollouts, n_solved)"""
    from grpo_rollout import run_parallel_ids
    f = eval_file or f"{BASE}/eval.jsonl"
    tasks = [json.loads(l) for l in open(f) if l.strip()]
    if n_limit:
        tasks = tasks[:n_limit]
    jobs = [t for t in tasks for _ in range(k)]
    st = run_parallel_ids(jobs, llm, max_turns=max_turns,
                          workers=min(24, len(jobs)), root="/tmp/grpo_eval",
                          lora_request=lora_request, temperature=temperature)
    n_ok = sum(1 for s in st if s["resolved"])
    return n_ok / max(len(st), 1), len(st), n_ok
