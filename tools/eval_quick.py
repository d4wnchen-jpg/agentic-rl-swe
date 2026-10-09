#!/usr/bin/env python3
"""eval_quick.py — 评测（默认跑指定文件的前 n_limit 道，每题 1 次）"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE = os.environ.get("SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1]))


def quick_eval(llm, lora_request, eval_file=None, n_limit=None, max_turns=40, k=1,
               temperature=0.7, dump=None):
    """temperature=0.7 对齐 agl 的 rollout.val_kwargs（训练用 1.0）

    dump: 若给定路径，把每条 rollout 的明细写成 jsonl（供失败模式分析）

    返回 (solve_rate, n_rollouts, n_solved)
    """
    from grpo_rollout import run_parallel_ids
    f = eval_file or f"{BASE}/eval.jsonl"
    tasks = [json.loads(l) for l in open(f) if l.strip()]
    if n_limit:
        tasks = tasks[:n_limit]
    jobs = [t for t in tasks for _ in range(k)]
    st = run_parallel_ids(jobs, llm, max_turns=max_turns,
                          workers=min(24, len(jobs)), root="/tmp/grpo_eval",
                          lora_request=lora_request, temperature=temperature)
    if dump:
        with open(dump, "w") as fh:
            for x in st:
                t = x["task"]
                det = x.get("judge") or {}
                res = det.get("res") or {}
                f2p, p2p = t.get("FAIL_TO_PASS", []), t.get("PASS_TO_PASS", [])
                # ⭐ 把"没修好"和"弄坏了"分开统计
                f2p_bad = [n for n in f2p if res.get(n) not in ("PASSED", "XFAIL")]
                p2p_bad = [n for n in p2p if res.get(n) not in ("PASSED", "XFAIL")]
                fh.write(json.dumps({
                    "instance_id": t["instance_id"],
                    "resolved": x["resolved"],
                    "n_fail": x.get("n_fail", -1),
                    "n_missing": x.get("n_missing", -1),
                    "nturns": x.get("nturns", 0),
                    "repo": t.get("repo", ""),
                    "n_f2p": len(f2p), "n_p2p": len(p2p),
                    "n_f2p_bad": len(f2p_bad), "n_p2p_bad": len(p2p_bad),
                    "test_status": res,
                    "messages": x["messages"],
                }, ensure_ascii=False) + chr(10))
        print(f"  明细已写 {dump}", flush=True)

    n_ok = sum(1 for s in st if s["resolved"])
    return n_ok / max(len(st), 1), len(st), n_ok
