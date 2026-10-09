#!/usr/bin/env python3
"""grpo_rollout.py — 录制 token ids + logprob 的并行 rollout（GRPO 专用）

与 agent_par.run_parallel 的区别：
  1. 不 elide 历史（保证 token 级前缀链）
  2. 每轮记录 [prompt_ids, response_ids, response_logprobs]
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_par import (prep_sandbox, run_tools, evaluate, init_msgs,
                       parse_actions, repo_of, ENVS, STOP_STRINGS, restore_tests)
from agl_util import FORMAT_HINT, MAX_FORMAT_ERR

MAX_CTX_CHARS = int(os.environ.get("GRPO_MAX_CTX", "200000"))


def run_parallel_ids(tasks, llm, max_turns=40, workers=16, root="/tmp/grpo",
                     lora_request=None, verbose=False, temperature=None):
    os.makedirs(root, exist_ok=True)
    stamp = int(time.time())

    states = []
    for i, t in enumerate(tasks):
        wd = Path(root) / f"{stamp}_{i}"
        try:
            prep_sandbox(t, wd)
        except Exception as e:
            print(f"  [{i}] 沙箱失败: {e}", flush=True)
            continue
        states.append({
            "task": t, "wd": wd, "messages": init_msgs(t),
            "done": False, "last_sig": None, "rep": 0, "fmt_err": 0,
            "turns": [],                       # [[prompt_ids, response_ids, logprobs], ...]
            "resolved": False, "n_fail": -1, "n_missing": -1, "nturns": 0,
        })
    print(f"已准备 {len(states)} 个沙箱", flush=True)

    ex = ThreadPoolExecutor(max_workers=workers)
    t0 = time.time()

    for turn in range(max_turns):
        alive = [s for s in states if not s["done"]]
        if not alive:
            break

        for s in alive:
            tot = sum(len(str(m.get("content", ""))) for m in s["messages"])
            if tot > MAX_CTX_CHARS:
                print(f"  [warn] 上下文 {tot} 字符超预算，结束该 episode", flush=True)
                s["done"] = True
        alive = [s for s in states if not s["done"]]
        if not alive:
            break

        try:
            outs = llm.batch_chat_ids([s["messages"] for s in alive],
                                      stop=STOP_STRINGS,
                                      lora_request=lora_request,
                                      temperature=temperature)
        except Exception as e:
            print(f"[warn] 整批失败({type(e).__name__}): {str(e)[:120]}", flush=True)
            outs = []
            for s in alive:
                try:
                    outs += llm.batch_chat_ids([s["messages"]], stop=STOP_STRINGS,
                                               lora_request=lora_request,
                                               temperature=temperature)
                except Exception as e2:
                    print(f"[warn] 单条失败({type(e2).__name__}): {str(e2)[:100]}", flush=True)
                    outs.append(([], [], None, []))

        jobs = []
        for s, tup in zip(alive, outs):
            tup = list(tup) + [None, [], [], []]
            pid, rid, text, lps = tup[0], tup[1], tup[2], tup[3]
            if text is None:
                s["done"] = True
                continue
            s["nturns"] += 1
            s["messages"].append({"role": "assistant", "content": text})
            s["turns"].append([list(pid), list(rid), list(lps)])

            acts = parse_actions(text)
            if verbose:
                print(f"\n--- [{s['task']['instance_id'][:36]}] turn {s['nturns']} ---\n{text[:300]}", flush=True)
            if not acts:
                s["fmt_err"] = s.get("fmt_err", 0) + 1
                if s["fmt_err"] >= MAX_FORMAT_ERR:
                    s["done"] = True
                    continue
                s["messages"].append({"role": "user", "content": FORMAT_HINT})
                continue
            s["fmt_err"] = 0
            if any(n == "submit" for n, _ in acts):
                s["done"] = True
                continue
            sig = json.dumps(acts, ensure_ascii=False)
            if sig == s["last_sig"]:
                s["rep"] += 1
                if s["rep"] >= 3:
                    s["done"] = True
                    continue
            else:
                s["rep"], s["last_sig"] = 0, sig
            jobs.append((s, acts))

        futs = [(s, ex.submit(run_tools, acts, s["wd"])) for s, acts in jobs]
        for s, f in futs:
            try:
                obs = f.result()
            except Exception as e:
                obs = [f"<tool error: {e}>"]
            s["messages"].append({"role": "user",
                                  "content": "OBSERVATION:\n" + "\n".join(obs)})

        print(f"[turn {turn+1:3}] 活跃 {len(alive):3}  "
              f"完成 {sum(1 for s in states if s['done']):3}  "
              f"{time.time()-t0:6.0f}s", flush=True)

    def fin(s):
        repo = repo_of(s["task"])
        try:
            restore_tests(repo, s["wd"])
            ok, det = evaluate(s["task"], ENVS[repo], s["wd"])
        except Exception as e:
            print(f"  [warn] 判分失败({type(e).__name__}): {str(e)[:100]}", flush=True)
            ok, det = False, {"n_fail": -1, "n_missing": -1}
        s["resolved"] = ok
        s["n_fail"] = det.get("n_fail", -1)
        s["n_missing"] = det.get("n_missing", -1)
        s["judge"] = det            # ⭐ 保留逐测试状态（res），供失败分析用
        return s

    list(ex.map(fin, states))
    ex.shutdown()
    return states
