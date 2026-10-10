#!/usr/bin/env python3
"""probe_par.py — 并行 pass rate 探测

用法:
  python tools/probe_par.py <题数> <每题次数> <最大轮数> <并发数> [adapter路径]

注意：所有执行逻辑必须放在 main() 里 —— vLLM 用 spawn 启动子进程，
子进程会重新导入 __main__，顶层代码会被重复执行导致 RuntimeError。
"""
import os, sys, json, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import HFBatch, VLLMBatch
from agent_par import run_parallel

BASE = os.environ.get("SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1]))


def main():
    N     = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    K     = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    WORK  = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    ADPT  = sys.argv[5] if len(sys.argv) > 5 else None

    # 默认接 verify_tasks.py 的产出（验证通过的任务）
    TASK_FILE = os.environ.get("TASK_FILE", f"{BASE}/tasks_ok.jsonl")
    tasks = [json.loads(l) for l in open(TASK_FILE)][:N]
    print(f"[题库] {TASK_FILE}", flush=True)
    jobs  = [t for t in tasks for _ in range(K)]
    print(f"题数 {len(tasks)} × 每题 {K} 次 = {len(jobs)} 个 rollout | "
          f"并发 {WORK} | 轮数上限 {TURNS}", flush=True)

    BACKEND = os.environ.get("LLM_BACKEND", "vllm").lower()
    if BACKEND == "vllm":
        llm = VLLMBatch(f"{BASE}/models/Qwen3.5-9B",
                        max_model_len=int(os.environ.get("MAX_MODEL_LEN", 32768)),
                        gpu_mem=float(os.environ.get("GPU_MEM", 0.95)),
                        max_new_tokens=1024)
    else:
        llm = HFBatch(f"{BASE}/models/Qwen3.5-9B", adapter=ADPT, max_new_tokens=1024)

    t0 = time.time()
    RESUME = os.environ.get("RESUME", "0") == "1"
    RESULT = f"{BASE}/probe_par_result.jsonl"
    prior_rows = []
    if RESUME and os.path.exists(RESULT):
        for l in open(RESULT):
            try:
                prior_rows.append(json.loads(l))
            except Exception:
                pass
        done_ids = {r["instance_id"] for r in prior_rows}
        before = len(jobs)
        jobs = [j for j in jobs if j["task"]["instance_id"] not in done_ids]
        print(f"[resume] 已完成 {len(done_ids)} 道题，跳过；"
              f"待跑 {len(jobs)} 个 rollout（原 {before}）", flush=True)

    def _dump(sts, path):
        """写盘：先补上续跑前的旧结果，再写本轮 states（每次全量重写，不重复）"""
        with open(path, "w") as f:
            for r in prior_rows:
                f.write(json.dumps(r, ensure_ascii=False) + chr(10))
            for x in sts:
                f.write(json.dumps({"instance_id": x["task"]["instance_id"],
                                    "resolved": x["resolved"], "n_fail": x["n_fail"],
                                    "n_missing": x.get("n_missing", 0),
                                    "nturns": x["nturns"],
                                    "messages": x["messages"]}, ensure_ascii=False) + chr(10))

    WAVE = int(os.environ.get("WAVE", "0"))
    if WAVE > 0 and len(jobs) > WAVE:
        states, nw = [], (len(jobs) + WAVE - 1) // WAVE
        for i in range(0, len(jobs), WAVE):
            batch = jobs[i:i+WAVE]
            print(f"\n===== 波次 {i//WAVE+1}/{nw}  ({len(batch)} 个 rollout) =====", flush=True)
            states += run_parallel(batch, llm, max_turns=TURNS,
                                   workers=min(WORK, len(batch)), root="/tmp/par")
            _dump(states, f"{BASE}/probe_par_result.jsonl")     # 每波落盘
            print(f"[已保存 {len(states)} 条]", flush=True)
    else:
        states = run_parallel(jobs, llm, max_turns=TURNS, workers=WORK, root="/tmp/par")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    per = defaultdict(list)
    for s in states:
        per[s["task"]["instance_id"]].append(s)

    print("\n===== 每题结果 =====")
    print(f"{'instance_id':52} {'成功/次数':>9} {'失败测试':>18} {'轮数':>5}")
    n_mixed = n_solved = 0
    for iid, ss in per.items():
        ok = sum(1 for s in ss if s["resolved"])
        nf = [s["n_fail"] for s in ss]
        nt = sum(s["nturns"] for s in ss) / len(ss)
        if ok == len(ss): n_solved += 1
        if 0 < ok < len(ss): n_mixed += 1
        print(f"{iid[:52]:52} {ok:>4}/{len(ss):<4} {str(nf):>18} {nt:5.1f}")

    tot_ok = sum(1 for s in states if s["resolved"])
    print("\n===== 总计 =====")
    print(f"  rollout 级 pass rate : {tot_ok}/{len(states)} = {tot_ok/len(states)*100:.1f}%")
    print(f"  全对（无学习信号）    : {n_solved} / {len(per)} 道")
    print(f"  部分对（有学习信号）⭐ : {n_mixed} / {len(per)} 道")
    print(f"  全错                  : {len(per)-n_solved-n_mixed} / {len(per)} 道")

    with open(f"{BASE}/probe_par_result.jsonl", "w") as f:
        for s in states:
            f.write(json.dumps({"instance_id": s["task"]["instance_id"],
                                "resolved": s["resolved"], "n_fail": s["n_fail"],
                                "n_missing": s["n_missing"], "nturns": s["nturns"],
                                "messages": s["messages"]}, ensure_ascii=False) + "\n")
    print(f"\n明细已存 → {BASE}/probe_par_result.jsonl")


if __name__ == "__main__":
    main()
