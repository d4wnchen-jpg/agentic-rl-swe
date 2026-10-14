#!/usr/bin/env python3
"""对指定适配器做评测（可指定题库 / 道数 / 每题次数）

用法：
  python tools/eval_adapter.py --adapter grpo_adapters/step000 --n 115 --k 1 --tag base
  python tools/eval_adapter.py --n 115 --k 1 --tag base_raw        # 不给 adapter = 基座
"""
import os, sys, json, argparse
from pathlib import Path
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE = os.environ.get("SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="适配器目录；不给则用基座（不挂 LoRA）")
    ap.add_argument("--tasks", default=f"{BASE}/eval.jsonl")
    ap.add_argument("--n", type=int, default=115)
    ap.add_argument("--k", type=int, default=1, help="每题跑几次")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--dump", default=None,
                    help="把每条 rollout 的明细写成 jsonl（默认 eval_<tag>_detail.jsonl）")
    a = ap.parse_args()

    # ⭐ 先落运行清单 —— 放在加载模型之前，这样即使 OOM 崩溃也有记录
    from run_manifest import collect, write
    write(collect(tag=a.tag, adapter=a.adapter, tasks=a.tasks,
                  n=a.n, k=a.k, turns=a.turns, gpu_mem=a.gpu_mem),
          f"{BASE}/eval_{a.tag}.meta.json")

    from grpo_llm import GRPOLlm
    from eval_quick import quick_eval

    llm = GRPOLlm(f"{BASE}/models/Qwen3.5-9B", max_model_len=81920,
                  gpu_mem=a.gpu_mem, max_new_tokens=16384, max_num_seqs=128)
    req = llm.LoRARequest(a.tag, 777, a.adapter) if a.adapter else None

    print(f"[{a.tag}] adapter={a.adapter or '(基座)'}  n={a.n}  k={a.k}", flush=True)
    dump = a.dump or f"{BASE}/eval_{a.tag}_detail.jsonl"
    rate, n, ok = quick_eval(llm, req, eval_file=a.tasks, n_limit=a.n,
                             max_turns=a.turns, k=a.k, dump=dump)
    print(f"[{a.tag}] {ok}/{n} = {rate*100:.1f}%", flush=True)
    with open(f"{BASE}/eval_{a.tag}.json", "w") as f:
        json.dump({"tag": a.tag, "adapter": a.adapter, "n": n, "ok": ok,
                   "rate": rate, "k": a.k, "turns": a.turns,
                   "detail": dump}, f, ensure_ascii=False, indent=2)
    print(f"已写 {BASE}/eval_{a.tag}.json  和  {dump}", flush=True)


if __name__ == "__main__":
    main()
