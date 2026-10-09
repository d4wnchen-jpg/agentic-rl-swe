#!/usr/bin/env python3
"""grpo_llm.py — 支持 LoRA 和 token-id/logprob 记录的 vLLM 封装"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # sm_120 必需
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_CU13 = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cu13")
if os.path.isdir(_CU13):
    os.environ["CUDA_HOME"] = _CU13
    os.environ["PATH"] = f"{_CU13}/bin:" + os.environ.get("PATH", "")


class GRPOLlm:
    def __init__(self, model_path, max_model_len=81920, gpu_mem=0.40,
                 max_new_tokens=1024, max_lora_rank=16, max_loras=2,
                 max_num_seqs=256, enable_thinking=None):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest
        self.max_new_tokens = max_new_tokens
        # ⚠️ None = 不传该参数（推荐）。
        #    传 False 会让 Qwen3.5 的模板在 prompt 尾部插入完整的空思考块
        #    <think>\n\n</think>\n\n，模型有概率复读它。
        #    详见 docs/RESULTS.md 第 8 节。
        self.enable_thinking = enable_thinking

        print(f"[GRPOLlm] loading {model_path}  max_model_len={max_model_len}  "
              f"gpu_mem={gpu_mem}  lora=True", flush=True)
        self.llm = LLM(
            model=model_path, trust_remote_code=True, dtype="bfloat16",
            max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
            enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
            enable_lora=True, max_lora_rank=max_lora_rank, max_loras=max_loras,
            gdn_prefill_backend="triton", max_num_batched_tokens=8192,
            max_num_seqs=max_num_seqs,     # ⚠️ Qwen3.5 的 Mamba cache 块数有限
            disable_log_stats=True,
        )
        print("[GRPOLlm] ready", flush=True)

    def _ct_kwargs(self):
        """chat template 的额外参数；None 表示不传"""
        return (None if self.enable_thinking is None
                else {"enable_thinking": self.enable_thinking})

    def _sp(self, temperature, stop, want_logprobs=False):
        sp = self.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=(float(os.environ.get("GEN_TEMPERATURE", "1.0"))
                         if temperature is None else temperature),
            stop=stop or [],
            include_stop_str_in_output=True,
        )
        if want_logprobs:
            sp.logprobs = 0
        return sp

    def batch_chat(self, batch_messages, stop=None, temperature=None,
                   lora_request=None):
        """兼容 agent_par.run_parallel 的接口：只返回文本"""
        outs = self.llm.chat(
            batch_messages, sampling_params=self._sp(temperature, stop),
            lora_request=lora_request,
            chat_template_kwargs=self._ct_kwargs())
        return [o.outputs[0].text if o.outputs else "" for o in outs]

    def batch_chat_ids(self, batch_messages, stop=None, temperature=None,
                       lora_request=None, want_logprobs=True):
        """返回 [(prompt_ids, response_ids, text, logprobs), ...]

        logprobs[k] = response_ids[k] 在【采样时策略】下的 logprob
                    = GRPO 的 old_logprob（不需要额外 forward）
        """
        outs = self.llm.chat(
            batch_messages,
            sampling_params=self._sp(temperature, stop, want_logprobs),
            lora_request=lora_request,
            chat_template_kwargs=self._ct_kwargs())
        res = []
        for o in outs:
            pid = list(o.prompt_token_ids or [])
            out0 = o.outputs[0] if o.outputs else None
            rid = list(out0.token_ids) if out0 else []
            txt = out0.text if out0 else ""
            lps = []
            if want_logprobs and out0 is not None and out0.logprobs:
                for d in out0.logprobs:
                    lps.append(float(next(iter(d.values())).logprob) if d else 0.0)
            res.append((pid, rid, txt, lps))
        return res
