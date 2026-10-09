#!/usr/bin/env python3
"""llm_client.py — 批量生成客户端

HFBatch: 一次 generate 处理整个 batch（左 padding + attention_mask）
关键点：enable_thinking 默认【不传】。
        传 False 会让 Qwen3.5 的模板在 prompt 尾部插入完整的空思考块
        <think>\n\n</think>\n\n，模型有概率复读它
        （本项目实测：训练后 96% 的 rollout 首轮复读；见 docs/RESULTS.md 第 8 节）
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys


class HFBatch:
    def __init__(self, model_path, adapter=None, max_new_tokens=1536,
                 enable_thinking=None, dtype=torch.bfloat16):
        print(f"加载 tokenizer/模型: {model_path}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"                     # 批量生成必须左 padding
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=dtype, device_map="cuda")
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, device_map="cuda")

        if adapter:
            from peft import PeftModel
            print(f"挂载 adapter: {adapter}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

        self.max_new_tokens  = max_new_tokens
        self.enable_thinking = enable_thinking
        self.tpl_has_thinking = "enable_thinking" in (self.tok.chat_template or "")
        print(f"  enable_thinking 参数可用: {self.tpl_has_thinking}", flush=True)

    def _render(self, messages):
        kw = dict(add_generation_prompt=True, tokenize=False)
        if self.tpl_has_thinking and self.enable_thinking is not None:
            kw["enable_thinking"] = self.enable_thinking
        return self.tok.apply_chat_template(messages, **kw)

    def batch_chat(self, batch_messages, temperature=None, stop=None):
        texts = [self._render(m) for m in batch_messages]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        ids  = enc["input_ids"].to("cuda")
        attn = enc["attention_mask"].to("cuda")
        gen_kw = dict(max_new_tokens=self.max_new_tokens,
                      pad_token_id=self.tok.pad_token_id)
        if stop:
            gen_kw["stop_strings"] = stop          # 生成到这些串就停
            gen_kw["tokenizer"]    = self.tok
        if temperature is None:
            gen_kw["do_sample"] = False
        else:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=attn, **gen_kw)
        return [self.tok.decode(g, skip_special_tokens=True)
                for g in out[:, ids.shape[1]:]]


class VLLMBatch:
    """vLLM 后端：分页 KV + 连续批处理

    在 CUDA 12.4 上跑 Qwen3.5 需要四个 workaround（都已验证）：
      1. CUDA_HOME 指向 vLLM 自带的 CUDA 13.4（否则 FlashInfer JIT 编译失败：
         nvcc fatal: Unknown option '--compress-mode=size'）
      2. VLLM_USE_FLASHINFER_SAMPLER=0（跳过 FlashInfer sampling 内核）
      3. gdn_prefill_backend='triton'（跳过 FlashInfer GDN prefill 内核）
      4. enforce_eager=True（权重 17.66G 占卡 75%，没空间给 CUDA graph）
    """
    CU13 = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cu13")

    def __init__(self, model_path, max_model_len=16384, gpu_mem=0.95,
                 max_new_tokens=1024, enable_thinking=None):
        import os
        if os.path.isdir(self.CU13):                    # 必须在 import vllm 之前
            os.environ["CUDA_HOME"] = self.CU13
            os.environ["PATH"] = f"{self.CU13}/bin:" + os.environ.get("PATH", "")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        self.enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1"
        print(f"[VLLMBatch] enforce_eager={self.enforce_eager} (1=关CUDA graph, 0=开)", flush=True)

        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        print(f"vLLM loading {model_path}  max_model_len={max_model_len}  gpu_mem={gpu_mem}", flush=True)
        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=True,
            gdn_prefill_backend="triton",
            enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
        )
        self.max_new_tokens  = max_new_tokens
        self.enable_thinking = enable_thinking

    def batch_chat(self, batch_messages, stop=None, temperature=None):
        sp = self.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=(float(os.environ.get("GEN_TEMPERATURE", "1.0"))
                         if temperature is None else temperature),   # 对齐 agent-lightning: 1.0
            stop=stop or [],
            include_stop_str_in_output=True,
        )
        outs = self.llm.chat(
            batch_messages,
            sampling_params=sp,
            chat_template_kwargs=(None if self.enable_thinking is None
                                  else {"enable_thinking": self.enable_thinking}),
        )
        return [o.outputs[0].text if o.outputs else "" for o in outs]
