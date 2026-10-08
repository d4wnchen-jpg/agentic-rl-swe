#!/usr/bin/env python3
"""train_grpo.py — 单卡 LoRA GRPO（含中途评测）

超参出处：
  from agl  = agent-lightning (microsoft/agent-lightning, examples/swe_smith)
  单卡适配  = 因 1x96GB 而非 4x192GB 而做的缩放
"""
from __future__ import annotations
import os, sys, json, time, math, shutil, random, datetime
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

BASE = os.environ.get("SWESMITH_RL_HOME", str(Path(__file__).resolve().parents[1]))
MODEL_PATH  = f"{BASE}/models/Qwen3.5-9B"
TRAIN_FILE  = f"{BASE}/train.jsonl"
VAL_FILE    = f"{BASE}/val_mixed.jsonl"      # 中途评测（17 道）
FULL_EVAL   = f"{BASE}/eval.jsonl"           # 最终评测（115 道）
PROBE_FILE  = f"{BASE}/probe_par_result.jsonl"
ADAPTER_ROOT = f"{BASE}/grpo_adapters"
LOG_JSONL   = f"{BASE}/log_grpo.jsonl"
RESULT_MD   = f"{BASE}/notes/GRPO_RESULT.md"

# ═══════════ 超参 ═══════════
# 单一来源：configs/default.yaml（由 config.py 加载）
# 覆盖方式：环境变量 GRPO_LR=2e-4 GRPO_STEPS=40 ...
# 可用键见 tools/config.py 的 DEFAULTS
from config import CFG  # noqa: E402


def plan_batches(tasks, n_prompts, steps, seed=0):
    """按 epoch 遍历地规划每步抽哪些题（对应 agl 的 total_epochs 迭代）

    每个 epoch: 把全部题打乱一次，顺序取；取完再打乱进入下一个 epoch。
    对比"有放回抽样"：
      72 道题、30 步 x 4 = 120 次抽取
        有放回: 只覆盖 72*(1-e^(-120/72)) = 58 道（81%）
        遍历  : epoch1 用满 72 道 + epoch2 用 48 道 = 100% 覆盖
    """
    rng = random.Random(seed)
    order = []
    while len(order) < steps * n_prompts:
        ep = list(tasks)
        rng.shuffle(ep)
        order.extend(ep)
    return [order[i * n_prompts:(i + 1) * n_prompts]
            for i in range(steps)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def baseline_from_probe():
    """val 集合的基线：直接取探测结果（无额外成本）"""
    ids = [json.loads(l)["instance_id"] for l in open(VAL_FILE) if l.strip()]
    want = set(ids)
    per = {}
    for l in open(PROBE_FILE):
        r = json.loads(l)
        if r["instance_id"] in want:
            per.setdefault(r["instance_id"], []).append(r["resolved"])
    if not per:
        return None
    tot = sum(sum(v) for v in per.values())
    n = sum(len(v) for v in per.values())
    return tot / n, n, len(per)


def init_result_md(cfg, n_train, base):
    os.makedirs(f"{BASE}/notes", exist_ok=True)
    with open(RESULT_MD, "w") as f:
        f.write(f"# GRPO 训练结果\n\n")
        f.write(f"开始时间：{datetime.datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## 配置\n\n")
        f.write(f"| 参数 | 值 | 出处 |\n|---|---|---|\n")
        src = {
            "n_rollouts": "from agl", "n_prompts": "单卡适配", "max_turns": "from agl",
            "max_row_tokens": "单卡适配", "lr": "from agl", "clip_low": "from agl",
            "clip_high": "from agl", "temperature": "from agl", "steps": "单卡适配",
            "gpu_mem_util": "单卡适配", "max_new_tokens": "本项目设置",
            "max_num_seqs": "Qwen3.5 限制",
        }
        for k, v in cfg.items():
            if k in src:
                f.write(f"| `{k}` | {v} | {src[k]} |\n")
        f.write(f"\n训练题：{n_train} 道\n\n")
        if base:
            f.write(f"**基线**（val, {base[2]} 道 x 4 次探测）：**{base[0]*100:.1f}%**\n\n")
        f.write("## 曲线\n\n")
        f.write("| step | epoch | train solve | **val solve** | loss | clip% | 用时(s) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        if base:
            f.write(f"| 0 | 0.0 | — | **{base[0]*100:.1f}%** | — | — | — |\n")


def append_md(step, epoch, train_rate, val_rate, loss, clip, sec):
    vs = f"**{val_rate*100:.1f}%**" if val_rate is not None else "—"
    ts = f"{train_rate*100:.1f}%" if train_rate is not None else "—"
    with open(RESULT_MD, "a") as f:
        f.write(f"| {step} | {epoch:.1f} | {ts} | {vs} | {loss:+.4f} | {clip*100:.1f} | {sec:.0f} |\n")


def main():
    from transformers import AutoConfig
    import transformers
    from peft import LoraConfig, get_peft_model
    from grpo_llm import GRPOLlm
    from grpo_rollout import run_parallel_ids
    from grpo_util import build_row_lcp, masked_logprobs
    from eval_quick import quick_eval

    random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

    tasks = [json.loads(l) for l in open(TRAIN_FILE) if l.strip()]
    base = baseline_from_probe()
    log(f"训练题 {len(tasks)} 道；val 基线 {base[0]*100:.1f}%" if base else "val 基线未知")
    init_result_md(CFG, len(tasks), base)
    os.makedirs(ADAPTER_ROOT, exist_ok=True)

    # ═══ vLLM ═══
    llm = GRPOLlm(MODEL_PATH, max_model_len=81920, gpu_mem=CFG["gpu_mem_util"],
                  max_new_tokens=CFG["max_new_tokens"], max_loras=2,
                  max_num_seqs=CFG["max_num_seqs"])
    log(f"vLLM 就绪 显存 {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ═══ 训练侧 ═══
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    Cls = getattr(transformers, cfg.architectures[0])
    log("载入训练侧模型...")
    model = Cls.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
    lcfg = LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules="all-linear")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    log(f"训练侧显存 {torch.cuda.memory_allocated()/1e9:.1f}GB")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"])

    cur_dir = f"{ADAPTER_ROOT}/step000"
    os.makedirs(cur_dir, exist_ok=True)
    model.save_pretrained(cur_dir)
    cur_lora = llm.LoRARequest("s0", 1, cur_dir)
    log(f"初始适配器 -> {cur_dir}")

    # ⭐ val_before_train（对应 agl 的 trainer.val_before_train: True）
    #    ⚠️ 默认跳过（SKIP_BASELINE=1）：先证明配置能跑，配置冻结后再补基线
    #    step000 是零 LoRA（基座），纯推理，顺序无关 -> 基线可事后补
    base_rate = None
    if os.environ.get("SKIP_BASELINE", "1") == "1":
        log("跳过 val_before_train（SKIP_BASELINE=1）")
        log("  配置冻结后补测: python tools/eval_adapter.py --adapter grpo_adapters/step000 --n 115 --k 1 --tag base")
    else:
        log("═══ val_before_train（step 0, 全量 115 道）═══")
        try:
            base_rate, n_b, ok_b = quick_eval(llm, cur_lora, eval_file=FULL_EVAL,
                                              n_limit=CFG["eval_n"],
                                              temperature=CFG["val_temperature"],
                                              max_turns=CFG["max_turns"])
            log(f"  同配置基线 {ok_b}/{n_b} = {base_rate*100:.1f}%")
        except Exception as e:
            log(f"  ⚠️ 基线评测失败: {type(e).__name__}: {str(e)[:120]}")

    BATCHES = plan_batches(tasks, CFG["n_prompts"], CFG["steps"], CFG["seed"])
    seen = len({t["instance_id"] for b in BATCHES for t in b})
    log(f"批次规划: {CFG['steps']} 步 x {CFG['n_prompts']} 题 = "
        f"{CFG['steps']*CFG['n_prompts']} 次抽取，覆盖 {seen}/{len(tasks)} 道 "
        f"({seen/len(tasks)*100:.0f}%)，epoch={CFG['steps']*CFG['n_prompts']/len(tasks):.2f}")

    history = []
    for step in range(CFG["steps"]):
        t_step = time.time()
        ep = (step + 1) * CFG["n_prompts"] / max(len(tasks), 1)
        log(f"═══ step {step+1}/{CFG['steps']}  (epoch {ep:.2f}) ═══")

        # 1) 取题（按 epoch 遍历，不是有放回抽样）
        picked = BATCHES[step]
        jobs = [t for t in picked for _ in range(CFG["n_rollouts"])]
        t0 = time.time()
        states = run_parallel_ids(jobs, llm, max_turns=CFG["max_turns"],
                                  workers=min(24, len(jobs)), root="/tmp/grpo",
                                  lora_request=cur_lora,
                                  temperature=CFG["temperature"])
        n_ok = sum(1 for s in states if s["resolved"])
        train_rate = n_ok / max(len(states), 1)
        log(f"  rollout {len(states)} 条  成功 {n_ok} ({train_rate*100:.1f}%)  {time.time()-t0:.0f}s")

        # 2) 优势
        groups = {}
        for s in states:
            groups.setdefault(s["task"]["instance_id"], []).append(s)
        n_zv = 0
        for tid, g in groups.items():
            r = [1.0 if x["resolved"] else 0.0 for x in g]
            mu = sum(r) / len(r)
            sd = math.sqrt(sum((x - mu) ** 2 for x in r) / len(r))
            if sd < 1e-6:
                n_zv += 1
            for x, ri in zip(g, r):
                x["adv"] = (ri - mu) / (sd + 1e-4)
        log(f"  零方差组 {n_zv}/{len(groups)}")

        # 3) 拼行
        rows = []
        for s in states:
            got = build_row_lcp(s["turns"], max_tokens=CFG["max_row_tokens"])
            if got:
                rows.append((got[0], got[1], got[2], s["adv"]))
        if not rows:
            log("  ⚠️ 无训练行，跳过")
            append_md(step + 1, ep, train_rate, None, 0.0, 0.0, time.time() - t_step)
            continue
        log(f"  训练行 {len(rows)} 条  平均长 {sum(len(r[0]) for r in rows)/len(rows):.0f}  "
            f"平均 loss token {sum(sum(r[1]) for r in rows)/len(rows):.0f}")

        # 4) PPO 更新（per-rollout mean loss）
        t0 = time.time()
        # ⭐ 打乱（对应 verl dataloader 的 shuffle）
        #    否则 rows 按题分组 -> 每个 mini-batch 只含 2 道题的梯度
        # ⭐ 打乱（对应 verl dataloader 的 shuffle）
        random.shuffle(rows)

        # ⭐⭐ 用【训练模型】重算 old_logprobs
        #    verl 的 actor_rollout_ref.rollout.calculate_log_probs 默认 False
        #    -> 用 vLLM 返回的 logprob 会混入 "vLLM vs HF" 的实现差异（实测 ~2.3% token）
        packed = []
        with torch.no_grad():
            for (row, mask, olp, A) in rows:
                ntok = int(sum(mask))
                if ntok == 0:
                    continue
                lp_o, pos = masked_logprobs(model, row, mask)
                if lp_o is None:
                    continue
                vllm_lp = torch.tensor([olp[i] for i in pos.tolist()],
                                       dtype=torch.float32, device="cuda")
                packed.append((row, mask, pos, lp_o.detach(), vllm_lp, A))
        if not packed:
            log("  ⚠️ 无可用行，跳过本步")
            continue

        # 诊断（只在 step 1 打一次）：量化两个实现的 logprob 差异
        if step == 0:
            gap = torch.cat([(x[4] - x[3]).abs() for x in packed])
            log(f"  [诊断] |vLLM_logp - 训练模型_logp|  "
                f"mean={gap.mean():.4f}  p50={gap.median():.4f}  "
                f"p99={gap.quantile(0.99):.4f}  max={gap.max():.4f}")
            log(f"  [诊断] 差异 > 0.246（旧方案会误判为 clip）的比例: "
                f"{(gap > 0.246).float().mean()*100:.2f}%")

        mb = CFG["mini_batch_size"]
        n_mb = min(CFG["max_ppo_update_times"], max(1, len(packed) // mb))
        chunks = [packed[i * mb:(i + 1) * mb] for i in range(n_mb)]
        dropped = len(packed) - n_mb * mb
        if dropped > 0:
            log(f"  （{dropped} 行不满 mini-batch 被丢弃，对应 agl 的对齐丢弃）")

        tot_loss = tot_tok = clip_hits = n_used = 0
        first_ratio_reported = False
        for ci, chunk in enumerate(chunks):
            opt.zero_grad(set_to_none=True)
            for (row, mask, pos, lp_old, vllm_lp, A) in chunk:
                ntok = int(sum(mask))
                if ntok == 0:
                    continue
                lp_new, _ = masked_logprobs(model, row, mask)
                if lp_new is None:
                    continue
                ratio = torch.exp(lp_new - lp_old)
                if step == 0 and ci == 0 and not first_ratio_reported:
                    log(f"  [校验] 第1步第1批 ratio: mean={ratio.mean():.6f} "
                        f"min={ratio.min():.6f} max={ratio.max():.6f} "
                        f"（应≈1.000000，因为策略还没动）")
                    first_ratio_reported = True
                un = ratio * A
                cl = torch.clamp(ratio, 1 - CFG["clip_low"], 1 + CFG["clip_high"]) * A
                loss = (-torch.min(un, cl)).sum() / ntok
                (loss / len(chunk)).backward()
                tot_loss += loss.item(); tot_tok += ntok; n_used += 1
                clip_hits += int((torch.abs(ratio - 1) >
                                  max(CFG["clip_low"], CFG["clip_high"])).sum().item())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        log(f"  做了 {len(chunks)} 次 mini-batch 更新（每批 {mb} 行）")
        avg_loss = tot_loss / max(n_used, 1)
        clip_frac = clip_hits / max(tot_tok, 1)
        log(f"  更新 {n_used} 行/{tot_tok} token  loss {avg_loss:+.4f}  clip {clip_frac*100:.1f}%  {time.time()-t0:.0f}s")

        # 5) 存适配器
        new_dir = f"{ADAPTER_ROOT}/step{step+1:03d}"
        shutil.rmtree(new_dir, ignore_errors=True)
        model.save_pretrained(new_dir)
        cur_lora = llm.LoRARequest(f"s{step+1}", step + 2, new_dir)

        # 6) 中途评测 + 写文档
        val_rate = None
        if (step + 1) % CFG["eval_every"] == 0 or (step + 1) == CFG["steps"]:
            log("  ── 中途评测 ──")
            try:
                val_rate, n_ev, ok_ev = quick_eval(llm, cur_lora, eval_file=FULL_EVAL,
                                                   n_limit=CFG["eval_n"],
                                                   temperature=CFG["val_temperature"],
                                                   max_turns=CFG["max_turns"])
                log(f"  val solve {ok_ev}/{n_ev} = {val_rate*100:.1f}%  "
                    f"(基线 {base[0]*100:.1f}%)" if base else "")
            except Exception as e:
                log(f"  ⚠️ 评测失败: {type(e).__name__}: {str(e)[:120]}")

        sec = time.time() - t_step
        append_md(step + 1, ep, train_rate, val_rate, avg_loss, clip_frac, sec)
        rec = dict(step=step + 1, epoch=round(ep, 3), train_rate=train_rate,
                   val_rate=val_rate, n_rollout=len(states), n_rows=len(rows),
                   n_zero_var=n_zv, loss=avg_loss, clip_frac=clip_frac, sec=sec)
        history.append(rec)
        with open(LOG_JSONL, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"  step 用时 {sec:.0f}s")

    # ═══ 最终评测（115 道 held-out）═══
    log("═══ 最终评测（115 道）═══")
    final_rate = None
    try:
        final_rate, n_f, ok_f = quick_eval(llm, cur_lora, eval_file=FULL_EVAL,
                                           n_limit=CFG["eval_n"],
                                           temperature=CFG["val_temperature"],
                                           max_turns=CFG["max_turns"])
        log(f"  最终 val {ok_f}/{n_f} = {final_rate*100:.1f}%")
        with open(RESULT_MD, "a") as f:
            f.write(f"\n## 最终评测（115 道 held-out）\n\n")
            f.write(f"- 结果：**{ok_f}/{n_f} = {final_rate*100:.1f}%**\n")
            if base:
                f.write(f"- （注意：早期探测基线 {base[0]*100:.1f}% 是 max_turns=25 配置，不可直接比）\n")
    except Exception as e:
        log(f"  ⚠️ 最终评测失败: {type(e).__name__}: {str(e)[:120]}")

    # ═══ 收尾 ═══
    with open(RESULT_MD, "a") as f:
        f.write(f"\n结束时间：{datetime.datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write("## 结论\n\n")
        if base:
            f.write(f"- 基线 val solve: **{base[0]*100:.1f}%**\n")
        last = [h for h in history if h.get("val_rate") is not None]
        if last:
            f.write(f"- 最终 val solve: **{last[-1]['val_rate']*100:.1f}%**\n")
            if base:
                d = (last[-1]["val_rate"] - base[0]) * 100
                f.write(f"- 变化: **{d:+.1f} pp**\n")
        f.write(f"\n适配器目录：`{ADAPTER_ROOT}`\n")

    log("=" * 60)
    log("训练完成")
    for h in history:
        vr = f"{h['val_rate']*100:.1f}%" if h.get("val_rate") is not None else "—"
        log(f"  step {h['step']:2}  train {h['train_rate']*100:5.1f}%  val {vr:>6}  "
            f"loss {h['loss']:+.4f}  {h['sec']:.0f}s")
    log(f"结果文档: {RESULT_MD}")


if __name__ == "__main__":
    main()
