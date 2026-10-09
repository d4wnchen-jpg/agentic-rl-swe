# Agentic RL on a Single GPU

[中文](README.md) | **English**

[![CI](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml/badge.svg)](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-1%C3%97%20RTX%20PRO%206000%2096GB-76B900.svg)]()
[![model](https://img.shields.io/badge/model-Qwen3.5--9B-8A2BE2.svg)]()
[![method](https://img.shields.io/badge/method-GRPO%20%2B%20LoRA-FF6F00.svg)]()

A working agentic RL loop on one 96GB GPU: multi-turn tool use → sandboxed grading → GRPO training → weight round-trip.

---

## Scope

Constraints:

| | |
|---|---|
| GPU | Single 96GB (RTX PRO 6000) |
| Environment | Container, no Docker |
| Model | Qwen3.5-9B (VL + linear-attention hybrid) |
| Compute budget | Tens of hours |

The goal is a working loop with the engineering pinned down, not benchmark numbers.

Deliverables: a runnable loop, a configuration that lets vLLM and the training model share one card, and a reproducible procedure with a pre-flight checklist.

---

## The loop

```
① Sandbox       checkout the bug branch → move .git out of the worktree
        ↓
② Rollout       up to 40 tool-call turns; record prompt/response token ids and per-token logprobs
        ↓
③ Grading       pytest; reward = 1.0 iff all F2P and all P2P pass
        ↓
④ Advantage     rollout-level, normalized within each group
        ↓
⑤ Row assembly  LCP prefix alignment; all turns merged into a single sequence
        ↓
⑥ Training      chunked logprobs + PPO clipping, 2 mini-batch updates per batch
        ↓
⑦ Weight sync   save LoRA → new LoRARequest → back to ②
```

---

## Results

| | |
|---|---|
| Loop | 20 steps completed; 32 rollouts per step; ~27 min/step |
| Memory | vLLM 43 GB + training 45 GB = 88 GB / 96 GB |
| Row assembly | 32 / 32 succeeded |
| Pass rate (113 held-out) | baseline 30.1% → 27.4% after training, SE = 4.3 pp |
| Training volume | 640 rollouts; the reference implementation uses 128,000 |
| Weight delta | ΔW/W ≈ 0.9% (LoRA) |

The loop runs. Within this project's compute budget, the held-out pass rate shows no measurable change.

Detecting a 12 pp effect would need roughly 15× more training (~140 hours, about 6 days, continuous). See [RESULTS.md](docs/RESULTS.md).

---

## Suggested improvements

Each item is grounded in measurements from this run.

| # | Improvement | Evidence | Cost |
|---|---|---|---|
| **1** | **Raise training volume to ≥10,000 rollouts** (currently 640) | Detecting a 12 pp change at this eval size needs roughly 15x more training | ~140 h on one GPU (about 6 days) |
| **2** | **Grow the eval set to ~150 tasks** (currently 113) | SE drops from 4.3 pp to ~3.7 pp | eval time only |
| **3** | **Take larger policy steps** (higher lr or more steps) | `clip_frac` stayed at 0.0–0.2% throughout: clipping never fired, so PPO degenerated into REINFORCE | free, needs tuning |
| **4** | **Make the task pool dynamic** — retire tasks the model reliably solves | Zero-variance groups (all 8 correct or all 8 wrong) give advantage 0 and produce no gradient; the training set still holds 12 all-fail tasks | small |
| **5** | **Scale the dataset to thousands of tasks** (currently 187) | The reference implementation uses 6,000; the mixed-task ratio (41%) drives GRPO's sample efficiency | data preparation |
| **6** | **Average k>1 eval samples** | Eval temperature is 0.7, so a single sample is noisy | eval time x k |

**Priority: 1 > 4 > 3 > 2 > 5 > 6**

Item 1 is the only change that can make a *result* possible; the rest improve efficiency or reduce noise.
Until it is done, the conclusion here stays engineering-level: **the loop runs, the effect is not measurable.**

> Item 4 is the idea behind DAPO's Dynamic Sampling ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)).

---

## Quick start

```bash
pip install -r requirements.txt
pip install vllm==0.30.0

export SWESMITH_RL_HOME=$(pwd)
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python tools/train_grpo.py
```

Evaluate any saved adapter:

```bash
python tools/eval_adapter.py --adapter grpo_adapters/step000 --n 115 --k 1 --tag base
```

`step000` is a zero LoRA (equivalent to the base model), used to measure a baseline under identical settings.

---

## Configuration

Hyperparameters live in `configs/default.yaml`:

```
configs/default.yaml  ──►  tools/config.py  ──►  tools/train_grpo.py
                              ▲
                   environment overrides
```

```bash
GRPO_LR=3e-4 GRPO_STEPS=40 python tools/train_grpo.py
```

`tests/test_config.py` checks that the YAML matches the code defaults. Loading fails loudly if `pyyaml` is missing.

---

## Tests

```bash
pytest tests/ -q     # 30 cases, CPU only
```

| Test | Coverage |
|---|---|
| `tests/test_row_lcp.py` | LCP alignment: no dropped responses, mask and logprob stay aligned, truncation keeps the task statement |
| `tests/test_config.py` | YAML matches defaults, environment overrides work, every key used by the training script is defined |

---

## Implementation highlights

| Area | Approach |
|---|---|
| Single-card coexistence | LoRA (59 MB adapters) instead of full fine-tuning; vLLM `gpu_memory_utilization=0.45` |
| Weight sync | Save adapter → new `LoRARequest` (fresh directory and `lora_int_id`) |
| Multi-turn row assembly | LCP alignment; 40 turns merged into one ~25k-token sequence, ~18× cheaper than per-turn rows |
| Logprob memory | Loss positions only, chunked (512), with `torch.utils.checkpoint` recomputation |
| PPO clipping | 2 mini-batch updates per batch, rows shuffled first |
| Sampling consistency | `old_logprobs` recomputed by the training model; first batch of step 1 should give `ratio == 1.000000` |
| Evaluation | Full held-out set, temperature 0.7; baseline measured with `step000` under identical settings |

Procedure details in [IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

---

## Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| `n_rollouts` | 8 | reference implementation |
| `max_turns` | 40 | reference implementation |
| `max_new_tokens` | 16384 | reference implementation |
| `temperature` / `val_temperature` | 1.0 / 0.7 | reference implementation |
| `clip_low` / `clip_high` | 0.2 / 0.28 | reference implementation |
| `mini_batch_size` | 16 | reference implementation |
| `max_ppo_update_times` | 2 | reference implementation |
| `lr` | 1e-4 | calibrated for LoRA (1e-6 for full fine-tuning) |
| `n_prompts` / `steps` | 4 / 20 | single-GPU adaptation (reference: 16 / 1000) |
| `gpu_mem_util` | 0.45 | single-GPU adaptation (reference: 0.8) |
| `max_num_seqs` | 128 | Qwen3.5 Mamba cache limit |
| `max_row_tokens` | 32768 | single-GPU adaptation (reference: 65536) |

Full table in [configs/default.yaml](configs/default.yaml).

---

## Documentation

| Document | Contents |
|---|---|
| [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | Step-by-step procedure |
| [DATA_FORMAT.md](docs/DATA_FORMAT.md) | Task fields, directory layout, environment variables |
| [RESULTS.md](docs/RESULTS.md) | Measurements and compute threshold |

---

## References

- **agent-lightning** — Microsoft, [examples/swe_smith](https://github.com/microsoft/agent-lightning/tree/main/examples/swe_smith)
  (Qwen3.5-9B / 4×B200 / full fine-tuning with FSDP). Hyperparameters here follow this implementation first.
- **SWE-smith** — task generation, [arXiv:2504.21798](https://arxiv.org/abs/2504.21798)
- **verl** — PPO/GRPO training framework, [GitHub](https://github.com/volcengine/verl)

---

## License

MIT
