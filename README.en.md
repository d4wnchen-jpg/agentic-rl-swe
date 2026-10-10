# Agentic RL on a Single GPU

[中文](README.md) | **English**

[![CI](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml/badge.svg)](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-1%C3%97%20RTX%20PRO%206000%2096GB-76B900.svg)]()
[![model](https://img.shields.io/badge/model-Qwen3.5--9B-8A2BE2.svg)]()
[![method](https://img.shields.io/badge/method-GRPO%20%2B%20LoRA-FF6F00.svg)]()

The full agentic RL loop on one 96GB GPU, in a container without Docker.

[Implementation](docs/IMPLEMENTATION.md) · [Results](docs/RESULTS.md) · [Lessons](docs/LESSONS.en.md) · [MIT License](LICENSE)

## ⚡ What this is

Training Qwen3.5-9B on SWE-smith tasks: sandbox → multi-turn rollout → grading → GRPO → weight reload.
LoRA + vLLM, with inference and training sharing one RTX PRO 6000.

| Constraint | |
|---|---|
| GPU | 1× 96GB |
| Environment | AutoDL container, no Docker |
| Model | Qwen3.5-9B (VL + hybrid linear attention) |
| Compute budget | Tens of hours |

## ⚡ Headline result

20 steps, 640 rollouts. Held-out pass rate 23.9% → 25.7%, a difference within noise.

74% of the eval set is beyond the base model, so the tier the training actually improved
(mixed) went from **5/17 to 9/17**. The total is diluted.

## ⚡ Two pitfalls that cost the most

**The two renderings of a chat template must agree.** Qwen3.5's native template uses one
code path at the generation point and another in history rendering, so consecutive turns
do not form a token prefix chain, and the prompt tail keeps a complete empty thinking
block that can be copied wholesale. After training, **90% of rollouts echoed it**. Fixed
with [`tools/tpl_chat.jinja`](tools/tpl_chat.jinja): repetition **90.3% → 0%**.

**Do not let the model depend on the grading tests.** When the test files are visible it
runs the suite; pytest emits tens of thousands of characters, `obs_cap` truncates that to
roughly the last 66 tests, and the model never gets a signal that it is done — **99% of
rollouts hit the turn limit**. The reference implementation removes the F2P test files
instead, and its prompt tells the model to write a minimal reproduction script.

Full analysis and the other four pitfalls: [LESSONS.md](docs/LESSONS.en.md).

## ⚡ Install

```bash
pip install -r requirements.txt
pip install vllm==0.30.0

export SWESMITH_RL_HOME=$(pwd)
bash scripts/run_train.sh
```

Needs PyTorch ≥ 2.9 (sm_120 support). Hyperparameters have a single source:
[configs/default.yaml](configs/default.yaml).

Evaluate any saved adapter:

```bash
python tools/eval_adapter.py --adapter grpo_adapters/step000 --n 115 --k 1 --tag base
```

## ⚡ The loop

```
┌──────────────────────────────────────────────────────────────────┐
│ ① Build sandbox                                                  │
│   checkout bug branch → apply patch → remove .git (anti-cheat)   │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ② Rollout (vLLM + LoRA, up to 40 turns)                          │
│   model emits an action → sandbox runs it → observation back     │
│   ⭐ record prompt_ids / response_ids / per-token logprob        │
│   ⭐ never trim history (keeps the token prefix chain)           │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ③ Grade (once, when the task ends)                               │
│   pytest --color=no -rA, COLUMNS=200                             │
│   restore the deleted F2P tests                                  │
│   reward = 1.0 if all F2P + P2P pass                             │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ④ Advantage (rollout level, normalised within group)             │
│   A_i = (r_i - mean(r_1..r_8)) / (std + 1e-4)                    │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑤ Assemble training row (LCP prefix alignment)                   │
│   40 turns → one ~25k-token sequence                             │
│   ~18× cheaper than one row per turn                             │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑥ PPO update (2 mini-batches, per-rollout mean loss)             │
│   chunked logprob, recomputed on the backward pass               │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑦ Save LoRA                                                      │
│   new LoRARequest → back to ②                                    │
└──────────────────────────────────────────────────────────────────┘
```

## ⚡ Results

| Item | Result |
|---|---|
| Steps / training volume | 20 steps · 640 rollouts (reference: 128,000) |
| Memory | vLLM 43 GB + training 45 GB / 96 GB |
| Pass rate (113 held-out) | 23.9% → 25.7% |
| mixed tier pass rate | 5/17 → 9/17 |
| Weight change | ΔW/W ≈ 0.9% (LoRA) |

Detecting a change of 12 pp would need roughly 24× the training volume (after correcting
for zero-variance groups).

## ⚡ Documentation

| Doc | Content |
|---|---|
| [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | Steps, single-GPU adaptations, the three chat template problems |
| [DATA_FORMAT.md](docs/DATA_FORMAT.md) | Task fields, directory layout, environment variables |
| [RESULTS.md](docs/RESULTS.md) | Measurements, compute thresholds, the full degeneration analysis |
| [LESSONS.en.md](docs/LESSONS.en.md) | Six pitfalls and what to change |

The first three are Chinese only.

## ⚡ Tests

```bash
pytest tests/ -q     # 55 cases, CPU only
```

| File | Covers |
|---|---|
| `tests/test_row_lcp.py` | LCP alignment: no response lost, mask and logprob stay aligned |
| `tests/test_config.py` | YAML matches defaults, environment overrides |
| `tests/test_template.py` | Prefix chain and repetition trigger for both chat templates |
| `tests/test_audit_trajectory.py` | The six trajectory classifications |

## ⚡ Switches

```bash
SMITH_HARNESS_V1=1        # fall back to the old configuration (reproduces pitfall 2)
SMITH_ENABLE_THINKING=0   # pass enable_thinking=False, which leaves the empty block
SMITH_OBS_CHAR_CAP=10000  # per-observation character cap
GRPO_LR=3e-4 GRPO_STEPS=40  # override any hyperparameter
```

## ⚡ References

- **agent-lightning** — Microsoft, [examples/swe_smith](https://github.com/microsoft/agent-lightning/tree/main/examples/swe_smith)
  (Qwen3.5-9B / 4×B200 / full fine-tune + FSDP). Hyperparameters follow it first;
  `tools/tpl_chat_agl.jinja` is adapted from its `swe_smith_chat_template.jinja` (MIT).
- **SWE-smith** — task generation, [arXiv:2504.21798](https://arxiv.org/abs/2504.21798)
- **verl** — PPO/GRPO training framework, [GitHub](https://github.com/volcengine/verl)

## ⚡ License

MIT
