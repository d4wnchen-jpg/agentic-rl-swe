# Agentic RL on a Single GPU

**中文** | [English](README.en.md)

[![CI](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml/badge.svg)](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-1%C3%97%20RTX%20PRO%206000%2096GB-76B900.svg)]()
[![model](https://img.shields.io/badge/model-Qwen3.5--9B-8A2BE2.svg)]()
[![method](https://img.shields.io/badge/method-GRPO%20%2B%20LoRA-FF6F00.svg)]()

在单张 96GB 显卡上跑通完整的 agentic RL 回路：多轮工具调用 → 沙箱判分 → GRPO 训练 → 权重回灌。

---

## 项目定位

约束条件：

| 项 | 限制 |
|---|---|
| GPU | 单张 96GB（RTX PRO 6000） |
| 环境 | AutoDL 容器，无 Docker |
| 模型 | Qwen3.5-9B（VL + 线性注意力混合架构） |
| 算力预算 | 数十小时量级 |

目标是把回路跑通并固定工程细节，不是刷指标。

产出：一条可运行的回路、一套单卡上让 vLLM 与训练侧共存的配置、一份可复现的步骤与验证清单。

---

## 回路

```
① 准备沙箱      checkout 缺陷分支 → 移出 .git
        ↓
② Rollout       最多 40 轮工具调用；记录 prompt/response token ids 与 per-token logprob
        ↓
③ 判分          pytest；F2P 全通过且 P2P 全通过 → reward = 1.0
        ↓
④ 优势          rollout 级，组内归一化
        ↓
⑤ 拼训练行      LCP 前缀链对齐，多轮合成一条序列
        ↓
⑥ 训练          分块 logprob + PPO 裁剪，每批 2 次 mini-batch 更新
        ↓
⑦ 权重回灌      保存 LoRA → 换 LoRARequest → 回到 ②
```

---

## 结论

| 项 | 结果 |
|---|---|
| 回路 | 20 步完整执行，每步 32 条 rollout，约 27 分钟/步 |
| 显存占用 | vLLM 43 GB + 训练侧 45 GB = 88 GB / 96 GB |
| 训练行拼接 | 32 / 32 全部成功 |
| 通过率（113 道 held-out） | 基线 30.1% → 训练后 27.4%，SE = 4.3 pp |
| 训练量 | 640 条 rollout；同类参照实现为 128,000 条 |
| 权重变化 | ΔW/W ≈ 0.9%（LoRA） |

回路可运行。在本项目的算力预算内，held-out 通过率没有可测量的变化。

要检测 12 pp 量级的变化，训练量需要约 15 倍（~45 小时连续训练）。详见 [RESULTS.md](docs/RESULTS.md)。

---

## 快速开始

```bash
pip install -r requirements.txt
pip install vllm==0.30.0

export SWESMITH_RL_HOME=$(pwd)
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python tools/train_grpo.py
```

评测任意存档的适配器：

```bash
python tools/eval_adapter.py --adapter grpo_adapters/step000 --n 115 --k 1 --tag base
```

`step000` 是零 LoRA（等价基座），用于测同配置基线。

---

## 配置

超参来源为 `configs/default.yaml`：

```
configs/default.yaml  ──►  tools/config.py  ──►  tools/train_grpo.py
                              ▲
                   环境变量覆盖
```

```bash
GRPO_LR=3e-4 GRPO_STEPS=40 python tools/train_grpo.py
```

`tests/test_config.py` 校验 YAML 与代码默认值一致。未安装 `pyyaml` 时加载直接报错。

---

## 测试

```bash
pytest tests/ -q     # 30 个用例，纯 CPU
```

| 测试 | 覆盖 |
|---|---|
| `tests/test_row_lcp.py` | LCP 对齐：response 不丢失、mask 与 logprob 不错位、超长截断保留题面 |
| `tests/test_config.py` | YAML 与默认值一致、环境变量覆盖、脚本用到的键均已定义 |

---

## 关键实现

| 点 | 做法 |
|---|---|
| 单卡共存 | LoRA（59 MB 适配器）替代全参；vLLM `gpu_memory_utilization=0.45` |
| 权重回灌 | 保存适配器 → 新建 `LoRARequest`（新目录 + 新 `lora_int_id`） |
| 多轮拼行 | LCP 对齐，40 轮合成一条 25k token 序列；比逐轮拼节省约 18 倍 |
| logprob 显存 | 只算 loss 位置 + 分块（512）+ `torch.utils.checkpoint` 反向重算 |
| PPO 裁剪 | 每批 2 次 mini-batch 更新，训练行先打乱 |
| 采样一致性 | `old_logprobs` 由训练模型重算；首步首批 `ratio` 应为 1.000000 |
| 评测 | 全部 held-out，温度 0.7；基线用 `step000` 同配置测量 |

实施步骤见 [IMPLEMENTATION.md](docs/IMPLEMENTATION.md)。

---

## 超参

| 参数 | 值 | 来源 |
|---|---|---|
| `n_rollouts` | 8 | 参照实现 |
| `max_turns` | 40 | 参照实现 |
| `max_new_tokens` | 16384 | 参照实现 |
| `temperature` / `val_temperature` | 1.0 / 0.7 | 参照实现 |
| `clip_low` / `clip_high` | 0.2 / 0.28 | 参照实现 |
| `mini_batch_size` | 16 | 参照实现 |
| `max_ppo_update_times` | 2 | 参照实现 |
| `lr` | 1e-4 | LoRA 标定（全参为 1e-6） |
| `n_prompts` / `steps` | 4 / 20 | 单卡适配（参照：16 / 1000） |
| `gpu_mem_util` | 0.45 | 单卡适配（参照：0.8） |
| `max_num_seqs` | 128 | Qwen3.5 Mamba cache 约束 |
| `max_row_tokens` | 32768 | 单卡适配（参照：65536） |

完整表见 [configs/default.yaml](configs/default.yaml)。

---

## 文档

| 文档 | 内容 |
|---|---|
| [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | 实施步骤 |
| [DATA_FORMAT.md](docs/DATA_FORMAT.md) | 任务字段、目录布局、环境变量 |
| [RESULTS.md](docs/RESULTS.md) | 实验数据与算力门槛 |

---

## 参考来源

- **agent-lightning** — Microsoft，[examples/swe_smith](https://github.com/microsoft/agent-lightning/tree/main/examples/swe_smith)
  （Qwen3.5-9B / 4×B200 / 全参 + FSDP）。本项目的超参优先对齐该实现。
- **SWE-smith** — 任务生成，[arXiv:2504.21798](https://arxiv.org/abs/2504.21798)
- **verl** — PPO/GRPO 训练框架，[GitHub](https://github.com/volcengine/verl)

---

## License

MIT
