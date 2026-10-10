# Agentic RL on a Single GPU

**中文** | [English](README.en.md)

[![CI](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml/badge.svg)](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-1%C3%97%20RTX%20PRO%206000%2096GB-76B900.svg)]()
[![model](https://img.shields.io/badge/model-Qwen3.5--9B-8A2BE2.svg)]()
[![method](https://img.shields.io/badge/method-GRPO%20%2B%20LoRA-FF6F00.svg)]()

在单张 96GB 显卡、无 Docker 的容器里跑通 agentic RL 全流程。

[实施步骤](docs/IMPLEMENTATION.md) · [实验数据](docs/RESULTS.md) · [踩坑记录](docs/LESSONS.md) · [MIT License](LICENSE)

## ⚡ 这是什么

用 SWE-smith 任务训练 Qwen3.5-9B：沙箱 → 多轮 rollout → 判分 → GRPO → 权重回灌。
LoRA + vLLM，推理侧与训练侧共用一张 RTX PRO 6000。

| 约束 | |
|---|---|
| GPU | 单张 96GB |
| 环境 | AutoDL 容器，无 Docker |
| 模型 | Qwen3.5-9B（VL + 线性注意力混合架构）|
| 算力预算 | 数十小时 |

## ⚡ 关键结论

20 步训练用了 640 条 rollout。held-out 通过率 23.9% → 25.7%，差异在噪声内。

这不是"训练没用"：评测集 74% 是基座做不动的题，而训练真正提升的那一层（mixed）
从 **5/17 涨到 9/17**。总分被稀释了。

## ⚡ 两个最容易踩的坑

**chat template 的两处渲染必须一致。** Qwen3.5 原生模板在生成点和历史渲染用两套逻辑，
相邻两轮不构成 token 前缀链，prompt 尾部还留下一个可整段复制的空思考块。
实测训练后 **90% 的 rollout 复读**它。用 [`tools/tpl_chat.jinja`](tools/tpl_chat.jinja)
修复，复读 **90.3% → 0%**。

**不要让模型依赖判分测试。** 测试文件对模型开放时，它会去跑测试套件；
pytest 输出几万字符，被 `obs_cap` 截断到只剩最后约 66 个测试 ——
拿不到"我完成了"的信号，轮数耗尽 **99%**。参照实现的做法是移除 F2P 测试文件，
并在题面引导模型自己写最小复现脚本。

完整推导与其余四个坑见 [LESSONS.md](docs/LESSONS.md)。

## ⚡ 安装

```bash
pip install -r requirements.txt
pip install vllm==0.30.0

export SWESMITH_RL_HOME=$(pwd)
bash scripts/run_train.sh
```

需要 PyTorch ≥ 2.9（sm_120 支持）。超参的单一来源是 [configs/default.yaml](configs/default.yaml)。

评测任意存档的适配器：

```bash
python tools/eval_adapter.py --adapter grpo_adapters/step000 --n 115 --k 1 --tag base
```

## ⚡ 回路

```
┌──────────────────────────────────────────────────────────────────┐
│ ① 造沙箱                                                         │
│   checkout 缺陷分支 → 打补丁 → 移除 .git等防作弊                 │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ② Rollout（vLLM + LoRA，最多 40 轮）                             │
│   模型出动作 → 沙箱执行 → 观察拼回                               │
│   ⭐ 记录 prompt_ids / response_ids / per-token logprob          │
│   ⭐ 不裁剪历史（保证 token 级前缀链）                           │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ③ 判分（任务结束时一次）                                         │
│   pytest --color=no -rA，COLUMNS=200                             │
│   恢复被删的 F2P 测试                                            │
│   reward = 1.0 if 全部 F2P + P2P 通过                            │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ④ 优势（rollout 级，组内归一化）                                 │
│   A_i = (r_i - mean(r_1..r_8)) / (std + 1e-4)                    │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑤ 拼训练行（LCP 前缀链对齐）                                     │
│   40 轮拼成一条约 25k token 的序列                               │
│   比“每轮一行”省约 18 倍                                         │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑥ PPO 更新（2 次 mini-batch，per-rollout mean loss）             │
│   分块 + 反向重算的 logprob                                      │
└──────────────────────────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑦ 存 LoRA                                                        │
│   换成新的 LoRARequest → 回 ②                                    │
└──────────────────────────────────────────────────────────────────┘
```

## ⚡ 结果

| 项 | 结果 |
|---|---|
| 步数 / 训练量 | 20 步 · 640 条 rollout（参照实现：128,000 条）|
| 显存占用 | vLLM 43 GB + 训练侧 45 GB / 96 GB |
| 通过率（113 道 held-out）| 23.9% → 25.7% |
| mixed 层通过率 | 5/17 → 9/17 |
| 权重变化 | ΔW/W ≈ 0.9%（LoRA）|

要检测 12 pp 量级的变化，训练量需要约 24 倍（含零方差组修正）。

## ⚡ 文档

| 文档 | 内容 |
|---|---|
| [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | 实施步骤、单卡适配、chat template 的三个问题 |
| [DATA_FORMAT.md](docs/DATA_FORMAT.md) | 任务字段、目录布局、环境变量 |
| [RESULTS.md](docs/RESULTS.md) | 实验数据、算力门槛、生成退化的完整推导 |
| [LESSONS.md](docs/LESSONS.md) | 六个坑与改进建议 |

## ⚡ 测试

```bash
pytest tests/ -q     # 55 个用例，纯 CPU
```

| 文件 | 覆盖 |
|---|---|
| `tests/test_row_lcp.py` | LCP 对齐：response 不丢失、mask 与 logprob 不错位 |
| `tests/test_config.py` | YAML 与默认值一致、环境变量覆盖 |
| `tests/test_template.py` | 两个 chat template 的前缀链、复读触发条件 |
| `tests/test_audit_trajectory.py` | 轨迹审计的六类判定 |

## ⚡ 配置开关

```bash
SMITH_HARNESS_V1=1        # 回退到旧配置（复现坑 2 用），默认是修复后的配置
SMITH_ENABLE_THINKING=0   # 传 enable_thinking=False（会让原生模板留下空思考块）
SMITH_OBS_CHAR_CAP=10000  # 单条 observation 的字符上限
GRPO_LR=3e-4 GRPO_STEPS=40  # 覆盖任意超参
```

## ⚡ 参考来源

- **agent-lightning** — Microsoft，[examples/swe_smith](https://github.com/microsoft/agent-lightning/tree/main/examples/swe_smith)
  （Qwen3.5-9B / 4×B200 / 全参 + FSDP）。本项目的超参优先对齐该实现；
  `tools/tpl_chat_agl.jinja` 改编自其 `swe_smith_chat_template.jinja`（MIT）。
- **SWE-smith** — 任务生成，[arXiv:2504.21798](https://arxiv.org/abs/2504.21798)
- **verl** — PPO/GRPO 训练框架，[GitHub](https://github.com/volcengine/verl)

## ⚡ License

MIT
