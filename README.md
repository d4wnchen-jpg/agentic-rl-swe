# Agentic RL on a Single GPU

**中文** | [English](README.en.md)

[![CI](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml/badge.svg)](https://github.com/d4wnchen-jpg/agentic-rl-swe/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-1%C3%97%20RTX%20PRO%206000%2096GB-76B900.svg)]()
[![model](https://img.shields.io/badge/model-Qwen3.5--9B-8A2BE2.svg)]()
[![method](https://img.shields.io/badge/method-GRPO%20%2B%20LoRA-FF6F00.svg)]()

在单张 96GB 显卡、无 Docker 的容器环境里跑通完整的 agentic RL 回路：
构造数据 → 沙箱 → 多轮 rollout → 判分 → GRPO 训练 → 权重回灌 → 结果分析。

---

## 项目定位

约束条件：

| 项 | 限制 |
|---|---|
| GPU | 单张 96GB（RTX PRO 6000） |
| 环境 | AutoDL 容器，**无 Docker**（seccomp 禁了 `unshare`） |
| 模型 | Qwen3.5-9B（VL + 线性注意力混合架构） |
| 算力预算 | 数十小时量级 |

**目标**：有限算力下完成 agentic RL 的整个流程，熟悉每个部分，并对结果做分析。

**产出**：

- 一条可运行的回路（含无 Docker 环境下的沙箱方案）
- 一套单卡上让 vLLM 与训练侧共存的配置
- 一份结果分析：逐题翻转、失败原因统计、生成退化定位
- 可复用的工具与测试（39 个用例 + 一个适配版的 chat template）

---

## 数据

| 步骤 | 结果 |
|---|---|
| 从 SWE-smith 导出 `func_basic` 任务 | 4 个仓库 |
| 可执行性验证 | 187 / 187 全部通过 |
| 难度探测（每题 4 次 rollout，748 条）| 通过率 25.9% |
| 按探测结果分层 | mixed **77** / allpass 14 / allfail 96 |

**切分构成**：

| | mixed | allpass | allfail | 合计 |
|---|---|---|---|---|
| **train** | 60 | 0 | 12 | **72** |
| **eval** | 17 | 14 | 84 | **115** |
加入少量allfail，模型提升也许能做对。

详见 [DATA_FORMAT.md](docs/DATA_FORMAT.md)。

---

## 回路

**① 准备沙箱** —— 复制仓库副本 → 注入缺陷补丁 → 删掉 `.git`。
没有 Docker，隔离靠独立目录而不是容器边界。防作弊四层：补丁写在 `/tmp` 且应用后立即删除、
删除 `.git`、命令正则拦截（git / 网络 / 装包 / 改测试）、判分前从原仓库恢复测试文件。

**② Rollout**（≤40 轮）—— 模型出动作 → 沙箱执行 → 观察拼回。
记录 **token ids** 而不是文本（RL 要在 token 级算概率）；**不裁剪历史**（一裁就不是前缀链）。

**③ 判分** —— pytest 跑 F2P + P2P，全通过则 reward = 1.0。
`COLUMNS=200` 防止 `PASSED` 折行导致假失败；同时检查 `n_fail` 与 `n_missing`。

**④ 优势** —— rollout 级，组内归一化。零方差组（8 条全对或全错）优势恒为 0。

**⑤ 拼训练行** —— 40 轮拼成一条约 25k token 的序列。
生成提示符与历史渲染不一致，用最长公共前缀对齐；历史 response 标回 `mask=1`。

**⑥ 训练** —— 梯度累积 16 条后更新一次，共 2 批。
micro-batch = 1（各条长度不一）；两级归一化；`old_logprobs` 由训练模型重算。

**⑦ 权重回灌** —— 保存 LoRA → 新建 `LoRARequest`。
必须换新的 `lora_int_id`，否则 vLLM 命中缓存，用的还是旧权重。

实施步骤见 [IMPLEMENTATION.md](docs/IMPLEMENTATION.md)。

---

## 结论

同一次评测内对比基座与训练后（113 道 held-out，40 轮，温度 0.7）：

| 项 | 结果 |
|---|---|
| 回路 | 20 步完整执行，每步 32 条 rollout，约 27 分钟/步 |
| 显存占用 | vLLM 43 GB + 训练侧 45 GB = 88 GB / 96 GB |
| **通过率** | 基座 27/113 = **23.9%** → step020 29/113 = **25.7%** |
| 训练量 | 640 条 rollout（参照实现为 128,000 条）|
| 权重变化 | ΔW/W ≈ 0.9%（LoRA）|

**这 1.8 pp 的差异不能算作提升** —— 同一配置重跑一次会得到相反方向的结论。

要检测 12 pp 量级的变化，训练量需要约 15 倍（~140 小时，约 6 天连续训练）。

---

## 结果分析

### 逐题翻转

| | 道数 |
|---|---|
| fail → pass | 13 |
| pass → fail | 11 |
| 两个都对 | 16 |
| 两个都错 | 73 |

净变化 +2，但总翻转 24 —— **翻转量是净变化的 12 倍**。变化主要是噪声，不是学习。

### 失败模式

| 结束原因 | 基座 | step020 |
|---|---|---|
| 轮数耗尽 | 112 (99%) | 85 (75%) |
| 主动提交 | 1 | 19 |
| 重复动作 ×3 | 0 | 8 |
| 上下文超限 | 0 | 1 |

**几乎所有 rollout 都跑满 40 轮** —— 通过的 27 条里也有 96% 跑满。所以通过率衡量的不是
"能不能修好"，而是"第 40 轮那一刻代码恰好对不对"。

两次都失败的 73 道题上，至少编辑过一次代码的比例：

| | 编辑过 | 比例 |
|---|---|---|
| 基座 | 47/73 | 64% |
| step020 | 26/73 | 36% |

训练后模型更少动手。这不是生成退化导致的，而是训练本身改变了策略。

### chat template：enable 与不 enable 都有问题

| 传参 | prompt 尾部 | 后果 |
|---|---|---|
| `False` | `<think>\n\n</think>\n\n` | 训练后 90% 的 rollout 复读这个空块 |
| 不传 / `True` | `<think>\n` | 打开思考模式，但历史渲染会把思考内容整个剥掉 |

另外，原生模板的生成提示符与历史渲染不一致，相邻两轮的 prompt 不构成 token 前缀链。

**推荐自定义 chat template** —— 让"生成提示符 = 内容为空的历史渲染"，三处问题一并消除。
参照实现（agent-lightning）就是这么做的；本项目提供适配版
[`tools/tpl_chat.jinja`](tools/tpl_chat.jinja)，含 9 个回归测试。

⚠️ 换模板需要重新训练 —— 已有权重是用旧配置训练的，直接换会训推不匹配。

完整推导见 [RESULTS.md](docs/RESULTS.md)。

---

## 改进建议

chat template 的改进见上一节。以下按**是否增加开销**分两类，每条都有实测数据作为依据。

**不增加算力开销：**

| # | 改进 | 依据 |
|---|---|---|
| **1** | 评测集扩到约 150 道（当前 113）| SE 从 4.3 pp 降到约 3.7 pp |
| **2** | 增大策略步长（更大 lr 或更多步）| 实测 `clip_frac` 全程 0.0–0.2%：裁剪从未触发，PPO 实际退化为 REINFORCE |
| **3** | 题目池动态化：已稳定做对的题移出 | 零方差组优势恒为 0；训练集仍含 12 道 allfail |
| **4** | 评测取 k>1 次平均 | 评测温度 0.7，单次采样方差大 |

**需要额外算力或数据：**

| # | 改进 | 依据 | 成本 |
|---|---|---|---|
| **5** | **先做 SFT 冷启动，再做 GRPO** | 基座通过率 23.9%，已有约 27 道成功轨迹可作监督数据。SFT 不需要组内比较，小数据下样本效率高于 GRPO | 数据准备 + 短训练 |
| **6** | 训练量提到 ≥10,000 条 rollout（当前 640）| 要检测 12 pp 的变化，训练量约需 15× | ~140 小时单卡（约 6 天）|
| **7** | 数据扩到上千道（当前 187）| 参照实现用 6,000 道；mixed 比例决定 GRPO 的样本效率 | 数据准备 |

在完成 SFT 冷启动并把训练量提上去之前，本项目的结论只能停在工程层面：
**回路可运行，效果不可测**。

> 第 3 条即 DAPO 的 Dynamic Sampling（[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)）的思路。

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

分析 rollout 的失败模式：

```bash
python tools/analyze_failures.py eval_base_detail.jsonl 40
```

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
pytest tests/ -q     # 39 个用例，纯 CPU
```

| 测试 | 覆盖 |
|---|---|
| `tests/test_row_lcp.py` | LCP 对齐：response 不丢失、mask 与 logprob 不错位、超长截断保留题面 |
| `tests/test_config.py` | YAML 与默认值一致、环境变量覆盖、脚本用到的键均已定义 |
| `tests/test_template.py` | 自定义模板的三条性质：生成点无 think 标签、历史保留思考、前缀链连续 |

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
| [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | 实施步骤（含 chat template 的三个问题与推荐做法）|
| [DATA_FORMAT.md](docs/DATA_FORMAT.md) | 任务字段、目录布局、环境变量 |
| [RESULTS.md](docs/RESULTS.md) | 实验数据、算力门槛、生成退化的完整推导 |

---

## 参考来源

- **agent-lightning** — Microsoft，[examples/swe_smith](https://github.com/microsoft/agent-lightning/tree/main/examples/swe_smith)
  （Qwen3.5-9B / 4×B200 / 全参 + FSDP）。本项目的超参优先对齐该实现；
  `tools/tpl_chat.jinja` 改编自其 `swe_smith_chat_template.jinja`（MIT）。
- **SWE-smith** — 任务生成，[arXiv:2504.21798](https://arxiv.org/abs/2504.21798)
- **verl** — PPO/GRPO 训练框架，[GitHub](https://github.com/volcengine/verl)

---

## License

MIT
