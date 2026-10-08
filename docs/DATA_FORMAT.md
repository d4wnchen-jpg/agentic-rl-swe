# 数据格式与目录布局

---

## 1. 目录布局

所有路径都相对于 `SWESMITH_RL_HOME`（默认 = 仓库根目录）：

```
$SWESMITH_RL_HOME/
├── models/Qwen3.5-9B/          # 基座模型（HuggingFace 格式）
├── swe_smith/                  # SWE-smith 原始数据（parquet）
│   └── data/*.parquet
├── train.jsonl                 # 训练任务
├── eval.jsonl                  # 评测任务（held-out）
├── probe_par_result.jsonl      # 难度探测结果（可选，用于算基线）
├── grpo_adapters/              # 训练产物：每步一个 LoRA 适配器
│   ├── step000/                #   零 LoRA（= 基座），用于测基线
│   └── step001/ ... step020/
├── log_grpo.jsonl              # 每步指标（机器可读）
└── notes/GRPO_RESULT.md        # 训练曲线（人类可读）
```

---

## 2. 任务格式（`train.jsonl` / `eval.jsonl`）

**每行一个 JSON 对象**，字段如下：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `instance_id` | str | ✅ | 唯一标识，格式 `<repo>__<bug_type>__<hash>` |
| `problem_statement` | str | ✅ | 给模型的题目描述（PR description） |
| `repo` | str | ✅ | 仓库名，决定用哪个 conda env 判分 |
| `FAIL_TO_PASS` | list[str] | ✅ | 缺陷修复后应该通过的测试节点 |
| `PASS_TO_PASS` | list[str] | ✅ | 原本就通过、修复后不能破坏的测试节点 |
| `image_name` | str | 🟡 | SWE-smith 原始字段，本地判分用不到 |

**示例**（字段结构，内容为示意）：
```json
{"instance_id": "mistune__func_basic__07jpr8n3",
 "problem_statement": "Fix the tokenizer so that nested emphasis ...",
 "repo": "mistune",
 "FAIL_TO_PASS": ["tests/test_core.py::test_nested_emphasis"],
 "PASS_TO_PASS": ["tests/test_core.py::test_basic", "tests/test_core.py::test_escape"],
 "image_name": "swesmith.x86_64.mistune.abc123"}
```

**测试节点格式**：pytest 的 nodeid，`<文件路径>::<测试函数>`。

---

## 3. 难度探测结果（`probe_par_result.jsonl`）

由 `tools/probe_par.py` 产生，每题跑 N 次（默认 4）。**可选** —— 只有需要自动算基线时才用到。

```json
{"instance_id": "mistune__func_basic__07jpr8n3",
 "resolved": false,
 "n_turns": 40,
 "n_fail": 1,
 "n_missing": 0,
 "messages": [{"role": "system", "content": "..."}, ...]}
```

`tools/split_train_eval.py` 依据同一题多次 `resolved` 的结果分层：

| 分类 | 判据 | 处理 |
|---|---|---|
| **mixed** | 0 < 通过次数 < N | ✅ 进训练集（有学习信号）|
| **allpass** | 通过次数 = N | ❌ 丢弃（基座已全对，无信号）|
| **allfail** | 通过次数 = 0 | 🟡 抽一批进训练集（防止偏简单）|

---

## 4. 训练产物

### `grpo_adapters/stepNNN/`
标准 PEFT 适配器目录（`adapter_config.json` + `adapter_model.safetensors`，约 59MB）。

- `step000` = **零 LoRA**，等价于基座 → 用它测**同配置基线**
- 每次评测都用新的目录 + 新的 `lora_int_id`，避免 vLLM 命中旧缓存

### `log_grpo.jsonl`
每步一行：
```json
{"step": 3, "epoch": 0.17, "train_rate": 0.469, "val_rate": null,
 "n_rollout": 32, "n_rows": 32, "n_zero_var": 2,
 "loss": 0.0001, "clip_frac": 0.0, "sec": 1931}
```

### `notes/GRPO_RESULT.md`
自动生成的训练曲线表（step / epoch / train solve / val solve / loss / clip% / 耗时）。

---

## 5. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SWESMITH_RL_HOME` | 仓库根目录 | 所有路径的基准 |
| `GRPO_CONFIG` | `configs/default.yaml` | 配置文件路径 |
| `GRPO_<超参名大写>` | — | 覆盖单个超参，如 `GRPO_LR=2e-4`、`GRPO_STEPS=40` |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | sm_120 必需 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 防显存碎片 |
| `SKIP_BASELINE` | `1` | 设为 `0` 则训练前先跑一次全量基线 |

---

## 6. 数据从哪来

本项目用的是 **SWE-smith** 的 `func_basic` 类任务：

1. 从 SWE-smith 发布的数据里筛选目标仓库的 parquet
   → `tools/export_more.py`
2. 验证任务可执行（在我们选定的 conda 环境里能跑起来）
   → `tools/verify_tasks.py`
3. 难度探测（每题跑 4 次）
   → `tools/probe_par.py`
4. 按探测结果分层切分
   → `tools/split_train_eval.py`

**注意**：`verify_tasks.py` 里的 `ENVS` 映射表写死了本项目用的 4 个仓库
（mistune / sqlparse / isodate / boltons）。换仓库要改这张表。
