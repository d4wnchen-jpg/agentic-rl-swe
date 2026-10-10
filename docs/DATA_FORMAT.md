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

四步，每步的输入是上一步的输出。**文件名是接得上的** —— 每一步直接用默认值跑即可。

```
swe_smith/data/*.parquet                    SWE-smith 原始数据（parquet）
        │
        │  python tools/export_more.py
        ▼
tasks_more.jsonl                            导出的任务（含 problem_statement）
        │
        │  python tools/verify_tasks.py
        ▼
tasks_ok.jsonl / tasks_bad.jsonl            可执行性验证（clean → 打 patch → F2P 挂/P2P 过 → 反向）
        │
        │  python tools/probe_par.py         每题跑 N 次，约 748 条 rollout
        ▼
probe_par_result.jsonl                      难度探测结果（每题的通过次数）
        │
        │  python tools/split_train_eval.py
        ▼
train.jsonl / eval.jsonl / val_mixed.jsonl  按难度分层切分
```

**每步的输入路径都可以覆盖**（换文件名或换数据源时用）：

| 脚本 | 输入 | 覆盖方式 |
|---|---|---|
| `export_more.py` | `swe_smith/data/*.parquet` | 改脚本里的仓库/数量字典 |
| `verify_tasks.py` | `tasks_more.jsonl` | `python tools/verify_tasks.py <路径>` |
| `probe_par.py` | `tasks_ok.jsonl` | 环境变量 `TASK_FILE=<路径>` |
| `split_train_eval.py` | `probe_par_result.jsonl` + `tasks_ok.jsonl` | `--probe <路径> --tasks <路径>` |

**辅助工具：**

- `tools/make_layered_sample.py` —— 按仓库分层采样（每仓库抽 N 道），
  输入 `tasks_ok.jsonl`，输出 `tasks_layered.jsonl`。想要各仓库均衡时用。

**⚠️ `verify_tasks.py` 里的 `ENVS` 映射表写死了本项目用的 4 个仓库**
（mistune / sqlparse / isodate / boltons），且需要对应的 conda 环境。
换仓库要改这张表。

**⚠️ 评测集构成会影响能不能测出效果** —— 本项目 113 道里 74% 是基座做不动的题，
导致训练真正的提升被稀释。切分时留意 mixed / allpass / allfail 的比例，
报告时分层看。详见 [LESSONS.md 坑 3](LESSONS.md#3-评测集构成掩盖效果)。
