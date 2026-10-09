# Implementation

单卡 96GB（RTX PRO 6000 Blackwell / sm_120）实测通过的完整实现路径。

---

## 1. 环境

```bash
conda create -n vllm python=3.12 -y
conda activate vllm

pip install vllm==0.30.0          # 需要 torch >= 2.9 才支持 sm_120
pip install peft accelerate safetensors

# 训练侧 kernel —— vLLM 只带推理 kernel，这个必须单独装
pip install flash-linear-attention
```

### 判分环境（每个仓库一个 conda env）

```bash
for r in mistune sqlparse isodate boltons; do
  conda create -n $r python=3.11 -y
  conda run -n $r pip install -e /path/to/$r
done
conda run -n boltons pip install "pytest<7"    # boltons 需要旧版 pytest
```

调用判分时用 `conda run -n <env> pytest ...`（**不要**加 `python -m`，
部分 conda 版本下会返回空输出）。

### 环境变量

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0            # sm_120 必需
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1                     # 崩溃时输出 Python 栈
export SWESMITH_RL_HOME=/path/to/workspace      # 本项目所有路径的根
```

`VLLM_USE_FLASHINFER_SAMPLER` 必须在 `import vllm` **之前**设置（放在模块顶部）。

---

## 2. 数据准备

```
SWE-smith 全量
    │
    ├─ 导出 func_basic 类任务（4 个仓库：mistune / sqlparse / isodate / boltons）
    │   → tools/export_more.py
    │
    ├─ 可执行性验证（187/187 通过）
    │   → tools/verify_tasks.py
    │
    ├─ 难度探测：每题跑 4 次
    │   → tools/probe_par.py      （748 条 rollout）
    │
    └─ 按探测结果分层切分
        → tools/split_train_eval.py

分类结果：
    mixed   77 道（41%）   ← 有学习信号，GRPO 的基础
    allpass 14 道          ← 去掉（基座已全对）
    allfail 96 道          ← 抽一批进训练集，防止偏简单

切分：
    train 72  = 60 mixed + 12 allfail
    eval  115 = held-out（含 mixed / allpass / allfail）
```

**评分标准**：`FAIL_TO_PASS` 全部通过 **且** `PASS_TO_PASS` 全部通过 → reward = 1.0。

---

## 3. 模型与显存

### 3.1 显存预算

| 组件 | 占用 |
|---|---|
| vLLM 权重（bf16）| 18 GB |
| vLLM KV + Mamba cache | ~20 GB |
| vLLM CUDA graph | ~3 GB |
| 训练侧权重（bf16，冻结）| 18 GB |
| LoRA + Adam 状态 | 0.5 GB |
| 激活（梯度检查点，16k token）| ~25 GB |
| **合计** | **~85 GB / 96 GB** |

### 3.2 vLLM 初始化

```python
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"     # 必须在 import vllm 之前

from vllm import LLM

llm = LLM(
    model=MODEL_PATH, trust_remote_code=True, dtype="bfloat16",
    max_model_len=81920,
    gpu_memory_utilization=0.45,       # 单卡 96GB 的取值
    enforce_eager=False,               # 启用 CUDA graph
    enable_lora=True, max_lora_rank=16, max_loras=2,
    max_num_seqs=128,                  # Qwen3.5 的 Mamba cache 约束，见下
    max_num_batched_tokens=8192,
    gdn_prefill_backend="triton",
    disable_log_stats=True,
)
```

**`max_num_seqs` 需要显式设置。** Qwen3.5 是混合架构（32 层中 24 层为 GDN 线性注意力），
使用 Mamba cache，容量单位为**序列数**而非 token 数：

| 缓存 | 服务对象 | 容量单位 |
|---|---|---|
| KV cache | 8 个全注意力层 | token 数 |
| Mamba cache | 24 个 GDN 层 | 序列数 |

`gpu_memory_utilization` 降低后 Mamba 块数减少，vLLM 默认的 `max_num_seqs=1024`
会超过可用块数。本项目取 128（并发上限 40）。

### 3.3 训练侧

```python
from peft import LoraConfig, get_peft_model

model = Cls.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",       # 必须用 all-linear
)
model = get_peft_model(model, lora_cfg)
model.config.use_cache = False
model.gradient_checkpointing_enable()
```

**`target_modules` 用 `"all-linear"`**：Qwen3.5 的层类型不统一（只有 8/32 层是 `self_attn`，
其余是 `linear_attn`），手写层名列表会漏掉大部分可训练参数。
让 peft 自己发现，实测得到 **716 张量 / 51.3M 参数 / 0.54%**。

---

## 4. Rollout

```
每轮：模型输出动作 → 沙箱执行 → 观察拼回 → 下一轮（最多 40 轮）
```

### 4.1 沙箱准备

```python
prep_sandbox(task, workdir)
#  ① git checkout <缺陷分支>
#  ② 应用缺陷补丁
#  ③ 把 .git 移出工作区（防止从 git 历史里找到参考修复）
```

### 4.2 关键：记录 token 级信息

**每一轮都要记录**（这是后续训练的前提）：

```python
(pid, rid, text, lps, finish_reason) = llm.batch_chat_ids([messages], ...)
#  pid           = vLLM 实际喂给模型的 prompt token ids
#  rid           = 模型实际生成的 token ids
#  lps           = 每个生成 token 的 logprob
#  finish_reason = 用于区分"撞输出上限"和一般格式错误
```

**必须关闭历史 elide** —— 一旦改写历史，多轮之间就不再构成 token 前缀链，
训练行只能退回"每轮一行"（成本高约 18 倍）。

### 4.3 chat template：不要传 `enable_thinking=False`

Qwen3.5 的模板在 `enable_thinking=False` 时会在 prompt 尾部插入一个**完整的空思考块**：

```
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

| 传参 | prompt 尾部 |
|---|---|
| **不传（推荐）** | `<\|im_start\|>assistant\n<think>\n` |
| `False` | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| `True` | 同"不传" |

**传 `False` 时，模型有概率复读这个完整模式。** 本项目实测：训练后 108/113（96%）的
rollout 出现复读（基座 0/113），其中 82% 在第一轮。该片段在训练数据里 `mask=0`，
因此复读不是学到的行为，机制未确定。详见 [RESULTS.md 第 8 节](RESULTS.md#8-生成退化一个观察到的现象)。

```python
# 推荐（需要与训练配置一致）
chat_template_kwargs = None
# 或显式关闭思考（接受复读风险）
chat_template_kwargs = {"enable_thinking": False}
```

#### 推荐做法：自定义 chat template

上面三个问题的根源是同一个 —— Qwen3.5 原生模板对思考块做了特殊处理。

**参照实现的做法是自己写 chat template**（agent-lightning 的
[`examples/swe_smith/swe_smith_chat_template.jinja`](https://github.com/microsoft/agent-lightning/blob/main/examples/swe_smith/swe_smith_chat_template.jinja)，
MIT 许可）。

本项目提供了一个适配版：[`tools/tpl_chat.jinja`](../tools/tpl_chat.jinja)
（去掉了参照实现里的 tools / tool_calls 分支，因为本项目从文本里解析 XML 动作）。

核心写法只有一条：**生成提示符 = 内容为空的历史渲染**，两处共用同一套结构。

| 问题 | 原生模板 | 自定义模板 |
|---|---|---|
| 生成点插入完整空块 | 有（触发复读）| 无 |
| 生成提示符 = 历史渲染？ | 否（前缀链断裂）| **是** |
| 历史渲染保留思考内容？ | 否（剥离）| **是** |

**前提与代价：**

- ⚠️ **换模板等于换格式，必须重新训练。** 已有权重是用原生模板训练的，直接换会出现
  训推不匹配（本项目实测：同题通过率 25.4% → 15.3%）。
- 切换前建议先用基座模型验证一次（确认模型在新格式下仍能正常输出动作）。
- 前缀链连续之后，`build_row_lcp` 的 LCP 对齐不再是必需的（保留也无害）。

**⚠️ 训练与推理必须用同一个配置。** 用 A 配置训练的权重，换 B 配置推理会得到明显更差的结果
（本项目实测：25.4% → 15.3%）。如果要切换配置，需要用新配置重新训练。

**⚠️ 另一个要注意的行为：历史渲染会剥离思考内容。**

Qwen3.5 的原生模板在渲染历史 assistant 消息时，会把内容里的 `<think>…</think>` 整个删掉：

```python
content = "<think>\n思考内容\n</think>\n\n<function=bash>ls</function>"
# 渲染结果（三种 enable_thinking 传参都一样）：
#   <|im_start|>assistant
#   <function=bash>ls</function><|im_end|>
```

**→ 模型在下一轮看不到自己上一轮的思考，必须重新推理。**

参照实现（agent-lightning）使用自定义模板，**原样保留** assistant 内容，没有这个问题。

如果需要保留思考历史，要自己写 chat template。

---

### 4.4 动作格式与容错

- 动作格式：`<function=name><parameter=key>value</parameter></function>`
- 解析失败 → 追加格式提示、不消耗工具轮次
- 撞输出上限（`finish_reason == "length"`）→ 给**针对性提示**（要求简洁），与一般格式错误区分
- 连续 3 次格式错误 → 结束该 episode

---

## 5. 防作弊

模型可能绕过任务本身去拿参考修复。四类动作被拦截：

| 类别 | 拦截内容 |
|---|---|
| **git** | `git` 调用、`--git-dir` / `--work-tree`、访问 `.git` |
| **网络** | `curl` / `wget` / `scp` / `nc` / `telnet` 等 |
| **安装** | `pip install` / `conda install` / `uv install` / `python -m pip install` |
| **改测试** | 写入 `conftest.py` / `pytest.ini` / `tox.ini` / `sitecustomize.py` / `setup.cfg` / `pyproject.toml` / `*.pth` |

被拦截的动作返回拒绝原因，**不消耗工具轮次**。

**判分前恢复被删的测试文件**：从缺陷提交的父提交里 `git checkout` 回测试路径
（只覆盖测试文件，保留模型的源码修改）。

---

## 6. 判分

```bash
COLUMNS=200 pytest --color=no -rA -p no:cacheprovider -n4 <test_nodes>
```

| 要点 | 原因 |
|---|---|
| `COLUMNS=200` | 防止 pytest 把 `PASSED test_xxx` 折成两行导致解析失败 |
| `--color=no` + 剥 ANSI | 颜色码会包住 `PASSED` 关键字 |
| `-n4` 上限 | 项目自带的 `-n auto` 会按**物理核数**起 worker，在容器里容易 OOM |
| `f2p_only` | 只判 `FAIL_TO_PASS` 所在文件的 `PASS_TO_PASS`，避免跨文件超时 |

**reward = 1.0** 当且仅当 `F2P` 全通过 **且** `P2P` 全部 PASSED/XFAIL。

---

## 7. GRPO 训练

### 7.1 优势（rollout 级，组内归一化）

$$A_i = \frac{r_i - \text{mean}(r_1..r_G)}{\text{std}(r_1..r_G) + 10^{-4}}$$

- $r_i$ = 第 i 条 rollout 的 reward（0 或 1）
- $G$ = 组大小（同一道题的 rollout 数，本实现为 8）
- 优势在**完整的组**上算好之后，才进入训练；之后怎么分批都不影响它的正确性

### 7.2 拼训练行（LCP 前缀链对齐）

**目标**：把 40 轮拼成**一条**序列（而不是每轮一条），省约 18 倍训练成本。

**做法**：chat template 在生成提示符与历史渲染两种场景下的输出可能不同，
采用**最长公共前缀**对齐：

```
row  = P₀ + R₀ ;  mask = [0]*|P₀| + [1]*|R₀|
对第 k 轮 (k≥1):
    L    = LCP(row, P_k)              # 最长公共前缀
    row  = row[:L] + P_k[L:]          # 用 P_k 作权威，丢弃被模板改写的 token
    mask = mask[:L] + [0]*(|P_k|-L)
    在 P_k 中找历史 response 的片段，标回 mask=1 并还原其 logprob
    row += R_k ;  mask += [1]*|R_k|
```

**符号**：$P_k$ = 第 k 轮 vLLM 实际喂的 prompt token ids；$R_k$ = 第 k 轮生成的 token ids。

**实测**：3 轮模拟下压缩 2.1 倍；40 轮时约 18 倍。

**超长处理**：保留 `system + 题面` + 尾部若干 token（尾部包含修复动作）。

### 7.3 logprob 计算

PPO 需要每个 token 的 logprob，但 `[S, V]` 张量太大：

| 量 | 值 |
|---|---|
| S（行长）| 25,000 |
| V（词表）| **248,320** |
| `[S, V]` fp32 | **24.8 GB**（不含反向）|

**两步处理**：

1. **只对 loss 位置算**（观察和系统提示不算 loss）→ 6539 个位置而非 25,000
2. **分块 + 反向重算**：每块 512 个位置，用 `torch.utils.checkpoint` 包住

```python
for i in range(0, n, chunk):
    idx = head_pos[i:i+chunk]
    o = checkpoint(_seg, x[:, idx, :], targets[i:i+chunk], use_reentrant=False)
```

**分块和反向重算必须配套**：只分块不重算，每块的中间结果仍要留到反向
（`0.5GB × 13 块 × 2 = 13GB`）。

**实现要点**：

```python
head_pos = pred_pos - 1       # 因果 LM：预测位置 i 的 logits 来自位置 i-1

# 拦截 lm_head 要 patch 模块的 forward，不是 setattr 到外层包装
head_mod = model.get_submodule(lm_head_name)
orig_fwd = head_mod.forward
head_mod.forward = lambda x: orig_fwd(x[:, head_pos, :])

# 加形状断言，拦截失败时立刻报错
assert o.shape[0] == head_pos.numel()
```

**验证**：与全量 logits 逐位对比，最大差异应为 `0.00000000`。

### 7.4 `old_logprobs` 由训练模型重算

```python
with torch.no_grad():
    lp_old, pos = masked_logprobs(model, row, mask)
```

**不要用推理引擎返回的 logprob**：推理和训练的 kernel 不同（FlashAttention vs 参考实现），
同一份权重下少数 token 的 logprob 会有可观差异。

**自带的正确性校验**：第 1 步第 1 批的 `ratio` 必须精确等于 `1.000000`（策略还没动）。

### 7.5 PPO 更新

| 符号 | 含义 |
|---|---|
| $\pi_{\theta_{old}}$ | 采样时的策略 |
| $\pi_\theta$ | 算 loss 时的策略 |
| $\rho_t = \pi_\theta / \pi_{\theta_{old}}$ | 重要性比 |
| $\hat A$ | rollout 级优势（广播到该 rollout 的所有 token）|
| $\epsilon_{low}, \epsilon_{high}$ | 0.2 / 0.28 |

$$\mathcal{L} = -\min\Big(\rho_t \hat A,\ \text{clip}(\rho_t,\ 1-\epsilon_{low},\ 1+\epsilon_{high})\,\hat A\Big)$$

**每批数据做 2 次 mini-batch 更新**（`mini_batch_size=16`，`max_ppo_update_times=2`）：

```
打乱全部训练行
→ 第 1 批 16 行 → 累加梯度 → opt.step()
→ 第 2 批 16 行 → 累加梯度 → opt.step()
```

**每批做 2 次更新。** 单次更新时，算 loss 的策略与采样策略完全相同，$\rho_t \equiv 1$，
`clip` 不生效；第 2 次更新在策略已变动之后进行，$\rho_t \neq 1$，裁剪开始约束。

**切批前先打乱。** 训练行按题分组，不打乱会导致每批只含 2 道题的梯度。

**损失归一化（per-rollout mean）**：

```
每条 rollout 的 loss 先除以它的 token 数
再除以该 mini-batch 的行数
```

每条 rollout 的权重与其训练行长度无关。

### 7.6 学习率

| 训练方式 | lr |
|---|---|
| 全参微调 | 1e-6 |
| **LoRA（本实现）** | **1e-4** |

peft 默认将 `lora_B` 初始化为 0，而 $\partial L/\partial A \propto B$。
沿用全参的 lr 时，20 步后 $\Delta W/W$ 仅 0.006%，低于 bf16 数值精度。

$$\Delta W \approx \sqrt{r}\cdot|B|\cdot|A|\cdot\frac{\alpha}{r}, \qquad |B| \approx N \cdot \text{lr}$$

**标定方法**（跑 1 步即可验证 lr 是否有效）：

```python
a = load_file("adapters/step000/adapter_model.safetensors")
b = load_file("adapters/step001/adapter_model.safetensors")
mb = max((a[k].float()-b[k].float()).abs().max().item() for k in a if "lora_B" in k)
# Adam 首步的预期值 ≈ lr
```

### 7.7 权重同步

```python
model.save_pretrained(f"adapters/step{step:03d}")
cur_lora = llm.LoRARequest(f"s{step}", step + 2, f"adapters/step{step:03d}")
```

每次用**新的目录和新的 `lora_int_id`**，避免 vLLM 命中旧缓存。

---

## 8. 评测

```python
# 评测温度 0.7（训练用 1.0）
quick_eval(llm, cur_lora, eval_file="eval.jsonl", n_limit=115, temperature=0.7)
```

**评测集用全部 held-out 样本**，并且**基线必须用同一套配置、同一个集合测**。

**基线可以事后补**：`adapters/step000` 是零 LoRA（等价基座），
用 `eval_adapter.py` 单独跑一次即可，顺序无关。

```bash
python tools/eval_adapter.py --adapter adapters/step000 --n 115 --k 1 --tag base
```

**统计功效**：$n$ 个样本的标准误

$$\text{SE} = \sqrt{\frac{p(1-p)}{n}}$$

113 个样本、$p \approx 0.3$ 时 SE ≈ 4.3pp。**报告结论前先确认样本量能检测出多大的变化。**

---

## 9. 启动

```bash
setsid nohup bash -c '
  cd $SWESMITH_RL_HOME
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONFAULTHANDLER=1
  python -u tools/train_grpo.py
  rc=$?
  echo "=== exit rc=$rc $(date +%H:%M:%S) ==="
  [ $rc -gt 128 ] && echo "=== signal $((rc-128)) ==="
' > log_grpo.txt 2>&1 &
```

`setsid` 让进程完全脱离终端；捕获退出码便于定位异常终止。

---

## 10. 上线前的自检清单

```
□ 1. vLLM 能加载 LoRA，且非零 LoRA 会改变 logprob
□ 2. 训练侧模型和 vLLM 能共存（不 OOM）
□ 3. 一条 rollout 能跑完并记录 prompt / response token ids
□ 4. 多轮 token 能拼成前缀链（LCP 校验通过）
□ 5. logprob 与全量 logits 逐位一致（最大差异 < 1e-3）
□ 6. 跑 1 步后适配器权重变化 ≈ lr
□ 7. 跑 1 步后第 1 批 ratio == 1.000000
□ 8. 用同配置测基座 → 得到可比基线
```
