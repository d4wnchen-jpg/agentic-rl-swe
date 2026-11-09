# Qwen3.5-9B pilot（demo 换学生模型）

目标：用 **Qwen3.5-9B** 在 `pallets__flask-5063` 上生成 5 条轨迹，确认三件事后，再决定是否把 demo 主数据（30 条 + 9 条清理实验）整体换成 Qwen。

## 0. 选 provider、配 key（只做一次）

三种接法任选，key 只配在 Codespace 环境变量里：

| provider | export | MODEL | 备注 |
|---|---|---|---|
| 硅基流动 SiliconFlow | `OPENAI_API_KEY=...`、`OPENAI_API_BASE=https://api.siliconflow.cn/v1` | `openai/Qwen/Qwen3.5-9B` | 支付宝/微信付款；litellm 用 openai 兼容前缀，不需要 `siliconflow/` |
| 阿里云百炼 DashScope | `DASHSCOPE_API_KEY=...` | `dashscope/qwen3.5-9b` | 官方 Qwen；支付宝付款 + 实名认证；模型 id 以百炼控制台为准 |
| OpenRouter（海外卡） | `OPENROUTER_API_KEY=...` | `openrouter/qwen/qwen3.5-9b` | $0.17/M 输入、$0.25/M 输出；只收 visa/mastercard/crypto |

> 硅基流动已确认有 `Qwen/Qwen3.5-9B`（模型页标签：推理模型、Tools、256K）。它是 thinking 模型，默认会输出推理 token；价格：单次请求上下文 <128K 时输入 ¥0.5/M、输出 ¥4/M，≥128K 时输入 ¥1.5/M、输出 ¥12/M。thinking token 按输出计费，按现有轨迹估算单条约 ¥0.1–0.3。

> 国内付款优先：先查百炼模型列表有没有 Qwen3.5-9B（官方、版本稳定）；没有再用硅基流动。硅基流动的 Qwen 模型名多是 `Qwen/Qwen3.5-9B-Instruct` 这种带 Instruct 后缀，以模型中心为准，`check_model.sh` 通了再往下跑。

先验证 endpoint：

```bash
cd /workspaces/agentic-rl-swe
export MODEL=openrouter/qwen/qwen3.5-9b       # 换成你选的
bash oracle/check_model.sh                    # 应输出 OK + usage
```

## 0.1 采样参数（Qwen 专用）

Qwen3.5-9B 是 thinking 模型，默认每步会输出推理 token（实测简单问题就 195 reasoning tokens，单步延迟 5–10s）。对 28 步的轨迹来说，thinking 会显著拖慢（单条可能 30 分钟+），pilot 建议先关掉：

```bash
export THINK=false     # 通过 extra_body 传 enable_thinking=false（已用 curl 验证硅基流动支持）
export TEMP=0.6        # Qwen 官方推荐温度；默认脚本里是 1.2
```

如果关掉 thinking 后命令格式错误率高，再开 thinking 跑 2 条对比。

## 1. 生成 5 条轨迹

```bash
cd /workspaces/agentic-rl-swe
export MODEL=openrouter/qwen/qwen3.5-9b       # 同上
bash oracle/gen_many.sh pilot pallets__flask-5063 1 5 3
```

- 产物：`mswa_pilot_1..5/pallets__flask-5063/*.traj.json`，日志 `pilot_gen_*.log`
- 预计 5–15 分钟/条（取决于 provider 延迟），5 条约 ¥10–30
- 失败就 `cat pilot_gen_1.log` 看是不是模型名/key/限额问题

## 2. 回放 + 逐轮标签

```bash
# 第 5 个参数 teacher=0（pilot 不做教师标注）
bash oracle/run_pipeline.sh pallets__flask-5063 \
  swebench/sweb.eval.x86_64.pallets_1776_flask-5063:latest 5 pilot 0
```

已有轨迹会自动跳过生成，直接重放（`--rerun 1 --counterfactual-pivots --counterfactual-recoveries`）。结果进 `results/pilot/`，每条一个 `*.json` + `SUMMARY.txt`。

## 3. 看四个数

```bash
cat results/pilot/SUMMARY.txt
```

| 指标 | 合格线 | 说明 |
|---|---|---|
| 失败率 | ≥ 40% | 太低说明 Qwen3.5-9B 太强/温度不够，需要调采样 |
| `baseline n_fail == len(F2P)=2` | 5/5 成立 | 否则该条的标签是噪声，丢弃 |
| 有害 pivot 密度 | 失败轨迹里 ≥ 1/3 有 harmful 编辑 | 决定 demo/训练数据能否从 Qwen 轨迹产出 |
| 重放保真 | 重放代码 == 提交 patch（忽略 `patch.txt`） | 与之前 DeepSeek 数据同一标准 |

另外看 `parse_failures`（命令解析失败次数）和每条耗时/token（`traj.json` 里有 usage）。

## 4. 决策

- **四项都合格** → 扩到 20–30 条，替换 demo 主数据；同时用这些轨迹跑 hint 实验。
- **失败率过低** → 提高 temperature（如 1.2 已是默认上限，可试 1.4）或换更难的实例。
- **有害 pivot 太少** → 只换 hint/清理实验用的候选轨迹，demo 诊断部分保留 DeepSeek 数据作为强模型对照。
- **重放保真不过** → 先查工具格式（`parse_failures`），必要时改用 Qwen 官方 chat template。

## 5. provenance（写进 demo/论文）

`model=Qwen3.5-9B`、provider、模型 id、采样日期、temperature=1.2、scaffold=mini-swe-agent 2.4.6 + `swebench.yaml`、dataset=SWE-bench_Lite。demo 定稿时若换 vLLM 重采，把 provider 改为本地 vLLM 并记录 checkpoint。

## 2026-11-09 与 PivotOPD 的设置对齐

生成侧（gen_many.sh / run_g1.sh / run_pipeline.sh）默认值已改为：

- `TEMP=0.6`：PivotOPD SWE 训练 rollout 温度（其评测档为 temp 1.0 / top-p 0.95 / 3 seeds，等跑最终数字时再切）
- `MAXTURNS=100`：PivotOPD SWE 的 100-turn episode 预算（mini-swe 原本是 250）
- `THINK=false`：PivotOPD vLLM rollout 关闭原生 thinking，推理放在 content
- `-c model.cost_tracking=ignore_errors`：Qwen 不在 litellm 价格表里

保持不同的地方（论文里要写明）：

- scaffold：PivotOPD SWE 用 OpenHands 风格 function-calling；我们用 mini-swe-agent 的 bash tool call（同为工具调用接口，但 prompt/上下文管理不同）。D.4 的 `<think>/<action>` 文本模板只用于 ALFWorld/WebShop/Search QA，不适用于 SWE。
- 方法差异正是我们的 gap：PivotOPD 在 SWE 只审计最终提交动作（m=1）、K=0；我们做逐轮 edit 级验证 + recovery 数据。
