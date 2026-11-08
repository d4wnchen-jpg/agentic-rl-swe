# Qwen3.5-9B pilot（demo 换学生模型）

目标：用 **Qwen3.5-9B** 在 `pallets__flask-5063` 上生成 5 条轨迹，确认三件事后，再决定是否把 demo 主数据（30 条 + 9 条清理实验）整体换成 Qwen。

## 0. 选 provider、配 key（只做一次）

三种接法任选，key 只配在 Codespace 环境变量里：

| provider | export | MODEL | 备注 |
|---|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY=...` | `openrouter/qwen/qwen3.5-9b` | $0.17/M 输入、$0.25/M 输出，262K ctx |
| SiliconFlow | `SILICONFLOW_API_KEY=...` | `siliconflow/Qwen/Qwen3.5-9B` | 国内付款；若 litellm 不认这个前缀，用下面的 openai 兼容写法 |
| SiliconFlow（兼容写法） | `OPENAI_API_KEY=...`、`OPENAI_API_BASE=https://api.siliconflow.cn/v1` | `openai/Qwen/Qwen3.5-9B` | 通用兜底 |
| DashScope | `DASHSCOPE_API_KEY=...` | `dashscope/qwen3.5-9b-instruct` | 官方；模型 id 以控制台为准 |

先验证 endpoint：

```bash
cd /workspaces/agentic-rl-swe
export MODEL=openrouter/qwen/qwen3.5-9b       # 换成你选的
bash oracle/check_model.sh                    # 应输出 OK + usage
```

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
