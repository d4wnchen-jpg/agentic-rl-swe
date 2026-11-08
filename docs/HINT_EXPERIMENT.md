# hint 实验：失败到底缺什么？

## 目的

v3 受控回滚（DeepSeek 轨迹）显示：把有害轮回滚到 baseline 状态后，`resolve` 仍是 0/9。
也就是说「擦掉错误」能回到起点，但 agent 自己走不到终点。问题出在三种可能：

1. **缺目标信息**：它不知道要修哪两个隐藏测试、预期行为是什么；
2. **缺预算**：它需要更多步才能探索到正确修法；
3. **缺能力**：Qwen3.5-9B 本身修不动这道题。

hint 实验就是分离开这三者，同时顺带产出 recovery 数据（教师/特权信息下的成功续跑）。

## 设计

所有条件都**从最终状态继续**（不回滚），只改变续跑时给 agent 的信息和步数预算。
续跑模型必须与生成轨迹的模型一致（Qwen3.5-9B），否则无法回答「这个学生缺什么」。

| 条件 | 额外信息 | 步数 | 含义 |
|---|---|---|---|
| `none` | 无 | 8 | 基线：只有 issue + 当前 diff |
| `legal` | 仓库自带测试的当前输出 | 8 | agent 自己跑 pytest 就能看到（可部署） |
| `oracle` | 隐藏 F2P 测试名 + test_patch 内容 | 8 | 上界：目标信号直接告诉它 |
| `legal_long` | 同 legal | 24 | 排除预算因素（legal 侧） |
| `oracle_long` | 同 oracle | 24 | 排除预算因素（oracle 侧） |

判读：

- `oracle >> none` → 失败主要因为**缺目标信息**（支持 privileged teacher / recovery 数据构造）
- `oracle_long ≈ oracle` → 不是预算问题
- `legal ≈ oracle` → demo 可以直接用合法提示，无需特权信息
- `none ≈ oracle ≈ oracle_long` → 能力瓶颈：9B 学生自己修不动，需要训练

## 命令

```bash
cd /workspaces/agentic-rl-swe

# 1) 选 case（所有失败轨迹；results/pilot 需已包含 1/3/4/5 的 json+traj）
python3 - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob('results/pilot/*.json')):
    b = os.path.basename(f).split('.')[0]
    if not b.isdigit() or f.endswith(('.traj.json', '_preds.json', '_teacher.json')):
        continue
    r = json.load(open(f))
    evs = r.get('evals') or []
    final = evs[-1].get('n_fail') if evs else None
    if final:
        rows.append({'run': b, 'final_n_fail': final})
json.dump(rows, open('demo/cases_pilot.json', 'w'), ensure_ascii=False, indent=1)
print(rows)
PY

# 2) Qwen 续跑环境（与生成轨迹一致）
export OPENAI_API_BASE=https://api.siliconflow.cn/v1
export OPENAI_API_KEY=<siliconflow key>
export THINK=false
export REPAIR_MODEL=openai/Qwen/Qwen3.5-9B
export REPAIR_TEMP=0.6

# 3) 跑 5 臂（串行，建议 tmux）
bash oracle/run_hint_batch.sh demo/cases_pilot.json \
  swebench/sweb.eval.x86_64.pallets_1776_flask-5063:latest 8

# 4) 汇总
python oracle/stats_hint.py results/hint_pilot
```

单臂约 3–8 分钟，4 失败 case × 5 臂约 1.5–3 小时（含 TPM 限流重试）。

## 产出

- 成功续跑（final n_fail 下降/归零）→ recovery 数据，可用于 SFT / 正向样本；
- 失败续跑 → 偏好对（坏状态下的教师/成功动作 vs 学生动作）；
- 汇总表 → 回答「缺信息 / 缺预算 / 缺能力」，决定 demo 主 claim 和论文训练方案。
