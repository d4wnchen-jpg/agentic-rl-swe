#!/bin/bash
# 并行生成轨迹：bash oracle/gen_many.sh <prefix> <instance> <start> <end> [workers]
# 模型：MODEL 环境变量（默认 deepseek/deepseek-chat）；
#   OpenRouter: MODEL=openrouter/qwen/qwen3.5-9b + OPENROUTER_API_KEY
#   SiliconFlow: MODEL=siliconflow/Qwen/Qwen3.5-9B + SILICONFLOW_API_KEY
#   DashScope:  MODEL=dashscope/qwen3.5-9b-instruct + DASHSCOPE_API_KEY
set -u
ROOT=${ROOT:-/workspaces/agentic-rl-swe}
cd "$ROOT"
MODEL=${MODEL:-deepseek/deepseek-chat}
THINK_ARGS=()
if [ "${THINK:-true}" = "false" ]; then
  THINK_ARGS=(-c model.model_kwargs.extra_body.enable_thinking=false)
fi
echo "model=$MODEL root=$ROOT"
PREFIX=${1:?prefix}
IID=${2:?instance}
START=${3:?start}
END=${4:?end}
WORKERS=${5:-4}

gen_one() {
  local i=$1
  local d="mswa_${PREFIX}_${i}"
  local traj="$d/$IID/$IID.traj.json"
  if [ -f "$traj" ]; then echo "[$i] 已存在"; return 0; fi
  mkdir -p "$d"
  mini-extra swebench --subset lite --split test --filter "$IID" --workers 1 \
    -m "$MODEL" -c swebench.yaml -c model.cost_tracking=ignore_errors \
    -c environment.pull_timeout=300 \
    -c model.model_kwargs.temperature="${TEMP:-1.2}" "${THINK_ARGS[@]}" \
    -o "$d" > "${PREFIX}_gen_$i.log" 2>&1
  if [ -f "$traj" ]; then echo "[$i] 生成完成"; else echo "[$i] 生成失败"; fi
}

for i in $(seq "$START" "$END"); do
  gen_one "$i" &
  while [ "$(jobs -r | wc -l)" -ge "$WORKERS" ]; do wait -n; done
done
wait
echo "=== 全部完成 ==="
for i in $(seq "$START" "$END"); do
  [ -f "mswa_${PREFIX}_${i}/$IID/$IID.traj.json" ] && echo "OK $i" || echo "MISS $i"
done
