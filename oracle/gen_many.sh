#!/bin/bash
# 并行生成轨迹：bash oracle/gen_many.sh <prefix> <instance> <start> <end> [workers]
set -u
cd /workspaces/agentic-rl-swe
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
    -m deepseek/deepseek-chat -c swebench.yaml \
    -c environment.pull_timeout=300 \
    -c model.model_kwargs.temperature=1.2 \
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
