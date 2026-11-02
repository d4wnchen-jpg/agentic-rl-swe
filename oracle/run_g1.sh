#!/bin/bash
# 批量生成 N 条轨迹 → 重放 → 反事实归因 → 汇总 G1
# 用法: bash oracle/run_g1.sh [instance_id] [image] [N] [prefix]
set -u
cd /workspaces/agentic-rl-swe
IID=${1:-pallets__flask-5063}
IMAGE=${2:-swebench/sweb.eval.x86_64.pallets_1776_flask-5063:latest}
N=${3:-5}
PREFIX=${4:-g1}

for i in $(seq 1 "$N"); do
  outdir="mswa_${PREFIX}_$i"
  echo "=== [$i/$N] 生成轨迹 → $outdir ==="
  mini-extra swebench --subset lite --split test --filter "$IID" --workers 1 \
    -m deepseek/deepseek-chat -c swebench.yaml -c environment.pull_timeout=300 \
    -c model.model_kwargs.temperature=1.2 -o "$outdir" > "${PREFIX}_gen_$i.log" 2>&1 \
    || { echo "生成失败，见 ${PREFIX}_gen_$i.log"; continue; }

  traj="$outdir/$IID/$IID.traj.json"
  [ -f "$traj" ] || { echo "没有轨迹文件，跳过"; continue; }

  echo "=== [$i/$N] 重放 + 反事实 ==="
  python oracle/replay_traj.py --traj "$traj" --task SWE-bench_Lite.jsonl \
    --instance-id "$IID" --image "$IMAGE" \
    --work "replay_${PREFIX}_$i" --out "out_${PREFIX}_$i" \
    --rerun 1 --counterfactual-pivots --counterfactual-recoveries \
    || echo "重放失败（out_${PREFIX}_$i）"
  rm -rf "replay_${PREFIX}_$i"
done

echo "=== 汇总 ==="
python oracle/summarize_g1.py 'out_g1_*'
