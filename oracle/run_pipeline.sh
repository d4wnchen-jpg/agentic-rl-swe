#!/bin/bash
# 全自动管线：生成 → 重放+归因 → 教师标注 → 汇总 → git 入库
# 用法: bash oracle/run_pipeline.sh [iid] [image] [N] [prefix] [teacher 1/0]
set -u
cd /workspaces/agentic-rl-swe
IID=${1:-pallets__flask-5063}
IMAGE=${2:-swebench/sweb.eval.x86_64.pallets_1776_flask-5063:latest}
N=${3:-10}
PREFIX=${4:-run}
TEACHER=${5:-1}
RERUN=${RERUN:-1}
RES="results/$PREFIX"
mkdir -p "$RES"

push() {
  git add -f "$RES" >/dev/null 2>&1
  git commit -qm "results: $PREFIX ($1)" >/dev/null 2>&1 || return 0
  git push -q >/dev/null 2>&1 || { git pull -q --rebase -X theirs && git push -q; } || true
}

for i in $(seq 1 "$N"); do
  outdir="mswa_${PREFIX}_$i"
  traj="$outdir/$IID/$IID.traj.json"
  if [ ! -f "$traj" ]; then
    echo "=== [$i/$N] 生成 ==="
    mini-extra swebench --subset lite --split test --filter "$IID" --workers 1 \
      -m deepseek/deepseek-chat -c swebench.yaml -c environment.pull_timeout=300 \
      -c model.model_kwargs.temperature=1.2 -o "$outdir" > "${PREFIX}_gen_$i.log" 2>&1
  fi
  [ -f "$traj" ] || { echo "[$i] 没有轨迹，跳过"; continue; }

  if [ -f "out_${PREFIX}_$i/traj_report.json" ]; then
    echo "[$i/$N] 已有报告，跳过重放"
  else
  echo "=== [$i/$N] 重放 + 归因 ==="
  python oracle/replay_traj.py --traj "$traj" --task SWE-bench_Lite.jsonl \
    --instance-id "$IID" --image "$IMAGE" \
    --work "replay_${PREFIX}_$i" --out "out_${PREFIX}_$i" \
    --rerun "$RERUN" --counterfactual-pivots --counterfactual-recoveries \
    > "${PREFIX}_replay_$i.log" 2>&1 || echo "[$i] 重放失败"
  rm -rf "replay_${PREFIX}_$i"
  fi

  cp "out_${PREFIX}_$i/TRAJ.md" "$RES/${i}.md" 2>/dev/null
  cp "out_${PREFIX}_$i/traj_report.json" "$RES/${i}.json" 2>/dev/null
  cp "$traj" "$RES/${i}.traj.json" 2>/dev/null
  cp "$outdir/preds.json" "$RES/${i}_preds.json" 2>/dev/null
  python oracle/summarize_g1.py "out_${PREFIX}_*" > "$RES/SUMMARY.txt" 2>&1
  push "$i/$N"
  echo "[$i/$N] 已入库"
done

if [ "$TEACHER" = "1" ]; then
  echo "=== 教师标注（G2） ==="
  : > "$RES/TEACHER.txt"
  for i in $(seq 1 "$N"); do
    r="out_${PREFIX}_$i/traj_report.json"
    t="mswa_${PREFIX}_$i/$IID/$IID.traj.json"
    [ -f "$r" ] && [ -f "$t" ] || continue
    python oracle/teacher_label.py --traj "$t" --task SWE-bench_Lite.jsonl \
      --instance-id "$IID" --report "$r" --out "out_teacher_${PREFIX}_$i" \
      >> "$RES/TEACHER.txt" 2>&1
    cp "out_teacher_${PREFIX}_$i/teacher_label.json" "$RES/${i}_teacher.json" 2>/dev/null
  done
  push "teacher-$N"
fi

echo "=== 完成 ==="
cat "$RES/SUMMARY.txt"
