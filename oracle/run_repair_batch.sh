#!/bin/bash
# 批量修复实验：对候选轨迹逐个跑 4 个条件，断点续跑 + 自动入库。
# 用法: bash oracle/run_repair_batch.sh <cases.json> <image> [steps] [outdir] [results_dir] [iid]
set -u
cd /workspaces/agentic-rl-swe
CASES=${1:?cases.json}
IMAGE=${2:?image}
STEPS=${3:-8}
OUTDIR=${4:-results/repair_d1a}
RESDIR=${5:-results/d1a}
IID=${6:-pallets__flask-5063}
mkdir -p "$OUTDIR"

python3 - "$CASES" > /tmp/repair_cases.txt <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
for r in rows:
    print(r["run"])
PY

for run in $(cat /tmp/repair_cases.txt); do
  traj="$RESDIR/$run.traj.json"
  report="$RESDIR/$run.json"
  if [ ! -f "$traj" ] || [ ! -f "$report" ]; then echo "skip $run (缺文件)"; continue; fi

  # 教师条件需要 teacher json，缺则现跑一次
  if [ ! -f "$RESDIR/${run}_teacher.json" ]; then
    python oracle/teacher_label.py --traj "$traj" --task SWE-bench_Lite.jsonl \
      --instance-id "$IID" --report "$report" --out "out_teacher_${run}" \
      > "teacher_${run}.log" 2>&1 || echo "teacher label fail $run"
    cp "out_teacher_${run}/teacher_label.json" "$RESDIR/${run}_teacher.json" 2>/dev/null
  fi

  for cond in restart continue revert_random revert_teacher revert_ours; do
    out="$OUTDIR/${run}_${cond}.json"
    if [ -f "$out" ]; then echo "skip $run/$cond"; continue; fi
    echo "=== $run / $cond ==="
    python oracle/repair_resume.py \
      --traj "$traj" --task SWE-bench_Lite.jsonl --instance-id "$IID" \
      --image "$IMAGE" --report "$report" \
      --teacher "$RESDIR/${run}_teacher.json" \
      --condition "$cond" --steps "$STEPS" --seed "$run" \
      --out "$out" || echo "FAIL $run/$cond"
    git add -f "$out" >/dev/null 2>&1
    git commit -qm "repair: $run $cond" >/dev/null 2>&1
    git push -q >/dev/null 2>&1 || true
  done
done

echo "=== 批量完成，汇总 ==="
python oracle/stats_repair.py "$OUTDIR"
