#!/bin/bash
# hint 实验：从最终状态继续，只改变给 agent 的信息量 / 步数预算。
# 用法: bash oracle/run_hint_batch.sh <cases.json> <image> [steps] [outdir] [results_dir] [iid]
set -u
cd ${ROOT:-/workspaces/agentic-rl-swe}
CASES=${1:?cases.json}
IMAGE=${2:?image}
STEPS=${3:-8}
STEPS_LONG=${STEPS_LONG:-24}
OUTDIR=${4:-results/hint_pilot}
RESDIR=${5:-results/pilot}
IID=${6:-pallets__flask-5063}
CONDS=${CONDS:-"none legal oracle legal_long oracle_long"}
mkdir -p "$OUTDIR"

python3 - "$CASES" > /tmp/hint_cases.txt <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])):
    print(r["run"])
PY

for run in $(cat /tmp/hint_cases.txt); do
  traj="$RESDIR/$run.traj.json"
  report="$RESDIR/$run.json"
  [ -f "$traj" ] || { echo "skip $run (no traj)"; continue; }
  [ -f "$report" ] || { echo "skip $run (no report)"; continue; }
  for cond in $CONDS; do
    condition=continue
    case "$cond" in
      none)        hint=none;   steps=$STEPS ;;
      legal)       hint=legal;  steps=$STEPS ;;
      oracle)      hint=oracle; steps=$STEPS ;;
      legal_long)  hint=legal;  steps=$STEPS_LONG ;;
      oracle_long) hint=oracle; steps=$STEPS_LONG ;;
      revert_none)   hint=none;   steps=$STEPS; condition=revert_ours ;;
      revert_legal)  hint=legal;  steps=$STEPS; condition=revert_ours ;;
      revert_oracle) hint=oracle; steps=$STEPS; condition=revert_ours ;;
      *) echo "unknown cond $cond"; continue ;;
    esac
    out="$OUTDIR/${run}_${cond}.json"
    if [ -f "$out" ]; then echo "skip $run/$cond"; continue; fi
    echo "=== $run / $cond (hint=$hint steps=$steps) ==="
    python oracle/repair_resume.py \
      --traj "$traj" --task SWE-bench_Lite.jsonl --instance-id "$IID" \
      --image "$IMAGE" --report "$report" \
      --condition "$condition" --hint "$hint" --steps "$steps" --seed "$run" \
      --out "$out" || echo "FAIL $run/$cond"
    git add -f "$out" >/dev/null 2>&1
    git commit -qm "hint: $run $cond" >/dev/null 2>&1
    git push -q >/dev/null 2>&1 || true
  done
done
echo "=== hint 批量完成 ==="
