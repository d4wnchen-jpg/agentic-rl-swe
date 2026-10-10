#!/usr/bin/env bash
# GRPO 训练监控（只读，不干扰训练）
#
# 用法:
#   export SWESMITH_RL_HOME=/path/to/workspace
#   bash scripts/watch_grpo.sh
#
# Ctrl-C 退出，不影响训练进程。

set -uo pipefail

ROOT="${SWESMITH_RL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

LOG="$ROOT/log_grpo.txt"

while pgrep -f train_grpo >/dev/null 2>&1; do
  clear
  echo "════════ GRPO 训练监控 $(date +%H:%M:%S) ════════"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi 不可用)"
  echo
  echo "── 日志尾部 ──"
  tail -20 "$LOG" 2>/dev/null || echo "  (还没有 $LOG)"
  echo
  echo "── 最近几步的指标 ──"
  if [ -f "$ROOT/log_grpo.jsonl" ]; then
    python3 - "$ROOT/log_grpo.jsonl" <<'PY' 2>/dev/null || tail -3 "$ROOT/log_grpo.jsonl"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
print(f"{'step':>4} {'train':>7} {'val':>7} {'clip%':>6} {'零方差':>6}")
for o in rows[-6:]:
    v = f"{o['val_rate']*100:.1f}%" if o.get("val_rate") is not None else "  -"
    print(f"{o['step']:>4} {o['train_rate']*100:>6.1f}% {v:>7} "
          f"{o.get('clip_frac', 0)*100:>5.2f}% {o.get('n_zero_var', 0):>6}")
PY
  else
    echo "  (还没有 log_grpo.jsonl)"
  fi
  sleep 60
done

echo "✅ 训练进程已结束"
tail -30 "$LOG" 2>/dev/null
