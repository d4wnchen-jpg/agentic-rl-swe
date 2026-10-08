#!/usr/bin/env bash
# GRPO 训练启动脚本
# 用法: export SWESMITH_RL_HOME=/path/to/workspace && bash scripts/run_train.sh

set -euo pipefail

ROOT="${SWESMITH_RL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1

LOG="$ROOT/log_grpo.txt"
[ -f "$LOG" ] && mv "$LOG" "${LOG%.txt}_$(date +%Y%m%d_%H%M%S).txt"

setsid nohup bash -c "cd '$ROOT'; python -u tools/train_grpo.py" > "$LOG" 2>&1 &

echo "训练已启动"
echo "  日志: $LOG"
echo "  查看: tail -f $LOG"
