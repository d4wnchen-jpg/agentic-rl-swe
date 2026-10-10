#!/usr/bin/env bash
# GRPO 训练启动脚本
#
# 用法:
#   export SWESMITH_RL_HOME=/path/to/workspace
#   PYTHON=$(which python) bash scripts/run_train.sh
#
# ⚠️ 必须用【装好 vllm / torch 的那个解释器】。
#    容器里常常有多个 python（例如 conda base 和某个 env），
#    用错了会报 ModuleNotFoundError: No module named 'vllm'。
#    不确定的话先跑:  $PYTHON -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
#
# 环境变量覆盖超参:  GRPO_LR=3e-4 GRPO_STEPS=40 bash scripts/run_train.sh
# 复现旧配置的坑:    SMITH_HARNESS_V1=1 bash scripts/run_train.sh

set -euo pipefail

ROOT="${SWESMITH_RL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

# ⚠️ 提前拦截"用错解释器"这个最常见的失败
if ! "$PYTHON" -c "import vllm" 2>/dev/null; then
  echo "❌ $PYTHON 里没有 vllm。"
  echo "   请用装好依赖的解释器，例如:"
  echo "     PYTHON=/path/to/env/bin/python bash scripts/run_train.sh"
  exit 1
fi

export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1

LOG="$ROOT/log_grpo.txt"
[ -f "$LOG" ] && mv "$LOG" "${LOG%.txt}_$(date +%Y%m%d_%H%M%S).txt"

setsid nohup bash -c "cd '$ROOT'; '$PYTHON' -u tools/train_grpo.py" > "$LOG" 2>&1 &

echo "训练已启动"
echo "  解释器: $PYTHON"
echo "  日志:   $LOG"
echo "  查看:   tail -f $LOG"
echo "  监控:   bash scripts/watch_grpo.sh"
