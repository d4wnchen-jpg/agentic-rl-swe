#!/usr/bin/env bash
cd /root/autodl-tmp/agentic
while pgrep -f train_grpo >/dev/null 2>&1; do
  clear
  echo "════════ GRPO 训练监控 $(date +%H:%M:%S) ════════"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  echo
  echo "── 结果文档 ──"
  tail -20 notes/GRPO_RESULT.md 2>/dev/null
  echo
  echo "── 最新日志 ──"
  tail -6 log_grpo.txt
  sleep 60
done
echo "✅ 训练结束"
tail -30 log_grpo.txt
