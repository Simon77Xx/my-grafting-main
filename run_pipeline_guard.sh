#!/bin/bash
# 流水线守护（断点续跑 v2）：负责反复拉起 run_master_v2.py。
#   - master 意外退出（崩溃/被杀）→ 60s 后自动续跑（skip 逻辑保证不重做已完成项）
#   - 磁盘金丝雀：启动前测试写文件，磁盘配额满 → 每 10 分钟重试，避免空转
#   - 全队列跑完但存在失败任务 → 最多再补跑 5 轮；无失败 → 退出
#
# 用法: setsid nohup bash run_pipeline_guard.sh > /tmp/opencode/guard.log 2>&1 &
cd "$(dirname "$0")" || exit 1
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
LOG=/tmp/opencode/master_v2_run2.log
DONE=/tmp/opencode/pipeline_done.log
FAIL=/tmp/opencode/v2_failures.log
MAX_EXTRA=5
extra=0

canary_ok() { echo "canary $(date +%s)" > /tmp/opencode/.canary 2>/dev/null; }

launch() {
  echo "[guard $(date '+%F %T')] launch master (extra_retry=$extra)" >> "$LOG"
  "$PY" run_master_v2.py >> "$LOG" 2>&1
  echo "[guard $(date '+%F %T')] master exited rc=$?" >> "$LOG"
}

launch
while true; do
  if grep -q "MASTER V2 COMPLETE" "$DONE" 2>/dev/null; then
    if [ -s "$FAIL" ] && [ "$extra" -lt "$MAX_EXTRA" ]; then
      extra=$((extra + 1))
      echo "[guard $(date '+%F %T')] complete but failures recorded; retry round $extra" >> "$LOG"
      sleep 120
      launch
      continue
    fi
    echo "[guard $(date '+%F %T')] pipeline COMPLETE, guard stops" >> "$LOG"
    exit 0
  fi
  until canary_ok; do
    echo "[guard $(date '+%F %T')] disk canary FAILED (quota full), wait 10min" >> "$LOG"
    sleep 600
  done
  sleep 60
  launch
done

