#!/bin/bash
# 复数流水线进度速览（v2：7 方法 × 6 面板 × 4 嫁接设置 × 5 种子）。用法: bash check_progress.sh
cd "$(dirname "$0")"
PANELS="OCTDL BreastMRI MCND SARAircraft NewFUSAR RetinalOCT"
METHODS="softmax reedl hedl rsnn iedl duq mcdropout deepens"
echo "===== $(date) ====="
echo "--- 编排进程 ---"
pgrep -af 'run_master_v2|run_master_5seed|run_plural_all' | sed 's/^/  /' || echo "  (无 - 已结束或中断)"
echo "--- 主编排日志 ---"
M=$(ls -t /tmp/opencode/master_v2*.log 2>/dev/null | head -1)
if [ -n "$M" ]; then echo "  (log: $M)"; tail -8 "$M"; else tail -5 /tmp/opencode/master_5seed.log 2>/dev/null; fi
echo "--- 各设置完成度（目标 210/210 json；deepens 为额外单文件）---"
for d in results_plural results_plural_g_swa50 results_plural_g_hyena50 results_plural_g_mamba2_50; do
  n=$(find "$d" -name '*.json' 2>/dev/null | grep -v deepens | wc -l)
  de=$(find "$d" -name 'deepens.json' 2>/dev/null | wc -l)
  printf "  %-28s %3d/210  (deepens: %d/24)\n" "$d" "$n" "$de"
done
echo "--- 各面板明细（按方法）---"
for p in $PANELS; do
  line="  $(printf '%-12s' $p):"
  for m in $METHODS; do
    n=$(ls results_plural/panel_$p/${m}_s*.json 2>/dev/null | wc -l)
    line="$line ${m}:${n}"
  done
  echo "$line"
done
echo "--- 各方法最新日志 ---"
for lg in /tmp/opencode/v2_logs/*.log; do
  [ -f "$lg" ] || continue
  last=$(tail -1 "$lg" | cut -c1-70)
  [ -n "$last" ] && echo "  $(basename $lg .log): $last"
done 2>/dev/null | tail -12
echo "--- GPU ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null