#!/bin/bash
# 方案B 全流程收尾：修复后的 HEDL / RS-NN 在 GPU3/GPU4 补齐 5 种子，
# 结束后自动 make_table.py + 渲染论文风格 table.png（仿 HEDL 主表效果图）。
set -u
cd "$(dirname "$0")"
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python

( $PY run_experiment.py --methods hedl --seeds 1000 1001 1002 1003 1004 --gpu 3 \
    > /tmp/opencode/dit_hedl_final.log 2>&1 ) &
H=$!
( $PY run_experiment.py --methods rsnn --seeds 1000 1001 1002 1003 1004 --gpu 4 \
    > /tmp/opencode/dit_rsnn_final.log 2>&1 ) &
R=$!

wait $H $R
$PY make_table.py > /tmp/opencode/dit_table_final.log 2>&1
$PY render_table_png.py > /tmp/opencode/dit_png_final.log 2>&1
echo "ALL DONE $(date)" >> /tmp/opencode/dit_final_done.log
