#!/bin/bash
# 全量收尾（含 Places365）：4 方法 × 5 种子，GPU1-4 各跑一个方法，
# 全部结束后自动 make_table.py + render_table_png.py。
set -u
cd "$(dirname "$0")"
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python

( $PY run_experiment.py --methods softmax --seeds 1000 1001 1002 1003 1004 --gpu 1 \
    > /tmp/opencode/final_softmax.log 2>&1 ) &
P1=$!
( $PY run_experiment.py --methods reedl --seeds 1000 1001 1002 1003 1004 --gpu 2 \
    > /tmp/opencode/final_reedl.log 2>&1 ) &
P2=$!
( $PY run_experiment.py --methods hedl --seeds 1000 1001 1002 1003 1004 --gpu 3 \
    > /tmp/opencode/final_hedl.log 2>&1 ) &
P3=$!
( $PY run_experiment.py --methods rsnn --seeds 1000 1001 1002 1003 1004 --gpu 4 \
    > /tmp/opencode/final_rsnn.log 2>&1 ) &
P4=$!

wait $P1 $P2 $P3 $P4
$PY make_table.py > /tmp/opencode/final_table.log 2>&1
$PY render_table_png.py > /tmp/opencode/final_png.log 2>&1
echo "ALL DONE $(date)" >> /tmp/opencode/final_done.log
