#!/bin/bash
# 方案B 收尾：修复后的 HEDL / RS-NN 头补齐 5 种子，结束后自动出表。
# hedl -> GPU1, rsnn -> GPU2（并行，各 ~15h）
set -u
cd "$(dirname "$0")"
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python

( $PY run_experiment.py --methods hedl --seeds 1000 1001 1002 1003 1004 --gpu 1 \
    > /tmp/opencode/dit_hedl_fix.log 2>&1 ) &
H=$!
( $PY run_experiment.py --methods rsnn --seeds 1000 1001 1002 1003 1004 --gpu 2 \
    > /tmp/opencode/dit_rsnn_fix.log 2>&1 ) &
R=$!

wait $H $R
$PY make_table.py > /tmp/opencode/dit_table_fix.log 2>&1
echo "ALL DONE $(date)" >> /tmp/opencode/dit_fix_done.log
