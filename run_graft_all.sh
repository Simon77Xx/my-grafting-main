#!/bin/bash
# 嫁接后全量训练：GRAFT=hyena50|swa50|mamba2_50，4 方法 × 6 种子，GPU1-4 各一方法。
# 用法: GRAFT=hyena50 bash run_graft_all.sh
# 结束后自动 make_table.py + render_table_png.py（同 GRAFT 环境 → results_g_<spec>/）。
set -u
cd "$(dirname "$0")"
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
export GRAFT=${GRAFT:?need GRAFT env, e.g. hyena50}

( $PY run_experiment.py --methods softmax --seeds 1000 1001 1002 1003 1004 1005 --gpu 1 \
    > /tmp/opencode/${GRAFT}_softmax.log 2>&1 ) &
P1=$!
( $PY run_experiment.py --methods reedl --seeds 1000 1001 1002 1003 1004 1005 --gpu 2 \
    > /tmp/opencode/${GRAFT}_reedl.log 2>&1 ) &
P2=$!
( $PY run_experiment.py --methods hedl --seeds 1000 1001 1002 1003 1004 1005 --gpu 3 \
    > /tmp/opencode/${GRAFT}_hedl.log 2>&1 ) &
P3=$!
( $PY run_experiment.py --methods rsnn --seeds 1000 1001 1002 1003 1004 1005 --gpu 4 \
    > /tmp/opencode/${GRAFT}_rsnn.log 2>&1 ) &
P4=$!

wait $P1 $P2 $P3 $P4
$PY make_table.py > /tmp/opencode/${GRAFT}_table.log 2>&1
$PY render_table_png.py > /tmp/opencode/${GRAFT}_png.log 2>&1
echo "ALL DONE $GRAFT $(date)" >> /tmp/opencode/graft_done.log
