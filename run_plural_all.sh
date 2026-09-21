#!/bin/bash
# 复数阶段全量：GRAFT=''|swa50|hyena50|mamba2_50，4 方法 × 9 面板 × 6 种子，GPU1-4 各一方法。
# 用法: GRAFT=swa50 bash run_plural_all.sh   （GRAFT 为空 = 嫁接前）
# run_experiment_plural.py 内置 skip：已有 json 自动跳过，可安全重试。
set -u
cd "$(dirname "$0")"
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
export GRAFT=${GRAFT-}
TAG=${GRAFT:-pre}

( GRAFT=$GRAFT CUDA_VISIBLE_DEVICES=1 $PY run_experiment_plural.py --methods softmax --gpu 0 \
    > /tmp/opencode/plural_${TAG}_softmax.log 2>&1 ) &
P1=$!
( GRAFT=$GRAFT CUDA_VISIBLE_DEVICES=2 $PY run_experiment_plural.py --methods reedl --gpu 0 \
    > /tmp/opencode/plural_${TAG}_reedl.log 2>&1 ) &
P2=$!
( GRAFT=$GRAFT CUDA_VISIBLE_DEVICES=3 $PY run_experiment_plural.py --methods hedl --gpu 0 \
    > /tmp/opencode/plural_${TAG}_hedl.log 2>&1 ) &
P3=$!
( GRAFT=$GRAFT CUDA_VISIBLE_DEVICES=4 $PY run_experiment_plural.py --methods rsnn --gpu 0 \
    > /tmp/opencode/plural_${TAG}_rsnn.log 2>&1 ) &
P4=$!

wait $P1 $P2 $P3 $P4
echo "DONE plural GRAFT=$GRAFT $(date)" >> /tmp/opencode/plural_done.log