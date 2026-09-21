#!/bin/bash
# mamba2_50 retry: Triton cache race fix (per-GPU TRITON_CACHE_DIR)
set -u
cd /supcon3/shixisheng/xjx/research/project
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
RESULTS=/supcon3/shixisheng/xjx/research/project/results_g_mamba2_50
mkdir -p "$RESULTS"

run_one () {
  local M=$1; local GPU=$2; local SEEDS=$3
  GRAFT=mamba2_50 CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_$GPU \
    nohup $PY run_experiment.py --methods $M --seeds $SEEDS \
    > /tmp/opencode/mamba2_50_${M}.log 2>&1 &
}

echo "[$(date)] mamba2_50 retry starting (canary softmax GPU1 first)..."
# Canary: softmax alone first, verify no Triton crash before launching the rest
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton_cache_1 \
  $PY run_experiment.py --methods softmax --seeds 1000 > /tmp/opencode/mamba2_50_softmax.log 2>&1
if [ ! -f "$RESULTS/softmax_s1000.json" ]; then
  echo "[$(date)] mamba2 canary FAILED - aborting"; exit 1
fi
echo "[$(date)] mamba2 canary PASS - launching full run..."

# softmax s1000 done; run softmax s1001-1005 on GPU1 alongside other methods
run_one softmax 1  "1001 1002 1003 1004 1005"
run_one reedl   2  "1000 1001 1002 1003 1004 1005"
run_one hedl    3  "1000 1001 1002 1003 1004 1005"
run_one rsnn    4  "1000 1001 1002 1003 1004 1005"
wait
echo "[$(date)] mamba2_50 all methods finished."

SEEDS_ALL="1000 1001 1002 1003 1004 1005"
$PY make_table.py --results_dir "$RESULTS" >> /tmp/opencode/mamba2_50_table.log 2>&1 \
  || SEEDS="$SEEDS_ALL" $PY make_table.py --results_dir "$RESULTS" >> /tmp/opencode/mamba2_50_table.log 2>&1
$PY render_table_png.py --results_dir "$RESULTS" >> /tmp/opencode/mamba2_50_table.log 2>&1
echo "ALL DONE mamba2_50 (retry) $(date)" >> /tmp/opencode/graft_done.log

echo "[$(date)] chaining pre-graft evaluation..."
bash /supcon3/shixisheng/xjx/research/project/run_pre_graft.sh > /tmp/opencode/pre_graft_runner.log 2>&1
echo "[$(date)] ALL PIPELINE COMPLETE"
echo "PIPELINE COMPLETE $(date)" > /tmp/opencode/pipeline_done.log