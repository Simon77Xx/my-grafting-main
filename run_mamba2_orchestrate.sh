#!/bin/bash
# mamba2_50 orchestrator v2: canary (softmax s1000) already running on GPU1.
# Launches reedl/hedl/rsnn now on GPU2-4, softmax s1001-1005 after canary exits,
# then table + pre-graft chain.
set -u
cd /supcon3/shixisheng/xjx/research/project
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
RESULTS=/supcon3/shixisheng/xjx/research/project/results_g_mamba2_50
mkdir -p "$RESULTS"

launch () {
  local M=$1; local GPU=$2; local SEEDS=$3
  GRAFT=mamba2_50 CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_$GPU \
    nohup $PY run_experiment.py --methods $M --seeds $SEEDS \
    > /tmp/opencode/mamba2_50_${M}.log 2>&1 &
}

echo "[$(date)] launching reedl(GPU2) hedl(GPU3) rsnn(GPU4)..."
launch reedl 2 "1000 1001 1002 1003 1004 1005"
launch hedl  3 "1000 1001 1002 1003 1004 1005"
launch rsnn  4 "1000 1001 1002 1003 1004 1005"

# wait for the canary (softmax s1000, GPU1) to finish its seed
echo "[$(date)] waiting for canary softmax s1000 (GPU1)..."
while pgrep -f "run_experiment.py --methods softmax --seeds 1000" > /dev/null; do sleep 60; done
if [ ! -f "$RESULTS/softmax_s1000.json" ]; then
  echo "[$(date)] CANARY FAILED (no json) - aborting"; exit 1
fi
echo "[$(date)] canary PASS - launching softmax s1001-1005 on GPU1"
launch softmax 1 "1001 1002 1003 1004 1005"

wait
echo "[$(date)] mamba2_50 all methods finished."

$PY make_table.py --results_dir "$RESULTS" >> /tmp/opencode/mamba2_50_table.log 2>&1
$PY render_table_png.py --results_dir "$RESULTS" >> /tmp/opencode/mamba2_50_table.log 2>&1
echo "ALL DONE mamba2_50 (retry) $(date)" >> /tmp/opencode/graft_done.log

echo "[$(date)] chaining pre-graft evaluation..."
bash /supcon3/shixisheng/xjx/research/project/run_pre_graft.sh > /tmp/opencode/pre_graft_runner.log 2>&1
echo "PIPELINE COMPLETE $(date)" > /tmp/opencode/pipeline_done.log