#!/bin/bash
# 方案B DiT 全量实验调度：4方法 × 5种子，按 GPU 并行。
# 每方法一个 GPU，顺序跑 5 种子。冒烟（softmax s1000）已单独跑过则跳过。
# 用法: bash run_dit_full.sh
set -u
cd "$(dirname "$0")"
source /home/shixisheng/miniforge3/etc/profile.d/conda.sh
conda activate env_pt
export HF_ENDPOINT=https://hf-mirror.com

GPU_SOFTMAX=1   # 冒烟已占；全量时换成空闲 GPU
GPU_REEDL=6
GPU_HEDL=7
GPU_RSNN=2

SEEDS="1000 1001 1002 1003 1004"

# 已在 GPU1 跑的 softmax s1000 冒烟 —— 全量时会覆盖写，故 softmax 全量不重跑 s1000。
# 为简单，这里 softmax 直接跑全部 5 种子（s1000 会重跑，约多 3h）。
setsid nohup python run_experiment.py --methods softmax --seeds $SEEDS --gpu $GPU_SOFTMAX \
  > /tmp/opencode/dit_softmax_full.log 2>&1 &
setsid nohup python run_experiment.py --methods reedl --seeds $SEEDS --gpu $GPU_REEDL \
  > /tmp/opencode/dit_reedl_full.log 2>&1 &
setsid nohup python run_experiment.py --methods hedl --seeds $SEEDS --gpu $GPU_HEDL \
  > /tmp/opencode/dit_hedl_full.log 2>&1 &
setsid nohup python run_experiment.py --methods rsnn --seeds $SEEDS --gpu $GPU_RSNN \
  > /tmp/opencode/dit_rsnn_full.log 2>&1 &
echo "launched: softmax(g$GPU_SOFTMAX) reedl(g$GPU_REEDL) hedl(g$GPU_HEDL) rsnn(g$GPU_RSNN)"