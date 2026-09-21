#!/bin/bash
# 复数（data_plural）全自动编排：
#   0. 等待实数管线收尾（results_g_mamba2_50 达 24 json + softmax-fix ALL DONE）
#   1. SD-VAE 编码 9 个面板（断点续跑，npy 已存在自动跳过）
#   2. 依序 4 个嫁接设置：嫁接前(原始DiT) → swa50 → hyena50 → mamba2_50
#      每个：GPU1-4 各绑一方法，逐面板×6种子；面板结束后校验 json，缺失种子补跑一次
#   3. 生成附表（实数）+ 主表（复数 9 面板）
# 用法: setsid nohup bash run_plural_orchestrate.sh > /tmp/opencode/plural_orchestrate.log 2>&1 &
set -u
cd /supcon3/shixisheng/xjx/research/project
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
RESULTS_M2=/supcon3/shixisheng/xjx/research/project/results_g_mamba2_50
PANELS="OCTDL BreastMRI MCND SARAircraft NewFUSAR labeledOCT RetinalOCT OpenSARUrban CADCardiac"
SEEDS="1000 1001 1002 1003 1004 1005"
LOG=/tmp/opencode

echo "[$(date)] ===== plural orchestrate started ====="

# ---- 0. 等实数收尾（mamba2_50 24/24 json 且 softmax-fix 完成）----
while true; do
  n=$(ls "$RESULTS_M2"/*.json 2>/dev/null | wc -l)
  done_fix=$(grep -c 'ALL DONE mamba2_50 softmax-fix' "$LOG/graft_done.log" 2>/dev/null || echo 0)
  if [ "$n" -ge 24 ] && [ "$done_fix" -ge 1 ]; then break; fi
  echo "[$(date)] waiting real pipeline: mamba2 jsons=$n/24 softmax_fix_done=$done_fix"
  sleep 300
done
echo "[$(date)] real pipeline complete. Encoding plural latents..."

# ---- 1. 编码（GPU1；断点续跑）----
$PY encode_vae_latents_plural.py --gpu 1 > "$LOG/encode_plural.log" 2>&1 \
  || { echo "[$(date)] ENCODE FAILED"; exit 1; }
# 校验 9 面板 × 5 个 npy 文件齐全
for p in $PANELS; do
  for s in train train_y train_flip test test_y; do
    [ -f "data/vae_latents_plural/${p}_${s}.npy" ] \
      || { echo "[$(date)] MISSING latent ${p}_${s}.npy"; exit 1; }
  done
done
echo "[$(date)] latents ready."

# ---- 2. 训练：4 个嫁接设置 ----
run_setting () {
  local SPEC=$1
  for M in softmax:1 reedl:2 hedl:3 rsnn:4; do
    local METHOD=${M%%:*} GPU=${M##*:}
    (
      for P in $PANELS; do
        export GRAFT="$SPEC" CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/triton_cache_$GPU
        $PY run_experiment_plural.py --panels $P --methods $METHOD --seeds $SEEDS \
          >> "$LOG/plural_${SPEC}_${METHOD}.log" 2>&1
        # 校验 6 个 json，缺则补跑一次（继承上面的 export 环境）
        MISSING=""
        for S in $SEEDS; do
          [ -f "results_plural${SPEC:+_g_$SPEC}/panel_$P/${METHOD}_s${S}.json" ] || MISSING="$MISSING $S"
        done
        if [ -n "$MISSING" ]; then
          echo "[$(date)] retry $SPEC/$METHOD/$P seeds:$MISSING" >> "$LOG/plural_retry.log"
          $PY run_experiment_plural.py --panels $P --methods $METHOD --seeds $MISSING \
            >> "$LOG/plural_${SPEC}_${METHOD}.log" 2>&1
        fi
      done
    ) &
  done
  wait
  # 全设置 json 总数校验
  local TOTAL=$(find results_plural${SPEC:+_g_$SPEC} -name '*.json' 2>/dev/null | wc -l)
  echo "[$(date)] setting '$SPEC' done: $TOTAL/216 jsons"
  [ "$TOTAL" -ge 216 ] || { echo "[$(date)] SETTING $SPEC INCOMPLETE ($TOTAL/216)"; exit 1; }
}

run_setting ""            # 嫁接前（原始 DiT）
run_setting swa50
run_setting hyena50
run_setting mamba2_50

# ---- 3. 两张大表 ----
SEEDS_CSV="1000,1001,1002,1003,1004,1005" $PY combine_tables.py --real \
  > "$LOG/combine_real.log" 2>&1
SEEDS_CSV="1000,1001,1002,1003,1004,1005" $PY combine_tables.py --plural \
  > "$LOG/combine_plural.log" 2>&1
$PY gen_report.py > "$LOG/gen_report.log" 2>&1
echo "[$(date)] ===== PLURAL PIPELINE COMPLETE ====="
echo "PLURAL COMPLETE $(date)" > "$LOG/plural_done.log"
