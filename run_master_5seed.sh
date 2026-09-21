#!/bin/bash
# 复数总编排（5 种子版：1000-1004）：
#   1. 等待 latent 编码（OpenSARUrban/CADCardiac 补码）完成并校验 9 面板 npy 齐全
#   2. 依次 4 个嫁接设置：嫁接前 → swa50 → hyena50 → mamba2_50
#      每个：GPU1-4 各绑一方法（softmax/reedl/hedl/rsnn），9 面板 × 5 种子
#   3. 生成复数主表 combine_tables.py --plural
# 用法: setsid nohup bash run_master_5seed.sh > /tmp/opencode/master_5seed.log 2>&1 &
set -u
cd /supcon3/shixisheng/xjx/research/project
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
PANELS="OCTDL BreastMRI MCND SARAircraft NewFUSAR labeledOCT RetinalOCT OpenSARUrban CADCardiac"

echo "[$(date)] step1: 等待 latent 编码完成..."
while pgrep -f encode_vae_latents_plural.py > /dev/null; do sleep 60; done
LAT=/supcon3/shixisheng/xjx/research/data/vae_latents_plural
for p in $PANELS; do
  for s in train train_y train_flip test test_y; do
    [ -f "${LAT}/${p}_${s}.npy" ] \
      || { echo "[$(date)] MISSING ${p}_${s}.npy - abort"; exit 1; }
  done
done
echo "[$(date)] latents ready (9 panels)."

echo "[$(date)] step2: 复数训练 4 个嫁接设置（每个 4 方法 × 9 面板 × 5 种子）"
for spec in "" swa50 hyena50 mamba2_50; do
  echo "[$(date)] plural GRAFT='$spec' START"
  GRAFT=$spec bash run_plural_all.sh > /tmp/opencode/plural_${spec:-pre}_all.log 2>&1
  n=$(find results_plural${spec:+_g_$spec} -name '*.json' 2>/dev/null | wc -l)
  echo "[$(date)] plural GRAFT='$spec' DONE: $n/180 jsons"
  [ "$n" -ge 180 ] || { echo "[$(date)] SETTING '$spec' INCOMPLETE ($n/180) - abort"; exit 1; }
done

echo "[$(date)] step3: 生成复数主表"
$PY combine_tables.py --plural > /tmp/opencode/combine_plural.log 2>&1 \
  || { echo "[$(date)] combine plural FAILED"; exit 1; }
echo "PLURAL 5-SEED COMPLETE $(date)" >> /tmp/opencode/pipeline_done.log
echo "[$(date)] ALL DONE"