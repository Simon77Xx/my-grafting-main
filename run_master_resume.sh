#!/bin/bash
# 总编排：① 实数 mamba2_50 补跑 s1005（hedl/reedl/rsnn 被杀 + softmax 在跑）
#         ② 生成 mamba2_50 表 + 实数附表（combine_tables.py --real）
#         ③ 复数全量：graft='' → swa50 → hyena50 → mamba2_50（各 4 方法 × 9 面板 × 6 种子）
#         ④ 复数主表（combine_tables.py --plural）
# 用法: setsid nohup bash run_master_resume.sh > /tmp/opencode/master_resume.log 2>&1 &
set -u
cd /supcon3/shixisheng/xjx/research/project
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
R=/supcon3/shixisheng/xjx/research/project/results_g_mamba2_50

echo "[$(date)] step1: 补跑 mamba2_50 hedl/reedl s1005 (GPU3/GPU4)"
GRAFT=mamba2_50 CUDA_VISIBLE_DEVICES=3 nohup $PY run_experiment.py --methods hedl --seeds 1005 \
  > /tmp/opencode/mamba2_50_hedl_s1005fix.log 2>&1 &
PH=$!
GRAFT=mamba2_50 CUDA_VISIBLE_DEVICES=4 nohup $PY run_experiment.py --methods reedl --seeds 1005 \
  > /tmp/opencode/mamba2_50_reedl_s1005fix.log 2>&1 &
PR=$!

echo "[$(date)] step2: 等复数 latent 编码完成 → 补跑 rsnn s1005 (GPU2)"
while pgrep -f encode_vae_latents_plural.py > /dev/null; do sleep 30; done
GRAFT=mamba2_50 CUDA_VISIBLE_DEVICES=2 nohup $PY run_experiment.py --methods rsnn --seeds 1005 \
  > /tmp/opencode/mamba2_50_rsnn_s1005fix.log 2>&1 &
PS=$!

echo "[$(date)] step3: 等实数 softmax s1005（GPU1 原进程）"
while pgrep -f "run_experiment.py --methods softmax --seeds 1001 1002 1003 1004 1005" > /dev/null; do sleep 60; done

wait $PH $PR $PS
echo "[$(date)] step4: 实数全部补跑完成，生成表"
$PY make_table.py --results_dir "$R" > /tmp/opencode/mamba2_50_table2.log 2>&1
$PY render_table_png.py --results_dir "$R" >> /tmp/opencode/mamba2_50_table2.log 2>&1
$PY combine_tables.py --real > /tmp/opencode/combine_real.log 2>&1
echo "REAL PART COMPLETE $(date)" >> /tmp/opencode/pipeline_done.log

echo "[$(date)] step5: 复数全量（4 个嫁接设置依次）"
for spec in "" swa50 hyena50 mamba2_50; do
  echo "[$(date)] plural GRAFT='$spec' START" >> /tmp/opencode/plural_progress.log
  GRAFT=$spec bash run_plural_all.sh > /tmp/opencode/plural_${spec:-pre}_all.log 2>&1
  echo "[$(date)] plural GRAFT='$spec' DONE" >> /tmp/opencode/plural_progress.log
done

echo "[$(date)] step6: 生成复数主表"
$PY combine_tables.py --plural > /tmp/opencode/combine_plural.log 2>&1
echo "ALL COMPLETE $(date)" >> /tmp/opencode/pipeline_done.log