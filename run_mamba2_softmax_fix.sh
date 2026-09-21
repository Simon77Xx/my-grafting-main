#!/bin/bash
# mamba2_50 softmax 补跑修复：金丝雀这次必须带 GRAFT=mamba2_50（上次根因：金丝雀漏了 GRAFT，
# 训成嫁接前模型且 json 写错目录 → 误判 CANARY FAILED → softmax 6 种子全没跑）。
# 表格阶段必须 export GRAFT，否则 make_table.py 会把 results_g_mamba2_50 的表双写到
# results/（嫁接前目录）造成覆写污染。
set -u
cd /supcon3/shixisheng/xjx/research/project
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
RESULTS=/supcon3/shixisheng/xjx/research/project/results_g_mamba2_50
mkdir -p "$RESULTS"

echo "[$(date)] softmax fix starting: canary s1000 with GRAFT=mamba2_50 (GPU1)..."
GRAFT=mamba2_50 CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton_cache_1 \
  $PY run_experiment.py --methods softmax --seeds 1000 > /tmp/opencode/mamba2_50_softmax_fix.log 2>&1
if [ ! -f "$RESULTS/softmax_s1000.json" ]; then
  echo "[$(date)] softmax fix canary FAILED (no json in $RESULTS) - aborting"
  exit 1
fi
python3 - "$RESULTS/softmax_s1000.json" <<'EOF' || { echo "[$(date)] canary json invalid"; exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
assert "acc" in d and "SVHN" in d and "Places365" in d, d.keys()
print("canary json OK:", {k: d[k] for k in ("acc",)})
EOF
echo "[$(date)] softmax fix canary PASS - launching s1001-1005..."
GRAFT=mamba2_50 CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton_cache_1 \
  nohup $PY run_experiment.py --methods softmax --seeds 1001 1002 1003 1004 1005 \
  >> /tmp/opencode/mamba2_50_softmax_fix.log 2>&1
echo "[$(date)] softmax 6 seeds done."

# 表格：必须 export GRAFT，让 config.RESULTS_DIR 指向 results_g_mamba2_50，避免双写污染 results/
export GRAFT=mamba2_50
SEEDS="1000,1001,1002,1003,1004,1005" $PY make_table.py --results_dir "$RESULTS" \
  >> /tmp/opencode/mamba2_50_table.log 2>&1
SEEDS="1000,1001,1002,1003,1004,1005" $PY render_table_png.py --results_dir "$RESULTS" \
  >> /tmp/opencode/mamba2_50_table.log 2>&1
echo "ALL DONE mamba2_50 softmax-fix $(date)" >> /tmp/opencode/graft_done.log
echo "[$(date)] mamba2_50 COMPLETE (24/24 jsons: $(ls $RESULTS/*.json 2>/dev/null | wc -l))"