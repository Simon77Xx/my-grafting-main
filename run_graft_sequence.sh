#!/bin/bash
# 按顺序跑三个 grafted 算子（每算子 GPU1-4，4 方法并行），串行衔接，自动出三张表。
# 用法: setsid nohup bash run_graft_sequence.sh > /tmp/opencode/graft_sequence.log 2>&1 &
set -u
cd "$(dirname "$0")"

for spec in hyena50 swa50 mamba2_50; do
    if [ -f /tmp/opencode/graft_done.log ] && grep -q "ALL DONE $spec" /tmp/opencode/graft_done.log; then
        echo "[$(date)] $spec already done, skip"
        continue
    fi
    echo "[$(date)] START $spec (4 methods x 6 seeds on GPU1-4)"
    export GRAFT=$spec
    bash run_graft_all.sh  # 内部 wait 全部子进程 + 自动 make_table + render => 完成后才返回
    echo "[$(date)] FINISHED $spec"
done
echo "[$(date)] ALL THREE GRAFTED VARIANTS COMPLETE"