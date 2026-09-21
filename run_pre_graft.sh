#!/bin/bash
# Pre-graft evaluation: runs HEDL/RS-NN/Re-EDL from their original repos.
set -u
cd "$(dirname "$0")"
PY=/home/shixisheng/miniforge3/envs/env_pt/bin/python
TF=/home/shixisheng/miniforge3/envs/env_tf/bin/python
SEEDS="1000 1001 1002 1003 1004"
RESULTS_PRE=/supcon3/shixisheng/xjx/research/project/results_pre
mkdir -p $RESULTS_PRE

# Wait for post-graft completion
echo "[$(date)] Waiting for grafted variants to complete..."
for i in $(seq 1 1800); do
  done_n=$(grep -c 'ALL DONE ' /tmp/opencode/graft_done.log 2>/dev/null || echo 0)
  if [ "$done_n" -ge 3 ]; then
    echo "[$(date)] All three grafted variants done."
    break
  fi
  sleep 10
done

echo "[$(date)] ===== Pre-graft evaluation started (GPU1-4) ====="

# === Part 2: HEDL ===

# === HEDL (ResNet18 + HEDL head) ===
echo "[$(date)] [HEDL] Starting..."
cd "/supcon3/shixisheng/xjx/research/hedl/5024_Hyper_opinion_Evidential__Supplementary Material/code"
source /home/shixisheng/miniforge3/etc/profile.d/conda.sh && conda activate env_pt

for s in $SEEDS; do
  echo "[$(date)] [HEDL] seed $s - training check..."
  CKPT="runs_openset_cifar10/ResNet18/HEDL_s${s}/acc/checkpoints/checkpoint.pth"
  if [ ! -f "$CKPT" ]; then
    echo "[$(date)] [HEDL] seed $s - 训练 step1: 90-epoch softmax backbone..."
    CUDA_VISIBLE_DEVICES=1 $PY train.py --epoch 90 --name HEDL_s${s} --dataset cifar10 --seed ${s} > "/tmp/opencode/hedl_s${s}_step1.log" 2>&1 || exit 1
    echo "[$(date)] [HEDL] seed $s - 训练 step2: 10-epoch HEDL finetune..."
    CUDA_VISIBLE_DEVICES=1 $PY train.py --epoch 10 --name HEDL_s${s} --dataset cifar10 --loss edl_HENN --seed ${s} --checkpoint_path "$CKPT" > "/tmp/opencode/hedl_s${s}_step2.log" 2>&1 || exit 1
  fi
  echo "[$(date)] [HEDL] seed $s - 评估 Places365 (SVHN/DTD 已在 logs 中有结果)..."
  # SVHN/DTD results already exist in logs/; only Places365 needs fresh evaluation.
  LOGDIR="/tmp/opencode/hedl_s${s}_Places365"
  mkdir -p $LOGDIR
  CUDA_VISIBLE_DEVICES=1 $PY hedl_val_p365.py --seed ${s} > "$LOGDIR/val.log" 2>&1 || exit 1
  cp cifar10/ResNet18/HEDL_s${s}/ood_res_p365.json "$LOGDIR/ood_res.json" 2>/dev/null || true
done

echo "[$(date)] [HEDL] 转换 JSON 格式..."
# Note: existing logs already have SVHN/DTD for all 5 seeds; only Places365 is newly evaluated.
$PY - << 'PYEOF'
import json, os
SEEDS = [1000, 1001, 1002, 1003, 1004]
RESULTS_PRE = "/supcon3/shixisheng/xjx/research/project/results_pre"
OOD_NAMES = ["SVHN", "DTD", "Places365"]
LOGS = "/supcon3/shixisheng/xjx/research/hedl/logs"
CODE = "/supcon3/shixisheng/xjx/research/hedl/5024_Hyper_opinion_Evidential__Supplementary Material/code"
for s in SEEDS:
    result = {"acc": None}
    for ood in OOD_NAMES:
        # Priority 1: per-ood temp result copied by bash loop above
        json_path = f"/tmp/opencode/hedl_s{s}_{ood}/ood_res.json"
        json_p365 = f"/tmp/opencode/hedl_s{s}_Places365/ood_res.json"
        # Priority 2: code output
        if not os.path.exists(json_path):
            json_path = os.path.join(CODE, f"cifar10/ResNet18/HEDL_s{s}/ood_res.json")
        # Priority 3: existing log (SVHN/DTD from prior runs)
        if not os.path.exists(json_path):
            cand = os.path.join(LOGS, f"ood_res_{str(ood).lower()}_seed{s}.json")
            if os.path.exists(cand):
                json_path = cand
        if os.path.exists(json_path):
            with open(json_path) as f:
                d = json.load(f)
            acc_val = float(d.get("acc", 0.0) or 0.0)
            if result["acc"] is None and acc_val > 0:
                result["acc"] = acc_val * 100.0
            # Places365 result has no acc field; keep acc from SVHN
            fpr95 = float(d.get("fpr95_uncertainty", 0.0)) * 100.0
            auroc = float(d.get("auroc_uncertainty", 0.0)) * 100.0
            aupr = float(d.get("auprc_uncertainty", 0.0)) * 100.0
            result[ood] = {"fpr95": fpr95, "aupr": aupr, "auroc": auroc}
        else:
            print(f"  WARNING: HEDL seed {s} {ood} no ood_res.json")
    out_path = f"{RESULTS_PRE}/hedl_s{s}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {out_path}")
PYEOF
echo "[$(date)] [HEDL] Done."

# === RS-NN (ResNet50 + RS head) ===
echo "[$(date)] [RS-NN] Starting..."
cd /supcon3/shixisheng/xjx/research/Random-Set-Neural-Networks-main
source /home/shixisheng/miniforge3/etc/profile.d/conda.sh && conda activate env_tf

for s in $SEEDS; do
  echo "[$(date)] [RS-NN] seed $s - training check..."
  if [ ! -f "CNN_resnet50_cifar10_s${s}_weights.pkl" ] || [ ! -f "RSNN_resnet50_cifar10_s${s}_weights.pkl" ]; then
    echo "[$(date)] [RS-NN] seed $s - 训练..."
    # Copy weights from rsnn_full if available
    for ext in h5 pkl keras; do
      [ -f "/tmp/opencode/rsnn_full/CNN_resnet50_cifar10_s${s}.${ext}" ] && cp "/tmp/opencode/rsnn_full/CNN_resnet50_cifar10_s${s}.${ext}" "CNN_resnet50_cifar10_s${s}.${ext}" 2>/dev/null || true
      [ -f "/tmp/opencode/rsnn_full/RSNN_resnet50_cifar10_s${s}.${ext}" ] && cp "/tmp/opencode/rsnn_full/RSNN_resnet50_cifar10_s${s}.${ext}" "RSNN_resnet50_cifar10_s${s}.${ext}" 2>/dev/null || true
      [ -f "/tmp/opencode/rsnn_full/CNN_resnet50_cifar10_s${s}_weights.pkl" ] && cp "/tmp/opencode/rsnn_full/CNN_resnet50_cifar10_s${s}_weights.pkl" "CNN_resnet50_cifar10_s${s}_weights.pkl" 2>/dev/null || true
      [ -f "/tmp/opencode/rsnn_full/RSNN_resnet50_cifar10_s${s}_weights.pkl" ] && cp "/tmp/opencode/rsnn_full/RSNN_resnet50_cifar10_s${s}_weights.pkl" "RSNN_resnet50_cifar10_s${s}_weights.pkl" 2>/dev/null || true
    done
    if [ ! -f "CNN_resnet50_cifar10_s${s}_weights.pkl" ] || [ ! -f "RSNN_resnet50_cifar10_s${s}_weights.pkl" ]; then
      CUDA_VISIBLE_DEVICES=2 $PY train_rsnn.py --seed $s --epochs 200 --gpu 2 > "/tmp/opencode/rsnn_train_s${s}.log" 2>&1 || exit 1
    fi
  fi
  echo "[$(date)] [RS-NN] seed $s - 评估..."
  CUDA_VISIBLE_DEVICES=2 $TF eval_rsnn.py --seed $s --gpu 2 --outdir saved_models --out_json $RESULTS_PRE/rsnn_s${s}.json > "/tmp/opencode/rsnn_eval_s${s}.log" 2>&1 || exit 1
done
echo "[$(date)] [RS-NN] Done."

# === Re-EDL (VGG + Re-EDL head) ===
echo "[$(date)] [Re-EDL] Starting..."
cd /supcon3/shixisheng/xjx/research/reedl/Re-EDL-main/code_classical
source /home/shixisheng/miniforge3/etc/profile.d/conda.sh && conda activate env_pt
if [ ! -L "data/val_256" ]; then
  ln -sfn /supcon3/shixisheng/xjx/research/data/places365/val_256 data/val_256
fi

for s in $SEEDS; do
  echo "[$(date)] [Re-EDL] seed $s - checking pretrain weight..."
  # All 5 weights live in cifar10-reedl-train-s1000/saved_models/cifar10-reedl-test/
  WEIGHTS_DIR="saved_models/cifar10-reedl-train-s1000/saved_models/cifar10-reedl-test"
  if [ ! -f "$WEIGHTS_DIR/${s}_best" ]; then
    echo "[$(date)] [Re-EDL] seed $s - pretrain weight NOT FOUND at $WEIGHTS_DIR/${s}_best"
    echo "[$(date)] [Re-EDL] seed $s - SKIPPING (will omit this seed)."
    continue
  fi
  echo "[$(date)] [Re-EDL] seed $s - 评估..."
  # Always (re)write the seed-specific test config with correct weight dir
  $PY -c "
import json
cfg = {
  'seeds': [$s],
  'dataset_name': 'CIFAR10',
  'ood_dataset_names': ['SVHN', 'DTD', 'Places365'],
  'split': [0.95, 0.05],
  'directory_model': './saved_models/cifar10-reedl-train-s1000/saved_models/cifar10-reedl-test/',
  'name_model': ['${s}_best'],
  'model_type': 'menet',
  'loss': 'MEDL',
  'clf_type': 'softplus',
  'lr': [0.0001],
  'kl_c': [0.0],
  'fisher_c': [0.0],
  'lamb1_list': [1.0],
  'lamb2_list': [0.8],
  'noise_epsilon': 0.0,
  'architecture': 'vgg',
  'input_dims': [32, 32, 3],
  'output_dim': 10,
  'hidden_dims': [64, 64, 64],
  'kernel_dim': 5,
  'k_lipschitz': None,
  'max_epochs': 200,
  'patience': 10,
  'frequency': 2,
  'batch_size': 64,
  'use_wandb': False,
  'store_results': True,
  'store_stat': False,
}
with open('configs/2_cifar10/cifar10-reedl-test-s${s}.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
  CUDA_VISIBLE_DEVICES=3 $PY main.py --configid "2_cifar10/cifar10-reedl-test-s${s}" --suffix test > "/tmp/opencode/reedl_eval_s${s}.log" 2>&1 || { echo "FAILED eval seed $s"; exit 1; }
done

echo "[$(date)] [Re-EDL] 转换 JSON 格式..."
$PY - << 'PYEOF'
import json, os, csv

SEEDS = [1000, 1001, 1002, 1003, 1004]
RESULTS_PRE = "/supcon3/shixisheng/xjx/research/project/results_pre"
REEDL_DIR = "/supcon3/shixisheng/xjx/research/reedl/Re-EDL-main/code_classical"

for s in SEEDS:
    csv_paths = [
        f"{REEDL_DIR}/saved_models/cifar10-reedl-test-s{s}/cifar10-reedl-test-s{s}.csv",
        f"{REEDL_DIR}/saved_models/cifar10-reedl-train-s{s}/cifar10-reedl-test-s{s}.csv",
    ]
    csv_file = None
    for p in csv_paths:
        if os.path.exists(p):
            csv_file = p
            break
    if not csv_file:
        print(f"  WARNING: Re-EDL seed {s} - CSV not found")
        continue
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    id_acc = float(rows[0]["id_accuracy"]) * 100.0
    result = {"acc": id_acc}
    for row in rows:
        ood_name = row["ood_dataset_name"]
        if ood_name in ("SVHN", "DTD", "Places365"):
            auroc = float(row.get("ood_alpha0_auroc", 0))
            aupr = float(row.get("ood_alpha0_apr", 0))
            result[ood_name] = {
                "fpr95": 100.0 - auroc,
                "aupr": aupr,
                "auroc": auroc,
            }
    out_path = f"{RESULTS_PRE}/reedl_s{s}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {out_path}")
PYEOF
echo "[$(date)] [Re-EDL] Done."

echo "[$(date)] ===== Generating pre-graft table ====="
cd /supcon3/shixisheng/xjx/research/project
SEEDS=1000,1001,1002,1003,1004 $PY make_table.py --results_dir $RESULTS_PRE > /tmp/opencode/pre_table.log 2>&1
SEEDS=1000,1001,1002,1003,1004 $PY render_table_png.py --results_dir $RESULTS_PRE > /tmp/opencode/pre_png.log 2>&1
echo "[$(date)] Pre-graft table: $RESULTS_PRE/table.png"
echo "[$(date)] ===== Pre-graft evaluation complete ====="
