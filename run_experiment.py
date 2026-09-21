"""统一实验：方法 × 种子 循环训练 + OOD 评估，结果存 results/。

SAVE_SCORES=1 时额外导出每个方法/种子的概率矩阵（ID + 各 OOD），
供 Deep Ensembles 集成与后续 post-hoc 方法复用（存 <results>/scores/）。
"""
import argparse
import json
import os
import random
import numpy as np
import torch

from config import (SEEDS, METHODS, OOD_NAMES, RESULTS_DIR, BACKBONE, IMAGE_SIZE, NUM_WORKERS)
from backbone import build_model
from data import get_id_dataset, get_ood_dataset, get_loader
from trainer import train, predict_uncertainty, softmax_probs
from evaluate import compute_ood_metrics, compute_acc
from torch.utils.data import DataLoader

SAVE_SCORES = os.environ.get("SAVE_SCORES", "0") == "1"


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def score_path(method, seed):
    return os.path.join(RESULTS_DIR, "scores", f"{method}_s{seed}_scores.npz")


def result_exists(method, seed):
    """json 存在即算完成；但 SAVE_SCORES 下 softmax 还要求分数文件存在
    （Deep Ensembles 需要全部 5 个种子的概率矩阵）。"""
    if not os.path.exists(os.path.join(RESULTS_DIR, f"{method}_s{seed}.json")):
        return False
    if SAVE_SCORES:
        return os.path.exists(score_path(method, seed))
    return True


def run_method_seed(method, seed, device):
    if result_exists(method, seed):
        print(f"skip {method} s{seed} (exists)", flush=True)
        return None
    set_seed(seed)
    # device 已经是正确的 cuda:X (考虑了 CUDA_VISIBLE_DEVICES)
    model = build_model(method, num_classes=10).to(device)

    train_loader = DataLoader(get_id_dataset(train=True), batch_size=128,
                              shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(get_id_dataset(train=False), batch_size=128,
                             shuffle=False, num_workers=NUM_WORKERS)

    model = train(model, method, train_loader, device, seed)

    # 收集 ID 测试集 preds/unc
    id_preds, id_unc = predict_uncertainty(model, method, test_loader, device)
    id_labels = torch.tensor([y for _, y in test_loader.dataset])
    acc = compute_acc(id_preds.numpy(), id_labels.numpy())
    id_unc = id_unc.numpy()

    result = {"method": method, "seed": seed, "acc": acc}
    ood_uncs = {}
    for ood_name in OOD_NAMES:
        ood_loader = DataLoader(get_ood_dataset(ood_name), batch_size=128,
                                shuffle=False, num_workers=NUM_WORKERS)
        _, ood_unc = predict_uncertainty(model, method, ood_loader, device)
        ood_uncs[ood_name] = ood_unc.numpy()
        fpr95, auroc, aupr = compute_ood_metrics(id_unc, ood_unc.numpy())
        result[ood_name] = {"fpr95": fpr95, "auroc": auroc, "aupr": aupr}
        print(f"  {method} s{seed} {ood_name}: FPR95 {fpr95:.2f} AUROC {auroc:.2f} AUPR {aupr:.2f}",
              flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{method}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {path}", flush=True)

    # 概率矩阵导出（DE 集成 / post-hoc 复用）
    if SAVE_SCORES:
        sp = score_path(method, seed)
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        dump = {"id_probs": softmax_probs(model, test_loader, device),
                "id_labels": id_labels.numpy()}
        for ood_name in OOD_NAMES:
            ood_loader = DataLoader(get_ood_dataset(ood_name), batch_size=128,
                                    shuffle=False, num_workers=NUM_WORKERS)
            dump[f"ood_{ood_name}"] = softmax_probs(model, ood_loader, device)
        np.savez_compressed(sp, **dump)
        print(f"saved {sp}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", nargs="*", default=[str(s) for s in SEEDS])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    for m in args.methods:
        for s in [int(x) for x in args.seeds]:
            run_method_seed(m, s, device)
