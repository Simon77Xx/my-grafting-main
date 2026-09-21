"""复数统一实验：面板 × 方法 × 种子 训练 + 跨域 OOD 评估。

与实数 run_experiment.py 协议一致，仅以下不同：
- ID = 指定面板（6 数据集轮流），num_classes 随面板自适应；
- OOD = 其他两域的 4 个数据集（各自 test latent）；
- RSNN focal 集用面板专属文件（rsnn_focal_plural/）；
- 结果存 results_plural[_g_<spec>]/panel_<key>/<method>_s<seed>.json。

SAVE_SCORES=1 时额外导出概率矩阵（ID + 各 OOD）到 panel 目录下 scores/，
供 Deep Ensembles 集成与 post-hoc 方法复用；softmax 的跳过条件同时要求
分数文件存在（DE 需要全部种子的概率矩阵）。

用法（GRAFT 环境变量语义与实数一致：''=嫁接前，swa50/hyena50/mamba2_50=嫁接后）:
  GRAFT=mamba2_50 python run_experiment_plural.py --panels OCTDL --methods rsnn \
      --seeds 1000 1001 --gpu 2
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import config_plural as C
from backbone import build_model
from latent_data_plural import get_panel_dataset, get_panel_ood_datasets
from trainer import train, predict_uncertainty, softmax_probs
from evaluate import compute_ood_metrics, compute_acc

SAVE_SCORES = os.environ.get("SAVE_SCORES", "0") == "1"


def score_path(panel, method, seed):
    return os.path.join(C.panel_dir(panel), "scores", f"{method}_s{seed}_scores.npz")


def result_exists(panel, method, seed):
    if not os.path.exists(os.path.join(C.panel_dir(panel), f"{method}_s{seed}.json")):
        return False
    if SAVE_SCORES:
        return os.path.exists(score_path(panel, method, seed))
    return True


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync_config():
    """把复数协议同步进 config（trainer/backbone 读的是 config 模块）。"""
    config.GRAFT = os.environ.get("GRAFT", "").strip()
    config.EPOCHS = C.EPOCHS
    config.LR = C.LR
    config.WEIGHT_DECAY = C.WEIGHT_DECAY
    config.LR_SCHEDULE = C.LR_SCHEDULE
    config.GRAD_ACCUM = C.GRAD_ACCUM
    config.USE_AMP = C.USE_AMP
    config.AMP_DTYPE = C.AMP_DTYPE
    config.NUM_WORKERS = C.NUM_WORKERS
    config.BACKBONE = "DiT-XL/2"
    config.IMAGE_SIZE = 32


def run_method_seed(panel, method, seed, device):
    set_seed(seed)
    sync_config()
    config.RSNN_NEW_CLASSES = C.focal_path(panel)  # RSNN 头用面板专属 focal 集
    nc = C.PANELS[panel]["n"]

    model = build_model(method, num_classes=nc).to(device)

    train_loader = DataLoader(get_panel_dataset(panel, train=True), batch_size=C.BATCH_SIZE,
                              shuffle=True, num_workers=C.NUM_WORKERS)
    test_loader = DataLoader(get_panel_dataset(panel, train=False), batch_size=C.BATCH_SIZE,
                             shuffle=False, num_workers=C.NUM_WORKERS)

    model = train(model, method, train_loader, device, seed)

    id_preds, id_unc = predict_uncertainty(model, method, test_loader, device)
    id_labels = torch.tensor([y for _, y in test_loader.dataset])
    acc = compute_acc(id_preds.numpy(), id_labels.numpy())
    id_unc = id_unc.numpy()

    result = {"method": method, "seed": seed, "panel": panel, "acc": acc}
    ood_uncs = {}
    for ood_key, ood_ds in get_panel_ood_datasets(panel).items():
        ood_loader = DataLoader(ood_ds, batch_size=C.BATCH_SIZE,
                                shuffle=False, num_workers=C.NUM_WORKERS)
        _, ood_unc = predict_uncertainty(model, method, ood_loader, device)
        ood_uncs[ood_key] = ood_unc.numpy()
        fpr95, auroc, aupr = compute_ood_metrics(id_unc, ood_unc.numpy())
        result[ood_key] = {"fpr95": fpr95, "auroc": auroc, "aupr": aupr}
        print(f"  {panel} {method} s{seed} {ood_key}: "
              f"FPR95 {fpr95:.2f} AUROC {auroc:.2f} AUPR {aupr:.2f}", flush=True)

    out_dir = C.panel_dir(panel)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{method}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {path}", flush=True)

    # 概率矩阵导出（DE 集成 / post-hoc 复用）
    if SAVE_SCORES:
        sp = score_path(panel, method, seed)
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        dump = {"id_probs": softmax_probs(model, test_loader, device),
                "id_labels": id_labels.numpy()}
        for ood_key, ood_ds in get_panel_ood_datasets(panel).items():
            ood_loader = DataLoader(ood_ds, batch_size=C.BATCH_SIZE,
                                    shuffle=False, num_workers=C.NUM_WORKERS)
            dump[f"ood_{ood_key}"] = softmax_probs(model, ood_loader, device)
        np.savez_compressed(sp, **dump)
        print(f"saved {sp}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=C.PANEL_ORDER)
    ap.add_argument("--methods", nargs="*", default=C.METHODS)
    ap.add_argument("--seeds", nargs="*", default=[str(s) for s in C.SEEDS])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    for panel in args.panels:
        for m in args.methods:
            for s in [int(x) for x in args.seeds]:
                if result_exists(panel, m, s):
                    print(f"skip {panel} {m} s{s} (exists)", flush=True)
                    continue
                run_method_seed(panel, m, s, device)
