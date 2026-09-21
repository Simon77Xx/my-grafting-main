"""生成 results/README.md：结果表 + 逐种子指标 + 训练时间。"""
import json
import os

import numpy as np

from config import RESULTS_DIR, METHODS, SEEDS

METRIC_NAMES = {"fpr95": "FPR95↓", "aupr": "AUPR↑", "auroc": "AUROC↑"}


def mean_std(vals):
    arr = np.array(vals, dtype=float)
    return f"{arr.mean():.2f} ± {arr.std():.2f}"


def load(method, seed):
    with open(os.path.join(RESULTS_DIR, f"{method}_s{seed}.json")) as f:
        return json.load(f)


def build():
    lines = []
    lines.append("# PlanA 复现结果")
    lines.append("")
    lines.append("环境：8×H200 (141GB)；`env_pt` (PyTorch 2.1.2 + cu121)、`env_tf` (TF 2.13.0 + cu11)。")
    lines.append("统一协议：ImageNet 预训练 ResNet18 骨干、CIFAR-10 (ID) / SVHN+DTD (OOD)、种子 [1000..1004]，每格 5 种子 `mean ± std`。")
    lines.append("")
    lines.append("## 统一协议结果表（阶段二/阶段三）")
    lines.append("")
    lines.append("ID Acc（%）、FPR95（%↓）、AUPR/AUROC（%↑），每列最优加粗。")
    lines.append("")

    table_md = os.path.join(RESULTS_DIR, "table.md")
    lines += [l.rstrip() for l in open(table_md)]
    lines.append("结论：统一协议下 Re-EDL 全面最优（Avg FPR95/AUPR/AUROC 均第一），RS-NN 次之，Softmax 基线第三，HEDL 明显落后（ID Acc 86.36、AUROC 82.98）。")
    lines.append("注意：HEDL 与 RS-NN 论文原协议骨干/轮数不同（ResNet18+100 / ResNet50+200），统一协议（ResNet18+60 轮）下 HEDL 收敛不足，属\"统一协议相对对比\"，不代表论文绝对数字。")
    lines.append("")

    lines.append("## 逐种子指标（统一协议）")
    lines.append("")
    for m in METHODS:
        lines.append(f"### {m}")
        lines.append("| seed | acc | SVHN FPR95 | SVHN AUROC | SVHN AUPR | DTD FPR95 | DTD AUROC | DTD AUPR |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in SEEDS:
            d = load(m, s)
            sv, dt = d["SVHN"], d["DTD"]
            lines.append(f"| {s} | {d['acc']:.2f} | {sv['fpr95']:.2f} | {sv['auroc']:.2f} | {sv['aupr']:.2f} | {dt['fpr95']:.2f} | {dt['auroc']:.2f} | {dt['aupr']:.2f} |")
        lines.append("")

    lines.append("## 训练时间")
    lines.append("")
    lines.append("| 方法 | 统一协议(60轮×5种子) | 论文原协议(阶段一) |")
    lines.append("|---|---|---|")
    lines.append("| softmax | ~2.5h | - |")
    lines.append("| reedl | ~2.5h | Re-EDL 阶段一 5种子见 reedl/ 日志 |")
    lines.append("| hedl | ~2.5h | HEDL 阶段一 5种子见 hedl/logs/ |")
    lines.append("| rsnn | ~2.5h | 阶段一C ResNet50@224 两阶段×200轮×5种子，单种子 ~6.2h（GPU 并行） |")
    lines.append("")
    lines.append("## 阶段一独立复现结果")
    lines.append("")
    lines.append("- **HEDL**（论文原协议，5 种子 × SVHN/DTD）：SVHN FPR95 23.93±9.72 / AUROC 95.92±1.51 / AUPR 98.01±0.74；DTD FPR95 34.37±12.67 / AUROC 91.80±3.08 / AUPR 99.56±0.17；平均 AUROC 93.86。ID Acc 用训练集评估（~99%，偏高，出表时说明）。")
    lines.append("- **Re-EDL**（5 种子）：ID Acc 90.13±0.25；SVHN（diff_ent）AUROC 90.27±0.89 / AUPR 87.55±0.73；DTD AUROC 89.68±0.63 / AUPR 97.96±0.13。")
    lines.append("- **RS-NN**（ResNet50@224，两阶段×200 轮，5 种子）：ID Acc 92.5±0.3；SVHN AUROC 0.930±0.005 / AUPR 0.914±0.006；Intel Image AUROC 0.991±0.02 / AUPR 0.917±0.1。")
    lines.append("")

    out = os.path.join(RESULTS_DIR, "README.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"written {out}")


if __name__ == "__main__":
    build()