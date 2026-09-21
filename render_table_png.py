"""渲染论文风格 OOD 对比表 PNG（仿 HEDL NeurIPS 2024 主表版式）。

版式对齐 b468b5021ddd0bf750180e6c09066116_720.png：
- 双层表头：OOD Datasets 跨 SVHN/Textures 分组（各 FPR95↓/AUPR↑/AUROC↑）+ Average + ID data Acc↑
- CIFAR-10 灰色条带
- 每格 5 种子 mean ± std；每列最优加粗（暗红）
- 输出 results/table.png

用法: python render_table_png.py [--ignore-missing]   # 后者允许缺方法时先出预览
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, METHODS, SEEDS, OOD_NAMES, GRAFT

# Allow SEEDS override from env (e.g., 5-seed mode)
_env_seeds = os.environ.get("SEEDS")
if _env_seeds:
    SEEDS = [int(s) for s in _env_seeds.split(",")]

DISPLAY = {"softmax": "Softmax", "reedl": "Re-EDL", "hedl": "HEDL", "rsnn": "RS-NN"}
if GRAFT:
    # 嫁接后行名加 -G 后缀以区分嫁接前（如 HEDL-G）
    DISPLAY = {"softmax": "Softmax-G", "reedl": "Re-EDL-G",
               "hedl": "HEDL-G", "rsnn": "RS-NN-G"}
OOD_TITLE = {"SVHN": "SVHN", "DTD": "Textures", "Places365": "Places365"}
METRICS = ["fpr95", "aupr", "auroc"]
MNAMES = {"fpr95": "FPR95\u2193", "aupr": "AUPR\u2191", "auroc": "AUROC\u2191"}
BEST_COLOR = "#c00000"


def load(method, seed, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR
    with open(os.path.join(results_dir, f"{method}_s{seed}.json")) as f:
        return json.load(f)


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def collect(methods, results_dir=None):
    """vals[m][colkey] = (mean, std)；colkey ∈ {('acc',), (ood, met), ('avg', met)}。"""
    vals = {}
    for m in methods:
        d = {s: load(m, s, results_dir) for s in SEEDS}
        v = {("acc",): mean_std([d[s]["acc"] for s in SEEDS])}
        for o in OOD_NAMES:
            for met in METRICS:
                v[(o, met)] = mean_std([d[s][o][met] for s in SEEDS])
        for met in METRICS:
            v[("avg", met)] = mean_std(
                [d[s][o][met] for s in SEEDS for o in OOD_NAMES])
        vals[m] = v
    return vals


def render(vals, out_path):
    methods = [m for m in METHODS if m in vals]
    n_metric_cols = len(OOD_NAMES) * 3 + 3 + 1        # 各 OOD 组3 + Average 3 + ID Acc 1
    col_w = [0.14] + [0.086] * n_metric_cols          # + Method 0.14 = 1.0
    fig_w = 13.5 if len(OOD_NAMES) <= 2 else 10.0 + 1.6 * len(OOD_NAMES)
    fig, ax = plt.subplots(figsize=(fig_w, 3.4), dpi=220)

    # ---- 列 x 坐标 ----
    xs = [0.0]
    for w in col_w:
        xs.append(xs[-1] + w)

    # ---- 行 y 坐标（自上而下）----
    rows = [1.00, 0.90, 0.80]                        # 表头两行 + CIFAR-10 条带
    rh = 0.085
    for _ in methods:
        rows.append(rows[-1] - rh)
    ax.set_xlim(0, 1)
    ax.set_ylim(rows[-1] - 0.02, 1.02)
    ax.axis("off")

    def cell_text(x0, x1, yc, s, bold=False, color="black", fontsize=8.5):
        ax.text((x0 + x1) / 2, yc, s, ha="center", va="center",
                fontsize=fontsize, color=color,
                fontweight="bold" if bold else "normal", family="DejaVu Sans")

    def hline(y, x0=0.0, x1=1.0, lw=0.8, color="black"):
        ax.plot([x0, x1], [y, y], color=color, lw=lw)

    y_hdr1_mid, y_hdr2_mid = 0.95, 0.85

    # ---- 双层表头 ----
    cell_text(xs[0], xs[1], (y_hdr1_mid + y_hdr2_mid) / 2, "Method", bold=True, fontsize=9.5)
    # OOD Datasets 跨 SVHN/DTD 分组
    ood_x0, ood_x1 = xs[1], xs[1 + 3 * len(OOD_NAMES)]
    cell_text(ood_x0, ood_x1, y_hdr1_mid, "OOD Datasets", bold=True, fontsize=9.5)
    # Average 组
    avg_x0, avg_x1 = ood_x1, ood_x1 + 3 * col_w[1]
    cell_text(avg_x0, avg_x1, y_hdr1_mid, "Average", bold=True, fontsize=9.5)
    # ID data（跨两行）
    cell_text(xs[-2], xs[-1], (y_hdr1_mid + y_hdr2_mid) / 2, "ID data", bold=True, fontsize=9.5)
    # 第二行：组名 + 指标名
    for gi, o in enumerate(OOD_NAMES):
        x0 = xs[1 + 3 * gi]
        cell_text(x0, xs[4 + 3 * gi], y_hdr2_mid, OOD_TITLE[o], bold=True, fontsize=9)
    cell_text(avg_x0, avg_x1, y_hdr2_mid, "Average", bold=True, fontsize=9)
    cell_text(xs[-2], xs[-1], y_hdr2_mid + 0.028, "Acc\u2191", bold=True, fontsize=8.5)
    k = 1
    for o in OOD_NAMES + ["avg"]:
        for met in METRICS:
            cell_text(xs[k], xs[k + 1], y_hdr2_mid - 0.026, MNAMES[met], fontsize=8)
            k += 1
    hline(rows[2])

    # ---- CIFAR-10 条带（位于表头之下、方法行之上）----
    ax.add_patch(plt.Rectangle((0, rows[2] - rh * 0.9), 1, rh * 0.9,
                               facecolor="#d9d9d9", edgecolor="none"))
    cell_text(0, 1, rows[2] - rh * 0.45, "CIFAR-10", bold=True, fontsize=9.5)

    # ---- 方法行 ----
    # 逐列最优（fpr95 取小，其余取大；含 ID Acc）
    colkeys = []
    for o in OOD_NAMES:
        colkeys += [(o, met) for met in METRICS]
    colkeys += [("avg", met) for met in METRICS]
    colkeys += [("acc",)]
    best_idx = {}
    for key in colkeys:
        means = [vals[m][key][0] for m in methods]
        best_idx[key] = int(np.argmin(means)) if key[-1] == "fpr95" else int(np.argmax(means))

    for ri, m in enumerate(methods):
        y = rows[3 + ri]
        yc = y - rh / 2
        cell_text(xs[0], xs[1], yc, DISPLAY.get(m, m), fontsize=9)
        k = 1
        for key in colkeys:
            mean, std = vals[m][key]
            s = f"{mean:.2f} \u00b1 {std:.2f}"
            is_best = (best_idx[key] == ri)
            cell_text(xs[k], xs[k + 1], yc, s, bold=is_best,
                      color=BEST_COLOR if is_best else "black", fontsize=8.2)
            k += 1
        hline(y, lw=0.5, color="#bbbbbb")
    hline(rows[-1])

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"written {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ignore-missing", action="store_true",
                    help="缺某方法 5 种子结果时跳过该方法（用于中途预览）")
    ap.add_argument("--results_dir", type=str, default=None,
                    help="自定义结果目录（如 results_pre）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results_dir = args.results_dir
    # Default output path
    if args.out:
        out_path = args.out
    elif results_dir:
        out_path = os.path.join(results_dir, "table.png")
    else:
        out_path = os.path.join(RESULTS_DIR, "table.png")

    methods = []
    for m in METHODS:
        try:
            for s in SEEDS:
                load(m, s, results_dir)
            methods.append(m)
        except FileNotFoundError:
            if args.ignore_missing:
                print(f"[skip] {m}: 结果不全")
            else:
                raise
    vals = collect(methods, results_dir)
    render(vals, out_path)


if __name__ == "__main__":
    main()