"""阶段三：汇总 mean±std，加粗最优，输出 csv + md。"""
import glob
import json
import os

import numpy as np

from config import SEEDS, METHODS, OOD_NAMES, RESULTS_DIR

OOD_METRICS = ["fpr95", "aupr", "auroc"]
ORDER = METHODS


def load(method, seed, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR
    p = os.path.join(results_dir, f"{method}_s{seed}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main(results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR


def mean_std(vals):
    arr = np.array(vals, dtype=float)
    return f"{arr.mean():.2f} ± {arr.std():.2f}"


def build_table(ods=None, results_dir=None):
    if ods is None:
        ods = OOD_NAMES
    cols = ["method", "acc"]
    for o in ods:
        for m in OOD_METRICS:
            cols.append(f"{o}_{m}")
    for m in OOD_METRICS:
        cols.append(f"avg_{m}")

    # collect raw numbers（某方法任一种子缺失则整行跳过，避免崩溃）
    table = {c: [] for c in cols}
    for method in ORDER:
        ds = [load(method, s, results_dir) for s in SEEDS]
        if any(d is None for d in ds):
            continue
        table["method"].append(method)
        table["acc"].append([d["acc"] for d in ds])
        for o in ods:
            for m in OOD_METRICS:
                table[f"{o}_{m}"].append([d[o][m] for d in ds])
        for m in OOD_METRICS:
            vals = []
            for o in ods:
                vals += [d[o][m] for d in ds]
            table[f"avg_{m}"].append(vals)

    # mean + std strings
    out = {}
    for c in cols:
        if c == "method":
            out[c] = table[c]
        else:
            out[c] = [mean_std(v) for v in table[c]]

    # best per numeric column (fpr95 min, else max)
    best = {}
    for c in cols:
        if c == "method":
            continue
        is_fpr = "fpr95" in c
        vals = [float(v.split(" ± ")[0]) for v in out[c]]
        best[c] = vals.index(min(vals)) if is_fpr else vals.index(max(vals))

    return out, best, cols


def render_md(out, best, cols, path):
    lines = []
    header = "| Method | " + " | ".join(cols[1:]) + " |"
    sep = "|" + "---|" * (len(cols))
    lines.append(header)
    lines.append(sep)
    for i, m in enumerate(out["method"]):
        cells = []
        for c in cols:
            if c == "method":
                cells.append(m)
            else:
                v = out[c][i]
                if best[c] == i:
                    v = f"**{v}**"
                cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return lines


def render_csv(out, best, cols, path):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, m in enumerate(out["method"]):
            row = [m]
            for c in cols[1:]:
                row.append(out[c][i])
            w.writerow(row)


def render_table(out, best, cols, ods=None):
    """论文风格：每 OOD 一组列 + Average 列。"""
    if ods is None:
        ods = OOD_NAMES
    metric_names = {"fpr95": "FPR95↓", "aupr": "AUPR↑", "auroc": "AUROC↑"}
    header_cols = ["Method", "ID Acc↑"]
    for o in ods:
        for m in OOD_METRICS:
            header_cols.append(f"{o} {metric_names[m]}")
    for m in OOD_METRICS:
        header_cols.append(f"Avg {metric_names[m]}")

    # remap from flat cols to paper cols
    def flat_val(method, o, met):
        return out[f"{o}_{met}"][ORDER.index(method)]

    def avg_val(method, met):
        return out[f"avg_{met}"][ORDER.index(method)]

    paper_best = {}
    for c in header_cols[1:]:
        if "ID Acc" in c:
            continue
        if "FPR95" in c:
            vals = []
            for method in ORDER:
                if c.startswith("Avg"):
                    vals.append(float(avg_val(method, "fpr95").split(" ± ")[0]))
                else:
                    o, met = c.split(" ")[0], "fpr95"
                    vals.append(float(flat_val(method, o, met).split(" ± ")[0]))
            paper_best[c] = vals.index(min(vals))
        else:
            vals = []
            for method in ORDER:
                if c.startswith("Avg"):
                    met = "aupr" if "AUPR" in c else "auroc"
                    vals.append(float(avg_val(method, met).split(" ± ")[0]))
                else:
                    o = c.split(" ")[0]
                    met = "aupr" if "AUPR" in c else "auroc"
                    vals.append(float(flat_val(method, o, met).split(" ± ")[0]))
            paper_best[c] = vals.index(max(vals))

    lines = ["| " + " | ".join(header_cols) + " |",
             "|" + "---|" * len(header_cols)]
    for i, method in enumerate(ORDER):
        row = [method, out["acc"][i]]
        for c in header_cols[1:]:
            if c.startswith("ID Acc"):
                continue
            if c.startswith("Avg"):
                met = "fpr95" if "FPR95" in c else ("aupr" if "AUPR" in c else "auroc")
                v = avg_val(method, met)
            else:
                o, met = c.split(" ")[0], ("fpr95" if "FPR95" in c else
                                           ("aupr" if "AUPR" in c else "auroc"))
                v = flat_val(method, o, met)
            if paper_best[c] == i:
                v = f"**{v}**"
            row.append(v)
        lines.append("| " + " | ".join(row) + " |")
    return lines


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None, help="Custom results directory")
    args = parser.parse_args()

    results_dir = args.results_dir
    out, best, cols = build_table(results_dir=results_dir)
    lines = render_md(out, best, cols, os.path.join(results_dir or RESULTS_DIR, "table.md"))
    render_csv(out, best, cols, os.path.join(results_dir or RESULTS_DIR, "table.csv"))
    lines = render_md(out, best, cols, os.path.join(RESULTS_DIR, "table.md"))
    render_csv(out, best, cols, os.path.join(RESULTS_DIR, "table.csv"))
    print("\n".join(lines))
    print()
    print("--- 论文风格 ---")
    print("\n".join(render_table(out, best, cols)))
