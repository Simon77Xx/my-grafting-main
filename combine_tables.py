"""复数/实数 前后对比大表：单面板渲染 + 两张最终大表（附表/主表）。

行结构（每面板 16 行，嫁接前 4 + 嫁接后 3 变体 × 4）：
  嫁接前:    Softmax / Re-EDL / HEDL / RS-NN          （原始 DiT-XL/2）
  嫁接后:    Softmax-G(SWA50/Hyena50/Mamba2) 等 × 4    （grafted DiT）

列结构：Method | ID Acc↑ | 每个 OOD (FPR95↓/AUPR↑/AUROC↑) | Average (同 3)
最优加粗（暗红），面板条带灰底。输出 png + csv + md。

用法:
  python combine_tables.py --real            # 附表（实数 CIFAR-10）
  python combine_tables.py --plural-panel OCTDL   # 单面板预览
  python combine_tables.py --plural          # 主表（9 面板大表）
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_plural import PANELS, PANEL_ORDER, ood_panels, DISPLAY_PANEL, SEEDS

ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS = ["fpr95", "aupr", "auroc"]
MNAMES = {"fpr95": "FPR95\u2193", "aupr": "AUPR\u2191", "auroc": "AUROC\u2191"}
BEST_COLOR = "#c00000"
OUT_DIR = os.path.join(ROOT, "final_tables")

METHODS = ["reedl", "hedl", "rsnn", "iedl", "duq", "mcdropout", "deepens"]
DISPLAY = {"reedl": "Re-EDL", "hedl": "HEDL", "rsnn": "RS-NN",
           "iedl": "I-EDL", "duq": "DUQ", "mcdropout": "MC-Dropout",
           "deepens": "Deep Ensembles"}
DEEP_ENS = "deepens"   # 单文件 deepens.json（5 成员集成，单值无 std）
GRAFT_GROUPS = [
    ("嫁接前", os.path.join(ROOT, "results"), "results"),            # 实数用
    ("嫁接后 SWA-50", os.path.join(ROOT, "results_g_swa50"), "swa50"),
    ("嫁接后 Hyena-50", os.path.join(ROOT, "results_g_hyena50"), "hyena50"),
    ("嫁接后 Mamba-2", os.path.join(ROOT, "results_g_mamba2_50"), "mamba2_50"),
]
GRAFT_GROUPS_PLURAL = [
    ("嫁接前", os.path.join(ROOT, "results_plural"), None),
    ("嫁接后 SWA-50", os.path.join(ROOT, "results_plural_g_swa50"), "swa50"),
    ("嫁接后 Hyena-50", os.path.join(ROOT, "results_plural_g_hyena50"), "hyena50"),
    ("嫁接后 Mamba-2", os.path.join(ROOT, "results_plural_g_mamba2_50"), "mamba2_50"),
]
SUFFIX = {"嫁接前": "", "嫁接后 SWA-50": " (SWA-G)",
          "嫁接后 Hyena-50": " (Hy-G)", "嫁接后 Mamba-2": " (Mamba-G)"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_group(groot, gspec, panel, oods, seeds):
    """一个嫁接组 × 一个面板 → {method: {key: (mean,std)}}；缺文件返回 None。

    key ∈ {('acc',), (ood, met), ('avg', met)}
    """
    base = os.path.join(groot, f"panel_{panel}") if gspec else groot
    vals = {}
    for m in METHODS:
        try:
            ds = [load_json(os.path.join(base, f"{m}_s{s}.json")) for s in seeds]
        except FileNotFoundError:
            return None

        def ms(key_fn):
            xs = [key_fn(d) for d in ds]
            return float(np.mean(xs)), float(np.std(xs))

        v = {("acc",): ms(lambda d: d["acc"])}
        for o in oods:
            for met in METRICS:
                v[(o, met)] = ms(lambda d, o=o, met=met: d[o][met])
        for met in METRICS:
            v[("avg", met)] = ms(
                lambda d, met=met: float(np.mean([d[o][met] for o in oods])))
        vals[m] = v
    return vals


def build_rows(panels, seeds):
    """收集所有面板所有组。

    panels: [{"key", "display", "oods", "nested"(可选), "groups": [(组名, 根目录)]}]
      nested=True 时 json 在 <根目录>/panel_<key>/ 下（复数）；
      否则直接在根目录下（实数）。
    返回: [{key, display, oods, rows: [(label, gname, vals|None)]}]
    """
    data = []
    for p in panels:
        rows = []
        for gname, groot in p["groups"]:
            base = (os.path.join(groot, f"panel_{p['key']}")
                    if p.get("nested") else groot)
            for m in METHODS:
                label = DISPLAY[m] + SUFFIX[gname]
                try:
                    if m == DEEP_ENS:
                        # Deep Ensembles：5 成员集成，单值无 std
                        ds = [load_json(os.path.join(base, "deepens.json"))]
                        single = True
                    else:
                        ds = [load_json(os.path.join(base, f"{m}_s{s}.json"))
                              for s in seeds]
                        single = False
                except FileNotFoundError:
                    rows.append((label, gname, None))
                    continue

                def ms(fn, ds=ds, single=single):
                    xs = [fn(d) for d in ds]
                    return (float(np.mean(xs)),
                            None if single else float(np.std(xs)))

                v = {("acc",): ms(lambda d: d["acc"])}
                for o in p["oods"]:
                    for met in METRICS:
                        v[(o, met)] = ms(lambda d, o=o, met=met: d[o][met])
                for met in METRICS:
                    # 与 render_table_png.py 口径一致：对 (种子 × OOD) 池化
                    pooled = [d[o][met] for d in ds for o in p["oods"]]
                    v[("avg", met)] = (float(np.mean(pooled)),
                                       float(np.std(pooled)))
                rows.append((label, gname, v))
        data.append({"key": p["key"], "display": p["display"],
                     "oods": p["oods"], "rows": rows})
    return data


def render_png(data, out_path, title=None):
    """纵向长表：双层表头 + 每面板灰色条带 + 16 行（缺数据的行画 '—'）。

    最优加粗：仅在该面板内「数据齐全的行」之间比较（FPR95 取小，其余取大）。
    """
    oods0 = data[0]["oods"]
    n_metric = len(oods0) * 3 + 3
    w_method, w_acc, w_m = 0.10, 0.06, (1 - 0.10 - 0.06) / n_metric
    xs = [0.0, w_method, w_method + w_acc]
    for _ in range(n_metric):
        xs.append(xs[-1] + w_m)

    # 行布局：每面板 = 条带 + 4 组（组条 + N 方法行）
    RH_ROW, RH_BAND, RH_GBAND = 0.032, 0.042, 0.024
    gsize = len(METHODS)
    GBAND_EN = {
        "嫁接前": "Pre-graft (original DiT-XL/2)",
        "嫁接后 SWA-50": "Post-graft SWA-50 (SWA-G)",
        "嫁接后 Hyena-50": "Post-graft Hyena-50 (Hy-G)",
        "嫁接后 Mamba-2": "Post-graft Mamba-2 (Mamba-G)",
    }
    panels_layout = []   # (band_y, [(gband_y, gname, [row_y x gsize]), ...])
    y = 0.0
    for p in data:
        band_y = y
        y -= RH_BAND
        groups = []
        for i in range(0, len(p["rows"]), gsize):
            g = p["rows"][i:i + gsize]
            gy = y
            y -= RH_GBAND
            rys = []
            for _ in g:
                rys.append(y)
                y -= RH_ROW
            groups.append((gy, g[0][1], rys))
        panels_layout.append((band_y, groups))
    bottom = y
    fig_h = max(3.0, (0.06 - bottom) * 16)
    fig, ax = plt.subplots(figsize=(16.5, fig_h), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom - 0.01, 0.06)
    ax.axis("off")

    def T(x0, x1, yc, s, bold=False, color="black", fs=7.2):
        ax.text((x0 + x1) / 2, yc, s, ha="center", va="center", fontsize=fs,
                color=color, fontweight="bold" if bold else "normal",
                family="DejaVu Sans")

    def H(yv, x0=0.0, x1=1.0, lw=0.7, color="black"):
        ax.plot([x0, x1], [yv, yv], color=color, lw=lw)

    # 双层表头（位于 y=0.045 / 0.015）
    y1, y2 = 0.045, 0.015
    T(xs[0], xs[1], (y1 + y2) / 2, "Method", bold=True, fs=8.5)
    T(xs[1], xs[2], (y1 + y2) / 2, "ID Acc\u2191", bold=True, fs=8.5)
    ood_x0, ood_x1 = xs[2], xs[2 + 3 * len(oods0)]
    T(ood_x0, ood_x1, y1, "OOD Datasets", bold=True, fs=8.5)
    T(ood_x1, xs[-1], y1, "Average", bold=True, fs=8.5)
    for gi, o in enumerate(oods0):
        T(xs[2 + 3 * gi], xs[5 + 3 * gi], y2, o, bold=True, fs=7.5)
    T(ood_x1, xs[-1], y2, "Avg", bold=True, fs=7.5)
    k = 2
    for o in oods0 + ["avg"]:
        for met in METRICS:
            T(xs[k], xs[k + 1], y2 - 0.020, MNAMES[met], fs=6.2)
            k += 1
    H(0.001)          # 表头与面板分界

    for pi, p in enumerate(data):
        band_y, groups = panels_layout[pi]
        # 面板条带
        ax.add_patch(plt.Rectangle((0, band_y - RH_BAND), 1, RH_BAND,
                                   facecolor="#d9d9d9", edgecolor="none"))
        T(0, 1, band_y - RH_BAND / 2,
          f"ID = {p['display']}", bold=True, fs=8.5)
        # 逐列最优（仅比较数据齐全的行）
        colkeys = [("acc",)] + [(o, met) for o in p["oods"] for met in METRICS] \
            + [("avg", met) for met in METRICS]
        best = {}
        for key in colkeys:
            cand = [(i, v[key][0]) for i, (_, _, v) in enumerate(p["rows"])
                    if v is not None]
            if len(cand) >= 2:
                best[key] = (min(cand, key=lambda t: t[1]) if key[-1] == "fpr95"
                             else max(cand, key=lambda t: t[1]))[0]
        # 组 × 方法：每组先画组标题条，再画 4 行
        gi0 = 0
        for gy, gname, rys in groups:
            fc = "#e8eef8" if gname == "嫁接前" else "#fbe9e9"
            ax.add_patch(plt.Rectangle((0, gy - RH_GBAND), 1, RH_GBAND,
                                       facecolor=fc, edgecolor="none"))
            T(0, 1, gy - RH_GBAND / 2, GBAND_EN.get(gname, gname),
              bold=True, fs=7.5, color="#444444")
            g_rows = p["rows"][gi0:gi0 + len(rys)]
            for ri, (label, _, v) in enumerate(g_rows):
                row_idx = gi0 + ri
                yr = rys[ri]
                yc = yr - RH_ROW / 2
                H(yr, lw=0.4, color="#bbbbbb")
                T(xs[0], xs[1], yc, label, fs=7.2,
                  bold=(gname != "嫁接前"),
                  color="#333333" if gname != "嫁接前" else "black")
                if v is None:
                    T(xs[1], xs[-1], yc, "\u2014", fs=7.2, color="#999999")
                else:
                    k = 1
                    for key in colkeys:
                        mean, std = v[key]
                        s = (f"{mean:.2f}" if std is None
                             else f"{mean:.2f} \u00b1 {std:.2f}")
                        is_best = (best.get(key) == row_idx)
                        T(xs[k], xs[k + 1], yc, s, bold=is_best,
                          color=BEST_COLOR if is_best else "black", fs=6.4)
                        k += 1
            gi0 += len(rys)
    if title:
        ax.set_title(title, fontsize=11, pad=10)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"written {out_path}")


def render_csv(data, out_csv):
    """csv: Panel,Group,Method,acc,每 OOD 3 指标,Avg 3。"""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        oods0 = data[0]["oods"]
        head = ["Panel", "Group", "Method", "ID Acc"]
        head += [f"{o}_{met}" for o in oods0 for met in METRICS]
        head += [f"Avg_{met}" for met in METRICS]
        w.writerow(head)
        for p in data:
            for label, gname, v in p["rows"]:
                row = [p["display"], gname, label]
                if v is None:
                    row += [""] * (3 * len(p["oods"]) + 4)
                else:
                    def fmt(pair):
                        m, s = pair
                        return f"{m:.2f}" if s is None else f"{m:.2f}±{s:.2f}"
                    row.append(fmt(v[("acc",)]))
                    for o in p["oods"]:
                        for met in METRICS:
                            row.append(fmt(v[(o, met)]))
                    for met in METRICS:
                        row.append(fmt(v[("avg", met)]))
                w.writerow(row)
    print(f"written {out_csv}")


# ---------- 附表（实数 CIFAR-10）----------
def real_panels():
    """实数：json 直接在 results*/ 目录下（nested=False）。"""
    oods = ["SVHN", "DTD", "Places365"]
    return [{"key": "CIFAR-10", "display": "CIFAR-10", "oods": oods,
             "groups": [("嫁接前", os.path.join(ROOT, "results")),
                        ("嫁接后 SWA-50", os.path.join(ROOT, "results_g_swa50")),
                        ("嫁接后 Hyena-50", os.path.join(ROOT, "results_g_hyena50")),
                        ("嫁接后 Mamba-2", os.path.join(ROOT, "results_g_mamba2_50"))]}]


# ---------- 主表（复数 9 面板）----------
def plural_panels(only=None):
    """复数：json 在 results_plural*/panel_<key>/ 下（nested=True）。"""
    keys = [only] if only else PANEL_ORDER
    panels = []
    for k in keys:
        panels.append({
            "key": k, "display": DISPLAY_PANEL[k], "oods": ood_panels(k),
            "nested": True,
            "groups": [("嫁接前", os.path.join(ROOT, "results_plural")),
                       ("嫁接后 SWA-50", os.path.join(ROOT, "results_plural_g_swa50")),
                       ("嫁接后 Hyena-50", os.path.join(ROOT, "results_plural_g_hyena50")),
                       ("嫁接后 Mamba-2", os.path.join(ROOT, "results_plural_g_mamba2_50"))],
        })
    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="生成附表（实数）")
    ap.add_argument("--plural", action="store_true", help="生成主表（复数 9 面板）")
    ap.add_argument("--plural-panel", type=str, default=None, help="单面板预览")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.real:
        data = build_rows(real_panels(), seeds)
        base = os.path.join(OUT_DIR, "table_real_combined")
        render_png(data, base + ".png",
                   title="Table S (real data): Pre- vs Post-Grafting on CIFAR-10")
        render_csv(data, base + ".csv")
    if args.plural:
        data = build_rows(plural_panels(), seeds)
        base = os.path.join(OUT_DIR, "table_plural_combined")
        render_png(data, base + ".png",
                   title="Table M (plural data): Pre- vs Post-Grafting across 6 Panels")
        render_csv(data, base + ".csv")
    if args.plural_panel:
        k = args.plural_panel
        data = build_rows(plural_panels(only=k), seeds)
        base = os.path.join(OUT_DIR, f"panel_{k}")
        render_png(data, base + ".png",
                   title=f"面板预览：ID={DISPLAY_PANEL[k]}")
        render_csv(data, base + ".csv")
    if not (args.real or args.plural or args.plural_panel):
        print("nothing to do: use --real / --plural / --plural-panel KEY")


if __name__ == "__main__":
    main()
