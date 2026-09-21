"""生成给导师的实验说明文档（report_实验说明.md）。

读取 final_tables/ 两张大表 CSV，自动填入关键数字：各面板最优方法、
嫁接前后 Avg 指标变化方向等。表格未完成时数字段会标注 [待补]。
用法: python gen_report.py [--out project/report_实验说明.md]
"""
import argparse
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "final_tables")

GRAFT_SHORT = {"嫁接前": "pre", "嫁接后 SWA-50": "swa", "嫁接后 Hyena-50": "hyena",
               "嫁接后 Mamba-2": "mamba2"}


def read_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def best_by_panel(rows, metric, higher_better):
    """{panel: (方法, 值)}，仅统计数据齐全的行。"""
    best = {}
    for r in rows:
        v = r.get(metric)
        if not v:
            continue
        val = float(v.split("±")[0])
        cur = best.get(r["Panel"])
        if cur is None or (val > cur[1] if higher_better else val < cur[1]):
            best[r["Panel"]] = (r["Method"], val)
    return best


def avg_metric(rows, col):
    """{panel: {group: mean}}。"""
    out = defaultdict(dict)
    for r in rows:
        if r.get(col):
            out[r["Panel"]][GRAFT_SHORT.get(r["Group"], r["Group"])] = \
                float(r[col].split("±")[0])
    return out


def group_best(rows, col, minimize=True):
    """{panel: {group: (method, 最优值)}}：组内 4 方法取最优。"""
    out = defaultdict(lambda: defaultdict(lambda: (None, None)))
    for r in rows:
        v = r.get(col)
        if not v:
            continue
        val = float(v.split("±")[0])
        g = GRAFT_SHORT.get(r["Group"], r["Group"])
        cur = out[r["Panel"]][g]
        if cur[1] is None or (val < cur[1] if minimize else val > cur[1]):
            out[r["Panel"]][g] = (r["Method"], val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "report_实验说明.md"))
    args = ap.parse_args()

    real = read_csv(os.path.join(OUT_DIR, "table_real_combined.csv"))
    plural = read_csv(os.path.join(OUT_DIR, "table_plural_combined.csv"))

    L = []
    A = L.append

    A("# 实验说明：不确定性头嫁接前后性能对比（实数 / 复数数据）")
    A("")
    A("> 本文档由 gen_report.py 自动生成，数字直接来自 final_tables/ 两张大表 CSV。")
    A("")
    A("## 1. 实验目的")
    A("在**完全公平**的协议下，比较「不确定性估计头」在骨干网络**嫁接前（原始 DiT-XL/2）**"
      "与**嫁接后（SWA-50 / Hyena-50 / Mamba-2 序列算子替换）**的性能差异：")
    A("1. 嫁接（原论文工作）是否保持/改变了不确定性估计能力；")
    A("2. 我们的创新点——把三种 2024/2025 最新不确定性头（HEDL / Re-EDL / RS-NN）与"
      "从其对比清单中选取的四个基线（I-EDL / DUQ / MC-Dropout / Deep Ensembles）"
      "嫁接到 DiT 特征上——在不同数据形态（实数 / 复数内涵）下表现如何。")
    A("")
    A("## 2. 方法与骨干")
    A("- **骨干**：grafting-main（原论文主框架）的 DiT-XL/2，吃 SD-VAE 预编码 latent（4×32×32）。")
    A("- **嫁接前**：原始 DiT（官方权重 DiT-XL-2-256x256.pt），去噪末层替换为不确定性头。")
    A("- **嫁接后**：按原论文机制（ReplacementFactory）把 14 个 MHA 块（奇数层 1..27）替换为")
    A("  SWA / Hyena / Mamba-2，加载官方 stage2 嫁接权重（DiT-XL-2-*-50p.pt）。")
    A("- **七个方法**：Re-EDL（TPAMI 2025）、HEDL（NeurIPS 2024）、RS-NN（ICLR 2025）、"
      "I-EDL（ICML 2023）、DUQ（ICML 2020）、MC-Dropout（ICML 2016）、"
      "Deep Ensembles（NeurIPS 2017），全部接在 DiT 池化特征（1152 维）上。")
    A("")
    A("> **基线选取依据**：原基线 Softmax（2017）已过旧，改从 Re-EDL / HEDL / RS-NN 三篇正式论文的"
      "对比清单中选取**它们已经评估筛选过**的方法作为新基线——I-EDL 直接移植自 Re-EDL 官方仓库的"
      " IEDL 分支，DUQ / MC-Dropout / Deep Ensembles 为三篇论文共同的对比基线（全部为开源实现）。"
      "Softmax 仅作为 Deep Ensembles 的集成成员保留训练，不再作为独立对比行出现。")
    A("")
    A("## 3. 公平性协议（实数/复数完全一致）")
    A("| 协议要素 | 设置 |")
    A("|---|---|")
    A("| 随机种子 | 1000-1004 共 5 个，全部方法/设置一致，报 mean±std（Deep Ensembles 为单值）|")
    A("| 训练超参 | 60 epoch、AdamW、lr 1e-4、wd 1e-4、cosine、batch 128、AMP bf16 |")
    A("| 数据编码 | 同一 SD-VAE（Resize256 bicubic + CenterCrop256 + /0.5 归一化）|")
    A("| 评估口径 | ID Acc + 各 OOD 的 FPR95↓/AUPR↑/AUROC↑ + Average |")
    A("| RS-NN focal 集 | 每数据集固定 seed=123 生成，所有方法/种子共享 |")
    A("| 数据划分 | 自带 train/test 用官方划分；否则按类分层 90/10（固定 seed）|")
    A("")
    A("即：**任何性能差异只能归因于「嫁接与否」与「不确定性头种类」**。")
    A("")
    A("## 4. 实验设计")
    A("### 附表（实数数据）")
    A("- ID = CIFAR-10；OOD = SVHN、DTD、Places365。"
      "7 方法（6 个逐种子训练 + Deep Ensembles）× 5 种子 × 4 设置。")
    A("### 主表（复数内涵数据 data_plural）")
    A("- 从 9 个数据集中选 **6 个**（MRI/OCT/SAR 各 2 个）**逐个轮 ID**（6 面板），"
      "OOD = 其他两域各 2 个、共 4 个数据集（跨域开放集，域差距大，更能区分不确定性质量）。")
    A("- 6 面板 × 7 方法 × 5 种子 × 4 设置。")
    A("")
    A("## 5. 关键结果（自动从两表 CSV 提取）")
    A("")

    # ---- 附表结果 ----
    if real:
        A("### 附表（实数，CIFAR-10）")
        for col, name, hb in [("ID Acc", "ID Acc", True),
                              ("Avg_fpr95", "Avg FPR95", False),
                              ("Avg_aupr", "Avg AUPR", True),
                              ("Avg_auroc", "Avg AUROC", True)]:
            b = best_by_panel(real, col, hb)
            if "CIFAR-10" in b:
                m, v = b["CIFAR-10"]
                A(f"- **{name} 最优**：{m} ({v:.2f})")
        d = group_best(real, "Avg_fpr95").get("CIFAR-10", {})
        if d:
            A("- 各设置**最优方法**的 Avg FPR95：" +
              ", ".join(f"{k}={v:.2f}({m})" for k, (m, v) in sorted(d.items())
                        if v is not None))
        A("")
    else:
        A("### 附表（实数）")
        A("[表格尚未生成]")
        A("")

    # ---- 主表结果 ----
    if plural:
        A("### 主表（复数，6 面板）")
        A("| 面板 | 嫁接前 | SWA-G | Hyena-G | Mamba2-G | 嫁接降低FPR95? |")
        A("|---|---|---|---|---|---|")
        gb = group_best(plural, "Avg_fpr95")
        for p in sorted(gb):
            d = gb[p]
            pre = d.get("pre", (None, None))[1]
            posts = [d.get(g, (None, None))[1] for g in ("swa", "hyena", "mamba2")]
            posts_s = [f"{x:.2f}" if x is not None else "—" for x in posts]
            if pre is not None and all(x is not None for x in posts):
                best_post = min(posts)
                verdict = ("是" if best_post < pre - 0.5 else
                           "持平" if abs(best_post - pre) <= 0.5 else "否")
            else:
                verdict = "—"
            pre_s = f"{pre:.2f}" if pre is not None else "—"
            A(f"| {p} | {pre_s} | {' | '.join(posts_s)} | {verdict} |")
        best = best_by_panel(plural, "Avg_fpr95", higher_better=False)
        for p in sorted(best):
            m, v = best[p]
            A(f"- {p}: Avg FPR95 最优 = **{m}** ({v:.2f})")
        A("")
    else:
        A("### 主表（复数）")
        A("[表格尚未生成]")
        A("")

    A("## 6. 结论怎么读")
    A("- 每格 = 5 种子 mean±std（Deep Ensembles 为单值）；**加粗暗红 = 该列最优**"
      "（FPR95 取小，其余取大）。")
    A("- 同一面板内前 7 行（嫁接前）vs 后 21 行（嫁接后）逐行可比——骨干/数据/种子/超参全同。")
    A("- 主表 6 面板按「小数据先行」排列：OCTDL → BreastMRI → MCND → SAR-Aircraft → "
      "New-FUSAR → RetinalOCT。")
    A("- 嫁接后 FPR95 系统性上升/下降 ⇒ 序列算子替换对不确定性估计有系统性影响；")
    A("  不同头方向不一致 ⇒ 影响与证据建模方式有关。")
    A("")
    A("## 7. 产出物")
    A("| 产出 | 路径 |")
    A("|---|---|")
    A("| 附表 PNG/CSV | `project/final_tables/table_real_combined.{png,csv}` |")
    A("| 主表 PNG/CSV | `project/final_tables/table_plural_combined.{png,csv}` |")
    A("| 实数逐设置原始表 | `project/results*/table.png` |")
    A("| 复数原始 json | `project/results_plural*/panel_*/<method>_s<seed>.json` |")
    A("| 复现脚本 | `encode_vae_latents_plural.py` / `run_experiment_plural.py` / "
      "`run_master_v2.py`（总编排）/ `combine_tables.py`（两张大表）|")
    A("")
    A("## 8. 五分钟口头汇报提纲")
    A("1. **背景**（30s）：原论文把 DiT 的 MHA 换成 SWA/Hyena/Mamba2（序列算子嫁接）；")
    A("   我们问：嫁接后的骨干还能不能可靠地「知道自己不知道」。")
    A("2. **创新**（60s）：把 HEDL/Re-EDL/RS-NN（2024/25 最新头）嫁接到 DiT 池化特征上，")
    A("   并引入三篇论文对比清单中的 I-EDL/DUQ/MC-Dropout/Deep Ensembles 作为新基线，")
    A("   建立 嫁接前/后 × 7 方法 × 5 种子 完全公平对比。")
    A("3. **协议**（60s）：同种子、同超参、同编码；实数=CIFAR10+3 OOD；")
    A("   复数=6 个 MRI/OCT/SAR 数据集逐个轮 ID、跨域 OOD。")
    A("4. **结果**（90s）：附表看「嫁接是否伤害不确定性质量」；主表看哪个头/变体在医学/遥感数据最稳。")
    A("5. **结论**（30s）：给出「嫁接 × 不确定性头」的适用边界与推荐组合。")

    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"written {args.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
