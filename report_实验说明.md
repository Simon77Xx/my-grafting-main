# 实验说明：不确定性头嫁接前后性能对比（实数 / 复数数据）

> 本文档由 gen_report.py 自动生成，数字直接来自 final_tables/ 两张大表 CSV。

## 1. 实验目的
在**完全公平**的协议下，比较「不确定性估计头」在骨干网络**嫁接前（原始 DiT-XL/2）**与**嫁接后（SWA-50 / Hyena-50 / Mamba-2 序列算子替换）**的性能差异：
1. 嫁接（原论文工作）是否保持/改变了不确定性估计能力；
2. 我们的创新点——把三种 2024/2025 最新不确定性头（HEDL / Re-EDL / RS-NN）与从其对比清单中选取的四个基线（I-EDL / DUQ / MC-Dropout / Deep Ensembles）嫁接到 DiT 特征上——在不同数据形态（实数 / 复数内涵）下表现如何。

## 2. 方法与骨干
- **骨干**：grafting-main（原论文主框架）的 DiT-XL/2，吃 SD-VAE 预编码 latent（4×32×32）。
- **嫁接前**：原始 DiT（官方权重 DiT-XL-2-256x256.pt），去噪末层替换为不确定性头。
- **嫁接后**：按原论文机制（ReplacementFactory）把 14 个 MHA 块（奇数层 1..27）替换为
  SWA / Hyena / Mamba-2，加载官方 stage2 嫁接权重（DiT-XL-2-*-50p.pt）。
- **七个方法**：Re-EDL（TPAMI 2025）、HEDL（NeurIPS 2024）、RS-NN（ICLR 2025）、I-EDL（ICML 2023）、DUQ（ICML 2020）、MC-Dropout（ICML 2016）、Deep Ensembles（NeurIPS 2017），全部接在 DiT 池化特征（1152 维）上。

> **基线选取依据**：原基线 Softmax（2017）已过旧，改从 Re-EDL / HEDL / RS-NN 三篇正式论文的对比清单中选取**它们已经评估筛选过**的方法作为新基线——I-EDL 直接移植自 Re-EDL 官方仓库的 IEDL 分支，DUQ / MC-Dropout / Deep Ensembles 为三篇论文共同的对比基线（全部为开源实现）。Softmax 仅作为 Deep Ensembles 的集成成员保留训练，不再作为独立对比行出现。

## 3. 公平性协议（实数/复数完全一致）
| 协议要素 | 设置 |
|---|---|
| 随机种子 | 1000-1004 共 5 个，全部方法/设置一致，报 mean±std（Deep Ensembles 为单值）|
| 训练超参 | 60 epoch、AdamW、lr 1e-4、wd 1e-4、cosine、batch 128、AMP bf16 |
| 数据编码 | 同一 SD-VAE（Resize256 bicubic + CenterCrop256 + /0.5 归一化）|
| 评估口径 | ID Acc + 各 OOD 的 FPR95↓/AUPR↑/AUROC↑ + Average |
| RS-NN focal 集 | 每数据集固定 seed=123 生成，所有方法/种子共享 |
| 数据划分 | 自带 train/test 用官方划分；否则按类分层 90/10（固定 seed）|

即：**任何性能差异只能归因于「嫁接与否」与「不确定性头种类」**。

## 4. 实验设计
### 附表（实数数据）
- ID = CIFAR-10；OOD = SVHN、DTD、Places365。7 方法（6 个逐种子训练 + Deep Ensembles）× 5 种子 × 4 设置。
### 主表（复数内涵数据 data_plural）
- 从 9 个数据集中选 **6 个**（MRI/OCT/SAR 各 2 个）**逐个轮 ID**（6 面板），OOD = 其他两域各 2 个、共 4 个数据集（跨域开放集，域差距大，更能区分不确定性质量）。
- 6 面板 × 7 方法 × 5 种子 × 4 设置。

## 5. 关键结果（自动从两表 CSV 提取）

### 附表（实数，CIFAR-10）
- **ID Acc 最优**：Re-EDL (97.52)
- **Avg FPR95 最优**：RS-NN (28.90)
- **Avg AUPR 最优**：Re-EDL (75.38)
- **Avg AUROC 最优**：Re-EDL (87.71)
- 各设置**最优方法**的 Avg FPR95：hyena=88.35(RS-NN (Hy-G)), mamba2=84.15(Re-EDL (Mamba-G)), pre=28.90(RS-NN), swa=83.33(RS-NN (SWA-G))

### 主表（复数，6 面板）
| 面板 | 嫁接前 | SWA-G | Hyena-G | Mamba2-G | 嫁接降低FPR95? |
|---|---|---|---|---|---|
| BreastMRI | 21.64 | — | — | — | — |
| MCND | 61.16 | — | — | — | — |
| OCTDL | 87.94 | — | — | — | — |
| SAR-Aircraft | 47.21 | — | — | — | — |
- BreastMRI: Avg FPR95 最优 = **HEDL** (21.64)
- MCND: Avg FPR95 最优 = **RS-NN** (61.16)
- OCTDL: Avg FPR95 最优 = **HEDL** (87.94)
- SAR-Aircraft: Avg FPR95 最优 = **Re-EDL** (47.21)

## 6. 结论怎么读
- 每格 = 5 种子 mean±std（Deep Ensembles 为单值）；**加粗暗红 = 该列最优**（FPR95 取小，其余取大）。
- 同一面板内前 7 行（嫁接前）vs 后 21 行（嫁接后）逐行可比——骨干/数据/种子/超参全同。
- 主表 6 面板按「小数据先行」排列：OCTDL → BreastMRI → MCND → SAR-Aircraft → New-FUSAR → RetinalOCT。
- 嫁接后 FPR95 系统性上升/下降 ⇒ 序列算子替换对不确定性估计有系统性影响；
  不同头方向不一致 ⇒ 影响与证据建模方式有关。

## 7. 产出物
| 产出 | 路径 |
|---|---|
| 附表 PNG/CSV | `project/final_tables/table_real_combined.{png,csv}` |
| 主表 PNG/CSV | `project/final_tables/table_plural_combined.{png,csv}` |
| 实数逐设置原始表 | `project/results*/table.png` |
| 复数原始 json | `project/results_plural*/panel_*/<method>_s<seed>.json` |
| 复现脚本 | `encode_vae_latents_plural.py` / `run_experiment_plural.py` / `run_master_v2.py`（总编排）/ `combine_tables.py`（两张大表）|

## 8. 五分钟口头汇报提纲
1. **背景**（30s）：原论文把 DiT 的 MHA 换成 SWA/Hyena/Mamba2（序列算子嫁接）；
   我们问：嫁接后的骨干还能不能可靠地「知道自己不知道」。
2. **创新**（60s）：把 HEDL/Re-EDL/RS-NN（2024/25 最新头）嫁接到 DiT 池化特征上，
   并引入三篇论文对比清单中的 I-EDL/DUQ/MC-Dropout/Deep Ensembles 作为新基线，
   建立 嫁接前/后 × 7 方法 × 5 种子 完全公平对比。
3. **协议**（60s）：同种子、同超参、同编码；实数=CIFAR10+3 OOD；
   复数=6 个 MRI/OCT/SAR 数据集逐个轮 ID、跨域 OOD。
4. **结果**（90s）：附表看「嫁接是否伤害不确定性质量」；主表看哪个头/变体在医学/遥感数据最稳。
5. **结论**（30s）：给出「嫁接 × 不确定性头」的适用边界与推荐组合。
