# Uncertainty-Head Grafting on DiT (HEDL / Re-EDL / RS-NN + Baselines)

在 [grafting-main](https://github.com/)（DiT-XL/2 序列算子嫁接：SWA / Hyena / Mamba-2）基础上，
把 **不确定性估计头** 嫁接到 DiT 池化特征（1152 维）上，建立
**嫁接前 / 嫁接后 × 方法 × 种子** 完全公平的不确定性质量对比（实数 + 复数内涵数据）。

## 方法（表格行）

| 方法 | 出处 | 角色 |
|---|---|---|
| Re-EDL | TPAMI 2025 | 2024/25 最新头 |
| HEDL | NeurIPS 2024 | 2024/25 最新头 |
| RS-NN | ICLR 2025 | 2024/25 最新头 |
| I-EDL | ICML 2023 | 基线（移植自 Re-EDL 官方仓库 IEDL 分支） |
| DUQ | ICML 2020 | 基线（RBF 质心 + EMA） |
| MC-Dropout | ICML 2016 | 基线（T=10 随机前向） |
| Deep Ensembles | NeurIPS 2017 | 基线（5 个 softmax 种子模型合成，`make_de_results.py`） |

基线选取原则：从 Re-EDL / HEDL / RS-NN 三篇正式论文的对比清单中选取（它们已评估筛选过一轮）。

## 目录结构

```
heads/            6 个不确定性头（hedl/reedl/rsnn/iedl/duq/mcdropout）
backbones/        DiT 骨干（原始 + 嫁接后：SWA-50 / Hyena-50 / Mamba-2-50）
config.py         实数协议（CIFAR-10 ID；SVHN/DTD/Places365 OOD）
config_plural.py  复数协议（6 面板轮 ID：OCTDL/BreastMRI/MCND/SAR-Aircraft/New-FUSAR/RetinalOCT）
encode_vae_latents*.py  SD-VAE 预编码
run_experiment*.py      单方法训练+评估（5 种子 1000-1004）
run_master_v2.py        总编排器（74 任务：补齐/新方法/DE 合成/出表）
run_pipeline_guard.sh   断点续跑守护（磁盘金丝雀 + 失败补跑）
make_de_results.py      Deep Ensembles 合成（读 scores/*.npz）
combine_tables.py       生成两张最终大表（--real / --plural）
verify_jsons.py         结果 json 完整性校验
rsnn_focal_plural/      RS-NN focal 集固定文件（seed=123）
final_tables/           最终表格（png+csv）
```

## 复现

```bash
# 1. 权重（不随仓库分发，见下）
#    DiT-XL-2-256x256.pt / DiT-XL-2-swa-50p.pt / DiT-XL-2-hyena_x-50p.pt / DiT-XL-2-mamba_2-50p.pt
#    放入 ../grafting-main/pretrained_models/
# 2. VAE 编码
python encode_vae_latents.py && python encode_vae_latents_plural.py
# 3. 训练+评估（总编排，4 卡）
setsid nohup bash run_pipeline_guard.sh > /tmp/opencode/guard.log 2>&1 &
# 4. 出表
python combine_tables.py --real && python combine_tables.py --plural
python gen_report.py
```

## 权重下载

| 文件 | 大小 | 链接 |
|---|---|---|
| DiT-XL-2-256x256.pt（官方） | 2.7G | [ModelScope/HF 链接待补] |
| DiT-XL-2-swa-50p.pt（官方嫁接权重） | 2.7G | [ModelScope/HF 链接待补] |
| DiT-XL-2-hyena_x-50p.pt（官方嫁接权重） | 2.7G | [ModelScope/HF 链接待补] |
| DiT-XL-2-mamba_2-50p.pt（官方嫁接权重） | 2.9G | [ModelScope/HF 链接待补] |

## 结果

见 `final_tables/table_real_combined.png`（实数）与 `table_plural_combined.png`（复数 6 面板），
每格 = 5 种子 mean±std，加粗暗红 = 列最优（FPR95 取小，其余取大）。
