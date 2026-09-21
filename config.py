"""统一协议配置 (阶段二)."""
import os

# 根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/supcon3/shixisheng/xjx/research/data"

# 种子（全部方法共用；6 种子，嫁接前/后一致）
# 允许通过环境变量 SEEDS 覆盖（如预览 / 5 种子模式）
_env_seeds = os.environ.get("SEEDS", "1000,1001,1002,1003,1004")
SEEDS = [int(s) for s in _env_seeds.split(",")]

# ID / OOD
ID_NAME = "cifar10"
OOD_NAMES = ["SVHN", "DTD", "Places365"]

# 嫁接配置：GRAFT 为空 = 原始 DiT（嫁接前）；否则为嫁接后骨干变体。
# 由环境变量 GRAFT 切换，使 run_experiment / make_table / render_table_png
# 无需改代码即可分别产出嫁接前后的结果与表格。
GRAFT = os.environ.get("GRAFT", "").strip()
GRAFT_SPECS = {
    # spec: (operator_name, operator_config_relpath, grafted_ckpt_filename)
    "hyena50":   ("hyena_x", "configs/sequence_mixers/hyena_x.yaml",
                  "DiT-XL-2-hyena_x-50p.pt"),
    "swa50":     ("swa", "configs/sequence_mixers/swa.yaml",
                  "DiT-XL-2-swa-50p.pt"),
    "mamba2_50": ("mamba2", "configs/sequence_mixers/mamba2.yaml",
                  "DiT-XL-2-mamba_2-50p.pt"),
}
# 主论文 50% 嫁接层（奇数层 1..27，来自 HF config.json grafted_layer_indices）
GRAFT_INDEXES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]

# --- DiT backbone（grafting-main 主架构作为分类骨干）----------------------------
DIT_VARIANT = "DiT-XL/2"           # 备选 "DiT-B/2"（需更大显存，可加载官方预训练权重）
DIT_PATCH_SIZE = 2
DIT_INPUT_MODE = "vae_latent"      # "vae_latent" 吃 SD-VAE 预编码 latent（复用预训练 patch_embed，推荐）| "rgb" 直接吃 RGB（patch_embed 重训，实测特征学不出分类性）
DIT_CKPT_PATH = os.path.join(ROOT, "..", "grafting-main", "pretrained_models", "DiT-XL-2-256x256.pt")  # grafting 官方预训练 DiT 权重
DIT_FREEZE_BLOCKS = False         # True=linear probe（blocks 冻结，只训 patch_embed/头）；False=全微调
DIT_NUM_TRAINABLE_BLOCKS = 0      # >0：FREEZE_BLOCKS=True 时额外解冻最后 N 个 block
DIT_POOL = "mean"                 # patch-token 池化方式
DIT_CONDITIONING = "zero"        # 固定 c=t(0)+y(0)，把 class-conditional AdaLN 退化为确定性 affine
GRAFTING_SRC = os.path.join(ROOT, "..", "grafting-main", "src")

# 骨干与训练
BACKBONE = "DiT-XL/2"          # ImageNet 预训练，去掉 fc
IMAGE_SIZE = 224               # 与预训练一致，全程一致
EPOCHS = 60                    # 全部方法一致
BATCH_SIZE = 128
USE_AMP = True                # 混合精度（DiT-XL/2 fp32 太慢且显存 108GB）
AMP_DTYPE = "bf16"            # bf16（H200 支持，fp32 指数范围无溢出）| fp16
LR = 1e-4
WEIGHT_DECAY = 1e-4
LR_SCHEDULE = "cosine"     # None=固定LR | "cosine"=cosine衰减（DiT 后期 loss 反弹，防发散）
GRAD_ACCUM = 1             # 梯度累积步数（等效 batch = BATCH_SIZE × GRAD_ACCUM）
OPTIMIZER = "adamw"
NUM_WORKERS = 0

# 各方法头参数
HEDL_W = 2.0                    # HEDL: W 先验强度
REEDL_LAMB = 0.8               # Re-EDL: lambda (lamb2)
REEDL_EVIDENCE = "softplus"
RSNN_NEW_CLASSES = os.path.join(ROOT, "..", "Random-Set-Neural-Networks-main", "new_classes.npy")

# Backbone-driven image size: DiT 变体吃 latent 32x32（SD-VAE 编码 256px 图得到，
# patch_size=2 → 16×16=256 个 patch-token，与 DiT 预训练一致）；ResNet18 仍保持 224。
if str(BACKBONE).startswith("DiT"):
    IMAGE_SIZE = 32
RSNN_ALPHA = 0.001              # mass 正则权重
RSNN_BETA = 0.001               # mass 和约束权重

# 方法列表（顺序即表格行序）
METHODS = ["softmax", "reedl", "hedl", "rsnn", "iedl", "duq", "mcdropout"]

# --- 新增较新基线超参（全部来自 RS-NN/HEDL/Re-EDL 三篇论文的对比清单）---------
# I-EDL: Deng et al., "Uncertainty Estimation by Fisher Information-based
#        Evidential Deep Learning", ICML 2023（移植自 Re-EDL 官方仓库 IEDL 分支）
IEDL_LAMB = 0.8                  # Dirichlet 先验权重 (lamb2)
IEDL_FISHER_C = 1.0              # Fisher 行列式正则权重
IEDL_KL_C = -1.0                 # -1 = 非目标证据 KL 按 epoch/10 线性爬升
# DUQ: van Amersfoort et al., "Uncertainty Estimation Using a Single Deep
#      Deterministic Neural Network", ICML 2020
DUQ_LENGTHSCALE = 0.3            # RBF 核长度尺度（特征已 L2 归一化）
DUQ_GAMMA = 0.999                # 质心 EMA 系数
DUQ_EPSILON = 0.05               # BCE 平滑目标 epsilon
DUQ_NORM_FEATURES = True         # L2 归一化骨干特征（与特征尺度解耦）
# MC-Dropout: Gal & Ghahramani, ICML 2016
MCD_DROPOUT_P = 0.1              # dropout 比例
MCD_T = int(os.environ.get("MCD_T", "10"))   # 推理期随机前向次数

# 结果目录：嫁接前 results/，嫁接后 results_g_<spec>/
RESULTS_DIR = os.path.join(ROOT, "results" if not GRAFT else f"results_g_{GRAFT}")
