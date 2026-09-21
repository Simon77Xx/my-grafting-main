"""复数（data_plural）实验协议配置。

与实数管线（config.py）协议完全对齐，保证可比性：
- 同种子（SEEDS 环境变量，默认 1000-1005）、同方法（softmax/reedl/hedl/rsnn）
- 同骨干（DiT-XL/2；GRAFT 空=嫁接前，swa50/hyena50/mamba2_50=嫁接后）
- 同训练超参（LR/WD/cosine/60 epoch/batch128/AMP bf16）
- 同 VAE 编码协议（Resize256 bicubic + CenterCrop256 + Normalize0.5 → 4x32x32 latent）

面板结构（方案B，逐数据集轮 ID）：9 个数据集轮流当 ID，
OOD = 其他两个域的全部数据集（每面板 6 个 OOD）。
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PLURAL_DATA_ROOT = "/supcon3/shixisheng/xjx/research/data_plural"
LATENT_ROOT = "/supcon3/shixisheng/xjx/research/data/vae_latents_plural"
FOCAL_ROOT = os.path.join(ROOT, "rsnn_focal_plural")

# 种子（与实数一致；env 可覆盖）
_env_seeds = os.environ.get("SEEDS", "1000,1001,1002,1003,1004")
SEEDS = [int(s) for s in _env_seeds.split(",")]

# 嫁接配置（与 config.py 相同语义）
GRAFT = os.environ.get("GRAFT", "").strip()
GRAFT_SPECS = {
    "hyena50":   ("hyena_x", "configs/sequence_mixers/hyena_x.yaml",
                  "DiT-XL-2-hyena_x-50p.pt"),
    "swa50":     ("swa", "configs/sequence_mixers/swa.yaml",
                  "DiT-XL-2-swa-50p.pt"),
    "mamba2_50": ("mamba2", "configs/sequence_mixers/mamba2.yaml",
                  "DiT-XL-2-mamba_2-50p.pt"),
}

# 数据集分层划分固定种子（所有方法共享同一划分，保证公平）
SPLIT_SEED = 123

# 面板注册表：key -> dict(domain, n, path, mode)
#   mode: builtin  = 数据集自带 train/test 目录
#         stratify = 按类分层 90/10 划分（seed=SPLIT_SEED，所有方法共享）
PANELS = {
    "OCTDL":        dict(domain="OCT", n=5,  path="OCT/OCTDL", mode="stratify"),
    "labeledOCT":   dict(domain="OCT", n=3,  path="OCT/labeled OCT", mode="stratify"),
    "RetinalOCT":   dict(domain="OCT", n=8,  path="OCT/RetinalOCT_128w_pad", mode="builtin"),
    "BreastMRI":    dict(domain="MRI", n=2,  path="MRI/Breast Cancer Patients MRI's", mode="stratify"),
    "CADCardiac":   dict(domain="MRI", n=2,  path="MRI/CAD Cardiac MRI", mode="stratify"),
    "MCND":         dict(domain="MRI", n=8,  path="MRI/MCND", mode="builtin"),
    "SARAircraft":  dict(domain="SAR", n=6,  path="SAR/Aircraft", mode="stratify"),
    "NewFUSAR":     dict(domain="SAR", n=10, path="SAR/New_FUSAR", mode="builtin"),
    "OpenSARUrban": dict(domain="SAR", n=10, path="SAR/OpenSARUrbanPatch", mode="stratify"),
}

# 表格行顺序（小面板在前，尽早出预览）
# 6 面板方案（每域 2 个）：避开 labeledOCT/OpenSARUrban/CADCardiac 三个最大数据集；
# 每个面板的 OOD = 其他两域各 2 个，共 4 个，口径统一。
PANEL_ORDER = ["OCTDL", "BreastMRI", "MCND", "SARAircraft", "NewFUSAR", "RetinalOCT"]

# 训练面板时每个面板的 OOD = 其他两域的数据集
def ood_panels(panel_key):
    dom = PANELS[panel_key]["domain"]
    return [k for k in PANEL_ORDER if PANELS[k]["domain"] != dom]

# 表格显示名
DISPLAY_PANEL = {
    "OCTDL": "OCTDL", "labeledOCT": "OCT-Labeled", "RetinalOCT": "RetinalOCT",
    "BreastMRI": "BreastMRI", "CADCardiac": "CAD-Cardiac", "MCND": "MCND",
    "SARAircraft": "SAR-Aircraft", "NewFUSAR": "New-FUSAR", "OpenSARUrban": "OpenSAR-Urban",
}

# 训练超参（与 config.py 完全一致；PLURAL_EPOCHS/PLURAL_BATCH 仅用于冒烟测试）
EPOCHS = int(os.environ.get("PLURAL_EPOCHS", "60"))
BATCH_SIZE = int(os.environ.get("PLURAL_BATCH", "128"))
LR = 1e-4
WEIGHT_DECAY = 1e-4
LR_SCHEDULE = "cosine"
GRAD_ACCUM = 1
USE_AMP = True
AMP_DTYPE = "bf16"
NUM_WORKERS = 0

METHODS = ["softmax", "reedl", "hedl", "rsnn", "iedl", "duq", "mcdropout"]

# 结果目录：嫁接前 results_plural/，嫁接后 results_plural_g_<spec>/
def results_root():
    return os.path.join(ROOT, "results_plural" if not GRAFT else f"results_plural_g_{GRAFT}")

def panel_dir(panel_key):
    return os.path.join(results_root(), f"panel_{panel_key}")

# RSNN focal 集文件（每个面板一份，整数索引 focal 集）
def focal_path(panel_key):
    return os.path.join(FOCAL_ROOT, f"{panel_key}_new_classes.npy")
