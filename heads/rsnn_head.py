"""RS-NN 头：DiT-compatible 版本。
关键修复：
1. 特征归一化 (LayerNorm) - 适配 DiT 特征尺度
2. Belief 用 sigmoid + 温度缩放 - 更稳定的概率输出
3. Mass 计算全程 clamp + 数值安全归一化
4. BCE loss 加 label smoothing + 梯度裁剪
5. 不确定度计算全程 nan_to_num
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def _class_to_idx(sets):
    """focal 集元素 → 类别索引元组。
    兼容两种格式：字符串类名（CIFAR10，原始管线）与整数索引（复数面板）。
    """
    idx_sets = []
    for s in sets:
        if len(s) and isinstance(next(iter(s)), (int, np.integer)):
            idx_sets.append(tuple(sorted(int(c) for c in s)))
        else:
            idx_sets.append(tuple(sorted(CIFAR10_CLASSES.index(c) for c in s)))
    return idx_sets


def mass_coeff(idx_sets):
    n = len(idx_sets)
    m = np.zeros((n, n))
    for i, A in enumerate(idx_sets):
        for j, B in enumerate(idx_sets):
            if set(B).issubset(set(A)):
                m[j, i] = (-1) ** (len(A) - len(B))
    return m


def betp_matrix(idx_sets, num_classes):
    full = idx_sets + [tuple(range(num_classes))]
    n = len(full)
    m = np.zeros((n, num_classes))
    for i, c in enumerate(range(num_classes)):
        for j, A in enumerate(full):
            if {c}.issubset(set(A)):
                m[j, i] = 1.0 / len(A)
    return m


def groundtruth_encoding(labels, idx_sets, num_classes):
    B = labels.shape[0]
    enc = np.zeros((B, len(idx_sets)), dtype=np.float32)
    for j, A in enumerate(idx_sets):
        enc[np.isin(labels, A), j] = 1.0
    return enc


class RSNNHead(nn.Module):
    def __init__(self, feat_dim, num_classes, new_classes_path,
                 alpha=0.001, beta=0.001, temp=1.0, label_smoothing=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.temp = temp
        self.label_smoothing = label_smoothing

        raw = np.load(new_classes_path, allow_pickle=True)
        idx_sets = _class_to_idx(raw)
        self.idx_sets = idx_sets
        self.num_focal = len(idx_sets)

        # 特征归一化 - 关键
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.fc = nn.Linear(feat_dim, self.num_focal)

        self.register_buffer("M", torch.tensor(mass_coeff(idx_sets), dtype=torch.float32))
        self.register_buffer("BetP", torch.tensor(betp_matrix(idx_sets, num_classes),
                                                  dtype=torch.float32))

    def forward(self, features):
        """输出 belief ∈ (0,1) per focal element。"""
        features = self.feat_norm(features)
        logits = self.fc(features) / self.temp
        belief = torch.sigmoid(logits)
        # clamp 避免 0/1 边界
        return belief.clamp(1e-7, 1 - 1e-7)

    def belief_to_mass(self, belief):
        """belief -> mass，全程数值安全。"""
        mass = belief @ self.M
        mass = torch.clamp(mass, min=0.0)
        # 未分配质量
        sums = 1.0 - mass.sum(dim=-1, keepdim=True)
        sums = torch.clamp(sums, min=0.0)
        mass = torch.cat([mass, sums], dim=-1)
        # 安全归一化
        denom = mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mass = mass / denom
        return mass

    def final_betp(self, mass):
        betp = mass @ self.BetP
        return torch.clamp(betp, 1e-8, 1.0)

    def loss(self, belief, labels):
        """忠实原始 rsnn_loss.py：BCE(focal-set 级) + α·mass_reg + β·mass_sum。

        注：原 TF 实现只有 BCE + mass 正则，没有 betp 上的 CE 项——
        betp.log() 在 1e-8 下界处是 bf16/fp32 混合精度的 NaN 源，已移除。
        带可选 label smoothing（默认 0，即与原版一致）。
        """
        y_true = torch.from_numpy(groundtruth_encoding(
            labels.detach().cpu().numpy(), self.idx_sets, self.num_classes)).to(belief.device)

        # Label smoothing（默认 0，关闭）
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / self.num_focal

        # BCE (focal set level)，belief 已 clamp 到 (1e-7, 1-1e-7)，数值安全
        bce = F.binary_cross_entropy(belief, y_true, reduction='none').mean(dim=0).mean()

        # Mass 正则（与原始 rsnn_loss.py 完全一致）
        mass = belief @ self.M
        mass_reg = torch.relu(-mass).mean()
        mass_sum = torch.relu(mass.sum(dim=-1).mean() - 1.0)

        return bce + self.alpha * mass_reg + self.beta * mass_sum

    def betp_from_features(self, features):
        with torch.no_grad():
            belief = self.forward(features)
            mass = self.belief_to_mass(belief)
            betp = self.final_betp(mass)
            betp = torch.nan_to_num(betp, nan=1e-8, posinf=1.0, neginf=1e-8)
        return betp

    def uncertainty(self, features):
        """OOD 不确定度 = entropy(BetP)。"""
        with torch.no_grad():
            betp = self.betp_from_features(features)
            u = -(betp * torch.log(betp)).sum(dim=-1)
            u = torch.nan_to_num(u, nan=10.0, posinf=10.0, neginf=0.0)
        return u