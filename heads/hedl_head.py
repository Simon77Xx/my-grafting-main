"""HEDL 头：HENN 投影 + digamma EDL 损失（忠实原始 losses.py 的 edl_HENN）。

对齐原始实现（hedl/5024_.../code/losses.py）：
- get_henn_fc：mask(fc.weight>0) / max(1, 每列正数个数)，bias = W/C；
- hedl_loss(edl_HENN)：A = Σ y·(digamma(S) − digamma(alpha))，即 digamma EDL 损失；
- uncertainty u = W / Σ alpha（val.py 的口径）。

DiT 适配点（在保持 HEDL 语义的前提下，均在 fp32 下计算）：
1. 证据 alpha = softplus(henn 投影) + 1e-6，保证 alpha>0。原始 henn 投影无激活，
   在 DiT pooled 特征上 alpha 可能 <=0，digamma 在 0/负整数附近发散 → NaN。
2. 保留一个 0.5×CE 辅助项，但作用在【真实分类 logits】上（原始代码是把 outputs.data
   替换成 alpha 再算梯度注入，DiT 上无法稳定收敛；CE 提供稳定的分类监督路径）。
3. 头与损失由 trainer 在【禁用 autocast 的 fp32 上下文】中调用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def henn_weight_norm(weight, num_classes, device):
    """HENN 投影权重：mask(fc.weight>0) / max(1, 每列正数个数)。忠实原始 get_henn_fc。"""
    with torch.no_grad():
        mask = (weight > 0).float()
        item_num = torch.sum(mask, dim=0)                       # (D,)
        item_num = item_num.unsqueeze(0).expand(num_classes, -1)
        weight_norm = mask / torch.max(torch.ones_like(item_num), item_num)
    return weight_norm.to(device)


class HEDLHead(nn.Module):
    """HEDL 头：HENN 证据投影（alpha）+ 真实 logits 辅助分类。"""

    def __init__(self, feat_dim, num_classes, W=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.W = W
        # 特征归一化：DiT pooled 特征过骨干层 LayerNorm 后再校准一次，稳定投影量级
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes)

    def get_weight(self):
        return self.fc.weight

    def forward(self, features):
        """真实分类 logits（辅助 CE 路径）。"""
        return self.fc(self.feat_norm(features))

    def henn_forward(self, features, device=None):
        """HENN 投影证据 alpha = softplus(feat @ weight_norm^T + W/C) + 1e-6 > 0。"""
        device = device or features.device
        feats = self.feat_norm(features).to(device)
        wn = henn_weight_norm(self.get_weight(), self.num_classes, device)
        bias = torch.full((self.num_classes,), self.W / self.num_classes, device=device)
        z = F.linear(feats, wn, bias)
        # softplus 保证 alpha>0：digamma 数值安全，且与 EDL softplus evidence 惯例一致
        return F.softplus(z) + 1e-6

    def hedl_ce(self, y_onehot, features, outputs=None, device=None):
        """EDL digamma 损失 + 0.5×CE(真实 logits)。

        与原 trainer 的调用签名保持一致：(y_onehot, features, outputs, device)。
        outputs 参数仅为兼容旧调用，本实现不再做 outputs.data 替换。
        """
        device = device or features.device
        alpha = self.henn_forward(features, device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(y_onehot.to(device) *
                      (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        edl_loss = A.mean()
        logits = self.forward(features)
        ce_loss = F.cross_entropy(logits, y_onehot.argmax(dim=1))
        return edl_loss + 0.5 * ce_loss

    def uncertainty(self, features, device=None):
        """OOD 不确定度 u = W / Σ alpha（HEDL 原口径）。"""
        with torch.no_grad():
            alpha = self.henn_forward(features, device)
            u = self.W / torch.clamp(torch.sum(alpha, dim=1), min=1e-6)
        return u