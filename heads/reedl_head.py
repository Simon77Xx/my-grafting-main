"""Re-EDL 头：EDLHead（softplus evidence -> alpha）+ ReEDLLoss（MSE）。

忠实还原 RE_EDL_SOURCE_ANALYSIS.md 的 EDLHead / ReEDLLoss / MEDLLoss。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EDLHead(nn.Module):
    def __init__(self, feat_dim, num_classes, lamb=0.8, evidence_fn="softplus"):
        super().__init__()
        self.num_classes = num_classes
        self.lamb = lamb
        self.evidence_fn = evidence_fn
        self.fc = nn.Linear(feat_dim, num_classes)

    def compute_evidence(self, logits):
        if self.evidence_fn == "softplus":
            return F.softplus(logits)
        elif self.evidence_fn == "exp":
            return torch.exp(torch.clamp(logits, max=10.0))
        elif self.evidence_fn == "relu":
            return F.relu(logits)
        raise ValueError(f"unsupported evidence_fn: {self.evidence_fn}")

    def forward(self, features):
        return self.fc(features)

    def alpha_prob_unc(self, logits):
        evidence = self.compute_evidence(logits)
        alpha = evidence + self.lamb
        alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / alpha0
        uncertainty = (self.num_classes * self.lamb) / alpha0
        return alpha, prob, uncertainty

    def uncertainty(self, features):
        with torch.no_grad():
            _, _, unc = self.alpha_prob_unc(self.forward(features))
        return unc.squeeze(-1)


class ReEDLLoss(nn.Module):
    def __init__(self, num_classes, lamb=0.8, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.lamb = lamb
        self.reduction = reduction

    def forward(self, evidence, labels):
        alpha = evidence + self.lamb
        alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / alpha0
        if labels.dim() == 1:
            labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        else:
            labels_onehot = labels.float()
        return F.mse_loss(prob, labels_onehot, reduction=self.reduction)


class MEDLLoss(nn.Module):
    """原始 Re-EDL 论文 Modified MSE Loss（lamb1=1.0 时退化为标准 P=α/Σα）。"""

    def __init__(self, num_classes, lamb1=1.0, lamb2=0.8, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.reduction = reduction

    def forward(self, evidence, labels):
        B, C = evidence.shape
        if labels.dim() == 1:
            labels_onehot = F.one_hot(labels, num_classes=C).float()
        else:
            labels_onehot = labels.float()
        evidence_sum = torch.sum(evidence, dim=-1, keepdim=True)
        modified_S = evidence + self.lamb1 * (evidence_sum - evidence) + self.lamb2 * self.num_classes
        alpha = evidence + self.lamb2
        prob = alpha / modified_S
        loss = (labels_onehot - prob).pow(2).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
