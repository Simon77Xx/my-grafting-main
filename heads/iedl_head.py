"""I-EDL 头（Fisher Information-based Evidential Deep Learning, Deng et al., ICML 2023）。

忠实移植自 Re-EDL 官方仓库 code_classical/models/ModifiedEvidentialN.py 的
loss == 'IEDL' 分支：Dirichlet alpha 上的 Fisher 信息加权 MSE + 方差项
+ Fisher 行列式正则 + 非目标证据 KL（前 10 epoch 线性爬升）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IEDLHead(nn.Module):
    def __init__(self, feat_dim, num_classes, lamb=0.8, fisher_c=1.0, kl_c=-1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lamb = lamb            # Dirichlet 先验权重 (lamb2)
        self.fisher_c = fisher_c
        self.kl_c = kl_c            # -1 = 按 epoch/10 线性爬升
        self.cur_epoch = 0          # 由 trainer 每个 epoch 更新（供 KL 爬升）
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

    def alpha_prob_unc(self, logits):
        evidence = F.softplus(logits)
        alpha = evidence + self.lamb
        alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / alpha0
        uncertainty = (self.num_classes * self.lamb) / alpha0
        return alpha, prob, uncertainty.squeeze(-1)

    def uncertainty(self, features):
        with torch.no_grad():
            _, _, unc = self.alpha_prob_unc(self.forward(features))
        return unc

    def loss(self, logits, labels):
        return IEDLLoss(num_classes=self.num_classes, lamb=self.lamb,
                        fisher_c=self.fisher_c, kl_c=self.kl_c,
                        cur_epoch=self.cur_epoch)(logits, labels)


class IEDLLoss(nn.Module):
    def __init__(self, num_classes, lamb=0.8, fisher_c=1.0, kl_c=-1.0, cur_epoch=0):
        super().__init__()
        self.num_classes = num_classes
        self.lamb = lamb
        self.fisher_c = fisher_c
        self.kl_c = kl_c
        self.cur_epoch = cur_epoch

    def compute_fisher_mse(self, labels_1hot, alpha):
        S = torch.sum(alpha, dim=-1, keepdim=True)
        gamma1_alpha = torch.polygamma(1, alpha)
        gamma1_S = torch.polygamma(1, S)
        gap = labels_1hot - alpha / S
        loss_mse = (gap.pow(2) * gamma1_alpha).sum(-1).mean()
        loss_var = (alpha * (S - alpha) * gamma1_alpha / (S * S * (S + 1))).sum(-1).mean()
        loss_det_fisher = -(torch.log(gamma1_alpha).sum(-1)
                            + torch.log(1.0 - (gamma1_S / gamma1_alpha).sum(-1))).mean()
        return loss_mse, loss_var, loss_det_fisher

    def compute_kl_loss(self, alphas, target_concentration, epsilon=1e-8):
        target_alphas = torch.ones_like(alphas) * target_concentration
        alp0 = torch.sum(alphas, dim=-1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)
        alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term,
                                torch.zeros_like(alp0_term))
        alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon)
                                - torch.lgamma(alphas + epsilon)
                                + (alphas - target_alphas)
                                * (torch.digamma(alphas + epsilon)
                                   - torch.digamma(alp0 + epsilon)),
                                dim=-1, keepdim=True)
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term,
                                  torch.zeros_like(alphas_term))
        return torch.squeeze(alp0_term + alphas_term).mean()

    def forward(self, logits, labels):
        labels_1hot = F.one_hot(labels, num_classes=self.num_classes).float()
        evidence = F.softplus(logits)
        alpha = evidence + self.lamb

        loss_mse, loss_var, loss_fisher = self.compute_fisher_mse(labels_1hot, alpha)
        grad_loss = loss_mse + loss_var + self.fisher_c * loss_fisher

        kl_alpha = evidence * (1 - labels_1hot) + self.lamb
        loss_kl = self.compute_kl_loss(kl_alpha, self.lamb)
        regr = self.kl_c if self.kl_c >= 0 else float(np.minimum(1.0, self.cur_epoch / 10.0))
        return grad_loss + regr * loss_kl
