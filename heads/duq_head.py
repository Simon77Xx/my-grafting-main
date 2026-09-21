"""DUQ 头（Uncertainty Estimation Using a Single Deep Deterministic Neural Network,
van Amersfoort et al., ICML 2020）。

按论文 Algorithm 1 实现：RBF 输出层（W 投影 + 每类一个标量质心 c_k），
质心用 EMA 更新（不参与梯度），损失 = 平滑目标的 BCE。
不确定性 = 1 - max_k y_k(x)。
特征做 L2 归一化，使 lengthscale 与骨干特征尺度无关。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DUQHead(nn.Module):
    def __init__(self, feat_dim, num_classes, lengthscale=0.3, gamma=0.999,
                 epsilon=0.05, norm_features=True):
        super().__init__()
        self.num_classes = num_classes
        self.lengthscale = lengthscale
        self.lam = lengthscale ** 2
        self.gamma = gamma
        self.epsilon = epsilon
        self.norm_features = norm_features
        self.W = nn.Parameter(0.05 * torch.randn(num_classes, feat_dim))
        self.register_buffer("centroids", 0.05 * torch.randn(num_classes))

    def normalize(self, features):
        if self.norm_features:
            return F.normalize(features, dim=-1)
        return features

    def projected(self, z):
        """z: (B, d) -> u: (B, K)，第 k 类的标量投影。"""
        return z @ self.W.t()

    def update_centroids(self, u, y):
        """EMA 更新质心（无梯度）。u: (B, K), y: (B,)"""
        with torch.no_grad():
            for k in range(self.num_classes):
                mask = y == k
                if mask.any():
                    mean_k = u[mask, k].mean()
                    self.centroids[k] = (self.gamma * self.centroids[k]
                                         + (1 - self.gamma) * mean_k)

    def kernel_probs(self, z):
        """y_k(x) = exp(-||W_k z - c_k||^2 / lam)，z 已归一化。"""
        u = self.projected(z)
        d2 = (u - self.centroids.unsqueeze(0)).pow(2)
        return torch.exp(-d2 / self.lam)

    def forward(self, features):
        return self.kernel_probs(self.normalize(features))

    def uncertainty(self, features):
        with torch.no_grad():
            prob = self.kernel_probs(self.normalize(features))
        return 1 - prob.max(dim=-1).values

    def loss_from_features(self, features, labels):
        """两步：① EMA 更新质心（无梯度）② 用更新后质心算 BCE（平滑目标）。"""
        z = self.normalize(features)
        with torch.no_grad():
            u = self.projected(z)
            self.update_centroids(u, labels)
        c = self.centroids.detach()
        d2 = (self.projected(z) - c.unsqueeze(0)).pow(2)
        y_pred = torch.exp(-d2 / self.lam).clamp(1e-6, 1 - 1e-6)
        target = (F.one_hot(labels, num_classes=self.num_classes).float()
                  * (1 - self.epsilon) + self.epsilon / self.num_classes)
        return F.binary_cross_entropy(y_pred, target)
