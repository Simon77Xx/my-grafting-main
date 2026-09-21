"""MC-Dropout 头（Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016）。

双层 MLP 头 + Dropout；训练与普通 CE 完全一致（dropout 自动生效），
推理时保持 dropout 开启做 T 次随机前向：
  prob = mean_t softmax(logits_t)
  pred = argmax(prob)
  uncertainty = predictive entropy（越大越 OOD）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropoutHead(nn.Module):
    def __init__(self, feat_dim, num_classes, p=0.1, hidden=512):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.hidden = hidden
        self.dropout1 = nn.Dropout(p)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.dropout2 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, features):
        x = self.dropout1(features)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

    def enable_mc_dropout(self):
        """推理期保持 dropout 生效（其余模块仍 eval）。"""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def mc_predict(self, features, t=10):
        """T 次随机前向 -> (prob_mean, predictive_entropy)。"""
        probs = []
        with torch.no_grad():
            for _ in range(t):
                logits = self.forward(features)
                probs.append(F.softmax(logits, dim=-1))
        prob = torch.stack(probs, dim=0).mean(dim=0)
        entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=-1)
        return prob, entropy
