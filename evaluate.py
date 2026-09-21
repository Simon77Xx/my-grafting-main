"""统一评估：Acc / FPR95 / AUPR / AUROC（sklearn，百分比）。"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


def compute_ood_metrics(id_unc, ood_unc):
    """id_unc/ood_unc: 每样本 uncertainty（越高越 OOD）。ID 当正类，分数取 -unc。"""
    scores = np.concatenate([-np.asarray(id_unc), -np.asarray(ood_unc)])
    labels = np.concatenate([np.ones(len(id_unc)), np.zeros(len(ood_unc))])
    fpr, tpr, _ = roc_curve(labels, scores)
    idx95 = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx95]
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    return fpr95 * 100, auroc * 100, aupr * 100


def compute_acc(preds, labels):
    return (np.asarray(preds) == np.asarray(labels)).mean() * 100
