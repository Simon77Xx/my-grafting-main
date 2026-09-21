"""统一骨干：ImageNet 预训练 ResNet18，去掉分类头，输出 512 维特征。"""
import torch.nn as nn
import torchvision.models as models


def get_backbone():
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Identity()  # 去掉分类头，输出 512 维
    return net


class BackboneWrapper(nn.Module):
    """带头的完整模型：骨干 + 方法头。"""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def build_model(method, num_classes=10):
    from heads.hedl_head import HEDLHead
    from heads.reedl_head import EDLHead
    from heads.rsnn_head import RSNNHead
    from config import (BACKBONE, HEDL_W, REEDL_LAMB, REEDL_EVIDENCE,
                        RSNN_NEW_CLASSES)

    # 统一骨干：ResNet18（原方案A）或 grafting-main 的 DiT（方案B）
    if BACKBONE == "ResNet18":
        backbone = get_backbone()
        feat_dim = 512
    elif str(BACKBONE).startswith("DiT"):
        import config as _cfg
        if _cfg.GRAFT:
            from backbones.dit_grafted_backbone import build_grafted_dit_backbone
            backbone = build_grafted_dit_backbone(_cfg)
        else:
            from backbones.dit_backbone import get_dit_backbone
            backbone = get_dit_backbone(_cfg)
        feat_dim = backbone.feat_dim
    else:
        raise ValueError(f"unknown backbone: {BACKBONE}")

    # 三个不确定性头 + softmax 基线，feat_dim 随骨干自动适配（这是"嫁接"的关键）
    if method == "softmax":
        head = nn.Linear(feat_dim, num_classes)
    elif method == "hedl":
        head = HEDLHead(feat_dim=feat_dim, num_classes=num_classes, W=HEDL_W)
    elif method == "reedl":
        head = EDLHead(feat_dim=feat_dim, num_classes=num_classes,
                       lamb=REEDL_LAMB, evidence_fn=REEDL_EVIDENCE)
    elif method == "rsnn":
        head = RSNNHead(feat_dim=feat_dim, num_classes=num_classes,
                        new_classes_path=RSNN_NEW_CLASSES)
    elif method == "iedl":
        from heads.iedl_head import IEDLHead
        head = IEDLHead(feat_dim=feat_dim, num_classes=num_classes,
                        lamb=getattr(_cfg, "IEDL_LAMB", 0.8),
                        fisher_c=getattr(_cfg, "IEDL_FISHER_C", 1.0),
                        kl_c=getattr(_cfg, "IEDL_KL_C", -1.0))
    elif method == "duq":
        from heads.duq_head import DUQHead
        head = DUQHead(feat_dim=feat_dim, num_classes=num_classes,
                       lengthscale=getattr(_cfg, "DUQ_LENGTHSCALE", 0.3),
                       gamma=getattr(_cfg, "DUQ_GAMMA", 0.999),
                       epsilon=getattr(_cfg, "DUQ_EPSILON", 0.05),
                       norm_features=bool(getattr(_cfg, "DUQ_NORM_FEATURES", True)))
    elif method == "mcdropout":
        from heads.mcdropout_head import MCDropoutHead
        head = MCDropoutHead(feat_dim=feat_dim, num_classes=num_classes,
                             p=getattr(_cfg, "MCD_DROPOUT_P", 0.1))
    else:
        raise ValueError(f"unknown method: {method}")
    return BackboneWrapper(backbone, head)
