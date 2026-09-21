"""Diffusion Transformer (DiT) used as a classification backbone.

Re-uses the pretrained DiT from <research>/grafting-main as a feature
extractor: we run the patch-embed + transformer-block stack of DiT, pool the
patch tokens, and expose a (N, D) feature vector. The three uncertainty heads
(HEDL / Re-EDL / RS-NN) plus the softmax baseline then attach on top of these
features, exactly as they did on ResNet18, so trainer.py / evaluate.py /
make_table.py are reused unchanged.

This mirrors how grafting-main's graft.py replaces a submodule of a pretrained
DiT: here the "graft point" is the end of the transformer stack and the
denoising FinalLayer is the submodule being replaced by an uncertainty head.
"""

import os
import sys

import torch
import torch.nn as nn

# Make grafting-main/src importable so `models.dit` resolves like in graft.py
_HERE = os.path.dirname(os.path.abspath(__file__))
GRAFTING_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "grafting-main", "src"))
if GRAFTING_SRC not in sys.path:
    sys.path.insert(0, GRAFTING_SRC)

from models.dit import DiT_models  # noqa: E402


# hidden_size per DiT variant (feat_dim of the pooled features)
DIT_HIDDEN = {
    "DiT-XL/2": 1152,
    "DiT-L/2": 1024,
    "DiT-B/2": 768,
    "DiT-S/2": 384,
}


class DiTBackbone(nn.Module):
    """Pretrained DiT run as an unconditional feature extractor.

    forward(x) returns pooled (N, D) features; freezing and conditioning are
    configurable to trade off compute against how much DiT adapts.
    """

    def __init__(
        self,
        variant="DiT-B/2",
        num_classes=10,
        input_mode="rgb",
        image_size=32,
        patch_size=2,
        ckpt_path=None,
        freeze_blocks=False,
        num_trainable_blocks=0,
        pool="mean",
        conditioning="zero",
        class_dropout_prob=0.0,
    ):
        super().__init__()
        assert variant in DiT_models, f"unknown DiT variant: {variant}"
        assert variant in DIT_HIDDEN, f"no feat_dim registered for {variant}"
        self.variant = variant
        self.input_mode = input_mode
        self.pool = pool
        self.conditioning = conditioning
        self.feat_dim = DIT_HIDDEN[variant]

        in_ch = 4 if input_mode == "vae_latent" else 3
        # DiT_models[variant] 工厂已把 patch_size 硬编码（如 DiT_XL_2 传 patch_size=2），
        # 不能再在 kwargs 里重复传 patch_size，否则 TypeError: multiple values。
        self.dit = DiT_models[variant](
            input_size=image_size,
            in_channels=in_ch,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=True,
        )

        # Optional ImageNet-class-conditional DiT pretrained weights
        # (https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) or any
        # grafted checkpoint from grafting-main. RGB / 10-class-mode shape
        # mismatches (patch conv, y_embedder, final_layer) are silently skipped
        # so the transformer-block weights are still restored.
        self._load_pretrained(ckpt_path)

        # Freeze blocks if requested, but keep the patch embedding trainable so
        # an RGB input distribution can be (re)learned.
        self._apply_freeze(freeze_blocks, num_trainable_blocks)

        # DiT is class-conditional via AdaLN. Using it as an *unconditional*
        # backbone means we must remove the label/timestep signal: we pin
        # c = t_embedder(0) + y_embedder(0) once and broadcast it, so the blocks
        # operate as a deterministic, label-free feature transform.
        self._build_fixed_conditioning()

        # RGB 模式下 patch_embed 随机初始化 + 固定 conditioning，会让 28 个预训练
        # block 输出的 pooled 特征量级极大（L2~1e4），直接进分类头 loss 爆炸。
        # 池化后加 LayerNorm 把特征量级校准到 O(1)，与 ViT linear-probe 常见做法一致。
        self._pool_ln = nn.LayerNorm(self.feat_dim)

    # ---- weights ---------------------------------------------------------------
    def _load_pretrained(self, ckpt_path):
        if ckpt_path is None:
            return
        sd = None
        if ckpt_path in {"DiT-XL-2-256x256.pt", "DiT-XL-2-512x512.pt"}:
            from models.download import find_model
            sd = find_model(ckpt_path)
        elif os.path.isfile(ckpt_path):
            obj = torch.load(ckpt_path, map_location="cpu")
            if isinstance(obj, dict):
                sd = obj.get("ema", obj.get("model", obj))
            else:
                sd = obj
        if not isinstance(sd, dict):
            print(f"[DiTBackbone] warning: could not load checkpoint {ckpt_path}")
            return
        own = self.dit.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        msg = self.dit.load_state_dict(keep, strict=False)
        print(
            f"[DiTBackbone] loaded {len(keep)}/{len(own)} tensors from {ckpt_path}; "
            f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
        )

    def _apply_freeze(self, freeze_blocks, num_trainable_blocks):
        for p in self.dit.x_embedder.parameters():
            p.requires_grad = True
        if freeze_blocks:
            for blk in self.dit.blocks:
                for p in blk.parameters():
                    p.requires_grad = False
            if num_trainable_blocks > 0:
                for blk in self.dit.blocks[-num_trainable_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
        else:
            for p in self.dit.parameters():
                p.requires_grad = True

    def _build_fixed_conditioning(self):
        with torch.no_grad():
            t = self.dit.t_embedder(torch.zeros(1, device="cpu"))
            y = self.dit.y_embedder(
                torch.zeros(1, dtype=torch.long, device="cpu"), False
            )
            c = (t + y).squeeze(0)  # (hidden_size,)
        self.register_buffer("fixed_c", c.clone(), persistent=False)

    # ---- forward ---------------------------------------------------------------
    def forward_features(self, x):
        # x: (N, C, H, W). Runs DiT up to but NOT including FinalLayer, so we
        # discard the denoising head and treat the patch tokens as features.
        tokens = self.dit.x_embedder(x) + self.dit.pos_embed  # (N, T, D)
        N = tokens.shape[0]
        c = self.fixed_c.unsqueeze(0).expand(N, -1)  # (N, D)
        for blk in self.dit.blocks:
            tokens = blk(tokens, c)
        if self.pool == "cls":
            return self._pool_ln(tokens[:, 0])
        return self._pool_ln(tokens.mean(dim=1))  # mean over patch tokens -> (N, D)

    def forward(self, x):
        return self.forward_features(x)


def get_dit_backbone(cfg):
    return DiTBackbone(
        variant=cfg.BACKBONE,
        num_classes=10,
        input_mode=cfg.DIT_INPUT_MODE,
        image_size=cfg.IMAGE_SIZE,
        patch_size=cfg.DIT_PATCH_SIZE,
        ckpt_path=cfg.DIT_CKPT_PATH,
        freeze_blocks=cfg.DIT_FREEZE_BLOCKS,
        num_trainable_blocks=cfg.DIT_NUM_TRAINABLE_BLOCKS,
        pool=cfg.DIT_POOL,
        conditioning=cfg.DIT_CONDITIONING,
    )
