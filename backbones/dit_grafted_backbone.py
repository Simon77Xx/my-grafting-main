"""嫁接后 DiT 骨干：用 grafting-main 的 ReplacementFactory 把指定 MHA 块替换成
替代算子（hyena_x / swa / mamba2），再加载 HF 嫁接 checkpoint 的 ema 权重。

与 grafting-main/src/graft.py 的 graft_dit() 等价，但：
  - 适配本项目的 DiTBackbone（vae_latent 32x32、固定 conditioning、池化特征）；
  - 嫁接权重 = 官方 stage2 微调后的完整 ckpt（model.pt → ['ema']），strict 加载
    到替换后的模型，保证算子权重与论文一致。

用法：环境变量 GRAFT=hyena50|swa50|mamba2_50 时，backbone.build_model 自动走这里。
"""
import os
import sys

import torch

from config import (GRAFT, GRAFT_SPECS, GRAFT_INDEXES, GRAFTING_SRC,
                    DIT_CKPT_PATH)

GRAFTING_ROOT = os.path.dirname(GRAFTING_SRC)
for _p in (GRAFTING_SRC, os.path.join(GRAFTING_SRC, "operators"),
           os.path.join(GRAFTING_SRC, "operators", "hyena")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CKPT_DIR = os.path.dirname(DIT_CKPT_PATH)


def build_grafted_dit_backbone(cfg):
    from backbones.dit_backbone import DiTBackbone

    assert GRAFT in GRAFT_SPECS, f"unknown GRAFT spec: {GRAFT}"
    operator_name, operator_config_rel, ckpt_name = GRAFT_SPECS[GRAFT]
    operator_config = os.path.join(GRAFTING_ROOT, operator_config_rel)
    grafted_ckpt = os.path.join(_CKPT_DIR, ckpt_name)
    assert os.path.isfile(grafted_ckpt), f"grafted ckpt missing: {grafted_ckpt}"

    # 1) 先构建不带预训练权重的骨干（结构与 DiTBackbone 完全一致）
    model = DiTBackbone(
        variant=cfg.BACKBONE,
        num_classes=10,
        input_mode=cfg.DIT_INPUT_MODE,
        image_size=cfg.IMAGE_SIZE,
        patch_size=cfg.DIT_PATCH_SIZE,
        ckpt_path=None,                 # 稍后加载嫁接 ckpt
        freeze_blocks=cfg.DIT_FREEZE_BLOCKS,
        num_trainable_blocks=cfg.DIT_NUM_TRAINABLE_BLOCKS,
        pool=cfg.DIT_POOL,
        conditioning=cfg.DIT_CONDITIONING,
    )

    # 2) 替换 MHA 块（与 graft.py 相同的工厂 + swap 逻辑）
    from replacement_factory import ReplacementFactory
    for index in GRAFT_INDEXES:
        factory = ReplacementFactory(operator_name, operator_config)
        operator = factory.operator
        model.dit.blocks[index].attn = operator
    print(f"[Graft] swapped {len(GRAFT_INDEXES)} MHA blocks -> {operator_name} "
          f"({GRAFT})", flush=True)

    # 3) 加载官方嫁接 checkpoint（['ema']）。官方 ckpt 的 y_embedder(1000类) /
    #    final_layer 与本项目 10 类模型形状不同，先按形状过滤（与 DiTBackbone
    #    的 _load_pretrained 策略一致）；其余所有键（含被替换 attn 算子权重）
    #    必须全部被消费，否则视为结构对齐失败。
    obj = torch.load(grafted_ckpt, map_location="cpu")
    sd = obj.get("ema", obj.get("model", obj)) if isinstance(obj, dict) else obj
    own = model.dit.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    dropped = sorted(set(sd) - set(keep))
    dropped_expected = [k for k in dropped
                        if k.startswith(("y_embedder.", "final_layer."))]
    msg = model.dit.load_state_dict(keep, strict=False)
    print(f"[Graft] loaded grafted ckpt {grafted_ckpt}: "
          f"{len(keep)}/{len(sd)} tensors, missing={len(msg.missing_keys)}, "
          f"dropped={len(dropped)} (y_embedder/final_layer={len(dropped_expected)})",
          flush=True)
    assert len(dropped) == len(dropped_expected), \
        f"unexpected dropped keys (structure mismatch): {[k for k in dropped if k not in dropped_expected][:8]}"
    assert len(msg.unexpected_keys) == 0, \
        f"grafted ckpt keys not consumed: {msg.unexpected_keys[:5]}"

    # 4) 冻结策略与 DiTBackbone 一致
    model._apply_freeze(cfg.DIT_FREEZE_BLOCKS, cfg.DIT_NUM_TRAINABLE_BLOCKS)
    return model
