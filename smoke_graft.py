"""嫁接骨干冒烟自检：构建替换后的 DiT 骨干，4 个头各跑一次 前向+反向+OOD预测。

用法: GRAFT=hyena50 python smoke_graft.py [--gpu 0] [--methods softmax hedl reedl rsnn]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, Subset

import config as cfg
from backbone import build_model
from latent_data import get_latent_id_dataset
from trainer import STEP_FN, predict_uncertainty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--methods", nargs="*", default=cfg.METHODS)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    assert cfg.GRAFT, "need GRAFT env, e.g. GRAFT=hyena50"
    print(f"[smoke] GRAFT={cfg.GRAFT} RESULTS_DIR={cfg.RESULTS_DIR}", flush=True)

    device = "cuda"
    ds = Subset(get_latent_id_dataset(train=False), range(args.batch * 2))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    for method in args.methods:
        torch.manual_seed(0)
        model = build_model(method, num_classes=10).to(device)
        # 训练语义：骨干走 bf16 autocast（flash MHA 要求 fp16/bf16 输入）
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = STEP_FN[method](model, x, y, device)
        assert torch.isfinite(loss), f"{method} loss not finite"
        loss.backward()
        preds, unc = predict_uncertainty(model, method, loader, device)
        assert torch.isfinite(unc).all(), f"{method} unc not finite"
        print(f"[smoke] {method}: loss={loss.item():.4f} unc_mean={unc.mean():.4f} "
              f"preds={preds[:5].tolist()} -> PASS", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"[smoke] ALL PASS {cfg.GRAFT}", flush=True)


if __name__ == "__main__":
    main()
