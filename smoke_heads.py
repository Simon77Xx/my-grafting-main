"""DiT 骨干下 HEDL / RS-NN 头冒烟自检：少量 batch 内 loss 无 NaN 且下降。

用法: python smoke_heads.py --methods hedl rsnn --steps 40 --gpu 1
"""
import argparse

import torch
from torch.utils.data import DataLoader, Subset

import config
from backbone import build_model
from data import get_id_dataset
from trainer import STEP_FN, predict_uncertainty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=["hedl", "rsnn"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(0)

    full = get_id_dataset(train=True)
    subset = Subset(full, list(range(args.batch * (args.steps + 4))))
    loader = DataLoader(subset, batch_size=args.batch, shuffle=True, num_workers=0)

    for method in args.methods:
        print(f"\n===== smoke: {method} =====", flush=True)
        torch.manual_seed(0)
        model = build_model(method, num_classes=10).to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        step_fn = STEP_FN[method]

        losses = []
        nan_count = 0
        it = iter(loader)
        model.train()
        for step in range(args.steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = step_fn(model, x, y, device)
            if not torch.isfinite(loss):
                nan_count += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            losses.append(loss.item())
            if (step + 1) % 10 == 0:
                print(f"  step {step+1}: loss {loss.item():.4f}", flush=True)

        # 评估路径自检（ID 测试集前 512 张）
        test = Subset(get_id_dataset(train=False), list(range(512)))
        tl = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)
        preds, unc = predict_uncertainty(model, method, tl, device)
        finite_unc = torch.isfinite(unc).all().item()
        acc = (preds == torch.tensor([y for _, y in test])).float().mean().item() * 100

        first, last = losses[0], sum(losses[-5:]) / 5
        status = "PASS" if (losses and finite_unc) else "FAIL"
        print(f"  [{method}] loss {first:.4f} -> {last:.4f} | nan_batches={nan_count} "
              f"| unc_finite={finite_unc} | smoke_acc={acc:.2f}% => {status}", flush=True)


if __name__ == "__main__":
    main()