"""最小复现 mamba2 嫁接骨干在 bf16 autocast + batch128 下 triton 崩溃的问题."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, Subset

import config as cfg
from backbone import build_model
from latent_data import get_latent_id_dataset
from trainer import STEP_FN

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--backward", action="store_true")
ap.add_argument("--no-amp", action="store_true")
args = ap.parse_args()

print(f"[repro] GRAFT={cfg.GRAFT} batch={args.batch} amp={not args.no_amp} backward={args.backward}", flush=True)

device = "cuda"
ds = Subset(get_latent_id_dataset(train=False), range(args.batch))
loader = DataLoader(ds, batch_size=args.batch, shuffle=False)
x, y = next(iter(loader))
x, y = x.to(device), y.to(device)

model = build_model("softmax", num_classes=10).to(device)
model.train()
ctx = torch.autocast("cuda", dtype=torch.bfloat16) if not args.no_amp else torch.autocast("cuda", enabled=False)
with ctx:
    loss = STEP_FN["softmax"](model, x, y, device)
print(f"[repro] fwd loss={loss.item():.4f} ok", flush=True)
if args.backward:
    loss.backward()
    print("[repro] backward ok", flush=True)
print("[repro] PASS", flush=True)
