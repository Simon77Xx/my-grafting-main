"""SD-VAE 预编码 Places365 val (256 small) → 32×32×4 latent，存 npy。

复用 encode_vae_latents.py 的协议：AutoencoderKL (sd-vae-ft-ema)，
latent * 0.18215，归一化 [0.5]/[0.5]，Resize256 + CenterCrop256。
输出：vae_latents/places365_test.npy (36500, 4, 32, 32)
用法: python encode_places365.py --gpu 1
"""
import argparse
import os
import time

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from encode_vae_latents import DATA_ROOT, OUT_ROOT, encode_loader

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    vae.enable_slicing()

    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    from places365_data import Places365Val

    ds = Places365Val(transform=tf)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
    z, y = encode_loader(vae, dl, device)
    np.save(f"{OUT_ROOT}/places365_test.npy", z.numpy())
    np.save(f"{OUT_ROOT}/places365_test_y.npy", y.numpy())
    print(f"places365_test {z.shape} done {time.strftime('%H:%M')}", flush=True)
