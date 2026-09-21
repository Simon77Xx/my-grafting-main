"""SD-VAE 预编码 CIFAR-10 / SVHN / DTD → 32×32×4 latent，存 npy。

方案 B DiT vae_latent 模式。与 grafting-main extract_vae_features.py 一致：
  - AutoencoderKL (stabilityai/sd-vae-ft-ema)
  - latent = vae.encode(x).latent_dist.sample() * 0.18215
  - 输入归一化 [0.5,0.5,0.5]/[0.5,0.5,0.5]
  - Resize 256 + 中心裁剪 256（CIFAR 32×32 放大到 256，VAE 编码后得 32×32×4
    latent，正好是 DiT-XL-2 原生输入：32/2=16 → 256 个 patch token）
训练集（CIFAR）额外编码一份水平翻转版本做增强。

输出（DATA_ROOT/vae_latents/）：
  cifar_train.npy / cifar_train_y.npy    (50000, 4, 32, 32)  原图
  cifar_train_flip.npy                   (50000, 4, 32, 32)  水平翻转
  cifar_test.npy  / cifar_test_y.npy     (10000, 4, 32, 32)
  svhn_test.npy   / svhn_test_y.npy
  dtd_test.npy    / dtd_test_y.npy
"""
import os
import time

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

DATA_ROOT = "/supcon3/shixisheng/xjx/research/data"
OUT_ROOT = os.path.join(DATA_ROOT, "vae_latents")
VAE_NAME = "stabilityai/sd-vae-ft-ema"
LATENT_SCALE = 0.18215
IMAGE_SIZE = 256  # 编码分辨率；VAE 输出 /8 → 32×32 latent
BATCH = 64


def center_crop(x, size):
    h, w = x.shape[-2], x.shape[-1]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return x[:, y0:y0 + size, x0:x0 + size]


def encode_loader(vae, loader, device, flip=False):
    latents, labels = [], []
    for x, y in tqdm(loader, desc="encode"):
        if flip:
            x = torch.flip(x, dims=[3])
        x = x.to(device)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.sample().mul_(LATENT_SCALE).cpu()
        latents.append(z)
        labels.append(y)
    return torch.cat(latents), torch.cat(labels)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    device = "cuda:1"
    print(f"loading VAE {VAE_NAME} ...", flush=True)
    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()
    vae.enable_slicing()

    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    # CIFAR-10 train (original + horizontal flip)
    cifar_train = datasets.CIFAR10(root=f"{DATA_ROOT}/cifar-10-python", train=True, download=False, transform=tf)
    dl = DataLoader(cifar_train, batch_size=BATCH, shuffle=False, num_workers=8)
    z, y = encode_loader(vae, dl, device)
    np.save(f"{OUT_ROOT}/cifar_train.npy", z.numpy()); np.save(f"{OUT_ROOT}/cifar_train_y.npy", y.numpy())
    print(f"cifar_train {z.shape} done {time.strftime('%H:%M')}", flush=True)
    dl = DataLoader(cifar_train, batch_size=BATCH, shuffle=False, num_workers=8)
    z, _ = encode_loader(vae, dl, device, flip=True)
    np.save(f"{OUT_ROOT}/cifar_train_flip.npy", z.numpy())
    print(f"cifar_train_flip {z.shape} done {time.strftime('%H:%M')}", flush=True)

    # CIFAR-10 test
    cifar_test = datasets.CIFAR10(root=f"{DATA_ROOT}/cifar-10-python", train=False, download=False, transform=tf)
    dl = DataLoader(cifar_test, batch_size=BATCH, shuffle=False, num_workers=8)
    z, y = encode_loader(vae, dl, device)
    np.save(f"{OUT_ROOT}/cifar_test.npy", z.numpy()); np.save(f"{OUT_ROOT}/cifar_test_y.npy", y.numpy())
    print(f"cifar_test {z.shape} done {time.strftime('%H:%M')}", flush=True)

    # SVHN test
    svhn = datasets.SVHN(root=f"{DATA_ROOT}/SVHN", split="test", download=False, transform=tf)
    dl = DataLoader(svhn, batch_size=BATCH, shuffle=False, num_workers=8)
    z, y = encode_loader(vae, dl, device)
    np.save(f"{OUT_ROOT}/svhn_test.npy", z.numpy()); np.save(f"{OUT_ROOT}/svhn_test_y.npy", y.numpy())
    print(f"svhn_test {z.shape} done {time.strftime('%H:%M')}", flush=True)

    # DTD test
    dtd = datasets.DTD(root=f"{DATA_ROOT}", split="test", download=False, transform=tf)
    dl = DataLoader(dtd, batch_size=BATCH, shuffle=False, num_workers=8)
    z, y = encode_loader(vae, dl, device)
    np.save(f"{OUT_ROOT}/dtd_test.npy", z.numpy()); np.save(f"{OUT_ROOT}/dtd_test_y.npy", y.numpy())
    print(f"dtd_test {z.shape} done {time.strftime('%H:%M')}", flush=True)


if __name__ == "__main__":
    main()