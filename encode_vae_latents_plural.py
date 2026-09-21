"""复数数据 SD-VAE 预编码：data_plural 9 个数据集 → 32x32x4 latent npy。

与实数管线 encode_vae_latents.py 完全同一编码协议（公平性核心）：
  - AutoencoderKL (stabilityai/sd-vae-ft-ema)
  - latent = vae.encode(x).latent_dist.sample() * 0.18215
  - 输入归一化 [0.5,0.5,0.5]；Resize 256 (bicubic) + CenterCrop 256
  - 灰度/RGBA 图统一 convert("RGB")
ID 训练集额外编码一份水平翻转版本做增强。

划分规则：
  builtin  = 用数据集自带 train/ test/（如 MCND、NewFUSAR、RetinalOCT）
  stratify = 按类分层 90/10 划分（seed=123 固定，所有方法共享同一划分）
划分清单存 data/vae_latents_plural/<key>_split.json 便于复现。

输出（每面板）：
  <key>_train.npy (N,4,32,32) fp32 / <key>_train_y.npy / <key>_train_flip.npy
  <key>_test.npy  / <key>_test_y.npy
支持断点续跑：已存在的 npy 自动跳过。
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config_plural import (PLURAL_DATA_ROOT, LATENT_ROOT, PANELS, PANEL_ORDER,
                           SPLIT_SEED)

VAE_NAME = "stabilityai/sd-vae-ft-ema"
LATENT_SCALE = 0.18215
IMAGE_SIZE = 256
BATCH = 64
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_class_images(root):
    """返回 sorted([(class_name, [img paths...])])，按类目录组织的数据集。"""
    out = []
    for c in sorted(os.listdir(root)):
        cd = os.path.join(root, c)
        if not os.path.isdir(cd):
            continue
        imgs = []
        for r, _, fs in os.walk(cd):
            imgs += [os.path.join(r, f) for f in sorted(fs)
                     if f.lower().endswith(IMG_EXT)]
        if imgs:
            out.append((c, sorted(imgs)))
    return out


def collect_split(key):
    """按 mode 收集 train/test 的 (path, label) 清单，划分可复现。"""
    info = PANELS[key]
    base = os.path.join(PLURAL_DATA_ROOT, info["path"])
    if info["mode"] == "builtin":
        train_root, test_root = os.path.join(base, "train"), os.path.join(base, "test")
        cls_images = list_class_images(train_root)
        classes = [c for c, _ in cls_images]
        cls_dirs = {c: i for i, c in enumerate(classes)}
        train = [(p, cls_dirs[c]) for c, paths in cls_images for p in paths]
        test = [(p, cls_dirs[c]) for c, paths in list_class_images(test_root)
                for p in paths]
    else:  # stratify: 分层 90/10
        cls_images = list_class_images(base)
        classes = [c for c, _ in cls_images]
        rng = random.Random(SPLIT_SEED)
        train, test = [], []
        for ci, (c, imgs) in enumerate(cls_images):
            imgs = sorted(imgs)
            rng.shuffle(imgs)
            n_test = max(1, int(round(len(imgs) * 0.10)))
            test += [(p, ci) for p in imgs[:n_test]]
            train += [(p, ci) for p in imgs[n_test:]]
    return classes, train, test


def load_image(p):
    """PIL 优先；GeoTIFF（float32 SAR，PIL 12 不支持）回落 tifffile → uint8 灰度。"""
    try:
        return Image.open(p)
    except Exception:
        if p.lower().endswith((".tif", ".tiff")):
            import tifffile
            arr = np.asarray(tifffile.imread(p), dtype=np.float32)
            if arr.ndim == 3:
                # shape[-1]∈{2,3,4} 视为通道维（如 OpenSARUrban (100,100,2)），否则视为多页 (P,H,W)
                arr = arr.mean(axis=-1) if arr.shape[-1] <= 4 else arr.mean(axis=0)
            lo, hi = np.percentile(arr, [1.0, 99.5])
            if not np.isfinite(lo) or hi <= lo:
                lo, hi = float(arr.min()), float(arr.max()) + 1e-6
            arr8 = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr8, mode="L")
        raise


class ListDataset(Dataset):
    """(path, label) 清单 + 统一变换。"""

    def __init__(self, items):
        self.items = items
        self.tf = transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        try:
            return self.tf(load_image(p)), y
        except Exception as e:
            print(f"[warn] bad image -> zeros: {p} ({e})", flush=True)
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), y


def encode_items(vae, items, device, flip=False, desc="encode"):
    ds = ListDataset(items)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=8)
    zs, ys = [], []
    for x, y in tqdm(dl, desc=desc):
        if flip:
            x = torch.flip(x, dims=[3])
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.sample().mul_(LATENT_SCALE).cpu()
        zs.append(z)
        ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def encode_split(vae, key, suffix, items, device):
    """编码并保存一个 split（npy 已存在则跳过，断点续跑）。"""
    arr_p = os.path.join(LATENT_ROOT, f"{key}_{suffix}.npy")
    y_p = os.path.join(LATENT_ROOT, f"{key}_{suffix}_y.npy")
    if os.path.isfile(arr_p) and os.path.isfile(y_p):
        print(f"[skip] {arr_p} exists", flush=True)
        return
    z, y = encode_items(vae, items, device, desc=f"{key}_{suffix}")
    np.save(arr_p, z)
    np.save(y_p, y)
    print(f"saved {arr_p} {z.shape} {time.strftime('%H:%M')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--panels", nargs="*", default=None, help="只编码指定面板（默认全部）")
    args = ap.parse_args()

    os.makedirs(LATENT_ROOT, exist_ok=True)
    device = f"cuda:{args.gpu}"
    print(f"loading VAE {VAE_NAME} on {device} ...", flush=True)
    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()
    vae.enable_slicing()

    keys = args.panels or PANEL_ORDER
    for key in keys:
        t0 = time.time()
        classes, train, test = collect_split(key)
        print(f"===== {key}: train={len(train)} test={len(test)} "
              f"classes={classes}", flush=True)
        with open(os.path.join(LATENT_ROOT, f"{key}_split.json"), "w") as f:
            json.dump({"classes": classes, "mode": PANELS[key]["mode"],
                       "n_train": len(train), "n_test": len(test),
                       "split_seed": SPLIT_SEED}, f, indent=2)
        encode_split(vae, key, "train", train, device)
        arr_p = os.path.join(LATENT_ROOT, f"{key}_train_flip.npy")
        if os.path.isfile(arr_p):
            print(f"[skip] {arr_p} exists", flush=True)
        else:
            z, _ = encode_items(vae, train, device, flip=True, desc=f"{key}_flip")
            np.save(arr_p, z)
            print(f"saved {arr_p} {z.shape} {time.strftime('%H:%M')}", flush=True)
        encode_split(vae, key, "test", test, device)
        print(f"===== {key} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
