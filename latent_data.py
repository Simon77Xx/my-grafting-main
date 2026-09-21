"""vae_latent 模式数据加载：从预编码的 SD-VAE latent (32,32,4) 读数据。

与 data.py 的 RGB 数据加载并存：vae_latent 时返回 latent 张量 (N,4,32,32)。
增强：训练集在 latent 空间做水平翻转（原始 + 翻转副本随机抽一份）。
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_ROOT = "/supcon3/shixisheng/xjx/research/data"
VAE_LATENT_ROOT = os.path.join(DATA_ROOT, "vae_latents")


class LatentDataset(Dataset):
    """从 npy 读 latent，可选翻转副本（增强用）。"""

    def __init__(self, arr_path, label_path=None, flip_path=None, train=False):
        self.x = torch.from_numpy(np.load(arr_path))
        self.y = None
        if label_path is not None:
            self.y = torch.from_numpy(np.load(label_path))
        self.flip_x = None
        if flip_path is not None:
            self.flip_x = torch.from_numpy(np.load(flip_path))
        self.train = train

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        if self.train and self.flip_x is not None and torch.rand(1).item() < 0.5:
            x = self.flip_x[i]
        else:
            x = self.x[i]
        if self.y is not None:
            return x, self.y[i]
        return x


def get_latent_id_dataset(train=True):
    if train:
        return LatentDataset(
            f"{VAE_LATENT_ROOT}/cifar_train.npy",
            label_path=f"{VAE_LATENT_ROOT}/cifar_train_y.npy",
            flip_path=f"{VAE_LATENT_ROOT}/cifar_train_flip.npy",
            train=True,
        )
    return LatentDataset(f"{VAE_LATENT_ROOT}/cifar_test.npy", label_path=f"{VAE_LATENT_ROOT}/cifar_test_y.npy")


def get_latent_ood_dataset(name):
    if name == "SVHN":
        return LatentDataset(f"{VAE_LATENT_ROOT}/svhn_test.npy", label_path=f"{VAE_LATENT_ROOT}/svhn_test_y.npy")
    if name == "DTD":
        return LatentDataset(f"{VAE_LATENT_ROOT}/dtd_test.npy", label_path=f"{VAE_LATENT_ROOT}/dtd_test_y.npy")
    if name == "Places365":
        return LatentDataset(f"{VAE_LATENT_ROOT}/places365_test.npy",
                             label_path=f"{VAE_LATENT_ROOT}/places365_test_y.npy")
    raise ValueError(f"unknown ood: {name}")
