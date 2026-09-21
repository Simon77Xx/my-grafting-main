"""复数管线数据加载：从预编码 SD-VAE latent 读 9 个面板数据。

latent 存放于 config_plural.LATENT_ROOT，文件命名：
  <key>_train.npy / <key>_train_y.npy / <key>_train_flip.npy   (ID 训练集)
  <key>_test.npy  / <key>_test_y.npy                           (ID 测试集，同时作为他面板的 OOD)
"""
import os

import torch

from config_plural import LATENT_ROOT, PANELS, PANEL_ORDER, ood_panels
from latent_data import LatentDataset


def get_panel_dataset(key, train=True):
    """返回某面板 train/test latent 数据集。"""
    if train:
        return LatentDataset(
            os.path.join(LATENT_ROOT, f"{key}_train.npy"),
            label_path=os.path.join(LATENT_ROOT, f"{key}_train_y.npy"),
            flip_path=os.path.join(LATENT_ROOT, f"{key}_train_flip.npy"),
            train=True,
        )
    return LatentDataset(
        os.path.join(LATENT_ROOT, f"{key}_test.npy"),
        label_path=os.path.join(LATENT_ROOT, f"{key}_test_y.npy"),
    )


def get_panel_ood_datasets(panel_key):
    """返回 {ood_key: LatentDataset}：其他两域的 6 个数据集（各自 test latent）。"""
    return {k: get_panel_dataset(k, train=False) for k in ood_panels(panel_key)}
