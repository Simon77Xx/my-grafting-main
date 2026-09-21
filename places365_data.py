"""Places365 val 专用轻量加载器：读取 places365_val.txt + val_256/ 平铺图片。

torchvision 官方 Places365 会校验 places365_val.txt 的 md5（含 ground-truth 标签），
本任务 OOD 检测不需要 OOD 标签，故用本地生成的列表（标签占位 0）绕开校验。
"""
import os

from PIL import Image
from torch.utils.data import Dataset

PLACE365_ROOT = "/supcon3/shixisheng/xjx/research/data/places365"


class Places365Val(Dataset):
    def __init__(self, transform=None, root=PLACE365_ROOT):
        self.transform = transform
        self.img_dir = os.path.join(root, "val_256")
        with open(os.path.join(root, "places365_val.txt")) as f:
            self.files = [line.split()[0] for line in f if line.strip()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.files[i])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
