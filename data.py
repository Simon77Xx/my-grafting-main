"""统一数据加载：CIFAR-10 (ID) + SVHN/DTD (OOD)，与阶段一 HEDL 保持一致。"""
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from config import DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, DIT_INPUT_MODE

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

if str(DIT_INPUT_MODE) == "vae_latent":
    from latent_data import get_latent_id_dataset, get_latent_ood_dataset


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_id_dataset(train=True):
    if str(DIT_INPUT_MODE) == "vae_latent":
        return get_latent_id_dataset(train)
    return datasets.CIFAR10(
        root=DATA_ROOT + "/cifar-10-python",
        train=train,
        download=False,
        transform=get_transforms(train),
    )


def get_ood_dataset(name):
    if str(DIT_INPUT_MODE) == "vae_latent":
        return get_latent_ood_dataset(name)
    if name == "SVHN":
        return datasets.SVHN(
            root=DATA_ROOT + "/SVHN",
            split="test",
            download=False,
            transform=get_transforms(train=False),
        )
    if name == "DTD":
        return datasets.DTD(
            root=DATA_ROOT,
            split="test",
            download=False,
            transform=get_transforms(train=False),
        )
    if name == "Places365":
        from places365_data import Places365Val
        return Places365Val(transform=get_transforms(train=False))
    raise ValueError(f"unknown ood: {name}")


def get_loader(dataset, train=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=train, num_workers=NUM_WORKERS)


def get_id_test_unc_loader():
    # 评估用 ID 测试集；阶段一 HEDL 用 train=True 当测试，这里用标准 test 集
    return get_loader(get_id_dataset(train=False))
