"""下载 Places365 (places365standard, val 256x256 small 版) 到 DATA_ROOT/places365。"""
import torchvision.datasets as datasets
from config import DATA_ROOT

ds = datasets.Places365(
    root=f"{DATA_ROOT}/places365",
    split="val",
    small=True,
    download=True,
)
print("places365 val images:", len(ds))
