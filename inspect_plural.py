"""只读检查 data_plural 9 个数据集的目录结构与类别数。"""
import os

BASE = "/supcon3/shixisheng/xjx/research/data_plural"
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def n_img(d):
    n = 0
    for r, ds, fs in os.walk(d):
        n += len([f for f in fs if f.lower().endswith(IMG_EXT)])
    return n


def tree(d, depth=0, maxd=2):
    if depth > maxd:
        return
    try:
        entries = sorted(os.listdir(d))
    except Exception as e:
        print("  " * depth + f"[err {e}]")
        return
    for e in entries[:15]:
        p = os.path.join(d, e)
        if os.path.isdir(p):
            print("  " * depth + f"{e}/ ({n_img(p)} imgs)")
            if depth < maxd:
                tree(p, depth + 1, maxd)


for dom in ["MRI", "OCT", "SAR"]:
    for ds in sorted(os.listdir(os.path.join(BASE, dom))):
        p = os.path.join(BASE, dom, ds)
        if os.path.isdir(p):
            print(f"===== {dom}/{ds} ===== total {n_img(p)} imgs")
            tree(p)
            print()
