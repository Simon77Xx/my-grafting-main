"""为复数 9 个面板生成 RS-NN focal 集文件（整数索引格式）。

风格与原始 RS-NN 仓库 new_classes.npy（CIFAR10: 10 单例 + 20 随机子集，共 30）
一致：n 个单例 + 2n 个随机子集（size2:75% / size3:25%），固定 seed=SPLIT_SEED，
所有方法/种子共享同一 focal 集。
"""
import os

import numpy as np

from config_plural import PANELS, PANEL_ORDER, FOCAL_ROOT, SPLIT_SEED


def gen_focal(n, rng):
    """n 个单例 + 至多 2n 个随机子集（size2:75% / size3:25%）。

    n 小（如 n=2）时可能的子集总数有限，目标数按可用子集数封顶，
    避免死循环。风格与原始 RS-NN 仓库 new_classes.npy 一致。
    """
    focal = [frozenset([c]) for c in range(n)]
    seen = set(focal)
    # 所有可能的真子集（不含单例与空集/全集）
    candidates = [frozenset(s) for r in range(2, n)
                  for s in _comb(range(n), r)]
    if not candidates:
        return focal
    rng.shuffle(candidates)
    target = min(n * 3, len(focal) + len(candidates))
    for s in candidates:
        if len(focal) >= target:
            break
        focal.append(s)
    return focal


def _comb(iterable, r):
    """itertools.combinations 返回 list。"""
    from itertools import combinations
    return list(combinations(iterable, r))


def main():
    os.makedirs(FOCAL_ROOT, exist_ok=True)
    for key in PANEL_ORDER:
        n = PANELS[key]["n"]
        rng = np.random.RandomState(SPLIT_SEED)
        focal = gen_focal(n, rng)
        out = np.array([set(map(int, s)) for s in focal], dtype=object)
        path = os.path.join(FOCAL_ROOT, f"{key}_new_classes.npy")
        np.save(path, out, allow_pickle=True)
        print(f"{key}: n_classes={n} focal_sets={len(focal)} -> {path}")


if __name__ == "__main__":
    main()
