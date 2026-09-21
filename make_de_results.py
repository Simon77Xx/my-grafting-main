"""Deep Ensembles 结果合成（Lakshminarayanan et al., NeurIPS 2017）。

集成成员 = 同一设置（同骨干）下 softmax 头的 5 个种子模型；
  集成概率 = 成员 softmax 概率的均值；
  预测     = argmax(集成概率)；
  不确定性 = predictive entropy（越大越 OOD）。

读取 <results_root>[/panel_<key>]/scores/softmax_s{seed}_scores.npz，
输出 deepens.json（单值无 std，combine_tables 会按单值渲染）。

用法:
  python make_de_results.py --results_dir results --oods SVHN DTD Places365
  python make_de_results.py --results_dir results_plural_g_swa50/panel_OCTDL \
      --oods BreastMRI MCND SARAircraft NewFUSAR
"""
import argparse
import json
import os

import numpy as np

from evaluate import compute_ood_metrics, compute_acc


def load_probs(results_dir, seed):
    p = os.path.join(results_dir, "scores", f"softmax_s{seed}_scores.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--oods", nargs="*", required=True)
    ap.add_argument("--seeds", nargs="*", type=int,
                    default=[1000, 1001, 1002, 1003, 1004])
    args = ap.parse_args()

    parts = [load_probs(args.results_dir, s) for s in args.seeds]

    id_probs = np.mean([p["id_probs"] for p in parts], axis=0)
    id_labels = parts[0]["id_labels"]
    id_unc = -(id_probs * np.log(id_probs.clip(min=1e-12))).sum(-1)
    preds = id_probs.argmax(-1)
    acc = compute_acc(preds, id_labels)

    result = {"method": "deepens", "acc": acc,
              "members": [int(s) for s in args.seeds]}
    for ood in args.oods:
        key = f"ood_{ood}"
        ood_probs = np.mean([p[key] for p in parts], axis=0)
        ood_unc = -(ood_probs * np.log(ood_probs.clip(min=1e-12))).sum(-1)
        fpr95, auroc, aupr = compute_ood_metrics(id_unc, ood_unc)
        result[ood] = {"fpr95": fpr95, "auroc": auroc, "aupr": aupr}
        print(f"  deepens {ood}: FPR95 {fpr95:.2f} AUROC {auroc:.2f} AUPR {aupr:.2f}",
              flush=True)

    out = os.path.join(args.results_dir, "deepens.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
