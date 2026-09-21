"""校验所有结果 json 的完整性（能否被 json.load 读取且含必需字段）。

磁盘配额打满时可能产生截断/损坏的 json；combine_tables 只看文件存在与否，
坏档会静默变成表格里的「—」。本脚本找出坏档并删除（删除后流水线续跑会
自动重做对应 seed）。先于 Phase D 运行；流水线运行期间运行也安全（只读，
删除动作仅针对坏档）。

用法: python verify_jsons.py [--fix]
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DIRS = ["results", "results_g_swa50", "results_g_hyena50", "results_g_mamba2_50",
        "results_plural", "results_plural_g_swa50", "results_plural_g_hyena50",
        "results_plural_g_mamba2_50"]


def check(path):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return "unreadable"
    if not isinstance(d, dict) or "acc" not in d:
        return "missing-fields"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true",
                    help="删除坏档（流水线续跑会自动重做）")
    args = ap.parse_args()

    bad = []
    total = 0
    for d in DIRS:
        for p in glob.glob(os.path.join(ROOT, d, "**", "*.json"), recursive=True):
            if os.path.basename(p) == "deepens.json":
                continue  # deepens 由 make_de_results 生成，Phase C 校验
            total += 1
            err = check(p)
            if err:
                bad.append((p, err))
    print(f"checked {total} json files, {len(bad)} bad")
    for p, err in bad:
        print(f"  [{err}] {p}")
        if args.fix:
            os.remove(p)
            print("    removed (will be re-trained on resume)")
    if bad and not args.fix:
        print("run with --fix to remove them")


if __name__ == "__main__":
    main()
