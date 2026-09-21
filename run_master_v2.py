#!/usr/bin/env python
"""总编排器 v2（7 方法 × 4 嫁接设置；实数 + 复数 6 面板）。

任务队列（4 个 worker，各绑一张 GPU：1/2/3/4）：
  Phase A  复数补齐：reedl/hedl/rsnn × 4 设置 × **全部 6 面板**，skip 已有
           （v2.1 修正：原先只写 NewFUSAR/RetinalOCT，漏掉 grafted 三设置下
             OCTDL/BreastMRI/MCND/SARAircraft 的 3 头 × 5 种子，共 180 次训练）
  Phase B1 实数：softmax(带分数)/iedl/duq/mcdropout × 4 设置，skip 已有
  Phase B2 复数：同上 4 方法 × 4 设置 × 6 面板，skip 已有
  Phase C  Deep Ensembles 合成（实数 4 目录 + 复数 4 根 × 6 面板）
  Phase D  出两张表 combine_tables.py --real / --plural

SAVE_SCORES 仅对 softmax 打开（Deep Ensembles 的成员是 softmax 头，
其余方法不需要概率矩阵；若对它们打开会多占评估时间且失败即重跑）。

用法: setsid nohup python run_master_v2.py > /tmp/opencode/master_v2.log 2>&1 &
断点续跑：直接重启即可（skip 逻辑按 json+分数文件存在性判断）。
"""
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = "/home/shixisheng/miniforge3/envs/env_pt/bin/python"
LOG_DIR = "/tmp/opencode/v2_logs"
GPUS = ["1", "2", "3", "4"]
SEEDS = ["1000", "1001", "1002", "1003", "1004"]
SPECS = ["", "swa50", "hyena50", "mamba2_50"]
PANELS = ["OCTDL", "BreastMRI", "MCND", "SARAircraft", "NewFUSAR", "RetinalOCT"]
HEADS = ["reedl", "hedl", "rsnn"]
NEW_METHODS = ["softmax", "iedl", "duq", "mcdropout"]
REAL_OODS = ["SVHN", "DTD", "Places365"]

os.makedirs(LOG_DIR, exist_ok=True)


def plural_root(spec):
    return os.path.join(ROOT, "results_plural" if not spec else f"results_plural_g_{spec}")


def real_root(spec):
    return os.path.join(ROOT, "results" if not spec else f"results_g_{spec}")


def plural_oods(panel):
    import config_plural as C
    dom = C.PANELS[panel]["domain"]
    return [k for k in C.PANEL_ORDER if C.PANELS[k]["domain"] != dom]


def build_jobs():
    jobs = []  # (name, cmd, env, cwd)

    # ---- Phase A: 复数补齐（3 头 × 4 设置 × 全部 6 面板）----
    for spec in SPECS:
        for head in HEADS:
            jobs.append((
                f"A_catchup_{spec or 'pre'}_{head}",
                [PY, "run_experiment_plural.py", "--panels", *PANELS,
                 "--methods", head, "--seeds", *SEEDS, "--gpu", "0"],
                {"GRAFT": spec, "SAVE_SCORES": "0"},
            ))

    # ---- Phase B1: 实数（4 新方法 × 4 设置）----
    for spec in SPECS:
        for m in NEW_METHODS:
            jobs.append((
                f"B1_real_{spec or 'pre'}_{m}",
                [PY, "run_experiment.py", "--methods", m, "--seeds", *SEEDS, "--gpu", "0"],
                {"GRAFT": spec, "SAVE_SCORES": "1" if m == "softmax" else "0"},
            ))

    # ---- Phase B2: 复数（4 新方法 × 4 设置 × 6 面板）----
    for spec in SPECS:
        for m in NEW_METHODS:
            jobs.append((
                f"B2_plural_{spec or 'pre'}_{m}",
                [PY, "run_experiment_plural.py", "--methods", m,
                 "--seeds", *SEEDS, "--gpu", "0"],
                {"GRAFT": spec, "SAVE_SCORES": "1" if m == "softmax" else "0"},
            ))

    # ---- Phase C: Deep Ensembles 合成 ----
    for spec in SPECS:
        tag = spec or "pre"
        jobs.append((
            f"C_de_real_{tag}",
            [PY, "make_de_results.py", "--results_dir", real_root(spec),
             "--oods", *REAL_OODS, "--seeds", *SEEDS],
            {"GRAFT": spec},
        ))
        for panel in PANELS:
            jobs.append((
                f"C_de_plural_{tag}_{panel}",
                [PY, "make_de_results.py",
                 "--results_dir", os.path.join(plural_root(spec), f"panel_{panel}"),
                 "--oods", *plural_oods(panel), "--seeds", *SEEDS],
                {"GRAFT": spec},
            ))

    # ---- Phase D: 两张大表 ----
    jobs.append((
        "D_combine_real",
        [PY, "combine_tables.py", "--real"], {},
    ))
    jobs.append((
        "D_combine_plural",
        [PY, "combine_tables.py", "--plural"], {},
    ))
    return jobs


gpu_q = queue.Queue()
for g in GPUS:
    gpu_q.put(g)


def run_job(name, cmd, env):
    gpu = gpu_q.get()
    try:
        log = os.path.join(LOG_DIR, f"{name}.log")
        full_env = dict(os.environ)
        full_env.update(env)
        full_env["CUDA_VISIBLE_DEVICES"] = gpu
        full_env["HF_HUB_OFFLINE"] = "1"
        full_env["TRANSFORMERS_OFFLINE"] = "1"
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] START {name} (GPU{gpu})",
              flush=True)
        with open(log, "w") as lf:
            r = subprocess.run(cmd, cwd=ROOT, env=full_env,
                               stdout=lf, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 3600
        status = "OK" if r.returncode == 0 else f"FAIL(rc={r.returncode})"
        print(f"[{time.strftime('%H:%M:%S')}] END   {name} [{status}] "
              f"({dt:.2f} h)", flush=True)
        return r.returncode == 0
    finally:
        gpu_q.put(gpu)


def worker():
    while True:
        try:
            name, cmd, env = job_q.get_nowait()
        except queue.Empty:
            return
        ok = run_job(name, cmd, env)
        if not ok:
            with open("/tmp/opencode/v2_failures.log", "a") as f:
                f.write(f"{name}\n")


job_q = queue.Queue()
for j in build_jobs():
    job_q.put(j)

if __name__ == "__main__":
    n_jobs = job_q.qsize()
    print(f"==== master v2: {n_jobs} jobs, {len(GPUS)} GPUs {GPUS} ====", flush=True)
    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"==== ALL DONE in {(time.time() - t0) / 3600:.2f} h ====", flush=True)
    with open("/tmp/opencode/pipeline_done.log", "a") as f:
        f.write(f"MASTER V2 COMPLETE {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
