#!/usr/bin/env python3
"""Mixing ratio ablation: vary % of natural data mixed into balanced training.

Tests 0%, 25%, 50%, 75%, 100% natural data ratios with DGCNN.
Uses the same fold splits and evaluates on both protocols.
"""
import subprocess
import sys
import time
import os
import json
import numpy as np
from pathlib import Path

SEEDS = [1337, 2020, 2021]
FOLDS = [0, 1, 2, 3, 4]
RATIOS = [0.0, 0.25, 0.50, 0.75, 1.0]  # fraction of natural data to include
GPUS = [0, 1]  # reserve GPU 3 for PointMLP
EPOCHS = 100
PATIENCE = 20

RUN_ROOT = Path("runs/mix_ablation")
SCRIPT = Path("scripts/phase3_train_raw_seg.py")
BAL_ROOT = Path("processed/raw_seg/v1")
NAT_ROOT = Path("processed/raw_seg/v2_natural")


def main():
    jobs = []
    for ratio in RATIOS:
        for seed in SEEDS:
            for fold in FOLDS:
                tag = f"ratio{int(ratio*100)}"
                exp_name = f"dgcnn_{tag}_s{seed}_f{fold}"
                result_dir = RUN_ROOT / exp_name
                result_file = result_dir / "results.json"
                if result_file.exists():
                    print(f"[SKIP] {exp_name}")
                    continue
                jobs.append({
                    "ratio": ratio, "seed": seed, "fold": fold,
                    "exp_name": exp_name, "tag": tag
                })

    print(f"[LAUNCH] {len(jobs)} jobs on {len(GPUS)} GPUs")
    if not jobs:
        print("All done!")
        return 0

    running = {}
    job_idx = 0
    completed = 0
    failed = 0

    while job_idx < len(jobs) or running:
        for gpu in GPUS:
            if gpu not in running and job_idx < len(jobs):
                job = jobs[job_idx]
                job_idx += 1
                ratio = job["ratio"]

                # Build command based on ratio
                cmd = [
                    sys.executable, str(SCRIPT),
                    "--model", "dgcnn_v2",
                    "--seed", str(job["seed"]),
                    "--fold", str(job["fold"]),
                    "--epochs", str(EPOCHS),
                    "--patience", str(PATIENCE),
                    "--device", "cuda:0",
                    "--run-dir", str(RUN_ROOT / job["exp_name"]),
                ]

                if ratio == 0.0:
                    # Balanced only
                    cmd += ["--data-root", str(BAL_ROOT)]
                elif ratio == 1.0:
                    # Full mixing (all natural + all balanced)
                    cmd += ["--data-root", str(BAL_ROOT),
                            "--mix-data-root", str(NAT_ROOT)]
                else:
                    # Partial mixing: use --mix-data-root with --mix-ratio
                    cmd += ["--data-root", str(BAL_ROOT),
                            "--mix-data-root", str(NAT_ROOT),
                            "--mix-ratio", str(ratio)]

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

                log_dir = RUN_ROOT / job["exp_name"]
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = open(log_dir / "train.log", "w")
                proc = subprocess.Popen(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT,
                    env=env, cwd=str(Path.cwd()))
                running[gpu] = (proc, job, log_file)
                print(f"[GPU {gpu}] Started {job['exp_name']} (pid={proc.pid})")

        done_gpus = []
        for gpu, (proc, job, log_file) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                if ret == 0:
                    completed += 1
                    print(f"[GPU {gpu}] Done {job['exp_name']} ({completed}/{len(jobs)})")
                else:
                    failed += 1
                    print(f"[GPU {gpu}] FAILED {job['exp_name']} (rc={ret})")
                done_gpus.append(gpu)
        for gpu in done_gpus:
            del running[gpu]

        if running:
            time.sleep(5)

    print(f"\n[SUMMARY] {completed} completed, {failed} failed, {len(jobs)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
