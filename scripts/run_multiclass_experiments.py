#!/usr/bin/env python3
"""Launch multi-class restoration segmentation experiments.

5-class: bg=0, filling=1, crown=2, post-core=3, onlay=4
Uses existing phase3_train_raw_seg.py with multiclass data directories.
"""
import subprocess
import sys
import os
import time
from pathlib import Path

SEEDS = [42, 7, 1337]
FOLDS = [0, 1, 2, 3, 4]
PROTOCOLS = [
    ("balanced", "processed/raw_seg/v1_multiclass"),
    ("natural", "processed/raw_seg/v2_natural_multiclass"),
]
GPUS = [0, 1, 3]
EPOCHS = 100
PATIENCE = 20
MODEL = "dgcnn_v2"
KFOLD = "metadata/splits_raw_case_kfold.json"
RUN_ROOT = Path("runs/multiclass_seg")
SCRIPT = Path("scripts/phase3_train_raw_seg.py")


def main():
    jobs = []
    for proto_name, data_root in PROTOCOLS:
        for seed in SEEDS:
            for fold in FOLDS:
                exp_name = f"dgcnn_{proto_name}_s{seed}_f{fold}"
                result_file = RUN_ROOT / exp_name / "results.json"
                if result_file.exists():
                    print(f"[SKIP] {exp_name}")
                    continue
                jobs.append({
                    "proto": proto_name, "data_root": data_root,
                    "seed": seed, "fold": fold, "exp_name": exp_name,
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
                val_fold = (job["fold"] + 1) % 5
                cmd = [
                    sys.executable, str(SCRIPT),
                    "--data-root", job["data_root"],
                    "--model", MODEL,
                    "--seed", str(job["seed"]),
                    "--fold", str(job["fold"]),
                    "--val-fold", str(val_fold),
                    "--kfold", KFOLD,
                    "--epochs", str(EPOCHS),
                    "--patience", str(PATIENCE),
                    "--device", "cuda:0",
                    "--run-root", str(RUN_ROOT),
                    "--exp-name", job["exp_name"],
                    "--focal-loss",
                    "--n-points", "8192",
                ]
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
                print(f"[GPU {gpu}] {job['exp_name']} (pid={proc.pid})")

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

    print(f"\n[SUMMARY] {completed} OK, {failed} FAIL, {len(jobs)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
