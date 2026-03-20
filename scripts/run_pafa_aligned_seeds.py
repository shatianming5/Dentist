#!/usr/bin/env python3
"""Launch PAFA experiments with aligned seeds {1337,2020,2021} to match Table 4."""
import subprocess
import sys
import time
import os
from pathlib import Path

SEEDS = [1337, 2020, 2021]  # Match Table 4 mixing baseline seeds
FOLDS = [0, 1, 2, 3, 4]
MODES = ["mixing", "pafa"]
GPUS = [0, 1, 3]  # available GPUs
EPOCHS = 100
PATIENCE = 20

RUN_ROOT = Path("runs/pafa_aligned")
SCRIPT = Path("scripts/train_pafa.py")


def main():
    jobs = []
    for mode in MODES:
        for seed in SEEDS:
            for fold in FOLDS:
                exp_name = f"dgcnn_{mode}_s{seed}_f{fold}"
                result_file = RUN_ROOT / exp_name / "results.json"
                if result_file.exists():
                    print(f"[SKIP] {exp_name} already done")
                    continue
                jobs.append({"mode": mode, "seed": seed, "fold": fold, "exp_name": exp_name})

    print(f"[LAUNCH] {len(jobs)} jobs on {len(GPUS)} GPUs")
    if not jobs:
        print("All jobs already complete!")
        return 0

    running = {}  # gpu_id -> (process, job_info)
    job_idx = 0
    completed = 0
    failed = 0

    while job_idx < len(jobs) or running:
        for gpu in GPUS:
            if gpu not in running and job_idx < len(jobs):
                job = jobs[job_idx]
                job_idx += 1
                cmd = [
                    sys.executable, str(SCRIPT),
                    "--mode", job["mode"],
                    "--seed", str(job["seed"]),
                    "--fold", str(job["fold"]),
                    "--device", "cuda:0",
                    "--epochs", str(EPOCHS),
                    "--patience", str(PATIENCE),
                    "--run-root", str(RUN_ROOT),
                    "--exp-name", job["exp_name"],
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
