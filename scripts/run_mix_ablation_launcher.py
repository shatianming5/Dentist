#!/usr/bin/env python3
"""Launch mixing ratio ablation: 5 ratios × 3 seeds × 5 folds = 75 jobs."""
import subprocess, sys, time, os
from pathlib import Path

SEEDS = [1337, 2020, 2021]
FOLDS = [0, 1, 2, 3, 4]
RATIOS = [0.0, 0.25, 0.50, 0.75, 1.0]
GPUS = [0, 1]
RUN_ROOT = Path("runs/mix_ablation")
SCRIPT = Path("scripts/train_mix_ablation.py")

def main():
    jobs = []
    for ratio in RATIOS:
        for seed in SEEDS:
            for fold in FOLDS:
                tag = f"ratio{int(ratio*100)}"
                exp = f"dgcnn_{tag}_s{seed}_f{fold}"
                if (RUN_ROOT / exp / "results.json").exists():
                    print(f"[SKIP] {exp}")
                    continue
                jobs.append({"ratio": ratio, "seed": seed, "fold": fold, "exp": exp})

    print(f"[LAUNCH] {len(jobs)} jobs on GPUs {GPUS}")
    if not jobs:
        return 0

    running = {}
    idx = 0
    ok = fail = 0
    while idx < len(jobs) or running:
        for gpu in GPUS:
            if gpu not in running and idx < len(jobs):
                j = jobs[idx]; idx += 1
                d = RUN_ROOT / j["exp"]; d.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(SCRIPT),
                       "--ratio", str(j["ratio"]), "--seed", str(j["seed"]),
                       "--fold", str(j["fold"]), "--device", "cuda:0",
                       "--run-dir", str(d)]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                lf = open(d / "train.log", "w")
                p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(Path.cwd()))
                running[gpu] = (p, j, lf)
                print(f"[GPU {gpu}] {j['exp']} (pid={p.pid})")
        done = []
        for gpu, (p, j, lf) in running.items():
            r = p.poll()
            if r is not None:
                lf.close()
                if r == 0: ok += 1; print(f"[GPU {gpu}] Done {j['exp']} ({ok}/{len(jobs)})")
                else: fail += 1; print(f"[GPU {gpu}] FAIL {j['exp']}")
                done.append(gpu)
        for g in done: del running[g]
        if running: time.sleep(5)
    print(f"\n[DONE] {ok} ok, {fail} fail, {len(jobs)} total")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
