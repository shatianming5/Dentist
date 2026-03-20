#!/usr/bin/env python3
"""Launch CurveNet benchmark: 2 protocols × 5 seeds × 5 folds = 50 jobs on GPU 3."""
import subprocess, sys, time, os, json
from pathlib import Path

SEEDS = [42, 123, 1337, 2020, 2021]
FOLDS = [0, 1, 2, 3, 4]
PROTOCOLS = [
    ("balanced", "processed/raw_seg/v1"),
    ("natural",  "processed/raw_seg/v2_natural"),
]
GPU = 3
RUN_ROOT = Path("runs/curvenet_benchmark")
SCRIPT = Path("scripts/phase3_train_raw_seg.py")

def main():
    jobs = []
    for proto_name, proto_dir in PROTOCOLS:
        for seed in SEEDS:
            for fold in FOLDS:
                exp = f"curvenet_{proto_name}_s{seed}_f{fold}"
                result = RUN_ROOT / exp / "results.json"
                if result.exists():
                    print(f"[SKIP] {exp}")
                    continue
                jobs.append({
                    "proto": proto_name, "dir": proto_dir,
                    "seed": seed, "fold": fold, "exp": exp
                })

    print(f"[LAUNCH] {len(jobs)} CurveNet jobs on GPU {GPU}")
    if not jobs:
        return 0

    ok = fail = 0
    for i, j in enumerate(jobs):
        d = RUN_ROOT / j["exp"]; d.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(SCRIPT),
            "--model", "curvenet_seg",
            "--data-root", j["dir"],
            "--kfold", "metadata/splits_raw_case_kfold.json",
            "--fold", str(j["fold"]),
            "--seed", str(j["seed"]),
            "--epochs", "100",
            "--patience", "15",
            "--lr", "0.001",
            "--run-root", str(RUN_ROOT),
            "--exp-name", j["exp"],
            "--device", "cuda:0",
            "--n-points", "8192",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(GPU)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        lf = open(d / "train.log", "w")
        print(f"[{i+1}/{len(jobs)}] {j['exp']}", end=" ", flush=True)
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(Path.cwd()))
        ret = p.wait()
        lf.close()
        if ret == 0:
            ok += 1
            rfile = d / "results.json"
            if rfile.exists():
                r = json.load(open(rfile))
                miou = r.get("test_metrics", {}).get("mean_iou", "?")
                print(f"OK mIoU={miou:.4f}" if isinstance(miou, float) else "OK")
            else:
                print("OK (no results.json)")
        else:
            fail += 1
            print("FAIL")

    print(f"\n[DONE] {ok} ok, {fail} fail, {len(jobs)} total")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
