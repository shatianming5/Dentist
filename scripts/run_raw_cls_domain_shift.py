#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_int_list(text: str) -> list[int]:
    s = str(text or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def parse_str_list(text: str) -> list[str]:
    s = str(text or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def fmt_float_tag(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def source_tag(s: str) -> str:
    mapping = {
        "普通标注": "norm",
        "专家标注": "expert",
    }
    key = str(s or "").strip()
    if key in mapping:
        return mapping[key]
    safe = "".join(ch for ch in key.lower() if ch.isalnum() or ch in {"_", "-"})
    return (safe[:12] or "src")


def run_logged(cmd: list[str], *, cwd: Path, log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY] {' '.join(cmd)}", flush=True)
        return
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.write(f"# cwd: {cwd}\n")
        f.write(f"# started_at: {datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}\n")
        f.flush()
        subprocess.check_call(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)


@dataclass(frozen=True)
class Job:
    exp_name: str
    seed: int
    model: str


def main() -> int:
    ap = argparse.ArgumentParser(description="Run raw_cls domain-shift experiments (train source → test source).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--data-root", type=Path, default=Path("processed/raw_cls/v13_main4"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/raw_cls_domain_shift"))
    ap.add_argument("--seeds", type=str, default="1337,2020,2021")
    ap.add_argument("--models", type=str, default="pointnet,dgcnn")
    ap.add_argument("--train-source", type=str, required=True, help="e.g. 普通标注")
    ap.add_argument("--test-source", type=str, required=True, help="e.g. 专家标注")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--extra-features", type=str, default="")
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    ap.add_argument("--tag", type=str, default="paper", help="Prefix tag for exp_name.")
    args = ap.parse_args()

    root = args.root.resolve()
    data_root = (root / args.data_root).resolve()
    runs_dir = (root / args.runs_dir).resolve()
    if not data_root.exists():
        raise SystemExit(f"Missing data_root: {data_root}")

    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("--seeds is empty")
    models = [m.lower() for m in parse_str_list(args.models)]
    if not models:
        raise SystemExit("--models is empty")

    train_source = str(args.train_source).strip()
    test_source = str(args.test_source).strip()
    if not train_source or not test_source:
        raise SystemExit("--train-source/--test-source must be non-empty")
    if not (0.0 < float(args.val_ratio) < 0.5):
        raise SystemExit("--val-ratio must be in (0,0.5)")

    dataset_tag = data_root.name
    base_tag = str(args.tag).strip() or "paper"
    tr_tag = source_tag(train_source)
    te_tag = source_tag(test_source)

    flags: list[str] = []
    if bool(args.balanced_sampler):
        flags.append("bal")
    if float(args.label_smoothing) > 0:
        flags.append(f"ls{fmt_float_tag(float(args.label_smoothing))}")
    if str(args.extra_features).strip():
        flags.append("xf")
    if int(args.tta) > 0:
        flags.append(f"tta{int(args.tta)}")
    flags.append(f"vr{fmt_float_tag(float(args.val_ratio))}")
    flag_tag = ("_" + "_".join(flags)) if flags else ""

    jobs: list[Job] = []
    for model in models:
        for seed in seeds:
            exp_name = (
                f"{base_tag}_rawcls_dom_{dataset_tag}_{model}_n{int(args.n_points)}_seed{int(seed)}_tr{tr_tag}_te{te_tag}"
                f"{flag_tag}"
            )
            jobs.append(Job(exp_name=exp_name, seed=int(seed), model=model))

    print(f"[PLAN] jobs={len(jobs)} (models={models}, seeds={seeds}, train={train_source}, test={test_source})", flush=True)

    py = sys.executable or "python3"
    log_dir = runs_dir / "_paper_logs" / f"raw_cls_domain_{utc_now_compact()}"
    done = 0
    skipped = 0
    for i, job in enumerate(jobs, start=1):
        out_dir = runs_dir / job.exp_name
        metrics_path = out_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            skipped += 1
            print(f"[{i:03d}/{len(jobs)}] skip existing: {job.exp_name}", flush=True)
            continue

        cmd = [
            py,
            "scripts/phase3_train_raw_cls_baseline.py",
            "--data-root",
            str(data_root),
            "--run-root",
            str(runs_dir),
            "--exp-name",
            job.exp_name,
            "--seed",
            str(job.seed),
            "--device",
            str(args.device),
            "--model",
            job.model,
            "--epochs",
            str(int(args.epochs)),
            "--patience",
            str(int(args.patience)),
            "--batch-size",
            str(int(args.batch_size)),
            "--n-points",
            str(int(args.n_points)),
            "--num-workers",
            str(int(args.num_workers)),
            "--label-smoothing",
            str(float(args.label_smoothing)),
            "--extra-features",
            str(args.extra_features or ""),
            "--source-train",
            train_source,
            "--source-test",
            test_source,
            "--source-val-ratio",
            str(float(args.val_ratio)),
        ]
        if bool(args.balanced_sampler):
            cmd.append("--balanced-sampler")
        if int(args.tta) > 0:
            cmd.extend(["--tta", str(int(args.tta))])

        log_path = log_dir / f"{job.exp_name}.log"
        print(f"[{i:03d}/{len(jobs)}] run: {job.exp_name}", flush=True)
        run_logged(cmd, cwd=root, log_path=log_path, dry_run=bool(args.dry_run))
        done += 1

    print(f"[DONE] completed={done} skipped={skipped} logs={log_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

