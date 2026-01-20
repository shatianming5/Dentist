#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def normalize_extra_features(text: str) -> tuple[str, str]:
    parts = parse_str_list(text)
    if not parts:
        return "", ""
    # Canonicalize to avoid exp-name collisions due to ordering/duplicates.
    seen: set[str] = set()
    parts = [p for p in parts if not (p in seen or seen.add(p))]
    known_order = ["scale", "log_scale", "points", "log_points", "objects_used"]
    rank = {name: i for i, name in enumerate(known_order)}
    parts = sorted(parts, key=lambda p: (0, rank[p]) if p in rank else (1, p))
    canon = ",".join(parts)
    abbrev = {
        "scale": "sc",
        "log_scale": "lsc",
        "points": "pts",
        "log_points": "lpts",
        "objects_used": "ou",
    }
    tags: list[str] = []
    for p in parts:
        safe = "".join(ch for ch in p.lower() if ch.isalnum() or ch in {"_", "-"})
        tags.append(abbrev.get(p, safe[:12] or "x"))
    tag = "xf" + "_".join(tags)
    return canon, tag


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
    fold: int


def main() -> int:
    ap = argparse.ArgumentParser(description="Run raw_cls baselines across K-fold splits and multiple seeds (paper protocol).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--data-root", type=Path, default=Path("processed/raw_cls/v13_main4"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/raw_cls_baseline"))
    ap.add_argument("--kfold", type=Path, default=Path("metadata/splits_raw_case_kfold.json"))
    ap.add_argument("--folds", type=str, default="all", help="Comma list like 0,1,2 or 'all' (default).")
    ap.add_argument("--seeds", type=str, default="1337,2020,2021", help="Comma list of seeds.")
    ap.add_argument("--models", type=str, default="pointnet,dgcnn")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--init-feat", type=Path, default=None, help="Optional: init PointNet feat from AE ckpt (pointnet only).")
    ap.add_argument("--freeze-feat-epochs", type=int, default=0, help="Freeze PointNet feat for first K epochs.")
    ap.add_argument("--dgcnn-k", type=int, default=20)
    ap.add_argument("--extra-features", type=str, default="")
    ap.add_argument("--tta", type=int, default=0, help="Test-time augmentation passes for final val/test eval (0=disable).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True, help="Skip runs that already have metrics.json (default).")
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", help="Do not skip existing runs.")
    ap.add_argument("--tag", type=str, default="paper", help="Prefix tag for exp_name.")
    args = ap.parse_args()

    root = args.root.resolve()
    data_root = (root / args.data_root).resolve()
    runs_dir = (root / args.runs_dir).resolve()
    kfold_path = (root / args.kfold).resolve()

    if not data_root.exists():
        raise SystemExit(f"Missing data_root: {data_root}")
    if not kfold_path.exists():
        raise SystemExit(f"Missing kfold file: {kfold_path}")

    kfold_obj = read_json(kfold_path)
    k = int(kfold_obj.get("k") or 0)
    if k < 2:
        raise SystemExit(f"Invalid k in kfold file: {k}")

    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("--seeds is empty")
    models = [m.lower() for m in parse_str_list(args.models)]
    if not models:
        raise SystemExit("--models is empty")

    if str(args.folds).strip().lower() == "all":
        folds = list(range(k))
    else:
        folds = parse_int_list(args.folds)
    if not folds:
        raise SystemExit("--folds is empty")
    bad_folds = [f for f in folds if f < 0 or f >= k]
    if bad_folds:
        raise SystemExit(f"Invalid folds for k={k}: {bad_folds}")

    dataset_tag = data_root.name
    base_tag = str(args.tag).strip() or "paper"

    jobs: list[Job] = []
    extra_features_canon, extra_features_tag = normalize_extra_features(str(args.extra_features or ""))
    init_feat_path = ""
    if args.init_feat is not None:
        init_feat_file = args.init_feat.expanduser().resolve()
        if not init_feat_file.is_file():
            raise SystemExit(f"Missing init checkpoint: {init_feat_file}")
        init_feat_path = str(init_feat_file)
    for model in models:
        for seed in seeds:
            for fold in folds:
                flags: list[str] = []
                if bool(args.balanced_sampler):
                    flags.append("bal")
                if float(args.label_smoothing) > 0:
                    flags.append(f"ls{fmt_float_tag(float(args.label_smoothing))}")
                if extra_features_tag:
                    flags.append(extra_features_tag)
                if init_feat_path:
                    flags.append("init")
                if int(args.freeze_feat_epochs) > 0:
                    flags.append(f"frz{int(args.freeze_feat_epochs)}")
                if int(args.tta) > 0:
                    flags.append(f"tta{int(args.tta)}")
                flag_tag = ("_" + "_".join(flags)) if flags else ""
                exp_name = f"{base_tag}_rawcls_{dataset_tag}_{model}_n{int(args.n_points)}_k{k}_fold{fold}_seed{int(seed)}{flag_tag}"
                jobs.append(Job(exp_name=exp_name, seed=int(seed), model=model, fold=int(fold)))

    print(f"[PLAN] jobs={len(jobs)} (models={models}, seeds={seeds}, folds={folds})", flush=True)

    py = sys.executable or "python3"
    log_dir = runs_dir / "_paper_logs" / f"raw_cls_kfold_{utc_now_compact()}"
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
            "--dgcnn-k",
            str(int(args.dgcnn_k)),
            "--label-smoothing",
            str(float(args.label_smoothing)),
            "--extra-features",
            str(extra_features_canon),
            "--kfold",
            str(kfold_path),
            "--fold",
            str(int(job.fold)),
        ]
        if bool(args.balanced_sampler):
            cmd.append("--balanced-sampler")
        if init_feat_path:
            cmd.extend(["--init-feat", init_feat_path])
        if int(args.freeze_feat_epochs) > 0:
            cmd.extend(["--freeze-feat-epochs", str(int(args.freeze_feat_epochs))])
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
