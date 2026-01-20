#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_str_list(text: str) -> list[str]:
    s = str(text or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


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


def has_min_d_metrics(eval_path: Path, *, clearance: float) -> bool:
    try:
        obj = read_json(eval_path)
        if float(obj.get("occlusion_clearance") or 0.0) != float(clearance):
            return False
        m = obj.get("metrics") or {}
        return isinstance(m, dict) and all(k in m for k in ("occlusion_min_d_p05", "occlusion_min_d_p50", "occlusion_min_d_p95"))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-evaluate Teeth3DS prep2target constraints runs with fixed metrics (paper protocol)."
    )
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/teeth3ds_prep2target_constraints"))
    ap.add_argument("--data-root", type=Path, default=Path("processed/teeth3ds_teeth/v1"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--splits", type=str, default="val,test", help="Comma list: val,test (default).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--ckpt", choices=["best", "final"], default="best")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--cut-q-min", type=float, default=0.7)
    ap.add_argument("--cut-q-max", type=float, default=0.7)
    ap.add_argument("--margin-band", type=float, default=0.02)
    ap.add_argument("--margin-points", type=int, default=64)
    ap.add_argument("--occlusion-clearance", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--continue-on-fail",
        action="store_true",
        default=True,
        help="Continue evaluating other runs if one eval command fails (default).",
    )
    ap.add_argument(
        "--stop-on-fail", action="store_false", dest="continue_on_fail", help="Stop at the first failure."
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip split eval when eval_<split>.json already has min_d quantiles for the requested clearance (default).",
    )
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", help="Do not skip existing eval.")
    ap.add_argument("--limit-runs", type=int, default=0, help="If >0, only process the first N run dirs (sorted).")
    args = ap.parse_args()

    root = args.root.resolve()
    runs_dir = (root / args.runs_dir).resolve()
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs dir: {runs_dir}")

    splits = [s.lower() for s in parse_str_list(args.splits)]
    if not splits:
        raise SystemExit("--splits is empty")
    bad = [s for s in splits if s not in {"val", "test"}]
    if bad:
        raise SystemExit(f"Invalid splits: {bad}")

    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")])
    if int(args.limit_runs) > 0:
        run_dirs = run_dirs[: int(args.limit_runs)]

    py = sys.executable or "python3"
    log_dir = runs_dir / "_paper_logs" / f"constraints_eval_{utc_now_compact()}"

    total = 0
    skipped = 0
    failed = 0
    for run_dir in run_dirs:
        if not (run_dir / "config.json").exists():
            continue
        if not ((run_dir / "ckpt_best.pt").exists() or (run_dir / "ckpt_final.pt").exists()):
            skipped += len(splits)
            print(f"[SKIP] {run_dir.name} (missing ckpt_*.pt)", flush=True)
            continue
        for split in splits:
            eval_path = run_dir / f"eval_{split}.json"
            if bool(args.skip_existing) and eval_path.exists() and has_min_d_metrics(
                eval_path, clearance=float(args.occlusion_clearance)
            ):
                skipped += 1
                print(f"[SKIP] {run_dir.name} split={split} (already has min_d metrics)", flush=True)
                continue

            cmd = [
                py,
                "scripts/phase4_eval_teeth3ds_constraints_run.py",
                "--root",
                str(root),
                "--run-dir",
                str(run_dir.relative_to(root)),
                "--data-root",
                str(args.data_root),
                "--device",
                str(args.device),
                "--split",
                split,
                "--batch-size",
                str(int(args.batch_size)),
                "--num-workers",
                str(int(args.num_workers)),
                "--ckpt",
                str(args.ckpt),
                "--cut-q-min",
                str(float(args.cut_q_min)),
                "--cut-q-max",
                str(float(args.cut_q_max)),
                "--margin-band",
                str(float(args.margin_band)),
                "--margin-points",
                str(int(args.margin_points)),
                "--occlusion-clearance",
                str(float(args.occlusion_clearance)),
            ]
            if bool(args.deterministic):
                cmd.append("--deterministic")

            log_path = log_dir / f"{run_dir.name}_{split}.log"
            print(f"[RUN] {run_dir.name} split={split}", flush=True)
            try:
                run_logged(cmd, cwd=root, log_path=log_path, dry_run=bool(args.dry_run))
                total += 1
            except subprocess.CalledProcessError as exc:
                failed += 1
                print(f"[FAIL] {run_dir.name} split={split} exit={exc.returncode} (see {log_path})", flush=True)
                if not bool(args.continue_on_fail):
                    raise

    print(f"[DONE] evaluated={total} skipped={skipped} failed={failed} logs={log_dir}", flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
