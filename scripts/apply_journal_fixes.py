#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply the repo's 12 journal-readiness fixes (idempotent where possible).")
    ap.add_argument("--root", type=Path, default=Path("."), help="Repo root (default: .)")
    ap.add_argument(
        "--build-raw-cls",
        action="store_true",
        help="(Optional) Rebuild a stable 4-class raw_cls dataset (can be slow, writes many npz files).",
    )
    ap.add_argument("--raw-cls-out", type=Path, default=Path("processed/raw_cls/v13_main4"))
    ap.add_argument("--raw-min-train-count", type=int, default=5)
    ap.add_argument("--kfold-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    root = args.root.resolve()
    if not (root / "scripts").exists():
        raise SystemExit(f"Not a repo root (missing scripts/): {root}")

    py = sys.executable or "python3"

    # 1) Freeze metadata splits (includes Teeth3DS patient-level splits).
    run([py, "scripts/phase0_freeze.py", "--root", "."], cwd=root)

    # 2) Ensure processed Teeth3DS tooth dataset uses patient-level split (rewrite index.jsonl split fields).
    teeth_data = root / "processed/teeth3ds_teeth/v1/index.jsonl"
    if teeth_data.exists():
        run(
            [
                py,
                "scripts/fix_teeth3ds_teeth_splits.py",
                "--data-root",
                "processed/teeth3ds_teeth/v1",
                "--splits",
                "metadata/splits_teeth3ds.json",
                "--inplace",
            ],
            cwd=root,
        )
    else:
        print("[SKIP] processed/teeth3ds_teeth/v1/index.jsonl not found; skip split rewrite", flush=True)

    # 3) Create stratified case-level K-fold splits for raw cases.
    run(
        [
            py,
            "scripts/make_raw_kfold_splits.py",
            "--root",
            ".",
            "--manifest",
            "converted/raw/manifest_with_labels.json",
            "--k",
            str(int(args.kfold_k)),
            "--seed",
            str(int(args.seed)),
            "--out",
            "metadata/splits_raw_case_kfold.json",
        ],
        cwd=root,
    )

    # 4) (Optional) rebuild stable raw_cls dataset (e.g. 4-class main task).
    if bool(args.build_raw_cls):
        out_path = args.raw_cls_out
        if not out_path.is_absolute():
            out_path = root / out_path
        if out_path.exists():
            print(f"[SKIP] raw_cls out exists: {out_path}", flush=True)
        else:
            run(
                [
                    py,
                    "scripts/phase1_build_raw_cls.py",
                    "--root",
                    ".",
                    "--out",
                    str(args.raw_cls_out),
                    "--include-name-regex",
                    "segmented$",
                    "--select-smallk",
                    "1",
                    "--merge-extraction-to-unknown",
                    "--min-train-count",
                    str(int(args.raw_min_train_count)),
                ],
                cwd=root,
            )

    # 5) Audit.
    run([py, "scripts/journal_audit.py", "--root", ".", "--out", "JOURNAL_AUDIT.md"], cwd=root)
    print("[OK] fixes applied; see JOURNAL_AUDIT.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

