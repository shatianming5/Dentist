#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnsembleResult:
    fold: int
    n: int
    accuracy: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def _parse_member(s: str) -> tuple[str, str]:
    raw = str(s or "").strip()
    if not raw or ":" not in raw:
        raise ValueError(f"Invalid --member (expected exp:model): {s!r}")
    exp, model = raw.split(":", 1)
    exp = exp.strip()
    model = model.strip()
    if not exp or not model:
        raise ValueError(f"Invalid --member (expected exp:model): {s!r}")
    return exp, model


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute multi-member (exp:model) + multi-seed ensemble metrics from raw_cls preds_test.jsonl."
    )
    ap.add_argument("--runs-root", type=Path, required=True, help="Root folder that contains exp/model/fold=*/seed=*/")
    ap.add_argument(
        "--member",
        action="append",
        default=[],
        help="Ensemble member as exp:model (repeatable).",
    )
    ap.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds to ensemble (e.g., 1337,2020,2021).")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated folds to include (default: 0..4).")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON file.")
    args = ap.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    members_in = list(args.member or [])
    if not members_in:
        raise SystemExit("--member must be provided at least once (exp:model).")
    members = [_parse_member(s) for s in members_in]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    folds = [int(s) for s in str(args.folds).split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds must be non-empty")
    if not folds:
        raise SystemExit("--folds must be non-empty")

    fold_to_cases: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    fold_to_true: dict[int, dict[str, int]] = defaultdict(dict)

    missing: list[str] = []
    for fold in folds:
        for exp, model in members:
            for seed in seeds:
                path = runs_root / exp / model / f"fold={fold}" / f"seed={seed}" / "preds_test.jsonl"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                for r in read_jsonl(path):
                    ck = str(r.get("case_key") or r.get("sample_npz") or "").strip()
                    if not ck:
                        continue
                    y = int(r.get("y_true", 0))
                    probs = np.asarray(r.get("probs") or [], dtype=np.float64)
                    if probs.ndim != 1 or probs.size == 0:
                        continue
                    fold_to_cases[int(fold)][ck].append(probs)
                    prev = fold_to_true[int(fold)].get(ck)
                    if prev is not None and int(prev) != int(y):
                        raise SystemExit(f"Inconsistent y_true for case_key={ck} fold={fold}: {prev} vs {y}")
                    fold_to_true[int(fold)][ck] = int(y)

    if missing:
        raise SystemExit(f"Missing preds_test.jsonl for some runs (first): {missing[0]} (total missing={len(missing)})")

    expected = int(len(members) * len(seeds))
    fold_results: list[EnsembleResult] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    for fold in folds:
        cases = fold_to_cases.get(int(fold)) or {}
        y_true: list[int] = []
        y_pred: list[int] = []
        for ck, plist in cases.items():
            if len(plist) != expected:
                continue
            p = np.mean(np.stack(plist, axis=0), axis=0)
            y_pred.append(int(np.argmax(p)))
            y_true.append(int(fold_to_true[int(fold)][ck]))
        if not y_true:
            continue
        y_true_np = np.asarray(y_true, dtype=np.int64)
        y_pred_np = np.asarray(y_pred, dtype=np.int64)
        acc = float(np.mean(y_true_np == y_pred_np))
        fold_results.append(EnsembleResult(fold=int(fold), n=int(y_true_np.shape[0]), accuracy=acc))
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    if not fold_results:
        raise SystemExit("No valid folds found (check --runs-root/--member/--seeds/--folds).")

    overall_true = np.asarray(all_true, dtype=np.int64)
    overall_pred = np.asarray(all_pred, dtype=np.int64)
    out = {
        "runs_root": str(runs_root),
        "members": [{"exp": exp, "model": model} for exp, model in members],
        "seeds": seeds,
        "folds": folds,
        "expected_members": expected,
        "per_fold": [r.__dict__ for r in fold_results],
        "mean_acc_over_folds": float(np.mean([r.accuracy for r in fold_results])),
        "std_acc_over_folds": float(np.std([r.accuracy for r in fold_results])),
        "overall_acc": float(np.mean(overall_true == overall_pred)),
        "overall_n": int(overall_true.shape[0]),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

