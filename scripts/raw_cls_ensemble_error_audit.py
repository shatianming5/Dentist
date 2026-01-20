#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    cm = np.asarray(cm, dtype=np.int64)
    k = int(cm.shape[0])
    f1s: list[float] = []
    recalls: list[float] = []
    for c in range(k):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(float(f1))
        recalls.append(float(rec))
    acc = float(np.trace(cm) / max(1, int(cm.sum())))
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "balanced_acc": float(np.mean(recalls))}


def entropy(p: np.ndarray) -> float:
    eps = 1e-12
    pp = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), eps, 1.0)
    pp = pp / float(np.sum(pp))
    return float(-np.sum(pp * np.log(pp)))


@dataclass(frozen=True)
class SplitData:
    case_keys: list[str]
    y_true: np.ndarray
    probs_by_member: list[np.ndarray]  # len=members, each [N, C]


def load_seed_averaged_split(
    *,
    runs_root: Path,
    exp: str,
    model: str,
    fold: int,
    seeds: list[int],
    split: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    by_case_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    by_case_true: dict[str, int] = {}

    missing: list[str] = []
    for seed in seeds:
        path = runs_root / exp / model / f"fold={fold}" / f"seed={seed}" / f"preds_{split}.jsonl"
        if not path.is_file():
            missing.append(str(path))
            continue
        for r in read_jsonl(path):
            ck = str(r.get("case_key") or r.get("sample_npz") or "").strip()
            if not ck:
                continue
            probs = np.asarray(r.get("probs") or [], dtype=np.float64)
            if probs.ndim != 1 or probs.size == 0:
                continue
            y = int(r.get("y_true", 0))
            by_case_probs[ck].append(probs)
            prev = by_case_true.get(ck)
            if prev is not None and int(prev) != int(y):
                raise SystemExit(f"Inconsistent y_true for case_key={ck} fold={fold}: {prev} vs {y}")
            by_case_true[ck] = int(y)

    if missing:
        raise SystemExit(f"Missing preds_{split}.jsonl for some runs (first): {missing[0]} (total missing={len(missing)})")

    case_keys = sorted([ck for ck, plist in by_case_probs.items() if len(plist) == len(seeds)])
    if not case_keys:
        raise SystemExit(f"No cases found for exp={exp} model={model} fold={fold} split={split}")

    probs = np.stack([np.mean(np.stack(by_case_probs[ck], axis=0), axis=0) for ck in case_keys], axis=0)
    y_true = np.asarray([by_case_true[ck] for ck in case_keys], dtype=np.int64)
    return case_keys, y_true, probs


def align_members_on_case_keys(datas: list[tuple[list[str], np.ndarray, np.ndarray]]) -> SplitData:
    if not datas:
        raise ValueError("no member data")
    common = set(datas[0][0])
    for ck, _yt, _pr in datas[1:]:
        common &= set(ck)
    common_keys = sorted(common)
    if not common_keys:
        raise SystemExit("No common case_keys across members (check runs/preds files).")

    y0_map = {k: int(y) for k, y in zip(datas[0][0], datas[0][1].tolist(), strict=True)}
    y_true = np.asarray([y0_map[k] for k in common_keys], dtype=np.int64)

    probs_by_member: list[np.ndarray] = []
    for mi, (case_keys, y, probs) in enumerate(datas):
        idx = {k: i for i, k in enumerate(case_keys)}
        y_m = np.asarray([int(y[idx[k]]) for k in common_keys], dtype=np.int64)
        if y_m.shape != y_true.shape or not np.all(y_m == y_true):
            raise SystemExit(f"y_true mismatch after aligning members (member index={mi})")
        probs_by_member.append(np.stack([probs[idx[k]] for k in common_keys], axis=0))
    return SplitData(case_keys=common_keys, y_true=y_true, probs_by_member=probs_by_member)


def ensemble_mean(probs_by_member: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probs_by_member, axis=0), axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit a mean-prob ensemble on raw_cls test folds (top errors / ambiguities).")
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True, help="Processed dataset root (for label_map.json).")
    ap.add_argument("--member", action="append", default=[], help="Ensemble member as exp:model (repeatable).")
    ap.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds (e.g., 1337,2020,2021).")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--topk", type=int, default=50, help="Top-K rows for error/ambiguity lists.")
    ap.add_argument("--min_conf", type=float, default=0.0, help="Optional filter for high-confidence errors.")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    args = ap.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    members_in = list(args.member or [])
    if not members_in:
        raise SystemExit("--member must be provided at least once (exp:model).")
    members = [_parse_member(s) for s in members_in]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    folds = [int(s) for s in str(args.folds).split(",") if s.strip()]
    topk = int(args.topk)
    if topk <= 0:
        raise SystemExit("--topk must be > 0")

    label_map = read_json(data_root / "label_map.json")
    label_to_id = {str(k): int(v) for k, v in label_map.items()}
    labels_by_id: list[str] = [None] * (max(label_to_id.values()) + 1)
    for lab, i in label_to_id.items():
        labels_by_id[int(i)] = str(lab)
    if any(x is None for x in labels_by_id):
        raise SystemExit("label_map.json must map to a contiguous 0..C-1 space")

    all_rows: list[dict[str, Any]] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for fold in folds:
        parts = []
        for exp, model in members:
            parts.append(
                load_seed_averaged_split(
                    runs_root=runs_root,
                    exp=exp,
                    model=model,
                    fold=int(fold),
                    seeds=seeds,
                    split="test",
                )
            )
        test = align_members_on_case_keys(parts)
        probs = ensemble_mean(test.probs_by_member)
        y_true = test.y_true
        y_pred = probs.argmax(axis=1)
        all_true.append(y_true)
        all_pred.append(y_pred)

        for ck, yt, pr in zip(test.case_keys, y_true.tolist(), probs.tolist(), strict=True):
            p = np.asarray(pr, dtype=np.float64)
            conf = float(np.max(p))
            order = np.argsort(-p)
            margin = float(p[order[0]] - p[order[1]]) if int(p.size) >= 2 else 0.0
            ent = entropy(p)
            all_rows.append(
                {
                    "fold": int(fold),
                    "case_key": str(ck),
                    "y_true": int(yt),
                    "y_true_label": labels_by_id[int(yt)],
                    "y_pred": int(np.argmax(p)),
                    "y_pred_label": labels_by_id[int(np.argmax(p))],
                    "p_pred": conf,
                    "margin": margin,
                    "entropy": ent,
                    "probs": [float(x) for x in p.tolist()],
                }
            )

    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    cm = confusion_matrix(y_true_all, y_pred_all, num_classes=len(labels_by_id))
    overall = metrics_from_confusion(cm)
    overall["n"] = int(y_true_all.shape[0])
    overall["confusion_matrix"] = cm.tolist()
    overall["labels_by_id"] = list(labels_by_id)

    errors = [r for r in all_rows if int(r["y_pred"]) != int(r["y_true"]) and float(r["p_pred"]) >= float(args.min_conf)]
    errors_sorted = sorted(errors, key=lambda r: (-float(r["p_pred"]), float(r["margin"]), float(r["entropy"])))
    ambiguous_sorted = sorted(all_rows, key=lambda r: (-float(r["entropy"]), float(r["margin"]), -float(r["p_pred"])))

    by_pair: dict[str, int] = defaultdict(int)
    for r in errors:
        key = f"{r['y_true_label']}→{r['y_pred_label']}"
        by_pair[key] += 1

    out = {
        "runs_root": str(runs_root),
        "data_root": str(data_root),
        "members": [{"exp": exp, "model": model} for exp, model in members],
        "seeds": seeds,
        "folds": folds,
        "method": "mean",
        "overall": overall,
        "error_pairs": dict(sorted(by_pair.items(), key=lambda kv: (-int(kv[1]), kv[0]))),
        "top_errors": errors_sorted[:topk],
        "top_ambiguous": ambiguous_sorted[:topk],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": overall, "top_errors": out["top_errors"][:5]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

