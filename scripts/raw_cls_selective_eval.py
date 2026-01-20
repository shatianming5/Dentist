#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def calibration_basic(probs: np.ndarray, y_true: np.ndarray, *, n_bins: int = 15) -> dict[str, Any]:
    if probs.ndim != 2:
        return {"total": int(y_true.shape[0]), "error": "invalid probs shape"}
    if y_true.ndim != 1 or y_true.shape[0] != probs.shape[0]:
        return {"total": int(y_true.shape[0]), "error": "invalid y_true shape"}

    eps = 1e-12
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)

    n_bins_i = max(2, int(n_bins))
    bins = np.linspace(0.0, 1.0, num=n_bins_i + 1)
    ece = 0.0
    for i in range(n_bins_i):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        w = float(np.mean(mask))
        bin_acc = float(np.mean(acc[mask]))
        bin_conf = float(np.mean(conf[mask]))
        ece += w * abs(bin_acc - bin_conf)

    p_true = probs[np.arange(y_true.shape[0]), y_true]
    nll = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
    onehot = np.zeros_like(probs, dtype=np.float64)
    onehot[np.arange(y_true.shape[0]), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    return {
        "total": int(y_true.shape[0]),
        "ece": float(ece),
        "nll": float(nll),
        "brier": float(brier),
        "mean_conf": float(np.mean(conf)),
        "accuracy": float(np.mean(acc)),
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> np.ndarray:
    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    c = int(cm.shape[0])
    supports = cm.sum(axis=1)
    total = int(cm.sum())
    correct = int(np.trace(cm))

    f1s_present: list[float] = []
    recalls_present: list[float] = []
    for i in range(c):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(supports[i]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if int(supports[i]) > 0:
            f1s_present.append(float(f1))
            recalls_present.append(float(rec))

    return {
        "total": total,
        "correct": correct,
        "accuracy": float(correct / total) if total > 0 else 0.0,
        "macro_f1_present": float(np.mean(f1s_present)) if f1s_present else 0.0,
        "balanced_accuracy_present": float(np.mean(recalls_present)) if recalls_present else 0.0,
    }


def temp_scale_probs(probs: np.ndarray, *, T: float) -> np.ndarray:
    eps = 1e-12
    tt = float(T)
    if not math.isfinite(tt) or tt <= 0:
        raise ValueError(f"Invalid temperature: {T}")
    logp = np.log(np.clip(probs, eps, 1.0)) / tt
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), eps, None)


@dataclass(frozen=True)
class SelectiveRow:
    coverage: float
    kept: int
    total: int
    min_conf_kept: float
    accuracy: float
    macro_f1_present: float
    balanced_accuracy_present: float
    ece: float
    nll: float
    brier: float


def render_md(rows: list[SelectiveRow], *, title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(
        "| coverage | kept | min_conf_kept | acc | macro_f1_present | bal_acc_present | ece | nll | brier |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.coverage:.2f} | {r.kept:d}/{r.total:d} | {r.min_conf_kept:.4f} | {r.accuracy:.4f} | {r.macro_f1_present:.4f} | {r.balanced_accuracy_present:.4f} | {r.ece:.4f} | {r.nll:.4f} | {r.brier:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Selective classification evaluation from raw_cls preds_test.jsonl.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--coverages", type=str, default="1.0,0.9,0.8,0.7")
    ap.add_argument("--use-calibrated", action="store_true", help="Apply temperature scaling from calib.json if present.")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None, help="Output selective.json (default: <run>/selective.json)")
    ap.add_argument("--out-md", type=Path, default=None, help="Output selective.md (default: <run>/selective.md)")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Missing run_dir: {run_dir}")
    preds_test = run_dir / "preds_test.jsonl"
    if not preds_test.is_file():
        raise SystemExit(f"Missing preds_test.jsonl: {preds_test}")

    rows_test = read_jsonl(preds_test)
    probs = np.asarray([r.get("probs") or [] for r in rows_test], dtype=np.float64)
    y_true = np.asarray([int(r.get("y_true", 0)) for r in rows_test], dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != y_true.shape[0]:
        raise SystemExit(f"Invalid probs shape: {probs.shape} vs y {y_true.shape}")

    notes: list[str] = []
    if bool(args.use_calibrated):
        calib_path = run_dir / "calib.json"
        if calib_path.is_file():
            obj = json.loads(calib_path.read_text(encoding="utf-8"))
            T = float(obj.get("temperature") or 0.0)
            probs = temp_scale_probs(probs, T=T)
            notes.append(f"used_calibrated: true (T={T})")
        else:
            notes.append("used_calibrated: requested but calib.json missing; fell back to raw probs")
    else:
        notes.append("used_calibrated: false")

    covs: list[float] = []
    for s in str(args.coverages or "").split(","):
        ss = s.strip()
        if not ss:
            continue
        try:
            covs.append(float(ss))
        except Exception as e:
            raise SystemExit(f"Invalid --coverages item: {s!r}: {e}") from e
    if not covs:
        raise SystemExit("No coverages provided.")

    total = int(y_true.shape[0])
    conf = probs.max(axis=1)
    order = np.argsort(-conf)  # desc

    out_rows: list[SelectiveRow] = []
    for cov in covs:
        c = float(cov)
        if not (0.0 < c <= 1.0):
            raise SystemExit(f"coverage must be in (0,1], got {c}")
        kept = int(max(1, math.ceil(c * total)))
        keep_idx = order[:kept]
        probs_k = probs[keep_idx]
        y_k = y_true[keep_idx]
        y_pred_k = probs_k.argmax(axis=1)
        cm = confusion_matrix(y_k, y_pred_k, num_classes=int(probs.shape[1]))
        m = metrics_from_confusion(cm)
        cal = calibration_basic(probs_k, y_k, n_bins=int(args.bins))
        min_conf = float(conf[keep_idx].min()) if keep_idx.size > 0 else 0.0
        out_rows.append(
            SelectiveRow(
                coverage=float(kept / total),
                kept=int(kept),
                total=int(total),
                min_conf_kept=float(min_conf),
                accuracy=float(m["accuracy"]),
                macro_f1_present=float(m["macro_f1_present"]),
                balanced_accuracy_present=float(m["balanced_accuracy_present"]),
                ece=float(cal.get("ece") or 0.0),
                nll=float(cal.get("nll") or 0.0),
                brier=float(cal.get("brier") or 0.0),
            )
        )

    out_json = args.out.expanduser().resolve() if args.out is not None else (run_dir / "selective.json")
    out_md = args.out_md.expanduser().resolve() if args.out_md is not None else (run_dir / "selective.md")

    obj = {
        "notes": notes,
        "bins": int(args.bins),
        "coverages": [float(r.coverage) for r in out_rows],
        "rows": [r.__dict__ for r in out_rows],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(out_rows, title="Selective classification (test)"), encoding="utf-8")

    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

