#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def fmt_ms(m: float, s: float, *, digits: int = 4) -> str:
    return f"{m:.{digits}f}±{s:.{digits}f}"


def calibration_basic(probs: np.ndarray, y_true: np.ndarray, *, n_bins: int = 15) -> dict[str, Any]:
    if probs.size == 0 or y_true.size == 0:
        return {"total": int(y_true.shape[0])}
    if probs.ndim != 2 or probs.shape[0] != y_true.shape[0]:
        return {"total": int(y_true.shape[0]), "error": "invalid probs shape"}
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


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
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
        "total": float(total),
        "accuracy": float(correct / total) if total > 0 else 0.0,
        "macro_f1_present": float(np.mean(f1s_present)) if f1s_present else 0.0,
        "balanced_accuracy_present": float(np.mean(recalls_present)) if recalls_present else 0.0,
    }


@dataclass(frozen=True)
class CoverageRow:
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


def eval_selective(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    coverages: list[float],
    bins: int,
) -> list[CoverageRow]:
    if probs.ndim != 2 or y_true.ndim != 1 or probs.shape[0] != y_true.shape[0]:
        raise ValueError(f"invalid shapes: probs={probs.shape} y_true={y_true.shape}")
    total = int(y_true.shape[0])
    conf = probs.max(axis=1)
    order = np.argsort(-conf)
    out: list[CoverageRow] = []
    for cov in coverages:
        c = float(cov)
        if not (0.0 < c <= 1.0):
            raise ValueError(f"coverage must be in (0,1], got {c}")
        kept = int(max(1, math.ceil(c * total)))
        keep_idx = order[:kept]
        probs_k = probs[keep_idx]
        y_k = y_true[keep_idx]
        y_pred_k = probs_k.argmax(axis=1)
        cm = confusion_matrix(y_k, y_pred_k, num_classes=int(probs.shape[1]))
        m = metrics_from_confusion(cm)
        cal = calibration_basic(probs_k, y_k, n_bins=int(bins))
        min_conf = float(conf[keep_idx].min()) if keep_idx.size > 0 else 0.0
        out.append(
            CoverageRow(
                coverage=float(c),
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
    return out


def render_md(
    *,
    title: str,
    coverages: list[float],
    mean_rows: dict[float, dict[str, str]],
    ensemble_rows: dict[float, CoverageRow] | None,
) -> str:
    lines: list[str] = [f"# {title}", "", f"- generated_at: {utc_now_iso()}", ""]
    lines.append("## Mean±std over seeds (k-fold merged)")
    lines.append("| coverage | acc | macro_f1_present | bal_acc_present | ece | nll | brier |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for cov in coverages:
        r = mean_rows.get(float(cov)) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(cov):.2f}",
                    r.get("accuracy", "0.0000±0.0000"),
                    r.get("macro_f1_present", "0.0000±0.0000"),
                    r.get("balanced_accuracy_present", "0.0000±0.0000"),
                    r.get("ece", "0.0000±0.0000"),
                    r.get("nll", "0.0000±0.0000"),
                    r.get("brier", "0.0000±0.0000"),
                ]
            )
            + " |"
        )
    lines.append("")

    if ensemble_rows is not None:
        lines.append("## Ensemble over seeds (avg probs)")
        lines.append("| coverage | kept | min_conf_kept | acc | macro_f1_present | bal_acc_present | ece | nll | brier |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cov in coverages:
            r = ensemble_rows.get(float(cov))
            if r is None:
                continue
            lines.append(
                f"| {r.coverage:.2f} | {r.kept:d}/{r.total:d} | {r.min_conf_kept:.4f} | {r.accuracy:.4f} | {r.macro_f1_present:.4f} | {r.balanced_accuracy_present:.4f} | {r.ece:.4f} | {r.nll:.4f} | {r.brier:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for item in str(s or "").split(","):
        t = item.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for item in str(s or "").split(","):
        t = item.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def load_kfold_merged(
    *,
    runs_root: Path,
    exp: str,
    model: str,
    fold: int,
    seed: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    run_dir = (runs_root / exp / model / f"fold={int(fold)}" / f"seed={int(seed)}").resolve()
    preds_path = run_dir / "preds_test.jsonl"
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing preds_test.jsonl: {preds_path}")
    rows = read_jsonl(preds_path)
    keys: list[str] = []
    y_true: list[int] = []
    probs: list[list[float]] = []
    for r in rows:
        keys.append(str(r.get("case_key") or r.get("id") or ""))
        y_true.append(int(r.get("y_true", 0)))
        probs.append([float(x) for x in (r.get("probs") or [])])
    return keys, np.asarray(y_true, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Selective classification report from k-fold merged preds_test.jsonl (raw_cls).")
    ap.add_argument("--runs-root", type=Path, required=True, help="E.g. runs/raw_cls/v13_main4")
    ap.add_argument("--exp", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="1337,2020,2021")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--coverages", type=str, default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    exp = str(args.exp).strip()
    model = str(args.model).strip()
    seeds = parse_int_list(args.seeds)
    folds = parse_int_list(args.folds)
    coverages = parse_float_list(args.coverages)
    if not seeds or not folds or not coverages:
        raise SystemExit("Empty seeds/folds/coverages.")

    per_seed: dict[int, dict[str, Any]] = {}
    merged_by_seed: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        key_to: dict[str, tuple[int, np.ndarray]] = {}
        num_classes: int | None = None
        for fold in folds:
            keys_f, y_f, p_f = load_kfold_merged(runs_root=runs_root, exp=exp, model=model, fold=fold, seed=seed)
            if num_classes is None:
                num_classes = int(p_f.shape[1])
            if int(p_f.shape[1]) != int(num_classes):
                raise SystemExit(f"num_classes mismatch: seed={seed} fold={fold} got {p_f.shape[1]} expected {num_classes}")
            for k, yt, pr in zip(keys_f, y_f.tolist(), p_f, strict=True):
                if not k:
                    continue
                prev = key_to.get(k)
                if prev is not None and int(prev[0]) != int(yt):
                    raise SystemExit(f"y_true mismatch for case_key={k} seed={seed}: {prev[0]} vs {yt}")
                key_to[k] = (int(yt), np.asarray(pr, dtype=np.float64))

        if not key_to or num_classes is None:
            raise SystemExit(f"No merged preds for seed={seed}")

        case_keys = sorted(key_to.keys())
        y_true = np.asarray([key_to[k][0] for k in case_keys], dtype=np.int64)
        probs = np.stack([key_to[k][1] for k in case_keys], axis=0).astype(np.float64, copy=False)
        rows = eval_selective(probs, y_true, coverages=coverages, bins=int(args.bins))
        per_seed[int(seed)] = {"n": int(y_true.shape[0]), "rows": [r.__dict__ for r in rows]}
        merged_by_seed[int(seed)] = {"case_keys": case_keys, "y_true": y_true, "probs": probs}

    mean_rows: dict[float, dict[str, str]] = {}
    for cov in coverages:
        accs: list[float] = []
        f1s: list[float] = []
        bals: list[float] = []
        eces: list[float] = []
        nlls: list[float] = []
        briers: list[float] = []
        for seed in seeds:
            rows = per_seed[int(seed)]["rows"]
            row = next((r for r in rows if abs(float(r["coverage"]) - float(cov)) < 1e-9), None)
            if row is None:
                continue
            accs.append(float(row["accuracy"]))
            f1s.append(float(row["macro_f1_present"]))
            bals.append(float(row["balanced_accuracy_present"]))
            eces.append(float(row["ece"]))
            nlls.append(float(row["nll"]))
            briers.append(float(row["brier"]))
        m_acc, s_acc = mean_std(accs)
        m_f1, s_f1 = mean_std(f1s)
        m_bal, s_bal = mean_std(bals)
        m_ece, s_ece = mean_std(eces)
        m_nll, s_nll = mean_std(nlls)
        m_br, s_br = mean_std(briers)
        mean_rows[float(cov)] = {
            "accuracy": fmt_ms(m_acc, s_acc),
            "macro_f1_present": fmt_ms(m_f1, s_f1),
            "balanced_accuracy_present": fmt_ms(m_bal, s_bal),
            "ece": fmt_ms(m_ece, s_ece),
            "nll": fmt_ms(m_nll, s_nll),
            "brier": fmt_ms(m_br, s_br),
        }

    # Ensemble: average probs across seeds.
    common_keys = set(merged_by_seed[int(seeds[0])]["case_keys"])
    for seed in seeds[1:]:
        common_keys &= set(merged_by_seed[int(seed)]["case_keys"])
    ensemble_rows: dict[float, CoverageRow] | None = None
    if common_keys:
        keys = sorted(common_keys)
        y0 = None
        probs_sum = None
        for seed in seeds:
            obj = merged_by_seed[int(seed)]
            key_to_idx = {k: i for i, k in enumerate(obj["case_keys"])}
            idx = np.asarray([key_to_idx[k] for k in keys], dtype=np.int64)
            y = obj["y_true"][idx]
            p = obj["probs"][idx]
            if y0 is None:
                y0 = y
                probs_sum = np.zeros_like(p, dtype=np.float64)
            else:
                if not np.all(y0 == y):
                    raise SystemExit("Ensemble y_true mismatch across seeds after aligning case_keys.")
            assert probs_sum is not None
            probs_sum += p
        assert y0 is not None and probs_sum is not None
        probs_avg = probs_sum / float(len(seeds))
        rows = eval_selective(probs_avg, y0, coverages=coverages, bins=int(args.bins))
        ensemble_rows = {float(r.coverage): r for r in rows}

    out_md = args.out_md.expanduser().resolve()
    out_json = args.out_json.expanduser().resolve() if args.out_json is not None else out_md.with_suffix(".json")
    title = f"Selective classification — {exp}/{model}"
    out_md.write_text(render_md(title=title, coverages=coverages, mean_rows=mean_rows, ensemble_rows=ensemble_rows), encoding="utf-8")
    write_json(
        out_json,
        {
            "generated_at": utc_now_iso(),
            "runs_root": str(runs_root),
            "exp": exp,
            "model": model,
            "seeds": seeds,
            "folds": folds,
            "coverages": coverages,
            "bins": int(args.bins),
            "per_seed": per_seed,
            "mean_rows": mean_rows,
            "ensemble": {"rows": [r.__dict__ for r in (ensemble_rows or {}).values()]} if ensemble_rows is not None else None,
        },
    )
    print(f"[OK] wrote: {out_md}")
    print(f"[OK] wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
