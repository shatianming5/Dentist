#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}")
    c = int(num_classes)
    if c <= 0:
        raise ValueError(f"invalid num_classes: {num_classes}")
    flat = c * y_true.astype(np.int64, copy=False) + y_pred.astype(np.int64, copy=False)
    cm = np.bincount(flat, minlength=c * c).reshape(c, c)
    return cm.astype(np.int64, copy=False)


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    num_classes = int(cm.shape[0])
    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = float(correct / total) if total > 0 else 0.0

    f1s_all: list[float] = []
    f1s_present: list[float] = []
    recalls_present: list[float] = []
    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        support = float(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s_all.append(float(f1))
        if support > 0:
            f1s_present.append(float(f1))
            recalls_present.append(float(recall))

    macro_f1_all = float(np.mean(f1s_all)) if f1s_all else 0.0
    macro_f1_present = float(np.mean(f1s_present)) if f1s_present else 0.0
    bal_acc_present = float(np.mean(recalls_present)) if recalls_present else 0.0
    return {
        "accuracy": float(acc),
        "macro_f1_all": float(macro_f1_all),
        "macro_f1_present": float(macro_f1_present),
        "balanced_accuracy_present": float(bal_acc_present),
    }


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


def parse_time_ts(iso: str | None, *, fallback_path: Path) -> float:
    s = str(iso or "").strip()
    if s:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return float(dt.timestamp())
        except Exception:
            pass
    try:
        return float(fallback_path.stat().st_mtime)
    except Exception:
        return 0.0


@dataclass(frozen=True)
class SeedPreds:
    case_keys: list[str]
    key_to_idx: dict[str, int]
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray


def metric_value(metric: str, *, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> float:
    if metric == "accuracy":
        return float(np.mean((y_pred == y_true).astype(np.float64))) if y_true.size else 0.0
    if metric == "macro_f1_present":
        cm = confusion_matrix(y_true, y_pred, num_classes=int(probs.shape[1]))
        return float(metrics_from_confusion(cm).get("macro_f1_present") or 0.0)
    if metric == "macro_f1_all":
        cm = confusion_matrix(y_true, y_pred, num_classes=int(probs.shape[1]))
        return float(metrics_from_confusion(cm).get("macro_f1_all") or 0.0)
    if metric == "balanced_accuracy_present":
        cm = confusion_matrix(y_true, y_pred, num_classes=int(probs.shape[1]))
        return float(metrics_from_confusion(cm).get("balanced_accuracy_present") or 0.0)
    if metric == "ece":
        return float(calibration_basic(probs, y_true).get("ece") or 0.0)
    raise ValueError(f"unsupported metric: {metric}")


def paired_bootstrap_diff(
    *,
    a: dict[int, SeedPreds],
    b: dict[int, SeedPreds],
    metric: str,
    n_bootstrap: int,
    seed: int,
    warnings: list[str],
) -> dict[str, Any]:
    seeds = sorted(set(a.keys()) & set(b.keys()))
    if not seeds:
        return {"error": "no common seeds"}

    aligned: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for s in seeds:
        pa = a[int(s)]
        pb = b[int(s)]
        common = sorted(set(pa.case_keys) & set(pb.case_keys))
        if not common:
            warnings.append(f"comparison metric={metric}: seed={s} has no common case_keys")
            continue
        ia = np.asarray([pa.key_to_idx[k] for k in common], dtype=np.int64)
        ib = np.asarray([pb.key_to_idx[k] for k in common], dtype=np.int64)
        yt_a = pa.y_true[ia]
        yt_b = pb.y_true[ib]
        if yt_a.shape != yt_b.shape or not np.all(yt_a == yt_b):
            warnings.append(f"comparison metric={metric}: seed={s} y_true mismatch after aligning case_keys")
            continue
        aligned.append((int(s), yt_a, pa.y_pred[ia], pa.probs[ia], pb.y_pred[ib], pb.probs[ib]))

    if not aligned:
        return {"error": "no aligned seeds"}

    # Observed delta averaged over seeds (per-seed metric computed on full case set).
    deltas_seed: list[float] = []
    for _s, yt, yp_a, pr_a, yp_b, pr_b in aligned:
        ma = metric_value(metric, y_true=yt, y_pred=yp_a, probs=pr_a)
        mb = metric_value(metric, y_true=yt, y_pred=yp_b, probs=pr_b)
        deltas_seed.append(float(ma - mb))
    observed = float(np.mean(np.asarray(deltas_seed, dtype=np.float64)))

    # Hierarchical bootstrap: resample seeds (outer) and cases (inner).
    rng = np.random.default_rng(int(seed))
    deltas: list[float] = []
    n_seeds = len(aligned)
    for _ in range(int(n_bootstrap)):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds, endpoint=False)
        ds: list[float] = []
        for j in seed_idx.tolist():
            _s, yt, yp_a, pr_a, yp_b, pr_b = aligned[int(j)]
            n = int(yt.shape[0])
            idx = rng.integers(0, n, size=n, endpoint=False)
            ma = metric_value(metric, y_true=yt[idx], y_pred=yp_a[idx], probs=pr_a[idx])
            mb = metric_value(metric, y_true=yt[idx], y_pred=yp_b[idx], probs=pr_b[idx])
            ds.append(float(ma - mb))
        deltas.append(float(np.mean(np.asarray(ds, dtype=np.float64))))

    arr = np.asarray(deltas, dtype=np.float64)
    lo = float(np.quantile(arr, 0.025))
    hi = float(np.quantile(arr, 0.975))
    p_ge0 = float(np.mean(arr >= 0.0))
    p_le0 = float(np.mean(arr <= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_ge0, p_le0)))
    return {
        "metric": metric,
        "delta_mean": float(np.mean(arr)),
        "delta_observed": float(observed),
        "delta_ci95": {"lo": lo, "hi": hi},
        "p_two_sided": p_two,
        "seeds": [int(s) for s in seeds],
        "n_seeds": int(len(seeds)),
        "n_cases_per_seed": [int(x[1].shape[0]) for x in aligned],
    }


def bootstrap_ci_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    n = int(y_true.shape[0])
    if n <= 0:
        return {}
    rng = np.random.default_rng(int(seed))
    accs: list[float] = []
    f1s: list[float] = []
    bals: list[float] = []
    eces: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        yt = y_true[idx]
        yp = y_pred[idx]
        pr = probs[idx]
        cm = confusion_matrix(yt, yp, num_classes=num_classes)
        met = metrics_from_confusion(cm)
        cal = calibration_basic(pr, yt)
        accs.append(float(met["accuracy"]))
        f1s.append(float(met["macro_f1_present"]))
        bals.append(float(met["balanced_accuracy_present"]))
        eces.append(float(cal.get("ece") or 0.0))

    def _ci(xs: list[float]) -> dict[str, float]:
        arr = np.asarray(xs, dtype=np.float64)
        lo = float(np.quantile(arr, 0.025))
        hi = float(np.quantile(arr, 0.975))
        return {"mean": float(np.mean(arr)), "lo": lo, "hi": hi}

    return {
        "accuracy": _ci(accs),
        "macro_f1_present": _ci(f1s),
        "balanced_accuracy_present": _ci(bals),
        "ece": _ci(eces),
    }


@dataclass(frozen=True)
class RunRef:
    run_dir: Path
    exp_name: str
    generated_at: str
    sort_ts: float
    data_tag: str
    model: str
    n_points: int
    seed: int
    kfold_k: int
    test_fold: int
    balanced: bool
    label_smoothing: float
    extra_features: str
    tta: int


def scan_runs(runs_dir: Path, *, data_tag_filter: str, exp_prefix: str) -> list[RunRef]:
    out: list[RunRef] = []
    for d in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        cfg_path = d / "config.json"
        preds_path = d / "preds_test.jsonl"
        if not cfg_path.exists() or not preds_path.exists():
            continue
        try:
            cfg = read_json(cfg_path)
        except Exception:
            continue
        exp_name = str(cfg.get("exp_name") or d.name)
        if str(exp_prefix).strip() and not exp_name.startswith(str(exp_prefix).strip()):
            continue
        gen_at = str(cfg.get("generated_at") or "")
        ts = parse_time_ts(gen_at, fallback_path=cfg_path)
        data_root = str(cfg.get("data_root") or "")
        data_tag = Path(data_root).name if data_root else ""
        if str(data_tag_filter).strip() and data_tag != str(data_tag_filter).strip():
            continue
        kfold_k = int(cfg.get("kfold_k") or 0)
        if kfold_k <= 0:
            continue
        model = str(cfg.get("model") or "unknown").lower()
        seed = int(cfg.get("seed", 0))
        test_fold = int(cfg.get("kfold_test_fold", -1))
        if test_fold < 0 or test_fold >= kfold_k:
            continue
        out.append(
            RunRef(
                run_dir=d,
                exp_name=str(exp_name),
                generated_at=str(gen_at),
                sort_ts=float(ts),
                data_tag=data_tag,
                model=model,
                n_points=int(cfg.get("n_points") or 0),
                seed=seed,
                kfold_k=kfold_k,
                test_fold=test_fold,
                balanced=bool(cfg.get("balanced_sampler") or False),
                label_smoothing=float(cfg.get("label_smoothing") or 0.0),
                extra_features=",".join(cfg.get("extra_features") or []),
                tta=int(cfg.get("tta") or 0),
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Journal-style raw_cls report: merge k-fold test predictions per seed, add bootstrap CI and by-source breakdown.")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/raw_cls_baseline"))
    ap.add_argument("--out-prefix", type=Path, default=Path("paper_tables/raw_cls_kfold_merged_report_v13_main4"))
    ap.add_argument("--data-tag", type=str, default="v13_main4")
    ap.add_argument("--exp-prefix", type=str, default="paper_rawcls_", help="Only include runs whose exp_name starts with this prefix (default: paper_rawcls_).")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed for bootstrap.")
    args = ap.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs dir: {runs_dir}")

    runs = scan_runs(runs_dir, data_tag_filter=str(args.data_tag), exp_prefix=str(args.exp_prefix))
    if not runs:
        raise SystemExit(f"No k-fold runs found under {runs_dir} (data_tag={args.data_tag})")

    # Group by configuration (same as paper_table_raw_cls.py).
    groups: dict[tuple[Any, ...], list[RunRef]] = {}
    for r in runs:
        key = (
            r.data_tag,
            r.model,
            r.n_points,
            r.kfold_k,
            bool(r.balanced),
            round(float(r.label_smoothing), 6),
            r.extra_features,
            int(r.tta),
        )
        groups.setdefault(key, []).append(r)

    # Keep merged predictions for paired comparisons (not written to JSON).
    preds_by_group: dict[tuple[Any, ...], dict[int, SeedPreds]] = {}

    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "runs_dir": str(runs_dir),
        "data_tag": str(args.data_tag),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "groups": [],
        "comparisons": [],
        "warnings": [],
    }

    md_lines: list[str] = []
    md_lines.append("# raw_cls k-fold merged report (per-seed)")
    md_lines.append("")
    md_lines.append(f"- generated_at: {report['generated_at']}")
    md_lines.append(f"- runs_dir: `{runs_dir}`")
    md_lines.append(f"- data_tag: `{args.data_tag}`")
    md_lines.append(f"- bootstrap: n={int(args.n_bootstrap)} seed={int(args.seed)}")
    md_lines.append("")

    md_lines.append("## Overall (mean±std over seeds)")
    md_lines.append("")
    md_lines.append(
        "| macro_f1_present (mean±std) | macro_f1_all (mean±std) | bal_acc_present (mean±std) | acc (mean±std) | ece (mean±std) | seeds | n_cases | model | extra_features | tta |"
    )
    md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|---:|")

    for key in sorted(groups.keys(), key=lambda k: (k[1], k[6], k[7], k[2])):
        data_tag, model, n_points, kfold_k, balanced, label_smoothing, extra_features, tta = key
        items = groups[key]
        seed_to_folds: dict[int, dict[int, RunRef]] = {}
        for r in items:
            fm = seed_to_folds.setdefault(int(r.seed), {})
            prev = fm.get(int(r.test_fold))
            if prev is None:
                fm[int(r.test_fold)] = r
                continue
            keep, drop = (r, prev) if (float(r.sort_ts), str(r.exp_name)) > (float(prev.sort_ts), str(prev.exp_name)) else (prev, r)
            fm[int(r.test_fold)] = keep
            report["warnings"].append(
                f"duplicate run for group={model} extra={extra_features or '(none)'} seed={int(r.seed)} fold={int(r.test_fold)}: "
                f"keep={keep.exp_name} (t={keep.generated_at or 'mtime'}) drop={drop.exp_name} (t={drop.generated_at or 'mtime'})"
            )

        seed_entries: dict[str, Any] = {}
        seed_preds: dict[int, SeedPreds] = {}
        seed_metrics: list[dict[str, float]] = []
        seed_eces: list[float] = []
        n_cases_ref: int | None = None

        for seed, fold_map in sorted(seed_to_folds.items()):
            if int(kfold_k) > 0 and len(fold_map) < int(kfold_k):
                report["warnings"].append(
                    f"missing folds for group={model} extra={extra_features or '(none)'} seed={seed}: have {sorted(fold_map.keys())}"
                )
                continue
            merged: dict[str, dict[str, Any]] = {}
            for fold in sorted(fold_map.keys()):
                rr = fold_map[fold]
                rows = read_jsonl(rr.run_dir / "preds_test.jsonl")
                for r0 in rows:
                    ck = str(r0.get("case_key") or f"{fold}:{len(merged)}")
                    if ck in merged:
                        raise SystemExit(f"Duplicate case_key while merging (seed={seed}, fold={fold}): {ck}")
                    merged[ck] = r0

            case_keys = sorted(merged.keys())
            merged_rows = [merged[k] for k in case_keys]
            if not merged_rows:
                continue
            if n_cases_ref is None:
                n_cases_ref = len(merged_rows)
            probs = np.asarray([r0.get("probs") or [] for r0 in merged_rows], dtype=np.float64)
            y_true = np.asarray([int(r0.get("y_true") or 0) for r0 in merged_rows], dtype=np.int64)
            y_pred = np.asarray([int(r0.get("y_pred") or 0) for r0 in merged_rows], dtype=np.int64)
            if probs.ndim != 2 or probs.shape[0] != y_true.shape[0]:
                raise SystemExit(f"Invalid probs shape in merged rows (seed={seed}): {probs.shape}")
            num_classes = int(probs.shape[1])

            cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
            overall = metrics_from_confusion(cm)
            cal = calibration_basic(probs, y_true)
            ci = bootstrap_ci_metrics(
                y_true=y_true,
                y_pred=y_pred,
                probs=probs,
                num_classes=num_classes,
                n_bootstrap=int(args.n_bootstrap),
                seed=int(args.seed) + int(seed),
            )

            by_source: dict[str, Any] = {}
            by_source_cal: dict[str, Any] = {}
            sources = sorted({str(r0.get("source") or "") for r0 in merged_rows})
            for src in sources:
                rows_s = [r0 for r0 in merged_rows if str(r0.get("source") or "") == src]
                if not rows_s:
                    continue
                probs_s = np.asarray([r0.get("probs") or [] for r0 in rows_s], dtype=np.float64)
                yt_s = np.asarray([int(r0.get("y_true") or 0) for r0 in rows_s], dtype=np.int64)
                yp_s = np.asarray([int(r0.get("y_pred") or 0) for r0 in rows_s], dtype=np.int64)
                cm_s = confusion_matrix(yt_s, yp_s, num_classes=num_classes)
                by_source[src or "(missing)"] = metrics_from_confusion(cm_s) | {"total": int(yt_s.shape[0])}
                by_source_cal[src or "(missing)"] = calibration_basic(probs_s, yt_s)

            seed_entries[str(seed)] = {
                "seed": int(seed),
                "n_cases": int(y_true.shape[0]),
                "num_classes": int(num_classes),
                "overall": overall,
                "overall_calibration": cal,
                "ci95": ci,
                "by_source": by_source,
                "by_source_calibration": by_source_cal,
            }
            seed_preds[int(seed)] = SeedPreds(
                case_keys=list(case_keys),
                key_to_idx={k: i for i, k in enumerate(case_keys)},
                y_true=y_true,
                y_pred=y_pred,
                probs=probs,
            )
            seed_metrics.append(
                {
                    "accuracy": float(overall["accuracy"]),
                    "macro_f1_all": float(overall["macro_f1_all"]),
                    "macro_f1_present": float(overall["macro_f1_present"]),
                    "balanced_accuracy_present": float(overall["balanced_accuracy_present"]),
                    "ece": float(cal.get("ece") or 0.0),
                }
            )
            seed_eces.append(float(cal.get("ece") or 0.0))

        if not seed_entries:
            continue

        acc_m, acc_s = mean_std([m["accuracy"] for m in seed_metrics])
        f1p_m, f1p_s = mean_std([m["macro_f1_present"] for m in seed_metrics])
        f1a_m, f1a_s = mean_std([m["macro_f1_all"] for m in seed_metrics])
        bal_m, bal_s = mean_std([m["balanced_accuracy_present"] for m in seed_metrics])
        ece_m, ece_s = mean_std(seed_eces)
        seeds_list = sorted(int(s) for s in seed_entries.keys())
        n_cases_val = int(n_cases_ref or 0)

        md_lines.append(
            "| "
            + " | ".join(
                [
                    fmt_ms(f1p_m, f1p_s),
                    fmt_ms(f1a_m, f1a_s),
                    fmt_ms(bal_m, bal_s),
                    fmt_ms(acc_m, acc_s),
                    fmt_ms(ece_m, ece_s),
                    str(len(seeds_list)),
                    str(n_cases_val),
                    str(model),
                    str(extra_features or "(none)"),
                    str(int(tta)),
                ]
            )
            + " |"
        )

        report["groups"].append(
            {
                "key": {
                    "data_tag": data_tag,
                    "model": model,
                    "n_points": int(n_points),
                    "kfold_k": int(kfold_k),
                    "balanced": bool(balanced),
                    "label_smoothing": float(label_smoothing),
                    "extra_features": str(extra_features),
                    "tta": int(tta),
                },
                "seeds": seed_entries,
                "seed_mean": {
                    "accuracy": acc_m,
                    "macro_f1_present": f1p_m,
                    "macro_f1_all": f1a_m,
                    "balanced_accuracy_present": bal_m,
                    "ece": ece_m,
                },
                "seed_std": {
                    "accuracy": acc_s,
                    "macro_f1_present": f1p_s,
                    "macro_f1_all": f1a_s,
                    "balanced_accuracy_present": bal_s,
                    "ece": ece_s,
                },
            }
        )

        preds_by_group[key] = seed_preds

    if report["warnings"]:
        md_lines.append("")
        md_lines.append("## Warnings")
        md_lines.append("")
        for w in report["warnings"]:
            md_lines.append(f"- {w}")

    # Paired comparisons (journal-style): hierarchical bootstrap over (seed, case).
    md_lines.append("")
    md_lines.append("## Paired comparisons (hierarchical bootstrap over seeds×cases)")
    md_lines.append("")
    md_lines.append("| metric | delta(mean) | CI95 | p(two-sided) | A | B | seeds | family |")
    md_lines.append("|---|---:|---:|---:|---|---|---:|---|")

    def family_of(k: tuple[Any, ...]) -> tuple[Any, ...]:
        data_tag, _model, n_points, kfold_k, balanced, label_smoothing, _extra, tta = k
        return (data_tag, int(n_points), int(kfold_k), bool(balanced), float(label_smoothing), int(tta))

    def name_of(k: tuple[Any, ...]) -> str:
        _data_tag, model, n_points, kfold_k, balanced, label_smoothing, extra, tta = k
        parts = [
            str(model),
            f"n={int(n_points)}",
            f"k={int(kfold_k)}",
            "bal" if bool(balanced) else "unbal",
            f"ls={float(label_smoothing):g}",
            f"tta={int(tta)}",
            f"extra={extra or '(none)'}",
        ]
        return " ".join(parts)

    # Index groups by family/model/extra for auto comparisons.
    fam_index: dict[tuple[Any, ...], dict[tuple[str, str], tuple[Any, ...]]] = {}
    for gkey in preds_by_group.keys():
        fam = family_of(gkey)
        _dt, model, _np, _k, _bal, _ls, extra, _tta = gkey
        fam_index.setdefault(fam, {})[(str(model), str(extra))] = gkey

    comparisons: list[tuple[tuple[Any, ...], tuple[Any, ...], str]] = []
    for fam, mm in sorted(fam_index.items(), key=lambda x: str(x[0])):
        # Model compare: all pairs within the same extra.
        extras = sorted({extra for (_model, extra) in mm.keys()})
        for extra in extras:
            models = sorted({m for (m, e) in mm.keys() if e == extra})
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    a = mm.get((models[i], extra))
                    b = mm.get((models[j], extra))
                    if a and b:
                        comparisons.append((a, b, "model_pair"))

        # Extra compare: meta vs none (same model).
        none = ""
        for model in sorted({m for (m, _e) in mm.keys()}):
            base = mm.get((model, none))
            if not base:
                continue
            for extra in extras:
                if not extra:
                    continue
                a = mm.get((model, extra))
                if a:
                    comparisons.append((a, base, "extra_vs_none"))

    seen_specs: set[str] = set()
    for a_key, b_key, kind in comparisons:
        spec = f"{kind}:{a_key}->{b_key}"
        if spec in seen_specs:
            continue
        seen_specs.add(spec)

        # Stable per-comparison RNG seed.
        crc = zlib.crc32(spec.encode("utf-8")) & 0xFFFFFFFF
        base_seed = int(args.seed) + int(crc % 1_000_000)

        for metric in ["macro_f1_present", "accuracy"]:
            res = paired_bootstrap_diff(
                a=preds_by_group[a_key],
                b=preds_by_group[b_key],
                metric=metric,
                n_bootstrap=int(args.n_bootstrap),
                seed=base_seed + (0 if metric == "macro_f1_present" else 17),
                warnings=report["warnings"],
            )
            if "error" in res:
                continue
            report["comparisons"].append(
                {
                    "kind": str(kind),
                    "family": {
                        "data_tag": str(family_of(a_key)[0]),
                        "n_points": int(family_of(a_key)[1]),
                        "kfold_k": int(family_of(a_key)[2]),
                        "balanced": bool(family_of(a_key)[3]),
                        "label_smoothing": float(family_of(a_key)[4]),
                        "tta": int(family_of(a_key)[5]),
                    },
                    "a": {"model": str(a_key[1]), "extra_features": str(a_key[6] or "")},
                    "b": {"model": str(b_key[1]), "extra_features": str(b_key[6] or "")},
                    "result": res,
                }
            )

            md_lines.append(
                "| "
                + " | ".join(
                    [
                        metric,
                        f"{float(res['delta_mean']):.4f}",
                        f"[{float(res['delta_ci95']['lo']):.4f},{float(res['delta_ci95']['hi']):.4f}]",
                        f"{float(res['p_two_sided']):.4f}",
                        name_of(a_key),
                        name_of(b_key),
                        str(int(res.get('n_seeds') or 0)),
                        " ".join([f"{k}={v}" for k, v in report["comparisons"][-1]["family"].items()]),
                    ]
                )
                + " |"
            )

    out_prefix = args.out_prefix.resolve() if args.out_prefix.is_absolute() else (Path.cwd() / args.out_prefix).resolve()
    out_json = out_prefix.with_suffix(".json")
    out_md = out_prefix.with_suffix(".md")
    write_json(out_json, report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_md}")
    print(f"[OK] wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
