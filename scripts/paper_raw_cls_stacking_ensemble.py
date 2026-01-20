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


def softmax_np(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    # Standard multiclass macro-F1 + balanced accuracy (mean recall).
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
    return {"macro_f1": float(np.mean(np.asarray(f1s))), "balanced_acc": float(np.mean(np.asarray(recalls)))}


def calibration_basic(probs: np.ndarray, y_true: np.ndarray, *, n_bins: int = 15) -> dict[str, float]:
    if probs.size == 0 or y_true.size == 0:
        return {"ece": 0.0, "nll": 0.0, "brier": 0.0}
    eps = 1e-12
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)

    n_bins_i = max(2, int(n_bins))
    bins = np.linspace(0.0, 1.0, num=n_bins_i + 1)
    ece = 0.0
    for i in range(n_bins_i):
        lo = float(bins[i])
        hi = float(bins[i + 1])
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
    return {"ece": float(ece), "nll": float(nll), "brier": float(brier)}


@dataclass(frozen=True)
class SplitData:
    case_keys: list[str]
    y_true: np.ndarray
    probs_by_member: list[np.ndarray]  # len=members, each [N, C]
    meta: dict[str, list[str]]  # e.g. source/tooth_position


def load_seed_averaged_split(
    *,
    runs_root: Path,
    exp: str,
    model: str,
    fold: int,
    seeds: list[int],
    split: str,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, list[str]]]:
    # Returns (case_keys, y_true, probs_mean_over_seeds, meta)
    # Only keeps cases that appear in all seeds.
    by_case_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    by_case_true: dict[str, int] = {}
    by_case_meta: dict[str, dict[str, str]] = defaultdict(dict)

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
            # Record minimal grouping metadata if present (first seen wins).
            if "source" in r and "source" not in by_case_meta[ck]:
                by_case_meta[ck]["source"] = str(r.get("source") or "(missing)")
            if "tooth_position" in r and "tooth_position" not in by_case_meta[ck]:
                by_case_meta[ck]["tooth_position"] = str(r.get("tooth_position") or "(missing)")

    if missing:
        raise SystemExit(f"Missing preds_{split}.jsonl for some runs (first): {missing[0]} (total missing={len(missing)})")

    case_keys = sorted([ck for ck, plist in by_case_probs.items() if len(plist) == len(seeds)])
    if not case_keys:
        raise SystemExit(f"No cases found for exp={exp} model={model} fold={fold} split={split}")

    probs = np.stack([np.mean(np.stack(by_case_probs[ck], axis=0), axis=0) for ck in case_keys], axis=0)
    y_true = np.asarray([by_case_true[ck] for ck in case_keys], dtype=np.int64)

    meta: dict[str, list[str]] = {}
    for key in ("source", "tooth_position"):
        meta[key] = [str((by_case_meta.get(ck) or {}).get(key) or "(missing)") for ck in case_keys]
    return case_keys, y_true, probs, meta


def align_members_on_case_keys(datas: list[tuple[list[str], np.ndarray, np.ndarray, dict[str, list[str]]]]) -> SplitData:
    # Align N across members by intersection of case_keys, preserving deterministic order.
    if not datas:
        raise ValueError("no member data")
    common = set(datas[0][0])
    for ck, _yt, _pr, _meta in datas[1:]:
        common &= set(ck)
    common_keys = sorted(common)
    if not common_keys:
        raise SystemExit("No common case_keys across members (check runs/preds files).")

    # y_true must match across members.
    y0_map = {k: int(y) for k, y in zip(datas[0][0], datas[0][1].tolist())}
    y_true = np.asarray([y0_map[k] for k in common_keys], dtype=np.int64)

    probs_by_member: list[np.ndarray] = []
    meta: dict[str, list[str]] = {}
    for mi, (case_keys, y, probs, m) in enumerate(datas):
        idx = {k: i for i, k in enumerate(case_keys)}
        y_m = np.asarray([int(y[idx[k]]) for k in common_keys], dtype=np.int64)
        if y_m.shape != y_true.shape or not np.all(y_m == y_true):
            raise SystemExit(f"y_true mismatch after aligning members (member index={mi})")
        probs_by_member.append(np.stack([probs[idx[k]] for k in common_keys], axis=0))
        # meta: take from member0 only (should be identical)
        if mi == 0:
            for key, vals in (m or {}).items():
                meta[key] = [vals[idx[k]] for k in common_keys] if vals else ["(missing)"] * len(common_keys)

    return SplitData(case_keys=common_keys, y_true=y_true, probs_by_member=probs_by_member, meta=meta)


def ensemble_mean(probs_by_member: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probs_by_member, axis=0), axis=0)


def ensemble_weighted(probs_by_member: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != len(probs_by_member):
        raise ValueError("weights size mismatch")
    if np.any(w < 0) or not math.isfinite(float(np.sum(w))):
        raise ValueError("invalid weights")
    s = float(np.sum(w))
    if s <= 0:
        raise ValueError("sum(weights) must be > 0")
    w = w / s
    p = np.zeros_like(probs_by_member[0], dtype=np.float64)
    for wi, pr in zip(w.tolist(), probs_by_member):
        p += float(wi) * pr
    return p


def apply_logit_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64).reshape(1, -1)
    if p.ndim != 2:
        raise ValueError("probs must be 2D")
    if b.shape[1] != p.shape[1]:
        raise ValueError("bias shape mismatch")
    eps = 1e-12
    logits = np.log(np.clip(p, eps, 1.0)) + b
    return softmax_np(logits, axis=1)


def grid_search_logit_bias(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    step: float,
    bias_max: float,
    objective: str,
) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if p.ndim != 2 or y.ndim != 1 or p.shape[0] != y.shape[0]:
        raise ValueError("shape mismatch")
    c = int(p.shape[1])
    if c != 4:
        raise ValueError("logit-bias grid search currently supports exactly 4 classes")
    step = float(step)
    bias_max = float(bias_max)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("--bias-step must be > 0")
    if not math.isfinite(bias_max) or bias_max <= 0:
        raise ValueError("--bias-max must be > 0")
    if objective not in {"nll", "acc"}:
        raise ValueError(f"unsupported objective: {objective}")

    eps = 1e-12

    def score(p2: np.ndarray) -> tuple[float, float]:
        pred = p2.argmax(axis=1)
        acc = float(np.mean((pred == y).astype(np.float64)))
        p_true = p2[np.arange(y.shape[0]), y]
        nll = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
        if objective == "nll":
            return (nll, -acc)  # minimize
        return (-acc, nll)  # minimize

    grid = np.arange(-bias_max, bias_max + 1e-9, step, dtype=np.float64)
    best_bias: np.ndarray | None = None
    best_score: tuple[float, float] | None = None

    # Fix b0=0 to remove softmax invariance to constant shift.
    for b1 in grid:
        for b2 in grid:
            for b3 in grid:
                bias = np.asarray([0.0, float(b1), float(b2), float(b3)], dtype=np.float64)
                p2 = apply_logit_bias(p, bias)
                sc = score(p2)
                if best_score is None or sc < best_score:
                    best_score = sc
                    best_bias = bias

    if best_bias is None:
        raise RuntimeError("bias search failed")
    return best_bias


def grid_search_weights(
    probs_by_member: list[np.ndarray],
    y_true: np.ndarray,
    *,
    step: float,
    objective: str,
) -> np.ndarray:
    m = len(probs_by_member)
    if m < 2 or m > 4:
        raise ValueError("grid search supports 2..4 members")
    step = float(step)
    if step <= 0 or step > 0.5:
        raise ValueError("--grid-step must be in (0, 0.5]")

    eps = 1e-12

    def score(p: np.ndarray) -> tuple[float, float]:
        # (primary, tie_breaker)
        pred = p.argmax(axis=1)
        acc = float(np.mean((pred == y_true).astype(np.float64)))
        p_true = p[np.arange(y_true.shape[0]), y_true]
        nll = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
        if objective == "nll":
            return (nll, -acc)  # minimize
        if objective == "acc":
            return (-acc, nll)  # minimize
        raise ValueError(f"unsupported objective: {objective}")

    best_w: np.ndarray | None = None
    best_score: tuple[float, float] | None = None
    grid = np.arange(0.0, 1.0 + 1e-9, step, dtype=np.float64)

    if m == 2:
        for w0 in grid:
            w = np.asarray([w0, 1.0 - w0], dtype=np.float64)
            p = ensemble_weighted(probs_by_member, w)
            sc = score(p)
            if best_score is None or sc < best_score:
                best_score = sc
                best_w = w
    elif m == 3:
        for w0 in grid:
            for w1 in grid:
                s = float(w0 + w1)
                if s > 1.0 + 1e-9:
                    continue
                w2 = 1.0 - s
                w = np.asarray([w0, w1, w2], dtype=np.float64)
                p = ensemble_weighted(probs_by_member, w)
                sc = score(p)
                if best_score is None or sc < best_score:
                    best_score = sc
                    best_w = w
    else:  # m == 4
        for w0 in grid:
            for w1 in grid:
                for w2 in grid:
                    s = float(w0 + w1 + w2)
                    if s > 1.0 + 1e-9:
                        continue
                    w3 = 1.0 - s
                    w = np.asarray([w0, w1, w2, w3], dtype=np.float64)
                    p = ensemble_weighted(probs_by_member, w)
                    sc = score(p)
                    if best_score is None or sc < best_score:
                        best_score = sc
                        best_w = w

    if best_w is None:
        raise SystemExit("grid search found no feasible weights")
    return best_w


def fit_softmax_stacker(
    *,
    probs_by_member: list[np.ndarray],
    y_true: np.ndarray,
    feature: str,
    l2: float,
    lr: float,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Train a linear softmax classifier on concatenated member features.
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"stacking requires torch (import failed: {e})") from e

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    eps = 1e-12
    feats: list[np.ndarray] = []
    for p in probs_by_member:
        if feature == "probs":
            feats.append(p.astype(np.float32))
        elif feature == "logprobs":
            feats.append(np.log(np.clip(p, eps, 1.0)).astype(np.float32))
        else:
            raise ValueError(f"unsupported --stack-feature: {feature}")
    x = np.concatenate(feats, axis=1)
    y = y_true.astype(np.int64)
    n, d = x.shape
    k = int(probs_by_member[0].shape[1])

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    w = torch.zeros((k, d), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((k,), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=float(lr), weight_decay=float(l2))

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logits = x_t @ w.t() + b
        loss = torch.nn.functional.cross_entropy(logits, y_t)
        loss.backward()
        opt.step()

    w_np = w.detach().cpu().numpy().astype(np.float64)
    b_np = b.detach().cpu().numpy().astype(np.float64)
    return w_np, b_np


def apply_softmax_stacker(*, probs_by_member: list[np.ndarray], w: np.ndarray, b: np.ndarray, feature: str) -> np.ndarray:
    eps = 1e-12
    feats: list[np.ndarray] = []
    for p in probs_by_member:
        if feature == "probs":
            feats.append(p.astype(np.float64))
        elif feature == "logprobs":
            feats.append(np.log(np.clip(p, eps, 1.0)).astype(np.float64))
        else:
            raise ValueError(f"unsupported --stack-feature: {feature}")
    x = np.concatenate(feats, axis=1)
    logits = x @ w.T + b.reshape(1, -1)
    return softmax_np(logits, axis=1)


def eval_probs(probs: np.ndarray, y_true: np.ndarray, *, calib_bins: int) -> dict[str, Any]:
    y_pred = probs.argmax(axis=1).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, num_classes=int(probs.shape[1]))
    cal = calibration_basic(probs, y_true, n_bins=int(calib_bins))
    base = {
        "n": int(y_true.shape[0]),
        "accuracy": float(np.mean((y_pred == y_true).astype(np.float64))),
        "macro_f1": float(metrics_from_confusion(cm)["macro_f1"]),
        "balanced_acc": float(metrics_from_confusion(cm)["balanced_acc"]),
        "ece": float(cal["ece"]),
        "nll": float(cal["nll"]),
        "brier": float(cal["brier"]),
        "confusion_matrix": cm.tolist(),
    }
    return base


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train a per-fold stacking/weighted ensemble on val preds and evaluate on test preds (raw_cls)."
    )
    ap.add_argument("--runs-root", type=Path, required=True, help="Root folder that contains exp/model/fold=*/seed=*/")
    ap.add_argument(
        "--member",
        action="append",
        default=[],
        help="Ensemble member as exp:model (repeatable). Seeds are averaged within each member before ensembling.",
    )
    ap.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds (e.g., 1337,2020,2021).")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated folds to include (default: 0..4).")
    ap.add_argument(
        "--method",
        type=str,
        default="weighted",
        choices=["mean", "weighted", "stacking", "bias"],
        help="Ensemble method: mean / weighted (grid search on val) / stacking (softmax linear on val) / bias (logit-bias on val).",
    )
    ap.add_argument("--objective", type=str, default="nll", choices=["nll", "acc"], help="Val objective for weighted.")
    ap.add_argument("--grid-step", type=float, default=0.02, help="Simplex grid step for weighted (2..4 members).")
    ap.add_argument("--bias-step", type=float, default=0.1, help="Grid step for bias (logit-bias) search (4 classes).")
    ap.add_argument("--bias-max", type=float, default=2.0, help="Bias range is [-bias-max, bias-max] (4 classes).")
    ap.add_argument("--stack-feature", type=str, default="logprobs", choices=["probs", "logprobs"])
    ap.add_argument("--stack-l2", type=float, default=1e-3, help="L2 weight decay for stacking.")
    ap.add_argument("--stack-lr", type=float, default=0.1, help="Learning rate for stacking.")
    ap.add_argument("--stack-steps", type=int, default=2000, help="Optimization steps for stacking.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for stacking optimization.")
    ap.add_argument("--calibration-bins", type=int, default=15, help="Bins for ECE metrics.")
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

    per_fold: list[dict[str, Any]] = []
    all_test_true: list[np.ndarray] = []
    all_test_probs: list[np.ndarray] = []

    weights_by_fold: dict[int, list[float]] = {}
    stacker_by_fold: dict[int, dict[str, Any]] = {}
    bias_by_fold: dict[int, list[float]] = {}

    for fold in folds:
        # Load member seed-ensembled val/test, then align across members.
        val_parts = []
        test_parts = []
        for exp, model in members:
            val_parts.append(
                load_seed_averaged_split(
                    runs_root=runs_root,
                    exp=exp,
                    model=model,
                    fold=int(fold),
                    seeds=seeds,
                    split="val",
                )
            )
            test_parts.append(
                load_seed_averaged_split(
                    runs_root=runs_root,
                    exp=exp,
                    model=model,
                    fold=int(fold),
                    seeds=seeds,
                    split="test",
                )
            )

        val = align_members_on_case_keys(val_parts)
        test = align_members_on_case_keys(test_parts)

        if int(test.y_true.shape[0]) == 0 or int(val.y_true.shape[0]) == 0:
            raise SystemExit(f"empty split after alignment (fold={fold})")

        if args.method == "mean":
            probs_test = ensemble_mean(test.probs_by_member)
            probs_val = ensemble_mean(val.probs_by_member)
        elif args.method == "weighted":
            w = grid_search_weights(val.probs_by_member, val.y_true, step=float(args.grid_step), objective=str(args.objective))
            weights_by_fold[int(fold)] = [float(x) for x in w.tolist()]
            probs_test = ensemble_weighted(test.probs_by_member, w)
            probs_val = ensemble_weighted(val.probs_by_member, w)
        elif args.method == "bias":
            base_val = ensemble_mean(val.probs_by_member)
            base_test = ensemble_mean(test.probs_by_member)
            bias = grid_search_logit_bias(
                base_val,
                val.y_true,
                step=float(args.bias_step),
                bias_max=float(args.bias_max),
                objective=str(args.objective),
            )
            bias_by_fold[int(fold)] = [float(x) for x in bias.tolist()]
            probs_val = apply_logit_bias(base_val, bias)
            probs_test = apply_logit_bias(base_test, bias)
        else:  # stacking
            w_stack, b_stack = fit_softmax_stacker(
                probs_by_member=val.probs_by_member,
                y_true=val.y_true,
                feature=str(args.stack_feature),
                l2=float(args.stack_l2),
                lr=float(args.stack_lr),
                steps=int(args.stack_steps),
                seed=int(args.seed) + int(fold) * 997,
            )
            stacker_by_fold[int(fold)] = {
                "w": w_stack.tolist(),
                "b": b_stack.tolist(),
                "feature": str(args.stack_feature),
                "l2": float(args.stack_l2),
                "lr": float(args.stack_lr),
                "steps": int(args.stack_steps),
            }
            probs_test = apply_softmax_stacker(probs_by_member=test.probs_by_member, w=w_stack, b=b_stack, feature=str(args.stack_feature))
            probs_val = apply_softmax_stacker(probs_by_member=val.probs_by_member, w=w_stack, b=b_stack, feature=str(args.stack_feature))

        fold_out: dict[str, Any] = {"fold": int(fold)}
        fold_out["val"] = eval_probs(probs_val, val.y_true, calib_bins=int(args.calibration_bins))
        fold_out["test"] = eval_probs(probs_test, test.y_true, calib_bins=int(args.calibration_bins))
        per_fold.append(fold_out)

        all_test_true.append(test.y_true)
        all_test_probs.append(probs_test)

    overall_true = np.concatenate(all_test_true, axis=0)
    overall_probs = np.concatenate(all_test_probs, axis=0)
    overall = eval_probs(overall_probs, overall_true, calib_bins=int(args.calibration_bins))

    out = {
        "runs_root": str(runs_root),
        "members": [{"exp": exp, "model": model} for exp, model in members],
        "seeds": seeds,
        "folds": folds,
        "method": str(args.method),
        "objective": str(args.objective) if str(args.method) in {"weighted", "bias"} else None,
        "grid_step": float(args.grid_step) if str(args.method) == "weighted" else None,
        "stacking": stacker_by_fold if str(args.method) == "stacking" else None,
        "weights": weights_by_fold if str(args.method) == "weighted" else None,
        "bias": bias_by_fold if str(args.method) == "bias" else None,
        "bias_step": float(args.bias_step) if str(args.method) == "bias" else None,
        "bias_max": float(args.bias_max) if str(args.method) == "bias" else None,
        "per_fold": per_fold,
        "overall": overall,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
