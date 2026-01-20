#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _lib.io import read_json
from _lib.time import utc_now_iso
from _lib.yaml_cfg import read_yaml


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def fmt_ms(m: float, s: float) -> str:
    return f"{m:.4f}±{s:.4f}"


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _parse_int_filter(spec: str) -> set[int] | None:
    raw = str(spec or "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.add(int(s))
        except Exception as exc:
            raise SystemExit(f"Invalid int filter item: {s!r} (spec={raw!r})") from exc
    return out or None


@dataclass(frozen=True)
class Run:
    run_dir: Path
    rel: str
    dataset: str
    task: str
    domain_tag: str
    exp_name: str
    model: str
    fold: int
    seed: int
    point_features: str
    extra_features: str
    balanced: bool
    label_smoothing: float
    tta: int
    test_acc: float
    test_macro_f1: float
    test_bal_acc: float
    test_ece: float
    test_brier: float
    test_nll: float
    # prep2target (regression) metrics
    test_total: float
    test_chamfer: float
    test_margin: float
    test_occlusion: float
    lambda_margin: float
    lambda_occlusion: float
    occlusion_clearance: float
    n_points: int
    latent_dim: int
    cond_label: bool


def infer_task(root: Path, *, task_arg: str) -> str:
    t = str(task_arg or "").strip().lower()
    if t and t != "auto":
        return t
    parts = {p.lower() for p in root.parts}
    if "raw_cls" in parts:
        return "raw_cls"
    if "domain_shift" in parts:
        return "domain_shift"
    if "prep2target" in parts:
        return "prep2target"
    return "unknown"


def parse_layout(root: Path, run_dir: Path, *, task: str) -> tuple[str, str, str, int, int]:
    rel_parts = run_dir.relative_to(root).parts

    fold = -1
    seed = -1
    exp_name = ""
    model = ""
    domain_tag = ""

    def _parse_int(prefix: str, s: str) -> int:
        if not s.startswith(prefix):
            return -1
        try:
            return int(s[len(prefix) :])
        except Exception:
            return -1

    if len(rel_parts) >= 2 and rel_parts[-1].startswith("seed=") and rel_parts[-2].startswith("fold="):
        seed = _parse_int("seed=", rel_parts[-1])
        fold = _parse_int("fold=", rel_parts[-2])
        model = rel_parts[-3] if len(rel_parts) >= 3 else ""
        exp_name = rel_parts[-4] if len(rel_parts) >= 4 else ""
        if task == "domain_shift" and len(rel_parts) >= 5:
            domain_tag = rel_parts[-5]
        return domain_tag, exp_name, model, fold, seed

    # seed-only layout: .../<exp>/<model>/seed=<seed>
    if len(rel_parts) >= 1 and rel_parts[-1].startswith("seed="):
        seed = _parse_int("seed=", rel_parts[-1])
        fold = -1
        model = rel_parts[-2] if len(rel_parts) >= 2 else ""
        exp_name = rel_parts[-3] if len(rel_parts) >= 3 else ""
        if task == "domain_shift" and len(rel_parts) >= 4:
            domain_tag = rel_parts[-4]
    return domain_tag, exp_name, model, fold, seed


def load_run(root: Path, metrics_path: Path, *, task: str) -> Run | None:
    run_dir = metrics_path.parent
    rel = str(run_dir.relative_to(root))

    cfg_yaml = run_dir / "config.yaml"
    cfg_json = run_dir / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_yaml.exists():
        try:
            cfg = read_yaml(cfg_yaml)
        except Exception:
            cfg = {}
    elif cfg_json.exists():
        try:
            cfg = read_json(cfg_json)
        except Exception:
            cfg = {}

    try:
        met = read_json(metrics_path)
    except Exception:
        return None

    domain_tag, exp_name, model, fold, seed = parse_layout(root, run_dir, task=task)
    if not exp_name:
        exp_name = str(_get(cfg, "exp.name", _get(cfg, "exp_name", "")) or "")
    if not model:
        model = str(_get(cfg, "model.name", _get(cfg, "model", "")) or "")
    if seed < 0:
        seed = int(_get(cfg, "repro.seed", _get(cfg, "seed", -1)) or -1)
    if fold < 0:
        fold = int(_get(cfg, "repro.fold", _get(cfg, "kfold_test_fold", -1)) or -1)

    dataset = str(_get(cfg, "data.version", "") or "")
    if not dataset:
        data_root = str(_get(cfg, "data.root", _get(cfg, "data_root", "")) or "")
        dataset = Path(data_root).name if data_root else ""

    point_features = _get(cfg, "features.point_features", "") or ""
    if isinstance(point_features, list):
        point_features = ",".join(str(x) for x in point_features)
    point_features = str(point_features)
    if not point_features:
        pf2 = (met.get("sanity") or {}).get("point_features") if isinstance(met, dict) else None
        if isinstance(pf2, list):
            point_features = ",".join(str(x) for x in pf2)
        elif isinstance(pf2, str):
            point_features = pf2

    extra_features = _get(cfg, "features.extra_features", _get(cfg, "extra_features", "")) or ""
    if isinstance(extra_features, list):
        extra_features = ",".join(str(x) for x in extra_features)
    extra_features = str(extra_features)

    balanced = bool(_get(cfg, "sampler.balanced", _get(cfg, "balanced_sampler", False)) or False)
    label_smoothing = float(_get(cfg, "loss.label_smoothing", _get(cfg, "label_smoothing", 0.0)) or 0.0)
    tta = int(_get(cfg, "eval.tta", _get(cfg, "tta", 0)) or 0)

    test = met.get("test") or {}
    cal = met.get("test_calibration") or {}
    test_total = _safe_float(test.get("total"))
    test_chamfer = _safe_float(test.get("chamfer"))
    test_margin = _safe_float(test.get("margin"))
    test_occlusion = _safe_float(test.get("occlusion"))
    lambda_margin = float(_get(cfg, "constraints.lambda_margin", 0.0) or 0.0)
    lambda_occlusion = float(_get(cfg, "constraints.lambda_occlusion", 0.0) or 0.0)
    occlusion_clearance = float(_get(cfg, "constraints.occlusion_clearance", 0.0) or 0.0)
    n_points = int(_get(cfg, "data.n_points", _get(cfg, "n_points", 0)) or 0)
    latent_dim = int(_get(cfg, "model.latent_dim", 0) or 0)
    cond_label = bool(_get(cfg, "model.cond_label", False) or False)
    return Run(
        run_dir=run_dir,
        rel=rel,
        dataset=str(dataset),
        task=str(task),
        domain_tag=str(domain_tag),
        exp_name=str(exp_name),
        model=str(model),
        fold=int(fold),
        seed=int(seed),
        point_features=str(point_features),
        extra_features=str(extra_features),
        balanced=bool(balanced),
        label_smoothing=float(label_smoothing),
        tta=int(tta),
        test_acc=_safe_float(test.get("accuracy")),
        test_macro_f1=_safe_float(test.get("macro_f1_present")),
        test_bal_acc=_safe_float(test.get("balanced_accuracy_present")),
        test_ece=_safe_float(cal.get("ece")),
        test_brier=_safe_float(cal.get("brier")),
        test_nll=_safe_float(cal.get("nll")),
        test_total=float(test_total),
        test_chamfer=float(test_chamfer),
        test_margin=float(test_margin),
        test_occlusion=float(test_occlusion),
        lambda_margin=float(lambda_margin),
        lambda_occlusion=float(lambda_occlusion),
        occlusion_clearance=float(occlusion_clearance),
        n_points=int(n_points),
        latent_dim=int(latent_dim),
        cond_label=bool(cond_label),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate nested run trees into a compact paper-style markdown table.")
    ap.add_argument("--root", type=Path, required=True, help="Run tree root (e.g., runs/raw_cls/v13_main4).")
    ap.add_argument("--out", type=Path, required=True, help="Output markdown path.")
    ap.add_argument("--task", type=str, default="auto", help="auto|raw_cls|domain_shift|prep2target|unknown")
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated seed filter (e.g., 1337,2020,2021). Empty means all.",
    )
    ap.add_argument(
        "--folds",
        type=str,
        default="",
        help="Optional comma-separated fold filter (e.g., 0,1,2,3,4). Empty means all.",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    task = infer_task(root, task_arg=str(args.task))
    metrics_paths = sorted(root.rglob("metrics.json"))
    runs: list[Run] = []
    for mp in metrics_paths:
        r = load_run(root, mp, task=task)
        if r is not None:
            runs.append(r)

    seed_filter = _parse_int_filter(str(args.seeds))
    fold_filter = _parse_int_filter(str(args.folds))
    if seed_filter is not None:
        runs = [r for r in runs if int(r.seed) in seed_filter]
    if fold_filter is not None:
        runs = [r for r in runs if int(r.fold) in fold_filter]

    out_lines: list[str] = []
    out_lines.append(f"# Aggregate runs ({task})")
    out_lines.append("")
    out_lines.append(f"- generated_at: {utc_now_iso()}")
    out_lines.append(f"- root: `{root}`")
    out_lines.append(f"- runs: {len(runs)}")
    if seed_filter is not None:
        out_lines.append(f"- seed_filter: {sorted(seed_filter)}")
    if fold_filter is not None:
        out_lines.append(f"- fold_filter: {sorted(fold_filter)}")
    out_lines.append("")

    if task == "prep2target":
        groups2: dict[tuple[Any, ...], list[Run]] = {}
        for r in runs:
            key = (
                r.dataset,
                r.exp_name,
                r.model,
                int(r.n_points),
                int(r.latent_dim),
                "cond" if r.cond_label else "plain",
                f"lm={r.lambda_margin:g}",
                f"lo={r.lambda_occlusion:g}",
                f"clr={r.occlusion_clearance:g}",
            )
            groups2.setdefault(key, []).append(r)

        out_lines.append(f"- groups: {len(groups2)}")
        out_lines.append("")
        out_lines.append("| test_total (mean±std) | test_chamfer (mean±std) | test_margin (mean±std) | test_occlusion (mean±std) | n | dataset | exp | model | n_points | latent_dim | cond_label | lambda_margin | lambda_occlusion | clearance |")
        out_lines.append("|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|")

        rows2: list[tuple[float, float, int, tuple[Any, ...]]] = []
        for key, items in groups2.items():
            rows2.append(
                (
                    statistics.mean([x.test_total for x in items]) if items else 0.0,
                    statistics.mean([x.test_chamfer for x in items]) if items else 0.0,
                    len(items),
                    key,
                )
            )
        rows2.sort(key=lambda t: (t[0], t[1], -t[2], t[3]))

        for _m_total, _m_cd, n, key in rows2:
            dataset, exp_name, model, n_points, latent_dim, cond_tag, lm_tag, lo_tag, clr_tag = key
            items = groups2[key]
            tot_m, tot_s = mean_std([x.test_total for x in items])
            cd_m, cd_s = mean_std([x.test_chamfer for x in items])
            m_m, m_s = mean_std([x.test_margin for x in items])
            o_m, o_s = mean_std([x.test_occlusion for x in items])
            out_lines.append(
                "| "
                + " | ".join(
                    [
                        fmt_ms(tot_m, tot_s),
                        fmt_ms(cd_m, cd_s),
                        fmt_ms(m_m, m_s),
                        fmt_ms(o_m, o_s),
                        str(n),
                        str(dataset),
                        str(exp_name),
                        str(model),
                        str(int(n_points)),
                        str(int(latent_dim)),
                        str(cond_tag),
                        str(float(lm_tag.split("=", 1)[1]) if "=" in lm_tag else 0.0),
                        str(float(lo_tag.split("=", 1)[1]) if "=" in lo_tag else 0.0),
                        str(float(clr_tag.split("=", 1)[1]) if "=" in clr_tag else 0.0),
                    ]
                )
                + " |"
            )
    else:
        # raw_cls / domain_shift
        groups: dict[tuple[str, ...], list[Run]] = {}
        for r in runs:
            key = (
                r.dataset,
                r.domain_tag,
                r.exp_name,
                r.model,
                r.point_features or "(default)",
                r.extra_features or "(none)",
                "bal" if r.balanced else "unbal",
                f"ls={r.label_smoothing:g}",
                f"tta={r.tta}",
            )
            groups.setdefault(key, []).append(r)

        rows: list[tuple[float, float, int, tuple[str, ...]]] = []
        for key, items in groups.items():
            rows.append(
                (
                    statistics.mean([x.test_macro_f1 for x in items]) if items else 0.0,
                    statistics.mean([x.test_acc for x in items]) if items else 0.0,
                    len(items),
                    key,
                )
            )
        rows.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3]))

        out_lines.append(f"- groups: {len(groups)}")
        out_lines.append("")
        out_lines.append(
            "| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |"
        )
        out_lines.append("|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|")
        for _m_f1, _m_acc, n, key in rows:
            dataset, domain_tag, exp_name, model, point_features, extra_features, sampler_tag, ls_tag, tta_tag = key
            items = groups[key]
            f1_m, f1_s = mean_std([x.test_macro_f1 for x in items])
            acc_m, acc_s = mean_std([x.test_acc for x in items])
            bal_m, bal_s = mean_std([x.test_bal_acc for x in items])
            ece_m, ece_s = mean_std([x.test_ece for x in items])
            out_lines.append(
                "| "
                + " | ".join(
                    [
                        fmt_ms(f1_m, f1_s),
                        fmt_ms(acc_m, acc_s),
                        fmt_ms(bal_m, bal_s),
                        fmt_ms(ece_m, ece_s),
                        str(n),
                        str(dataset),
                        str(domain_tag or "-"),
                        str(exp_name),
                        str(model),
                        str(point_features),
                        str(extra_features),
                        str(sampler_tag),
                        str(ls_tag),
                        str(int(tta_tag.split("=", 1)[1]) if "=" in tta_tag else 0),
                    ]
                )
                + " |"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
