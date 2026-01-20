#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _lib.yaml_cfg import deep_merge, interpolate_cfg, load_with_defaults, set_nested, write_yaml


def write_preds_csv_from_jsonl(jsonl_path: Path, out_csv: Path) -> None:
    if not jsonl_path.is_file():
        return
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_key",
                "split",
                "label",
                "y_true",
                "y_pred",
                "p_pred",
                "sample_npz",
            ],
        )
        w.writeheader()
        for r in rows:
            probs = r.get("probs") or []
            y_pred = int(r.get("y_pred") or 0)
            p_pred = float(probs[y_pred]) if isinstance(probs, list) and 0 <= y_pred < len(probs) else 0.0
            w.writerow(
                {
                    "case_key": r.get("case_key"),
                    "split": r.get("split"),
                    "label": r.get("label"),
                    "y_true": r.get("y_true"),
                    "y_pred": r.get("y_pred"),
                    "p_pred": f"{p_pred:.6f}",
                    "sample_npz": r.get("sample_npz"),
                }
            )


def postprocess_run_dir(run_dir: Path) -> None:
    # Match README artifact naming.
    best = run_dir / "model_best.pt"
    ckpt = run_dir / "ckpt_best.pt"
    if best.exists() and not ckpt.exists():
        try:
            ckpt.symlink_to(best.name)
        except Exception:
            ckpt.write_bytes(best.read_bytes())

    preds_jsonl = run_dir / "preds_test.jsonl"
    preds_csv = run_dir / "preds.csv"
    if preds_jsonl.exists() and not preds_csv.exists():
        write_preds_csv_from_jsonl(preds_jsonl, preds_csv)


def _run_temp_scaling(*, cfg: dict[str, Any], repo_root: Path, run_dir: Path) -> int:
    cal = cfg.get("calibration") or {}
    if not bool(cal.get("temperature_scaling") or False):
        return 0

    preds_val = run_dir / "preds_val.jsonl"
    preds_test = run_dir / "preds_test.jsonl"
    if not preds_val.is_file() or not preds_test.is_file():
        raise SystemExit(f"temperature_scaling requested but missing preds in: {run_dir}")

    bins = int(cal.get("bins") or (cfg.get("eval") or {}).get("calibration_bins") or 15)
    cmd = [
        sys.executable,
        str((repo_root / "scripts" / "raw_cls_temperature_scaling.py").resolve()),
        "--run-dir",
        str(run_dir),
        "--bins",
        str(bins),
    ]
    return run_subprocess(cmd, cwd=repo_root, log_path=run_dir / "temp_scaling.txt")


def _run_selective_eval(*, cfg: dict[str, Any], repo_root: Path, run_dir: Path) -> int:
    eval_cfg = cfg.get("eval") or {}
    sel = eval_cfg.get("selective") if isinstance(eval_cfg, dict) else None
    if not isinstance(sel, dict) or not bool(sel.get("enabled") or False):
        return 0

    preds_test = run_dir / "preds_test.jsonl"
    if not preds_test.is_file():
        raise SystemExit(f"selective eval requested but missing preds_test.jsonl in: {run_dir}")

    cov = sel.get("coverages")
    if isinstance(cov, list):
        cov_str = ",".join(str(x) for x in cov)
    else:
        cov_str = str(cov or "1.0,0.9,0.8,0.7")

    bins = int(
        sel.get("bins")
        or eval_cfg.get("calibration_bins")
        or (cfg.get("calibration") or {}).get("bins")
        or 15
    )
    cmd = [
        sys.executable,
        str((repo_root / "scripts" / "raw_cls_selective_eval.py").resolve()),
        "--run-dir",
        str(run_dir),
        "--coverages",
        cov_str,
        "--bins",
        str(bins),
    ]
    if bool(sel.get("use_calibrated") or False):
        cmd.append("--use-calibrated")
    return run_subprocess(cmd, cwd=repo_root, log_path=run_dir / "selective.txt")


def run_subprocess(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        return int(p.wait())


def ensure_fresh_run_dir(run_dir: Path) -> bool:
    """Ensure we can write a new run into `run_dir`.

    Returns True if a new run should be executed.
    Returns False if the run appears complete and can be skipped.
    """
    metrics = run_dir / "metrics.json"
    if metrics.exists():
        return False
    if run_dir.exists():
        # Preserve incomplete outputs and keep the canonical path for the retry.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = run_dir.with_name(run_dir.name + f"_incomplete_{ts}")
        run_dir.rename(backup)
    run_dir.mkdir(parents=True, exist_ok=True)
    return True


def _to_bool(x: Any, *, default: bool = False) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _to_float(x: Any, *, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _maybe_augment_scale_from_minmax(augment: dict[str, Any]) -> dict[str, Any]:
    if "scale" in augment and augment.get("scale") is not None:
        return augment
    smin = augment.get("scale_min")
    smax = augment.get("scale_max")
    if smin is None and smax is None:
        return augment
    lo = _to_float(smin, default=1.0)
    hi = _to_float(smax, default=1.0)
    lo, hi = (lo, hi) if lo <= hi else (hi, lo)
    s = max(0.0, abs(1.0 - lo), abs(hi - 1.0))
    out = dict(augment)
    out["scale"] = float(s)
    return out


def _normalize_augment_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    augment = dict(cfg.get("augment") or {})
    # README template compatibility: rotate_z_deg -> rotate_z (bool).
    if ("rotate_z" not in augment or augment.get("rotate_z") is None) and "rotate_z_deg" in augment:
        deg = _to_float(augment.get("rotate_z_deg"), default=0.0)
        augment["rotate_z"] = bool(deg > 0.0)
    augment = _maybe_augment_scale_from_minmax(augment)
    return augment


def _normalize_precompute_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    features = cfg.get("features") or {}
    pre = {}
    if isinstance(features, dict):
        pre = features.get("precompute") or {}
    # README template compatibility: root-level precompute.*
    if not pre:
        pre = cfg.get("precompute") or {}
    return dict(pre) if isinstance(pre, dict) else {}


def _normalize_features_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    features = dict(cfg.get("features") or {})

    # README template compatibility: use_normals/use_curvature/use_radius -> point_features list.
    if "point_features" not in features:
        pf = ["xyz"]
        if _to_bool(features.get("use_normals"), default=False):
            pf.append("normals")
        if _to_bool(features.get("use_curvature"), default=False):
            pf.append("curvature")
        if _to_bool(features.get("use_radius"), default=False):
            pf.append("radius")
        features["point_features"] = pf

    # README template compatibility: use_global_scale_token/global_scale_fields -> extra_features
    extra = features.get("extra_features") or ""
    extra_list: list[str] = []
    if isinstance(extra, list):
        extra_list = [str(x).strip() for x in extra if str(x).strip()]
    else:
        extra_list = [s.strip() for s in str(extra).split(",") if s.strip()]

    if not extra_list and _to_bool(features.get("use_global_scale_token"), default=False):
        fields = features.get("global_scale_fields") or ["scale"]
        if isinstance(fields, list):
            cand = [str(x).strip() for x in fields if str(x).strip()]
        else:
            cand = [s.strip() for s in str(fields).split(",") if s.strip()]
        # This repo only supports a small, auditable set of meta features.
        allowed = {"scale", "log_scale", "points", "log_points", "objects_used"}
        extra_list = [x for x in cand if x in allowed]
        if not extra_list and "scale" in cand:
            extra_list = ["scale"]

    if extra_list:
        features["extra_features"] = extra_list
    return features


def run_raw_cls(cfg: dict[str, Any], *, repo_root: Path) -> int:
    data = cfg.get("data") or {}
    exp = cfg.get("exp") or {}
    repro = cfg.get("repro") or {}
    runtime = cfg.get("runtime") or {}
    optim = cfg.get("optim") or {}
    train = cfg.get("train") or {}
    logging_cfg = cfg.get("logging") or {}
    loss = cfg.get("loss") or {}
    sampler = cfg.get("sampler") or {}
    augment = _normalize_augment_cfg(cfg)
    model = cfg.get("model") or {}
    features = _normalize_features_cfg(cfg)
    precompute = _normalize_precompute_cfg(cfg)
    domain = cfg.get("domain") or {}
    calibration = cfg.get("calibration") or {}

    version = str(data.get("version") or "").strip()
    data_root = str(data.get("root") or "").strip()
    if not version:
        raise SystemExit("Missing config: data.version")
    if not data_root:
        raise SystemExit("Missing config: data.root")

    exp_name = str(exp.get("name") or "").strip() or "baseline"
    out_root = str(exp.get("out_root") or "").strip() or "runs/raw_cls"

    model_name = str(model.get("name") or "pointnet").strip()
    fold = int(repro.get("fold") if repro.get("fold") is not None else 0)
    seed = int(repro.get("seed") if repro.get("seed") is not None else 1337)

    model_root = (repo_root / out_root / version / exp_name / model_name).resolve()
    run_rel = f"fold={fold}/seed={seed}"
    run_dir = (model_root / run_rel).resolve()
    run_needed = ensure_fresh_run_dir(run_dir)

    if run_needed:
        cfg_out = dict(cfg)
        set_nested(cfg_out, "resolved.run_dir", str(run_dir))
        # Persist effective config after README-template compatibility mapping.
        set_nested(cfg_out, "augment", dict(augment))
        set_nested(cfg_out, "features", dict(features))
        if precompute:
            set_nested(cfg_out, "features.precompute", dict(precompute))
        write_yaml(run_dir / "config.yaml", cfg_out)

        eval_cfg = cfg.get("eval") or {}
        calib_bins = int(eval_cfg.get("calibration_bins") or calibration.get("bins") or 15) if isinstance(eval_cfg, dict) else 15

        supcon_w = float(loss.get("supcon_weight") or 0.0)
        two_views = bool(supcon_w > 0) and isinstance(augment.get("view1"), dict) and isinstance(augment.get("view2"), dict)

        cmd = [
            sys.executable,
            str((repo_root / "scripts" / "phase3_train_raw_cls_baseline.py").resolve()),
            "--data-root",
            data_root,
            "--run-root",
            str(model_root),
            "--exp-name",
            run_rel,
            "--seed",
            str(seed),
            "--device",
            str(runtime.get("device") or "auto"),
            "--lr-scheduler",
            str(optim.get("scheduler") or "none"),
            "--warmup-epochs",
            str(int(optim.get("warmup_epochs") or 0)),
            "--min-lr",
            str(float(optim.get("min_lr") or 0.0)),
            "--model",
            model_name,
            "--epochs",
            str(int(train.get("epochs") if train.get("epochs") is not None else 120)),
            "--patience",
            str(int(train.get("patience") if train.get("patience") is not None else 25)),
            "--early-stop-min-delta",
            str(float(train.get("early_stop_min_delta") if train.get("early_stop_min_delta") is not None else 1e-6)),
            "--save-best-metric",
            str(train.get("save_best_metric") if train.get("save_best_metric") is not None else "macro_f1_present"),
            "--batch-size",
            str(int(train.get("batch_size") if train.get("batch_size") is not None else 64)),
            "--grad-clip",
            str(float(train.get("grad_clip") if train.get("grad_clip") is not None else (train.get("grad_clip_norm") or 0.0))),
            "--log-every",
            str(int(train.get("log_every") if train.get("log_every") is not None else (logging_cfg.get("log_every") or 0))),
            "--lr",
            str(float(optim.get("lr") if optim.get("lr") is not None else 1e-3)),
            "--weight-decay",
            str(float(optim.get("weight_decay") if optim.get("weight_decay") is not None else 1e-4)),
            "--dropout",
            str(float(train.get("dropout") if train.get("dropout") is not None else 0.3)),
            "--n-points",
            str(int(data.get("n_points") if data.get("n_points") is not None else 4096)),
            "--num-workers",
            str(int(runtime.get("num_workers") if runtime.get("num_workers") is not None else 2)),
            "--label-smoothing",
            str(float(loss.get("label_smoothing") if loss.get("label_smoothing") is not None else 0.0)),
            "--ce-weighting",
            str(loss.get("ce_weighting") if loss.get("ce_weighting") is not None else "auto"),
            "--tta",
            str(int(eval_cfg.get("tta") if isinstance(eval_cfg, dict) and eval_cfg.get("tta") is not None else 0)),
            "--calibration-bins",
            str(calib_bins),
        ]
        det = _to_bool(runtime.get("deterministic"), default=True)
        bench = _to_bool(runtime.get("cudnn_benchmark"), default=False)
        if not det:
            cmd.append("--no-deterministic")
            if bench:
                cmd.append("--cudnn-benchmark")
        if model_name == "dgcnn":
            dgcnn_k = int(model.get("dgcnn_k") or 0)
            if dgcnn_k > 0:
                cmd += ["--dgcnn-k", str(dgcnn_k)]
        if model_name == "pointnet2":
            sa1_npoint = int(model.get("pointnet2_sa1_npoint") or 0)
            sa1_nsample = int(model.get("pointnet2_sa1_nsample") or 0)
            sa2_npoint = int(model.get("pointnet2_sa2_npoint") or 0)
            sa2_nsample = int(model.get("pointnet2_sa2_nsample") or 0)
            if sa1_npoint > 0:
                cmd += ["--pointnet2-sa1-npoint", str(sa1_npoint)]
            if sa1_nsample > 0:
                cmd += ["--pointnet2-sa1-nsample", str(sa1_nsample)]
            if sa2_npoint > 0:
                cmd += ["--pointnet2-sa2-npoint", str(sa2_npoint)]
            if sa2_nsample > 0:
                cmd += ["--pointnet2-sa2-nsample", str(sa2_nsample)]
        if model_name == "point_transformer":
            pt_dim = int(model.get("pt_dim") or 0)
            pt_depth = int(model.get("pt_depth") or 0)
            pt_k = int(model.get("pt_k") or 0)
            pt_ffn_mult = float(model.get("pt_ffn_mult") or 0.0)
            if pt_dim > 0:
                cmd += ["--pt-dim", str(pt_dim)]
            if pt_depth > 0:
                cmd += ["--pt-depth", str(pt_depth)]
            if pt_k > 0:
                cmd += ["--pt-k", str(pt_k)]
            if pt_ffn_mult > 0:
                cmd += ["--pt-ffn-mult", str(pt_ffn_mult)]
        if model_name == "pointmlp":
            pmlp_dim = int(model.get("pmlp_dim") or 0)
            pmlp_depth = int(model.get("pmlp_depth") or 0)
            pmlp_k = int(model.get("pmlp_k") or 0)
            pmlp_ffn_mult = float(model.get("pmlp_ffn_mult") or 0.0)
            if pmlp_dim > 0:
                cmd += ["--pmlp-dim", str(pmlp_dim)]
            if pmlp_depth > 0:
                cmd += ["--pmlp-depth", str(pmlp_depth)]
            if pmlp_k > 0:
                cmd += ["--pmlp-k", str(pmlp_k)]
            if pmlp_ffn_mult > 0:
                cmd += ["--pmlp-ffn-mult", str(pmlp_ffn_mult)]

        kfold = str(data.get("kfold") or data.get("split_case_kfold_json") or "").strip()
        if kfold:
            cmd += ["--kfold", kfold, "--fold", str(fold)]

        if bool(sampler.get("balanced") or False):
            cmd.append("--balanced-sampler")

        # For SupCon with explicit view1/view2 aug, disable dataset aug to avoid double-augmentation.
        if two_views:
            cmd.append("--no-aug-rotate-z")
            cmd += ["--aug-scale", "0"]
            cmd += ["--aug-jitter-sigma", "0"]
            cmd += ["--aug-jitter-clip", "0"]
            cmd += ["--aug-dropout-ratio", "0"]
        else:
            if bool(augment.get("rotate_z", True)) is False:
                cmd.append("--no-aug-rotate-z")
            cmd += ["--aug-scale", str(float(augment.get("scale") if augment.get("scale") is not None else 0.0))]
            cmd += ["--aug-jitter-sigma", str(float(augment.get("jitter_sigma") if augment.get("jitter_sigma") is not None else 0.01))]
            cmd += ["--aug-jitter-clip", str(float(augment.get("jitter_clip") if augment.get("jitter_clip") is not None else 0.05))]
            cmd += ["--aug-dropout-ratio", str(float(augment.get("dropout_ratio") if augment.get("dropout_ratio") is not None else 0.1))]

        extra_features = features.get("extra_features") or ""
        if isinstance(extra_features, list):
            extra_features = ",".join(str(x) for x in extra_features if str(x).strip())
        extra_features = str(extra_features).strip()
        if extra_features:
            cmd += ["--extra-features", extra_features]

        point_features = features.get("point_features") or ""
        if isinstance(point_features, list):
            point_features = ",".join(str(x) for x in point_features if str(x).strip())
        point_features = str(point_features).strip()
        if point_features and point_features.lower() not in {"xyz", "x,y,z"}:
            cmd += ["--point-features", point_features]

        input_normalize = str(data.get("input_normalize") or "").strip()
        if input_normalize:
            cmd += ["--input-normalize", input_normalize]
        if bool(data.get("input_pca_align") or data.get("pca_align") or False):
            cmd.append("--input-pca-align")
        if bool(data.get("input_pca_align_globalz") or data.get("pca_align_globalz") or False):
            cmd.append("--input-pca-align-globalz")

        if bool(precompute.get("enabled") or False):
            cmd += ["--precompute-features"]
            cache_dir = str(precompute.get("cache_dir") or "").strip()
            if cache_dir:
                cmd += ["--feature-cache-dir", cache_dir]
            k_neighbors = int(precompute.get("k_neighbors") or 0)
            if k_neighbors > 0:
                cmd += ["--feature-k", str(k_neighbors)]

        init_feat = str(train.get("init_feat") or "").strip()
        if init_feat:
            init_path = (repo_root / init_feat).resolve() if not Path(init_feat).is_absolute() else Path(init_feat).resolve()
            if not init_path.is_file():
                raise SystemExit(f"Missing init_feat checkpoint: {init_path} (config: train.init_feat)")
            cmd += ["--init-feat", str(init_path)]
        freeze_epochs = int(train.get("freeze_feat_epochs") or 0)
        if freeze_epochs > 0:
            cmd += ["--freeze-feat-epochs", str(freeze_epochs)]

        if supcon_w and supcon_w > 0:
            cmd += ["--supcon-weight", str(float(supcon_w))]
            # README template compatibility: temperature/proj_dim.
            cmd += ["--supcon-temp", str(float(loss.get("supcon_temperature") or loss.get("temperature") or 0.07))]
            cmd += ["--supcon-proj-dim", str(int(loss.get("supcon_proj_dim") or loss.get("proj_dim") or 128))]
            if two_views:
                v1 = dict(augment.get("view1") or {})
                v2 = dict(augment.get("view2") or {})
                v1 = _maybe_augment_scale_from_minmax(v1)
                v2 = _maybe_augment_scale_from_minmax(v2)
                cmd += ["--supcon-aug-rotate-z", str(bool(augment.get("rotate_z", True))).lower()]
                cmd += ["--supcon-aug-scale", str(float(augment.get("scale") if augment.get("scale") is not None else 0.0))]
                cmd += ["--supcon-aug-jitter-sigma1", str(float(v1.get("jitter_sigma") if v1.get("jitter_sigma") is not None else augment.get("jitter_sigma") or 0.01))]
                cmd += ["--supcon-aug-jitter-sigma2", str(float(v2.get("jitter_sigma") if v2.get("jitter_sigma") is not None else augment.get("jitter_sigma") or 0.01))]
                cmd += ["--supcon-aug-jitter-clip1", str(float(v1.get("jitter_clip") if v1.get("jitter_clip") is not None else augment.get("jitter_clip") or 0.05))]
                cmd += ["--supcon-aug-jitter-clip2", str(float(v2.get("jitter_clip") if v2.get("jitter_clip") is not None else augment.get("jitter_clip") or 0.05))]
                cmd += ["--supcon-aug-dropout-ratio1", str(float(v1.get("dropout_ratio") if v1.get("dropout_ratio") is not None else augment.get("dropout_ratio") or 0.1))]
                cmd += ["--supcon-aug-dropout-ratio2", str(float(v2.get("dropout_ratio") if v2.get("dropout_ratio") is not None else augment.get("dropout_ratio") or 0.1))]

        tp_drop = float(domain.get("tooth_position_dropout") or 0.0)
        if tp_drop > 0:
            cmd += ["--tooth-position-dropout", str(tp_drop)]

        method = str(domain.get("method") or "").strip().lower()
        if method:
            cmd += ["--domain-method", method]
            group_key = str(domain.get("group_key") or "").strip()
            if group_key:
                cmd += ["--domain-group-key", group_key]
            if method == "groupdro":
                cmd += ["--groupdro-eta", str(float(domain.get("groupdro_eta") or 0.1))]
            if method == "coral":
                cmd += ["--coral-weight", str(float(domain.get("coral_weight") or 0.0))]
                cmd += ["--coral-proj-dim", str(int(domain.get("coral_proj_dim") or 128))]

        rc = run_subprocess(cmd, cwd=repo_root, log_path=run_dir / "logs.txt")
        if rc != 0:
            return rc

    postprocess_run_dir(run_dir)
    rc = _run_temp_scaling(cfg=cfg, repo_root=repo_root, run_dir=run_dir)
    if rc != 0:
        return rc
    return _run_selective_eval(cfg=cfg, repo_root=repo_root, run_dir=run_dir)


def run_prep2target(cfg: dict[str, Any], *, repo_root: Path) -> int:
    data = cfg.get("data") or {}
    exp = cfg.get("exp") or {}
    repro = cfg.get("repro") or {}
    runtime = cfg.get("runtime") or {}
    optim = cfg.get("optim") or {}
    train = cfg.get("train") or {}
    model = cfg.get("model") or {}
    constraints = cfg.get("constraints") or {}
    aux = cfg.get("aux") or {}

    version = str(data.get("version") or "").strip()
    data_root = str(data.get("root") or "").strip()
    if not version:
        raise SystemExit("Missing config: data.version")
    if not data_root:
        raise SystemExit("Missing config: data.root")

    exp_name = str(exp.get("name") or "").strip() or "baseline"
    out_root = str(exp.get("out_root") or "").strip() or "runs/prep2target"

    seed = int(repro.get("seed") if repro.get("seed") is not None else 1337)

    latent_dim = int(model.get("latent_dim") if model.get("latent_dim") is not None else 256)
    cond_label = bool(model.get("cond_label") or False)
    init_ckpt = str(model.get("init_ckpt") or "").strip()
    model_name = "p2t_cond" if cond_label else "p2t"
    model_name = str(model.get("name") or model_name).strip()

    model_root = (repo_root / out_root / version / exp_name / model_name).resolve()
    run_rel = f"seed={seed}"
    run_dir = (model_root / run_rel).resolve()
    run_needed = ensure_fresh_run_dir(run_dir)

    cfg_out = dict(cfg)
    set_nested(cfg_out, "resolved.run_dir", str(run_dir))
    if run_needed or not (run_dir / "config.yaml").exists():
        write_yaml(run_dir / "config.yaml", cfg_out)

    cmd = [
        sys.executable,
        str((repo_root / "scripts" / "phase4_train_raw_prep2target_finetune.py").resolve()),
        "--root",
        str(repo_root),
        "--data-root",
        data_root,
        "--device",
        str(runtime.get("device") or "auto"),
        "--seed",
        str(seed),
        "--n-points",
        str(int(data.get("n_points") if data.get("n_points") is not None else 512)),
        "--latent-dim",
        str(int(latent_dim)),
        "--batch-size",
        str(int(train.get("batch_size") if train.get("batch_size") is not None else 32)),
        "--epochs",
        str(int(train.get("epochs") if train.get("epochs") is not None else 200)),
        "--lr",
        str(float(optim.get("lr") if optim.get("lr") is not None else 1e-3)),
        "--weight-decay",
        str(float(optim.get("weight_decay") if optim.get("weight_decay") is not None else 1e-4)),
        "--num-workers",
        str(int(runtime.get("num_workers") if runtime.get("num_workers") is not None else 2)),
        "--patience",
        str(int(train.get("patience") if train.get("patience") is not None else 40)),
        "--lambda-margin",
        str(float(constraints.get("lambda_margin") if constraints.get("lambda_margin") is not None else 0.0)),
        "--lambda-occlusion",
        str(float(constraints.get("lambda_occlusion") if constraints.get("lambda_occlusion") is not None else 0.0)),
        "--occlusion-clearance",
        str(float(constraints.get("occlusion_clearance") if constraints.get("occlusion_clearance") is not None else 0.5)),
        "--aux-weight-margin",
        str(float(aux.get("weight_margin") if aux.get("weight_margin") is not None else 0.0)),
        "--aux-weight-occlusion",
        str(float(aux.get("weight_occlusion") if aux.get("weight_occlusion") is not None else 0.0)),
        "--aux-hidden-dim",
        str(int(aux.get("hidden_dim") if aux.get("hidden_dim") is not None else 128)),
        "--runs-dir",
        str(model_root),
        "--exp-name",
        run_rel,
        "--preview-samples",
        str(int(train.get("preview_samples") if train.get("preview_samples") is not None else 16)),
    ]
    if cond_label:
        cmd.append("--cond-label")
    if init_ckpt:
        cmd += ["--init-ckpt", init_ckpt]

    log_path = run_dir / ("logs.txt" if run_needed else "refresh.txt")
    rc = run_subprocess(cmd, cwd=repo_root, log_path=log_path)
    if rc == 0:
        postprocess_run_dir(run_dir)
    return rc


def run_domain_shift(cfg: dict[str, Any], *, repo_root: Path) -> int:
    data = cfg.get("data") or {}
    exp = cfg.get("exp") or {}
    repro = cfg.get("repro") or {}
    runtime = cfg.get("runtime") or {}
    optim = cfg.get("optim") or {}
    train = cfg.get("train") or {}
    logging_cfg = cfg.get("logging") or {}
    loss = cfg.get("loss") or {}
    sampler = cfg.get("sampler") or {}
    augment = _normalize_augment_cfg(cfg)
    model = dict(cfg.get("model") or {})
    domain = dict(cfg.get("domain") or {})
    features = _normalize_features_cfg(cfg)
    precompute = _normalize_precompute_cfg(cfg)
    calibration = cfg.get("calibration") or {}

    version = str(data.get("version") or "").strip()
    data_root = str(data.get("root") or "").strip()
    if not version:
        raise SystemExit("Missing config: data.version")
    if not data_root:
        raise SystemExit("Missing config: data.root")

    exp_name = str(exp.get("name") or "").strip() or "baseline"
    out_root = str(exp.get("out_root") or "").strip() or "runs/domain_shift"

    # README template compatibility: allow train_source/test_source under `data.*`.
    train_source = str(domain.get("train_source") or data.get("train_source") or "").strip()
    test_source = str(domain.get("test_source") or data.get("test_source") or "").strip()
    if not train_source or not test_source:
        raise SystemExit("domain_shift requires domain.train_source and domain.test_source (or CLI overrides)")

    # README template compatibility: `algo.*` / `position.*` / `model.moe.*` → `domain.*` + model selection.
    algo = cfg.get("algo") or {}
    if isinstance(algo, dict):
        if not str(domain.get("method") or "").strip() and str(algo.get("name") or "").strip():
            domain["method"] = str(algo.get("name") or "").strip()
        if str(domain.get("method") or "").strip().lower() == "groupdro" and domain.get("groupdro_eta") is None and algo.get("eta") is not None:
            domain["groupdro_eta"] = algo.get("eta")
        if str(domain.get("method") or "").strip().lower() == "coral" and domain.get("coral_weight") is None and algo.get("coral_weight") is not None:
            domain["coral_weight"] = algo.get("coral_weight")

    position = cfg.get("position") or {}
    if isinstance(position, dict) and _to_bool(position.get("enabled"), default=False):
        domain.setdefault("method", "pos_moe")
        if domain.get("tooth_position_dropout") is None and position.get("dropout_prob_on_present") is not None:
            domain["tooth_position_dropout"] = position.get("dropout_prob_on_present")

    moe = model.get("moe") if isinstance(model, dict) else None
    if isinstance(moe, dict) and _to_bool(moe.get("enabled"), default=False):
        domain.setdefault("method", "pos_moe")

    method0 = str(domain.get("method") or "").strip().lower()
    if method0 == "dsbn":
        model.setdefault("name", "pointnet_dsbn")
    if method0 == "pos_moe":
        model.setdefault("name", "pointnet_pos_moe")
        extra = features.get("extra_features") or []
        if isinstance(extra, str):
            extra_list = [s.strip() for s in extra.split(",") if s.strip()]
        else:
            extra_list = [str(x).strip() for x in (extra or []) if str(x).strip()]
        for need in ["tooth_position_premolar", "tooth_position_molar", "tooth_position_missing"]:
            if need not in extra_list:
                extra_list.append(need)
        features["extra_features"] = extra_list

    model_name = str(model.get("name") or "pointnet").strip()

    fold = int(repro.get("fold") if repro.get("fold") is not None else 0)
    seed = int(repro.get("seed") if repro.get("seed") is not None else 1337)

    a2b = f"A2B_{train_source}_to_{test_source}"
    model_root = (repo_root / out_root / version / a2b / exp_name / model_name).resolve()
    run_rel = f"fold={fold}/seed={seed}"
    run_dir = (model_root / run_rel).resolve()
    run_needed = ensure_fresh_run_dir(run_dir)

    if run_needed:
        cfg_out = dict(cfg)
        set_nested(cfg_out, "resolved.run_dir", str(run_dir))
        set_nested(cfg_out, "augment", dict(augment))
        set_nested(cfg_out, "features", dict(features))
        set_nested(cfg_out, "domain", dict(domain))
        set_nested(cfg_out, "model", dict(model))
        if precompute:
            set_nested(cfg_out, "features.precompute", dict(precompute))
        write_yaml(run_dir / "config.yaml", cfg_out)

        eval_cfg = cfg.get("eval") or {}
        calib_bins = int(eval_cfg.get("calibration_bins") or calibration.get("bins") or 15) if isinstance(eval_cfg, dict) else 15

        cmd = [
            sys.executable,
            str((repo_root / "scripts" / "phase3_train_raw_cls_baseline.py").resolve()),
            "--data-root",
            data_root,
            "--run-root",
            str(model_root),
            "--exp-name",
            run_rel,
            "--seed",
            str(seed),
            "--device",
            str(runtime.get("device") or "auto"),
            "--lr-scheduler",
            str(optim.get("scheduler") or "none"),
            "--warmup-epochs",
            str(int(optim.get("warmup_epochs") or 0)),
            "--min-lr",
            str(float(optim.get("min_lr") or 0.0)),
            "--model",
            model_name,
            "--epochs",
            str(int(train.get("epochs") if train.get("epochs") is not None else 120)),
            "--patience",
            str(int(train.get("patience") if train.get("patience") is not None else 25)),
            "--save-best-metric",
            str(train.get("save_best_metric") if train.get("save_best_metric") is not None else "macro_f1_present"),
            "--batch-size",
            str(int(train.get("batch_size") if train.get("batch_size") is not None else 32)),
            "--grad-clip",
            str(float(train.get("grad_clip") if train.get("grad_clip") is not None else (train.get("grad_clip_norm") or 0.0))),
            "--log-every",
            str(int(train.get("log_every") if train.get("log_every") is not None else (logging_cfg.get("log_every") or 0))),
            "--lr",
            str(float(optim.get("lr") if optim.get("lr") is not None else 1e-3)),
            "--weight-decay",
            str(float(optim.get("weight_decay") if optim.get("weight_decay") is not None else 1e-4)),
            "--dropout",
            str(float(train.get("dropout") if train.get("dropout") is not None else 0.3)),
            "--n-points",
            str(int(data.get("n_points") if data.get("n_points") is not None else 4096)),
            "--num-workers",
            str(int(runtime.get("num_workers") if runtime.get("num_workers") is not None else 2)),
            "--label-smoothing",
            str(float(loss.get("label_smoothing") if loss.get("label_smoothing") is not None else 0.0)),
            "--ce-weighting",
            str(loss.get("ce_weighting") if loss.get("ce_weighting") is not None else "auto"),
            "--tta",
            str(int(eval_cfg.get("tta") if isinstance(eval_cfg, dict) and eval_cfg.get("tta") is not None else 0)),
            "--calibration-bins",
            str(calib_bins),
            "--source-train",
            train_source,
            "--source-test",
            test_source,
            "--source-val-ratio",
            str(float(domain.get("val_ratio") if domain.get("val_ratio") is not None else 0.1)),
        ]
        det = _to_bool(runtime.get("deterministic"), default=True)
        bench = _to_bool(runtime.get("cudnn_benchmark"), default=False)
        if not det:
            cmd.append("--no-deterministic")
            if bench:
                cmd.append("--cudnn-benchmark")
        if model_name == "dgcnn":
            dgcnn_k = int(model.get("dgcnn_k") or 0)
            if dgcnn_k > 0:
                cmd += ["--dgcnn-k", str(dgcnn_k)]
        if model_name == "pointnet2":
            sa1_npoint = int(model.get("pointnet2_sa1_npoint") or 0)
            sa1_nsample = int(model.get("pointnet2_sa1_nsample") or 0)
            sa2_npoint = int(model.get("pointnet2_sa2_npoint") or 0)
            sa2_nsample = int(model.get("pointnet2_sa2_nsample") or 0)
            if sa1_npoint > 0:
                cmd += ["--pointnet2-sa1-npoint", str(sa1_npoint)]
            if sa1_nsample > 0:
                cmd += ["--pointnet2-sa1-nsample", str(sa1_nsample)]
            if sa2_npoint > 0:
                cmd += ["--pointnet2-sa2-npoint", str(sa2_npoint)]
            if sa2_nsample > 0:
                cmd += ["--pointnet2-sa2-nsample", str(sa2_nsample)]

        source_split_seed = int(domain.get("split_seed") or 0)
        if source_split_seed:
            cmd += ["--source-split-seed", str(source_split_seed)]

        if bool(sampler.get("balanced") or False):
            cmd.append("--balanced-sampler")

        if bool(augment.get("rotate_z", True)) is False:
            cmd.append("--no-aug-rotate-z")
        cmd += ["--aug-scale", str(float(augment.get("scale") if augment.get("scale") is not None else 0.0))]
        cmd += ["--aug-jitter-sigma", str(float(augment.get("jitter_sigma") if augment.get("jitter_sigma") is not None else 0.01))]
        cmd += ["--aug-jitter-clip", str(float(augment.get("jitter_clip") if augment.get("jitter_clip") is not None else 0.05))]
        cmd += ["--aug-dropout-ratio", str(float(augment.get("dropout_ratio") if augment.get("dropout_ratio") is not None else 0.1))]

        extra_features = features.get("extra_features") or ""
        if isinstance(extra_features, list):
            extra_features = ",".join(str(x) for x in extra_features if str(x).strip())
        extra_features = str(extra_features).strip()
        if extra_features:
            cmd += ["--extra-features", extra_features]

        point_features = features.get("point_features") or ""
        if isinstance(point_features, list):
            point_features = ",".join(str(x) for x in point_features if str(x).strip())
        point_features = str(point_features).strip()
        if point_features and point_features.lower() not in {"xyz", "x,y,z"}:
            cmd += ["--point-features", point_features]

        input_normalize = str(data.get("input_normalize") or "").strip()
        if input_normalize:
            cmd += ["--input-normalize", input_normalize]
        if bool(data.get("input_pca_align") or data.get("pca_align") or False):
            cmd.append("--input-pca-align")
        if bool(data.get("input_pca_align_globalz") or data.get("pca_align_globalz") or False):
            cmd.append("--input-pca-align-globalz")

        if bool(precompute.get("enabled") or False):
            cmd += ["--precompute-features"]
            cache_dir = str(precompute.get("cache_dir") or "").strip()
            if cache_dir:
                cmd += ["--feature-cache-dir", cache_dir]
            k_neighbors = int(precompute.get("k_neighbors") or 0)
            if k_neighbors > 0:
                cmd += ["--feature-k", str(k_neighbors)]

        init_feat = str(train.get("init_feat") or "").strip()
        if init_feat:
            init_path = (repo_root / init_feat).resolve() if not Path(init_feat).is_absolute() else Path(init_feat).resolve()
            if not init_path.is_file():
                raise SystemExit(f"Missing init_feat checkpoint: {init_path} (config: train.init_feat)")
            cmd += ["--init-feat", str(init_path)]
        freeze_epochs = int(train.get("freeze_feat_epochs") or 0)
        if freeze_epochs > 0:
            cmd += ["--freeze-feat-epochs", str(freeze_epochs)]

        supcon_w = float(loss.get("supcon_weight") or 0.0)
        if supcon_w and supcon_w > 0:
            cmd += ["--supcon-weight", str(float(supcon_w))]
            cmd += ["--supcon-temp", str(float(loss.get("supcon_temperature") or loss.get("temperature") or 0.07))]
            cmd += ["--supcon-proj-dim", str(int(loss.get("supcon_proj_dim") or loss.get("proj_dim") or 128))]

        tp_drop = float(domain.get("tooth_position_dropout") or 0.0)
        if tp_drop > 0:
            cmd += ["--tooth-position-dropout", str(tp_drop)]

        method = str(domain.get("method") or "").strip().lower()
        if method:
            cmd += ["--domain-method", method]
            group_key = str(domain.get("group_key") or "").strip()
            if group_key:
                cmd += ["--domain-group-key", group_key]
            if method == "groupdro":
                cmd += ["--groupdro-eta", str(float(domain.get("groupdro_eta") or 0.1))]
            if method == "coral":
                cmd += ["--coral-weight", str(float(domain.get("coral_weight") or 0.0))]
                cmd += ["--coral-proj-dim", str(int(domain.get("coral_proj_dim") or 128))]

        rc = run_subprocess(cmd, cwd=repo_root, log_path=run_dir / "logs.txt")
        if rc != 0:
            return rc

    postprocess_run_dir(run_dir)
    rc = _run_temp_scaling(cfg=cfg, repo_root=repo_root, run_dir=run_dir)
    if rc != 0:
        return rc
    return _run_selective_eval(cfg=cfg, repo_root=repo_root, run_dir=run_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Unified YAML-config runner for dentist repo (raw_cls/domain_shift/prep2target).")
    ap.add_argument("--config", type=Path, required=True, help="Path to experiment YAML (supports `defaults`).")
    ap.add_argument("--seed", type=int, default=None, help="Override repro.seed")
    ap.add_argument("--fold", type=int, default=None, help="Override repro.fold")
    ap.add_argument("--train_source", type=str, default="", help="Override domain.train_source (domain_shift)")
    ap.add_argument("--test_source", type=str, default="", help="Override domain.test_source (domain_shift)")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config key with dotted path, e.g. --set train.epochs=5 (repeatable)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_with_defaults((repo_root / args.config).resolve() if not args.config.is_absolute() else args.config)
    cfg = interpolate_cfg(cfg)

    if args.seed is not None:
        set_nested(cfg, "repro.seed", int(args.seed))
    if args.fold is not None:
        set_nested(cfg, "repro.fold", int(args.fold))
    if str(args.train_source).strip():
        set_nested(cfg, "domain.train_source", str(args.train_source).strip())
    if str(args.test_source).strip():
        set_nested(cfg, "domain.test_source", str(args.test_source).strip())
    for item in args.set:
        s = str(item or "").strip()
        if not s or "=" not in s:
            raise SystemExit(f"Invalid --set item (expected k=v): {item!r}")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Basic type coercion.
        if v.lower() in {"true", "false"}:
            vv: Any = v.lower() == "true"
        else:
            try:
                vv = int(v)
            except Exception:
                try:
                    vv = float(v)
                except Exception:
                    vv = v
        set_nested(cfg, k, vv)

    task = str((cfg.get("data") or {}).get("task") or "").strip().lower()
    if task in {"raw_cls", "raw-cls", "rawcls"}:
        return run_raw_cls(cfg, repo_root=repo_root)
    if task in {"domain_shift", "domain-shift", "domainshift"}:
        return run_domain_shift(cfg, repo_root=repo_root)
    if task in {"prep2target", "prep-2-target", "prep_to_target", "prep2t"}:
        return run_prep2target(cfg, repo_root=repo_root)
    raise SystemExit(f"Unsupported task: {task!r} (supported: raw_cls, domain_shift, prep2target)")


if __name__ == "__main__":
    raise SystemExit(main())
