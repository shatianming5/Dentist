# Paper Reproducibility Protocol (Non-data fixes)

This file defines a **journal-style, reproducible protocol** for running baselines, reporting stable statistics (k-fold + multi-seed), and generating paper tables **from the current codebase**.

For claim/implementation alignment, read `PAPER_SCOPE.md`.

## Environment

- Python: 3.10+ recommended (repo also runs on newer Python)
- Dependencies: `numpy`, `torch`
  - Minimal list: `requirements.txt`
- Optional (conda): use `environment.yml` for a closer-to-journal reproduction setup:

```bash
conda env create -f environment.yml
conda activate dentist
```

## 1) raw_cls (CCB2-extracted point clouds) — k-fold × multi-seed

### 1.1 Train a paper suite (k-fold × multi-seed)

Example (5-fold, 3 seeds, PointNet + DGCNN):

```bash
python3 scripts/run_raw_cls_kfold.py \
  --root . \
  --data-root processed/raw_cls/v13_main4 \
  --kfold metadata/splits_raw_case_kfold.json \
  --models pointnet,dgcnn \
  --seeds 1337,2020,2021 \
  --folds all \
  --device auto \
  --epochs 120 \
  --patience 25 \
  --batch-size 64 \
  --n-points 4096 \
  --balanced-sampler \
  --label-smoothing 0.1 \
  --tta 8
```

Outputs:
- run dirs under `runs/raw_cls_baseline/`
- logs under `runs/raw_cls_baseline/_paper_logs/`

### 1.1b Meta-feature augmented baseline (recommended on small data)

The Phase1 builder already records auditable metadata per case (e.g., `scale`, point counts, `n_objects_used`). On this small dataset, enabling these **extra features** can materially improve classification stability.

Example:

```bash
python3 scripts/run_raw_cls_kfold.py \
  --root . \
  --data-root processed/raw_cls/v13_main4 \
  --kfold metadata/splits_raw_case_kfold.json \
  --models pointnet,dgcnn \
  --seeds 1337,2020,2021 \
  --folds all \
  --device auto \
  --epochs 120 \
  --patience 25 \
  --batch-size 64 \
  --n-points 4096 \
  --balanced-sampler \
  --label-smoothing 0.1 \
  --tta 8 \
  --extra-features scale,log_points,objects_used
```

### 1.2 Generate the paper summary table (mean±std)

```bash
python3 scripts/paper_table_raw_cls.py --runs-dir runs/raw_cls_baseline --out paper_tables/raw_cls_table.md
```

Notes:
- For k-fold runs, `scripts/paper_table_raw_cls.py` deduplicates by `(seed, test_fold)` and keeps the latest run to avoid accidental table contamination when old exp_name/tag variants coexist.

### 1.3 Confidence intervals (bootstrap) and paired comparisons

Single-run CI (uses `preds_test.jsonl`):

```bash
python3 scripts/raw_cls_bootstrap_ci.py \
  --run-dir runs/raw_cls_baseline/<exp_name> \
  --split test \
  --n-bootstrap 5000 \
  --out paper_tables/raw_cls_ci_<exp_name>.json
```

Paired bootstrap difference (same test set keys):

```bash
python3 scripts/raw_cls_bootstrap_ci.py \
  --run-dir runs/raw_cls_baseline/<exp_A> \
  --compare runs/raw_cls_baseline/<exp_B> \
  --metric macro_f1_present \
  --split test \
  --n-bootstrap 5000
```

### 1.4 Domain shift / calibration reporting

`scripts/phase3_train_raw_cls_baseline.py` writes, into `metrics.json`:
- `test_by_source` + `test_by_source_calibration`
- `test_by_tooth_position` (if available)
- `test_calibration` (`ece`, `nll`, `brier`)

These are paper-ready diagnostics for “普通标注 vs 专家标注” domain differences.

### 1.5 k-fold merged report (per-seed, recommended for journals)

To avoid treating each fold as an independent sample, you can merge the k-fold test predictions **per seed** (each case appears once across folds) and report:
- overall metrics (mean±std over seeds),
- bootstrap CIs on merged cases,
- by-source metrics + calibration.

```bash
python3 scripts/paper_raw_cls_kfold_merged_report.py \
  --runs-dir runs/raw_cls_baseline \
  --out-prefix paper_tables/raw_cls_kfold_merged_report_v13_main4 \
  --data-tag v13_main4
```

The merged report also includes **paired comparisons** (hierarchical bootstrap over seeds×cases) for key baselines/ablations, with CI95 and a two-sided p-value.

### 1.6 Meta-only baseline (shortcut audit)

To quantify how much of the signal comes from Phase1-recorded metadata alone (e.g., `scale`, point counts, `objects_used`), run a meta-only model that does **not** load point clouds:

```bash
python3 scripts/run_raw_cls_kfold.py \
  --root . \
  --data-root processed/raw_cls/v13_main4 \
  --kfold metadata/splits_raw_case_kfold.json \
  --models meta_mlp \
  --seeds 1337,2020,2021 \
  --folds all \
  --device auto \
  --epochs 120 \
  --patience 25 \
  --batch-size 64 \
  --n-points 0 \
  --balanced-sampler \
  --label-smoothing 0.1 \
  --tta 0 \
  --extra-features scale,log_points,objects_used
```

## 2) Teeth3DS synthetic completion + constraints

This repo implements a **synthetic** prep→target benchmark (cut a tooth and reconstruct). Use it as a controlled benchmark / pretraining stage (not a direct clinical claim).

- Synthetic completion baseline: `scripts/phase3_train_teeth3ds_prep2target.py`
- Constraints training: `scripts/phase4_train_teeth3ds_prep2target_constraints.py`
- Fixed evaluation: `scripts/phase4_eval_teeth3ds_constraints_run.py`

### 2.1 Re-evaluate constraints runs with improved metrics (min_d quantiles)

If you have existing constraints runs under `runs/teeth3ds_prep2target_constraints/`, re-run evaluation to populate:
- `occlusion_contact_ratio`
- `occlusion_min_d_p05/p50/p95`

Important: metric spaces / units (paper wording must match this):
- `chamfer` / `margin` are computed in the **processed (normalized) tooth space** (the model input/output space).
- `occlusion_*` is computed after de-normalizing predictions back to the **original Teeth3DS OBJ coordinate space** and comparing to opposing jaw/tooth points from the OBJ.
  - Treat these values as **OBJ coordinate units** (often mm-like, but not clinically calibrated here).

```bash
python3 scripts/run_constraints_eval_suite.py \
  --root . \
  --runs-dir runs/teeth3ds_prep2target_constraints \
  --data-root processed/teeth3ds_teeth/v1 \
  --device auto \
  --splits val,test \
  --deterministic \
  --cut-q-min 0.7 --cut-q-max 0.7 \
  --margin-band 0.02 --margin-points 64 \
  --occlusion-clearance 0.5
```

### 2.2 Summarize constraints runs into CSV/MD

```bash
python3 scripts/phase4_summarize_constraints_runs.py \
  --runs-dir runs/teeth3ds_prep2target_constraints \
  --out-prefix paper_tables/constraints_summary
```

## 3) Final “journal readiness” audit

```bash
python3 scripts/journal_audit.py --root . --out JOURNAL_AUDIT.md
```

## 4) Non-data paper audit (protocol/tools)

```bash
python3 scripts/paper_audit.py --root . --out PAPER_AUDIT.md
```
