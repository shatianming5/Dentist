# raw_cls baselines (paper summary)

- generated_at: 2026-01-17T06:17:30Z
- runs_dir: `/home/ubuntu/tiasha/dentist/runs/raw_cls_strong1`
- total_runs: 46
- total_groups: 4
- dedup_dropped: 0 (k-fold duplicates)

| test_macro_f1 (mean±std) | test_acc (mean±std) | n | model | n_points | kfold_k | balanced | label_smoothing | extra_features | tta | dataset |
|---:|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0.3321±0.0488 | 0.3451±0.0467 | 15 | dgcnn | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.3040±0.0906 | 0.3235±0.0794 | 15 | pointnet | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.2921±0.0870 | 0.3176±0.0728 | 15 | dgcnn | 4096 | 5 | yes | 0.100 | scale,log_points,objects_used | 8 | v13_main4 |
| 0.2900±0.0000 | 0.2885±0.0000 | 1 | pointnet | 4096 | 5 | yes | 0.100 | scale,log_points,objects_used | 8 | v13_main4 |

Notes:
- `n` counts runs; for k-fold it is typically `seeds × folds`.
- For journal reporting, consider adding bootstrap CIs from `preds_test.jsonl` (see `scripts/raw_cls_bootstrap_ci.py`).

