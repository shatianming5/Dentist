# raw_cls baselines (paper summary)

- generated_at: 2026-01-16T12:46:04Z
- runs_dir: `/home/ubuntu/tiasha/dentist/runs/raw_cls_baseline`
- total_runs: 34
- total_groups: 21

| test_macro_f1 (mean±std) | test_acc (mean±std) | n | model | n_points | kfold_k | balanced | label_smoothing | extra_features | tta | dataset |
|---:|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0.3203±0.0000 | 0.3333±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v12 |
| 0.2875±0.0000 | 0.4062±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v8 |
| 0.2813±0.0455 | 0.3017±0.0433 | 5 | dgcnn | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.2802±0.0000 | 0.3438±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v6b |
| 0.2731±0.0000 | 0.2903±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v11 |
| 0.2702±0.0609 | 0.3516±0.0469 | 4 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v6 |
| 0.2596±0.0000 | 0.3750±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v7 |
| 0.2596±0.0383 | 0.2825±0.0333 | 5 | pointnet | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.2582±0.0000 | 0.3000±0.0000 | 1 | pointnet | 512 | 0 | no | 0.000 | (none) | 0 | v13_main4 |
| 0.2362±0.0000 | 0.3438±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | log_scale,log_points | 0 | v6 |
| 0.2171±0.0000 | 0.2188±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v10 |
| 0.2148±0.0000 | 0.3125±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v3 |
| 0.1858±0.0000 | 0.1875±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v4 |
| 0.1812±0.0000 | 0.2500±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v9 |
| 0.1707±0.0000 | 0.2500±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v2 |
| 0.1614±0.0000 | 0.2188±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v1 |
| 0.1596±0.0000 | 0.2188±0.0000 | 1 | pointnet | 2048 | 0 | no | 0.000 | (none) | 0 | v6 |
| 0.1111±0.0000 | 0.1562±0.0000 | 1 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v5 |
| 0.1061±0.0000 | 0.2692±0.0000 | 1 | pointnet | 512 | 5 | no | 0.000 | (none) | 0 | v13_main4 |
| 0.0972±0.0000 | 0.2333±0.0000 | 1 | pointnet | 256 | 0 | no | 0.000 | (none) | 4 | v13_main4 |
| 0.0667±0.0000 | 0.2500±0.0000 | 3 | pointnet | 4096 | 0 | no | 0.000 | (none) | 0 | v6_smoke |

Notes:
- `n` counts runs; for k-fold it is typically `seeds × folds`.
- For journal reporting, consider adding bootstrap CIs from `preds_test.jsonl` (see `scripts/raw_cls_bootstrap_ci.py`).

