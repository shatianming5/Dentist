# raw_cls baselines (paper summary)

- generated_at: 2026-01-16T17:43:30Z
- runs_dir: `/home/ubuntu/tiasha/dentist/runs/raw_cls_baseline`
- total_runs: 95
- total_groups: 7
- dedup_dropped: 2 (k-fold duplicates)

| test_macro_f1 (mean±std) | test_acc (mean±std) | n | model | n_points | kfold_k | balanced | label_smoothing | extra_features | tta | dataset |
|---:|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0.2825±0.0494 | 0.3019±0.0461 | 15 | pointnet | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.2821±0.0735 | 0.2996±0.0725 | 15 | dgcnn | 4096 | 5 | yes | 0.100 | scale,log_points,objects_used | 8 | v13_main4 |
| 0.2794±0.0725 | 0.2957±0.0636 | 15 | pointnet | 4096 | 5 | yes | 0.100 | scale,log_points,objects_used | 8 | v13_main4 |
| 0.2491±0.0468 | 0.2806±0.0445 | 15 | dgcnn | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.2053±0.0273 | 0.2256±0.0287 | 15 | meta_mlp | 0 | 5 | yes | 0.100 | scale,log_points,objects_used | 0 | v13_main4 |
| 0.1885±0.0324 | 0.2134±0.0297 | 15 | geom_mlp | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.1061±0.0000 | 0.2692±0.0000 | 1 | pointnet | 512 | 5 | no | 0.000 | (none) | 0 | v13_main4 |

Notes:
- `n` counts runs; for k-fold it is typically `seeds × folds`.
- k-fold dedup: keep the latest run per (seed,test_fold); older duplicates are dropped.
  - dropped duplicate: keep=`paper_rawcls_v13_main4_pointnet_n4096_k5_fold0_seed1337_bal_ls0p1_xfsc_lpts_ou_tta8` (t=2026-01-16T14:38:31Z) drop=`paper_rawcls_v13_main4_pointnet_n4096_k5_fold0_seed1337_bal_ls0p1_xfSLO_tta8` (t=2026-01-16T14:30:06Z)
  - dropped duplicate: keep=`smoke_rawcls_v13_main4_pointnet_n512_k5_fold0_seed1337` (t=2026-01-16T13:17:54Z) drop=`pointnet_n512_seed1337_20260116_104425_k5_fold0` (t=2026-01-16T10:44:25Z)
- For journal reporting, consider adding bootstrap CIs from `preds_test.jsonl` (see `scripts/raw_cls_bootstrap_ci.py`).

