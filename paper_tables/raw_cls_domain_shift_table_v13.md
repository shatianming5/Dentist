# raw_cls domain-shift baselines (paper summary)

- generated_at: 2026-01-16T17:43:50Z
- runs_dir: `/home/ubuntu/tiasha/dentist/runs/raw_cls_domain_shift`
- total_runs: 12
- total_groups: 4

| test_macro_f1_present (mean±std) | test_macro_f1_all (mean±std) | bal_acc_present (mean±std) | test_acc (mean±std) | ece (mean±std) | n | model | n_points | tta | train_source | test_source | val_ratio | dataset |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---:|---|
| 0.2321±0.1046 | 0.2321±0.1046 | 0.3095±0.0516 | 0.3212±0.0321 | 0.2985±0.0835 | 3 | dgcnn | 4096 | 8 | 专家标注 | 普通标注 | 0.100 | v13_main4 |
| 0.2314±0.0583 | 0.2314±0.0583 | 0.3806±0.1348 | 0.2771±0.0552 | 0.2351±0.0960 | 3 | dgcnn | 4096 | 8 | 普通标注 | 专家标注 | 0.100 | v13_main4 |
| 0.2293±0.0281 | 0.2293±0.0281 | 0.3428±0.0218 | 0.3213±0.1237 | 0.1985±0.0476 | 3 | pointnet | 4096 | 8 | 普通标注 | 专家标注 | 0.100 | v13_main4 |
| 0.2277±0.0204 | 0.2277±0.0204 | 0.3283±0.0325 | 0.3253±0.0126 | 0.2118±0.0348 | 3 | pointnet | 4096 | 8 | 专家标注 | 普通标注 | 0.100 | v13_main4 |

Notes:
- `n` counts seeds (each run is one seed under a fixed train_source→test_source split).
- Full per-run details live under each run dir (config.json/metrics.json/preds_test.jsonl).

