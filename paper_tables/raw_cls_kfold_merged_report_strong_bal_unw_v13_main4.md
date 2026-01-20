# raw_cls k-fold merged report (per-seed)

- generated_at: 2026-01-17T06:17:53Z
- runs_dir: `/home/ubuntu/tiasha/dentist/runs/raw_cls_strong1`
- data_tag: `v13_main4`
- bootstrap: n=2000 seed=1337

## Overall (mean±std over seeds)

| macro_f1_present (mean±std) | macro_f1_all (mean±std) | bal_acc_present (mean±std) | acc (mean±std) | ece (mean±std) | seeds | n_cases | model | extra_features | tta |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 0.3444±0.0234 | 0.3444±0.0234 | 0.3752±0.0396 | 0.3454±0.0222 | 0.1601±0.0335 | 3 | 248 | dgcnn | (none) | 8 |
| 0.3141±0.0183 | 0.3141±0.0183 | 0.3317±0.0150 | 0.3185±0.0185 | 0.1553±0.0134 | 3 | 248 | dgcnn | scale,log_points,objects_used | 8 |
| 0.3207±0.0131 | 0.3207±0.0131 | 0.3478±0.0165 | 0.3253±0.0084 | 0.1319±0.0135 | 3 | 248 | pointnet | (none) | 8 |

## Warnings

- missing folds for group=pointnet extra=scale,log_points,objects_used seed=1337: have [0]

## Paired comparisons (hierarchical bootstrap over seeds×cases)

| metric | delta(mean) | CI95 | p(two-sided) | A | B | seeds | family |
|---|---:|---:|---:|---|---|---:|---|
| macro_f1_present | 0.0237 | [-0.0191,0.0648] | 0.2740 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| accuracy | 0.0210 | [-0.0215,0.0618] | 0.3340 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| macro_f1_present | -0.0298 | [-0.0783,0.0167] | 0.2350 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| accuracy | -0.0261 | [-0.0739,0.0242] | 0.3120 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
