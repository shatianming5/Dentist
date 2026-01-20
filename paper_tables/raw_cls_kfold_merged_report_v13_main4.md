# raw_cls k-fold merged report (per-seed)

- generated_at: 2026-01-16T17:43:33Z
- runs_dir: `/home/ubuntu/tiasha/dentist/runs/raw_cls_baseline`
- data_tag: `v13_main4`
- bootstrap: n=2000 seed=1337

## Overall (mean±std over seeds)

| macro_f1_present (mean±std) | macro_f1_all (mean±std) | bal_acc_present (mean±std) | acc (mean±std) | ece (mean±std) | seeds | n_cases | model | extra_features | tta |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 0.2734±0.0288 | 0.2734±0.0288 | 0.3438±0.0223 | 0.2809±0.0246 | 0.2209±0.0595 | 3 | 248 | dgcnn | (none) | 8 |
| 0.2990±0.0165 | 0.2990±0.0165 | 0.3635±0.0383 | 0.2984±0.0176 | 0.2155±0.0084 | 3 | 248 | dgcnn | scale,log_points,objects_used | 8 |
| 0.1968±0.0123 | 0.1968±0.0123 | 0.3335±0.0074 | 0.2137±0.0107 | 0.2003±0.0022 | 3 | 248 | geom_mlp | (none) | 8 |
| 0.2067±0.0053 | 0.2067±0.0053 | 0.3432±0.0030 | 0.2258±0.0040 | 0.2126±0.0317 | 3 | 248 | meta_mlp | scale,log_points,objects_used | 0 |
| 0.2841±0.0196 | 0.2841±0.0196 | 0.3912±0.0256 | 0.3024±0.0176 | 0.1619±0.0136 | 3 | 248 | pointnet | (none) | 8 |
| 0.2916±0.0223 | 0.2916±0.0223 | 0.3678±0.0069 | 0.2970±0.0153 | 0.2014±0.0064 | 3 | 248 | pointnet | scale,log_points,objects_used | 8 |

## Warnings

- duplicate run for group=pointnet extra=scale,log_points,objects_used seed=1337 fold=0: keep=paper_rawcls_v13_main4_pointnet_n4096_k5_fold0_seed1337_bal_ls0p1_xfsc_lpts_ou_tta8 (t=2026-01-16T14:38:31Z) drop=paper_rawcls_v13_main4_pointnet_n4096_k5_fold0_seed1337_bal_ls0p1_xfSLO_tta8 (t=2026-01-16T14:30:06Z)

## Paired comparisons (hierarchical bootstrap over seeds×cases)

| metric | delta(mean) | CI95 | p(two-sided) | A | B | seeds | family |
|---|---:|---:|---:|---|---|---:|---|
| macro_f1_present | 0.0764 | [0.0299,0.1266] | 0.0000 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | geom_mlp n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| accuracy | 0.0671 | [0.0242,0.1129] | 0.0040 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | geom_mlp n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| macro_f1_present | -0.0095 | [-0.0600,0.0433] | 0.7020 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| accuracy | -0.0219 | [-0.0699,0.0309] | 0.4040 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| macro_f1_present | -0.0866 | [-0.1192,-0.0533] | 0.0000 | geom_mlp n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| accuracy | -0.0890 | [-0.1250,-0.0551] | 0.0000 | geom_mlp n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| macro_f1_present | 0.0078 | [-0.0364,0.0504] | 0.7110 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| accuracy | 0.0015 | [-0.0403,0.0417] | 0.9590 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| macro_f1_present | 0.0250 | [-0.0138,0.0612] | 0.1960 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| accuracy | 0.0175 | [-0.0215,0.0551] | 0.3730 | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | dgcnn n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| macro_f1_present | 0.0068 | [-0.0266,0.0410] | 0.6880 | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
| accuracy | -0.0055 | [-0.0417,0.0309] | 0.7620 | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=scale,log_points,objects_used | pointnet n=4096 k=5 bal ls=0.1 tta=8 extra=(none) | 3 | data_tag=v13_main4 n_points=4096 kfold_k=5 balanced=True label_smoothing=0.1 tta=8 |
