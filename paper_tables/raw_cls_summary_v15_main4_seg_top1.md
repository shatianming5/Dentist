# Aggregate runs (raw_cls)

- generated_at: 2026-01-18T02:40:45Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v15_main4_seg_top1`
- runs: 15
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 1

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.3519±0.0756 | 0.3891±0.0624 | 0.3826±0.0793 | 0.3806±0.0878 | 15 | v15_main4_seg_top1 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_top1 | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
