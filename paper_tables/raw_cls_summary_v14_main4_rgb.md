# Aggregate runs (raw_cls)

- generated_at: 2026-01-18T02:31:51Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v14_main4_rgb`
- runs: 15
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 1

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.3051±0.0666 | 0.3702±0.0571 | 0.3287±0.0567 | 0.3998±0.1135 | 15 | v14_main4_rgb | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_rgb | pointnet | xyz,rgb | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
