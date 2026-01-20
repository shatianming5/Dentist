# Aggregate runs (raw_cls)

- generated_at: 2026-01-18T02:47:14Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v16_main4_seg_all`
- runs: 15
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 1

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.4012±0.0844 | 0.4336±0.0658 | 0.4145±0.0828 | 0.3879±0.0699 | 15 | v16_main4_seg_all | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
