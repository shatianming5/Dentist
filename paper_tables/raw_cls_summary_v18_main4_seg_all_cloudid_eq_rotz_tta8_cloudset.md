# Aggregate runs (raw_cls)

- generated_at: 2026-01-18T06:57:52Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v18_main4_seg_all_cloudid_eq`
- runs: 166
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 12

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.4583±0.0628 | 0.4919±0.0608 | 0.4896±0.0568 | 0.2176±0.0465 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 8 |
| 0.4553±0.0651 | 0.4852±0.0470 | 0.4666±0.0701 | 0.3278±0.0648 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.4472±0.0667 | 0.4840±0.0603 | 0.4693±0.0706 | 0.2590±0.0612 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_dsbn | pointnet_dsbn | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 8 |
| 0.4424±0.0893 | 0.4834±0.0596 | 0.4953±0.0874 | 0.1989±0.0647 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_cloudset | pointnet_cloudset | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 8 |
| 0.4167±0.0716 | 0.4553±0.0762 | 0.4437±0.0711 | 0.3344±0.0689 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_ce_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.4121±0.0713 | 0.4527±0.0512 | 0.4242±0.0679 | 0.3273±0.0639 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_curvrad | pointnet | xyz,cloud_id,curvature,radius | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.4049±0.0579 | 0.4523±0.0576 | 0.4263±0.0551 | 0.3225±0.0632 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_meta | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing,scale,log_points,objects_used | bal | ls=0 | 0 |
| 0.4017±0.0680 | 0.4347±0.0650 | 0.4222±0.0810 | 0.3774±0.0789 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_tpdrop0p5_seg_all_cloudid_eq | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3973±0.0577 | 0.4401±0.0692 | 0.4173±0.0646 | 0.3529±0.0729 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_ae_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3829±0.0645 | 0.4753±0.0643 | 0.4086±0.0563 | 0.3110±0.0804 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_bestacc_seg_all_cloudid_eq | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3593±0.0575 | 0.4082±0.0512 | 0.3683±0.0545 | 0.3050±0.0881 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_cloud_geom_mlp_seg_all_cloudid_eq | cloud_geom_mlp | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3455±0.0000 | 0.4038±0.0000 | 0.3589±0.0000 | 0.3833±0.0000 | 1 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_dgcnn | dgcnn | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
