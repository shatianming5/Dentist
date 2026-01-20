# Aggregate runs (raw_cls)

- generated_at: 2026-01-17T18:33:12Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v13_main4`
- runs: 333
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 24

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.3796±0.0630 | 0.4176±0.0365 | 0.3838±0.0570 | 0.3649±0.0848 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1 | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3785±0.0751 | 0.4077±0.0556 | 0.3816±0.0751 | 0.3432±0.1061 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0 | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3694±0.0826 | 0.4116±0.0699 | 0.3785±0.0763 | 0.2778±0.0734 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0.1 | 0 |
| 0.3620±0.0862 | 0.4127±0.0644 | 0.3673±0.0821 | 0.3900±0.1072 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_bbox | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3545±0.0913 | 0.3820±0.0715 | 0.3698±0.0922 | 0.3268±0.1210 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_unbal_ls0 | pointnet | xyz | (none) | unbal | ls=0 | 0 |
| 0.3521±0.0874 | 0.3928±0.0678 | 0.3712±0.0805 | 0.3403±0.1084 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_unbal_ls0 | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | unbal | ls=0 | 0 |
| 0.3515±0.1005 | 0.3837±0.0963 | 0.3667±0.1000 | 0.3682±0.1465 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_bboxpca | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3510±0.0660 | 0.3553±0.0631 | 0.3982±0.0686 | 0.1376±0.0546 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_finetune_lr3e4 | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3483±0.0699 | 0.3668±0.0674 | 0.3875±0.0667 | 0.1687±0.0829 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_finetune | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3425±0.0678 | 0.3925±0.0631 | 0.3502±0.0669 | 0.3014±0.0741 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0 | pointnet | xyz | (none) | bal | ls=0.1 | 0 |
| 0.3421±0.0726 | 0.3452±0.0649 | 0.3835±0.0686 | 0.1874±0.0626 | 15 | v13_main4 | - | supcon | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3386±0.0799 | 0.3952±0.0675 | 0.3468±0.0715 | 0.3678±0.1043 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0 | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3323±0.0685 | 0.3472±0.0750 | 0.3623±0.0711 | 0.1623±0.0649 | 15 | v13_main4 | - | baseline | dgcnn | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3297±0.0724 | 0.3442±0.0784 | 0.3731±0.0808 | 0.1788±0.0900 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_finetune_freeze10 | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3279±0.0730 | 0.3857±0.0665 | 0.3481±0.0713 | 0.4011±0.0902 | 15 | v13_main4 | - | dgcnn_supcon_norot_tta0_posfeat_n1024_ls0_dropout0p1 | dgcnn | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3261±0.0680 | 0.3403±0.0661 | 0.3706±0.0864 | 0.1768±0.0551 | 15 | v13_main4 | - | scale_token | pointnet | xyz | scale | bal | ls=0.1 | 8 |
| 0.3260±0.0627 | 0.3629±0.0611 | 0.3461±0.0621 | 0.1752±0.0502 | 15 | v13_main4 | - | feat_normcurv | pointnet | xyz,normals,curvature,radius | (none) | bal | ls=0.1 | 8 |
| 0.3224±0.0844 | 0.3318±0.0690 | 0.3625±0.0863 | 0.1794±0.0627 | 15 | v13_main4 | - | baseline | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3218±0.1233 | 0.3522±0.1228 | 0.3397±0.1132 | 0.3081±0.1178 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_posmoe | pointnet_pos_moe | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.3200±0.0303 | 0.3942±0.0408 | 0.3625±0.0581 | 0.3006±0.0957 | 2 | v13_main4 | - | dgcnn_supcon_norot_tta0_posfeat | dgcnn | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0.1 | 0 |
| 0.3195±0.0755 | 0.3386±0.0686 | 0.3622±0.0740 | 0.1836±0.0727 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3059±0.1005 | 0.3614±0.0881 | 0.3222±0.0952 | 0.3165±0.1002 | 15 | v13_main4 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_normcurv | pointnet | xyz,normals,curvature,radius | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0.1 | 0 |
| 0.2841±0.0557 | 0.3045±0.0519 | 0.3296±0.0616 | 0.1904±0.0713 | 15 | v13_main4 | - | pretrain_finetune | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.1061±0.0000 | 0.2692±0.0000 | 0.2500±0.0000 | 0.0910±0.0000 | 1 | v13_main4 | - | smoke_bs32 | dgcnn | xyz | (none) | bal | ls=0.1 | 0 |
