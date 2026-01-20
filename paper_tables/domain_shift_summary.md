# Aggregate runs (domain_shift)

- generated_at: 2026-01-17T15:26:45Z
- root: `/home/ubuntu/tiasha/dentist/runs/domain_shift/v13_main4`
- runs: 48
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0]

- groups: 16

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.3662±0.1685 | 0.4333±0.1528 | 0.4232±0.1338 | 0.2018±0.0680 | 3 | v13_main4 | A2B_专家标注_to_普通标注 | groupdro | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3597±0.1544 | 0.4000±0.0866 | 0.3815±0.1332 | 0.1857±0.1226 | 3 | v13_main4 | A2B_普通标注_to_普通标注 | baseline | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3532±0.1306 | 0.2667±0.1155 | 0.2917±0.1102 | 0.4462±0.1639 | 3 | v13_main4 | A2B_普通标注_to_专家标注 | baseline | dgcnn | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3490±0.0159 | 0.3333±0.0577 | 0.3611±0.1203 | 0.1844±0.0350 | 3 | v13_main4 | A2B_专家标注_to_专家标注 | baseline | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3422±0.1727 | 0.3000±0.1732 | 0.3333±0.1909 | 0.2960±0.0418 | 3 | v13_main4 | A2B_普通标注_to_专家标注 | coral | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.3407±0.0449 | 0.3333±0.0577 | 0.3889±0.1049 | 0.2745±0.0796 | 3 | v13_main4 | A2B_普通标注_to_专家标注 | pos_moe | pointnet_pos_moe | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0.1 | 8 |
| 0.3007±0.1168 | 0.3500±0.0866 | 0.3458±0.0807 | 0.1484±0.0077 | 3 | v13_main4 | A2B_专家标注_to_普通标注 | coral | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.2953±0.0099 | 0.3000±0.0000 | 0.2958±0.0134 | 0.3202±0.0499 | 3 | v13_main4 | A2B_普通标注_to_普通标注 | baseline | dgcnn | xyz | (none) | bal | ls=0.1 | 8 |
| 0.2904±0.1913 | 0.3667±0.1756 | 0.3887±0.1393 | 0.1975±0.0799 | 3 | v13_main4 | A2B_专家标注_to_普通标注 | baseline | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.2565±0.0593 | 0.2333±0.0577 | 0.2778±0.0481 | 0.3248±0.0590 | 3 | v13_main4 | A2B_专家标注_to_专家标注 | baseline | dgcnn | xyz | (none) | bal | ls=0.1 | 8 |
| 0.2532±0.0881 | 0.1667±0.0577 | 0.1944±0.0636 | 0.3775±0.0449 | 3 | v13_main4 | A2B_普通标注_to_专家标注 | groupdro | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.2526±0.1277 | 0.2000±0.1000 | 0.2083±0.1102 | 0.2749±0.1141 | 3 | v13_main4 | A2B_普通标注_to_专家标注 | dsbn | pointnet_dsbn | xyz | (none) | bal | ls=0.1 | 8 |
| 0.2456±0.0906 | 0.3167±0.0764 | 0.2857±0.0804 | 0.0798±0.0580 | 3 | v13_main4 | A2B_专家标注_to_普通标注 | pos_moe | pointnet_pos_moe | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0.1 | 8 |
| 0.2369±0.0972 | 0.2000±0.1000 | 0.2083±0.0722 | 0.3233±0.0751 | 3 | v13_main4 | A2B_普通标注_to_专家标注 | baseline | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.2146±0.0535 | 0.3167±0.0289 | 0.3024±0.0655 | 0.2223±0.0480 | 3 | v13_main4 | A2B_专家标注_to_普通标注 | baseline | dgcnn | xyz | (none) | bal | ls=0.1 | 8 |
| 0.1296±0.0000 | 0.3500±0.0000 | 0.2500±0.0000 | 0.3754±0.1888 | 3 | v13_main4 | A2B_专家标注_to_普通标注 | dsbn | pointnet_dsbn | xyz | (none) | bal | ls=0.1 | 8 |
