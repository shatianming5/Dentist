# Dataset 统计报告
生成时间：2025-12-13T01:36:11Z
## 总览
- `data/`：29.17GB
- `archives/`：8.29GB
- 文件数（`data/`）：4056
- 目录数（`data/`）：2257

## Teeth3DS（mesh + 分割）
- `OBJ` 数：1900
- `JSON` 数：1800
- 上下颌 ID：upper=950（json=900）, lower=950（json=900）, unique_ids=968, both_jaws=932, only_upper=18, only_lower=18
- OBJ 顶点数（upper）：count=950, min=16282, p25=117142, median=131176, p75=148909.75, p95=175881.70, max=259938, mean=131519.76
- OBJ 面片数（upper）：count=950, min=32456, p25=232485.25, median=261361, p75=296981.50, p95=351661.65, max=519453, mean=261984.91
- OBJ 顶点数（lower）：count=950, min=13034, p25=89101, median=99314, p75=110471.75, p95=132027.95, max=241117, mean=99797.99
- OBJ 面片数（lower）：count=950, min=25940, p25=177499.50, median=196963.50, p75=219780, p95=263625.00, max=481822, mean=198366.39
- 分割标签（upper，汇总到所有带 json 的样本）：labels=[0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
- 分割标签（lower，汇总到所有带 json 的样本）：labels=[0, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
- `label=0`（牙龈/背景）比例：upper=0.4957, lower=0.4004
- 每个样本的牙齿实例数（instances>0）：upper=count=900, min=8, p25=12, median=14, p75=14, p95=15, max=16, mean=13.23, lower=count=900, min=10, p25=13, median=14, p75=14, p95=15, max=16, mean=13.45

## Landmarks（3DTeethLand kpt）
- kpt 文件数：340
- 点数量（所有 kpt 文件）：count=340, min=42, p25=84, median=92.50, p75=96, p95=106, max=118, mean=89.81
- classes：['Cusp', 'Distal', 'FacialPoint', 'InnerPoint', 'Mesial', 'OuterPoint']

## Split 覆盖情况
- `data/splits/3DTeethLand_challenge_train_test_split/testing_lower.txt`：lines=50, unique=50, dups=0, exists=50, has_json=0
- `data/splits/3DTeethLand_challenge_train_test_split/testing_upper.txt`：lines=50, unique=50, dups=0, exists=50, has_json=0
- `data/splits/3DTeethLand_challenge_train_test_split/training_lower.txt`：lines=120, unique=120, dups=0, exists=120, has_json=120
- `data/splits/3DTeethLand_challenge_train_test_split/training_upper.txt`：lines=120, unique=120, dups=0, exists=120, has_json=120
- `data/splits/3DTeethSeg22_challenge_train_test_split/private-testing-set.txt`：lines=600, unique=600, dups=0, exists=600, has_json=600
- `data/splits/3DTeethSeg22_challenge_train_test_split/public-training-set-1.txt`：lines=600, unique=600, dups=0, exists=600, has_json=600
- `data/splits/3DTeethSeg22_challenge_train_test_split/public-training-set-2.txt`：lines=600, unique=600, dups=0, exists=600, has_json=600
- `data/splits/Teeth3DS_train_test_split/testing_lower.txt`：lines=300, unique=300, dups=0, exists=300, has_json=300
- `data/splits/Teeth3DS_train_test_split/testing_upper.txt`：lines=300, unique=300, dups=0, exists=300, has_json=300
- `data/splits/Teeth3DS_train_test_split/training_lower.txt`：lines=600, unique=600, dups=0, exists=600, has_json=600
- `data/splits/Teeth3DS_train_test_split/training_upper.txt`：lines=600, unique=600, dups=0, exists=600, has_json=600

## 备注
- 许可证见：`data/splits/license.txt`
- 更细的 per-case 统计见：`DATASET_STATS.json`
