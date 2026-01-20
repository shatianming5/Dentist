# 数据说明（已解压并整理）

## 项目入口（只看 README）

- 快速开始（smoke / SOTA）：见下文「快速开始」
- 训练/评估入口：`scripts/train.py`
- 配置（YAML）：`configs/`（索引：`configs/INDEX.md`）
- 结果与表格：`paper_tables/`（索引：`paper_tables/INDEX.md`）
- runs 证据：`runs/`（说明：`runs/README.md`）
- 脚本索引：`scripts/INDEX.md`

## 快速开始

### 0) 环境

推荐 conda：

```bash
cd configs/env
conda env create -f environment.yml
conda activate dentist
```

或 pip（需要你自行安装匹配 CUDA 的 PyTorch）：

```bash
cd configs/env
pip install -r requirements.txt
```

### 1) 最小 smoke（CPU，~分钟级）

用于验证：配置系统/训练脚本/落盘产物都正常。

```bash
python3 scripts/train.py \
  --config configs/raw_cls/exp/baseline.yaml \
  --fold 0 --seed 0 \
  --set runtime.device=cpu \
  --set runtime.num_workers=0 \
  --set train.epochs=1 \
  --set train.patience=1 \
  --set train.batch_size=8 \
  --set data.n_points=256
```

检查输出目录（示例）：

- `runs/raw_cls/v13_main4/baseline/pointnet/fold=0/seed=0/metrics.json`
- `runs/raw_cls/v13_main4/baseline/pointnet/fold=0/seed=0/logs.txt`

### 2) raw_cls 当前 SOTA（GPU，耗时取决于机器）

SOTA 在 `processed/raw_cls/v18_main4_seg_all_cloudid_eq` 上，通过 3 个成员模型的 mean-prob ensemble 得到。

1) 跑 3 个成员 full（k=5×seeds=3）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_rawcls_v18_main4_seg_all_cloudid_eq_rotz_tta8_xyzonly.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_rawcls_v18_main4_seg_all_cloudid_eq_rotz_tta8_xyz_curvrad.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_rawcls_v18_main4_seg_all_cloudid_eq_rotz_tta8_pointnet2_xyzonly_n1024.sh
```

2) 计算 multi-member ensemble：

```bash
python3 scripts/paper_raw_cls_multi_ensemble.py \
  --runs-root runs/raw_cls/v18_main4_seg_all_cloudid_eq \
  --member teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly:pointnet \
  --member teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyz_curvrad:pointnet \
  --member teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly_pointnet2_n1024:pointnet2 \
  --seeds 1337,2020,2021 \
  --folds 0,1,2,3,4 \
  --out paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json
```

## 这个仓库做什么？

- 以“期刊可复现”为目标，提供牙科点云任务的 **数据整理、训练/评估协议、审计与表格产物**。
- 主要可复现任务：
  - raw_cls：修复体类型分类（4 类）
  - domain_shift：普通↔专家标注域差异
  - prep2target / constraints：synthetic proxy

## 这是什么数据？

这是牙列（上颌/下颌）**三维表面网格**数据：每个样本是一份 `OBJ` 网格（不是 CT/CBCT 的体数据），并带有逐顶点的牙齿分割标注（语义 + 实例）。同时包含一部分 3DTeethLand 的关键点（landmarks/kpt）标注，以及 Teeth3DS / 3DTeethLand / 3DTeethSeg22 的官方划分名单。

## 目录结构（数据视角）

```
.
├── archives/                         # 原始压缩包（未改内容）
└── data/
    ├── teeth3ds/                     # 主数据集（mesh + 分割标签）
    │   ├── upper/<ID>/<ID>_upper.obj
    │   ├── upper/<ID>/<ID>_upper.json
    │   ├── lower/<ID>/<ID>_lower.obj
    │   └── lower/<ID>/<ID>_lower.json
    ├── splits/                       # 官方划分/名单（txt）
    │   ├── Teeth3DS_train_test_split/
    │   ├── 3DTeethLand_challenge_train_test_split/
    │   ├── 3DTeethSeg22_challenge_train_test_split/
    │   └── license.txt
    ├── landmarks/                    # 3DTeethLand landmarks（关键点 kpt）
    │   ├── train/{upper,lower}/<ID>/*__kpt.json
    │   └── test/{upper,lower}/<ID>/*__kpt.json
    └── teeth3ds_sample/              # 小样例（含 kpt 示例）
```

## 文件格式要点

- `data/teeth3ds/**/<ID>_<jaw>.obj`：三角网格（OBJ）
- `OBJ` 顶点行形如 `v x y z r g b`（带顶点颜色；部分解析器会忽略颜色）
- `data/teeth3ds/**/<ID>_<jaw>.json`：分割标注（示例键：`id_patient`、`jaw`、`labels`、`instances`）
  - `labels`/`instances` 长度与 OBJ 顶点数一致，并按 OBJ 顶点顺序一一对应
- `data/**/__kpt.json`：关键点/landmarks 标注（含 `objects[].coord` 3D 坐标）

## 数据统计（已生成）

- Teeth3DS 相关统计：`archives/_meta/root_files/DATASET_STATS.md`（原始 JSON：`archives/_meta/root_files/DATASET_STATS.json`）
- raw/converted/processed 相关统计：`archives/_meta/root_files/RAW_DATASET_STATS.md`（原始 JSON：`archives/_meta/root_files/RAW_DATASET_STATS.json`）

## 从压缩包重新解压（可选）

如果需要从 `archives/` 重新生成 `data/`（会覆盖现有目录）：

```bash
rm -rf data
mkdir -p data/teeth3ds data/splits data/landmarks/train data/landmarks/test

for f in archives/data_part_*.zip; do
  bsdtar -xf "$f" -C data/teeth3ds
done

bsdtar -xf archives/teeth3ds_sample.zip -C data
bsdtar -xf archives/osfstorage-archive.zip -C data/splits
bsdtar -xf archives/3DTeethLand_landmarks_train.zip -C data/landmarks/train
bsdtar -xf archives/3DTeethLand_landmarks_test.zip -C data/landmarks/test
```

## raw/（修复体标注，CloudCompare BIN）

`raw/` 下是 CloudCompare 的 `CCB2` 二进制工程/实体文件（`*.bin`），配套 Excel 给了修复体类型标签。

已提供纯 Python 的批量提取脚本与转换结果：

- 转换脚本：`scripts/convert_ccb2_bin.py`
  - 示例：`python3 scripts/convert_ccb2_bin.py --input raw --output converted/raw --formats npz,ply --select all --min-points 500`
- 转换产物：`converted/raw/`
  - 汇总清单：`converted/raw/manifest.json`
  - 加上标签的清单：`converted/raw/manifest_with_labels.json`
  - 标签表：`converted/raw/labels.csv` / `converted/raw/labels.json`

## 下一步（从“能跑起来”到“能复现结论”）

- 跑 smoke：见上文「快速开始」
- 看结果：优先看 `paper_tables/INDEX.md` 与本 README 的「已尝试过的模型、方法与表现」
- 用户验收：进入任意 `runs/**/` 检查是否包含 `metrics.json / preds_*.jsonl / config.yaml / logs.txt`（说明见 `runs/README.md`）

## 结果速览（当前最佳）

> 口径：以 `paper_tables/` 聚合产物为准；raw_cls 越大越好，其余回归/约束类指标越小越好。

| task | best | evidence |
|---|---:|---|
| raw_cls（4 类分类） | overall_acc=0.6371 / macro_f1=0.6128 / bal_acc=0.6261 / ece=0.1255 (n=248) | `paper_tables/raw_cls_ensemble_eval_mean_v18_best.json` |
| prep2target（synthetic proxy） | test_total=0.0628±0.0004 (seeds=3) | `paper_tables/prep2target_summary.md` |
| constraints（teeth3ds_prep2target_constraints） | eval_test_total=0.057178 | `paper_tables/constraints_summary.md` |

## 已尝试过的模型、方法与表现（完整清单）

> 数据来源：`paper_tables/` 下的聚合表与 JSON（每个文件顶部 `generated_at` 可追溯）；run 级证据见 `runs/`。

### raw_cls（修复体 4 类分类）

- 当前 repo 内 SOTA（v18 multi-member mean-prob ensemble）：
  - overall_acc=0.6371 / macro_f1=0.6128 / bal_acc=0.6261 / ece=0.1255（n=248）
  - 证据：`paper_tables/raw_cls_ensemble_eval_mean_v18_best.json`（完整指标）与 `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json`（acc 汇总）

#### raw_cls 方法索引（读表用）

- Backbone：`pointnet` / `pointnet2` / `dgcnn` / `dgcnn_v2` / `pointmlp` / `point_transformer`
- 多云聚合：mean pool（baseline）/ `cloudset` / `cloudmil` / `cloudattn` / `dsbn`
- 输入特征：`xyz`、`xyz+cloud_id`、`xyz+curvature+radius`、`xyz+normals+curvature+radius`、`meta`
- 训练：CE / SupCon、label smoothing（`ls=*`）、dropout、balanced vs unbalanced sampler
- 推理与集成：`rotz tta8`、temperature scaling、seed ensemble、多成员 ensemble（mean/weighted/bias/stacking）

#### 提升方向（不跑新实验也能先做的）

- 误差结构驱动：优先解决“全冠 → 桩核冠/高嵌体”的系统性混淆（清标签歧义/降噪、让输入更像目标修复体）
- 集成升级：把简单均值融合升级为 val-opt 的加权/stacking（用每 fold 的 val 学权重/温度再评估 test fold）
- 提供新信息：在 out-of-fold 真提升的 member 里加入稳定局部几何（curvature/radius 等），避免无效堆 backbone

<details>
<summary><b>raw_cls v18：单模型尝试（k-fold×multi-seed 聚合）</b></summary>

Source: `paper_tables/raw_cls_summary_v18_main4_seg_all_cloudid_eq_rotz_tta8_cloudmil.md`

| test_acc | macro_f1 | bal_acc | ece | n | model | recipe | variant | input | extra | sampler |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 0.5672±0.0574 | 0.5407±0.0550 | 0.5553±0.0584 | 0.2253±0.0788 | 15 | pointnet2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_pointnet2_n1024_sa2_256 | xyz | pos | bal |
| 0.5394±0.0769 | 0.5152±0.0840 | 0.5485±0.0834 | 0.1976±0.0468 | 15 | pointnet2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyz_curvrad_pointnet2_n1024_sa2_256 | xyz,curvature,radius | pos | bal |
| 0.5414±0.0732 | 0.5146±0.0779 | 0.5388±0.0691 | 0.2319±0.0643 | 15 | pointnet2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_pointnet2_n1024 | xyz | pos | bal |
| 0.5349±0.0543 | 0.4968±0.0600 | 0.5230±0.0593 | 0.2223±0.0680 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly | xyz | pos | bal |
| 0.5163±0.0530 | 0.4922±0.0573 | 0.5248±0.0579 | 0.2179±0.0538 | 15 | pointnet2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyz_curvrad_pointnet2_n1024 | xyz,curvature,radius | pos | bal |
| 0.5283±0.0432 | 0.4901±0.0697 | 0.5215±0.0769 | 0.2105±0.0494 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p3 | xyzonly | xyz | pos | bal |
| 0.5215±0.0736 | 0.4898±0.0711 | 0.5134±0.0522 | 0.2554±0.0788 | 15 | dgcnn | dgcnn_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_n1024 | xyz | pos | bal |
| 0.5216±0.0491 | 0.4853±0.0602 | 0.5113±0.0564 | 0.2279±0.0606 | 15 | pointnet | teeth3ds_fdi_pretrain_ce_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly | xyz | pos | bal |
| 0.5252±0.0440 | 0.4852±0.0587 | 0.5187±0.0473 | 0.2350±0.0774 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_unbalanced_ce_none | xyz | pos | unbal |
| 0.5354±0.0368 | 0.4802±0.0504 | 0.4972±0.0526 | 0.2015±0.0546 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyz_curvrad | xyz,curvature,radius | pos | bal |
| 0.5010±0.0678 | 0.4591±0.0853 | 0.4980±0.0881 | 0.2093±0.0694 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyz_normcurvrad | xyz,normals,curvature,radius | pos | bal |
| 0.4919±0.0608 | 0.4583±0.0628 | 0.4896±0.0568 | 0.2176±0.0465 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | (base) | xyz,cloud_id | pos | bal |
| 0.4852±0.0470 | 0.4553±0.0651 | 0.4666±0.0701 | 0.3278±0.0648 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1 | (base) | xyz,cloud_id | pos | bal |
| 0.4936±0.0400 | 0.4484±0.0457 | 0.4643±0.0221 | 0.2330±0.0669 | 3 | pointnet_cloudattn | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | cloudattn | xyz,cloud_id | pos | bal |
| 0.4840±0.0603 | 0.4472±0.0667 | 0.4693±0.0706 | 0.2590±0.0612 | 15 | pointnet_dsbn | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | dsbn | xyz,cloud_id | pos | bal |
| 0.4791±0.0619 | 0.4458±0.0438 | 0.4735±0.0524 | 0.2410±0.0755 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | dropfix | xyz,cloud_id | pos | bal |
| 0.4834±0.0596 | 0.4424±0.0893 | 0.4953±0.0874 | 0.1989±0.0647 | 15 | pointnet_cloudset | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | cloudset | xyz,cloud_id | pos | bal |
| 0.5234±0.0446 | 0.4402±0.0805 | 0.4712±0.0796 | 0.2471±0.0758 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_bestacc | xyz | pos | bal |
| 0.4477±0.0826 | 0.4295±0.0849 | 0.4717±0.0885 | 0.1595±0.0538 | 15 | pointnet_cloudmil | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | cloudmil | xyz,cloud_id | pos | bal |
| 0.4627±0.0733 | 0.4266±0.0702 | 0.4840±0.0534 | 0.2510±0.0390 | 15 | pointnet2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_pointnet2_n2048_esdelta | xyz | pos | bal |
| 0.4553±0.0762 | 0.4167±0.0716 | 0.4437±0.0711 | 0.3344±0.0689 | 15 | pointnet | teeth3ds_fdi_pretrain_ce_norot_tta0_posfeat_ls0_dropout0p1 | (base) | xyz,cloud_id | pos | bal |
| 0.4487±0.0111 | 0.4149±0.0411 | 0.4714±0.0460 | 0.2685±0.0768 | 3 | pointnet_cloudattn | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | cloudattn_curvrad | xyz,cloud_id,curvature,radius | pos | bal |
| 0.4527±0.0512 | 0.4121±0.0713 | 0.4242±0.0679 | 0.3273±0.0639 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1 | curvrad | xyz,cloud_id,curvature,radius | pos | bal |
| 0.4523±0.0576 | 0.4049±0.0579 | 0.4263±0.0551 | 0.3225±0.0632 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1 | meta | xyz,cloud_id | pos+meta | bal |
| 0.4347±0.0650 | 0.4017±0.0680 | 0.4222±0.0810 | 0.3774±0.0789 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_tpdrop0p5 | (base) | xyz,cloud_id | pos | bal |
| 0.4401±0.0692 | 0.3973±0.0577 | 0.4173±0.0646 | 0.3529±0.0729 | 15 | pointnet | teeth3ds_ae_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1 | (base) | xyz,cloud_id | pos | bal |
| 0.4889±0.0509 | 0.3838±0.0437 | 0.4165±0.0468 | 0.3276±0.0201 | 3 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_fixedsplit | xyz | pos | bal |
| 0.4753±0.0643 | 0.3829±0.0645 | 0.4086±0.0563 | 0.3110±0.0804 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_bestacc | (base) | xyz,cloud_id | pos | bal |
| 0.4082±0.0512 | 0.3593±0.0575 | 0.3683±0.0545 | 0.3050±0.0881 | 15 | cloud_geom_mlp | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_cloud_geom_mlp | (base) | xyz,cloud_id | pos | bal |
| 0.4038±0.0000 | 0.3455±0.0000 | 0.3589±0.0000 | 0.3833±0.0000 | 1 | dgcnn | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1 | dgcnn | xyz,cloud_id | pos | bal |
| 0.3641±0.1118 | 0.2593±0.0876 | 0.3326±0.0754 | 0.2024±0.1265 | 15 | point_transformer | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_pointtransformer_n1024_v2 | xyz | pos | bal |
| 0.4063±0.0605 | 0.2381±0.0588 | 0.3270±0.0589 | 0.2516±0.1585 | 15 | dgcnn_v2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_dgcnn_v2_n1024 | xyz | pos | bal |
| 0.3702±0.0793 | 0.2255±0.0567 | 0.3086±0.0570 | 0.1952±0.0783 | 15 | pointmlp | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_pointmlp_n1024_v2 | xyz | pos | bal |
| 0.3846±0.0000 | 0.1389±0.0000 | 0.2500±0.0000 | 0.5901±0.0000 | 1 | point_transformer | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_pointtransformer_n1024 | xyz | pos | bal |
| 0.2885±0.0000 | 0.1315±0.0000 | 0.2625±0.0000 | 0.1905±0.0000 | 1 | pointnet2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_pointnet2_n1024 | xyz | pos | bal |
| 0.2692±0.0000 | 0.1061±0.0000 | 0.2500±0.0000 | 0.0058±0.0000 | 1 | pointnet_cloudattn | smoke_cloudattn | (base) | xyz,cloud_id | pos | bal |
| 0.2308±0.0000 | 0.0938±0.0000 | 0.2500±0.0000 | 0.5188±0.0000 | 1 | pointmlp | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1 | xyzonly_pointmlp_n1024_v1 | xyz | pos | bal |
| 0.1154±0.0000 | 0.0517±0.0000 | 0.2500±0.0000 | 0.6708±0.0000 | 1 | dgcnn_v2 | smoke_dgcnn_v2_initfeat | (base) | xyz | pos | bal |

</details>

<details>
<summary><b>raw_cls v18：ensemble 尝试（汇总）</b></summary>

Ensemble eval（同一组成员，比较 mean/weighted/stacking 等策略）：

| method | acc | macro_f1 | bal_acc | ece | n | file |
|---|---:|---:|---:|---:|---:|---|
| mean_prob | 0.6371 | 0.6128 | 0.6261 | 0.1255 | 248 | `paper_tables/raw_cls_ensemble_eval_mean_v18_best.json` |
| weighted (val-opt) | 0.5685 | 0.5471 | 0.5553 | 0.0638 | 248 | `paper_tables/raw_cls_ensemble_eval_weighted_v18_best.json` |
| weighted_acc (val-opt) | 0.5726 | 0.5378 | 0.5452 | 0.0425 | 248 | `paper_tables/raw_cls_ensemble_eval_weighted_acc_v18_best.json` |
| bias (val-opt) | 0.5726 | 0.5235 | 0.5240 | 0.0672 | 248 | `paper_tables/raw_cls_ensemble_eval_bias_v18_best.json` |
| bias_nll (val-opt) | 0.5806 | 0.5440 | 0.5407 | 0.0738 | 248 | `paper_tables/raw_cls_ensemble_eval_bias_nll_v18_best.json` |
| stacking | 0.5242 | 0.5072 | 0.5072 | 0.3182 | 248 | `paper_tables/raw_cls_ensemble_eval_stacking_v18_best.json` |

Multi-member mean-prob ensemble（不同 member 组合的 overall_acc）：

| overall_acc | file |
|---:|---|
| 0.6371 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json` |
| 0.6169 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_cloudmil.json` |
| 0.6129 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_pointnet2_curvrad.json` |
| 0.6129 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_pointnet2_n2048.json` |
| 0.6089 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet2.json` |
| 0.6089 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_pointtransformer.json` |
| 0.6089 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_seed1337_2020.json` |
| 0.6048 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_dgcnn_v2.json` |
| 0.6048 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_pointnet2_sa2_256.json` |
| 0.6048 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet_normcurvrad_pointnet2.json` |
| 0.6008 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_pointmlp.json` |
| 0.6008 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_pointnet2_curvrad_sa2_256.json` |
| 0.5968 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_sa2_256.json` |
| 0.5887 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet2_dgcnn.json` |
| 0.5887 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet2_sa2_256.json` |
| 0.5887 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_curvrad_sa2_256.json` |
| 0.5847 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_pointnet2_curvrad_seed1337_v2.json` |
| 0.5766 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_pointnet2_curvrad_seed1337.json` |
| 0.5565 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_seed1337.json` |
| 0.5565 | `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2_seed1337_v2.json` |

Seed ensemble（同一 exp:model，仅跨 seed 做 mean-prob）：

| overall_acc | model | exp | file |
|---:|---|---|---|
| 0.5766 | pointnet2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly_pointnet2_n1024 | `paper_tables/raw_cls_seed_ensemble_v18_pointnet2_n1024.json` |
| 0.5645 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly | `paper_tables/raw_cls_seed_ensemble_v18_xyzonly.json` |
| 0.5484 | dgcnn | dgcnn_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly_n1024 | `paper_tables/raw_cls_seed_ensemble_v18_dgcnn_n1024.json` |
| 0.5363 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyz_curvrad | `paper_tables/raw_cls_seed_ensemble_v18_xyz_curvrad.json` |
| 0.5323 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyz_normcurvrad | `paper_tables/raw_cls_seed_ensemble_v18_xyz_normcurvrad.json` |
| 0.5282 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly_bestacc | `paper_tables/raw_cls_seed_ensemble_v18_pointnet_bestacc.json` |
| 0.5242 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly_unbalanced_ce_none | `paper_tables/raw_cls_seed_ensemble_v18_pointnet_unbalanced_ce_none.json` |
| 0.5161 | pointnet_cloudset | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_cloudset | `paper_tables/raw_cls_seed_ensemble_v18_pointnet_cloudset.json` |
| 0.5040 | pointnet2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyz_curvrad_pointnet2_n1024 | `paper_tables/raw_cls_seed_ensemble_v18_pointnet2_xyz_curvrad_seed1337.json` |
| 0.5040 | pointnet_dsbn | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_dsbn | `paper_tables/raw_cls_seed_ensemble_v18_pointnet_dsbn.json` |
| 0.4032 | dgcnn_v2 | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly_dgcnn_v2_n1024 | `paper_tables/raw_cls_seed_ensemble_v18_dgcnn_v2_n1024.json` |
| 0.3427 | pointmlp | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly_pointmlp_n1024_v2 | `paper_tables/raw_cls_seed_ensemble_v18_pointmlp_n1024_v2.json` |
| 0.3226 | point_transformer | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly_pointtransformer_n1024_v2 | `paper_tables/raw_cls_seed_ensemble_v18_pointtransformer_n1024_v2.json` |

</details>

<details>
<summary><b>raw_cls v13：已跑过的 baselines（paper summary）</b></summary>

Source: `paper_tables/raw_cls_table_v13.md`

| test_macro_f1 (mean±std) | test_acc (mean±std) | n | model | n_points | kfold_k | balanced | label_smoothing | extra_features | tta | dataset |
|---:|---:|---:|---|---:|---:|---|---:|---|---:|---|
| 0.2825±0.0494 | 0.3019±0.0461 | 15 | pointnet | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.2821±0.0735 | 0.2996±0.0725 | 15 | dgcnn | 4096 | 5 | yes | 0.100 | scale,log_points,objects_used | 8 | v13_main4 |
| 0.2794±0.0725 | 0.2957±0.0636 | 15 | pointnet | 4096 | 5 | yes | 0.100 | scale,log_points,objects_used | 8 | v13_main4 |
| 0.2491±0.0468 | 0.2806±0.0445 | 15 | dgcnn | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.2053±0.0273 | 0.2256±0.0287 | 15 | meta_mlp | 0 | 5 | yes | 0.100 | scale,log_points,objects_used | 0 | v13_main4 |
| 0.1885±0.0324 | 0.2134±0.0297 | 15 | geom_mlp | 4096 | 5 | yes | 0.100 | (none) | 8 | v13_main4 |
| 0.1061±0.0000 | 0.2692±0.0000 | 1 | pointnet | 512 | 5 | no | 0.000 | (none) | 0 | v13_main4 |

</details>

<details>
<summary><b>raw_cls v13：readme full suite（k-fold×multi-seed 聚合）</b></summary>

Source: `paper_tables/raw_cls_summary.md`

| test_acc | macro_f1 | bal_acc | ece | n | model | exp | input | extra | sampler | ls | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| 0.4176±0.0365 | 0.3796±0.0630 | 0.3838±0.0570 | 0.3649±0.0848 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1 | xyz | pos | bal | ls=0 | 0 |
| 0.4077±0.0556 | 0.3785±0.0751 | 0.3816±0.0751 | 0.3432±0.1061 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0 | xyz | pos | bal | ls=0 | 0 |
| 0.4116±0.0699 | 0.3694±0.0826 | 0.3785±0.0763 | 0.2778±0.0734 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat | xyz | pos | bal | ls=0.1 | 0 |
| 0.4127±0.0644 | 0.3620±0.0862 | 0.3673±0.0821 | 0.3900±0.1072 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_bbox | xyz | pos | bal | ls=0 | 0 |
| 0.3820±0.0715 | 0.3545±0.0913 | 0.3698±0.0922 | 0.3268±0.1210 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_unbal_ls0 | xyz | - | unbal | ls=0 | 0 |
| 0.3928±0.0678 | 0.3521±0.0874 | 0.3712±0.0805 | 0.3403±0.1084 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_unbal_ls0 | xyz | pos | unbal | ls=0 | 0 |
| 0.3837±0.0963 | 0.3515±0.1005 | 0.3667±0.1000 | 0.3682±0.1465 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_bboxpca | xyz | pos | bal | ls=0 | 0 |
| 0.3553±0.0631 | 0.3510±0.0660 | 0.3982±0.0686 | 0.1376±0.0546 | 15 | pointnet | teeth3ds_fdi_pretrain_finetune_lr3e4 | xyz | - | bal | ls=0.1 | 8 |
| 0.3668±0.0674 | 0.3483±0.0699 | 0.3875±0.0667 | 0.1687±0.0829 | 15 | pointnet | teeth3ds_fdi_pretrain_finetune | xyz | - | bal | ls=0.1 | 8 |
| 0.3925±0.0631 | 0.3425±0.0678 | 0.3502±0.0669 | 0.3014±0.0741 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0 | xyz | - | bal | ls=0.1 | 0 |
| 0.3452±0.0649 | 0.3421±0.0726 | 0.3835±0.0686 | 0.1874±0.0626 | 15 | pointnet | supcon | xyz | - | bal | ls=0.1 | 8 |
| 0.3952±0.0675 | 0.3386±0.0799 | 0.3468±0.0715 | 0.3678±0.1043 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0 | xyz | pos | bal | ls=0 | 0 |
| 0.3472±0.0750 | 0.3323±0.0685 | 0.3623±0.0711 | 0.1623±0.0649 | 15 | dgcnn | baseline | xyz | - | bal | ls=0.1 | 8 |
| 0.3442±0.0784 | 0.3297±0.0724 | 0.3731±0.0808 | 0.1788±0.0900 | 15 | pointnet | teeth3ds_fdi_pretrain_finetune_freeze10 | xyz | - | bal | ls=0.1 | 8 |
| 0.3857±0.0665 | 0.3279±0.0730 | 0.3481±0.0713 | 0.4011±0.0902 | 15 | dgcnn | dgcnn_supcon_norot_tta0_posfeat_n1024_ls0_dropout0p1 | xyz | pos | bal | ls=0 | 0 |
| 0.3403±0.0661 | 0.3261±0.0680 | 0.3706±0.0864 | 0.1768±0.0551 | 15 | pointnet | scale_token | xyz | meta | bal | ls=0.1 | 8 |
| 0.3629±0.0611 | 0.3260±0.0627 | 0.3461±0.0621 | 0.1752±0.0502 | 15 | pointnet | feat_normcurv | xyz,normals,curvature,radius | - | bal | ls=0.1 | 8 |
| 0.3318±0.0690 | 0.3224±0.0844 | 0.3625±0.0863 | 0.1794±0.0627 | 15 | pointnet | baseline | xyz | - | bal | ls=0.1 | 8 |
| 0.3522±0.1228 | 0.3218±0.1233 | 0.3397±0.1132 | 0.3081±0.1178 | 15 | pointnet_pos_moe | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_posmoe | xyz | pos | bal | ls=0 | 0 |
| 0.3942±0.0408 | 0.3200±0.0303 | 0.3625±0.0581 | 0.3006±0.0957 | 2 | dgcnn | dgcnn_supcon_norot_tta0_posfeat | xyz | pos | bal | ls=0.1 | 0 |
| 0.3386±0.0686 | 0.3195±0.0755 | 0.3622±0.0740 | 0.1836±0.0727 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon | xyz | - | bal | ls=0.1 | 8 |
| 0.3614±0.0881 | 0.3059±0.1005 | 0.3222±0.0952 | 0.3165±0.1002 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_normcurv | xyz,normals,curvature,radius | pos | bal | ls=0.1 | 0 |
| 0.3045±0.0519 | 0.2841±0.0557 | 0.3296±0.0616 | 0.1904±0.0713 | 15 | pointnet | pretrain_finetune | xyz | - | bal | ls=0.1 | 8 |
| 0.2692±0.0000 | 0.1061±0.0000 | 0.2500±0.0000 | 0.0910±0.0000 | 1 | dgcnn | smoke_bs32 | xyz | - | bal | ls=0.1 | 0 |

</details>

<details>
<summary><b>raw_cls：其他数据版本（已跑过汇总）</b></summary>

每个版本的完整配置/聚合统计请以对应 `paper_tables/raw_cls_summary_*.md` 为准。

v14_main4_rgb（`paper_tables/raw_cls_summary_v14_main4_rgb.md`）：

| test_acc | macro_f1 | bal_acc | ece | n | model | exp | input | extra | sampler | ls | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| 0.3702±0.0571 | 0.3051±0.0666 | 0.3287±0.0567 | 0.3998±0.1135 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_rgb | xyz,rgb | pos | bal | ls=0 | 0 |

v15_main4_seg_top1（`paper_tables/raw_cls_summary_v15_main4_seg_top1.md`）：

| test_acc | macro_f1 | bal_acc | ece | n | model | exp | input | extra | sampler | ls | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| 0.3891±0.0624 | 0.3519±0.0756 | 0.3826±0.0793 | 0.3806±0.0878 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_top1 | xyz | pos | bal | ls=0 | 0 |

v16_main4_seg_all sweep（`paper_tables/raw_cls_summary_v16_main4_seg_all_sweep.md`）：

| test_acc | macro_f1 | bal_acc | ece | n | model | exp | input | extra | sampler | ls | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| 0.4336±0.0658 | 0.4012±0.0844 | 0.4145±0.0828 | 0.3879±0.0699 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all | xyz | pos | bal | ls=0 | 0 |
| 0.4308±0.0547 | 0.3865±0.0775 | 0.3984±0.0806 | 0.3721±0.0561 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0_seg_all | xyz | pos | bal | ls=0 | 0 |
| 0.4212±0.0732 | 0.3852±0.0910 | 0.4081±0.0992 | 0.3599±0.1002 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_seg_all | xyz | pos | bal | ls=0 | 0 |

v17_main4_seg_all_cloudid ablate（`paper_tables/raw_cls_summary_v17_main4_seg_all_cloudid_ablate.md`）：

| test_acc | macro_f1 | bal_acc | ece | n | model | exp | input | extra | sampler | ls | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| 0.4711±0.0502 | 0.4393±0.0498 | 0.4537±0.0570 | 0.3251±0.0511 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid | xyz,cloud_id | pos | bal | ls=0 | 0 |
| 0.4398±0.0700 | 0.3889±0.0781 | 0.4105±0.0798 | 0.3242±0.0754 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_onehot | xyz,cloud_id_onehot | pos | bal | ls=0 | 0 |

v19_main4_seg_all_cloudid_eq_n8192（`paper_tables/raw_cls_summary_v19_main4_seg_all_cloudid_eq_n8192_rotz_tta8_xyz_curvrad.md`）：

| test_acc | macro_f1 | bal_acc | ece | n | model | exp | input | extra | sampler | ls | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| 0.5119±0.0759 | 0.4709±0.0932 | 0.5014±0.0841 | 0.1907±0.0635 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_n8192_xyzonly | xyz | pos | bal | ls=0 | 8 |
| 0.5013±0.0582 | 0.4583±0.0683 | 0.4868±0.0770 | 0.2395±0.0495 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_n8192_xyz_curvrad | xyz,curvature,radius | pos | bal | ls=0 | 8 |
| 0.4656±0.0768 | 0.4361±0.0812 | 0.4540±0.0613 | 0.3292±0.0952 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_n8192 | xyz,cloud_id | pos | bal | ls=0 | 0 |

v20_main4_seg_all_cloudid_eq_pool16384（`paper_tables/raw_cls_summary_v20_main4_seg_all_cloudid_eq_pool16384_rotz_tta8_xyzonly.md`）：

| test_acc | macro_f1 | bal_acc | ece | n | model | exp | input | extra | sampler | ls | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| 0.5184±0.0545 | 0.4792±0.0648 | 0.5018±0.0612 | 0.2273±0.0558 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_pool16384_xyzonly | xyz | pos | bal | ls=0 | 8 |
| 0.4657±0.0691 | 0.4503±0.0686 | 0.4534±0.0645 | 0.3217±0.0643 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_pool16384 | xyz,cloud_id | pos | bal | ls=0 | 0 |

v21_main4_seg_small1_cloudid_eq（`paper_tables/raw_cls_summary_v21_main4_seg_small1_cloudid_eq_rotz_tta8_xyzonly.md`）：

| test_acc | macro_f1 | bal_acc | ece | n | model | exp | input | extra | sampler | ls | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| 0.3941±0.0737 | 0.3788±0.0649 | 0.3958±0.0804 | 0.2593±0.0903 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_small1_cloudid_eq_xyzonly | xyz | pos | bal | ls=0 | 8 |

v22_main4_seg_small3_cloudid_eq（`paper_tables/raw_cls_summary_v22_main4_seg_small3_cloudid_eq_rotz_tta8_xyzonly.md`）：

| test_acc | macro_f1 | bal_acc | ece | n | model | exp | input | extra | sampler | ls | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| 0.4777±0.0852 | 0.4185±0.0745 | 0.4488±0.0702 | 0.2401±0.1072 | 15 | pointnet | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_small3_cloudid_eq_xyzonly | xyz | pos | bal | ls=0 | 8 |

</details>

### domain_shift（普通↔专家标注 A→B）

<details>
<summary><b>domain_shift：各方法表现（fold=0, seeds=3 聚合）</b></summary>

Source: `paper_tables/domain_shift_summary.md`

| direction | exp | model | test_acc | macro_f1 | bal_acc | ece | n | input | extra | ls | tta |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|
| A2B_专家标注_to_普通标注 | groupdro | pointnet | 0.4333±0.1528 | 0.3662±0.1685 | 0.4232±0.1338 | 0.2018±0.0680 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_普通标注_to_普通标注 | baseline | pointnet | 0.4000±0.0866 | 0.3597±0.1544 | 0.3815±0.1332 | 0.1857±0.1226 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_普通标注_to_专家标注 | baseline | dgcnn | 0.2667±0.1155 | 0.3532±0.1306 | 0.2917±0.1102 | 0.4462±0.1639 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_专家标注_to_专家标注 | baseline | pointnet | 0.3333±0.0577 | 0.3490±0.0159 | 0.3611±0.1203 | 0.1844±0.0350 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_普通标注_to_专家标注 | coral | pointnet | 0.3000±0.1732 | 0.3422±0.1727 | 0.3333±0.1909 | 0.2960±0.0418 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_普通标注_to_专家标注 | pos_moe | pointnet_pos_moe | 0.3333±0.0577 | 0.3407±0.0449 | 0.3889±0.1049 | 0.2745±0.0796 | 3 | xyz | pos | ls=0.1 | 8 |
| A2B_专家标注_to_普通标注 | coral | pointnet | 0.3500±0.0866 | 0.3007±0.1168 | 0.3458±0.0807 | 0.1484±0.0077 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_普通标注_to_普通标注 | baseline | dgcnn | 0.3000±0.0000 | 0.2953±0.0099 | 0.2958±0.0134 | 0.3202±0.0499 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_专家标注_to_普通标注 | baseline | pointnet | 0.3667±0.1756 | 0.2904±0.1913 | 0.3887±0.1393 | 0.1975±0.0799 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_专家标注_to_专家标注 | baseline | dgcnn | 0.2333±0.0577 | 0.2565±0.0593 | 0.2778±0.0481 | 0.3248±0.0590 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_普通标注_to_专家标注 | groupdro | pointnet | 0.1667±0.0577 | 0.2532±0.0881 | 0.1944±0.0636 | 0.3775±0.0449 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_普通标注_to_专家标注 | dsbn | pointnet_dsbn | 0.2000±0.1000 | 0.2526±0.1277 | 0.2083±0.1102 | 0.2749±0.1141 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_专家标注_to_普通标注 | pos_moe | pointnet_pos_moe | 0.3167±0.0764 | 0.2456±0.0906 | 0.2857±0.0804 | 0.0798±0.0580 | 3 | xyz | pos | ls=0.1 | 8 |
| A2B_普通标注_to_专家标注 | baseline | pointnet | 0.2000±0.1000 | 0.2369±0.0972 | 0.2083±0.0722 | 0.3233±0.0751 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_专家标注_to_普通标注 | baseline | dgcnn | 0.3167±0.0289 | 0.2146±0.0535 | 0.3024±0.0655 | 0.2223±0.0480 | 3 | xyz | - | ls=0.1 | 8 |
| A2B_专家标注_to_普通标注 | dsbn | pointnet_dsbn | 0.3500±0.0000 | 0.1296±0.0000 | 0.2500±0.0000 | 0.3754±0.1888 | 3 | xyz | - | ls=0.1 | 8 |

</details>

<details>
<summary><b>domain_shift：cross - in 的掉分（baseline）</b></summary>

Source: `paper_tables/domain_shift_delta.md`

| direction | model | n | acc_in (mean±std) | acc_cross (mean±std) | Δacc (cross-in) | macro_f1_in (mean±std) | macro_f1_cross (mean±std) | Δmacro_f1 | ece_in (mean±std) | ece_cross (mean±std) | Δece |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 专家标注→普通标注 | dgcnn | 3 | 0.3000±0.0000 | 0.3167±0.0289 | 0.0167±0.0289 | 0.2953±0.0099 | 0.2146±0.0535 | -0.0808±0.0624 | 0.3202±0.0499 | 0.2223±0.0480 | -0.0979±0.0932 |
| 专家标注→普通标注 | pointnet | 3 | 0.4000±0.0866 | 0.3667±0.1756 | -0.0333±0.1155 | 0.3597±0.1544 | 0.2904±0.1913 | -0.0693±0.0718 | 0.1857±0.1226 | 0.1975±0.0799 | 0.0118±0.1606 |
| 普通标注→专家标注 | dgcnn | 3 | 0.2333±0.0577 | 0.2667±0.1155 | 0.0333±0.1528 | 0.2565±0.0593 | 0.3532±0.1306 | 0.0967±0.1583 | 0.3248±0.0590 | 0.4462±0.1639 | 0.1214±0.1681 |
| 普通标注→专家标注 | pointnet | 3 | 0.3333±0.0577 | 0.2000±0.1000 | -0.1333±0.0577 | 0.3490±0.0159 | 0.2369±0.0972 | -0.1121±0.1118 | 0.1844±0.0350 | 0.3233±0.0751 | 0.1389±0.0932 |

</details>

### prep2target（synthetic proxy）

- 当前最优（test_total 越小越好）：`constraints_occlusion`（λ_occ=0.1, clearance=0.5）→ test_total=0.0628±0.0004（seeds=3），见 `paper_tables/prep2target_summary.md`

<details>
<summary><b>prep2target：各方法表现（seeds=3 聚合）</b></summary>

Source: `paper_tables/prep2target_summary.md`

| test_total (mean±std) | test_chamfer (mean±std) | test_margin (mean±std) | test_occlusion (mean±std) | n | dataset | exp | model | n_points | latent_dim | cond_label | lambda_margin | lambda_occlusion | clearance |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|
| 0.0628±0.0004 | 0.0627±0.0004 | 0.0270±0.0006 | 0.0002±0.0000 | 3 | v1 | constraints_occlusion | p2t | 512 | 256 | plain | 0.0 | 0.1 | 0.5 |
| 0.0629±0.0003 | 0.0629±0.0003 | 0.0273±0.0001 | 0.0004±0.0001 | 3 | v1 | baseline | p2t | 512 | 256 | plain | 0.0 | 0.0 | 0.5 |
| 0.0659±0.0003 | 0.0633±0.0004 | 0.0258±0.0005 | 0.0004±0.0001 | 3 | v1 | constraints_margin | p2t | 512 | 256 | plain | 0.1 | 0.0 | 0.5 |
| 0.0661±0.0009 | 0.0631±0.0010 | 0.0262±0.0007 | 0.0003±0.0000 | 3 | v1 | multitask_constraints | p2t | 512 | 256 | plain | 0.1 | 0.1 | 0.5 |

</details>

### constraints（teeth3ds_prep2target_constraints）

- 当前最优（eval_test_total 越小越好）：0.057178（jaw + cut=z，λ_margin=0，λ_occ=0.1），exp=`p2tC_n256_seed1337_20251214_073510`，见 `paper_tables/constraints_summary.md`

<details>
<summary><b>constraints：不同 λ/切割模式的评估汇总</b></summary>

Source: `paper_tables/constraints_summary.md`

| eval_test_total | chamfer(val) | margin(val) | occ_contact(val) | min_d_p05(val) | λ_margin | λ_occ | occ_mode | cut_mode | exp |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 0.057178 | 0.057197 | 0.025533 | 0.028815 | 0.609575 | 0.000000 | 0.100000 | jaw | z | p2tC_n256_seed1337_20251214_073510 |
| 0.057351 | 0.057667 | 0.025988 | 0.028957 | 0.610356 | 0.000000 | 0.000000 | jaw | z | p2tC_n256_seed1337_20251214_035634 |
| 0.057683 | 0.057855 | 0.025507 | 0.028323 | 0.612372 | 0.000000 | 0.050000 | jaw | z | p2tC_n256_seed1337_20251214_044131 |
| 0.059680 | 0.057572 | 0.024341 | 0.028541 | 0.612372 | 0.100000 | 0.000000 | jaw | z | p2tC_n256_seed1337_20251214_040657 |
| 0.059917 | 0.057402 | 0.024314 | 0.028803 | 0.611974 | 0.100000 | 0.100000 | jaw | z | p2tC_n256_seed1337_20251214_074825 |
| 0.068937 | 0.058836 | 0.021187 | 0.029239 | 0.613965 | 0.500000 | 0.000000 | jaw | z | p2tC_n256_seed1337_20251214_041713 |
| 0.081473 | 0.061951 | 0.019878 | 0.029340 | 0.607569 | 1.000000 | 0.000000 | jaw | z | p2tC_n256_seed1337_20251214_042818 |
| 0.122454 | 0.104892 | 0.036090 | 0.029235 | 0.612372 | 0.500000 | 0.100000 | jaw | z | p2tC_n256_seed1337_20251214_001951 |
| 0.260095 | 0.247442 | 0.157751 | 0.032760 | 0.586718 | 0.100000 | 0.100000 | jaw | z | smoke_p2t_constraints_20260116_090938 |
| 0.260095 | 0.247442 | 0.157751 | 0.032760 | 0.586718 | 0.100000 | 0.100000 | jaw | z | smoke_p2t_constraints_20260116_091242 |
| 0.277740 | 0.280509 | 0.181870 | 0.001856 | 3.320355 | 0.000000 | 0.100000 | tooth | plane | p2tC_plane_tooth_n128_seed1337_20260116_104533 |
|  |  |  |  |  | 0.500000 | 0.100000 | jaw | z | p2tC_n256_seed1337_20251214_001741 |
|  |  |  |  |  | 0.500000 | 0.100000 | jaw | z | p2tC_n256_seed1337_20251214_001821 |
|  |  |  |  |  | 0.500000 | 0.100000 | jaw | z | p2tC_n256_seed1337_20251214_001850 |

</details>
