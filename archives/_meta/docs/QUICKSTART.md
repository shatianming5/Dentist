# Quickstart

面向第一次打开本仓库的用户：**如何验证能跑、能出结果、能复现关键结论**。

## 0) 环境

推荐 conda：

```bash
conda env create -f environment.yml
conda activate dentist
```

或 pip（需要你自行安装匹配 CUDA 的 PyTorch）：

```bash
pip install -r requirements.txt
```

## 1) 最小 smoke（CPU，~分钟级）

这条命令用于验证：配置系统/训练脚本/落盘产物都正常。

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

检查输出目录：

- `runs/raw_cls/v13_main4/baseline/pointnet/fold=0/seed=0/metrics.json`
- `runs/raw_cls/v13_main4/baseline/pointnet/fold=0/seed=0/logs.txt`

## 2) raw_cls 当前 SOTA（GPU，耗时取决于机器）

raw_cls 的 repo 内 SOTA 在 `v18_main4_seg_all_cloudid_eq` 上，通过 3 个成员模型的 mean-prob ensemble 得到。

1) 先跑 3 个成员 full（k=5×seeds=3）：

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

结果文件：

- `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json`
- 更完整指标（含 confusion_matrix）：`paper_tables/raw_cls_ensemble_eval_mean_v18_best.json`

## 3) 看“项目是否可评估/可审计”

建议按这个顺序浏览：

- `docs/INDEX.md`：项目地图 + 关键入口
- `docs/experiment.md`：有哪些实验、怎么跑、看哪些指标
- `paper_tables/`：汇总表/结论证据是否齐全
- `runs/`：是否每个 run 都有 `metrics.json`、`preds_*.jsonl`、`config.yaml`、`logs.txt`

