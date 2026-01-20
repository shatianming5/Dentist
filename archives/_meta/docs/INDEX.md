# Project Index

本仓库包含两条主线：

1) **Teeth3DS / 3DTeethLand（公开数据）**：数据整理、统计、预训练与可复现实验协议。  
2) **raw/（CloudCompare CCB2 bin + 修复体类型标签）**：点云提取、raw_cls（修复体类型分类）基线/消融/集成与期刊级报告产物。

> 研究蓝图（不等于已实现）：`plan_report.md`。论文范围对齐：`PAPER_SCOPE.md`。

## 目录速览

- `archives/`：原始压缩包备份（不改内容）。
- `data/`：Teeth3DS + splits + landmarks（已解压整理）。
- `raw/`：修复体 CCB2 `*.bin` + 标注表（Excel）。
- `converted/`：`raw/` 转换产物（npz/ply + manifest + labels）。
- `processed/`：训练用“版本化数据集”（含 `index.jsonl` / `label_map.json` / `samples/*.npz`）。
- `metadata/`：切分文件、统计中间产物等（如 `metadata/splits_raw_case_kfold.json`）。
- `configs/`：统一 YAML 配置（`scripts/train.py` 读取；支持 defaults 链）。
- `scripts/`：数据构建/训练/评估/汇总/表格脚本（主要入口都在这里）。
- `runs/`：所有训练/评估 run 的落盘目录（按 `dataset/exp/model/fold/seed` 组织）。
- `paper_tables/`：汇总表、审计表、集成结果 JSON（期刊-ready 输出）。
- `docs/`：闭环文档（plan/mohu/experiment/proof）。
- `.rd_queue/`：tmux 队列运行产物（logs/results/queue*.json）。

## 阅读顺序（用户视角）

1) `docs/QUICKSTART.md`：先跑 smoke，确认能出 `metrics.json`
2) `docs/USER_REVIEW.md`：用户视角的“可读/可跑/可复现/可审计”验收
3) `docs/RESULTS.md`：知道“结果看哪里”
4) `docs/REPO_MAP.md`：知道“代码入口在哪里/目录是什么”
5) `paper_tables/INDEX.md`：知道“表格与证据都有哪些”

## raw_cls（修复体类型分类）主流程

### 1) raw → converted（点云提取）

- 入口：`scripts/convert_ccb2_bin.py`
- 标注对齐：`scripts/label_converted_raw.py`
- 产物：`converted/raw/manifest_with_labels.json` + `converted/raw/labels.csv`

### 2) converted → processed/raw_cls（版本化数据集）

- 入口：`scripts/phase1_build_raw_cls.py`
- 关键产物：
  - `processed/raw_cls/<version>/index.jsonl`（每个 case 一行，含 split/统计/used_clouds 等）
  - `processed/raw_cls/<version>/label_map.json`（类别→id）
  - `processed/raw_cls/<version>/samples/**/*.npz`（训练用点云）

常用版本：
- `v13_main4`：早期 baseline（4 类：充填/全冠/桩核冠/高嵌体）。
- `v18_main4_seg_all_cloudid_eq`：seg-all + cloud_id + equal cloud sampling 的强版本（当前 raw_cls SOTA 所在）。

### 3) 训练/评估（统一入口 scripts/train.py）

- 统一 runner：`scripts/train.py --config <yaml> --fold <k> --seed <s> [--set k=v]`
- raw_cls 训练主体：`scripts/phase3_train_raw_cls_baseline.py`
- 输出目录（典型）：
  - `runs/raw_cls/<dataset>/<exp>/<model>/fold=<k>/seed=<s>/`
  - 必含：`metrics.json`, `preds_{val,test}.jsonl`, `config.yaml`, `logs.txt`

### 4) 汇总与表格

- 汇总 runs：`scripts/aggregate_runs.py` → `paper_tables/raw_cls_summary_*.md`
- 期刊级表格：`scripts/paper_table_raw_cls.py`（v13 主表）

### 5) 集成（ensemble）与误差审计

- 多 member × 多 seed mean-prob ensemble：`scripts/paper_raw_cls_multi_ensemble.py`
- per-fold stacking/weighted/bias（会用 val 学权重再测 test）：`scripts/paper_raw_cls_stacking_ensemble.py`
- SOTA 误差/不确定样本清单：`scripts/raw_cls_ensemble_error_audit.py` → `paper_tables/raw_cls_error_audit_*.json`

## 当前 raw_cls SOTA（repo 内）

数据：`processed/raw_cls/v18_main4_seg_all_cloudid_eq`（n=248）

- **SOTA（multi-member × multi-seed mean-prob）**  
  结果：`paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json`
  - overall_acc=0.6371（mean_acc_over_folds=0.6384±0.0449）
  - members：
    - PointNet xyz-only：`teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly:pointnet`
    - PointNet xyz+curv+rad：`..._xyz_curvrad:pointnet`
    - PointNet2 xyz-only (n=1024)：`..._xyzonly_pointnet2_n1024:pointnet2`

复现（先跑成员 full，再算 ensemble）：

1) 跑 3 个成员 full：
   - `CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_rawcls_v18_main4_seg_all_cloudid_eq_rotz_tta8_xyzonly.sh`
   - `CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_rawcls_v18_main4_seg_all_cloudid_eq_rotz_tta8_xyz_curvrad.sh`
   - `CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_rawcls_v18_main4_seg_all_cloudid_eq_rotz_tta8_pointnet2_xyzonly_n1024.sh`
2) 计算 multi-ensemble：
   - `python3 scripts/paper_raw_cls_multi_ensemble.py --runs-root runs/raw_cls/v18_main4_seg_all_cloudid_eq --member teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly:pointnet --member teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyz_curvrad:pointnet --member teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyzonly_pointnet2_n1024:pointnet2 --seeds 1337,2020,2021 --folds 0,1,2,3,4 --out paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json`

## 文档入口（闭环）

- `docs/plan.md`：本仓库“期刊可复现”闭环计划（P####/C####）。
- `docs/experiment.md`：实验台账（E####，含可跑命令、Smoke/Full checkbox）。
- `docs/mohu.md`：plan↔实现差距/模糊点清单（M####）。
- `docs/proof.md`：claims→evidence 的 proof 审计产物。
- `docs/USER_REVIEW.md`：从用户角度对仓库可读性/可复现性的评估与验收路径。
- `PAPER_PROTOCOL.md`：复现协议（环境/命令/产物约定）。
- `PAPER_SCOPE.md`：论文叙事边界（implemented vs proposed）。
- `plan.md` / `plan_report.md`：更偏“研究/工程路线图”（包含未实现内容；用于长期规划，不作为 claims 证据来源）。
- `runs/README.md`：run 目录结构与应当包含的证据文件。
- `paper_tables/INDEX.md`：`paper_tables/` 的快速索引。

## 开发/扩展（最常见改动点）

- 增加新 raw_cls 模型：`scripts/phase3_train_raw_cls_baseline.py`（添加 model 分支 + 保存指标）
- 新实验配置：`configs/raw_cls/exp/*.yaml`（建议复用 defaults，避免参数漂移）
- 新实验一键脚本：`scripts/run_full_rawcls_*.sh`（产出汇总到 `paper_tables/`）
- 新集成/分析：优先写到 `scripts/paper_*.py` 或 `scripts/*_audit.py`，输出落 `paper_tables/`

更多入口索引：

- `scripts/INDEX.md`
- `configs/INDEX.md`
