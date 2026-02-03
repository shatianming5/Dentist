# scripts/ Index

`scripts/` 里包含本仓库几乎所有“可复现入口”：数据构建、训练/评估、汇总表格、误差审计与文档审计。

## 最常用入口（用户视角）

- 统一训练/评估入口：`scripts/train.py`
- raw/ 转点云：`scripts/convert_ccb2_bin.py`
- raw_cls 构建数据集：`scripts/phase1_build_raw_cls.py`
- raw_cls 训练主逻辑：`scripts/phase3_train_raw_cls_baseline.py`
- 体内库（STL）索引：`scripts/internal_db/index_internal_db.py`
- 体内库（STL）整牙可视化（HTML）：`scripts/vis_internal_db_case.py`
- 体内库（STL）热力图上色/点云叠加（HTML+截图）：`scripts/vis_internal_db_heatmap.py`
- 汇总 runs：`scripts/aggregate_runs.py`
- 期刊表格：`scripts/paper_table_raw_cls.py`
- 集成（multi-member mean-prob）：`scripts/paper_raw_cls_multi_ensemble.py`
- 集成（stacking/weighted）：`scripts/paper_raw_cls_stacking_ensemble.py`
- 误差/不确定样本审计：`scripts/raw_cls_ensemble_error_audit.py`

## 一键脚本（批跑）

- raw_cls readme 批跑：`scripts/run_full_rawcls_readme.sh`
- domain_shift readme 批跑：`scripts/run_full_domain_shift_readme.sh`
- prep2target readme 批跑：`scripts/run_full_prep2target_readme.sh`

## 审计/自检

- 文档闭环审计（docs-spec）：`scripts/docs_audit.py`
- 论文范围/协议审计：`scripts/paper_audit.py`
- 期刊自检报告：`scripts/journal_audit.py`
