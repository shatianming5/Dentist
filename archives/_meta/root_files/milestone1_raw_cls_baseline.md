# Milestone 1：raw 修复体分类闭环（执行记录）

目标：按 `plan.md` 的 Milestone 1，把 `raw/`（CCB2）数据做成**可训练的分类数据集**并跑通一个**可复现 baseline**，输出指标与误差分析，用结果反推“样本定义/清洗规则”是否合理。

---

## 1) 已完成的落地产物

### 1.1 数据集构建脚本（Phase 1）
- 脚本：`scripts/phase1_build_raw_cls.py`
- 输出：`processed/raw_cls/<version>/index.jsonl` + `samples/**/*.npz` + `report.md`
- 新增能力：
  - `--include-name-regex/--exclude-name-regex`：按 `exported_clouds[].name` 过滤子点云
  - `--select-topk/--select-smallk`：按点数选最大/最小 K 个子点云后再合并
  - `--prefer-name-regex`：如果存在匹配的子点云，优先只从这些子点云里选

### 1.2 baseline 训练脚本（Phase 3）
- 脚本：`scripts/phase3_train_raw_cls_baseline.py`
- 输出目录：`runs/raw_cls_baseline/<exp_name>/`
  - `config.json`：实验配置（含 seed/增强等）
  - `metrics.json`：acc / macro-F1 / confusion（含按 `source=普通标注/专家标注` 的 test 子集统计）
  - `confusion_test.csv`：测试集混淆矩阵
  - `errors_test.csv`：错分样本列表（可回溯到 `case_key` 与 `sample_npz`）
  - `preds_test.jsonl`：每个 test 样本的预测概率（便于进一步分析/可视化）

---

## 2) 关键发现：选“最小 segmented$ 子点云”效果最好

对比多种样本定义后发现：把每个 `*.bin` 的样本定义为 **“segmented$ 子点云里点数最少的那一个”**（而不是全合并、也不是固定选 `Mesh.sampled.segmented`）能显著提升分类效果。

这很符合直觉：修复体目标往往是场景里更小的一个对象（全口/上下颌背景会很大）。

---

## 3) 推荐可复现命令（当前最优）

### 3.1 构建数据集（推荐版本：v6）
```bash
python3 scripts/phase1_build_raw_cls.py \
  --out processed/raw_cls/v6 \
  --include-name-regex 'segmented$' \
  --select-smallk 1
```

### 3.2 跑 baseline（PointNet）
```bash
python3 scripts/phase3_train_raw_cls_baseline.py \
  --data-root processed/raw_cls/v6 \
  --device auto \
  --epochs 120 \
  --patience 25 \
  --batch-size 64
```

对应一次可参考的结果目录：
- `runs/raw_cls_baseline/pointnet_n4096_seed1337_20251213_150439/metrics.json`

---

## 4) 下一步建议（仍在 Milestone 1 范围内）

1) **收敛标签体系**：`拔除` 样本极少（仅 1），会导致宏平均指标不稳定；建议合并到“未知/其他”或从分类任务里暂时移除。
2) **域差异分析**：`metrics.json` 已输出 test 按 `source` 的指标，建议针对“专家标注/普通标注”分别看错分样本（可能存在标注口径或预处理差异）。
3) **样本定义再升级（Definition C/D）**：如果后续要做“paired 生成”（stump→crown），需要进一步把对象语义结构化（至少识别 stump/target/opposing）。

---

## 5) 更稳指标：移除极少类后的评估/训练

### 5.1 仅用于“评估更稳”的做法（推荐）
保持训练/数据不变（如 `processed/raw_cls/v6`），但在汇报指标时**忽略极少类**（例如：`拔除/未知`），只看主任务 4 类（`充填/全冠/桩核冠/高嵌体`）。

一次示例（基于 `runs/raw_cls_baseline/pointnet_n4096_seed1337_20251213_150439/` 的 test 预测）：
- 去掉 `拔除/未知` 后：样本数 30，accuracy≈0.367，macro-F1≈0.403

### 5.2 直接从数据集中移除极少类（训练也变成少类）
如果希望训练/评估都只针对主任务类，可以在 Phase 1 构建时丢弃标签：

- 仅移除 `拔除`：
```bash
python3 scripts/phase1_build_raw_cls.py \
  --out processed/raw_cls/v11 \
  --include-name-regex 'segmented$' \
  --select-smallk 1 \
  --drop-labels 拔除
```

- 同时移除 `拔除,未知`（只保留 4 类，更稳定）：
```bash
python3 scripts/phase1_build_raw_cls.py \
  --out processed/raw_cls/v12 \
  --include-name-regex 'segmented$' \
  --select-smallk 1 \
  --drop-labels 拔除,未知
```

