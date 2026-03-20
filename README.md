# Dentist — Protocol-Induced Domain Shift in Dental Restoration Segmentation

# 牙科修复体分割中的协议诱导域偏移

> **Paper / 论文**: *Protocol-Induced Domain Shift in Dental Restoration Segmentation: A Multi-Method Benchmark with Feature-Space Analysis and Data-Centric Mitigation*
>
> **Target journals / 目标期刊**: CMPB / CIBM (~7 IF) or DMFR / J Dentistry / Dental Materials (~3-5 IF)
>
> **Status / 状态**: Manuscript ready / 论文就绪 (8.5/10 after 5 review rounds), 511 experimental configurations / 实验配置

---

## Highlights / 亮点

- **First dual-protocol benchmark / 首个双协议基准测试** for dental restoration segmentation on 3D intraoral scans (balanced vs. natural scanning protocols) / 用于3D口内扫描的牙科修复体分割（标准化 vs. 临床常规扫描协议）
- **7 methods / 7种方法** compared: DGCNN, PointNet, PointNet++, Point Transformer, Random Forest, DINOv2-probe, MV-ViT (fine-tuned)
- **Feature-space analysis / 特征空间分析**: UMAP visualization, A-distance, MMD² quantifying protocol-induced domain gap / UMAP可视化、A-距离、MMD²量化协议诱导的域差距
- **Data-centric finding / 数据为中心的发现**: Simple protocol mixing (+18.9% mIoU) outperforms all model-centric mitigations / 简单协议混合训练（+18.9% mIoU）优于所有模型策略（BN适配、预训练、对抗对齐）
- **PAFA negative result / PAFA负结果**: Protocol-Adversarial Feature Alignment significantly hurts (Δ = −0.047, p < 0.001) / 对抗特征对齐显著降低性能，强化了数据为中心的结论
- **511 experiments / 511个实验** across 8 tables with paired statistical testing (Wilcoxon, bootstrap CI, Cohen's d) / 8个表格，配对统计检验

## Key Results / 核心结果

| Strategy / 策略 | Balanced mIoU | Natural mIoU | Verdict / 结论 |
|----------|:------------:|:------------:|---------|
| Within-protocol only / 仅协议内训练 | 0.789 | 0.637 | Baseline / 基线 |
| Protocol mixing / 协议混合 | **0.810** | **0.758** | ✅ Best / 最优 (+18.9% natural) |
| BN adaptation / BN适配 | — | 0.589 | ❌ Worse / 更差 |
| Teeth3DS pre-training / 预训练 | — | 0.755 | ❌ No gain / 无提升 |
| PAFA (adversarial DA / 对抗DA) | 0.701 | 0.659 | ❌ Significantly hurts / 显著降低 |

---

## Repository Structure / 仓库结构

```
Dentist/
├── paper_draft.md          # Full manuscript / 完整论文 (~8,800 words, 8 tables)
├── paper_tables/           # All experimental results / 所有实验结果 (JSON/MD, 80+ files)
├── scripts/                # Training & analysis scripts / 训练与分析脚本 (50+)
│   ├── phase3_train_raw_seg.py      # Main segmentation training / 主分割训练
│   ├── train_pafa.py                # PAFA adversarial alignment / PAFA对抗对齐
│   ├── run_pafa_experiments.py      # Parallel experiment launcher / 并行实验启动器
│   └── ...
├── configs/                # YAML experiment configurations / YAML实验配置
├── assets/                 # Figures / 图表 (UMAP, boxplots, qualitative)
├── runs/                   # Experiment outputs / 实验输出 (gitignored, local only)
├── processed/              # Processed datasets / 处理后数据集 (gitignored, local only)
├── AUTO_REVIEW.md          # 5 rounds of automated review / 5轮自动审稿记录
├── REVIEW_STATE.json       # Current review status / 当前审稿状态
├── RESEARCH_PIPELINE_REPORT.md  # Full research log / 完整研究日志
├── IDEA_REPORT.md          # Initial idea exploration / 初始想法探索
└── TRAINING_INFRASTRUCTURE.md   # Model architecture docs / 模型架构文档
```

## Quick Navigation / 快速导航

| What / 内容 | Where / 位置 |
|------|-------|
| **Full paper / 完整论文** | [`paper_draft.md`](paper_draft.md) |
| **All results / 所有结果** | [`paper_tables/`](paper_tables/) |
| **Review history / 审稿历史** | [`AUTO_REVIEW.md`](AUTO_REVIEW.md) |
| **Research log / 研究日志** | [`RESEARCH_PIPELINE_REPORT.md`](RESEARCH_PIPELINE_REPORT.md) |
| **Training scripts / 训练脚本** | [`scripts/`](scripts/) |
| **Configs / 配置** | [`configs/`](configs/) |
| **Figures / 图表** | [`assets/`](assets/) |

---

## Quick Start / 快速开始

### 0) Environment / 环境

Conda recommended / 推荐 conda：

```bash
cd configs/env
conda env create -f environment.yml
conda activate dentist

# Optional: visualization dependencies / 可选：可视化依赖
pip install -r requirements_vis.txt
```

Or pip (install PyTorch with matching CUDA first) / 或 pip（需先安装匹配CUDA的PyTorch）：

```bash
cd configs/env
pip install -r requirements.txt
pip install -r requirements_vis.txt  # optional / 可选
```

### 1) Smoke Test / 冒烟测试 (CPU, ~minutes)

Verify configs, training scripts, and output artifacts / 验证配置、训练脚本和输出产物：

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

Check output / 检查输出:

- `runs/raw_cls/v13_main4/baseline/pointnet/fold=0/seed=0/metrics.json`
- `runs/raw_cls/v13_main4/baseline/pointnet/fold=0/seed=0/logs.txt`

### 2) Segmentation Benchmark / 分割基准测试 (GPU)

Train the dual-protocol benchmark (5-fold CV x 3 seeds x 7 methods) / 训练双协议基准（5折交叉验证 × 3种子 × 7方法）：

```bash
# DGCNN on balanced protocol / DGCNN标准化协议
python3 scripts/phase3_train_raw_seg.py \
  --model dgcnn_v2 \
  --data-root processed/raw_seg/v1 \
  --seed 1337 --fold 0 --epochs 100

# DGCNN on natural protocol / DGCNN临床常规协议
python3 scripts/phase3_train_raw_seg.py \
  --model dgcnn_v2 \
  --data-root processed/raw_seg/v2_natural \
  --seed 1337 --fold 0 --epochs 100

# PAFA adversarial alignment / PAFA对抗特征对齐
python3 scripts/train_pafa.py \
  --mode pafa --seed 42 --fold 0 --gpu 0
```

### 3) Reproduce All 511 Experiments / 复现全部511个实验

```bash
# Full expanded benchmark / 完整扩展基准 (25 configs x 5 folds x 3 seeds)
bash scripts/run_expanded_seg_experiments.sh

# PAFA experiments / PAFA实验 (2 modes x 5 folds x 3 seeds)
python3 scripts/run_pafa_experiments.py
```

---

## Paper / 论文

The full manuscript is in [`paper_draft.md`](paper_draft.md) (~8,800 words, 8 tables, 28 references).

完整论文在 [`paper_draft.md`](paper_draft.md)（约8,800词，8个表格，28篇参考文献）。

### Abstract / 摘要

Automated segmentation of dental restorations from intraoral 3D scans is essential for digital dentistry workflows. This study investigates how scanning protocol—balanced vs. natural—affects segmentation method ranking, reliability, and feature-space representations. Across 511 experimental configurations spanning 7 method families, we find that protocol mixing is the only reliable mitigation strategy, while model-centric approaches (BN adaptation, external pre-training, adversarial feature alignment) consistently fail.

口内3D扫描中牙科修复体的自动分割对数字化牙科工作流至关重要。本研究探讨扫描协议（标准化 vs. 临床常规）如何影响分割方法排名、可靠性和特征空间表征。在跨7个方法族的511个实验配置中，我们发现协议混合训练是唯一可靠的缓解策略，而所有模型策略（BN适配、外部预训练、对抗特征对齐）均失败。

### Tables / 表格

| Table / 表 | Content / 内容 | Data / 数据 |
|-------|---------|------|
| Table 1 | Dataset description / 数据集描述 | `paper_tables/dataset_description.md` |
| Table 2 | Main benchmark (7 methods x 2 protocols) / 主基准 | `paper_tables/expanded_benchmark_n25.json` |
| Table 3 | Pairwise statistical tests / 配对统计检验 | `paper_tables/pairwise_statistics_n25.json` |
| Table 4 | Protocol mixing results / 协议混合结果 | `paper_tables/protocol_mixing_summary.json` |
| Table 5 | BN adaptation / BN适配 | `paper_tables/bn_adaptation_results.json` |
| Table 6 | External pre-training (Teeth3DS) / 外部预训练 | `paper_tables/teeth3ds_pretraining_results.json` |
| Table 7 | Cross-dataset domain gap / 跨数据集域差距 | `paper_tables/cross_dataset_domain_gap.json` |
| Table 8 | PAFA adversarial alignment / PAFA对抗对齐 | `paper_tables/pafa_results.json` |

### Figures / 图表

| Figure / 图 | Description / 描述 | File / 文件 |
|--------|-------------|------|
| Figure 1 | Protocol comparison / 协议对比示例 | `assets/figure1_protocol_comparison.png` |
| Figure 2 | Per-case mIoU distribution / 逐案例mIoU分布 | `assets/figure2_boxplot_comparison.png` |
| Figure 3 | UMAP protocol gap / UMAP协议差距 | `assets/umap_protocol_gap.png` |
| Figure 4 | UMAP point features / UMAP点特征 | `assets/umap_point_features.png` |

---

## Data / 数据

### Dental Restoration Segmentation (Primary) / 牙科修复体分割（主数据集）

79 dental arch 3D surface meshes with per-point restoration labels, collected under two scanning protocols:

79个牙弓3D表面网格，含逐点修复体标签，在两种扫描协议下采集：

- **Balanced protocol / 标准化协议** (`processed/raw_seg/v1/`): Standardized, equal class representation / 标准化，类别均衡
- **Natural protocol / 临床常规协议** (`processed/raw_seg/v2_natural/`): Clinically routine scanning / 临床常规扫描

Each sample is an NPZ file containing / 每个样本为NPZ文件，包含:
- `points` (Mx3): XYZ coordinates / XYZ坐标
- `labels` (M,): Per-point binary restoration labels / 逐点二值修复体标签

### Teeth3DS (External / 外部数据集)

Public dental mesh dataset used for transfer learning experiments (§3.8).

用于迁移学习实验的公开牙科网格数据集（§3.8节）。

### Raw Data / 原始数据 (CloudCompare BIN)

Original CloudCompare CCB2 binary files in `raw/` / 原始CloudCompare CCB2文件在 `raw/` 下：
- Conversion script / 转换脚本: `scripts/convert_ccb2_bin.py`
- Converted meshes / 转换产物: `converted/raw/`

---

## Methods Benchmarked / 基准方法

| Method / 方法 | Type / 类型 | Reference / 参考 |
|--------|------|-----------|
| **DGCNN** | Dynamic graph CNN / 动态图CNN | Wang et al., 2019 |
| **PointNet** | Point-based MLP / 基于点的MLP | Qi et al., 2017a |
| **PointNet++** | Hierarchical point / 分层点 | Qi et al., 2017b |
| **Point Transformer** | Attention-based / 注意力机制 | Zhao et al., 2021 |
| **Random Forest** | Hand-crafted features / 手工特征 | Breiman, 2001 |
| **DINOv2-probe** | Foundation model probe / 基础模型探针 | Oquab et al., 2024 |
| **MV-ViT** | Multi-view fine-tuned / 多视角微调 | He et al., 2022 |

---

## Review History / 审稿历史

The paper went through 5 automated review rounds using the `aris-reviewer` agent:

论文经过5轮 `aris-reviewer` 自动审稿：

| Round / 轮次 | Score / 分数 | Verdict / 判定 | Key Issues / 主要问题 |
|-------|:-----:|---------|------------|
| 1 | 7.0 | NOT READY | Missing statistical tests / 缺少统计检验 |
| 2 | 8.0 | ALMOST | Needed feature-space analysis / 需要特征空间分析 |
| 3 | 7.5 | ALMOST | A-distance inconsistency / A-距离不一致 |
| 4 | 8.5 | READY | Minor fixes / 小修（运行数、种子说明） |
| 5 | 8.5 | READY | Seed footnote, limitations / 种子脚注、局限性 |

Full review logs / 完整审稿日志: [`AUTO_REVIEW.md`](AUTO_REVIEW.md)

---

## Citation / 引用

If you use this benchmark or dataset, please cite / 如使用本基准或数据集，请引用:

```bibtex
@article{dentist2026protocol,
  title={Protocol-Induced Domain Shift in Dental Restoration Segmentation:
         A Multi-Method Benchmark with Feature-Space Analysis
         and Data-Centric Mitigation},
  author={Authors},
  journal={Journal},
  year={2026}
}
```

## License / 许可

Dataset and code are provided for academic research purposes. See individual data sources for their respective licenses.

数据集和代码仅供学术研究使用。各数据来源的许可证请参见对应说明。
