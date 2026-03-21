# Dentist — Protocol-Induced Domain Shift in Dental Restoration Segmentation

# 牙科修复体分割中的扫描协议诱导域偏移

> **Paper / 论文**: *Protocol-Induced Domain Shift in Dental Restoration Segmentation: A Multi-Method Benchmark with Feature-Space Analysis and Data-Centric Mitigation*
>
> 📄 [English manuscript / 英文论文](paper_draft.md) ｜ 📄 [中文论文](paper_draft_zh.md)
>
> **Target journals / 目标期刊**: CMPB / CIBM (~7 IF) or DMFR / J Dentistry / Dental Materials (~3-5 IF)
>
> **Status / 状态**: ✅ Manuscript ready / 论文就绪 — **9.0/10** after 12 automated review rounds / 经12轮自动审稿, **715 experimental configurations / 实验配置**

---

## Highlights / 亮点

- **First dual-protocol benchmark / 首个双协议基准测试** for dental restoration segmentation on 3D intraoral scans (balanced vs. natural scanning protocols) / 用于3D口内扫描的牙科修复体分割（标准化 vs. 临床常规扫描协议）
- **9 methods, 7 paradigms / 9种方法，7种范式** compared: RF, PointNet, PointNet++, DGCNN, CurveNet, PointMLP, Point Transformer, DINOv2-MV, MV-ViT-ft / 对比：随机森林、PointNet、PointNet++、DGCNN、CurveNet、PointMLP、Point Transformer、DINOv2-MV、MV-ViT-ft
- **715 experiments / 715个实验** with 36 pairwise statistical tests (Mann–Whitney U, BH-FDR corrected) / 36组配对统计检验（Mann–Whitney U，BH-FDR校正）
- **Feature-space analysis / 特征空间分析**: UMAP visualization, A-distance, MMD² quantifying protocol-induced domain gap / UMAP可视化、A-距离、MMD²量化协议诱导的域差距
- **Data-centric finding / 数据为中心的发现**: Simple protocol mixing (+0.058 mIoU, p<0.001) outperforms all model-centric mitigations / 简单协议混合训练优于所有模型策略（BN适配、预训练、对抗对齐）
- **Triple negative result / 三重负结果**: BN adaptation (−0.293), PAFA adversarial alignment (−0.046), Teeth3DS pre-training (no effect) all fail / BN适配、对抗特征对齐、外部预训练均失败
- **Ranking reversal / 排名反转**: PointNet++ drops 2nd→7th, MV-ViT-ft rises 4th→1st under natural protocol / PointNet++从第2降至第7，MV-ViT-ft从第4升至第1

## Key Results / 核心结果

### Main Benchmark (Table 1) / 主基准测试（表1）

| Method / 方法 | Balanced mIoU | Natural mIoU | Gap | Drop% |
|--------|:---:|:---:|:---:|:---:|
| DGCNN | **0.955** | 0.690 | 0.265 | 27.8% |
| PointNet++ | 0.948 | 0.566 | 0.382 | 40.3% |
| RF | 0.910 | 0.548 | 0.362 | 39.8% |
| MV-ViT-ft | 0.908 | **0.743** | **0.165** | **18.2%** |
| CurveNet | 0.903 | 0.624 | 0.279 | 30.9% |
| DINOv2-MV | 0.876 | 0.657 | 0.219 | 25.0% |
| PointNet | 0.843 | 0.661 | 0.182 | 21.6% |
| PT | 0.620 | 0.571 | 0.049 | 7.9% |
| PointMLP | 0.476 | 0.412 | 0.065 | 13.6% |

### Mitigation Strategies / 缓解策略

| Strategy / 策略 | Effect / 效果 | Verdict / 结论 |
|----------|:---:|---------|
| Protocol mixing / 协议混合 | DGCNN: +0.058 mIoU (p<0.001) | ✅ Effective / 有效 |
| BN adaptation / BN适配 | DGCNN: −0.293 | ❌ Worse / 更差 |
| PAFA (adversarial / 对抗) | −0.046 (p=0.001) | ❌ Significantly hurts / 显著降低 |
| Teeth3DS pre-training / 预训练 | −0.001 (p=0.679) | ❌ No effect / 无效果 |

---

## Repository Structure / 仓库结构

```
Dentist/
├── paper_draft.md          # Full manuscript (English) / 完整论文（英文）
├── paper_draft_zh.md       # Full manuscript (Chinese) / 完整论文（中文）
├── paper_tables/           # All experimental results / 所有实验结果 (JSON, 90+ files)
├── scripts/                # Training & analysis scripts / 训练与分析脚本 (60+)
│   ├── phase3_train_raw_seg.py      # Main segmentation training / 主分割训练 (9 architectures)
│   ├── train_pafa.py                # PAFA adversarial alignment / PAFA对抗对齐
│   ├── train_mix_ablation.py        # Mixing ratio ablation / 混合比例消融
│   ├── generate_paper_figures.py    # Figure generation / 图表生成
│   └── ...
├── configs/                # YAML experiment configurations / YAML实验配置
├── assets/                 # Figures / 图表 (7 main + 1 supplementary)
├── runs/                   # Experiment outputs / 实验输出 (gitignored, local only)
├── processed/              # Processed datasets / 处理后数据集 (gitignored, local only)
├── AUTO_REVIEW.md          # 12 rounds of automated review / 12轮自动审稿记录
├── REVIEW_STATE.json       # Current review status / 当前审稿状态
├── RESEARCH_PIPELINE_REPORT.md  # Full research log / 完整研究日志
├── IDEA_REPORT.md          # Initial idea exploration / 初始想法探索
└── TRAINING_INFRASTRUCTURE.md   # Model architecture docs / 模型架构文档
```

## Quick Navigation / 快速导航

| What / 内容 | Where / 位置 |
|------|-------|
| **Full paper (EN) / 完整论文（英）** | [`paper_draft.md`](paper_draft.md) |
| **Full paper (ZH) / 完整论文（中）** | [`paper_draft_zh.md`](paper_draft_zh.md) |
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

### 3) Reproduce All 715 Experiments / 复现全部715个实验

```bash
# Full benchmark: 9 methods x 2 protocols x 5 folds x 5 seeds = 450 runs
# 完整基准：9种方法 × 2种协议 × 5折 × 5种子 = 450次运行
bash scripts/run_expanded_seg_experiments.sh

# Mixing ratio ablation: 5 ratios x 3 seeds x 5 folds = 75 runs
# 混合比例消融：5种比例 × 3种子 × 5折 = 75次运行
python3 scripts/run_mix_ablation_launcher.py

# PAFA experiments / PAFA实验 (2 modes x 5 folds x 3 seeds = 30 runs)
python3 scripts/run_pafa_experiments.py
```

---

## Paper / 论文

The full manuscript is available in two languages / 完整论文提供两种语言版本:

- 📄 **English**: [`paper_draft.md`](paper_draft.md) (~10,000 words, 10 tables + 5 supplementary, 7 figures, 30 references)
- 📄 **中文**: [`paper_draft_zh.md`](paper_draft_zh.md)（约15,000字，10个正文表格 + 5个补充表格，7张图，30篇参考文献）

### Abstract / 摘要

Automated segmentation of dental restorations from intraoral 3D scans is essential for digital dentistry workflows. This study investigates how scanning protocol—balanced vs. natural—affects segmentation method ranking, reliability, and feature-space representations. Across 715 experimental configurations spanning 9 methods and 7 paradigms, we find that scanning protocol choice induces dramatic ranking reversals (PointNet++ 2nd→7th, MV-ViT-ft 4th→1st). Protocol mixing is the only effective mitigation strategy, while three model-centric approaches (BN adaptation, adversarial alignment, external pre-training) all fail.

口内3D扫描中牙科修复体的自动分割对数字化牙科工作流至关重要。本研究探讨扫描协议（标准化 vs. 临床常规）如何影响分割方法排名、可靠性和特征空间表征。在涵盖9种方法和7种范式的715个实验配置中，我们发现扫描协议的选择引发了显著的方法排名反转（PointNet++从第2降至第7，MV-ViT-ft从第4升至第1）。协议混合训练是唯一有效的缓解策略，而三种模型策略（BN适配、对抗特征对齐、外部预训练）均失败。

### Tables / 表格

| Table / 表 | Content / 内容 | Data / 数据 |
|-------|---------|------|
| Table 1 | Main benchmark (9 methods × 2 protocols) / 主基准（9方法×2协议） | `paper_tables/full_benchmark_n25_all.json` |
| Table 2 | Restoration-class IoU / 修复体类IoU | `paper_tables/full_benchmark_n25_all.json` |
| Table 3 | Cross-protocol transfer / 跨协议迁移 | `paper_tables/cross_protocol_transfer.json` |
| Table 4 | Protocol mixing / 协议混合 | `paper_tables/protocol_mixing_summary.json` |
| Table 5 | BN adaptation / BN适配 | `paper_tables/bn_adaptation_results.json` |
| Table 6 | Teeth3DS pre-training / Teeth3DS预训练 | `paper_tables/teeth3ds_pretraining_results.json` |
| Table 7 | Cross-dataset domain gap / 跨数据集域差距 | `paper_tables/cross_dataset_domain_gap.json` |
| Table 8 | PAFA adversarial alignment / PAFA对抗对齐 | `paper_tables/pafa_aligned_results.json` |
| Table 9 | Mixing ratio ablation / 混合比例消融 | `paper_tables/mix_ablation_results.json` |
| Table S1-S5 | Supplementary (Dice, pairwise stats, methods, per-type, costs) / 补充材料 | `paper_tables/` |

### Figures / 图表

| Figure / 图 | Description / 描述 | File / 文件 |
|--------|-------------|------|
| Figure 1 | Qualitative segmentation examples / 分割定性示例 | `assets/figure1_protocol_comparison.png` |
| Figure 2 | Box plots (8 DL methods) / 箱线图（8种DL方法） | `assets/figure2_boxplot_comparison.png` |
| Figure 3 | UMAP protocol gap / UMAP协议差距可视化 | `assets/umap_protocol_gap.png` |
| Figure 4 | Ranking bump chart / 排名变化凸凹图 | `assets/figure_bump_chart.png` |
| Figure 5 | PAFA scatter plot / PAFA散点图 | `assets/figure_pafa_scatter.png` |
| Figure 6 | Mixing ablation curve / 混合消融曲线 | `assets/figure_mix_ablation.png` |
| Figure 7 | 3D segmentation rendering / 3D分割渲染 | `assets/figure_3d_segmentation.png` |

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

| Method / 方法 | Paradigm / 范式 | Params | Reference / 参考 |
|--------|------|:---:|-----------|
| **RF** | Classical ML / 经典机器学习 | — | Breiman, 2001 |
| **PointNet** | Point-based / 基于点 | 372K | Qi et al., CVPR 2017 |
| **PointNet++** | Hierarchical / 分层 | 1.40M | Qi et al., NeurIPS 2017 |
| **DGCNN** | Graph-based / 基于图 | 651K | Wang et al., 2019 |
| **CurveNet** | Curve-based / 基于曲线 | 140K | Xiang et al., ICCV 2021 |
| **PointMLP** | MLP-based / 基于MLP | 187K | Ma et al., ICLR 2022 |
| **Point Transformer** | Transformer | 388K | Zhao et al., ICCV 2021 |
| **DINOv2-MV** | Multi-view frozen / 多视角冻结 | 21.7M (132K trainable) | Oquab et al., 2024 |
| **MV-ViT-ft** | Multi-view fine-tuned / 多视角微调 | 21.7M (7.2M trainable) | Dosovitskiy et al., 2021 |

---

## Review History / 审稿历史

The paper went through 12 automated review rounds using the `aris-reviewer` agent (Claude Opus 4.6):

论文经过12轮 `aris-reviewer` 自动审稿（Claude Opus 4.6）：

| Round / 轮次 | Score / 分数 | Verdict / 判定 | Key Changes / 主要改动 |
|:---:|:---:|:---:|------------|
| 1 | 7.0 | NOT READY | Initial benchmark / 初始基准 |
| 2 | 8.0 | ALMOST | Added feature-space analysis / 加入特征空间分析 |
| 3 | 7.5 | ALMOST | Fixed A-distance / 修正A-距离 |
| 4 | 8.5 | READY | Expanded to n=25 / 扩展至n=25 |
| 5 | 8.5 | READY | DGCNN bug fix / DGCNN修复 |
| 6 | 8.0 | ALMOST | Teeth3DS integration / Teeth3DS整合 |
| 7 | 8.5 | READY | PAFA seed alignment / PAFA种子对齐 |
| 8 | 8.0 | ALMOST | PointMLP added / 加入PointMLP |
| 9 | 8.0 | ALMOST | Mixing ablation / 混合消融 |
| 10 | 8.5 | READY | All fixes / 全部修正 |
| 11 | 8.5 | READY | CurveNet + Table S2 expansion / CurveNet + 表S2扩展 |
| 12 | **9.0** | **READY** | Computational costs + final polish / 计算成本 + 最终打磨 |

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
