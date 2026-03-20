# Idea Discovery Report

**Direction**: 分类和分割 for dental point clouds, narrowed to restoration-type classification with point-level restoration segmentation.
**Date**: 2026-03-17
**Pipeline**: local repo audit -> literature survey -> pilot implementation -> pilot experiments
**AUTO_PROCEED**: selected Idea 1

## Executive Summary

The repo already has a strong restoration classification line (`raw_cls`) and a separate point-level restoration segmentation line (`raw_seg`), but the two are not coupled in a learnable way. Recent dental 3D papers mostly optimize segmentation, numbering, or foundation-model transfer; I did not find a paper in this search set that directly tackles restoration-type classification from full-case 3D intraoral point clouds with segmentation-guided pooling. That is an inference from the surveyed sources below, not a proof of absence.

The strongest local pilot signal is positive: on the same `processed/raw_seg/v1` fixed split and the same PointNet classifier, using only restoration-region points improves test accuracy from `0.0625` to `0.3125`, and a deployable predicted-mask crop improves it to `0.1250`. This is enough to recommend a segmentation-guided classification direction, but not enough to claim paper-level performance yet.

## Literature Landscape

### What recent dental 3D work emphasizes

| Area | Representative paper | Date | Why it matters |
|---|---|---:|---|
| Dental 3D segmentation with geometry + image cues | CrossTooth, CVPR 2025, https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CrossTooth_Cross-modal_Vision_Transformer_for_3D_Dental_Model_Segmentation_with_CVPR_2025_paper.html | 2025 | Shows the field is pushing strong tooth segmentation with cross-modal priors. |
| Foundation-model transfer to dental segmentation | 3DTeethSAM, arXiv, https://arxiv.org/abs/2508.10049 | 2025 | Confirms SAM-style 2D priors are being adapted to 3D dental segmentation. |
| Foundation-model fusion for dental segmentation | SOFTooth, arXiv, https://arxiv.org/abs/2506.18871 | 2025 | Reinforces that local semantic transfer is currently focused on segmentation rather than restoration typing. |
| Joint detection/numbering on digital impressions | Automated tooth detection and numbering in digital impressions of children using artificial intelligence, https://pubmed.ncbi.nlm.nih.gov/39812496/ | 2025 | Closest dental "classification + localization" precedent, but target is tooth identity, not restoration type. |
| 3D tooth identification/classification | 3D tooth identification for forensic dentistry using artificial intelligence, https://pmc.ncbi.nlm.nih.gov/articles/PMC12339297/ | 2025 | Shows 3D classification exists, but on isolated teeth / identity tasks. |
| Restoration classification in dentistry | Automated classification of dental restorations on panoramic radiographs using deep learning, https://pubmed.ncbi.nlm.nih.gov/34856747/ | 2021 | Restoration-type classification exists in 2D radiographs, which suggests the target is clinically meaningful, but not in 3D IOS point clouds. |

### Main gap

The surveyed dental 3D literature is concentrated on:

- tooth segmentation / numbering,
- cross-modal transfer from 2D foundation models to 3D geometry,
- tooth identity classification rather than restoration-type classification.

**Inference from the above**: there is room for a method that uses point-level restoration localization as a structural prior for restoration-type classification on full-case 3D scans.

## Local Evidence

### Existing repo evidence before new pilots

- `raw_cls` repo SOTA on `v18_main4_seg_all_cloudid_eq`: accuracy `0.6384`, macro-F1 `0.6122`, balanced-acc `0.6266`, ECE `0.1771`.
- `raw_seg` already exists and is strong:
  - historical PointNet + DGCNNv2 mean test mIoU across local runs: `0.8995`
  - DINOv3 segmentation mean test mIoU across local runs: `0.8756`
- DINOv3 does **not** transfer directly to restoration classification:
  - local `runs/raw_cls_dinov3/*`: mean test accuracy `0.3347`, macro-F1 `0.1867`

### New pilot implemented in this turn

Added dataset builder:

- `scripts/phase3_build_raw_cls_from_raw_seg.py`

Built datasets:

- `processed/raw_cls_from_raw_seg/v1_all`
- `processed/raw_cls_from_raw_seg/v1_gt_seg`
- `processed/raw_cls_from_raw_seg/v1_pred_topk_pointnetseg`

Ran pilots:

- segmentation teacher:
  - `runs/research_segcls_pilot/rawseg_pointnet_seg_s1337`
  - test mIoU `0.8596`
- classification on all points:
  - `runs/research_segcls_pilot/rawseg_all_pointnet_s1337`
- classification on oracle restoration crop:
  - `runs/research_segcls_pilot/rawseg_gtseg_pointnet_s1337`
- classification on predicted top-k restoration crop:
  - `runs/research_segcls_pilot/rawseg_predtopk_pointnet_s1337`

### Pilot table

| Pilot | Test acc | Test macro-F1 | Test bal-acc | Interpretation |
|---|---:|---:|---:|---|
| all-points PointNet | 0.0625 | 0.0294 | 0.2500 | Full-case raw input collapses under background clutter. |
| pred-topk PointNet | 0.1250 | 0.1625 | 0.3750 | Predicted restoration crop helps, but hard top-k is noisy. |
| gt-seg PointNet | 0.3125 | 0.2442 | 0.4653 | Oracle localization provides a clear positive signal. |

## Ranked Ideas

### 1. Restoration-Aware Soft Segmentation Pooling — RECOMMENDED

- **Core idea**: train one shared point encoder with a binary restoration segmentation head and a case-level classification head. Use predicted restoration probabilities for **soft pooling** into restoration/context/global descriptors instead of hard cropping.
- **Why it is best**:
  - supported by a strong oracle pilot,
  - deployable from full-case scans,
  - addresses the main failure mode of `pred_topk`: hard crop discards useful context and amplifies mask errors.
- **Local pilot signal**:
  - `all -> gt_seg` gives a large jump,
  - `all -> pred_topk` is already positive,
  - remaining gap suggests the final method should use soft weighting, not hard top-k.
- **Novelty position**:
  - dental 3D work already uses segmentation and tooth identity tasks,
  - restoration-type classification from full-case 3D scans with segmentation-aware pooling appears underexplored in this search set.
- **Next step**:
  - implement joint model, run k-fold x multi-seed on `processed/raw_seg/v1`, then compare against hard-crop and classifier-only baselines.

### 2. Segmentation-Feature Distillation from Foundation Models — BACKUP

- **Core idea**: use a strong segmentation teacher (DINOv3 or DGCNNv2) to supervise the classifier's local features or pooling weights.
- **Why it is interesting**:
  - local evidence says DINOv3 is good at segmentation but poor at direct classification,
  - this suggests the representation contains useful local cues but weak global task alignment.
- **Risk**:
  - more engineering and more moving parts than Idea 1,
  - not yet supported by a positive end-to-end classification pilot.

### 3. Uncertainty-Aware Crop / Curriculum — BACKUP

- **Core idea**: replace hard top-k with a mix of high-confidence restoration points, uncertain boundary points, and a small context budget.
- **Why it matters**:
  - current `pred_topk` underperforms the oracle,
  - the gap likely comes from over-aggressive cropping and calibration error.
- **Risk**:
  - may help incrementally, but on its own is more of an ablation than a full thesis.

## Eliminated Ideas

- **Direct full-case classifier without segmentation prior**:
  - negative pilot on `v1_all`.
- **Direct DINOv3 global classification**:
  - negative local evidence in `runs/raw_cls_dinov3/*`.

## Reviewer-Style Critique

- Current pilot uses the repository's fixed split, not k-fold x multi-seed.
- The new classification pilots are on `raw_seg`-derived data, so they are **not** directly comparable to the repo's `raw_cls v18` SOTA.
- `pred_topk` is only a two-stage proxy. It demonstrates feasibility, but not the final method.
- Test set is small (`n=16` on the fixed split), so effect sizes need k-fold and paired bootstrap before any paper claim.

## Recommendation

Proceed with **Idea 1**. The implementation target should be a joint segmentation-classification model with soft restoration-aware pooling, while keeping the two-stage crop builder as a baseline and debugging tool.

