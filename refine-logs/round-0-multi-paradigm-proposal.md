# Multi-Paradigm Dental Restoration Assessment: Proposal

## Problem Statement

3D intraoral scanning is becoming standard in restorative dentistry, but automated 
assessment of existing restorations on raw point clouds remains unsolved. Two key 
challenges exist:

1. **Segmentation**: Distinguishing restoration surfaces from natural tooth
2. **Classification**: Identifying the type of restoration (filling, full crown, 
   post-core crown, onlay)

These tasks are complicated by a **protocol gap**: laboratory-balanced datasets 
(50:50 class ratio) dramatically overestimate real-world performance on natural-
distribution scans (~19% restoration).

## Method Summary

We benchmark three paradigm families on 79 clinical intraoral scans:

| Paradigm | Method | Approach |
|----------|--------|----------|
| Point Cloud (shallow) | PointNet | Per-point MLP, no local context |
| Point Cloud (deep) | DGCNN | Dynamic graph conv, k-NN edges |
| Vision Foundation Model | DINOv3 | Multi-view rendering → ViT-S/16 → back-project |

Each method is evaluated under both balanced (50:50) and natural (~19:81) protocols.
For classification, we additionally test DINOv3's dense features with segmentation-
guided pooling.

## Current Evidence (15 runs per cell, 5-fold × 3 seeds)

### Segmentation (fold-level mIoU)

| Method   | Balanced     | Natural      | Gap   | Rel.Drop |
|----------|-------------|--------------|-------|----------|
| PointNet | 0.843±0.039 | 0.662±0.030  | 0.181 | 21.5%    |
| DGCNN    | 0.956±0.044 | 0.690±0.038  | 0.266 | 27.8%    |
| DINOv3   | 0.876±0.048 | 0.655±0.043  | 0.220 | 25.1%    |

Natural Protocol Per-Class IoU:
- PointNet: bg=0.828, res=0.496
- DGCNN: bg=0.851, res=0.529
- DINOv3: bg=0.842, res=0.469

Statistical: DGCNN > DINOv3 on natural (p=0.03), DGCNN > PointNet (p=0.04)

### Classification (4-class, balanced protocol)

| Method | F1 | Approach |
|--------|------|---------|
| Geometric features (from seg) | 0.279 | Hand-crafted features from point cloud |
| DINOv3 CLS token + MLP | 0.359 | Global image features |
| DINOv3 seg-guided + pooling | 0.533 | Dense features + seg mask + mean/max/std |
| DINOv3 binary (filling/indirect) | 0.604 | Clinically actionable binary |

### Negative Results (11 experiments)
- All transfer learning approaches: NS (PointNet, DGCNN, domain adaptation)
- Multi-task learning: hurts both tasks
- Boundary geometry: no type-discriminative features

## Proposed Paper Story

**Title**: "Multi-Paradigm Assessment of 3D Dental Restorations: Point Cloud Networks 
vs. Vision Foundation Models under Protocol Shift"

**Central Claim**: The choice of analysis paradigm should be task-dependent in dental 
restoration assessment. For segmentation, end-to-end trained point cloud networks 
(DGCNN) outperform vision foundation models. For restoration type classification, 
DINOv3's dense visual features with segmentation-guided pooling achieve F1=0.533, 
nearly doubling the best point-cloud approach (F1=0.279).

**Key Contributions**:
1. First multi-paradigm benchmark for dental restoration segmentation across two 
   clinically distinct protocols (balanced vs. natural)
2. Demonstration that protocol gap (18-29%) is universal across paradigm families, 
   establishing it as the central barrier to clinical deployment
3. DINOv3 seg-guided classification pipeline achieving F1=0.533, the first 
   meaningful automated restoration typing from 3D scans
4. Comprehensive negative results on transfer learning, multi-task learning, and 
   boundary geometry

**Target Journal**: Journal of Dental Research (Short Communication or Full Paper)

## Known Weaknesses
1. DINOv3 uses frozen backbone — fine-tuning could improve results
2. Classification tested only on balanced protocol
3. Small dataset (n=79 cases)
4. No external validation cohort
5. DINOv3 adds rendering complexity (6 views × 512²) vs direct point cloud processing

## Open Questions for Reviewer
1. Is the multi-paradigm angle strong enough for a standalone paper?
2. Should we fine-tune DINOv3 backbone before submitting?
3. Is the classification improvement (0.279 → 0.533) clinically meaningful enough?
4. Would this be better as a short communication or full paper?
