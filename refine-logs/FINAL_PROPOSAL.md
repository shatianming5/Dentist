# Final Proposal

## Title

Restoration-Aware Soft Segmentation Pooling for Dental Point-Cloud Classification

## Problem Anchor

We want to classify restoration type from a **full-case** intraoral point cloud, but the restoration region occupies a small and variable fraction of the scan. A classifier trained on all points wastes capacity on irrelevant geometry and can collapse under background dominance.

## Method Thesis

Use point-level restoration segmentation as a **learned attention prior** for case-level restoration classification: the classifier should pool restoration points more strongly than non-restoration points, while still retaining context.

## Dominant Contribution

The contribution is not "another better segmenter" and not "another plain classifier." The contribution is a **bridge** from dense restoration localization to case-level restoration typing:

- shared encoder over the full point cloud,
- binary restoration head,
- soft restoration-aware pooling for classification,
- end-to-end joint optimization.

## Minimum Viable Model

### Inputs

- full-case point cloud from `processed/raw_seg/v1`
- case-level restoration label from `index.jsonl`
- binary point labels from `samples/*.npz`

### Architecture

1. Shared point encoder `f_i`.
2. Segmentation head predicts per-point restoration probability `p_i`.
3. Build three pooled descriptors:
   - restoration descriptor: weighted by `p_i`
   - context descriptor: weighted by `1 - p_i`
   - global descriptor: standard max/mean pool
4. Concatenate descriptors and predict restoration type.

### Loss

- `L = L_cls + lambda_seg * L_seg`
- `L_seg` starts as cross-entropy; Dice can be added if needed.

## Why This Is Better Than Hard Top-K

The local pilot already shows:

- hard predicted top-k helps over all-points,
- oracle restoration-only crop is much better,
- therefore localization is useful,
- but hard cropping is too brittle and throws away context.

Soft pooling is the simplest way to preserve the good part of the oracle while remaining deployable.

## What Is Out of Scope

- claiming superiority over the repo's `raw_cls v18` SOTA in this turn,
- claiming cross-dataset generalization,
- claiming submission readiness from fixed-split pilots.

## Acceptance Criteria

- beats the same-backbone all-points classifier on `processed/raw_seg/v1`,
- keeps segmentation quality near the standalone segmenter,
- remains stable across k-fold x multi-seed,
- produces auditable run artifacts under `runs/`.

## Immediate Implementation Plan

1. Add a new joint training script or extend the existing classification trainer with a joint mode.
2. Keep `scripts/phase3_build_raw_cls_from_raw_seg.py` as the two-stage baseline builder.
3. Run smoke on fold 0 / seed 1337.
4. If positive, expand to k-fold.

