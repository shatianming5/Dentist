# Publication-Oriented Method Proposal

## Title

Localization-Consistent Joint Restoration Analysis for Full-Case Dental Point Clouds

## Target Claim

From a full-case 3D intraoral point cloud, the model can:

- localize restoration-related regions,
- classify restoration type at the case level,
- provide evidence that is inspectable enough for a dental journal audience.

## Why This Version Is Better Than the Previous Proposal

The previous proposal correctly identified restoration-aware pooling as the core direction, but the full-run evidence now says the harder problem is not just which pooled descriptor to use. The failure mode is the mismatch between:

- training-time access to good localization signal,
- inference-time dependence on noisy predicted localization.

The paper-worthy method should explicitly address that mismatch.

## Method Thesis

Train the deployment branch on **predicted restoration localization**, but simultaneously supervise a teacher branch on **GT restoration localization** during training. Force the two classification views to stay aligned.

This yields a clinically interpretable and technically simple bridge from point-level segmentation to case-level restoration typing.

## Minimum Viable Method

1. Shared point encoder over the full scan.
2. Segmentation head predicts restoration probability per point.
3. Deployment classification branch pools with predicted restoration mask.
4. Teacher classification branch pools with GT restoration mask during training only.
5. Training objective:
   - case-level classification loss on deployment branch,
   - auxiliary classification loss on teacher branch,
   - consistency loss between deployment and teacher predictions,
   - segmentation loss.

## Why This Fits a Strong Dental Journal Better

- It is easy to explain in clinician terms:
  - "learn from where the restoration truly is,"
  - "deploy from where the model thinks it is,"
  - "penalize disagreement."
- It preserves the full-case workflow instead of assuming a perfect crop.
- It keeps the model simple enough that the paper can emphasize clinical utility and reliability rather than architecture novelty for its own sake.

## Current Evidence

- Plain `pred`-mask joint training was weak on the strongest probes.
- The new localization-consistent variant improved over plain `pred` training on both key units:
  - fold2/seed2021: test macro-F1 `0.2319` vs plain `pred` `0.2333` and much better test accuracy `0.5000`
  - fold3/seed1337: test macro-F1 `0.2458` vs plain `pred` `0.1507`
- However it still does not beat the current best `joint_pointnet` baseline:
  - fold2/seed2021 baseline: `0.5251`
  - fold3/seed1337 baseline: `0.3393`

## Interpretation

This method is a **real next paper candidate**, not because it already wins, but because:

- it directly matches the diagnosed bottleneck,
- it is simple and publishable,
- its first probe is directionally better than the plain predicted-mask alternative.

## Promotion Rule

Promote this variant to the next all-fold stage only if one refined configuration can:

1. beat plain `pred` training consistently on fold2 and fold3,
2. recover at least one fold to within a small margin of the current `joint_pointnet` baseline,
3. keep segmentation quality from collapsing.

## Immediate Next Experiments

1. Tune `aux_gt_cls_weight` and `consistency_weight`.
2. Add optional coupled checkpoint selection for this variant only.
3. Compare top-k pooling vs factorized pooling under the same localization-consistency training.
4. If one setting is positive on fold2 and fold3, rerun folds `0..4` with seed `1337`.
