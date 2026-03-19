# Experiment Plan

## Goal

Validate whether segmentation-aware pooling improves restoration-type classification from full-case dental point clouds.

## Datasets and Protocol

- Primary data: `processed/raw_seg/v1`
- Splits for final claim: `metadata/splits_raw_case_kfold.json`
- Seeds: `1337, 2020, 2021`
- Input size:
  - joint model: `8192` points
  - two-stage hard crop baseline: keep `4096` points

## Metrics

### Classification

- accuracy
- macro-F1
- balanced accuracy
- ECE / NLL / Brier

### Segmentation

- mean IoU
- per-class IoU
- per-class F1

## Baselines

1. All-points classifier on `raw_seg`-derived classification data.
2. Oracle `gt_seg` crop classifier.
3. Hard `pred_topk` crop classifier.
4. Existing repo `raw_cls v13/v18` results as contextual reference only, not paired comparison.

## Proposed Variants

1. Joint model, no segmentation loss.
2. Joint model + segmentation loss.
3. Joint model + restoration-only soft pool.
4. Joint model + restoration/context/global pools.
5. Optional teacher variants:
   - PointNet segmentation teacher
   - DGCNNv2 segmentation teacher
   - DINOv3 segmentation teacher

## Ablations

1. Pooling:
   - hard top-k
   - soft restoration-only
   - soft restoration + context
2. Point budget:
   - 2048
   - 4096
   - 8192
3. Loss weight `lambda_seg`:
   - 0.1
   - 0.5
   - 1.0
4. Teacher strength:
   - PointNetSeg
   - DGCNNv2Seg
   - DINOv3 MLP

## Run Order

1. Smoke:
   - fold 0, seed 1337, 10 epochs
   - verify metrics and artifact layout
2. Pilot:
   - fold 0, seed 1337, full patience
   - compare all-points vs hard crop vs joint model
3. Medium confidence:
   - folds 0..4, seed 1337
4. Final claim:
   - folds 0..4, seeds 1337/2020/2021
   - paired bootstrap CI on macro-F1 / balanced accuracy

## Compute Estimate

- Two-stage builder: minutes or less.
- Segmentation training on this dataset: minutes on a single 4090-class GPU.
- Classification training on this dataset: minutes on a single GPU.
- Full k-fold x multi-seed is affordable and should be treated as the default for any claim.

## Evidence to Save

- `metrics.json`
- `config.yaml` or config JSON
- confusion matrices
- calibration outputs
- best checkpoints
- one paper-ready markdown summary table

## Post-Full Findings

- `all_pointnet`: weak and unstable
- `gtseg_pointnet`: robustly better than `all_pointnet`
- `joint_pointnet`: better mean than `all_pointnet`, but not robust enough and still below `gtseg_pointnet`
- pairwise macro-F1 deltas:
  - `gtseg - all`: `+0.1390 [0.0826, 0.1933]`
  - `joint - all`: `+0.0486 [-0.0151, 0.1269]`
  - `joint - gtseg`: `-0.0904 [-0.1661, -0.0019]`
- failure mode:
  - joint seg mIoU is not strongly predictive of joint cls macro-F1
  - next work should focus on classifier-mask coupling rather than stronger segmentation alone

## Next Iteration Run Order

1. Focused probe on fold2 and fold3 only:
   - `cls_train_mask = gt`
   - `cls_train_mask = pred`
   - top-k ratio `0.35 / 0.50 / 0.65`
2. Probe result on 2026-03-17:
   - simple mask / top-k knob changes did not beat the current joint baseline
   - do not promote these settings to all folds
3. Next probe should be architectural:
   - richer restoration pooling features
   - improved small-val checkpoint selection / calibration
4. Additional probe result on 2026-03-17:
   - two-stage `pred_topk` crop classification was unstable and not promotable
   - `factorized` pooling under the default selection rule also failed
   - coupled selection `val_macro_f1 + 0.5 * val_seg_mIoU` can rescue a degenerate variant, but did not give a general win on the original top-k joint model
5. New primary method line:
   - localization-consistent joint training: predicted-mask deployment branch + GT-mask auxiliary branch + consistency loss
   - first probe improved over plain `pred` training on fold3, but still stayed below the current best joint baseline
6. Focused locc sweep result on 2026-03-17:
   - fold2 best: `aux050_cons010`
   - fold3 best: `aux025_cons025`
   - there is still no single setting that wins on both probe folds
   - coupled selection on the promising `aux050_cons010` fold3 run was negative
7. Additional focused probes on 2026-03-17:
   - feature-consistency regularization produced a more balanced but weaker unified setting
   - best feature-consistency config: `aux050_cons010_feat010`
   - fold2: macro-F1 `0.2222`, seg mIoU `0.8736`
   - fold3: macro-F1 `0.2214`, seg mIoU `0.7291`
   - `hybrid + locc` did not beat top-k locc on either fold
   - best hybrid fold2: `0.1852`; best hybrid fold3: `0.1458`
8. Teeth3DS transfer probe on 2026-03-18:
   - built `processed/teeth3ds_teeth/v1` and trained `runs/pretrain/teeth3ds_fdi_pointnet_seed1337/ckpt_best.pt`
   - direct `init_feat` transfer into the locc joint trainer was negative on both tuned probe folds
   - fold2 `aux050_cons010`: `0.1481` vs old `0.3399`
   - fold3 `aux025_cons025`: `0.2333` vs old `0.2603`
   - do not promote plain Teeth3DS FDI initialization to all folds
9. Teeth3DS local-geometry distillation probe on 2026-03-18:
   - teacher: frozen Teeth3DS PointNet feature extractor on GT-restoration-centroid local crops
   - fold2 `aux050_cons010`: `t3d010=0.1176`, `t3d025=0.1404`, both below old `0.3399`
   - fold3 `aux025_cons025`: `t3d010=0.1987`, `t3d025=0.1987`, both below old `0.2603`
   - do not promote frozen Teeth3DS local distillation to all folds
10. If any focused probe shows stable gain over current joint baseline:
   - rerun that variant on all folds with seed `1337`
   - require the gain to hold on both fold2 and fold3 under the same selection rule
11. Only then:
   - expand to `5 folds x 3 seeds`
   - recompute paired CI against `all_pointnet` and `gtseg_pointnet`
12. Calibration / stronger-teacher follow-up on 2026-03-18:
   - implemented calibration-aware checkpoint selection in the joint trainer
   - implemented automatic validation-fit temperature scaling outputs (`calib.json`, calibrated preds, calibrated metrics)
   - result on tuned locc probe folds:
     - checkpoint choice did not change under tested ECE-weighted selection settings
     - test macro-F1 stayed unchanged on both fold2 and fold3
     - test ECE improved after temperature scaling:
       - fold2: `0.3145 -> 0.2632`
       - fold3: `0.3342 -> 0.1015`
   - interpretation:
     - keep calibration as an evaluation / confidence improvement component
     - do not promote calibration-aware selection itself as a new all-fold method yet
13. Stronger raw segmentation teacher follow-up on 2026-03-18:
   - trained reusable DGCNNv2 raw segmentation teachers with checkpoints
     - fold2/seed2021 test mIoU `0.9823`
     - fold3/seed1337 test mIoU `0.9836`
   - two-stage DGCNN-teacher `pred_topk` classification:
     - fold2 improved slightly: `0.0833 -> 0.1053`
     - fold3 regressed: `0.2527 -> 0.1762`
     - decision: do not promote hard stronger-teacher crops
   - direct joint seg-teacher distillation was implemented and probed after fixing the KL scaling bug
     - fold2:
       - `tseg010_v2 = 0.0833`
       - `tseg025_v2 = 0.2611`
       - both below old locc baseline `0.3399`
     - fold3:
       - `tseg010_v2 = 0.2701`
       - `tseg025_v2 = 0.1214`
       - mild distillation slightly beat old locc baseline `0.2603`
     - decision:
       - stronger teacher transfer is partially promising only in the light-weight joint-distillation regime
       - still not promotable to all folds because the fold2/fold3 behavior is not unified
