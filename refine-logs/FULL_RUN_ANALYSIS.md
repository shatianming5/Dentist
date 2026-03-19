# Full Run Analysis

**Scope**: `runs/research_segcls_full`
**Summary source**: `paper_tables/research_segcls_full_summary.md`
**Date**: 2026-03-17

## Aggregate Results

| Variant | Test Acc | Test Macro-F1 | Test Bal Acc | Test Seg mIoU |
|---|---:|---:|---:|---:|
| `all_pointnet` | `0.2239±0.0886` | `0.1413±0.0598` | `0.2096±0.0997` | - |
| `gtseg_pointnet` | `0.3092±0.1108` | `0.2803±0.0861` | `0.4102±0.1703` | - |
| `joint_pointnet` | `0.2411±0.1106` | `0.1899±0.1158` | `0.2496±0.1633` | `0.7603±0.0938` |
| `seg_pointnet` | `0.9060±0.0592` | - | - | `0.8323±0.0877` |

## Pairwise Evidence

- `gtseg_pointnet - all_pointnet`
  - macro-F1 delta: `+0.1390 [0.0826, 0.1933]`
  - balanced-accuracy delta: `+0.2006 [0.0931, 0.3065]`
  - macro-F1 wins/ties/losses across 15 aligned fold-seed runs: `14 / 0 / 1`
- `joint_pointnet - all_pointnet`
  - macro-F1 delta: `+0.0486 [-0.0151, 0.1269]`
  - balanced-accuracy delta: `+0.0399 [-0.0384, 0.1197]`
  - macro-F1 wins/ties/losses: `8 / 0 / 7`
- `joint_pointnet - gtseg_pointnet`
  - macro-F1 delta: `-0.0904 [-0.1661, -0.0019]`
  - balanced-accuracy delta: `-0.1606 [-0.2664, -0.0700]`
  - macro-F1 wins/ties/losses: `2 / 0 / 13`

## Fold-Level Pattern

- `all_pointnet` per-fold macro-F1 means:
  - fold0 `0.1687`
  - fold1 `0.1270`
  - fold2 `0.1365`
  - fold3 `0.0853`
  - fold4 `0.1889`
- `gtseg_pointnet` per-fold macro-F1 means:
  - fold0 `0.2402`
  - fold1 `0.3044`
  - fold2 `0.2591`
  - fold3 `0.2893`
  - fold4 `0.3086`
- `joint_pointnet` per-fold macro-F1 means:
  - fold0 `0.1391`
  - fold1 `0.1487`
  - fold2 `0.2329`
  - fold3 `0.2687`
  - fold4 `0.1602`

Interpretation:

- Joint gains are concentrated in fold2 and fold3.
- Oracle localization is consistently strong across all folds.
- The current joint method is not uniformly recovering the localization benefit.

## Failure Mode Diagnosis

- Joint classification quality is only weakly correlated with joint segmentation quality.
  - Pearson correlation between joint test seg mIoU and joint test macro-F1: `0.21`
- Counterexamples show that better segmentation does not reliably translate to better classification:
  - `joint_pointnet_fold2_seed1337`: seg mIoU `0.8576`, macro-F1 `0.0000`
  - `joint_pointnet_fold0_seed2020`: seg mIoU `0.8160`, macro-F1 `0.1389`
  - `joint_pointnet_fold4_seed2020`: seg mIoU `0.8309`, macro-F1 `0.1465`

Interpretation:

- The main bottleneck is no longer "can the model segment the restoration region?"
- The bottleneck is "can the classifier use the mask signal stably across folds?"
- This points to train/inference mask mismatch and pooling design as the next critical targets.

## Best Runs

- `all_pointnet`
  - `all_pointnet_fold0_seed2020`: macro-F1 `0.2566`, acc `0.3750`
- `gtseg_pointnet`
  - `gtseg_pointnet_fold1_seed2021`: macro-F1 `0.4333`, acc `0.4375`
- `joint_pointnet`
  - `joint_pointnet_fold2_seed2021`: macro-F1 `0.5251`, acc `0.3125`, seg mIoU `0.8598`
- `seg_pointnet`
  - `seg_pointnet_fold2_seed1337`: mIoU `0.8999`, acc `0.9474`

## Next Iteration Plan

1. Prioritize classifier-mask coupling ablations, not stronger segmentation first.
   - compare `cls_train_mask=gt` vs `pred` vs scheduled mixture
2. Ablate restoration pooling shape.
   - current top-k pooling
   - top-k + global max only
   - soft/hard hybrid pooling
3. Keep probe cost low at first.
   - run fold2 and fold3 first
   - only expand to full `5 x 3` when the paired delta over `all_pointnet` becomes stable
4. Promote a new joint variant only if:
   - paired macro-F1 delta over `all_pointnet` has CI fully above `0`
   - gap to `gtseg_pointnet` is materially reduced

## Focused Probe Results

Targeted probes were run on the two strongest full-run units:

- baseline A: `joint_pointnet_fold2_seed2021`
  - full-run test macro-F1 `0.5251`, acc `0.3125`, seg mIoU `0.8598`
- baseline B: `joint_pointnet_fold3_seed1337`
  - full-run test macro-F1 `0.3393`, acc `0.3750`, seg mIoU `0.8152`

Probed variants:

- `pred_05`
  - fold2/seed2021: macro-F1 `0.2333`
  - fold3/seed1337: macro-F1 `0.1507`
- `gt_035`
  - fold2/seed2021: macro-F1 `0.1846`
  - fold3/seed1337: macro-F1 `0.1750`
- `gt_065`
  - fold2/seed2021: macro-F1 `0.1261`
  - fold3/seed1337: macro-F1 `0.2667`
- `mix_05`
  - fold2/seed2021: macro-F1 `0.2407`
  - fold3/seed1337: macro-F1 `0.0625`

Interpretation:

- None of the simple knob changes beat the current full-run baseline on either probe unit.
- `pred_05` and `mix_05` partially improved validation behavior but did not translate into stronger test classification.
- `gt_065` severely damaged one fold by collapsing segmentation quality.
- `gt_035` preserved segmentation, but classification stayed weak.

Decision:

- Do not promote `pred_05`, `mix_05`, `gt_035`, or `gt_065` into the next all-fold run.
- The next change must be architectural, not just a mask/training schedule knob.
- The two most promising architectural directions are:
  - richer restoration pooling features
  - better checkpoint selection / calibration for tiny validation folds

## Two-Stage and Architectural Follow-Ups

Additional probes were run after the mask/top-k sweep.

- Two-stage `pred_topk` classifier with PointNet segmentation teacher:
  - fold2/seed2021: test macro-F1 `0.0833`
  - fold3/seed1337: test macro-F1 `0.2527`
- `factorized` joint pooling with the default selection rule:
  - fold2/seed2021: test macro-F1 `0.0000`, test seg mIoU `0.8548`
  - fold3/seed1337: test macro-F1 `0.1767`, test seg mIoU `0.2500`
- `factorized` joint pooling with coupled selection
  - selection score: `val_macro_f1_present + 0.5 * val_seg_mIoU`
  - fold3/seed1337: test macro-F1 `0.3333`, test seg mIoU `0.7623`
- original `topk` joint with the same coupled selection:
  - fold3/seed1337: test macro-F1 `0.2500`, test seg mIoU `0.7372`
  - current full-run baseline on the same unit remains stronger: test macro-F1 `0.3393`, test seg mIoU `0.8152`
- localization-consistent joint training (`pred` main branch + GT auxiliary branch + consistency loss):
  - fold2/seed2021: test macro-F1 `0.2319`, test seg mIoU `0.7590`
  - fold3/seed1337: test macro-F1 `0.2458`, test seg mIoU `0.6177`
  - compared with plain `pred` training:
    - fold2/seed2021: `0.2319` vs `0.2333`
    - fold3/seed1337: `0.2458` vs `0.1507`

Interpretation:

- Hard `pred_topk` crops remain too unstable to replace the joint path.
- The `factorized` head is not robust enough to promote, especially because fold2 still collapsed on classification.
- Checkpoint selection does matter:
  - the plain factorized fold3 run selected a checkpoint with `val seg mIoU = 0.25`
  - coupled selection avoided that degenerate checkpoint and nearly recovered the baseline classification score
- But coupled selection is not a universal win:
  - applying the same rule to the original `topk` joint probe did not beat the existing baseline
- Localization-consistent training is a more credible next paper method than plain `pred` training:
  - it clearly improves the weaker fold3 predicted-mask result,
  - but it still does not catch the current best `joint_pointnet` baseline
- A compact locc sweep confirmed that the method is tunable but not yet promotable:
  - fold2 best: `aux050_cons010`, test macro-F1 `0.3399`, seg mIoU `0.8901`
  - fold3 best: `aux025_cons025`, test macro-F1 `0.2603`, seg mIoU `0.7951`
  - no single locc configuration was best on both folds
  - coupled selection on `aux050_cons010` was strongly negative on fold3 (`0.0500`)
- Feature-consistency regularization on top of locc was also probed:
  - best fold2 run: `aux050_cons010_feat010`, test macro-F1 `0.2222`, test seg mIoU `0.8736`
  - best fold3 run: `aux050_cons010_feat010`, test macro-F1 `0.2214`, test seg mIoU `0.7291`
  - interpretation: feature consistency produced a more balanced configuration across fold2 and fold3, but it over-regularized the classifier and stayed below the earlier locc best on both folds
- `hybrid` pooling under locc was also negative:
  - best fold2 run: `aux025_cons025`, test macro-F1 `0.1852`, test seg mIoU `0.8571`
  - best fold3 run: `aux050_cons010`, test macro-F1 `0.1458`, test seg mIoU `0.7747`
  - interpretation: richer hybrid pooling did not beat the simpler top-k locc path and remains non-promotable
- Teeth3DS single-tooth FDI pretraining was also tested as a direct PointNet backbone initializer for the locc line:
  - pretrain checkpoint: `runs/pretrain/teeth3ds_fdi_pointnet_seed1337/ckpt_best.pt`
  - pretrain quality: val acc `0.6981`, test acc `0.7476`, test macro-F1 `0.6807`
  - fold2 best-config transfer (`aux050_cons010`): test macro-F1 `0.1481`, test seg mIoU `0.8093`
  - fold3 best-config transfer (`aux025_cons025`): test macro-F1 `0.2333`, test seg mIoU `0.7342`
  - compared with the corresponding no-init locc baselines:
    - fold2: `0.1481` vs `0.3399`
    - fold3: `0.2333` vs `0.2603`
  - interpretation:
    - naive single-tooth FDI feature transfer does not help the current restoration-focused joint task
    - fold3 validation improved (`0.3328` vs `0.2455`) but the test metric still regressed, so this is not a promotion candidate
    - Teeth3DS remains plausible as an upstream geometry resource, but not via plain PointNet `init_feat` alone
- A more task-aligned Teeth3DS transfer was then probed via frozen local-geometry distillation:
  - teacher signal: frozen Teeth3DS PointNet feature extractor on a GT-restoration-centroid local crop (`1024` nearest points, bbox-diag normalize + PCA align)
  - fold2 / `aux050_cons010`:
    - `t3d010`: test macro-F1 `0.1176`, test seg mIoU `0.8864`
    - `t3d025`: test macro-F1 `0.1404`, test seg mIoU `0.8736`
    - old no-teacher baseline: `0.3399`, test seg mIoU `0.8901`
  - fold3 / `aux025_cons025`:
    - `t3d010`: test macro-F1 `0.1987`, test seg mIoU `0.5655`
    - `t3d025`: test macro-F1 `0.1987`, test seg mIoU `0.5908`
    - old no-teacher baseline: `0.2603`, test seg mIoU `0.7951`
  - interpretation:
    - even when the teacher sees a tooth-local neighborhood instead of the raw restoration mask, the transfer is still negative on both probe folds
    - fold2 validation briefly improved to `0.4214`, but this did not survive to test
    - fold3 suffered a large segmentation regression, so the teacher signal is still misaligned with the restoration objective

Decision:

- Keep `--pooling-mode factorized` as an architectural probe path only.
- Keep `--selection-seg-weight` as an opt-in guardrail for future unstable variants.
- Promote localization-consistent training as the next primary probe family.
- Do not promote feature-consistency regularization into the next all-fold stage yet.
- Do not promote hybrid pooling into the next all-fold stage yet.
- Do not promote plain Teeth3DS FDI `init_feat` transfer into the next all-fold stage yet.
- Do not promote frozen Teeth3DS local-geometry distillation into the next all-fold stage yet.
- Do not promote a locc configuration to all-fold yet; the method still needs a unified setting that survives both fold2 and fold3.
- Do not promote `pred_topk`, `factorized`, or selection-only changes into the next all-fold run yet.
- Calibration-aware checkpoint selection and stronger raw segmentation teachers were then executed explicitly as the next two probe families:
  - calibration-aware locc selection:
    - new trainer support:
      - selection score can now subtract a calibration penalty: `val_macro_f1_present + w_seg * val_seg_mIoU - w_cal * val_{ece|nll|brier}`
      - final runs now also fit validation temperature scaling and save calibrated metrics/predictions
    - probe results on the existing tuned locc units:
      - fold2 `aux050_cons010`:
        - all tested selection variants (`ece010`, `ece020`, `seg025_ece010`) selected the same best epoch `29`
        - test macro-F1 stayed `0.3399`, unchanged from the old baseline
        - test ECE improved only after post-hoc temperature scaling: `0.3145 -> 0.2632` with `T=1.1532`
      - fold3 `aux025_cons025`:
        - all tested selection variants selected the same best epoch `16`
        - test macro-F1 stayed `0.2603`, unchanged from the old baseline
        - post-hoc temperature scaling strongly improved calibration: `0.3342 -> 0.1015` with `T=10.0`
    - interpretation:
      - calibration-aware selection, as tested here, did not change checkpoint choice on either probe fold
      - the implementation is still useful because it now exposes calibrated outputs directly from the joint trainer
      - temperature scaling gives a real calibration gain, but not a classification gain
  - stronger raw segmentation teacher:
    - trained reusable DGCNNv2 teachers with checkpoints:
      - fold2/seed2021: test mIoU `0.9823`, acc `0.9911`
      - fold3/seed1337: test mIoU `0.9836`, acc `0.9917`
    - two-stage `pred_topk` classification with DGCNN teacher:
      - fold2: test macro-F1 `0.1053` vs old PointNet-teacher `0.0833`
      - fold3: test macro-F1 `0.1762` vs old PointNet-teacher `0.2527`
      - interpretation:
        - stronger segmentation alone does not make hard crop classification reliable
        - this remains non-promotable as a standalone replacement for the joint path
    - stronger teacher directly distilled into the locc joint model:
      - first implementation attempt used an incorrectly scaled KL term and was discarded
      - corrected `v2` runs used mean per-point KL distillation from frozen DGCNNv2 teacher logits
      - fold2:
        - `tseg010_v2`: test macro-F1 `0.0833`, seg mIoU `0.6276`
        - `tseg025_v2`: test macro-F1 `0.2611`, seg mIoU `0.8317`
        - old locc baseline remained better: `0.3399`, seg mIoU `0.8901`
      - fold3:
        - `tseg010_v2`: test macro-F1 `0.2701`, seg mIoU `0.7863`, ECE `0.1668`
        - `tseg025_v2`: test macro-F1 `0.1214`, seg mIoU `0.5032`
        - old locc baseline: test macro-F1 `0.2603`, seg mIoU `0.7951`, ECE `0.3342`
      - interpretation:
        - mild seg-teacher distillation (`0.10`) is the first stronger-teacher variant that shows a small positive signal on fold3 while also sharply improving confidence calibration
        - the same idea is negative on fold2, so it is not yet a promotable unified method
        - stronger raw segmentation evidence helps only when the transfer path is aligned and softly weighted; hard crops and heavier distillation remain unstable

## Current Decision

- The research hypothesis is supported.
- The paper story should currently claim:
  - localization matters
  - oracle localization helps a lot
  - current joint model recovers part of that gain, but not robustly enough yet
- The next engineering effort should be framed as checkpoint robustness / calibration and stronger evidence transfer, not as a search for stronger raw segmentation accuracy or richer pooling alone.
- More specifically:
  - calibration is now worth keeping as an evaluation / deployment component
  - stronger raw segmentation teachers should only be continued through carefully weighted joint evidence transfer, not through hard `pred_topk` crops alone
