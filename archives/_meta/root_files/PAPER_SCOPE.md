# Paper Scope (Journal-Ready) — What Is Implemented vs Proposed

This repo currently contains **working baselines + evaluation tooling** for (1) point-cloud classification on CCB2-extracted dental objects and (2) synthetic tooth completion (prep→target) with lightweight functional constraints. It also contains a longer research proposal (`plan_report.md`) which includes **ideas that are not implemented** in the current codebase.

The goal of this file is to make the **paper claims consistent with the repository**.

## Implemented in this repository (can be reproduced)

### A) raw/ (CCB2 → converted → classification)
- Conversion + labeling: `scripts/convert_ccb2_bin.py`, `scripts/label_converted_raw.py`
- Dataset build (case-level, configurable extraction heuristics): `scripts/phase1_build_raw_cls.py`
- Baseline training (PointNet + DGCNN), imbalance options, feature init, k-fold support:
  - `scripts/phase3_train_raw_cls_baseline.py`
  - Key options: `--model pointnet|dgcnn`, `--balanced-sampler`, `--label-smoothing`, `--init-feat`, `--kfold/--fold`
- Run analysis/summaries:
  - `scripts/phase3_analyze_raw_cls_run.py`, `scripts/summarize_raw_cls_runs.py`

### B) Teeth3DS (single-tooth dataset → AE / prep→target / constraints)
- Single-tooth dataset build (+ deterministic normalization/PCA options):
  - `scripts/phase2_build_teeth3ds_teeth.py`
- Tooth AE baseline (self-supervised shape prior):
  - `scripts/phase2_train_teeth3ds_ae.py`
- Synthetic prep→target baseline (z-cut or random plane cut):
  - `scripts/phase3_train_teeth3ds_prep2target.py`
- Constraints training + evaluation (margin + occlusion proxies, richer metrics):
  - `scripts/phase4_train_teeth3ds_prep2target_constraints.py`
  - `scripts/phase4_eval_teeth3ds_constraints_run.py`, `scripts/phase4_summarize_constraints_runs.py`

### C) Reproducibility utilities
- Split freezing & metadata snapshotting: `scripts/phase0_freeze.py`
- Environment logging inside training runs: `env.json` produced by training scripts
- Audit checklist: `scripts/journal_audit.py` (writes `JOURNAL_AUDIT.md`)

## Proposed / Not implemented (do NOT claim as completed)

These appear in `plan_report.md` as a research blueprint, but are not currently implemented end-to-end in code:
- **VQ-VAE** with discrete codebook on 3D SDF/voxels for tooth morphology
- **Latent diffusion / DiT** for restoration generation in latent space
- **Differentiable SDF-based occlusion loss** (signed penetration, watertight meshes, differentiable marching cubes)
- Deep semantic parsing of proprietary CAD intent from CCB2 (insertion axis, cement gap, margin line extraction) beyond current point exports
- Clinical-grade evaluation in microns/mm with validated measurement protocols

## Task definition & limitation (important for reviewers)

- The current Teeth3DS “prep→target” pipeline is a **synthetic completion proxy** (cut a target tooth and reconstruct).
  - This is useful as **pretraining / method prototyping**, but it is **not identical** to real prep→crown design.
- For a journal submission, the safest framing is:
  - Claim the synthetic task as a **controlled benchmark** and/or **pretraining stage**, and
  - Clearly state what real-task signals are missing (true prep geometry, CAD crown, opposing/neighbor context).

## Recommended paper claim boundaries (practical)

If you want “journal-safe” claims without overreach, the current repo supports:
- A reproducible baseline suite for **dental restoration point-cloud classification** (with k-fold + multi-seed reporting).
- A reproducible baseline suite for **synthetic tooth completion** with **explicit constraint proxies** and ablations.

