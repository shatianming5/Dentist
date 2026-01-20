# Journal Submission Readiness: 12 Issues → Fixes

This repo was hardened for CS/AI journal submission by addressing 12 paper-blocking issues (data leakage, unstable labels/metrics, missing baselines, weak experimental protocol, weak constraints/metrics, and reproducibility gaps).

## One-command audit (recommended)

```bash
python3 scripts/journal_audit.py --root . --out JOURNAL_AUDIT.md
```

## 12 issues and the implemented fixes

1) Teeth3DS split leakage (cross-jaw / same patient in different splits)
   - Fix: patient-level derived splits + rewrite processed index split using patient-level mapping
   - Code: `scripts/phase0_freeze.py`, `scripts/fix_teeth3ds_teeth_splits.py`, `scripts/phase2_build_teeth3ds_teeth.py`
   - Verify: `python3 scripts/journal_audit.py --root .`

2) raw classification has rare / train-missing labels (macro-F1 instability, “test-only class”)
   - Fix: Phase1 adds `--min-train-count` auto-drop after canonicalization/merge
   - Code: `scripts/phase1_build_raw_cls.py`
   - Verify: rebuild dataset with `--min-train-count` and run audit

3) Baseline model is too weak for publication (single PointNet only)
   - Fix: add DGCNN baseline (`--model dgcnn`)
   - Code: `scripts/phase3_train_raw_cls_baseline.py`

4) No imbalance handling (biased training + unstable macro metrics)
   - Fix: inverse-frequency balanced sampler (`--balanced-sampler`) + label smoothing (`--label-smoothing`)
   - Code: `scripts/phase3_train_raw_cls_baseline.py`

5) No self-supervised pretraining to mitigate small labeled raw data
   - Fix: raw AE pretraining; classifier supports `--init-feat` + `--freeze-feat-epochs`
   - Code: `scripts/phase2_train_raw_ae.py`, `scripts/phase3_train_raw_cls_baseline.py`

6) Experimental design lacks K-fold evaluation protocol
   - Fix: generate stratified case-level k-fold splits; training supports `--kfold/--fold/--val-fold`
   - Code: `scripts/make_raw_kfold_splits.py`, `scripts/phase3_train_raw_cls_baseline.py`

7) Teeth3DS prep→target synthesis is too toy (z-cut only)
   - Fix: random plane cut synthesis (`--cut-mode plane`) and config logging
   - Code: `scripts/phase3_train_teeth3ds_prep2target.py`, `scripts/phase4_train_teeth3ds_prep2target_constraints.py`

8) Occlusion constraint is too weak / semantically wrong (using full opposing jaw only)
   - Fix: `--occlusion-mode tooth` uses opposing *tooth* points by FDI; adds cache keying and safe fallbacks
   - Code: `scripts/phase4_train_teeth3ds_prep2target_constraints.py`, `scripts/phase4_eval_teeth3ds_constraints_run.py`

9) Constraint metrics are not sensitive enough; missing/incorrect val metrics in runs
   - Fix: log occlusion contact ratio and `min_d` quantiles; ensure metrics.json contains both `val` and `test`
   - Code: `scripts/phase4_train_teeth3ds_prep2target_constraints.py`, `scripts/phase4_eval_teeth3ds_constraints_run.py`

10) PCA alignment instability (axis/sign ambiguity hurts consistency)
   - Fix: optional `--pca-align-globalz` for more stable PCA axis ordering/sign
   - Code: `scripts/phase2_build_teeth3ds_teeth.py`

11) Missing dataset diagnostics & run summarization utilities
   - Fix: add converted/raw validator + Teeth3DS jaw alignment check + run summarizers
   - Code: `scripts/validate_converted_raw.py`, `scripts/check_teeth3ds_jaw_alignment.py`, `scripts/summarize_raw_cls_runs.py`, `scripts/phase4_summarize_constraints_runs.py`

12) Reproducibility gaps (deps/env not recorded; CLI default pitfalls)
   - Fix: add `requirements.txt`; training scripts write `env.json`; fix optional Path arg defaults (e.g., `--kfold`)
   - Code: `requirements.txt`, `scripts/phase3_train_raw_cls_baseline.py`, `scripts/phase4_train_teeth3ds_prep2target_constraints.py`

