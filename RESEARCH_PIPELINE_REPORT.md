# Research Pipeline Report — Dentist Project

**Last updated**: 2026-03-22  
**Status**: ✅ READY FOR SUBMISSION  
**Review score**: 8.5/10 (Round 11)

## Active Direction

Protocol-induced domain shift benchmark for dental restoration segmentation.  
9 methods × 2 protocols × 5-fold CV × 5 seeds = 450 core benchmark runs.  
Additional experiments: mixing, PAFA, BN adaptation, Teeth3DS pre-training, cross-dataset, mixing ratio ablation = 265 more runs.  
**Total: 715 experimental configurations.**

## Implemented Changes (This Session)

### New Experiments
1. **PointMLP benchmark** — 50 runs (2 protocols × 5 seeds × 5 folds)
   - Balanced: 0.476 ± 0.039, Natural: 0.412 ± 0.071
   - Added as 8th method, 6th paradigm (MLP-based)

2. **Mixing ratio ablation** — 75 runs (5 ratios × 3 seeds × 5 folds)
   - 0%→25%→50%→75%→100% natural data
   - Peak at 75% (nat mIoU = 0.658), non-monotonic
   - Balanced performance stable (0.957→0.945)

3. **PAFA seed alignment** — 30 runs (aligned seeds for Table 8)
   - Δ = −0.046, p = 0.001, d = −1.19

### Paper Updates
- 8 methods (added PointMLP), 6 paradigms, 665 configs
- Table 9 (mixing ablation), Figure 6 (dose-response curve)
- All supplementary tables updated (S1: 8-method Dice, S3: +PointMLP row)
- Reference [24] added (Ma et al. PointMLP, ICLR 2022), refs renumbered
- 3 new figures: protocol gap barplot, PAFA scatter, bump chart

### Documentation
- README.md: bilingual EN/CN rewrite
- .gitignore: configured for large file exclusion

## Score Progression

| Round | Score | Verdict | Key Change |
|-------|-------|---------|------------|
| 1 | 7.0 | NOT READY | Initial review |
| 2 | 8.0 | ALMOST | Statistical rigor |
| 3 | 7.5 | ALMOST | A-distance inconsistency |
| 4 | 8.5 | READY | Paired bootstrap, JDR framing |
| 5 | 8.5 | READY | Stable |
| 6 | 8.0 | ALMOST | Table 4/8 discrepancy |
| 7 | 8.5 | READY | Focal loss documented |
| 8 | 8.0 | ALMOST | PointMLP integration artifacts |
| 9 | 8.0 | ALMOST | Figure/table count fixes |
| 10 | 8.5 | READY | Final micro-fixes |

## Remaining Blockers

1. **Institutional placeholders** — Lines 41 and 348: `[Institution]`, `[XXX]`
2. **LaTeX formatting** — Convert from markdown to target journal template
3. **Supplementary materials** — Package separately if required

## Target Journals

| Journal | IF | Probability |
|---------|---:|:-----------:|
| DMFR | ~3.1 | 70-85% |
| J Dentistry | ~4.8 | 70-85% |
| Dental Materials | ~5.0 | 70-85% |
| CMPB / CIBM | ~7.0 | 40-55% |
