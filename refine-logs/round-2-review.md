# Round 2 Review — Enhanced Benchmarking Paper

## Score: 7.5/10 (up from 4.5)
## Verdict: ALMOST

## What Improved
- Correctly returned to focused benchmarking paper
- Learning curve experiment with real data
- Dice scores computed (clinical standard)
- Sample size guidance is constructive

## Remaining Weaknesses (all writing-level)
1. W1: Power law fit is over-parameterized (3 params, 4 points, b hit boundary)
2. W2: Learning curve uses 1 seed while main results use 3 seeds
3. W3: "Power analysis" label is informal — rename to "sample size estimation"
4. W4: Inconsistent mIoU baselines (0.690 vs 0.732 vs 0.770) across analyses

## Minimum Fixes
- Fix 1: Soften extrapolation, add caveats, use ranges not point estimates
- Fix 2: Add seed-count reconciliation sentence
- Fix 3: Rename "power analysis" → "sample size estimation"
- Fix 4: Declare canonical aggregation method

## Estimated effort: ~35 minutes of writing edits, zero new experiments
