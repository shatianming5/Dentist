# Round 1 Review — Clinical Pivot Proposal

## Score: 4.5/10
## Verdict: NOT READY

## Key Takeaways
1. **Don't pivot** — the existing 8.5/10 benchmarking paper is the strongest asset
2. Comprehensive "what works/doesn't" framing is a **strict downgrade** from focused benchmarking
3. 4 of 5 proposed claims are null/negative — reads as "nothing works except basic seg"
4. Boundary analysis lacks physical validation (point cloud metric vs ADA threshold is category error)
5. Classification negative is expected (n=79, 4 classes) — provides no clinical insight beyond "collect more data"

## Recommended Action (Option B — Cheapest, Highest ROI)
Add to the existing benchmarking paper:
1. Clinical context paragraphs (pre-visit chart review, screening workflow)
2. Learning curve analysis (train on 20/40/60/79 → predict when mIoU >0.85)
3. Dice scores (clinical standard, not just mIoU)
4. Sample size requirements for restoration typing (reframe cls negative as planning tool)
5. Per-case boundary CIs with bootstrap

## What to NOT do
- Don't write a new paper
- Don't add classification/transfer/multi-task negative results to main body
- Don't claim boundary metrics are "clinically acceptable" without physical validation
