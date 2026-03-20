# Round 1 Refinement — All Fixes Applied

## Fixes Implemented

### W1: Honest 3-Condition Table (CRITICAL → FIXED)
- Replaced the unfair DINOv2-MV comparison with 3-condition table
- Three conditions: {bal-only transfer, within-protocol, mixed}
- DINOv2-MV true mixing lift: +0.008 (was incorrectly +0.117)
- Only DGCNN significantly benefits from mixing (+0.058, p<0.001)
- Saved: paper_tables/honest_3condition_table.json

### W2: DGCNN BN Adaptation (HIGH → FIXED)
- Ran BN adaptation on 10 DGCNN checkpoints (seeds 42, 7 × 5 folds)
- Result: 0.415 → 0.122 (Δ = -0.293, -71%) — even worse than DINOv2-MV
- Two negative results across architecturally distinct models
- Saved: paper_tables/bn_adaptation_dgcnn.json

### W3: Class-Ratio Confound (HIGH → FIXED)
- Added explicit paragraph in §3.5 acknowledging scan-level confound
- Cited point-level analysis as resolution
- Added §4.2 discussion section on nature of domain gap

### W4: Statistical Tests (MEDIUM → FIXED)
- Mann-Whitney U + Cohen's d for all 3 mixing comparisons in Table 4
- DGCNN: p<0.001, d=1.47 (significant)
- MV-ViT-ft: p=0.171, d=0.40 (not significant — honest)
- DINOv2-MV: p=0.255, d=0.05 (negligible)
- Saved: paper_tables/mixing_statistical_tests.json

### W5: Paper Rewrite (MEDIUM → FIXED)
- Added §2.6-2.8 (methods), §3.5-3.7 (results), §4.2-4.3 (discussion)
- Updated title, abstract, conclusions
- Total ~5,900 words

### W6: Catastrophic Outliers (MEDIUM → PARTIAL)
- Discussed in context of mixing results
- Full per-scan failure analysis deferred
