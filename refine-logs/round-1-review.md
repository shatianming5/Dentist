## Round 1 Review — Enhanced Benchmark for CMPB/CIBM

**Score: 7.0/10 — ALMOST**

### Critical Fixes Required:
1. W1 CRITICAL: DINOv2-MV mixing baseline unfair (bal-only vs mixed, not nat-only vs mixed). True lift = +0.008.
   → Fix: 3-condition table {bal-only, nat-only, mixed} × {eval-bal, eval-nat} for all methods
2. W2 HIGH: BN adaptation only on DINOv2-MV → Run on DGCNN too
3. W3 HIGH: Scan-level MMD confounded with class composition → Discuss + point-level evidence
4. W4 MEDIUM: No p-values on mixing improvements → Add Mann-Whitney U + Cohen's d
5. W5 MEDIUM: Paper not yet rewritten for CMPB → Write new sections
6. W6 MEDIUM: Catastrophic outliers (mIoU≈0.015) not acknowledged → Failure analysis

### Cheapest path to 8+:
- Fix mixing table with existing data (0 GPU-hrs)
- Run DGCNN BN adaptation (0.5 GPU-hrs)
- Add statistical tests (0 compute)
- Add class-ratio confound paragraph
- Write new manuscript sections
