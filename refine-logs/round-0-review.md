# Round 0 Review — aris-reviewer

**Score: 4/10 | Verdict: NOT READY**

## Key Findings

### Critical Weaknesses
1. **Zero architectural novelty** — Protocol-BN (2018), gradient reversal (2015), contrastive consistency are all off-the-shelf. ProtoSeg is an assembly, not an invention.
2. **Benchmark paper IS the contribution** — The ranking reversal finding + 350 runs + deployment framework is already strong. ProtoSeg is bolted on to justify a venue upgrade.
3. **n=79 too small for contrastive learning** — 63 pairs per fold with 8192×384-D features dominated by sampling noise.
4. **Teeth3DS integration incoherent** — Different task (tooth vs restoration), different scanner, different population. Not a "third protocol."
5. **Protocol shift may not be feature-space domain gap** — Could be data composition shift (class ratios), not visual domain shift.

### Reviewer's Alternative Direction
Submit enhanced benchmark to CMPB/CIBM (~7 IF) AS a benchmark paper:
- Add naive protocol mixing experiment
- Add test-time BN adaptation baseline  
- Add feature-space analysis (UMAP + MMD)
- Add Teeth3DS as external validation (binary tooth segmentation, honest about task difference)
- Total cost: ~10 GPU-hours

### Decision
**ACCEPT reviewer recommendation.** Pivot from ProtoSeg methods paper to enhanced benchmark at CMPB/CIBM.

### Cheapest Next Steps (go/no-go gates)
1. Feature-space domain gap analysis (UMAP + MMD) — 2 GPU-hours
2. Naive protocol mixing baseline — 2 GPU-hours
3. Test-time BN adaptation — 1 GPU-hour
4. Power analysis — 30 minutes
