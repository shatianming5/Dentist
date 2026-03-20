# Round 0 Refinement — Pivot to Enhanced Benchmark for CMPB/CIBM

## Revised Direction

**Drop ProtoSeg methods paper. Instead: upgrade the existing benchmark paper from DMFR (3 IF) to CMPB/CIBM (7 IF) by adding:**

1. **Protocol-mixing experiments** — Does simply training on both protocols close the gap?
2. **Feature-space analysis** — UMAP/t-SNE + MMD to visualize and quantify the protocol domain gap
3. **Test-time BN adaptation** — Zero-cost baseline: adapt BN statistics at test time
4. **Teeth3DS external validation** — Cross-dataset generalization (binary tooth vs gingiva)
5. **More sophisticated statistical analysis** — Bootstrap CIs, effect size forest plots

## Why This Is Stronger Than ProtoSeg

1. **Honest framing** — Benchmark papers with comprehensive analysis get high citations (cf. nnU-Net)
2. **No novelty attack surface** — No one can say "Protocol-BN is from 2018"
3. **The ranking reversal finding IS novel** — No one has shown this in dental segmentation
4. **Paired scan dataset IS unique** — Same 79 patients under two protocols
5. **350 runs already done** — Just need ~10 GPU-hours of additions
6. **Teeth3DS adds scale** — From 79 to ~1900 scans for external validation

## Experiment Plan

### Phase 1: Feature-Space Analysis (go/no-go gate) — ~2 GPU-hours
- Extract MV-ViT-ft features for all 79×2 scans
- UMAP visualization colored by protocol
- Compute MMD (Maximum Mean Discrepancy) between protocols
- Compute proxy A-distance
- If large gap → protocol shift IS feature-distributional → supports DA narrative
- If small gap → shift is compositional → supports data-centric narrative (both interesting)

### Phase 2: Protocol-Mixing Baseline — ~3 GPU-hours
- Train MV-ViT-ft on balanced+natural jointly (158 scans, no special modules)
- 5-fold × 3 seeds = 15 runs
- Compare: does joint training improve natural-protocol performance?
- Also test: DGCNN and PointNet++ with mixed training

### Phase 3: Test-Time BN Adaptation — ~1 GPU-hour
- Take balanced-trained MV-ViT-ft
- At test time: replace running BN stats with test batch stats from natural data
- Measure: how much of the gap closes for free?

### Phase 4: Teeth3DS Integration — ~5 GPU-hours  
- Download/process Teeth3DS+ data
- Convert to binary (tooth vs gingiva/background)
- Train MV-ViT-ft, DGCNN on Teeth3DS
- Cross-dataset: train on ours → test on Teeth3DS (and vice versa)

### Phase 5: Paper Rewrite for CMPB/CIBM
- Expand from DMFR format (3,700 words) to CMPB format (6,000-8,000 words)
- Add new experiment sections
- Add feature-space analysis figures
- Strengthen related work section (DA literature)
- Add deployment recommendation framework as formal contribution

## Target Paper Structure
1. Introduction (expanded: connect to DA literature)
2. Related Work (new section: dental AI + domain shift + benchmarking)
3. Materials and Methods
   3.1 Dataset (79 paired scans + Teeth3DS)
   3.2 Scanning Protocols
   3.3 Segmentation Methods (7 baselines)
   3.4 Protocol-Mixing Training (new)
   3.5 Test-Time Adaptation (new)
   3.6 Feature-Space Analysis (new)
   3.7 Evaluation Framework
4. Results
   4.1 Within-Protocol Benchmark (existing)
   4.2 Protocol-Induced Ranking Reversal (existing)
   4.3 Feature-Space Domain Gap (new)
   4.4 Protocol-Mixing Experiments (new)
   4.5 Test-Time Adaptation (new)
   4.6 Cross-Protocol Transfer (existing)
   4.7 Cross-Dataset Generalization (new, Teeth3DS)
   4.8 Training Stability (existing)
5. Discussion
6. Conclusions

## Compute Budget
- Phase 1: 2 GPU-hours
- Phase 2: 3 GPU-hours
- Phase 3: 1 GPU-hour
- Phase 4: 5 GPU-hours
- **Total: ~11 GPU-hours → ~4 wall-clock hours on 3 GPUs**
