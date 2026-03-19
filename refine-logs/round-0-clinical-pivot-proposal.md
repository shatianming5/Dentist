# Round 0: Clinical Pivot Proposal — Automated Dental Restoration Assessment on 3D Intraoral Scans

## Problem Anchor

Current dental clinical workflow for assessing dental restorations (fillings, crowns, post-cores, onlays) relies entirely on 2D radiographs and visual inspection. 3D intraoral scanning is rapidly adopted but lacks computational tools for restoration assessment. No prior work has systematically evaluated deep learning pipelines for **restoration detection, classification, and margin quality assessment** on 3D point cloud data from intraoral scans.

## Current Bottleneck

Our prior work established a segmentation benchmark (DGCNN mIoU 0.765 on realistic scans). However:
1. The benchmarking paper was rejected for lacking clinical relevance
2. We conducted 11 systematic experiments (D1-D7 + sub-experiments) to find clinical angles
3. Key findings: segmentation works, everything else fails (classification, transfer, multi-task, domain adaptation)

## Proposed Paper Thesis

**"Comprehensive Deep Learning Assessment of Dental Restorations on 3D Point Clouds: What Works, What Doesn't, and Why"**

Target: JDR or DMFR (dental research journals with computational readership)

### Key Claims
1. **Segmentation is deployment-ready** — DGCNN achieves mIoU 0.765 on realistic (natural-ratio) scans with 5-fold CV, enabling automated restoration boundary delineation
2. **Classification is fundamentally data-limited** — 11 systematic approaches (learned features, handcrafted features, multi-task, seg-guided) all achieve F1 ≈ 0.25-0.28 on 4-class typing, which is near random. This is a data scarcity problem (n=79, 4 imbalanced classes), not a model problem.
3. **Transfer learning provides no significant benefit** — Neither Teeth3DS tooth pretraining (PointNet Δ=+0.012 p=0.48; DGCNN Δ=+0.007 p=0.63) nor cross-protocol pretraining (Δ=+0.003 p=0.72) helps. Domain gap is too large.
4. **Margin geometry is clinically acceptable** — All restoration types show mean step heights <55μm (within ADA threshold <120μm), with no significant between-type differences (Kruskal-Wallis p>0.05). This is a clinically meaningful negative result.
5. **Multi-task learning is counterproductive** — Joint seg+cls training hurts both tasks (seg mIoU drops 0.75→0.50 on natural data; cls F1=0.14-0.17)

### Clinical Significance
- First systematic evaluation of end-to-end restoration analysis pipeline on 3D scans
- Clear evidence-based guidance: segmentation + boundary analysis is viable; classification needs larger datasets
- Quantitative margin quality assessment from 3D scans (novel clinical measurement tool)
- Honest negative results guide future data collection priorities

## Evidence Summary (11 experiments)

| ID | Experiment | Key Result | Status |
|----|-----------|------------|--------|
| D1 | E2E Pipeline | seg 0.732, cls F1=0.23 | ✅ |
| D2 | Boundary Analysis | all <55μm, KW p>0.05 | ✅ |
| D3 | PointNet Transfer | Δ=+0.012, p=0.48 NS | ✅ |
| D4 | Learned Classification (5 configs) | best F1=0.279≈random | ✅ |
| D5 | Multi-task Seg+Cls | hurts both tasks | ✅ |
| D6 | DGCNN Teeth3DS Pretrain | val_acc=0.724 | ✅ |
| D6.2 | DGCNN Transfer | Δ=+0.007, p=0.63 NS | ✅ |
| D6.3 | PN vs DGCNN Transfer | both NS | ✅ |
| D7 | Domain Adaptation | Δ=+0.003, p=0.72 NS | ✅ |

## Minimal Experiment Package (Already Complete)
All experiments are done. Paper assembly needed:
1. Introduction framing clinical need
2. Methods: segmentation, classification, transfer, boundary analysis pipelines
3. Results: Tables 1-5 covering all experiments
4. Discussion: clinical implications, data collection recommendations
5. Supplementary: per-fold details, per-class boundary statistics

## Compute/Risk
- **Compute**: Zero — all experiments are complete
- **Risk**: Reviewer may say "too many negative results" → mitigate by framing as systematic evidence and clinical guidance
- **Opportunity**: Very few papers in dental informatics do this level of systematic negative-result reporting

## Format Options
1. **Full research article** (JDR/DMFR) — comprehensive with all 11 experiments
2. **Focused segmentation + boundary paper** — drop classification, focus on what works
3. **Systematic evaluation paper** — frame as "lessons learned" / benchmark study

