# Round 0: Initial Proposal — Protocol-Adaptive Multi-View ViT for Dental Restoration Segmentation

## Problem Anchor

Our completed benchmark (350 runs, 7 methods, 2 protocols) demonstrates that scanning protocol choice induces **dramatic ranking reversals** among dental restoration segmentation methods. DGCNN ranks #1 under balanced protocol but drops to #2 under natural/clinical protocol, while MV-ViT-ft rises from #4 to #1. Cross-protocol deployment causes **27–72% mIoU loss**.

**The gap**: We have comprehensively diagnosed the problem, but we don't solve it. A 7+ IF journal (MedIA, CIBM, TMI) requires a **method contribution** — not just a finding.

## Current Bottleneck

1. **No solution proposed** — the paper stops at "protocol matters" without offering a protocol-robust method
2. **Single-center, small dataset** — 79 scans lacks generalizability for top venues
3. **Binary segmentation only** — limits clinical impact
4. **No public dataset validation** — reproducibility concern

## Proposed Method Thesis

### ProtoSeg: Protocol-Adaptive Multi-View Vision Transformer for Dental 3D Segmentation

**Core idea**: Extend MV-ViT-ft with a lightweight **Protocol-Adaptive Module (PAM)** that learns protocol-invariant representations through:

1. **Multi-Protocol Training (MPT)**: Joint training on balanced + natural protocol data with protocol-conditioned batch normalization (Protocol-BN), allowing the model to learn shared representations while adapting statistics per protocol
2. **Adversarial Protocol Alignment (APA)**: A gradient-reversal domain discriminator on the multi-view aggregated features that forces the backbone to learn protocol-invariant point features
3. **Cross-Protocol Consistency (CPC)**: For the 79 paired scans (same tooth, two protocols), enforce feature consistency between protocol views of the same anatomical structure via contrastive loss

**Architecture**:
```
Input: Point Cloud (8192 pts)
  → Multi-View Renderer (6 orthographic views, 512×512)
    → ViT-S/16 Backbone (DINOv3 pretrained, last 4 blocks unfrozen)
      → Patch Features (6 views × 1024 patches × 384-D)
        → [NEW] Protocol-BN (per-protocol normalization statistics)
        → Multi-View Backprojection → Point Features (8192 × 384)
          → [NEW] Adversarial Protocol Discriminator (gradient reversal)
          → [NEW] Cross-Protocol Contrastive Loss (paired scans)
        → MLP Segmentation Head → Labels (8192 × 2)
```

**Why this works**:
- Protocol-BN is proven in domain adaptation (Li et al., 2018) — near-zero overhead
- Adversarial alignment is the simplest effective DA approach — one extra MLP
- Cross-protocol consistency exploits our unique paired dataset — no other dental study has this
- Everything builds on MV-ViT-ft which already shows the best protocol robustness

## Public Dataset Integration: Teeth3DS+

**Teeth3DS+ dataset** (MICCAI 2022/2024 challenge):
- ~1,800 intraoral scans, 900 patients
- Per-tooth segmentation (multi-class, FDI numbering)
- Publicly available (CC BY-NC-ND 4.0)
- Already partially processed in `data/teeth3ds/` and `processed/teeth3ds_teeth/`

**Usage plan**:
- **Cross-dataset generalization**: Train on our data → test on Teeth3DS (zero-shot)
- **Multi-dataset training**: Joint training on our data + Teeth3DS to demonstrate scalability
- **Domain shift simulation**: Teeth3DS as a third "protocol" (different scanner, different population)
- Re-frame task: tooth segmentation (not just restoration) to match Teeth3DS labels

**Note**: Teeth3DS has tooth-level labels (32 classes), not restoration labels. We need to either:
- (a) Convert to binary (tooth vs background) — simpler, direct comparison
- (b) Use as external domain for pre-training + DA — more flexible

## Story Arc for 7+ IF Paper

### Title (working): "Protocol-Adaptive Multi-View Vision Transformer for Robust Dental 3D Segmentation Across Scanning Conditions"

### Narrative:
1. **Motivation** (from benchmark): Protocol choice dramatically affects segmentation — ranking reversal problem (our existing finding, condensed to 1 page)
2. **Method**: ProtoSeg — protocol-adaptive ViT with Protocol-BN + adversarial alignment + cross-protocol consistency
3. **Experiments**:
   - Private dataset: 79 scans × 2 protocols, 5-fold CV
   - Public dataset: Teeth3DS+ (~1800 scans), cross-dataset generalization
   - Ablation: each module's contribution
   - Comparison: 7 baselines (our existing benchmark) + ProtoSeg
4. **Key claims**:
   - ProtoSeg reduces cross-protocol performance gap by X%
   - Protocol-BN alone recovers Y% of the gap at zero computational cost
   - Cross-dataset evaluation confirms generalizability
   - First study to demonstrate and solve protocol-induced ranking reversal in dental segmentation

### Target venues (ordered):
1. **Computer Methods and Programs in Biomedicine** (~7 IF) — strong fit, accepts DA papers
2. **Computers in Biology and Medicine** (~7 IF) — similar scope
3. **Medical Image Analysis** (~13 IF) — reach, needs strong novelty
4. **IEEE Trans Medical Imaging** (~8 IF) — needs deeper technical contribution

## Minimal Experiment Package

### Phase A: Baseline establishment (reuse existing)
- [x] MV-ViT-ft on balanced and natural (already done, n=25 each)
- [x] 6 other methods for comparison (already done)
- [ ] MV-ViT-ft joint training (balanced+natural, no adaptation) — new baseline

### Phase B: Protocol-Adaptive Modules (core contribution)
- [ ] Protocol-BN implementation + training
- [ ] Adversarial protocol alignment training
- [ ] Cross-protocol contrastive loss training
- [ ] Full ProtoSeg (all three modules) training
- [ ] Ablation: each module individually

### Phase C: Teeth3DS integration
- [ ] Download and preprocess Teeth3DS+ data
- [ ] Convert to compatible format (binary tooth vs background)
- [ ] Cross-dataset zero-shot evaluation
- [ ] Joint training with Teeth3DS

### Phase D: Analysis
- [ ] Per-class IoU analysis
- [ ] t-SNE/UMAP visualization of protocol-invariant features
- [ ] Computational overhead analysis
- [ ] Statistical significance tests

## Compute/Risk Estimate

**Compute**:
- MV-ViT-ft training: ~20 min per run (1 GPU, RTX 4090)
- 5-fold × 5 seeds × ~6 conditions = ~150 runs = ~50 GPU-hours
- Teeth3DS preprocessing: ~2 hours
- Teeth3DS training: ~100 additional runs = ~33 GPU-hours
- **Total**: ~85 GPU-hours on 3 free GPUs → **~28 wall-clock hours**

**Risks**:
1. **Protocol-BN may not help much** — balanced/natural may differ in more than batch statistics → MITIGATION: adversarial alignment as backup
2. **Teeth3DS label mismatch** — tooth segmentation ≠ restoration segmentation → MITIGATION: use binary formulation or treat as auxiliary task
3. **79 paired scans may be too few for contrastive learning** → MITIGATION: heavy augmentation + simplest possible contrastive formulation
4. **Adversarial training instability** → MITIGATION: gradient penalty + learning rate scheduling
5. **Novelty concern** — Protocol-BN + adversarial is well-known → MITIGATION: the cross-protocol paired consistency loss and the dental application context add novelty; focus on the paired scan design as unique contribution

## Decision Point

If Phase B shows that ProtoSeg reduces the cross-protocol gap by ≥30% relative (e.g., MV-ViT-ft gap 0.165 → ≤0.115), the paper is viable for 7+ IF. If the improvement is marginal (<15%), we pivot to either:
- (a) Stronger DA: test-time adaptation, style transfer on rendered views
- (b) Different framing: focus on multi-dataset generalization (our data + Teeth3DS + simulated protocols)
