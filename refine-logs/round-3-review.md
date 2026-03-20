# Round 3 Review — aris-reviewer

**Score: 7.5/10 ALMOST** (down from 8.0 due to new section inconsistencies)

## Weaknesses
- W1 (HIGH): Two incompatible feature spaces for A-distance (DINOv2=0.987 vs geometric=2.000) — contradictory abstract
- W2 (HIGH): Table 7 A-distances ceiling-saturated — comparison uninformative, use MMD² instead
- W3 (MODERATE): Pre-training negative result over-generalized (only DGCNN tested)
- W4 (MINOR): Data-volume confound in mixing not discussed
- W5 (MINOR): Abstract too dense (~280 words)
- W6 (MINOR): Reference [27] format inconsistency

## Fixes Applied
- F1: Added bridge paragraph in §3.9 explaining feature-space switch
- F2: Reframed Table 7 with MMD² as primary metric, A-distance as supporting
- F3: Abstract now uses "DINOv2 embeddings" for 0.987, "geometric features" for MMD² comparison
- F4: Narrowed pre-training claims to "DGCNN pre-training" throughout
- F5: Added data-volume confound sentence in §3.6
- Added Teeth3DS pre-training limitation in §4.6
