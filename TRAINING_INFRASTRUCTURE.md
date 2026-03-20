# Dentist Training Infrastructure Analysis

## Overview
This codebase has a well-structured multi-phase training pipeline:
- **Phase 0**: Freeze (preprocessing)
- **Phase 1**: Build raw classification datasets
- **Phase 2**: Teeth3DS pretraining (FDI classification)
- **Phase 3**: Raw segmentation & classification training
- **Phase 4**: Fine-tuning on specialized tasks (prep2target)

---

## 1. SEGMENTATION TRAINING: `phase3_train_raw_seg.py`

### Architecture Definitions

#### **PointNetSeg** (Lines 34-63)
```python
# Per-point feature extraction → global pooling → classification head
Encoder:  Conv1d 3→64→128→256→512
Output: Global feature pooling (max) + per-point features
Head: 640 dims (128 local + 512 global) → 256→128→num_classes
```

#### **DGCNNv2Seg** (Lines 66-110)
```python
# Dynamic Graph CNN with edge convolution
Encoder: 
  - conv1: 6 dims (2*in_channels via edge feature) → 64
  - conv2: 128 → 64
  - conv3: 128 → 128
  - conv4: 256 → 256
  - conv5: 512 → emb_dims (default 512)

Output: Concatenates [x1, x2, x3, x4] (512 dims) + emb_dims via maxpool
Head: 512+emb_dims → 256→128→num_classes

Key: Uses get_graph_feature() for k-NN graph construction (k=20 default)
```

#### **PointNet2Seg** (Lines 113-153)
```python
# Set Abstraction (downsampling) + Feature Propagation (upsampling)
Encoder (3 stages):
  - SA1: npoint=1024, nsample=32, output=128
  - SA2: npoint=256, nsample=64, output=256
  - SA3: npoint=0 (all), nsample=0, output=1024 (global)

Decoder (3 stages):
  - FP3: 1024+256 → 256
  - FP2: 256+128 → 128
  - FP1: 128 → 128 (per-point features)

Head: 128 → 128→num_classes

Key: Uses 3-NN inverse-distance weighting for interpolation
```

#### **PointTransformerSeg** (Lines 202-278)
```python
# Lightweight transformer with k-NN attention
Embed: Conv1d 3 → dim (default 96)
Blocks: 4 attention blocks with:
  - kNN attention (k=16) over spatial neighbors
  - Position encoding via relative positions
  - FFN layers (ffn_mult=2.0)
Head: dim → 128 → num_classes
```

### Dataset Loading
```python
class RawSegDataset (Lines 285-326):
  - Loads from: processed/raw_seg/v1 (or v2_natural)
  - Each sample: .npz file with "points" (M,3) and "labels" (M,)
  - Resampling: Random choice to fixed n_points (8192 default)
  - Augmentation (train only):
    * Random Z-rotation: theta ∈ [0, 2π]
    * Random scale: [0.8, 1.2]
    * Jitter: N(0, 0.01)
```

### Training Loop Details (Lines 376-400)
```python
Optimizer: AdamW (lr=1e-3, wd=1e-4)
Scheduler: CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
Loss: CrossEntropyLoss with optional class_weights or focal_loss
  - Class weights: inverse frequency from training split
  - Focal loss: (1-pt)^gamma * CE (gamma=2.0)

Training step:
  1. Forward pass: logits = model(points)  # (B, C, N)
  2. Loss: F.cross_entropy(logits, labels, weight=class_weights)
  3. Backward + grad clip (1.0)
  4. Metrics: per-class IoU, precision, recall, F1

Validation: compute_seg_metrics() - confusion matrix → metrics
Early stopping: patience=20 on val mIoU
```

### Checkpoint Format (Lines 612-614)
```python
torch.save({
    "model": best_state,  # model.state_dict()
    "best_val_miou": float
}, "ckpt_best.pt")
```

---

## 2. TEETH3DS CLASSIFICATION TRAINING: `phase2_train_teeth3ds_fdi_cls.py`

### Model Architectures Supported
1. **PointNetFDIClassifier** (Lines 207-243)
   - Encoder: 3→64→128→256→512 (per-point conv)
   - Global pooling: max(x, dim=2)
   - Head: 512 → 256→128→num_classes

2. **PointNet2Classifier** (imported from `pointnet2.py`)
   - Same as segmentation but with classification head
   - Feature: 1024 global dims → 512→256→num_classes

3. **PointTransformerClassifier** (imported)
   - Same transformer encoder
   - Head: dim → 128 → num_classes

4. **PointMLPClassifier** (imported)
   - MLP-based backbone with relational aggregation

5. **DGCNNv2Classifier** (Lines 564-572)
   - Same encoder as DGCNNv2Seg
   - Output: 2*emb_dims (max + mean pooling of final features)
   - Head: 2*emb_dims → 512→256→num_classes

### Dataset Loading (Lines 456-494)
```python
class Teeth3DSToothFDIDataset:
  - Root: processed/teeth3ds_teeth/v1/
  - Index: index.jsonl with rows[split, fdi, case_key, sample_npz]
  - FDI mapping: fdi_values.json → {fdi_str: class_id}
  - Resampling: n_points (1024 default)
  
  - Augmentation (train=True):
    * Rotate Z: angle ∈ [0, 2π] (opt-in)
    * Scale: [1-aug_scale, 1+aug_scale] (0.02 default)
    * Jitter: sigma=0.005, clip=0.02
    * Point dropout: max_ratio=0.1
```

### Class Weighting Strategy (Lines 500-509)
```python
if balanced_sampler:
    # WeightedRandomSampler: 1/count per class
    weights = [1.0 / counts[fdi] for fdi in train_fdis]
    sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
    shuffle = False
else:
    # Standard shuffle + no weighting in loss
    shuffle = True
```

### Encoder Reusability for Segmentation
✓ **YES** - The PointNetFDIClassifier and others have a clear `.feat` module:
```python
# In PointNetFDIClassifier:
self.feat = nn.Sequential(
    nn.Conv1d(3, 64, ...),
    nn.Conv1d(64, 128, ...),
    ...
    nn.Conv1d(256, 512, ...)
)
# Can extract and transfer directly
```

---

## 3. RAW CLASSIFICATION TRAINING: `phase3_train_raw_cls_baseline.py`

### Models Supported
1. **PointNetClassifier** (Lines 132-187)
   - Encoder: 3→64→128→256→512 → max_pool → feature_dim=512
   - Head: 512 → 256→128→num_classes

2. **PointNet2Classifier** (with SA1/SA2 options)

3. **PointTransformerClassifier**

4. **PointMLPClassifier**

5. **DGCNNClassifier** (Lines 832-911)
   - Encoder: kNN graph edges → conv layers
   - Output: max_pool + mean_pool of final features → feature_dim=2048
   - Head: 2048 → 512→256→num_classes

6. **PointNetCloudSetClassifier** (Lines 190-330)
   - Multi-cloud aggregation with per-cloud max+mean pooling
   - Requires `cloud_id` in point features

7. **PointNetCloudAttnClassifier** (Lines 333-496)
   - Multi-cloud with attention-weighted aggregation

8. **PointNetCloudMILClassifier** (Lines 499-649)
   - Per-cloud logits aggregated via logsumexp (MIL)

9. **GeomMLPClassifier** (Lines 957-1022)
   - Hand-crafted geometric features: centroid, std, bbox, eigenvalues

10. **CloudGeomMLPClassifier** (Lines 1025-1124)
    - Per-cloud geometric features concatenated

### Dataset Loading (Lines 1196-1351)
```python
class RawClsDataset:
  - Root: processed/raw_cls/v1/
  - Index: index.jsonl [split, label, case_key, sample_npz]
  - Label map: label_map.json
  - N-points: variable (4096 default)
  
  - Point features: xyz, normals, curvature, radius, rgb, cloud_id
  - Input preprocessing: max_norm, bbox_diag, PCA alignment
  - Feature caching: Optional (.npz per sample with derived features)
  
  - Augmentation (train=True):
    * Rotate Z: [0, 2π]
    * Scale: [1-aug_scale, 1+aug_scale] (0.2 default)
    * Jitter: sigma=0.01, clip=0.05
    * Point dropout: max_ratio=0.1
    * Tooth position dropout: (simulates missing teeth)
```

---

## 4. TRANSFER LEARNING & PRETRAIN→FINETUNE

### Transfer Learning Pattern (Lines 2378-2432 in phase3_train_raw_cls_baseline.py)

```python
# Step 1: Load pretrained checkpoint
if cfg.init_feat:
    ckpt = torch.load(Path(cfg.init_feat), map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("model") or ckpt

# Step 2: Extract encoder weights (feat.* keys)
feat_state = {
    k[len("feat."):]: v 
    for k, v in state.items() 
    if isinstance(k, str) and k.startswith("feat.")
}

# Step 3: Adapt first conv if input channels differ
# (e.g., xyz→xyz+normals+curvature+radius)
def _adapt_first_conv_weight(src_state, tgt_state, key):
    w_src = src_state[key]
    w_tgt = tgt_state[key]
    if src.shape[1] != tgt.shape[1]:
        # Pad missing channels with mean of existing channels
        new_w = tgt.clone()
        c = min(src.shape[1], tgt.shape[1])
        new_w[:, :c, ...] = w_src[:, :c, ...]
        if tgt.shape[1] > src.shape[1]:
            fill = w_src.mean(dim=1, keepdim=True)
            new_w[:, c:, ...] = fill.repeat(...)
        return new_w

# Step 4: Load encoder with strict=True (others must match)
feat_module.load_state_dict(feat_state, strict=True)

# Step 5: Optional freezing for first K epochs
if cfg.freeze_feat_epochs > 0 and epoch <= cfg.freeze_feat_epochs:
    for p in model.feat.parameters():
        p.requires_grad = False
```

### Existing Transfer Learning Scripts
- **phase4_train_raw_prep2target_finetune.py**: Fine-tunes a pre-trained model on prep→target prediction
  - Load checkpoint with `--init-ckpt` flag
  - Checkpoint format: `{"model": state_dict}`
  - Supports strict=False loading for flexibility

---

## 5. MODEL CHECKPOINT FORMAT

### Standard Format (Classification & Segmentation)
```python
# Saved by phase2_train_teeth3ds_fdi_cls.py (Line 631):
torch.save({
    "model_state": model.state_dict(),  # Full state dict
    "epoch": int(epoch),
    "val_acc": float,
    "val_macro_f1": float
}, "ckpt_best.pt")

# Saved by phase3_train_raw_cls_baseline.py (Lines 2736-2746):
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": opt.state_dict(),
    "labels_by_id": labels_by_id,
    "config": asdict(cfg),
    "val_metrics": val_metrics,
}, best_path)

# Saved by phase3_train_raw_seg.py (Line 614):
torch.save({
    "model": model.state_dict(),  # Note: "model", not "model_state"
    "best_val_miou": float
}, "ckpt_best.pt")
```

### Loading Pattern
```python
# Standard load
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state"])  # or ckpt["model"]

# For transfer learning, extract .feat submodule
feat_state = {
    k[len("feat."):]: v 
    for k, v in state.items() 
    if k.startswith("feat.")
}
model.feat.load_state_dict(feat_state, strict=True)
```

---

## 6. DGCNN LAYER NAMES & ENCODER SEPARATION

### DGCNNv2Encoder Architecture (dgcnn_v2.py, Lines 17-72)
```python
class DGCNNv2Encoder(nn.Module):
    self.conv1 = nn.Sequential(...)  # 2*in_channels → 64
    self.conv2 = nn.Sequential(...)  # 128 → 64
    self.conv3 = nn.Sequential(...)  # 128 → 128
    self.conv4 = nn.Sequential(...)  # 256 → 256
    self.conv5 = nn.Sequential(...)  # 512 → emb_dims (1024)
    
    def forward(points):
        x1 = conv1(get_graph_feature(x, k=20)).max(dim=-1).values
        x2 = conv2(get_graph_feature(x1, k=20)).max(dim=-1).values
        x3 = conv3(get_graph_feature(x2, k=20)).max(dim=-1).values
        x4 = conv4(get_graph_feature(x3, k=20)).max(dim=-1).values
        x_cat = concat([x1, x2, x3, x4])  # (B, 512, N)
        feat = conv5(x_cat)              # (B, 1024, N)
        return concat([max_pool(feat), mean_pool(feat)])  # (B, 2048)
```

### DGCNNv2Classifier Composition (dgcnn_v2.py, Lines 75-124)
```python
class DGCNNv2Classifier(nn.Module):
    self.feat = DGCNNv2Encoder(in_channels, params)
    # self.feat outputs (B, 2*emb_dims) = (B, 2048) by default
    
    self.head = nn.Sequential(
        nn.Linear(2048 + extra_dim, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
        nn.Linear(512, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )
    
    def forward_features(points, extra, domains):
        x = self.feat(points)  # (B, 2048)
        if extra: x = concat([x, extra])
        return x
    
    def forward(points, extra, domains):
        feats = self.forward_features(points, extra, domains)
        return self.head(feats)
```

### Key Layer Names for Transfer Learning
```python
# Full encoder state_dict keys:
feat.conv1.0.weight  # Conv2d
feat.conv1.1.weight  # BatchNorm2d (weight)
feat.conv1.1.bias
...
feat.conv5.0.weight  # Conv1d
feat.conv5.1.weight  # BatchNorm1d
feat.conv5.1.bias

# To extract only encoder:
feat_state = {
    k[len("feat."):]: v 
    for k, v in full_state.items() 
    if k.startswith("feat.")
}
# This gives keys like: "conv1.0.weight", "conv1.1.weight", etc.

# To load into fresh model:
dgcnn_model.feat.load_state_dict(feat_state, strict=True)
```

---

## Summary: Building Your Pipelines

### Transfer Learning: Teeth3DS FDI → Raw Segmentation
```bash
# Step 1: Pretrain on Teeth3DS
python scripts/phase2_train_teeth3ds_fdi_cls.py \
  --model pointnet \
  --epochs 200 \
  --runs-dir runs/pretrain

# Step 2: Fine-tune on raw segmentation with frozen encoder
python scripts/phase3_train_raw_seg.py \
  --model pointnet_seg \
  --init-feat runs/pretrain/ckpt_best.pt \
  --freeze-feat-epochs 10 \
  --epochs 100
```

### End-to-End Restoration Detection
```bash
# Option 1: Classification (tooth presence/absence)
python scripts/phase3_train_raw_cls_baseline.py \
  --model pointnet \
  --data-root processed/raw_cls/v1 \
  --epochs 80

# Option 2: Segmentation (restoration boundary)
python scripts/phase3_train_raw_seg.py \
  --model dgcnn_v2 \
  --data-root processed/raw_seg/v1 \
  --epochs 100

# Option 3: Joint (segmentation + classification)
python scripts/phase3_train_raw_segcls_joint.py \
  --model pointnet \
  --epochs 100
```

### Checkpoint Compatibility
- **SegModels** (PointNetSeg, DGCNNv2Seg, PointNet2Seg):
  - Save format: `{"model": state_dict}`
  - Key: no `.feat` submodule (directly embed in forward)

- **ClassificationModels** (PointNetClassifier, DGCNNv2Classifier):
  - Save format: `{"model_state": state_dict}`
  - Key: `.feat` + `.head` structure
  - **Transferable**: Extract `.feat.*` keys directly

- **PointNet2Classifier**:
  - Key: `.feat` module with SA layers
  - Transferable: Extract `.feat.*` for new head
