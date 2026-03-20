#!/usr/bin/env python3
"""Compute computational cost (params + inference time) for all 9 segmentation methods.

Outputs paper_tables/computational_cost.json with per-method:
  - params_total, params_trainable
  - inference_ms (single-sample forward pass on GPU, median of 100 runs)
  - train_epoch_s (from existing run logs, if available)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/compute_computational_cost.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure scripts/ is on path for local imports
sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POINTS = 8192
NUM_CLASSES = 2
WARMUP = 20
N_TRIALS = 100


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_inference_ms(model: nn.Module, dummy_input, warmup=WARMUP, n=N_TRIALS):
    """Measure median inference time in ms using CUDA events."""
    model.eval()
    # Warmup
    for _ in range(warmup):
        _ = model(*dummy_input) if isinstance(dummy_input, (list, tuple)) else model(dummy_input)
    torch.cuda.synchronize()

    times = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model(*dummy_input) if isinstance(dummy_input, (list, tuple)) else model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return float(np.median(times))


def get_epoch_times_from_logs(model_key: str, run_dirs: list[str]):
    """Extract avg epoch time from existing run logs."""
    all_dts = []
    for run_dir in run_dirs:
        results_path = Path(run_dir) / "results.json"
        if not results_path.exists():
            continue
        try:
            d = json.load(open(results_path))
            history = d.get("history", [])
            for h in history:
                if "dt" in h:
                    all_dts.append(h["dt"])
        except Exception:
            pass
    return float(np.mean(all_dts)) if all_dts else None


def find_run_dirs(base_dir: Path, prefix: str):
    """Find all run directories matching a model prefix."""
    if not base_dir.exists():
        return []
    return [str(p) for p in sorted(base_dir.iterdir()) if p.is_dir() and p.name.startswith(prefix)]


# ── Point cloud models ────────────────────────────────────────────────────

def measure_pointnet():
    from phase3_train_raw_seg import PointNetSeg
    model = PointNetSeg(num_classes=NUM_CLASSES, dropout=0.3).to(DEVICE)
    dummy = torch.randn(1, N_POINTS, 3, device=DEVICE)
    total, trainable = count_params(model)
    ms = measure_inference_ms(model, dummy)
    epoch_s = get_epoch_times_from_logs(
        "pointnet_seg",
        find_run_dirs(Path("runs/raw_seg"), "pointnet_seg_"))
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"name": "PointNet", "params_total": total, "params_trainable": trainable,
            "inference_ms": round(ms, 3), "train_epoch_s": epoch_s}


def measure_pointnet2():
    from phase3_train_raw_seg import PointNet2Seg
    model = PointNet2Seg(num_classes=NUM_CLASSES, dropout=0.3).to(DEVICE)
    dummy = torch.randn(1, N_POINTS, 3, device=DEVICE)
    total, trainable = count_params(model)
    ms = measure_inference_ms(model, dummy)
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"name": "PointNet++", "params_total": total, "params_trainable": trainable,
            "inference_ms": round(ms, 3), "train_epoch_s": None}


def measure_dgcnn():
    from phase3_train_raw_seg import DGCNNv2Seg
    model = DGCNNv2Seg(num_classes=NUM_CLASSES, dropout=0.3, k=20, emb_dims=512).to(DEVICE)
    dummy = torch.randn(1, N_POINTS, 3, device=DEVICE)
    total, trainable = count_params(model)
    ms = measure_inference_ms(model, dummy)
    epoch_s = get_epoch_times_from_logs(
        "dgcnn_v2",
        find_run_dirs(Path("runs/raw_seg"), "dgcnn_v2_"))
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"name": "DGCNN", "params_total": total, "params_trainable": trainable,
            "inference_ms": round(ms, 3), "train_epoch_s": epoch_s}


def measure_curvenet():
    from phase3_train_raw_seg import CurveNetSeg
    model = CurveNetSeg(num_classes=NUM_CLASSES, dropout=0.3, dim=64,
                        depth=4, k=16, curve_len=4).to(DEVICE)
    dummy = torch.randn(1, N_POINTS, 3, device=DEVICE)
    total, trainable = count_params(model)
    ms = measure_inference_ms(model, dummy)
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"name": "CurveNet", "params_total": total, "params_trainable": trainable,
            "inference_ms": round(ms, 3), "train_epoch_s": None}


def measure_pointmlp():
    from phase3_train_raw_seg import PointMLPSeg
    model = PointMLPSeg(num_classes=NUM_CLASSES, dropout=0.3, dim=64,
                        depth=4, k=16, ffn_mult=2.0).to(DEVICE)
    dummy = torch.randn(1, N_POINTS, 3, device=DEVICE)
    total, trainable = count_params(model)
    ms = measure_inference_ms(model, dummy)
    epoch_s = get_epoch_times_from_logs(
        "pointmlp_seg",
        find_run_dirs(Path("runs/raw_seg"), "pointmlp_seg_"))
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"name": "PointMLP", "params_total": total, "params_trainable": trainable,
            "inference_ms": round(ms, 3), "train_epoch_s": epoch_s}


def measure_point_transformer():
    from phase3_train_raw_seg import PointTransformerSeg
    model = PointTransformerSeg(num_classes=NUM_CLASSES, dropout=0.3, dim=96,
                                depth=4, k=16, ffn_mult=2.0).to(DEVICE)
    dummy = torch.randn(1, N_POINTS, 3, device=DEVICE)
    total, trainable = count_params(model)
    ms = measure_inference_ms(model, dummy)
    del model; gc.collect(); torch.cuda.empty_cache()
    return {"name": "PT (Point Transformer)", "params_total": total, "params_trainable": trainable,
            "inference_ms": round(ms, 3), "train_epoch_s": None}


# ── Multi-view models ─────────────────────────────────────────────────────

def measure_dinov3_mv():
    """DINOv2-MV: frozen ViT backbone + MLP head. Only the head is trainable."""
    from dinov3_seg_pipeline import MLPSegHead
    import timm

    backbone = timm.create_model("vit_small_patch16_dinov3", pretrained=True,
                                 num_classes=0, img_size=512).to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head = MLPSegHead(in_dim=384, num_classes=NUM_CLASSES, hidden=256, dropout=0.3).to(DEVICE)

    # Count params
    bb_total = sum(p.numel() for p in backbone.parameters())
    bb_trainable = 0
    head_total, head_trainable = count_params(head)
    total = bb_total + head_total
    trainable = bb_trainable + head_trainable

    # Measure inference: backbone + head combined
    # Simulate: 6 views × 512×512 image → patch features → gather → head
    n_views = 6
    patch_size = 16
    n_patches = 512 // patch_size  # 32
    n_patch_tokens = n_patches * n_patches  # 1024

    dummy_imgs = torch.randn(n_views, 3, 512, 512, device=DEVICE)
    dummy_feat = torch.randn(1, N_POINTS, 384, device=DEVICE)

    # Measure backbone time (6 views)
    backbone.eval()
    head.eval()

    @torch.no_grad()
    def full_forward():
        feats = backbone.forward_features(dummy_imgs)
        patch_feat = feats[:, 1:1+n_patch_tokens, :]  # (V, 1024, 384)
        # Simulate point gathering (use pre-computed features for head)
        out = head(dummy_feat)
        return out

    # Warmup
    for _ in range(WARMUP):
        full_forward()
    torch.cuda.synchronize()

    times = []
    for _ in range(N_TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        full_forward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    ms = float(np.median(times))

    del backbone, head; gc.collect(); torch.cuda.empty_cache()
    return {"name": "DINOv2-MV", "params_total": total, "params_trainable": trainable,
            "inference_ms": round(ms, 3), "train_epoch_s": None,
            "note": "Frozen ViT-S/16 backbone; only MLP head trainable"}


def measure_mvvit_ft():
    """MV-ViT-ft: ViT backbone with last 4 blocks unfrozen + MLP head."""
    from train_mvvit_ft4 import MVViTFT4

    model = MVViTFT4(num_classes=NUM_CLASSES, unfreeze_blocks=4,
                     hidden=256, dropout=0.3, img_size=512).to(DEVICE)
    total, trainable = count_params(model)

    # Measure inference: 6 views of 512×512
    n_views = 6
    dummy_imgs = torch.randn(1, n_views, 3, 512, 512, device=DEVICE)
    dummy_mappings = torch.randint(0, 1024, (1, n_views, N_POINTS),
                                   device=DEVICE, dtype=torch.long)

    model.eval()

    @torch.no_grad()
    def fwd():
        return model(dummy_imgs, dummy_mappings)

    for _ in range(WARMUP):
        fwd()
    torch.cuda.synchronize()

    times = []
    for _ in range(N_TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fwd()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    ms = float(np.median(times))

    del model; gc.collect(); torch.cuda.empty_cache()
    return {"name": "MV-ViT-ft", "params_total": total, "params_trainable": trainable,
            "inference_ms": round(ms, 3), "train_epoch_s": None,
            "note": "ViT-S/16 last 4 blocks unfrozen + MLP head"}


# ── Random Forest ─────────────────────────────────────────────────────────

def measure_rf():
    """RF: sklearn RandomForest — not a neural network, report config."""
    return {
        "name": "RF (Random Forest)",
        "params_total": None,
        "params_trainable": None,
        "inference_ms": None,
        "train_epoch_s": None,
        "n_estimators": 100,
        "max_depth": 20,
        "feature_dims": 11,
        "note": "sklearn RandomForestClassifier; per-point features (11 dims)"
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    methods = []
    runners = [
        ("RF", measure_rf),
        ("PointNet", measure_pointnet),
        ("PointNet++", measure_pointnet2),
        ("DGCNN", measure_dgcnn),
        ("CurveNet", measure_curvenet),
        ("PointMLP", measure_pointmlp),
        ("PT", measure_point_transformer),
        ("DINOv2-MV", measure_dinov3_mv),
        ("MV-ViT-ft", measure_mvvit_ft),
    ]

    for label, fn in runners:
        print(f"\n{'='*60}")
        print(f"  Measuring: {label}")
        print(f"{'='*60}")
        try:
            result = fn()
            methods.append(result)
            pt = result.get("params_total")
            ptr = result.get("params_trainable")
            ms = result.get("inference_ms")
            pt_str = f"{pt:,}" if pt is not None else "N/A"
            ptr_str = f"{ptr:,}" if ptr is not None else "N/A"
            ms_str = f"{ms:.3f} ms" if ms is not None else "N/A"
            print(f"  Total params:     {pt_str}")
            print(f"  Trainable params: {ptr_str}")
            print(f"  Inference time:   {ms_str}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            methods.append({"name": label, "error": str(e)})

    # Save
    out_dir = Path("paper_tables")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "computational_cost.json"

    output = {
        "description": "Computational cost comparison for 9 segmentation methods",
        "n_points": N_POINTS,
        "num_classes": NUM_CLASSES,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timing_config": {"warmup": WARMUP, "n_trials": N_TRIALS, "batch_size": 1},
        "methods": methods,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<25} {'Total Params':>14} {'Trainable':>14} {'Infer (ms)':>12} {'Epoch (s)':>10}")
    print(f"{'-'*80}")
    for m in methods:
        name = m["name"]
        pt = m.get("params_total")
        ptr = m.get("params_trainable")
        ms = m.get("inference_ms")
        ep = m.get("train_epoch_s")
        pt_s = f"{pt:,}" if pt is not None else "—"
        ptr_s = f"{ptr:,}" if ptr is not None else "—"
        ms_s = f"{ms:.3f}" if ms is not None else "—"
        ep_s = f"{ep:.2f}" if ep is not None else "—"
        print(f"{name:<25} {pt_s:>14} {ptr_s:>14} {ms_s:>12} {ep_s:>10}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
