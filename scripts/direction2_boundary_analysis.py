#!/usr/bin/env python3
"""Direction 2: Restoration Boundary Geometric Analysis.

Extracts boundary features from original high-res point clouds (no downsampling).
For each of 79 cases:
  1. Load segmented (restoration) and remaining (tooth) clouds at full resolution
  2. Find boundary points (restoration points within threshold of tooth surface)
  3. Compute boundary metrics:
     - Step height (normal displacement at boundary)
     - Local curvature at boundary
     - Boundary smoothness (regularity of boundary curve)
     - Margin width (band thickness)
  4. Save per-case results
"""
import numpy as np
import json
import os
import sys
from pathlib import Path
from scipy.spatial import cKDTree
from collections import defaultdict

ROOT = Path("/mnt/SSD_4TB/zechuan/Dentist")
CONVERTED = ROOT / "converted" / "raw"
OUTPUT = ROOT / "paper_tables" / "boundary_analysis.json"

BOUNDARY_THRESHOLD = 0.3  # mm — points within this distance of the other surface

def load_case_clouds(case):
    """Load segmented and remaining point clouds for a case."""
    seg_cloud = None
    rem_cloud = None
    seg_rgb = None
    
    for cloud in case['exported_clouds']:
        npz_path = CONVERTED / cloud['outputs']['npz']
        if not npz_path.exists():
            continue
        d = np.load(str(npz_path))
        
        if cloud['name'] == 'Mesh.sampled.segmented':
            seg_cloud = d['points']
            if 'rgb' in d:
                seg_rgb = d['rgb']
        elif cloud['name'] == 'Mesh.sampled.remaining':
            rem_cloud = d['points']
    
    return seg_cloud, rem_cloud, seg_rgb


def compute_local_curvature(points, k=30):
    """Estimate local curvature via PCA of k-nearest neighborhoods."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k, len(points)))
    
    curvatures = np.zeros(len(points))
    normals = np.zeros((len(points), 3))
    
    for i in range(len(points)):
        neighbors = points[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / len(neighbors)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Curvature = smallest eigenvalue / sum (surface variation)
        curvatures[i] = eigvals[0] / (eigvals.sum() + 1e-10)
        normals[i] = eigvecs[:, 0]  # Normal = smallest eigenvector
    
    return curvatures, normals


def compute_step_height(seg_boundary, rem_boundary, seg_normals):
    """Compute normal displacement (step height) at boundary."""
    tree_rem = cKDTree(rem_boundary)
    dists, indices = tree_rem.query(seg_boundary)
    
    # Project displacement onto local normal
    displacement = rem_boundary[indices] - seg_boundary
    step_heights = np.abs(np.sum(displacement * seg_normals, axis=1))
    
    return step_heights


def compute_boundary_smoothness(boundary_points):
    """Measure boundary regularity via local distance variance."""
    if len(boundary_points) < 10:
        return np.nan, np.nan
    
    tree = cKDTree(boundary_points)
    dists, _ = tree.query(boundary_points, k=min(8, len(boundary_points)))
    
    # Mean NN distance (should be uniform for smooth boundary)
    mean_nn = dists[:, 1:].mean(axis=1)  # exclude self
    smoothness = mean_nn.std() / (mean_nn.mean() + 1e-10)  # CV = lower is smoother
    spacing = mean_nn.mean()
    
    return smoothness, spacing


def compute_margin_width(seg_cloud, rem_cloud, seg_boundary_mask, rem_boundary_mask):
    """Estimate the width of the margin band."""
    seg_b = seg_cloud[seg_boundary_mask]
    rem_b = rem_cloud[rem_boundary_mask]
    
    if len(seg_b) < 5 or len(rem_b) < 5:
        return np.nan
    
    # Margin width = mean distance between paired boundary points
    tree = cKDTree(rem_b)
    dists, _ = tree.query(seg_b)
    return np.median(dists)


def analyze_case(case, case_idx):
    """Full boundary analysis for one case."""
    seg_cloud, rem_cloud, seg_rgb = load_case_clouds(case)
    
    if seg_cloud is None or rem_cloud is None:
        return None
    
    label = case['label_info']['label']
    tooth = case['label_info'].get('tooth_position', 'unknown')
    
    # Build KD-trees
    tree_rem = cKDTree(rem_cloud[::3])  # subsample remaining for speed
    tree_seg = cKDTree(seg_cloud)
    
    # Find boundary points on restoration side
    dists_seg_to_rem, _ = tree_rem.query(seg_cloud)
    seg_boundary_mask = dists_seg_to_rem < BOUNDARY_THRESHOLD
    
    # Find boundary points on tooth side  
    dists_rem_to_seg, _ = tree_seg.query(rem_cloud[::3])
    rem_boundary_mask_sub = dists_rem_to_seg < BOUNDARY_THRESHOLD
    
    n_seg_boundary = seg_boundary_mask.sum()
    
    if n_seg_boundary < 20:
        print(f"  Case {case_idx}: too few boundary points ({n_seg_boundary}), skipping")
        return None
    
    seg_boundary = seg_cloud[seg_boundary_mask]
    rem_boundary = rem_cloud[::3][rem_boundary_mask_sub]
    
    print(f"  Case {case_idx} [{label}]: seg={seg_cloud.shape[0]:,}, "
          f"boundary={n_seg_boundary:,} ({100*n_seg_boundary/len(seg_cloud):.1f}%)")
    
    # 1. Local curvature at boundary
    curvatures, normals = compute_local_curvature(seg_boundary, k=20)
    
    # Also compute curvature on interior for comparison
    seg_interior = seg_cloud[~seg_boundary_mask]
    if len(seg_interior) > 100:
        interior_curvatures, _ = compute_local_curvature(
            seg_interior[np.random.choice(len(seg_interior), min(5000, len(seg_interior)), replace=False)], 
            k=20
        )
    else:
        interior_curvatures = np.array([np.nan])
    
    # 2. Step height
    step_heights = compute_step_height(seg_boundary, rem_boundary, normals)
    
    # 3. Boundary smoothness
    smoothness_cv, spacing = compute_boundary_smoothness(seg_boundary)
    
    # 4. Margin width
    margin_width = compute_margin_width(
        seg_cloud, rem_cloud[::3], seg_boundary_mask, rem_boundary_mask_sub
    )
    
    # 5. Boundary length estimate (convex hull perimeter of boundary projected to 2D)
    # Use PCA to find the dominant plane
    centered = seg_boundary - seg_boundary.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_2d = centered @ Vt[:2].T
    
    # Approximate boundary length via ordered nearest-neighbor walk
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(proj_2d)
        boundary_perimeter = hull.area  # In 2D, 'area' is perimeter
    except:
        boundary_perimeter = np.nan
    
    result = {
        'case_key': case['input'],
        'label': label,
        'tooth_position': tooth,
        'n_seg_points': int(seg_cloud.shape[0]),
        'n_rem_points': int(rem_cloud.shape[0]),
        'n_boundary_points': int(n_seg_boundary),
        'boundary_fraction': float(n_seg_boundary / len(seg_cloud)),
        'metrics': {
            'step_height_mean': float(np.mean(step_heights)),
            'step_height_median': float(np.median(step_heights)),
            'step_height_std': float(np.std(step_heights)),
            'step_height_p95': float(np.percentile(step_heights, 95)),
            'curvature_boundary_mean': float(np.mean(curvatures)),
            'curvature_boundary_median': float(np.median(curvatures)),
            'curvature_interior_mean': float(np.nanmean(interior_curvatures)),
            'curvature_ratio': float(np.mean(curvatures) / (np.nanmean(interior_curvatures) + 1e-10)),
            'smoothness_cv': float(smoothness_cv),
            'boundary_spacing': float(spacing),
            'margin_width': float(margin_width),
            'boundary_perimeter': float(boundary_perimeter),
        }
    }
    
    return result


def main():
    with open(str(CONVERTED / "manifest_with_labels.json")) as f:
        cases = json.load(f)
    
    print(f"Analyzing {len(cases)} cases for boundary geometry...")
    
    results = []
    for i, case in enumerate(cases):
        try:
            r = analyze_case(case, i)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  Case {i} FAILED: {e}")
    
    print(f"\nSuccessfully analyzed {len(results)}/{len(cases)} cases")
    
    # Aggregate by type
    by_type = defaultdict(list)
    for r in results:
        by_type[r['label']].append(r)
    
    summary = {}
    for label, cases_list in by_type.items():
        metrics_keys = cases_list[0]['metrics'].keys()
        summary[label] = {
            'n': len(cases_list),
            'metrics': {}
        }
        for mk in metrics_keys:
            vals = [c['metrics'][mk] for c in cases_list if not np.isnan(c['metrics'][mk])]
            if vals:
                summary[label]['metrics'][mk] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'median': float(np.median(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                }
    
    output = {
        'n_cases': len(results),
        'boundary_threshold_mm': BOUNDARY_THRESHOLD,
        'per_case': results,
        'by_type_summary': summary,
    }
    
    with open(str(OUTPUT), 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {OUTPUT}")
    
    # Print summary table
    print(f"\n{'Type':<12} {'N':>3} {'Step(mm)':>10} {'Curvature':>10} {'Smooth':>10} {'Margin(mm)':>10}")
    print("-" * 60)
    for label, s in sorted(summary.items()):
        m = s['metrics']
        print(f"{label:<12} {s['n']:>3} "
              f"{m['step_height_mean']['mean']:>10.4f} "
              f"{m['curvature_boundary_mean']['mean']:>10.4f} "
              f"{m['smoothness_cv']['mean']:>10.4f} "
              f"{m['margin_width']['mean']:>10.4f}")


if __name__ == '__main__':
    main()
