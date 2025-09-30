#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from typing import Iterable, List, Tuple


def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center to centroid and scale to unit sphere (robust, overflow-safe).

    Returns normalized points, center, combined scale.
    """
    pts64 = pts.astype(np.float64, copy=False)
    c = pts64.mean(axis=0)
    q = pts64 - c
    s0 = np.max(np.abs(q))
    if not np.isfinite(s0) or s0 == 0:
        s0 = 1.0
    q = q / s0
    # second-stage radius scale
    r = np.linalg.norm(q, axis=1).max()
    if not np.isfinite(r) or r == 0:
        r = 1.0
    q = q / r
    s = float(s0 * r)
    return q.astype(np.float32), c.astype(np.float32), s


def sample_points(pts: np.ndarray, n: int) -> np.ndarray:
    m = pts.shape[0]
    if m >= n:
        idx = np.random.choice(m, n, replace=False)
    else:
        pad = np.random.choice(m, n - m, replace=True)
        idx = np.concatenate([np.arange(m), pad])
    return pts[idx]


def label_from_name(name: str) -> int:
    n = name.lower()
    if 'upper' in n:
        return 1
    if 'lower' in n:
        return 0
    return -1


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description='Export BIN to NPZ (normalized point clouds)')
    ap.add_argument('--src', default='data', help='Source folder with .bin files')
    ap.add_argument('--dst', default='data_npz', help='Destination folder for .npz files')
    ap.add_argument('--points', type=int, default=2048, help='Points per sample')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of files (for quick test)')
    args = ap.parse_args(list(argv))

    # import reader
    scripts_dir = os.path.join(os.getcwd(), 'scripts')
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    from ccbin_v2_reader import load_points  # type: ignore

    os.makedirs(args.dst, exist_ok=True)
    files = [f for f in os.listdir(args.src) if f.lower().endswith('.bin')]
    files.sort()
    if args.limit:
        files = files[: args.limit]

    count = 0
    for fn in files:
        src_path = os.path.join(args.src, fn)
        try:
            pts, _ = load_points(src_path, peek=0)
        except Exception as e:
            print(f'[skip] {fn}: {e}')
            continue
        if len(pts) == 0:
            print(f'[skip] {fn}: empty')
            continue
        arr = np.asarray(pts, dtype=np.float32)
        # drop non-finite points
        mask = np.isfinite(arr).all(axis=1)
        arr = arr[mask]
        if arr.size == 0:
            print(f'[skip] {fn}: no finite points')
            continue
        arr = sample_points(arr, args.points)
        arr, center, scale = normalize_points(arr)
        out_name = os.path.splitext(fn)[0] + '.npz'
        out_path = os.path.join(args.dst, out_name)
        label = label_from_name(fn)
        np.savez(out_path, points=arr, center=center, scale=np.float32(scale), label=np.int32(label), filename=fn)
        count += 1
    print(f'Exported {count} samples to {args.dst}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
