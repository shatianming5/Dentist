#!/usr/bin/env python3
"""
Generate dataset split files (train/val[/test]) listing sample paths.

Features
- Supports PLY / NPY / NPZ (and any extension via --ext)
- Recursive scan of a source directory
- Shuffle with fixed seed and ratio-based splitting
- Relative or absolute paths in output lists

Usage examples
- Default (PLY/NPY/NPZ under data_ply, 90/10 split):
  python scripts/generate_splits.py --src data_ply --out splits

- Include NPZ under data_npz and make test split:
  python scripts/generate_splits.py --src data_npz --ext .npz --val-ratio 0.1 --test-ratio 0.1 --out splits

- Allow multiple extensions:
  python scripts/generate_splits.py --src data_npz --ext .npy,.npz --out splits
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Iterable, List


def find_files(src: str, exts: List[str], recursive: bool = True) -> List[str]:
    out: List[str] = []
    exts_l = [e.lower() for e in exts]
    if recursive:
        for root, _, files in os.walk(src):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts_l:
                    out.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(src):
            p = os.path.join(src, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in exts_l:
                out.append(p)
    return sorted(out)


def to_rel(paths: List[str], base: str, forward_slash: bool = True) -> List[str]:
    rels = [os.path.relpath(p, base) for p in paths]
    if forward_slash and os.sep != '/':
        rels = [r.replace(os.sep, '/') for r in rels]
    return rels


def write_list(paths: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in paths:
            f.write(p + '\n')


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description='Generate train/val[/test] split lists')
    ap.add_argument('--src', required=True, help='Source directory containing files')
    ap.add_argument('--out', default='splits', help='Output directory for split files')
    ap.add_argument('--ext', default='.ply,.npy,.npz', help='Comma-separated extensions to include')
    ap.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio (0-1)')
    ap.add_argument('--test-ratio', type=float, default=0.0, help='Test ratio (0-1)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    ap.add_argument('--no-shuffle', action='store_true', help='Disable shuffling before split')
    ap.add_argument('--absolute', action='store_true', help='Write absolute paths instead of relative')
    ap.add_argument('--no-recursive', action='store_true', help='Do not scan subdirectories')
    args = ap.parse_args(list(argv))

    src = os.path.abspath(args.src)
    exts = [e.strip() for e in args.ext.split(',') if e.strip()]
    files = find_files(src, exts, recursive=(not args.no_recursive))

    if not files:
        print(f'No files found in {src} with extensions {exts}')
        return 1

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(files)

    n = len(files)
    n_test = int(round(n * args.test_ratio))
    n_val = int(round(n * args.val_ratio))

    if n_test + n_val >= n:
        print('Invalid ratios: not enough samples left for train set')
        return 1

    test_files = files[:n_test]
    val_files = files[n_test:n_test + n_val]
    train_files = files[n_test + n_val:]

    if args.absolute:
        train_out = train_files
        val_out = val_files
        test_out = test_files
    else:
        train_out = to_rel(train_files, src)
        val_out = to_rel(val_files, src)
        test_out = to_rel(test_files, src)

    os.makedirs(args.out, exist_ok=True)
    write_list(train_out, os.path.join(args.out, 'train.txt'))
    write_list(val_out, os.path.join(args.out, 'val.txt'))
    if n_test > 0:
        write_list(test_out, os.path.join(args.out, 'test.txt'))

    print(f'Total: {n} | train: {len(train_out)} val: {len(val_out)} test: {len(test_out) if n_test>0 else 0}')
    print(f'Wrote: {os.path.join(args.out, "train.txt")}, {os.path.join(args.out, "val.txt")}')
    if n_test > 0:
        print(f'Wrote: {os.path.join(args.out, "test.txt")}')
    if not args.absolute:
        print(f'Paths are relative to: {src}')
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))

