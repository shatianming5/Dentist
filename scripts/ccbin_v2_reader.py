#!/usr/bin/env python3
"""
Native reader (best-effort) for CloudCompare .bin v2 files.

Notes
- This is a pragmatic reader focused on extracting point clouds (XYZ)
  and optional RGB from typical CC v2 BIN files like the ones in ./data.
- It does not re-implement the full CC object graph; instead it scans the
  serialized stream for the point array blocks written by CC's
  ccSerializationHelper::GenericArrayToFile/FromFile.
- For robust, standards-based workflows, prefer converting to PLY with
  CloudCompare CLI, then using scripts/read_ply_points.py.

Usage
  python scripts/ccbin_v2_reader.py data/Group-10.bin --summary
  python scripts/ccbin_v2_reader.py data/Group-10.bin --peek 5
  python scripts/ccbin_v2_reader.py data --glob "*.bin" --limit 2 --summary

Output
- Prints the detected point count and previews coordinates (+RGB if found).

Limitations
- Heuristic scanning may miss clouds in exotic files.
- RGB extraction is attempted when a matching color array follows the
  points block with a compatible element count.
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import struct
from typing import Iterable, List, Optional, Sequence, Tuple


def _read_le_u32(b: bytes, off: int) -> Tuple[int, int]:
    return struct.unpack_from('<I', b, off)[0], off + 4


def _read_header(path: str) -> Tuple[int, int]:
    with open(path, 'rb') as f:
        head = f.read(12)
    if len(head) < 12 or head[:3] != b'CCB':
        raise ValueError('Not a CloudCompare v2 BIN (missing CCBx header)')
    flags = head[3]
    ver = struct.unpack_from('<I', head, 4)[0]
    return flags, ver


def _scan_points(b: bytes, coord_is_double: bool) -> Tuple[int, int, int]:
    """Return (offset, count, stride_bytes) of the first plausible XYZ block.

    The array layout written by CC for points is:
      - 1 byte: componentCount (must be 3)
      - 4 bytes LE: elementCount (N)
      - N * 3 * (4 or 8) bytes: interleaved XYZ values
    """
    comp_size = 8 if coord_is_double else 4
    nmin, nmax = 10_000, 50_000_000  # plausible bounds
    i = 0
    end = len(b) - 1 - 4
    candidates: List[Tuple[int, int, int]] = []  # (off,n,stride)
    while i < end:
        if b[i] != 3:
            i += 1
            continue
        n = struct.unpack_from('<I', b, i + 1)[0]
        if not (nmin <= n <= nmax):
            i += 1
            continue
        total = 1 + 4 + n * 3 * comp_size
        if i + total > len(b):
            i += 1
            continue
        # light sanity check: parse first few coords and ensure they are finite
        try:
            j = i + 5
            if coord_is_double:
                xs = struct.unpack_from('<d', b, j)[0]
                ys = struct.unpack_from('<d', b, j + 8)[0]
                zs = struct.unpack_from('<d', b, j + 16)[0]
            else:
                xs = struct.unpack_from('<f', b, j)[0]
                ys = struct.unpack_from('<f', b, j + 4)[0]
                zs = struct.unpack_from('<f', b, j + 8)[0]
            # very loose plausibility: non-NaN and finite
            for v in (xs, ys, zs):
                if not (v == v):  # NaN check
                    raise ValueError
        except Exception:
            i += 1
            continue
        # keep the largest plausible candidate; points arrays are usually the largest 3-comp float/double array
        candidates.append((i, n, comp_size * 3))
        i += 1
    if not candidates:
        raise RuntimeError('No plausible XYZ array block found')
    # rank candidates: prefer ones followed by a plausible RGB array; then prefer sane value ranges; then larger n
    def score(cand: Tuple[int, int, int]) -> Tuple[int, int, int]:
        off, n, stride = cand
        s = 0
        if _maybe_following_rgb(b, off + 1 + 4 + n * stride, n) is not None:
            s += 2
        # range plausibility check on first few points
        try:
            j = off + 5
            ok = 0
            limit = min(n, 32)
            for _ in range(limit):
                if coord_is_double:
                    x = struct.unpack_from('<d', b, j)[0]; y = struct.unpack_from('<d', b, j + 8)[0]; z = struct.unpack_from('<d', b, j + 16)[0]
                    j += 24
                else:
                    x = struct.unpack_from('<f', b, j)[0]; y = struct.unpack_from('<f', b, j + 4)[0]; z = struct.unpack_from('<f', b, j + 8)[0]
                    j += 12
                if all(v == v and abs(v) < 1e6 for v in (x, y, z)):
                    ok += 1
            if ok >= max(3, limit // 2):
                s += 1
        except Exception:
            pass
        # return tuple for max(): (score, n, -off) to prefer earlier occurrence when tie
        return (s, n, -off)

    best = max(candidates, key=score)
    return best


def _maybe_following_rgb(b: bytes, after_off: int, n: int) -> Optional[Tuple[int, int]]:
    """Try to detect an RGB array right after the points block.

    For colors, CC writes:
      - 1 byte: hasColorsArray (bool)
      - if true: classID (8 or 4 bytes depending on version) then ccArray
        header (1 byte compCount=3, 4 bytes count=n) then raw RGB bytes (n*3)
    We don't need the classID value, only to skip it. We heuristically try 8
    then 4.
    Returns (offset, length_bytes) of the RGB payload if matched.
    """
    p = after_off
    if p >= len(b):
        return None
    has = b[p]
    p += 1
    if has not in (0, 1):
        return None
    if has == 0:
        return None

    # try skip 8-byte classID first
    for class_len in (8, 4):
        p2 = p + class_len
        if p2 + 1 + 4 > len(b):
            continue
        if b[p2] != 3:
            continue
        cnt = struct.unpack_from('<I', b, p2 + 1)[0]
        if cnt != n:
            continue
        payload_off = p2 + 1 + 4
        payload_len = n * 3  # ColorCompType is uint8
        if payload_off + payload_len <= len(b):
            return payload_off, payload_len
    return None


def load_points(path: str, peek: int = 0) -> Tuple[List[Tuple[float, float, float]], Optional[List[Tuple[int, int, int]]]]:
    flags, ver = _read_header(path)
    coord_is_double = (flags & 0x1) != 0
    with open(path, 'rb') as f:
        data = f.read()

    # Find the first plausible XYZ block
    off, n, stride = _scan_points(data, coord_is_double)
    j = off + 1 + 4
    points: List[Tuple[float, float, float]] = []
    if coord_is_double:
        for k in range(n):
            x = struct.unpack_from('<d', data, j)[0]
            y = struct.unpack_from('<d', data, j + 8)[0]
            z = struct.unpack_from('<d', data, j + 16)[0]
            points.append((float(x), float(y), float(z)))
            if peek and len(points) >= peek:
                break
            j += 24
    else:
        for k in range(n):
            x = struct.unpack_from('<f', data, j)[0]
            y = struct.unpack_from('<f', data, j + 4)[0]
            z = struct.unpack_from('<f', data, j + 8)[0]
            points.append((x, y, z))
            if peek and len(points) >= peek:
                break
            j += 12

    # Try to parse RGB right after the points block
    colors: Optional[List[Tuple[int, int, int]]] = None
    rgb_span = _maybe_following_rgb(data, off + 1 + 4 + n * stride, n)
    if rgb_span is not None:
        rgb_off, rgb_len = rgb_span
        colors = []
        for k in range(min(n, rgb_len // 3)):
            r = data[rgb_off + 3 * k + 0]
            g = data[rgb_off + 3 * k + 1]
            b = data[rgb_off + 3 * k + 2]
            colors.append((r, g, b))
            if peek and len(colors) >= peek:
                break

    return points, colors


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description='Read CloudCompare .bin v2 (best-effort)')
    ap.add_argument('path', help='A .bin file or a folder')
    ap.add_argument('--glob', default='*.bin', help='Glob pattern when path is a folder')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of files when scanning a folder')
    ap.add_argument('--peek', type=int, default=0, help='Print first N points (+RGB) instead of whole cloud')
    ap.add_argument('--summary', action='store_true', help='Print only summary for each file')
    args = ap.parse_args(list(argv))

    files: List[str] = []
    if os.path.isdir(args.path):
        files = sorted(_glob.glob(os.path.join(args.path, args.glob)))
        if args.limit:
            files = files[: args.limit]
    else:
        files = [args.path]

    for p in files:
        try:
            pts, cols = load_points(p, peek=args.peek)
            n = len(pts) if args.peek == 0 else len(pts)
            if args.summary:
                print(f'- {p}: {n} point sample{" +RGB" if cols else ""}')
            elif args.peek:
                print(f'- {p}: first {args.peek} points{" +RGB" if cols else ""}')
                for i, xyz in enumerate(pts[: args.peek]):
                    if cols and i < len(cols):
                        print(f'  {i}: ({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}) rgb={cols[i]}')
                    else:
                        print(f'  {i}: ({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f})')
            else:
                print(f'- {p}: loaded {len(pts)} points{" +RGB" if cols else ""}')
        except Exception as e:
            print(f'- {p}: ERROR {e}')

    return 0


if __name__ == '__main__':
    import sys

    raise SystemExit(main(sys.argv[1:]))
