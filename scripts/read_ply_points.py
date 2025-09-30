#!/usr/bin/env python3
"""
Minimal PLY reader for point clouds (x,y,z + optional RGB).

Supports ASCII and binary_little_endian PLY. Keeps dependencies to stdlib.

Usage examples:
  python scripts/read_ply_points.py data_ply/Group-10.ply --summary
  python scripts/read_ply_points.py data_ply --glob "*.ply" --limit 2
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PLY_DTYPE = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "int8": ("b", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


@dataclass
class PlyHeader:
    fmt: str  # 'ascii' or 'binary_little_endian'
    version: str
    vertex_count: int
    properties: List[Tuple[str, str]]  # list of (name, type)
    header_len: int  # byte offset of data start


def parse_header(fp) -> PlyHeader:
    line = fp.readline().decode("ascii", errors="strict").strip()
    if line != "ply":
        raise ValueError("Not a PLY file (missing 'ply' header)")
    fmt_line = fp.readline().decode("ascii", errors="strict").strip().split()
    if len(fmt_line) != 3 or fmt_line[0] != "format":
        raise ValueError("Invalid PLY format line")
    fmt, version = fmt_line[1], fmt_line[2]
    if fmt not in ("ascii", "binary_little_endian"):
        raise NotImplementedError(f"Unsupported PLY format: {fmt}")

    vertex_count = 0
    properties: List[Tuple[str, str]] = []
    header_bytes = len("ply\n") + len(" ".join(fmt_line)) + 1

    while True:
        pos = fp.tell()
        raw = fp.readline()
        if not raw:
            raise ValueError("Unexpected EOF in PLY header")
        header_bytes += len(raw)
        line = raw.decode("ascii", errors="strict").strip()
        if line == "end_header":
            break
        if line.startswith("comment"):
            continue
        if line.startswith("element"):
            parts = line.split()
            if parts[1] == "vertex":
                vertex_count = int(parts[2])
                properties = []
        elif line.startswith("property"):
            parts = line.split()
            if parts[1] == "list":
                # lists not supported for vertices in this minimal reader
                raise NotImplementedError("List properties not supported")
            else:
                ptype, pname = parts[1], parts[2]
                properties.append((pname, ptype))

    return PlyHeader(fmt=fmt, version=version, vertex_count=vertex_count, properties=properties, header_len=header_bytes)


def read_ascii_points(fp, hdr: PlyHeader):
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    colors: Optional[List[Tuple[int, int, int]]] = [] if any(n in dict(hdr.properties) for n in ("red", "green", "blue")) else None
    pnames = [n for n, _ in hdr.properties]
    for i in range(hdr.vertex_count):
        line = fp.readline().decode("ascii", errors="strict").strip()
        if not line:
            continue
        vals = line.split()
        m = dict(zip(pnames, vals))
        xs.append(float(m.get("x", 0.0)))
        ys.append(float(m.get("y", 0.0)))
        zs.append(float(m.get("z", 0.0)))
        if colors is not None:
            try:
                r = int(m.get("red"))
                g = int(m.get("green"))
                b = int(m.get("blue"))
                colors.append((r, g, b))
            except Exception:
                colors.append((0, 0, 0))
    return xs, ys, zs, colors


def read_binary_le_points(fp, hdr: PlyHeader):
    # Build struct format for one vertex
    fmt_codes = []
    offsets: Dict[str, int] = {}
    offset = 0
    for name, ptype in hdr.properties:
        code, size = PLY_DTYPE[ptype]
        fmt_codes.append(code)
        offsets[name] = offset
        offset += size
    rec_fmt = "<" + "".join(fmt_codes)
    rec_size = struct.calcsize(rec_fmt)
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    have_rgb = all(k in offsets for k in ("red", "green", "blue"))
    colors: Optional[List[Tuple[int, int, int]]] = [] if have_rgb else None
    for i in range(hdr.vertex_count):
        raw = fp.read(rec_size)
        if len(raw) != rec_size:
            raise ValueError("Unexpected EOF in binary PLY data")
        vals = struct.unpack(rec_fmt, raw)
        # Map by position
        pmap = {name: vals[idx] for idx, (name, _) in enumerate(hdr.properties)}
        xs.append(float(pmap.get("x", 0.0)))
        ys.append(float(pmap.get("y", 0.0)))
        zs.append(float(pmap.get("z", 0.0)))
        if colors is not None:
            colors.append((int(pmap["red"]), int(pmap["green"]), int(pmap["blue"])) )
    return xs, ys, zs, colors


def read_ply(path: str):
    with open(path, "rb") as fp:
        hdr = parse_header(fp)
        if hdr.fmt == "ascii":
            return read_ascii_points(fp, hdr)
        elif hdr.fmt == "binary_little_endian":
            return read_binary_le_points(fp, hdr)
        else:
            raise NotImplementedError(f"Unsupported PLY format: {hdr.fmt}")


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description="Read PLY points (x,y,z [+RGB])")
    ap.add_argument("path", help="PLY file or folder")
    ap.add_argument("--glob", default="*.ply", help="Glob when path is a folder")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files in a folder")
    ap.add_argument("--summary", action="store_true", help="Print only summary")
    args = ap.parse_args(list(argv))

    files: List[str] = []
    if os.path.isdir(args.path):
        files = sorted(_glob.glob(os.path.join(args.path, args.glob)))
        if args.limit:
            files = files[: args.limit]
    else:
        files = [args.path]

    for p in files:
        xs, ys, zs, colors = read_ply(p)
        n = len(xs)
        if args.summary:
            has_rgb = colors is not None
            print(f"- {p}: {n} points{' +RGB' if has_rgb else ''}")
        else:
            print(f"- {p}: first 5 points")
            for i in range(min(5, n)):
                if colors is not None:
                    print(f"  {i}: ({xs[i]:.6f}, {ys[i]:.6f}, {zs[i]:.6f}) rgb={colors[i]}")
                else:
                    print(f"  {i}: ({xs[i]:.6f}, {ys[i]:.6f}, {zs[i]:.6f})")

    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

