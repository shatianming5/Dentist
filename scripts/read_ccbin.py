#!/usr/bin/env python3
"""
Lightweight inspector for CloudCompare .bin files.

- Verifies the magic header.
- Extracts a few UTF-16LE strings from the header to confirm content.
- Prints a concise summary you can build on.

Note: This is not a full parser; CloudCompare's .bin format is complex.
Recommended workflow is to convert .bin to a standard format (e.g. PLY)
with CloudCompare CLI, then read point clouds with your preferred library.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Tuple


MAGIC = b"CCB2"  # CloudCompare BIN v2 signature observed at file start


def is_ccbin(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        return head == MAGIC
    except OSError:
        return False


def extract_utf16le_strings(data: bytes, min_len: int = 3) -> List[str]:
    """Extract readable UTF-16LE strings from a small byte window.

    This is heuristic: scans for sequences of the form c"\x00"c"\x00"...
    """
    out: List[str] = []
    i = 0
    n = len(data)
    while i + 1 < n:
        start = i
        buf = bytearray()
        while i + 1 < n:
            lo, hi = data[i], data[i + 1]
            # typical UTF-16LE for ASCII subset => hi == 0, printable lo
            if 32 <= lo <= 126 and hi == 0:
                buf.append(lo)
                i += 2
            else:
                break
        if len(buf) >= min_len:
            try:
                s = buf.decode("ascii", errors="ignore")
                if s:
                    out.append(s)
            except Exception:
                pass
        i = max(i + 2, start + 2)
    return out


def peek_strings(path: str, window: int = 65536) -> List[str]:
    with open(path, "rb") as f:
        blob = f.read(window)
    return extract_utf16le_strings(blob)


def summarize_bin(path: str) -> Tuple[bool, List[str], int]:
    ok = is_ccbin(path)
    strs: List[str] = []
    size = 0
    try:
        size = os.path.getsize(path)
    except OSError:
        pass
    if ok:
        strs = peek_strings(path)
    return ok, strs, size


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description="Probe CloudCompare .bin files")
    ap.add_argument("paths", nargs="*", help="Files or folders to scan (default: ./data)")
    args = ap.parse_args(list(argv))

    targets: List[str] = []
    if not args.paths:
        base = os.path.join(os.getcwd(), "data")
        if os.path.isdir(base):
            for name in os.listdir(base):
                p = os.path.join(base, name)
                if os.path.isfile(p) and p.lower().endswith(".bin"):
                    targets.append(p)
        else:
            print(f"No data folder at {base}")
            return 1
    else:
        for p in args.paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        if fn.lower().endswith(".bin"):
                            targets.append(os.path.join(root, fn))
            elif os.path.isfile(p):
                targets.append(p)

    if not targets:
        print("No .bin files found")
        return 1

    for p in sorted(targets):
        ok, strs, size = summarize_bin(p)
        tag = "CloudCompare BIN v2" if ok else "Unknown .bin"
        print(f"- {p} [{size} bytes] -> {tag}")
        if ok and strs:
            # Show a few meaningful strings if present
            hits = [s for s in strs if any(k in s for k in ("Group", "Mesh", "sampled"))]
            if hits:
                print("  header strings:")
                for s in hits[:6]:
                    print(f"    • {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

