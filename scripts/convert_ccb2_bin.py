#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DF_POINT_COORDS_64_BITS = 1
DF_SCALAR_VAL_32_BITS = 2


def human_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    n = float(num)
    for u in units:
        if n < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(n)}{u}"
            return f"{n:.2f}{u}"
        n /= 1024
    return f"{n:.2f}PB"


def sanitize_filename(name: str, max_len: int = 120) -> str:
    s = name.strip()
    s = s.replace("\\", "_").replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    s = s.strip("._-")
    if not s:
        s = "unnamed"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")
    return s


def read_qstring(data: bytes, offset: int) -> tuple[str | None, int]:
    """Reads a Qt QDataStream QString (default big-endian)."""
    if offset + 4 > len(data):
        raise ValueError("out of bounds")
    (length,) = struct.unpack_from(">I", data, offset)
    offset += 4
    if length == 0xFFFFFFFF:
        return None, offset
    if length == 0:
        return "", offset
    if length % 2 != 0:
        raise ValueError("invalid QString length (odd)")
    end = offset + length
    if end > len(data):
        raise ValueError("QString out of bounds")
    raw = data[offset:end]
    s = raw.decode("utf-16-be", errors="strict")
    return s, end


def find_prev_qstring(data: bytes, start: int, window: int) -> str | None:
    lo = max(0, start - window)
    for off in range(start - 4, lo - 1, -1):
        try:
            s, _ = read_qstring(data, off)
        except Exception:
            continue
        if not s:
            continue
        # reject control chars (keep tabs/newlines if any)
        if any(ord(ch) < 9 or (13 < ord(ch) < 32) for ch in s):
            continue
        return s
    return None


def parse_ccb_header(data: bytes) -> dict[str, Any]:
    if len(data) < 8:
        raise ValueError("file too small")
    magic = data[:4]
    if magic[:3] != b"CCB":
        raise ValueError(f"not a CloudCompare BIN file (magic={magic!r})")
    try:
        flags = int(chr(magic[3]))
    except Exception as e:
        raise ValueError(f"invalid BIN header 4th byte: {magic[3:4]!r}") from e
    if flags < 0 or flags > 8:
        raise ValueError(f"invalid deserialization flags value: {flags}")
    (bin_version,) = struct.unpack_from("<I", data, 4)
    coord_is_double = bool(flags & DF_POINT_COORDS_64_BITS)
    scalar_is_float = bool(flags & DF_SCALAR_VAL_32_BITS)
    return {
        "magic": magic.decode("ascii", errors="replace"),
        "flags": flags,
        "bin_version": int(bin_version),
        "coord_is_double": coord_is_double,
        "scalar_is_float": scalar_is_float,
    }


@dataclass(frozen=True)
class CloudCandidate:
    offset: int
    point_count: int
    name: str | None


def try_extract_rgb_u8_after_points(
    data: bytes,
    *,
    coord_end: int,
    n: int,
    search_window: int,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Best-effort extraction of a packed per-point 1D payload after the XYZ block.

    Empirically, many CCB2 BINs in this repo store a 1-component array (byte `0x01`)
    with length==n shortly after the coordinate block. The payload is 4 bytes per point
    and decodes well as packed 24-bit RGB in little-endian (low bytes = R,G,B; alpha
    byte often 0). This function locates `0x01 + <u32 n>` within a small forward window
    and returns an `rgb_u8` array (Nx3) aligned with the points.
    """
    if coord_end < 0 or n <= 0:
        return None
    end = min(len(data), int(coord_end) + max(0, int(search_window)))
    if end <= coord_end:
        return None
    pattern = b"\x01" + struct.pack("<I", int(n))
    pos = data.find(pattern, int(coord_end), end)
    if pos == -1:
        return None
    payload_off = pos + len(pattern)
    payload_len = int(n) * 4
    if payload_off + payload_len > len(data):
        return None
    packed = np.frombuffer(data, dtype="<u4", count=int(n), offset=payload_off)
    rgb = np.empty((int(n), 3), dtype=np.uint8)
    rgb[:, 0] = packed & 255
    rgb[:, 1] = (packed >> 8) & 255
    rgb[:, 2] = (packed >> 16) & 255
    meta = {"payload_offset": int(payload_off), "delta_from_coord_end": int(pos - int(coord_end))}
    return rgb, meta


def iter_cloud_candidates(
    data: bytes,
    coord_dtype: np.dtype,
    min_points: int,
    name_window: int,
) -> Iterable[CloudCandidate]:
    coord_size = int(coord_dtype.itemsize)
    total_len = len(data)
    tiny_threshold = 1e-30 if coord_size == 4 else 1e-300

    pos = data.find(b"\x03", 0)  # componentCount == 3
    while pos != -1:
        if pos + 5 <= total_len:
            (n,) = struct.unpack_from("<I", data, pos + 1)
            if n >= min_points:
                payload = 3 * coord_size * n
                end = pos + 5 + payload
                if end <= total_len:
                    sample_vals = min(3000, n * 3)
                    arr = np.frombuffer(data, dtype=coord_dtype, count=sample_vals, offset=pos + 5)
                    # use float64 for stable stats (avoid float32 overflow in reductions)
                    arr64 = np.asarray(arr, dtype=np.float64)
                    if np.isfinite(arr64).all():
                        abs_arr = np.abs(arr64)
                        tiny_ratio = float(np.mean(abs_arr < tiny_threshold))
                        max_abs = float(abs_arr.max(initial=0.0))
                        std = float(arr64.std())
                        # Heuristics: point coords shouldn't be almost all subnormal nor explode
                        if math.isfinite(std) and std > 1e-6 and tiny_ratio < 0.8 and max_abs < 1e6:
                            name = find_prev_qstring(data, pos, name_window)
                            if name:
                                name = name.strip()
                            yield CloudCandidate(offset=pos, point_count=int(n), name=name)
        pos = data.find(b"\x03", pos + 1)


def extract_points(data: bytes, offset: int, n: int, coord_dtype: np.dtype) -> np.ndarray:
    coord_size = int(coord_dtype.itemsize)
    expected = offset + 5 + (n * 3 * coord_size)
    if expected > len(data):
        raise ValueError("point array out of bounds")
    pts = np.frombuffer(data, dtype=coord_dtype, count=n * 3, offset=offset + 5).reshape(n, 3)
    # normalize to float32 for downstream
    return np.asarray(pts, dtype=np.float32, order="C")


def write_ply_points(path: Path, points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.asarray(points, dtype="<f4", order="C").tobytes())


def write_ply_points_rgb(path: Path, points: np.ndarray, rgb_u8: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if rgb_u8.ndim != 2 or rgb_u8.shape[1] != 3 or rgb_u8.shape[0] != points.shape[0]:
        raise ValueError("rgb_u8 must be Nx3 and aligned with points")
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    verts = np.empty(
        (int(n),),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pts = np.asarray(points, dtype="<f4", order="C")
    rgb = np.asarray(rgb_u8, dtype=np.uint8, order="C")
    verts["x"] = pts[:, 0]
    verts["y"] = pts[:, 1]
    verts["z"] = pts[:, 2]
    verts["red"] = rgb[:, 0]
    verts["green"] = rgb[:, 1]
    verts["blue"] = rgb[:, 2]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())


def write_obj_points(path: Path, points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in points:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")


def parse_formats(value: str) -> list[str]:
    fmts = [v.strip().lower() for v in value.split(",") if v.strip()]
    allowed = {"npz", "ply", "obj"}
    unknown = sorted(set(fmts) - allowed)
    if unknown:
        raise ValueError(f"unknown formats: {unknown} (allowed: {sorted(allowed)})")
    # stable order
    out = []
    for k in ["npz", "ply", "obj"]:
        if k in fmts:
            out.append(k)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract point clouds from CloudCompare CCB2 .bin files.")
    parser.add_argument("--input", default="raw", help="Input directory (default: raw)")
    parser.add_argument("--output", default="converted/raw", help="Output directory (default: converted/raw)")
    parser.add_argument(
        "--formats",
        default="npz,ply",
        help="Comma-separated output formats: npz,ply,obj (default: npz,ply)",
    )
    parser.add_argument(
        "--select",
        choices=["all", "largest"],
        default="all",
        help="Export all detected clouds or only the largest one per .bin (default: all)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=500,
        help="Minimum points for a cloud candidate (default: 500)",
    )
    parser.add_argument(
        "--name-window",
        type=int,
        default=8192,
        help="Backward search window (bytes) to find a preceding Qt QString name (default: 8192)",
    )
    parser.add_argument(
        "--export-rgb",
        action="store_true",
        help="Try to extract per-point packed RGB (if present) and export it in NPZ (and in PLY if enabled).",
    )
    parser.add_argument(
        "--rgb-window",
        type=int,
        default=8192,
        help="Forward search window (bytes) after XYZ block to find the packed RGB header (default: 8192).",
    )
    parser.add_argument(
        "--include-regex",
        default="",
        help="Only export clouds whose extracted name matches this regex (optional)",
    )
    parser.add_argument(
        "--exclude-regex",
        default="",
        help="Skip clouds whose extracted name matches this regex (optional)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N .bin files (0=all)")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    formats = parse_formats(args.formats)

    include_re = re.compile(args.include_regex) if args.include_regex else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    bin_files = sorted(input_dir.rglob("*.bin"))
    if args.limit and args.limit > 0:
        bin_files = bin_files[: args.limit]

    if not bin_files:
        raise SystemExit(f"no .bin files under: {input_dir}")

    manifest: list[dict[str, Any]] = []
    total_out_bytes = 0
    total_clouds = 0

    for i_file, bin_path in enumerate(bin_files, start=1):
        rel = bin_path.relative_to(input_dir)
        out_case_dir = output_dir / rel.parent / bin_path.stem

        data = bin_path.read_bytes()
        hdr = parse_ccb_header(data)
        coord_dtype = np.dtype("<f8") if hdr["coord_is_double"] else np.dtype("<f4")

        clouds = list(
            iter_cloud_candidates(
                data=data,
                coord_dtype=coord_dtype,
                min_points=args.min_points,
                name_window=args.name_window,
            )
        )
        # de-dup by offset (defensive)
        uniq: dict[int, CloudCandidate] = {}
        for c in clouds:
            uniq.setdefault(c.offset, c)
        clouds = list(uniq.values())
        clouds.sort(key=lambda c: c.point_count, reverse=True)

        if args.select == "largest":
            clouds = clouds[:1]

        exported: list[dict[str, Any]] = []
        for idx, c in enumerate(clouds):
            name = c.name or f"cloud_{idx}"
            name = name.strip()
            if name.startswith(("$", "*")):
                continue
            if include_re and not include_re.search(name):
                continue
            if exclude_re and exclude_re.search(name):
                continue

            safe_name = sanitize_filename(name)
            points = extract_points(data, c.offset, c.point_count, coord_dtype)
            coord_end = c.offset + 5 + (c.point_count * 3 * int(coord_dtype.itemsize))
            rgb_pack = (
                try_extract_rgb_u8_after_points(
                    data,
                    coord_end=int(coord_end),
                    n=int(c.point_count),
                    search_window=int(args.rgb_window),
                )
                if args.export_rgb
                else None
            )
            rgb_u8: np.ndarray | None = rgb_pack[0] if rgb_pack is not None else None
            rgb_meta: dict[str, Any] | None = rgb_pack[1] if rgb_pack is not None else None

            base = f"{idx:02d}__{safe_name}"
            rec: dict[str, Any] = {
                "name": name,
                "safe_name": safe_name,
                "points": int(points.shape[0]),
                "offset": int(c.offset),
                "outputs": {},
            }
            if rgb_u8 is not None:
                rec["rgb"] = {"dtype": "u8", "shape": [int(rgb_u8.shape[0]), int(rgb_u8.shape[1])], **(rgb_meta or {})}

            if "npz" in formats:
                out_npz = out_case_dir / f"{base}.npz"
                out_npz.parent.mkdir(parents=True, exist_ok=True)
                arrays: dict[str, Any] = {"points": points}
                if rgb_u8 is not None:
                    arrays["rgb"] = rgb_u8
                np.savez_compressed(out_npz, **arrays)
                rec["outputs"]["npz"] = str(out_npz.relative_to(output_dir))
                total_out_bytes += out_npz.stat().st_size

            if "ply" in formats:
                out_ply = out_case_dir / f"{base}.ply"
                if rgb_u8 is not None and args.export_rgb:
                    write_ply_points_rgb(out_ply, points, rgb_u8)
                else:
                    write_ply_points(out_ply, points)
                rec["outputs"]["ply"] = str(out_ply.relative_to(output_dir))
                total_out_bytes += out_ply.stat().st_size

            if "obj" in formats:
                out_obj = out_case_dir / f"{base}.obj"
                write_obj_points(out_obj, points)
                rec["outputs"]["obj"] = str(out_obj.relative_to(output_dir))
                total_out_bytes += out_obj.stat().st_size

            exported.append(rec)

        manifest.append(
            {
                "input": str(rel),
                "input_bytes": len(data),
                "header": hdr,
                "exported_clouds": exported,
            }
        )
        total_clouds += len(exported)

        if i_file % 10 == 0 or i_file == len(bin_files):
            print(
                f"[{i_file}/{len(bin_files)}] {rel} -> clouds={len(exported)} "
                f"(total_clouds={total_clouds}, out~={human_bytes(total_out_bytes)})"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
