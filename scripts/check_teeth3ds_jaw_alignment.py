#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def parse_obj_centroid(obj_path: Path, *, stop_at_faces: bool = True) -> np.ndarray:
    s = np.zeros((3,), dtype=np.float64)
    n = 0
    seen_v = False
    with obj_path.open("rb") as f:
        for line in f:
            if line.startswith(b"v "):
                seen_v = True
                xyz = np.fromstring(line[2:], sep=" ", count=3, dtype=np.float64)
                if xyz.shape[0] == 3 and np.isfinite(xyz).all():
                    s += xyz
                    n += 1
                continue
            if stop_at_faces and seen_v and line.startswith(b"f "):
                break
    if n <= 0:
        raise ValueError(f"no vertices parsed from: {obj_path}")
    return (s / float(n)).astype(np.float64, copy=False)


def fmt_summary(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"count": 0}
    xs2 = sorted(xs)
    return {
        "count": len(xs2),
        "min": float(xs2[0]),
        "p25": float(xs2[int(0.25 * (len(xs2) - 1))]),
        "median": float(statistics.median(xs2)),
        "p75": float(xs2[int(0.75 * (len(xs2) - 1))]),
        "p95": float(xs2[int(0.95 * (len(xs2) - 1))]),
        "max": float(xs2[-1]),
        "mean": float(sum(xs2) / len(xs2)),
    }


@dataclass(frozen=True)
class Row:
    case_id: str
    dist: float


def main() -> int:
    ap = argparse.ArgumentParser(description="Check whether Teeth3DS upper/lower jaws share a consistent coordinate frame.")
    ap.add_argument("--teeth3ds-dir", type=Path, default=Path("data/teeth3ds"))
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    ap.add_argument("--topk", type=int, default=30, help="Show top-K largest centroid diffs.")
    args = ap.parse_args()

    root = args.teeth3ds_dir.resolve()
    upper_dir = root / "upper"
    lower_dir = root / "lower"
    if not upper_dir.exists() or not lower_dir.exists():
        raise SystemExit(f"Missing teeth3ds dir: {root} (expected upper/ and lower/)")

    upper_ids = sorted([p.name for p in upper_dir.iterdir() if p.is_dir()])
    lower_ids = sorted([p.name for p in lower_dir.iterdir() if p.is_dir()])
    common = sorted(set(upper_ids) & set(lower_ids))

    rows: list[Row] = []
    dists: list[float] = []
    for cid in common:
        u = upper_dir / cid / f"{cid}_upper.obj"
        l = lower_dir / cid / f"{cid}_lower.obj"
        if not u.exists() or not l.exists():
            continue
        try:
            cu = parse_obj_centroid(u)
            cl = parse_obj_centroid(l)
        except Exception:
            continue
        d = float(np.linalg.norm((cl - cu).astype(np.float64)))
        if not math.isfinite(d):
            continue
        dists.append(d)
        rows.append(Row(case_id=cid, dist=d))

    rows.sort(key=lambda r: r.dist, reverse=True)
    summary = {
        "total_common_cases": len(common),
        "evaluated": len(rows),
        "centroid_diff_norm": fmt_summary(dists),
        "topk": [{"case_id": r.case_id, "centroid_diff_norm": r.dist} for r in rows[: int(args.topk)]],
    }

    if args.out is not None:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote: {out_path}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
