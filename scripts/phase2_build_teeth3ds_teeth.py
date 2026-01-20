#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_int_seed(text: str, base_seed: int) -> int:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ int(base_seed)) & 0xFFFFFFFF


def parse_obj_vertices(obj_path: Path, *, stop_at_faces: bool = True) -> np.ndarray:
    verts: list[np.ndarray] = []
    seen_v = False
    with obj_path.open("rb") as f:
        for line in f:
            if line.startswith(b"v "):
                seen_v = True
                xyz = np.fromstring(line[2:], sep=" ", count=3, dtype=np.float32)
                if xyz.shape[0] == 3 and np.isfinite(xyz).all():
                    verts.append(xyz)
                continue
            if stop_at_faces and seen_v and line.startswith(b"f "):
                break
    if not verts:
        raise ValueError(f"no vertices parsed from: {obj_path}")
    return np.vstack(verts).astype(np.float32, copy=False)


def mode_int(values: np.ndarray) -> int | None:
    if values.size == 0:
        return None
    vals = values.astype(np.int32, copy=False)
    mn = int(vals.min(initial=0))
    mx = int(vals.max(initial=0))
    if mx < 0:
        return None
    # labels are small ints (0..48), bincount is fast.
    if mn >= 0 and mx <= 1024:
        c = np.bincount(vals.clip(min=0))
        if c.size == 0:
            return None
        return int(c.argmax())
    # fallback
    counter: Counter[int] = Counter(int(x) for x in vals.tolist())
    return int(counter.most_common(1)[0][0]) if counter else None


def normalize_points(points: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = points - centroid
    if mode == "bbox_diag":
        mn = centered.min(axis=0)
        mx = centered.max(axis=0)
        scale = float(np.linalg.norm((mx - mn).astype(np.float64)))
    elif mode == "max_norm":
        norms = np.linalg.norm(centered.astype(np.float64), axis=1)
        scale = float(np.max(norms)) if norms.size else 1.0
    else:
        raise ValueError(f"unknown normalize mode: {mode}")
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    normalized = (centered / scale).astype(np.float32, copy=False)
    return normalized, centroid, scale


def pca_align(points: np.ndarray, *, align_globalz: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (aligned_points, R) where aligned = points @ R, and R is 3x3 orthonormal."""
    x = points.astype(np.float64, copy=False)
    cov = (x.T @ x) / max(1, x.shape[0])
    w, v = np.linalg.eigh(cov)  # ascending eigenvalues
    order = np.argsort(w)[::-1]
    r = v[:, order].astype(np.float32, copy=False)  # columns are axes

    r = r.copy()
    if bool(align_globalz):
        # Reorder axes to align with global Z/X as much as possible for stability across symmetric shapes.
        jz = int(np.argmax(np.abs(r[2, :])))
        remaining = [i for i in range(3) if i != jz]
        jx = remaining[int(np.argmax(np.abs(r[0, remaining])))]
        jy = [i for i in remaining if i != jx][0]
        r = r[:, [jx, jy, jz]]
        if float(r[2, 2]) < 0:
            r[:, 2] = -r[:, 2]
        if float(r[0, 0]) < 0:
            r[:, 0] = -r[:, 0]
    else:
        # Deterministic sign: make largest-magnitude component of each axis positive.
        for i in range(3):
            axis = r[:, i]
            j = int(np.argmax(np.abs(axis)))
            if axis[j] < 0:
                r[:, i] = -axis

    # Ensure right-handed coordinate system.
    if float(np.linalg.det(r.astype(np.float64))) < 0:
        r[:, 1] = -r[:, 1]

    aligned = (points @ r).astype(np.float32, copy=False)
    return aligned, r


def downsample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    m = int(points.shape[0])
    if m == n:
        return points
    replace = m < n
    idx = rng.choice(m, size=int(n), replace=replace)
    return points[idx]


@dataclass(frozen=True)
class BuildConfig:
    seed: int
    target_points: int
    min_points_per_tooth: int
    normalize: str
    pca: bool
    pca_align_globalz: bool
    limit_cases: int


def build_split_map(splits_path: Path, *, mode: str) -> dict[str, str]:
    obj = read_json(splits_path)
    derived = obj.get("derived") or {}
    patient = (obj.get("patient") or {}).get("derived") or {}

    def jaw_level_map() -> dict[str, str]:
        m: dict[str, str] = {}
        for jaw in ["upper", "lower"]:
            jaw_splits = derived.get(jaw) or {}
            for split_name, items in jaw_splits.items():
                for item in items:
                    m[str(item)] = str(split_name)
        return m

    split_mode = str(mode or "").strip().lower()
    if split_mode not in {"patient", "jaw"}:
        raise ValueError(f"unknown splits mode: {mode} (allowed: patient,jaw)")

    if split_mode == "patient":
        if not patient:
            # Fall back to jaw-level if patient-level splits are not present.
            return jaw_level_map()
        m: dict[str, str] = {}
        for split_name, ids in patient.items():
            for pid in ids:
                for jaw in ["upper", "lower"]:
                    m[f"{pid}_{jaw}"] = str(split_name)
        return m
    if split_mode == "jaw":
        return jaw_level_map()
    raise AssertionError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2: build Teeth3DS single-tooth point dataset.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Repo root (default: .)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/teeth3ds"))
    ap.add_argument("--splits", type=Path, default=Path("metadata/splits_teeth3ds.json"))
    ap.add_argument(
        "--splits-mode",
        choices=["patient", "jaw"],
        default="patient",
        help="Which split definition to use. 'patient' avoids cross-jaw leakage if available.",
    )
    ap.add_argument("--out", type=Path, default=Path("processed/teeth3ds_teeth/v1"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--target-points", type=int, default=1024)
    ap.add_argument("--min-points-per-tooth", type=int, default=500)
    ap.add_argument("--normalize", choices=["bbox_diag", "max_norm"], default="bbox_diag")
    ap.add_argument("--no-pca", action="store_true", help="Disable PCA alignment.")
    ap.add_argument(
        "--pca-align-globalz",
        action="store_true",
        help="When PCA is enabled, reorder axes to align with global Z/X for more stable orientation.",
    )
    ap.add_argument("--jaws", default="upper,lower", help="Comma-separated jaws to process: upper,lower.")
    ap.add_argument("--limit-cases", type=int, default=0, help="Only process first N cases per jaw (0=all).")
    ap.add_argument(
        "--index-mode",
        choices=["write", "append"],
        default="write",
        help="Write (overwrite) or append to index.jsonl (useful for per-jaw runs).",
    )
    ap.add_argument(
        "--skip-existing-npz",
        action="store_true",
        help="Do not overwrite existing sample .npz files (still writes index rows).",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    data_dir = (root / args.data_dir).resolve()
    splits_path = (root / args.splits).resolve()
    out_root = (root / args.out).resolve()
    out_teeth = out_root / "teeth"
    out_teeth.mkdir(parents=True, exist_ok=True)

    cfg = BuildConfig(
        seed=int(args.seed),
        target_points=int(args.target_points),
        min_points_per_tooth=int(args.min_points_per_tooth),
        normalize=str(args.normalize),
        pca=not bool(args.no_pca),
        pca_align_globalz=bool(args.pca_align_globalz),
        limit_cases=int(args.limit_cases),
    )

    if not data_dir.exists():
        raise SystemExit(f"Missing Teeth3DS data dir: {data_dir}")
    if not splits_path.exists():
        raise SystemExit(f"Missing splits: {splits_path} (run scripts/phase0_freeze.py first)")

    split_of = build_split_map(splits_path, mode=str(args.splits_mode))

    generated_at = utc_now_iso()
    skipped_cases: list[dict[str, Any]] = []
    fdi_set: set[int] = set()
    stats = {
        "cases_total": 0,
        "cases_with_json": 0,
        "teeth_total": 0,
        "teeth_written": 0,
        "teeth_skipped_small": 0,
        "teeth_skipped_nolabel": 0,
    }

    selected_jaws = [s.strip() for s in str(args.jaws).split(",") if s.strip()]
    unknown = sorted(set(selected_jaws) - {"upper", "lower"})
    if unknown:
        raise SystemExit(f"Unknown jaws: {unknown} (allowed: upper,lower)")

    index_path = out_root / "index.jsonl"
    if args.index_mode == "write" and index_path.exists():
        index_path.unlink()
    index_mode = "a" if args.index_mode == "append" else "w"

    with index_path.open(index_mode, encoding="utf-8") as index_f:
        for jaw in selected_jaws:
            jaw_dir = data_dir / jaw
            if not jaw_dir.exists():
                continue
            case_dirs = sorted([p for p in jaw_dir.iterdir() if p.is_dir()])
            if cfg.limit_cases > 0:
                case_dirs = case_dirs[: cfg.limit_cases]

            for i_case, case_dir in enumerate(case_dirs, start=1):
                id_ = case_dir.name
                case_key = f"{id_}_{jaw}"
                stats["cases_total"] += 1

                obj_path = case_dir / f"{id_}_{jaw}.obj"
                json_path = case_dir / f"{id_}_{jaw}.json"
                if not obj_path.exists() or not json_path.exists():
                    skipped_cases.append(
                        {
                            "case_key": case_key,
                            "missing_obj": not obj_path.exists(),
                            "missing_json": not json_path.exists(),
                        }
                    )
                    continue

                stats["cases_with_json"] += 1
                try:
                    vertices = parse_obj_vertices(obj_path, stop_at_faces=True)
                    seg = read_json(json_path)
                    labels = np.asarray(seg.get("labels", []), dtype=np.int32)
                    instances = np.asarray(seg.get("instances", []), dtype=np.int32)
                    if labels.shape[0] != vertices.shape[0] or instances.shape[0] != vertices.shape[0]:
                        raise ValueError(
                            f"labels/instances length mismatch: v={vertices.shape[0]} labels={labels.shape[0]} inst={instances.shape[0]}"
                        )
                except Exception as e:
                    skipped_cases.append({"case_key": case_key, "error": str(e)})
                    continue

                split = split_of.get(case_key, "unknown")

                inst_ids = np.unique(instances)
                inst_ids = inst_ids[inst_ids > 0]

                for inst_id in inst_ids.tolist():
                    idx = np.nonzero(instances == int(inst_id))[0]
                    if idx.size == 0:
                        continue
                    pts_raw = vertices[idx].astype(np.float32, copy=False)
                    stats["teeth_total"] += 1
                    if int(pts_raw.shape[0]) < cfg.min_points_per_tooth:
                        stats["teeth_skipped_small"] += 1
                        continue

                    tooth_labels = labels[idx]
                    tooth_labels = tooth_labels[tooth_labels > 0]
                    fdi = mode_int(tooth_labels)
                    if fdi is None or int(fdi) <= 0:
                        stats["teeth_skipped_nolabel"] += 1
                        continue

                    pts_norm, centroid, scale = normalize_points(pts_raw, cfg.normalize)
                    if cfg.pca:
                        pts_norm, r = pca_align(pts_norm, align_globalz=cfg.pca_align_globalz)
                    else:
                        r = np.eye(3, dtype=np.float32)

                    seed = stable_int_seed(f"{case_key}__{inst_id}__{fdi}", cfg.seed)
                    rng = np.random.default_rng(seed)
                    pts_out = downsample(pts_norm, cfg.target_points, rng)

                    tooth_key = f"inst{int(inst_id):02d}_fdi{int(fdi)}"
                    rel_npz = Path("teeth") / case_key / f"{tooth_key}.npz"
                    out_path = out_root / rel_npz
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if not (args.skip_existing_npz and out_path.exists()):
                        np.savez_compressed(out_path, points=pts_out.astype(np.float32, copy=False))

                    row = {
                        "case_key": case_key,
                        "id_patient": seg.get("id_patient") or id_,
                        "jaw": jaw,
                        "split": split,
                        "split_mode": str(args.splits_mode),
                        "instance_id": int(inst_id),
                        "fdi": int(fdi),
                        "n_points_raw": int(pts_raw.shape[0]),
                        "target_points": int(cfg.target_points),
                        "normalize": cfg.normalize,
                        "centroid": [float(x) for x in centroid.tolist()],
                        "scale": float(scale),
                        "pca": bool(cfg.pca),
                        "pca_align_globalz": bool(cfg.pca_align_globalz),
                        "R": [[float(x) for x in row_] for row_ in r.tolist()],
                        "sample_npz": str(rel_npz),
                        "source_obj": str(obj_path.relative_to(root)),
                        "source_json": str(json_path.relative_to(root)),
                    }
                    index_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["teeth_written"] += 1
                    if stats["teeth_written"] % 200 == 0:
                        index_f.flush()
                    fdi_set.add(int(fdi))

                if i_case % 50 == 0:
                    print(
                        f"[{jaw}] processed {i_case}/{len(case_dirs)} cases, teeth_written={stats['teeth_written']}",
                        flush=True,
                    )

    # label map for AE training (optional). If we appended, union with existing file.
    fdi_path = out_root / "fdi_values.json"
    if fdi_path.exists():
        try:
            for x in read_json(fdi_path) or []:
                fdi_set.add(int(x))
        except Exception:
            pass
    write_json(out_root / "fdi_values.json", sorted(fdi_set))

    report_lines: list[str] = []
    report_lines.append(f"# teeth3ds_teeth {out_root.name} 构建报告")
    report_lines.append(f"- 生成时间：{generated_at}")
    report_lines.append(f"- data_dir：`{data_dir.relative_to(root)}`")
    report_lines.append(f"- splits：`{splits_path.relative_to(root)}`")
    report_lines.append(f"- splits_mode：{args.splits_mode}")
    report_lines.append(f"- 输出：`{out_root.relative_to(root)}`")
    report_lines.append("")
    report_lines.append("## 参数")
    report_lines.append(f"- seed：{cfg.seed}")
    report_lines.append(f"- target_points：{cfg.target_points}")
    report_lines.append(f"- min_points_per_tooth：{cfg.min_points_per_tooth}")
    report_lines.append(f"- normalize：{cfg.normalize}")
    report_lines.append(f"- pca：{cfg.pca}")
    report_lines.append(f"- pca_align_globalz：{cfg.pca_align_globalz}")
    report_lines.append(f"- limit_cases：{cfg.limit_cases}")
    report_lines.append("")
    report_lines.append("## 统计")
    for k in ["cases_total", "cases_with_json", "teeth_total", "teeth_written", "teeth_skipped_small", "teeth_skipped_nolabel"]:
        report_lines.append(f"- {k}: {stats[k]}")
    report_lines.append("")
    if skipped_cases:
        report_lines.append("## 跳过的 case（前 50）")
        for item in skipped_cases[:50]:
            report_lines.append(f"- {item}")
        if len(skipped_cases) > 50:
            report_lines.append(f"- ... (还有 {len(skipped_cases) - 50} 个)")
        report_lines.append("")

    (out_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {index_path}")
    print(f"[OK] teeth_written: {stats['teeth_written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
