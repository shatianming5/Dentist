#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_int_seed(text: str, base_seed: int) -> int:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ int(base_seed)) & 0xFFFFFFFF


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


def pca_align(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points.astype(np.float64, copy=False)
    cov = (x.T @ x) / max(1, x.shape[0])
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    r = v[:, order].astype(np.float32, copy=False)

    r = r.copy()
    for i in range(3):
        axis = r[:, i]
        j = int(np.argmax(np.abs(axis)))
        if axis[j] < 0:
            r[:, i] = -axis
    if float(np.linalg.det(r.astype(np.float64))) < 0:
        r[:, 2] = -r[:, 2]

    aligned = (points @ r).astype(np.float32, copy=False)
    return aligned, r


def downsample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    m = int(points.shape[0])
    if m == n:
        return points
    replace = m < n
    idx = rng.choice(m, size=int(n), replace=replace)
    return points[idx]


def kmeans_1d_two_means(z: np.ndarray, rng: np.random.Generator, iters: int = 20, sample: int = 50_000) -> tuple[float, float]:
    z = z.astype(np.float32, copy=False)
    if z.size == 0:
        return 0.0, 0.0
    if z.size > sample:
        idx = rng.choice(z.size, size=sample, replace=False)
        zz = z[idx]
    else:
        zz = z
    m1 = float(zz.min(initial=0.0))
    m2 = float(zz.max(initial=0.0))
    if not math.isfinite(m1) or not math.isfinite(m2):
        return 0.0, 0.0
    if abs(m2 - m1) < 1e-6:
        return m1, m2
    for _ in range(int(iters)):
        d1 = np.abs(zz - m1)
        d2 = np.abs(zz - m2)
        mask = d1 < d2
        if mask.sum() == 0 or (~mask).sum() == 0:
            break
        m1_new = float(zz[mask].mean())
        m2_new = float(zz[~mask].mean())
        if abs(m1_new - m1) < 1e-6 and abs(m2_new - m2) < 1e-6:
            break
        m1, m2 = m1_new, m2_new
    return (m1, m2) if m1 <= m2 else (m2, m1)


@dataclass(frozen=True)
class BuildConfig:
    seed: int
    prep_points: int
    target_points: int
    margin_points: int
    margin_band: float
    cut_q: float
    margin_mode: str
    margin_quantile: float
    margin_max_dist: float
    prep_mode: str
    patch_radius_mult: float
    overlap_voxel: float
    normalize: str
    pca: bool
    occlusion_points: int


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4: build raw synthetic prep->target dataset with opposing jaw extracted from raw scene.")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--raw-cls-index", type=Path, default=Path("processed/raw_cls/v6/index.jsonl"))
    ap.add_argument("--manifest", type=Path, default=Path("converted/raw/manifest_with_labels.json"))
    ap.add_argument("--converted-root", type=Path, default=Path("converted/raw"))
    ap.add_argument("--out", type=Path, default=Path("processed/raw_prep2target/v1"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--prep-points", type=int, default=2048, help="Number of points for prep_points (input).")
    ap.add_argument("--target-points", type=int, default=512, help="Number of points for target_points (output).")
    ap.add_argument("--margin-points", type=int, default=64)
    ap.add_argument("--margin-band", type=float, default=0.02)
    ap.add_argument("--cut-q", type=float, default=0.7)
    ap.add_argument(
        "--margin-mode",
        choices=["synthetic_cut", "contact"],
        default="contact",
        help="How to build margin points: synthetic cut-plane band, or nearest-contact band (recommended for scene_patch).",
    )
    ap.add_argument(
        "--margin-quantile",
        type=float,
        default=0.05,
        help="For margin-mode=contact: take closest q fraction of target points as margin candidates.",
    )
    ap.add_argument(
        "--margin-max-dist",
        type=float,
        default=2.0,
        help="For margin-mode=contact: clamp distance threshold (mm) to avoid extreme outliers.",
    )
    ap.add_argument(
        "--prep-mode",
        choices=["synthetic_cut", "scene_patch"],
        default="scene_patch",
        help="How to build prep input: synthetic from target, or local patch from raw scene (recommended).",
    )
    ap.add_argument(
        "--patch-radius-mult",
        type=float,
        default=0.5,
        help="For prep-mode=scene_patch: radius = patch_radius_mult * target_bbox_diag.",
    )
    ap.add_argument(
        "--overlap-voxel",
        type=float,
        default=0.05,
        help="For prep-mode=scene_patch: voxel size used to drop prep points overlapping target (in raw coords).",
    )
    ap.add_argument("--normalize", choices=["bbox_diag", "max_norm"], default="bbox_diag")
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--occlusion-points", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = args.root.resolve()
    raw_cls_index = (root / args.raw_cls_index).resolve()
    manifest_path = (root / args.manifest).resolve()
    converted_root = (root / args.converted_root).resolve()
    out_root = (root / args.out).resolve()
    out_samples = out_root / "samples"
    out_samples.mkdir(parents=True, exist_ok=True)

    cfg = BuildConfig(
        seed=int(args.seed),
        prep_points=int(args.prep_points),
        target_points=int(args.target_points),
        margin_points=int(args.margin_points),
        margin_band=float(args.margin_band),
        cut_q=float(args.cut_q),
        margin_mode=str(args.margin_mode),
        margin_quantile=float(args.margin_quantile),
        margin_max_dist=float(args.margin_max_dist),
        prep_mode=str(args.prep_mode),
        patch_radius_mult=float(args.patch_radius_mult),
        overlap_voxel=float(args.overlap_voxel),
        normalize=str(args.normalize),
        pca=not bool(args.no_pca),
        occlusion_points=int(args.occlusion_points),
    )

    if not raw_cls_index.exists():
        raise SystemExit(f"Missing raw_cls index: {raw_cls_index}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path} (run scripts/convert_ccb2_bin.py first)")
    if not converted_root.exists():
        raise SystemExit(f"Missing converted root: {converted_root}")

    raw_rows = read_jsonl(raw_cls_index)
    if args.limit and int(args.limit) > 0:
        raw_rows = raw_rows[: int(args.limit)]

    manifest = read_json(manifest_path)
    mani_map: dict[str, dict[str, Any]] = {}
    for item in manifest:
        inp = str(item.get("input") or "")
        if inp:
            mani_map[inp] = item

    built_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    stats = {
        "cases_total": 0,
        "cases_written": 0,
        "cases_missing_manifest": 0,
        "cases_missing_npz": 0,
        "cases_missing_opposing": 0,
    }

    for i, r in enumerate(raw_rows, start=1):
        case_key = str(r.get("case_key") or "")
        if not case_key:
            continue
        stats["cases_total"] += 1

        item = mani_map.get(case_key)
        if item is None:
            stats["cases_missing_manifest"] += 1
            skipped.append({"case_key": case_key, "reason": "missing_manifest"})
            continue

        exported = item.get("exported_clouds") or []
        if not exported:
            skipped.append({"case_key": case_key, "reason": "no_exported_clouds"})
            continue

        # target cloud: use Phase1 selection (smallest segmented$).
        used = (r.get("used_clouds") or [])
        if not used:
            skipped.append({"case_key": case_key, "reason": "no_used_clouds"})
            continue
        target_rel = str(used[0].get("npz") or "")
        if not target_rel:
            skipped.append({"case_key": case_key, "reason": "missing_target_npz_rel"})
            continue
        target_path = converted_root / target_rel
        if not target_path.exists():
            stats["cases_missing_npz"] += 1
            skipped.append({"case_key": case_key, "reason": "missing_target_npz", "path": str(target_path)})
            continue

        # scene cloud: largest exported cloud (usually contains both jaws).
        scene = max(exported, key=lambda x: int(x.get("points") or 0))
        scene_rel = str((scene.get("outputs") or {}).get("npz") or "")
        if not scene_rel:
            skipped.append({"case_key": case_key, "reason": "missing_scene_npz_rel"})
            continue
        scene_path = converted_root / scene_rel
        if not scene_path.exists():
            stats["cases_missing_npz"] += 1
            skipped.append({"case_key": case_key, "reason": "missing_scene_npz", "path": str(scene_path)})
            continue

        try:
            with np.load(target_path) as data:
                tgt_raw = np.asarray(data["points"], dtype=np.float32)
            with np.load(scene_path) as data:
                scene_raw = np.asarray(data["points"], dtype=np.float32)
        except Exception as e:
            skipped.append({"case_key": case_key, "reason": "npz_load_error", "error": str(e)})
            continue

        if tgt_raw.ndim != 2 or tgt_raw.shape[1] != 3 or tgt_raw.shape[0] < 32:
            skipped.append({"case_key": case_key, "reason": "bad_target_points", "shape": list(tgt_raw.shape)})
            continue
        if scene_raw.ndim != 2 or scene_raw.shape[1] != 3 or scene_raw.shape[0] < 1000:
            skipped.append({"case_key": case_key, "reason": "bad_scene_points", "shape": list(scene_raw.shape)})
            continue

        # Canonicalize target (store inverse transform for occlusion with raw opposing jaw).
        tgt_norm, centroid, scale = normalize_points(tgt_raw, cfg.normalize)
        if cfg.pca:
            tgt_norm, rmat = pca_align(tgt_norm)
        else:
            rmat = np.eye(3, dtype=np.float32)

        seed = stable_int_seed(case_key, cfg.seed)
        rng = np.random.default_rng(seed)
        tgt_out = downsample(tgt_norm, cfg.target_points, rng)

        q = float(np.clip(cfg.cut_q, 0.05, 0.95))
        z = tgt_norm[:, 2]
        thr = float(np.quantile(z, q))
        patch_raw: np.ndarray | None = None

        if cfg.prep_mode == "synthetic_cut":
            kept = tgt_norm[z <= thr]
            if kept.shape[0] < max(16, int(0.2 * cfg.target_points)):
                kept = tgt_norm
            prep_out = downsample(kept, cfg.prep_points, rng)
            prep_meta = {
                "mode": "synthetic_cut",
                "n_source": int(kept.shape[0]),
            }
        elif cfg.prep_mode == "scene_patch":
            # Use raw scene points (same jaw) as prep signal; transform into target-canonical coords.
            scene_z = scene_raw[:, 2]
            m1, m2 = kmeans_1d_two_means(scene_z, rng)
            if not math.isfinite(m1) or not math.isfinite(m2) or abs(m2 - m1) < 1e-3:
                stats["cases_missing_opposing"] += 1
                skipped.append({"case_key": case_key, "reason": "no_bimodal_z", "m1": m1, "m2": m2})
                continue
            thr_z = 0.5 * (m1 + m2)
            target_z_mean = float(tgt_raw[:, 2].mean())
            same_is_low = abs(target_z_mean - m1) <= abs(target_z_mean - m2)
            if same_is_low:
                opp_mask = scene_z > thr_z
                opp_cluster = "high"
            else:
                opp_mask = scene_z <= thr_z
                opp_cluster = "low"

            mn = tgt_raw.min(axis=0)
            mx = tgt_raw.max(axis=0)
            bbox_diag = float(np.linalg.norm((mx - mn).astype(np.float64)))
            base_radius = float(max(1e-3, cfg.patch_radius_mult * bbox_diag))
            d2 = np.sum((scene_raw - centroid[None, :]) ** 2, axis=1)
            patch = scene_raw
            radius = float("inf")
            for mul in [1.0, 1.5, 2.0, 3.0]:
                rad = float(base_radius * mul)
                cand = scene_raw[d2 <= rad * rad]
                if cand.shape[0] >= 2048:
                    patch = cand
                    radius = rad
                    break
                patch = cand
                radius = rad

            # Cap patch candidates to avoid pulling in full-arch context.
            max_patch = 50_000
            if patch.shape[0] > max_patch:
                d2_patch = np.sum((patch - centroid[None, :]) ** 2, axis=1)
                idx = np.argpartition(d2_patch, max_patch - 1)[:max_patch]
                patch = patch[idx]

            # Remove target-overlapping points by voxel hashing (in raw coords).
            if cfg.overlap_voxel > 0:
                vs = float(cfg.overlap_voxel)
                v_tgt = np.floor(tgt_raw / vs).astype(np.int64)
                v_patch = np.floor(patch / vs).astype(np.int64)
                v_min = np.minimum(v_tgt.min(axis=0), v_patch.min(axis=0))
                v_tgt = v_tgt - v_min
                v_patch = v_patch - v_min
                dims = np.maximum(v_tgt.max(axis=0), v_patch.max(axis=0)) + 1
                stride_y = int(dims[0])
                stride_z = int(dims[0] * dims[1])
                key_tgt = v_tgt[:, 0] + v_tgt[:, 1] * stride_y + v_tgt[:, 2] * stride_z
                key_patch = v_patch[:, 0] + v_patch[:, 1] * stride_y + v_patch[:, 2] * stride_z
                tgt_keys = np.unique(key_tgt)
                keep_mask = ~np.isin(key_patch, tgt_keys)
                patch2 = patch[keep_mask]
                patch = patch2 if patch2.shape[0] >= 64 else patch

            # Transform patch to canonical (same as target).
            patch_raw = patch
            patch_norm = ((patch - centroid[None, :]) / float(scale)).astype(np.float32, copy=False)
            patch_can = (patch_norm @ rmat).astype(np.float32, copy=False)
            prep_out = downsample(patch_can, cfg.prep_points, rng)
            prep_meta = {
                "mode": "scene_patch",
                "radius": float(radius),
                "patch_n": int(patch.shape[0]),
                "overlap_voxel": float(cfg.overlap_voxel),
                "target_bbox_diag": float(bbox_diag),
                "target_z_mean": float(target_z_mean),
                "scene_z_means": [float(m1), float(m2)],
                "scene_split_thr_z": float(thr_z),
                "opposing_cluster": opp_cluster,
            }
        else:
            raise ValueError(f"unknown prep_mode: {cfg.prep_mode}")

        if cfg.margin_mode == "contact" and patch_raw is not None and patch_raw.shape[0] >= 64:
            try:
                from scipy.spatial import cKDTree  # type: ignore

                kdt = cKDTree(patch_raw.astype(np.float64, copy=False))
                dists, _ = kdt.query(tgt_raw.astype(np.float64, copy=False), k=1)
                q_m = float(np.clip(cfg.margin_quantile, 0.001, 0.5))
                thr_d = float(np.quantile(dists, q_m))
                thr_d = float(min(thr_d, float(cfg.margin_max_dist)))
                cand_raw = tgt_raw[dists <= thr_d]
                if cand_raw.shape[0] < cfg.margin_points:
                    idx = np.argsort(dists)[: max(int(cfg.margin_points), 16)]
                    cand_raw = tgt_raw[idx]
                cand_norm = ((cand_raw - centroid[None, :]) / float(scale)).astype(np.float32, copy=False)
                cand_can = (cand_norm @ rmat).astype(np.float32, copy=False)
                margin_out = downsample(cand_can, cfg.margin_points, rng)
                margin_meta = {
                    "mode": "contact",
                    "quantile": q_m,
                    "thr_dist": float(thr_d),
                    "n_candidates": int(cand_raw.shape[0]),
                    "dist_p01": float(np.quantile(dists, 0.01)),
                    "dist_p05": float(np.quantile(dists, 0.05)),
                    "dist_p10": float(np.quantile(dists, 0.10)),
                }
            except Exception as e:
                cfg_fallback = str(e)
                band = max(1e-6, float(cfg.margin_band))
                margin_cand = tgt_norm[np.abs(z - thr) <= band]
                if margin_cand.shape[0] < 4:
                    margin_cand = tgt_norm[z <= thr]
                margin_out = downsample(margin_cand, cfg.margin_points, rng)
                margin_meta = {"mode": "synthetic_cut_fallback", "error": cfg_fallback}
        else:
            band = max(1e-6, float(cfg.margin_band))
            margin_cand = tgt_norm[np.abs(z - thr) <= band]
            if margin_cand.shape[0] < 4:
                margin_cand = tgt_norm[z <= thr]
            margin_out = downsample(margin_cand, cfg.margin_points, rng)
            margin_meta = {"mode": "synthetic_cut"}

        # opposing jaw extraction: split scene by z into 2 clusters, pick the one opposite target.
        target_z_mean = float(tgt_raw[:, 2].mean())
        scene_z = scene_raw[:, 2]
        m1, m2 = kmeans_1d_two_means(scene_z, rng)
        if not math.isfinite(m1) or not math.isfinite(m2) or abs(m2 - m1) < 1e-3:
            stats["cases_missing_opposing"] += 1
            skipped.append({"case_key": case_key, "reason": "no_bimodal_z", "m1": m1, "m2": m2})
            continue
        thr_z = 0.5 * (m1 + m2)
        same_is_low = abs(target_z_mean - m1) <= abs(target_z_mean - m2)
        if same_is_low:
            opp_mask = scene_z > thr_z
            same_mask = ~opp_mask
            opp_cluster = "high"
        else:
            opp_mask = scene_z <= thr_z
            same_mask = ~opp_mask
            opp_cluster = "low"
        opp_pts = scene_raw[opp_mask]
        if opp_pts.shape[0] < 64:
            stats["cases_missing_opposing"] += 1
            skipped.append({"case_key": case_key, "reason": "opposing_too_small", "opp_n": int(opp_pts.shape[0])})
            continue
        opp_out = downsample(opp_pts.astype(np.float32, copy=False), cfg.occlusion_points, rng)

        # write sample
        src_dir = Path(case_key).parent
        stem = Path(case_key).stem  # Group-xxx
        rel_npz = Path("samples") / src_dir / f"{stem}.npz"
        out_path = out_root / rel_npz
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            prep_points=prep_out.astype(np.float32, copy=False),
            target_points=tgt_out.astype(np.float32, copy=False),
            margin_points=margin_out.astype(np.float32, copy=False),
            centroid=centroid.astype(np.float32, copy=False),
            scale=np.asarray(scale, dtype=np.float32),
            R=rmat.astype(np.float32, copy=False),
            opp_points=opp_out.astype(np.float32, copy=False),
        )

        built_rows.append(
            {
                "case_key": case_key,
                "split": r.get("split") or "unknown",
                "source": r.get("source"),
                "label": r.get("label"),
                "tooth_position": r.get("tooth_position"),
                "sample_npz": str(rel_npz),
                "prep_points": int(cfg.prep_points),
                "target_points": int(cfg.target_points),
                "margin_points": int(cfg.margin_points),
                "occlusion_points": int(cfg.occlusion_points),
                "cut_q": float(q),
                "cut_thr_z": float(thr),
                "margin_band": float(cfg.margin_band),
                "margin": margin_meta,
                "prep": prep_meta,
                "normalize": cfg.normalize,
                "pca": bool(cfg.pca),
                "centroid": [float(x) for x in centroid.tolist()],
                "scale": float(scale),
                "R": [[float(x) for x in row_] for row_ in rmat.tolist()],
                "target_cloud": {
                    "name": used[0].get("name"),
                    "npz": target_rel,
                    "points_reported": used[0].get("points_reported"),
                },
                "scene_cloud": {
                    "name": scene.get("name"),
                    "npz": scene_rel,
                    "points_reported": int(scene.get("points") or 0),
                },
                "opposing": {
                    "cluster": opp_cluster,
                    "scene_z_means": [float(m1), float(m2)],
                    "scene_split_thr_z": float(thr_z),
                    "scene_n_same": int(scene_raw[same_mask].shape[0]),
                    "scene_n_opp": int(opp_pts.shape[0]),
                    "target_z_mean": float(target_z_mean),
                },
            }
        )
        stats["cases_written"] += 1

        if i % 25 == 0:
            print(f"[{i}/{len(raw_rows)}] written={stats['cases_written']} skipped={len(skipped)}", flush=True)

    write_jsonl(out_root / "index.jsonl", built_rows)
    write_json(out_root / "stats.json", {"generated_at": utc_now_iso(), "config": cfg.__dict__, "stats": stats, "skipped": skipped[:200]})

    report_lines: list[str] = []
    report_lines.append(f"# raw_prep2target {out_root.name} 构建报告")
    report_lines.append(f"- generated_at: {utc_now_iso()}")
    report_lines.append(f"- raw_cls_index: `{raw_cls_index.relative_to(root)}`")
    report_lines.append(f"- manifest: `{manifest_path.relative_to(root)}`")
    report_lines.append(f"- out: `{out_root.relative_to(root)}`")
    report_lines.append("")
    report_lines.append("## 参数")
    report_lines.append(f"- target_points: {cfg.target_points}")
    report_lines.append(f"- margin_points: {cfg.margin_points}")
    report_lines.append(f"- occlusion_points: {cfg.occlusion_points}")
    report_lines.append(f"- cut_q: {cfg.cut_q}")
    report_lines.append(f"- margin_band: {cfg.margin_band}")
    report_lines.append(f"- normalize: {cfg.normalize}")
    report_lines.append(f"- pca: {cfg.pca}")
    report_lines.append("")
    report_lines.append("## 统计")
    for k, v in stats.items():
        report_lines.append(f"- {k}: {v}")
    report_lines.append("")
    if skipped:
        report_lines.append("## 跳过样本（前 50）")
        for it in skipped[:50]:
            report_lines.append(f"- {it}")
        report_lines.append("")
    (out_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_root / 'index.jsonl'}")
    print(f"[OK] cases_written: {stats['cases_written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
