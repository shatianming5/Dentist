#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonicalize_label(label: str | None) -> str:
    if not label:
        return "未知"
    s = str(label).strip()
    mapping = {
        "嵌体/高嵌体": "高嵌体",
        "高嵌体": "高嵌体",
        "树脂充填修复": "充填",
        "充填": "充填",
        "全冠": "全冠",
        "桩核冠": "桩核冠",
        "拔除": "拔除",
        "实在看不清": "未知",
        "未知": "未知",
        "未标注": "未知",
    }
    return mapping.get(s, s)


@dataclass(frozen=True)
class BuildConfig:
    seed: int
    min_points: int
    max_points_per_cloud: int
    target_points: int
    normalize: str
    with_rgb: bool
    with_cloud_id: bool
    cloud_sampling: str
    min_train_count: int
    include_name_regex: str
    exclude_name_regex: str
    prefer_name_regex: str
    select_topk: int
    select_smallk: int
    drop_labels: list[str]


def load_cloud_npz(path: Path, *, with_rgb: bool) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path) as data:
        pts = data["points"]
        rgb = data["rgb"] if with_rgb and "rgb" in data.files else None
    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Invalid points shape {pts.shape} in {path}")
    if pts.dtype != np.float32 and pts.dtype != np.float64:
        pts = pts.astype(np.float32)
    pts = pts.astype(np.float32, copy=False)

    if not with_rgb:
        return pts, None
    if rgb is None:
        raise ValueError(f"Missing `rgb` in {path}")
    rgb_arr = np.asarray(rgb)
    if rgb_arr.ndim != 2 or rgb_arr.shape[1] != 3 or rgb_arr.shape[0] != pts.shape[0]:
        raise ValueError(f"Invalid rgb shape {rgb_arr.shape} in {path} for points {pts.shape}")
    if rgb_arr.dtype == np.uint8:
        rgb_u8 = rgb_arr
    else:
        rgb_f = rgb_arr.astype(np.float32, copy=False)
        if float(np.max(rgb_f, initial=0.0)) <= 1.5:
            rgb_f = rgb_f * 255.0
        rgb_u8 = np.clip(np.round(rgb_f), 0, 255).astype(np.uint8)
    return pts, rgb_u8


def downsample_points(rng: np.random.Generator, pts: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be > 0")
    m = int(pts.shape[0])
    if m == n:
        return pts
    replace = m < n
    idx = rng.choice(m, size=n, replace=replace)
    return pts[idx]


def downsample_aligned(
    rng: np.random.Generator,
    pts: np.ndarray,
    rgb_u8: np.ndarray | None,
    n: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if n <= 0:
        raise ValueError("n must be > 0")
    m = int(pts.shape[0])
    if m == n:
        return pts, rgb_u8
    replace = m < n
    idx = rng.choice(m, size=n, replace=replace)
    out_pts = pts[idx]
    out_rgb = rgb_u8[idx] if rgb_u8 is not None else None
    return out_pts, out_rgb


def downsample_aligned3(
    rng: np.random.Generator,
    pts: np.ndarray,
    rgb_u8: np.ndarray | None,
    cloud_id: np.ndarray | None,
    n: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if n <= 0:
        raise ValueError("n must be > 0")
    m = int(pts.shape[0])
    if m == n:
        return pts, rgb_u8, cloud_id
    replace = m < n
    idx = rng.choice(m, size=n, replace=replace)
    out_pts = pts[idx]
    out_rgb = rgb_u8[idx] if rgb_u8 is not None else None
    out_cid = cloud_id[idx] if cloud_id is not None else None
    return out_pts, out_rgb, out_cid


def normalize_points(points: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, float]:
    if points.size == 0:
        raise ValueError("empty points")
    centroid = points.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = points - centroid
    if mode == "max_norm":
        norms = np.linalg.norm(centered.astype(np.float64), axis=1)
        scale = float(np.max(norms)) if norms.size else 1.0
    elif mode == "bbox_diag":
        mn = centered.min(axis=0)
        mx = centered.max(axis=0)
        scale = float(np.linalg.norm((mx - mn).astype(np.float64)))
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    normalized = (centered / scale).astype(np.float32, copy=False)
    return normalized, centroid, scale


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1: build raw classification dataset (case-level).")
    ap.add_argument("--root", type=Path, default=Path("."), help="Repo root (default: .)")
    ap.add_argument("--manifest", type=Path, default=Path("converted/raw/manifest_with_labels.json"))
    ap.add_argument("--splits", type=Path, default=Path("metadata/splits_raw_case.json"))
    ap.add_argument("--converted-root", type=Path, default=Path("converted/raw"))
    ap.add_argument("--out", type=Path, default=Path("processed/raw_cls/v1"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min-points", type=int, default=500)
    ap.add_argument("--max-points-per-cloud", type=int, default=50_000)
    ap.add_argument("--target-points", type=int, default=4096)
    ap.add_argument("--normalize", choices=["max_norm", "bbox_diag"], default="max_norm")
    ap.add_argument("--with-rgb", action="store_true", help="Include per-point RGB from converted clouds (requires 'rgb' in NPZ).")
    ap.add_argument(
        "--with-cloud-id",
        action="store_true",
        help="Include per-point cloud_id (0..K-1) indicating which exported cloud a point came from (after filtering/selection).",
    )
    ap.add_argument(
        "--cloud-sampling",
        choices=["concat", "equal"],
        default="concat",
        help="How to sample points across multiple clouds before normalization: concat (proportional) or equal (per-cloud equal quota).",
    )
    ap.add_argument(
        "--min-train-count",
        type=int,
        default=0,
        help="Drop any canonical label that has <K samples in train split (after canonicalization/merge). 0=disable.",
    )
    ap.add_argument(
        "--include-name-regex",
        default="",
        help="Only include exported_clouds whose name matches this regex (optional).",
    )
    ap.add_argument(
        "--exclude-name-regex",
        default="",
        help="Exclude exported_clouds whose name matches this regex (optional).",
    )
    ap.add_argument(
        "--prefer-name-regex",
        default="",
        help="If any candidate cloud name matches this regex, only use those candidates (optional).",
    )
    ap.add_argument(
        "--select-topk",
        type=int,
        default=0,
        help="After filtering, keep only top-K clouds by points_used (0=all).",
    )
    ap.add_argument(
        "--select-smallk",
        type=int,
        default=0,
        help="After filtering, keep only smallest-K clouds by points_used (0=disabled).",
    )
    ap.add_argument(
        "--merge-extraction-to-unknown",
        action="store_true",
        help="Map label '拔除' to '未知' (useful when train has no '拔除').",
    )
    ap.add_argument(
        "--drop-labels",
        default="",
        help="Comma-separated canonical labels to drop (e.g. '拔除,未知'). Applied after canonicalization/merge.",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    manifest_path = (root / args.manifest).resolve()
    splits_path = (root / args.splits).resolve()
    converted_root = (root / args.converted_root).resolve()
    out_root = (root / args.out).resolve()

    cfg = BuildConfig(
        seed=args.seed,
        min_points=args.min_points,
        max_points_per_cloud=args.max_points_per_cloud,
        target_points=args.target_points,
        normalize=args.normalize,
        with_rgb=bool(args.with_rgb),
        with_cloud_id=bool(args.with_cloud_id),
        cloud_sampling=str(args.cloud_sampling or "concat").strip().lower() or "concat",
        min_train_count=int(args.min_train_count or 0),
        include_name_regex=str(args.include_name_regex or ""),
        exclude_name_regex=str(args.exclude_name_regex or ""),
        prefer_name_regex=str(args.prefer_name_regex or ""),
        select_topk=int(args.select_topk or 0),
        select_smallk=int(args.select_smallk or 0),
        drop_labels=[s.strip() for s in str(args.drop_labels or "").split(",") if s.strip()],
    )

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    if not splits_path.exists():
        raise SystemExit(f"Missing splits: {splits_path} (run scripts/phase0_freeze.py first)")
    if not converted_root.exists():
        raise SystemExit(f"Missing converted root: {converted_root}")

    out_samples = out_root / "samples"
    out_samples.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now_iso()
    rng = np.random.default_rng(cfg.seed)
    include_re = re.compile(cfg.include_name_regex) if cfg.include_name_regex else None
    exclude_re = re.compile(cfg.exclude_name_regex) if cfg.exclude_name_regex else None
    prefer_re = re.compile(cfg.prefer_name_regex) if cfg.prefer_name_regex else None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    split_of: dict[str, str] = {}
    for split_name, case_keys in (splits.get("splits") or {}).items():
        for k in case_keys:
            split_of[str(k)] = str(split_name)

    # Auto-drop labels that are too rare in train split (helps avoid unstable metrics due to test-only classes).
    auto_drop: set[str] = set()
    if cfg.min_train_count and cfg.min_train_count > 0:
        train_counts: Counter[str] = Counter()
        all_labels: Counter[str] = Counter()
        for entry in manifest:
            case_key = str(entry.get("input", "")).strip()
            if not case_key:
                continue
            label_info = entry.get("label_info") or {}
            label_raw = label_info.get("label")
            label = canonicalize_label(label_raw)
            if args.merge_extraction_to_unknown and label == "拔除":
                label = "未知"
            all_labels[label] += 1
            if split_of.get(case_key, "unknown") == "train":
                train_counts[label] += 1
        for lab in all_labels.keys():
            if int(train_counts.get(lab, 0)) < int(cfg.min_train_count):
                auto_drop.add(str(lab))

    drop_label_set = set(cfg.drop_labels) | set(auto_drop)

    index_path = out_root / "index.jsonl"
    report_path = out_root / "report.md"
    label_map_path = out_root / "label_map.json"

    label_counter: dict[str, Counter[str]] = {"train": Counter(), "val": Counter(), "test": Counter(), "unknown": Counter()}
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()

    for entry in manifest:
        case_key = str(entry.get("input", "")).strip()
        if not case_key:
            continue
        split = split_of.get(case_key, "unknown")
        label_info = entry.get("label_info") or {}
        label_raw = label_info.get("label")
        label = canonicalize_label(label_raw)
        if args.merge_extraction_to_unknown and label == "拔除":
            label = "未知"
        if drop_label_set and label in drop_label_set:
            dropped[label] += 1
            continue

        exported_clouds = entry.get("exported_clouds") or []
        candidates: list[tuple[np.ndarray, dict[str, Any]]] = []

        n_objects_total = len(exported_clouds)
        n_points_total_reported = 0
        n_points_after_cap_total = 0

        try:
            for cloud in exported_clouds:
                try:
                    n_points = int(cloud.get("points") or 0)
                except Exception:
                    n_points = 0
                n_points_total_reported += max(0, n_points)

                name = str(cloud.get("name") or "")
                if include_re and not include_re.search(name):
                    continue
                if exclude_re and exclude_re.search(name):
                    continue

                outputs = cloud.get("outputs") or {}
                rel_npz = outputs.get("npz")
                if not rel_npz:
                    continue
                npz_path = converted_root / str(rel_npz)
                if not npz_path.exists():
                    continue
                pts, rgb_u8 = load_cloud_npz(npz_path, with_rgb=cfg.with_rgb)
                m = int(pts.shape[0])
                if m < cfg.min_points:
                    continue
                if cfg.max_points_per_cloud > 0 and m > cfg.max_points_per_cloud:
                    pts, rgb_u8 = downsample_aligned(rng, pts, rgb_u8, cfg.max_points_per_cloud)
                n_points_after_cap_total += int(pts.shape[0])
                candidates.append(
                    (
                        pts,
                        rgb_u8,
                        {
                            "name": name,
                            "points_reported": n_points,
                            "points_used": int(pts.shape[0]),
                            "npz": str(rel_npz),
                        },
                    )
                )

            if not candidates:
                raise RuntimeError("no usable clouds after filtering")

            if prefer_re:
                preferred = [item for item in candidates if prefer_re.search(str(item[2].get("name") or ""))]
                if preferred:
                    candidates = preferred

            if cfg.select_smallk > 0 and cfg.select_topk > 0:
                raise ValueError("select_smallk and select_topk are mutually exclusive")
            if cfg.select_topk > 0 and len(candidates) > cfg.select_topk:
                candidates.sort(key=lambda item: int(item[2].get("points_used") or 0), reverse=True)
                candidates = candidates[: cfg.select_topk]
            if cfg.select_smallk > 0 and len(candidates) > cfg.select_smallk:
                candidates.sort(key=lambda item: int(item[2].get("points_used") or 0))
                candidates = candidates[: cfg.select_smallk]

            used_clouds = [meta for _pts, _rgb, meta in candidates]
            points_list = [pts for pts, _rgb, _meta in candidates]
            rgb_list = [rgb for _pts, rgb, _meta in candidates] if cfg.with_rgb else []
            cloud_id_list = (
                [np.full((int(pts.shape[0]),), i, dtype=np.int16) for i, (pts, _rgb, _meta) in enumerate(candidates)]
                if cfg.with_cloud_id
                else []
            )
            n_points_after_cap = int(sum(int(m.get("points_used") or 0) for m in used_clouds))

            if cfg.cloud_sampling == "equal" and len(candidates) > 1:
                k = int(len(candidates))
                base = int(cfg.target_points) // k
                rem = int(cfg.target_points) - (base * k)
                # assign remainder to larger clouds (more stable)
                order = sorted(range(k), key=lambda i: int(candidates[i][0].shape[0]), reverse=True)
                quotas = [base for _ in range(k)]
                for j in range(rem):
                    quotas[order[j]] += 1

                sampled_pts: list[np.ndarray] = []
                sampled_rgb: list[np.ndarray] = []
                sampled_cid: list[np.ndarray] = []
                for i, (pts, rgb_u8, _meta) in enumerate(candidates):
                    n_i = int(quotas[i])
                    pts_i, rgb_i = downsample_aligned(rng, pts, rgb_u8, n_i)
                    sampled_pts.append(pts_i.astype(np.float32, copy=False))
                    if cfg.with_rgb:
                        assert rgb_i is not None
                        sampled_rgb.append(rgb_i.astype(np.uint8, copy=False))
                    if cfg.with_cloud_id:
                        sampled_cid.append(np.full((n_i,), i, dtype=np.int16))

                points_all = np.concatenate(sampled_pts, axis=0).astype(np.float32, copy=False)
                rgb_all = np.concatenate(sampled_rgb, axis=0).astype(np.uint8, copy=False) if cfg.with_rgb else None
                cloud_id_all = np.concatenate(sampled_cid, axis=0).astype(np.int16, copy=False) if cfg.with_cloud_id else None
                if int(points_all.shape[0]) != int(cfg.target_points):
                    raise RuntimeError(f"Internal error: expected {cfg.target_points} points after equal sampling, got {points_all.shape[0]}")
                points_all, centroid, scale = normalize_points(points_all, cfg.normalize)
                perm = rng.permutation(int(points_all.shape[0]))
                points_out = points_all[perm]
                rgb_out = rgb_all[perm] if rgb_all is not None else None
                cloud_id_out = cloud_id_all[perm] if cloud_id_all is not None else None
            else:
                points_all = np.concatenate(points_list, axis=0).astype(np.float32, copy=False)
                rgb_all = np.concatenate(rgb_list, axis=0).astype(np.uint8, copy=False) if cfg.with_rgb else None
                cloud_id_all = np.concatenate(cloud_id_list, axis=0).astype(np.int16, copy=False) if cfg.with_cloud_id else None
                points_all, centroid, scale = normalize_points(points_all, cfg.normalize)
                points_out, rgb_out, cloud_id_out = downsample_aligned3(rng, points_all, rgb_all, cloud_id_all, cfg.target_points)

            sample_path = out_samples / (case_key[:-4] + ".npz" if case_key.lower().endswith(".bin") else case_key + ".npz")
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            arrays: dict[str, Any] = {"points": points_out}
            if rgb_out is not None:
                arrays["rgb"] = rgb_out
            if cloud_id_out is not None:
                arrays["cloud_id"] = cloud_id_out
            np.savez_compressed(sample_path, **arrays)

            row = {
                "case_key": case_key,
                "split": split,
                "source": label_info.get("source"),
                "label_raw": label_raw,
                "label": label,
                "tooth_position": label_info.get("tooth_position"),
                "input_bytes": entry.get("input_bytes"),
                "n_objects_total": n_objects_total,
                "n_objects_used": len(used_clouds),
                "n_points_total_reported": n_points_total_reported,
                "n_points_after_cap": n_points_after_cap,
                "n_points_after_cap_total": n_points_after_cap_total,
                "target_points": cfg.target_points,
                "normalize": cfg.normalize,
                "with_rgb": bool(cfg.with_rgb),
                "with_cloud_id": bool(cfg.with_cloud_id),
                "cloud_sampling": str(cfg.cloud_sampling),
                "include_name_regex": cfg.include_name_regex,
                "exclude_name_regex": cfg.exclude_name_regex,
                "prefer_name_regex": cfg.prefer_name_regex,
                "select_topk": cfg.select_topk,
                "select_smallk": cfg.select_smallk,
                "drop_labels": cfg.drop_labels,
                "centroid": [float(x) for x in centroid.tolist()],
                "scale": float(scale),
                "sample_npz": str(sample_path.relative_to(out_root)),
                "used_clouds": used_clouds,
            }
            rows.append(row)
            label_counter.setdefault(split, Counter())[label] += 1
        except Exception as e:
            failures.append({"case_key": case_key, "split": split, "label_raw": label_raw, "label": label, "error": str(e)})
            label_counter.setdefault(split, Counter())[label] += 1

    rows.sort(key=lambda r: r["case_key"])
    with index_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_labels = sorted({r["label"] for r in rows})
    label_to_id = {lab: i for i, lab in enumerate(all_labels)}
    label_map_path.write_text(json.dumps(label_to_id, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    total_cases = len([e for e in manifest if str(e.get("input", "")).strip()])
    success_cases = len(rows)
    failed_cases = len(failures)

    md_lines: list[str] = []
    md_lines.append(f"# raw_cls {out_root.name} 构建报告")
    md_lines.append(f"- 生成时间：{generated_at}")
    md_lines.append(f"- manifest：`{manifest_path.relative_to(root)}`")
    md_lines.append(f"- splits：`{splits_path.relative_to(root)}`")
    md_lines.append(f"- 输出目录：`{out_root.relative_to(root)}`")
    md_lines.append("")
    md_lines.append("## 参数")
    md_lines.append(f"- seed：{cfg.seed}")
    md_lines.append(f"- min_points：{cfg.min_points}")
    md_lines.append(f"- max_points_per_cloud：{cfg.max_points_per_cloud}")
    md_lines.append(f"- target_points：{cfg.target_points}")
    md_lines.append(f"- normalize：{cfg.normalize}")
    md_lines.append(f"- with_rgb：{bool(cfg.with_rgb)}")
    md_lines.append(f"- with_cloud_id：{bool(cfg.with_cloud_id)}")
    md_lines.append(f"- cloud_sampling：{str(cfg.cloud_sampling)}")
    md_lines.append(f"- min_train_count：{cfg.min_train_count}")
    md_lines.append(f"- include_name_regex：{cfg.include_name_regex or '(none)'}")
    md_lines.append(f"- exclude_name_regex：{cfg.exclude_name_regex or '(none)'}")
    md_lines.append(f"- prefer_name_regex：{cfg.prefer_name_regex or '(none)'}")
    md_lines.append(f"- select_topk：{cfg.select_topk}")
    md_lines.append(f"- select_smallk：{cfg.select_smallk}")
    md_lines.append(f"- drop_labels：{', '.join(cfg.drop_labels) if cfg.drop_labels else '(none)'}")
    if auto_drop:
        md_lines.append(f"- auto_drop_labels_by_min_train_count：{sorted(auto_drop)}")
    md_lines.append("")
    md_lines.append("## 结果")
    md_lines.append(f"- 总 case：{total_cases}")
    md_lines.append(f"- 成功：{success_cases}")
    md_lines.append(f"- 失败：{failed_cases}")
    if dropped:
        md_lines.append(f"- 丢弃（drop_labels）：{dict(dropped)}")
    md_lines.append("")
    md_lines.append("## 标签规范化（label_raw → label）")
    md_lines.append("- 嵌体/高嵌体 → 高嵌体")
    md_lines.append("- 树脂充填修复 → 充填")
    md_lines.append("- 实在看不清 / 未标注 / 空 → 未知")
    if args.merge_extraction_to_unknown:
        md_lines.append("- 拔除 → 未知（merge-extraction-to-unknown）")
    md_lines.append("")
    md_lines.append("## split 标签分布（按 canonical label）")
    for split_name in ["train", "val", "test", "unknown"]:
        c = label_counter.get(split_name) or Counter()
        if not c:
            continue
        md_lines.append(f"- {split_name}: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    md_lines.append("")
    if failures:
        md_lines.append("## 失败样本（前 50）")
        for item in failures[:50]:
            md_lines.append(f"- {item['case_key']}: {item['error']}")
        if len(failures) > 50:
            md_lines.append(f"- ... (还有 {len(failures) - 50} 个)")
        md_lines.append("")

    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote: {index_path}")
    print(f"[OK] Wrote: {label_map_path}")
    print(f"[OK] Wrote: {report_path}")
    print(f"[OK] Samples: {out_samples}")
    if failed_cases:
        print(f"[WARN] Failed cases: {failed_cases} (see report.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
