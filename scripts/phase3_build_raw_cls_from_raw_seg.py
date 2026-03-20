#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _lib.device import normalize_device
from _lib.io import read_json, read_jsonl, write_json, write_jsonl
from _lib.time import utc_now_iso
from phase3_train_raw_seg import DGCNNv2Seg, PointNetSeg, PointTransformerSeg


CLS_LABEL_ORDER = ["充填", "全冠", "桩核冠", "高嵌体"]


def safe_case_name(case_key: str) -> str:
    s = str(case_key).replace("\\", "/").strip("/")
    s = s.replace("/", "__").replace(" ", "_")
    s = re.sub(r"[^0-9A-Za-z_\-.\u4e00-\u9fff]+", "_", s)
    return s


def build_cls_label_map(rows: list[dict[str, Any]], preset_path: Path | None) -> dict[str, int]:
    labels = [str(r.get("label") or "").strip() for r in rows if str(r.get("label") or "").strip()]
    uniq = sorted(set(labels))
    if preset_path is not None and preset_path.exists():
        preset = read_json(preset_path)
        order = [lab for lab, _ in sorted(((str(k), int(v)) for k, v in preset.items()), key=lambda kv: kv[1]) if lab in uniq]
        rest = [lab for lab in uniq if lab not in order]
        order.extend(rest)
        return {lab: i for i, lab in enumerate(order)}
    order = [lab for lab in CLS_LABEL_ORDER if lab in uniq]
    rest = [lab for lab in uniq if lab not in order]
    order.extend(rest)
    return {lab: i for i, lab in enumerate(order)}


def load_case_meta(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    p = path.resolve()
    if not p.exists():
        return {}
    rows = read_jsonl(p)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_key = str(row.get("case_key") or "").strip()
        if not case_key:
            continue
        out[case_key] = {
            "source": str(row.get("source") or ""),
            "tooth_position": str(row.get("tooth_position") or ""),
            "label_raw": str(row.get("label_raw") or ""),
        }
    return out


def make_seg_model(name: str) -> torch.nn.Module:
    nm = str(name).strip().lower()
    if nm == "pointnet_seg":
        return PointNetSeg(num_classes=2)
    if nm == "dgcnn_v2":
        return DGCNNv2Seg(num_classes=2)
    if nm == "point_transformer":
        return PointTransformerSeg(num_classes=2)
    raise ValueError(f"Unsupported --seg-model: {name}")


@torch.no_grad()
def predict_restoration_prob(model: torch.nn.Module, points_np: np.ndarray, device: torch.device) -> np.ndarray:
    pts = torch.from_numpy(np.asarray(points_np, dtype=np.float32)).unsqueeze(0).to(device)
    logits = model(pts)
    prob = torch.softmax(logits, dim=1)[0, 1]
    return prob.detach().cpu().numpy().astype(np.float32, copy=False)


def summarize_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def write_report(path: Path, *, cfg: dict[str, Any], label_counts: Counter[str], split_counts: Counter[str], point_counts: list[int], fallback_count: int) -> None:
    mean_points = float(np.mean(point_counts)) if point_counts else 0.0
    min_points = int(min(point_counts)) if point_counts else 0
    max_points = int(max(point_counts)) if point_counts else 0
    lines = [
        f"# raw_cls_from_raw_seg {cfg['out_name']}",
        f"- generated_at: {cfg['generated_at']}",
        f"- seg_root: `{cfg['seg_root']}`",
        f"- out_root: `{cfg['out_root']}`",
        f"- mode: `{cfg['mode']}`",
        f"- topk: {cfg['topk']}",
        f"- seg_model: `{cfg['seg_model']}`",
        f"- seg_ckpt: `{cfg['seg_ckpt']}`",
        f"- device: `{cfg['device']}`",
        "",
        "## Counts",
        f"- samples: {sum(split_counts.values())}",
        f"- by_split: {summarize_counter(split_counts)}",
        f"- by_label: {summarize_counter(label_counts)}",
        f"- selected_points: mean={mean_points:.1f}, min={min_points}, max={max_points}",
        f"- empty_selection_fallback_to_all: {int(fallback_count)}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build classification datasets from raw segmentation data.")
    ap.add_argument("--seg-root", type=Path, default=Path("processed/raw_seg/v1"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["all", "gt_seg", "pred_topk"], default="all")
    ap.add_argument("--topk", type=int, default=4096, help="Keep top-K points for pred_topk mode.")
    ap.add_argument("--seg-model", type=str, default="dgcnn_v2", help="pointnet_seg|dgcnn_v2|point_transformer")
    ap.add_argument("--seg-ckpt", type=Path, default=None, help="Required for pred_topk mode.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--label-map-from", type=Path, default=Path("processed/raw_cls/v13_main4/label_map.json"))
    ap.add_argument("--meta-from", type=Path, default=Path("processed/raw_cls/v13_main4/index.jsonl"))
    args = ap.parse_args()

    seg_root = args.seg_root.resolve()
    out_root = args.out.resolve()
    rows = read_jsonl(seg_root / "index.jsonl")
    if not rows:
        raise SystemExit(f"No rows found under {seg_root}/index.jsonl")
    case_meta = load_case_meta(args.meta_from if args.meta_from is not None else None)

    label_map = build_cls_label_map(rows, args.label_map_from.resolve() if args.label_map_from else None)
    samples_dir = out_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device_str = normalize_device(str(args.device))
    device = torch.device(device_str)
    model = None
    if args.mode == "pred_topk":
        if args.seg_ckpt is None:
            raise SystemExit("--seg-ckpt is required when --mode=pred_topk")
        model = make_seg_model(str(args.seg_model))
        ckpt = torch.load(str(args.seg_ckpt.resolve()), map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()

    out_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    point_counts: list[int] = []
    fallback_count = 0

    for i, row in enumerate(rows, start=1):
        rel = str(row.get("sample_npz") or "")
        if not rel:
            raise SystemExit(f"Missing sample_npz for row {i}: {row}")
        src_npz = seg_root / rel
        with np.load(src_npz) as z:
            points = np.asarray(z["points"], dtype=np.float32)
            seg_labels = np.asarray(z["labels"], dtype=np.int64)

        selected = points
        score_mean = None
        score_p90 = None

        if args.mode == "gt_seg":
            keep = seg_labels == 1
            if np.any(keep):
                selected = points[keep]
            else:
                fallback_count += 1
        elif args.mode == "pred_topk":
            assert model is not None
            probs = predict_restoration_prob(model, points, device)
            k = int(args.topk)
            if k > 0 and k < int(points.shape[0]):
                idx = np.argpartition(-probs, k - 1)[:k]
                idx = idx[np.argsort(-probs[idx])]
            else:
                idx = np.argsort(-probs)
            if idx.size > 0:
                selected = points[idx]
                sel_prob = probs[idx]
                score_mean = float(np.mean(sel_prob))
                score_p90 = float(np.quantile(sel_prob, 0.90))
            else:
                fallback_count += 1

        if int(selected.shape[0]) <= 0:
            selected = points
            fallback_count += 1

        case_key = str(row.get("case_key") or f"sample_{i}")
        safe_name = safe_case_name(case_key) + ".npz"
        out_rel = f"samples/{safe_name}"
        np.savez_compressed(str(samples_dir / safe_name), points=selected.astype(np.float32, copy=False))

        out_row = dict(row)
        meta = case_meta.get(case_key) or {}
        if "source" not in out_row:
            out_row["source"] = str(meta.get("source") or "")
        if "tooth_position" not in out_row:
            out_row["tooth_position"] = str(meta.get("tooth_position") or "")
        if "label_raw" not in out_row and meta.get("label_raw"):
            out_row["label_raw"] = str(meta.get("label_raw") or "")
        out_row["sample_npz"] = out_rel
        out_row["n_points_selected"] = int(selected.shape[0])
        out_row["target_points"] = int(min(int(args.topk), int(selected.shape[0]))) if int(args.topk) > 0 else int(selected.shape[0])
        out_row["builder_mode"] = str(args.mode)
        if score_mean is not None:
            out_row["seg_score_mean"] = float(score_mean)
        if score_p90 is not None:
            out_row["seg_score_p90"] = float(score_p90)
        out_rows.append(out_row)

        label_counts[str(out_row.get("label") or "")] += 1
        split_counts[str(out_row.get("split") or "")] += 1
        point_counts.append(int(selected.shape[0]))

    write_json(out_root / "label_map.json", label_map)
    write_jsonl(out_root / "index.jsonl", out_rows)

    report_cfg = {
        "generated_at": utc_now_iso(),
        "out_name": out_root.name,
        "seg_root": str(seg_root),
        "out_root": str(out_root),
        "mode": str(args.mode),
        "topk": int(args.topk),
        "seg_model": str(args.seg_model),
        "seg_ckpt": str(args.seg_ckpt.resolve()) if args.seg_ckpt is not None else "",
        "device": device_str,
        "meta_from": str(args.meta_from.resolve()) if args.meta_from is not None and args.meta_from.exists() else "",
    }
    write_json(out_root / "build_config.json", report_cfg)
    write_report(
        out_root / "report.md",
        cfg=report_cfg,
        label_counts=label_counts,
        split_counts=split_counts,
        point_counts=point_counts,
        fallback_count=fallback_count,
    )
    print(f"[done] wrote dataset to {out_root}")
    print(f"[done] mode={args.mode} samples={len(out_rows)} fallback={fallback_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
