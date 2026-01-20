#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def quantile(sorted_vals: list[int], q: float) -> float | None:
    if not sorted_vals:
        return None
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def summarize_ints(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    total = sum(vals)
    return {
        "count": len(vals),
        "min": vals[0],
        "p05": quantile(vals, 0.05),
        "p25": quantile(vals, 0.25),
        "median": statistics.median(vals),
        "p75": quantile(vals, 0.75),
        "p95": quantile(vals, 0.95),
        "max": vals[-1],
        "mean": total / len(vals),
        "sum": total,
    }


def count_obj_vertices_faces(obj_path: Path) -> tuple[int, int]:
    v = 0
    f = 0
    with obj_path.open("rb") as fh:
        for line in fh:
            if line.startswith(b"v "):
                v += 1
            elif line.startswith(b"f "):
                f += 1
    return v, f


def parse_teeth3ds_seg_json(json_path: Path) -> dict[str, Any]:
    obj = json.loads(json_path.read_text())
    labels = obj.get("labels", [])
    instances = obj.get("instances", [])
    label_hist = Counter(labels)
    inst_hist = Counter(instances)

    unique_labels = len(label_hist)
    unique_instances = len(inst_hist)
    teeth_instances = unique_instances - (1 if 0 in inst_hist else 0)
    teeth_labels = unique_labels - (1 if 0 in label_hist else 0)

    return {
        "id_patient": obj.get("id_patient"),
        "jaw": obj.get("jaw"),
        "labels_len": len(labels),
        "instances_len": len(instances),
        "unique_labels": unique_labels,
        "unique_instances": unique_instances,
        "teeth_instances": teeth_instances,
        "teeth_labels": teeth_labels,
        "label_values": sorted(label_hist.keys()),
        "instance_values": sorted(inst_hist.keys()),
        "label_hist": dict(sorted(label_hist.items())),
        "instance_hist": dict(sorted(inst_hist.items())),
    }


def parse_kpt_json(json_path: Path) -> dict[str, Any]:
    obj = json.loads(json_path.read_text())
    objects = obj.get("objects", [])
    class_hist = Counter(o.get("class") for o in objects)
    class_hist.pop(None, None)
    return {
        "version": obj.get("version"),
        "description": obj.get("description"),
        "key": obj.get("key"),
        "points": len(objects),
        "class_hist": dict(sorted(class_hist.items())),
    }


def parse_split_txt(path: Path) -> list[str]:
    items: list[str] = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        items.append(s)
    return items


@dataclass(frozen=True)
class CaseKey:
    id_: str
    jaw: str  # "upper" | "lower"

    @property
    def item(self) -> str:
        return f"{self.id_}_{self.jaw}"


def item_to_case(item: str) -> CaseKey | None:
    if item.endswith("_upper"):
        return CaseKey(item[:-6], "upper")
    if item.endswith("_lower"):
        return CaseKey(item[:-6], "lower")
    return None


def write_stats_md(out_path: Path, stats: dict[str, Any]) -> None:
    def fmt_num(x: Any) -> str:
        if x is None:
            return "n/a"
        if isinstance(x, float):
            if x.is_integer():
                return str(int(x))
            return f"{x:.2f}"
        return str(x)

    def fmt_summary(s: dict[str, Any]) -> str:
        if not s or s.get("count", 0) == 0:
            return "count=0"
        return (
            f"count={s['count']}, min={fmt_num(s.get('min'))}, "
            f"p25={fmt_num(s.get('p25'))}, median={fmt_num(s.get('median'))}, "
            f"p75={fmt_num(s.get('p75'))}, p95={fmt_num(s.get('p95'))}, "
            f"max={fmt_num(s.get('max'))}, mean={fmt_num(s.get('mean'))}"
        )

    teeth = stats["teeth3ds"]
    land = stats["landmarks"]
    splits = stats["splits"]
    sizes = stats["sizes_bytes"]
    counts = stats["counts"]

    lines: list[str] = []
    lines.append("# Dataset 统计报告\n")
    lines.append(f"生成时间：{stats['generated_at_utc']}\n")

    lines.append("## 总览\n")
    lines.append(f"- `data/`：{human_bytes(sizes['data_total'])}\n")
    lines.append(f"- `archives/`：{human_bytes(sizes['archives_total'])}\n")
    lines.append(f"- 文件数（`data/`）：{counts['data_files']}\n")
    lines.append(f"- 目录数（`data/`）：{counts['data_dirs']}\n")
    lines.append("\n")

    lines.append("## Teeth3DS（mesh + 分割）\n")
    lines.append(f"- `OBJ` 数：{counts['teeth3ds_obj_files']}\n")
    lines.append(f"- `JSON` 数：{counts['teeth3ds_json_files']}\n")
    lines.append(
        "- 上下颌 ID："
        f"upper={teeth['upper']['ids_total']}（json={teeth['upper']['ids_with_json']}）, "
        f"lower={teeth['lower']['ids_total']}（json={teeth['lower']['ids_with_json']}）, "
        f"unique_ids={teeth['unique_ids']}, both_jaws={teeth['both_jaws']}, "
        f"only_upper={teeth['only_upper']}, only_lower={teeth['only_lower']}\n"
    )
    lines.append(
        "- OBJ 顶点数（upper）："
        f"{fmt_summary(teeth['upper']['mesh']['vertices_summary'])}\n"
    )
    lines.append(
        "- OBJ 面片数（upper）："
        f"{fmt_summary(teeth['upper']['mesh']['faces_summary'])}\n"
    )
    lines.append(
        "- OBJ 顶点数（lower）："
        f"{fmt_summary(teeth['lower']['mesh']['vertices_summary'])}\n"
    )
    lines.append(
        "- OBJ 面片数（lower）："
        f"{fmt_summary(teeth['lower']['mesh']['faces_summary'])}\n"
    )
    lines.append(
        "- 分割标签（upper，汇总到所有带 json 的样本）："
        f"labels={teeth['upper']['seg']['label_values']}\n"
    )
    lines.append(
        "- 分割标签（lower，汇总到所有带 json 的样本）："
        f"labels={teeth['lower']['seg']['label_values']}\n"
    )
    lines.append(
        "- `label=0`（牙龈/背景）比例："
        f"upper={teeth['upper']['seg']['label0_ratio']:.4f}, "
        f"lower={teeth['lower']['seg']['label0_ratio']:.4f}\n"
    )
    lines.append(
        "- 每个样本的牙齿实例数（instances>0）："
        f"upper={fmt_summary(teeth['upper']['seg']['teeth_instances_summary'])}, "
        f"lower={fmt_summary(teeth['lower']['seg']['teeth_instances_summary'])}\n"
    )
    lines.append("\n")

    lines.append("## Landmarks（3DTeethLand kpt）\n")
    lines.append(f"- kpt 文件数：{counts['landmarks_kpt_files']}\n")
    lines.append(
        "- 点数量（所有 kpt 文件）："
        f"{fmt_summary(land['points_summary'])}\n"
    )
    lines.append(f"- classes：{land['class_values']}\n")
    lines.append("\n")

    lines.append("## Split 覆盖情况\n")
    for rel, info in sorted(splits["files"].items()):
        lines.append(
            f"- `{rel}`：lines={info['lines']}, unique={info['unique']}, "
            f"dups={info['dups']}, exists={info['exists']}, has_json={info['has_json']}\n"
        )
    lines.append("\n")

    lines.append("## 备注\n")
    lines.append("- 许可证见：`data/splits/license.txt`\n")
    lines.append("- 更细的 per-case 统计见：`DATASET_STATS.json`\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute dataset statistics for this repo.")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--archives-dir", default="archives", help="Archives directory (default: archives)")
    parser.add_argument("--out-json", default="DATASET_STATS.json", help="Output JSON path")
    parser.add_argument("--out-md", default="DATASET_STATS.md", help="Output Markdown path")
    parser.add_argument("--skip-mesh", action="store_true", help="Skip scanning OBJ for vertices/faces")
    parser.add_argument("--skip-seg", action="store_true", help="Skip parsing Teeth3DS segmentation JSONs")
    parser.add_argument("--skip-landmarks", action="store_true", help="Skip parsing landmarks kpt JSONs")
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    data_dir = (repo_root / args.data_dir).resolve()
    archives_dir = (repo_root / args.archives_dir).resolve()

    teeth3ds_root = data_dir / "teeth3ds"
    splits_root = data_dir / "splits"
    landmarks_root = data_dir / "landmarks"

    out_json = (repo_root / args.out_json).resolve()
    out_md = (repo_root / args.out_md).resolve()

    # Index Teeth3DS cases
    cases: dict[str, dict[str, Any]] = {}
    teeth_ids: dict[str, set[str]] = {"upper": set(), "lower": set()}
    teeth_ids_with_json: dict[str, set[str]] = {"upper": set(), "lower": set()}

    mesh_vertices_by_jaw: dict[str, list[int]] = {"upper": [], "lower": []}
    mesh_faces_by_jaw: dict[str, list[int]] = {"upper": [], "lower": []}
    obj_size_by_jaw: dict[str, list[int]] = {"upper": [], "lower": []}
    json_size_by_jaw: dict[str, list[int]] = {"upper": [], "lower": []}

    seg_label_hist_by_jaw: dict[str, Counter[int]] = {"upper": Counter(), "lower": Counter()}
    seg_inst_hist_by_jaw: dict[str, Counter[int]] = {"upper": Counter(), "lower": Counter()}
    seg_teeth_instances_by_jaw: dict[str, list[int]] = {"upper": [], "lower": []}
    seg_label_values_by_jaw: dict[str, set[int]] = {"upper": set(), "lower": set()}

    data_files = 0
    data_dirs = 0
    data_total_size = 0
    archives_total_size = 0

    for p in data_dir.rglob("*"):
        if p.is_file():
            data_files += 1
            data_total_size += p.stat().st_size
        elif p.is_dir():
            data_dirs += 1

    for p in archives_dir.rglob("*"):
        if p.is_file():
            archives_total_size += p.stat().st_size

    # Teeth3DS scan
    for jaw in ["upper", "lower"]:
        jaw_dir = teeth3ds_root / jaw
        if not jaw_dir.exists():
            continue
        ids = sorted([p.name for p in jaw_dir.iterdir() if p.is_dir()])
        teeth_ids[jaw] = set(ids)
        for idx, id_ in enumerate(ids, start=1):
            obj_path = jaw_dir / id_ / f"{id_}_{jaw}.obj"
            json_path = jaw_dir / id_ / f"{id_}_{jaw}.json"
            has_obj = obj_path.exists()
            has_json = json_path.exists()
            teeth_case: dict[str, Any] = {
                "id": id_,
                "jaw": jaw,
                "obj": str(obj_path.relative_to(repo_root)) if has_obj else None,
                "json": str(json_path.relative_to(repo_root)) if has_json else None,
            }

            if has_obj:
                obj_size = obj_path.stat().st_size
                teeth_case["obj_size_bytes"] = obj_size
                obj_size_by_jaw[jaw].append(obj_size)

                if not args.skip_mesh:
                    v, f = count_obj_vertices_faces(obj_path)
                    teeth_case["vertices"] = v
                    teeth_case["faces"] = f
                    mesh_vertices_by_jaw[jaw].append(v)
                    mesh_faces_by_jaw[jaw].append(f)

            if has_json:
                teeth_ids_with_json[jaw].add(id_)
                json_size = json_path.stat().st_size
                teeth_case["json_size_bytes"] = json_size
                json_size_by_jaw[jaw].append(json_size)

                if not args.skip_seg:
                    seg = parse_teeth3ds_seg_json(json_path)
                    teeth_case["seg"] = {
                        "labels_len": seg["labels_len"],
                        "instances_len": seg["instances_len"],
                        "unique_labels": seg["unique_labels"],
                        "unique_instances": seg["unique_instances"],
                        "teeth_instances": seg["teeth_instances"],
                        "teeth_labels": seg["teeth_labels"],
                        "label_values": seg["label_values"],
                        "instance_values": seg["instance_values"],
                    }
                    seg_teeth_instances_by_jaw[jaw].append(seg["teeth_instances"])
                    seg_label_values_by_jaw[jaw].update(seg["label_values"])
                    seg_label_hist_by_jaw[jaw].update({int(k): int(v) for k, v in seg["label_hist"].items()})
                    seg_inst_hist_by_jaw[jaw].update({int(k): int(v) for k, v in seg["instance_hist"].items()})

                    # quick consistency checks (only if mesh counted)
                    if not args.skip_mesh and "vertices" in teeth_case:
                        teeth_case["labels_match_vertices"] = (
                            seg["labels_len"] == teeth_case["vertices"] == seg["instances_len"]
                        )

            cases[CaseKey(id_, jaw).item] = teeth_case

            if idx % 100 == 0:
                print(f"[teeth3ds] {jaw}: {idx}/{len(ids)}")

    upper_ids = teeth_ids["upper"]
    lower_ids = teeth_ids["lower"]

    # Splits coverage
    split_files_info: dict[str, Any] = {}
    if splits_root.exists():
        for path in sorted([p for p in splits_root.rglob("*.txt") if p.is_file()]):
            rel = str(path.relative_to(repo_root))
            if path.name == "license.txt":
                continue
            items = parse_split_txt(path)
            unique_items = set(items)
            exists = 0
            has_json = 0
            missing_items: list[str] = []
            for it in unique_items:
                ck = item_to_case(it)
                if ck is None:
                    continue
                d = teeth3ds_root / ck.jaw / ck.id_
                obj_path = d / f"{ck.id_}_{ck.jaw}.obj"
                json_path = d / f"{ck.id_}_{ck.jaw}.json"
                if obj_path.exists():
                    exists += 1
                    if json_path.exists():
                        has_json += 1
                else:
                    missing_items.append(it)
            split_files_info[rel] = {
                "lines": len(items),
                "unique": len(unique_items),
                "dups": len(items) - len(unique_items),
                "exists": exists,
                "has_json": has_json,
                "missing_sample": sorted(missing_items)[:20],
            }

    # Landmarks kpt
    landmarks_points: list[int] = []
    landmarks_class_hist: Counter[str] = Counter()
    landmarks_by_split_jaw: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(dict))

    if not args.skip_landmarks and landmarks_root.exists():
        for split in ["train", "test"]:
            for jaw in ["upper", "lower"]:
                base = landmarks_root / split / jaw
                if not base.exists():
                    continue
                kpt_files = sorted(base.rglob("*__kpt.json"))
                ids = set()
                points_list: list[int] = []
                class_hist = Counter()
                for idx, kpt_path in enumerate(kpt_files, start=1):
                    # infer id from parent folder name (…/<ID>/<file>)
                    if kpt_path.parent and kpt_path.parent.name:
                        ids.add(kpt_path.parent.name)
                    kpt = parse_kpt_json(kpt_path)
                    pts = int(kpt["points"])
                    points_list.append(pts)
                    landmarks_points.append(pts)
                    class_hist.update(kpt["class_hist"])
                    landmarks_class_hist.update(kpt["class_hist"])
                    if idx % 50 == 0:
                        print(f"[landmarks] {split}/{jaw}: {idx}/{len(kpt_files)}")
                landmarks_by_split_jaw[split][jaw] = {
                    "files": len(kpt_files),
                    "ids": len(ids),
                    "points_summary": summarize_ints(points_list),
                    "class_values": sorted(class_hist.keys()),
                    "class_hist": dict(sorted(class_hist.items())),
                }

    # Build stats object
    def ratio_zero(hist: Counter[int]) -> float:
        total = sum(hist.values())
        if total == 0:
            return 0.0
        return hist.get(0, 0) / total

    teeth_stats = {
        "unique_ids": len(upper_ids | lower_ids),
        "both_jaws": len(upper_ids & lower_ids),
        "only_upper": len(upper_ids - lower_ids),
        "only_lower": len(lower_ids - upper_ids),
        "upper": {
            "ids_total": len(upper_ids),
            "ids_with_json": len(teeth_ids_with_json["upper"]),
            "mesh": {
                "obj_size_bytes_summary": summarize_ints(obj_size_by_jaw["upper"]),
                "vertices_summary": summarize_ints(mesh_vertices_by_jaw["upper"]) if not args.skip_mesh else None,
                "faces_summary": summarize_ints(mesh_faces_by_jaw["upper"]) if not args.skip_mesh else None,
            },
            "seg": {
                "json_size_bytes_summary": summarize_ints(json_size_by_jaw["upper"]),
                "label_values": sorted(seg_label_values_by_jaw["upper"]),
                "label0_ratio": ratio_zero(seg_label_hist_by_jaw["upper"]),
                "label_hist": dict(sorted(seg_label_hist_by_jaw["upper"].items())),
                "instance_hist": dict(sorted(seg_inst_hist_by_jaw["upper"].items())),
                "teeth_instances_summary": summarize_ints(seg_teeth_instances_by_jaw["upper"]),
            },
        },
        "lower": {
            "ids_total": len(lower_ids),
            "ids_with_json": len(teeth_ids_with_json["lower"]),
            "mesh": {
                "obj_size_bytes_summary": summarize_ints(obj_size_by_jaw["lower"]),
                "vertices_summary": summarize_ints(mesh_vertices_by_jaw["lower"]) if not args.skip_mesh else None,
                "faces_summary": summarize_ints(mesh_faces_by_jaw["lower"]) if not args.skip_mesh else None,
            },
            "seg": {
                "json_size_bytes_summary": summarize_ints(json_size_by_jaw["lower"]),
                "label_values": sorted(seg_label_values_by_jaw["lower"]),
                "label0_ratio": ratio_zero(seg_label_hist_by_jaw["lower"]),
                "label_hist": dict(sorted(seg_label_hist_by_jaw["lower"].items())),
                "instance_hist": dict(sorted(seg_inst_hist_by_jaw["lower"].items())),
                "teeth_instances_summary": summarize_ints(seg_teeth_instances_by_jaw["lower"]),
            },
        },
    }

    stats: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paths": {
            "repo_root": str(repo_root),
            "data_dir": str(data_dir.relative_to(repo_root)),
            "archives_dir": str(archives_dir.relative_to(repo_root)),
        },
        "sizes_bytes": {
            "data_total": data_total_size,
            "archives_total": archives_total_size,
        },
        "counts": {
            "data_files": data_files,
            "data_dirs": data_dirs,
            "teeth3ds_obj_files": sum(1 for _ in teeth3ds_root.rglob("*.obj")) if teeth3ds_root.exists() else 0,
            "teeth3ds_json_files": sum(1 for _ in teeth3ds_root.rglob("*.json")) if teeth3ds_root.exists() else 0,
            "landmarks_kpt_files": sum(1 for _ in landmarks_root.rglob("*__kpt.json")) if landmarks_root.exists() else 0,
        },
        "teeth3ds": teeth_stats,
        "splits": {"files": split_files_info},
        "landmarks": {
            "points_summary": summarize_ints(landmarks_points),
            "class_values": sorted(landmarks_class_hist.keys()),
            "class_hist": dict(sorted(landmarks_class_hist.items())),
            "by_split_jaw": landmarks_by_split_jaw,
        },
        "cases": cases,
    }

    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_stats_md(out_md, stats)
    print(f"wrote {out_json.relative_to(repo_root)}")
    print(f"wrote {out_md.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

