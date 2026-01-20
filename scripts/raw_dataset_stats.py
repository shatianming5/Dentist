#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import struct
import zipfile
import xml.etree.ElementTree as ET
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
        "p25": quantile(vals, 0.25),
        "median": statistics.median(vals),
        "p75": quantile(vals, 0.75),
        "p95": quantile(vals, 0.95),
        "max": vals[-1],
        "mean": total / len(vals),
        "sum": total,
    }


XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def col_to_index(col: str) -> int:
    idx = 0
    for ch in col:
        if not ch.isalpha():
            break
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def iter_xlsx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", XLSX_NS):
                texts = []
                for t in si.findall(".//m:t", XLSX_NS):
                    texts.append(t.text or "")
                shared.append("".join(texts))

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in zf.namelist():
            sheets = [
                n
                for n in zf.namelist()
                if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
            ]
            if not sheets:
                return []
            sheet_name = sorted(sheets)[0]

        root = ET.fromstring(zf.read(sheet_name))
        rows: list[list[str]] = []
        for row in root.findall(".//m:sheetData/m:row", XLSX_NS):
            cells: dict[int, str] = {}
            for c in row.findall("m:c", XLSX_NS):
                ref = c.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                if not col:
                    continue
                idx = col_to_index(col)
                t = c.get("t")

                if t == "s":
                    v = c.find("m:v", XLSX_NS)
                    raw = v.text if v is not None else ""
                    try:
                        val = shared[int(raw)] if raw else ""
                    except Exception:
                        val = raw or ""
                elif t == "inlineStr":
                    tnode = c.find("m:is/m:t", XLSX_NS)
                    val = tnode.text if tnode is not None and tnode.text else ""
                else:
                    v = c.find("m:v", XLSX_NS)
                    val = v.text if v is not None and v.text is not None else ""

                cells[idx] = val

            if not cells:
                continue
            max_i = max(cells)
            rows.append([cells.get(i, "") for i in range(max_i + 1)])
        return rows


def norm(s: str) -> str:
    return (s or "").strip()


@dataclass(frozen=True)
class LabeledItem:
    source: str  # "普通标注" | "专家标注"
    path: str
    filename: str
    label: str
    tooth_position: str | None = None  # 磨牙/前磨牙/…
    note: str | None = None
    index: int | None = None  # 普通标注的 1..166


def read_ccb2_header_version(path: Path) -> int | None:
    try:
        b = path.read_bytes()[:8]
    except Exception:
        return None
    if len(b) < 8 or b[:4] != b"CCB2":
        return None
    return struct.unpack("<I", b[4:8])[0]


def write_stats_md(out_path: Path, stats: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Raw 数据统计报告\n")
    lines.append(f"生成时间：{stats['generated_at_utc']}\n")
    lines.append("\n")

    lines.append("## 这是什么数据？\n")
    lines.append(
        "- `raw/**.bin` 以 `CCB2` 开头，是 CloudCompare 的二进制工程/实体格式（通常包含点云/网格等 3D 对象）。\n"
    )
    lines.append(
        "- 这里的 `普通标注/专家标注` 配套的 Excel（`*.xlsx`）提供了“修复体类型”等标签，可用于修复体识别/分类任务。\n"
    )
    lines.append("\n")

    lines.append("## 总览\n")
    lines.append(f"- 总文件数：{stats['counts']['total_files']}\n")
    lines.append(f"- bin 文件数：{stats['counts']['bin_files']}\n")
    lines.append(f"- xlsx 文件数：{stats['counts']['xlsx_files']}\n")
    lines.append(f"- raw 总大小：{human_bytes(stats['sizes_bytes']['raw_total'])}\n")
    lines.append("\n")

    for src in ["普通标注", "专家标注"]:
        info = stats["by_source"][src]
        lines.append(f"## {src}\n")
        lines.append(f"- bin：{info['counts']['bin_files']}，大小：{human_bytes(info['sizes_bytes']['bin_total'])}\n")
        lines.append(
            "- 单文件大小（bin）："
            f"count={info['sizes_bytes']['bin_size_summary']['count']}, "
            f"min={human_bytes(info['sizes_bytes']['bin_size_summary']['min'])}, "
            f"median={human_bytes(int(info['sizes_bytes']['bin_size_summary']['median']))}, "
            f"p95={human_bytes(int(info['sizes_bytes']['bin_size_summary']['p95']))}, "
            f"max={human_bytes(info['sizes_bytes']['bin_size_summary']['max'])}\n"
        )
        if info.get("labels"):
            lines.append("- 标签分布：\n")
            for k, v in info["labels"]["label_counts"].items():
                lines.append(f"  - {k}: {v}\n")
        if src == "普通标注" and info.get("labels"):
            lines.append("- 修复牙位（普通标注）：\n")
            for k, v in info["labels"]["tooth_position_counts"].items():
                lines.append(f"  - {k or '(空)'}: {v}\n")
        if info.get("mapping_issues"):
            issues = info["mapping_issues"]
            if issues.get("missing_files"):
                lines.append(f"- 未在 Excel 找到标签的 bin：{len(issues['missing_files'])}\n")
                lines.append(f"  - 示例：{issues['missing_files'][:10]}\n")
            if issues.get("duplicate_rows"):
                lines.append(f"- Excel 中同一文件重复标注：{len(issues['duplicate_rows'])}\n")
                lines.append(f"  - 示例：{issues['duplicate_rows'][:1]}\n")
        lines.append("\n")

    lines.append("## CloudCompare bin 头版本\n")
    for k, v in stats["ccb2_header_version_counts"].items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n")

    lines.append("## 可做任务建议\n")
    lines.append(
        "- 3D 修复体类型分类（多分类）：输入每个 `Group-*.bin` 的 3D 几何（需先转换/解析 CloudCompare bin），输出修复体类型（如：全冠/桩核冠/嵌体或高嵌体/树脂充填）。\n"
    )
    lines.append(
        "- 牙位属性辅助分类：普通标注提供 `磨牙/前磨牙`，可做多任务学习（修复体类型 + 牙位类别）。\n"
    )
    lines.append(
        "- 噪声标签/一致性研究：普通标注 vs 专家标注是两套独立数据，可用于研究不同标注质量下的鲁棒训练（但不是同一批样本的对照）。\n"
    )
    lines.append("\n")

    lines.append("## 输出文件\n")
    lines.append("- 机器可读统计：`RAW_DATASET_STATS.json`\n")
    lines.append("- 统计摘要：`RAW_DATASET_STATS.md`\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute statistics for D:\\\\dentist\\\\raw data.")
    parser.add_argument("--raw-dir", default="raw", help="Raw directory (default: raw)")
    parser.add_argument("--out-json", default="RAW_DATASET_STATS.json", help="Output JSON path")
    parser.add_argument("--out-md", default="RAW_DATASET_STATS.md", help="Output Markdown path")
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    out_json = (repo_root / args.out_json).resolve()
    out_md = (repo_root / args.out_md).resolve()

    if not raw_dir.exists():
        raise SystemExit(f"raw dir not found: {raw_dir}")

    total_files = 0
    raw_total_size = 0
    ext_counts = Counter()

    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        total_files += 1
        raw_total_size += p.stat().st_size
        ext_counts[p.suffix.lower() or "(noext)"] += 1

    # CloudCompare bin header versions
    ccb_versions = Counter()

    def scan_source(src: str) -> dict[str, Any]:
        src_dir = raw_dir / src
        bin_files = sorted(src_dir.glob("*.bin"))
        xlsx_files = sorted(src_dir.glob("*.xlsx"))

        sizes = [p.stat().st_size for p in bin_files]
        for p in bin_files:
            v = read_ccb2_header_version(p)
            if v is not None:
                ccb_versions[v] += 1

        info: dict[str, Any] = {
            "counts": {"bin_files": len(bin_files), "xlsx_files": len(xlsx_files)},
            "sizes_bytes": {
                "bin_total": sum(sizes),
                "bin_size_summary": summarize_ints(sizes),
            },
        }

        # Parse labels
        mapping_issues: dict[str, Any] = {}
        labeled_items: list[LabeledItem] = []

        if src == "普通标注":
            # numbering: trailing integer 1..166 in file name
            index_to_filename: dict[int, str] = {}
            for p in bin_files:
                m = re.search(r"(\d+)(?=\.bin$)", p.name)
                if m:
                    index_to_filename[int(m.group(1))] = p.name

            xlsx = src_dir / "修复体标记_corrected.xlsx"
            if xlsx.exists():
                rows = iter_xlsx_rows(xlsx)
                if rows:
                    header = [norm(x) for x in rows[0]]
                    col = {h: i for i, h in enumerate(header)}
                    for r in rows[1:]:
                        idx = norm(r[col["编号"]]) if "编号" in col and col["编号"] < len(r) else ""
                        if not idx:
                            continue
                        i = int(idx)
                        label = norm(r[col["修复体类型"]]) if "修复体类型" in col and col["修复体类型"] < len(r) else ""
                        tooth_pos = (
                            norm(r[col["修复牙位"]])
                            if "修复牙位" in col and col["修复牙位"] < len(r)
                            else ""
                        )
                        fname = index_to_filename.get(i)
                        if fname:
                            labeled_items.append(
                                LabeledItem(
                                    source=src,
                                    path=str((src_dir / fname).relative_to(repo_root)),
                                    filename=fname,
                                    label=label or "未标注",
                                    tooth_position=tooth_pos or None,
                                    index=i,
                                )
                            )
                        else:
                            mapping_issues.setdefault("missing_files", []).append(f"index={i}")

                labeled_indices = {li.index for li in labeled_items if li.index is not None}
                missing = [index_to_filename[i] for i in sorted(index_to_filename) if i not in labeled_indices]
                if missing:
                    mapping_issues["missing_files"] = missing

        elif src == "专家标注":
            xlsx = src_dir / "标注_corrected.xlsx"
            if xlsx.exists():
                rows = iter_xlsx_rows(xlsx)
                if rows:
                    header = [norm(x) for x in rows[0]]
                    col = {h: i for i, h in enumerate(header)}
                    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
                    for r in rows[1:]:
                        fname = norm(r[col["文件名"]]) if "文件名" in col and col["文件名"] < len(r) else ""
                        if not fname:
                            continue
                        idx = norm(r[col["编号"]]) if "编号" in col and col["编号"] < len(r) else ""
                        label = (
                            norm(r[col["修复体类型"]])
                            if "修复体类型" in col and col["修复体类型"] < len(r)
                            else ""
                        )
                        note = norm(r[col["备注"]]) if "备注" in col and col["备注"] < len(r) else ""
                        by_file[fname.lower()].append(
                            {"编号": idx, "文件名": fname, "修复体类型": label, "备注": note}
                        )

                    duplicates = {k: v for k, v in by_file.items() if len(v) > 1}
                    if duplicates:
                        mapping_issues["duplicate_rows"] = list(duplicates.items())[:10]

                    def score(row: dict[str, str]) -> int:
                        s = 0
                        if row.get("编号"):
                            s += 2
                        if row.get("修复体类型") and row["修复体类型"] != "未知":
                            s += 1
                        return s

                    chosen: dict[str, dict[str, str]] = {}
                    for k, rows_ in by_file.items():
                        best = rows_[0]
                        for r in rows_[1:]:
                            if score(r) > score(best):
                                best = r
                        chosen[k] = best

                    file_lookup = {p.name.lower(): p.name for p in bin_files}
                    for key, row in chosen.items():
                        fname = row["文件名"]
                        actual = file_lookup.get(fname.lower())
                        if not actual:
                            # fallback: case-insensitive match without trusting Excel case
                            actual = file_lookup.get(key)
                        if not actual:
                            mapping_issues.setdefault("missing_files", []).append(fname)
                            continue
                        labeled_items.append(
                            LabeledItem(
                                source=src,
                                path=str((src_dir / actual).relative_to(repo_root)),
                                filename=actual,
                                label=row.get("修复体类型") or "未知",
                                note=row.get("备注") or None,
                                index=int(row["编号"]) if row.get("编号", "").isdigit() else None,
                            )
                        )

                labeled_set = {li.filename.lower() for li in labeled_items}
                missing = sorted([p.name for p in bin_files if p.name.lower() not in labeled_set])
                if missing:
                    mapping_issues["missing_files"] = missing

        if labeled_items:
            label_counts = Counter(li.label for li in labeled_items)
            labels_info: dict[str, Any] = {
                "label_counts": dict(label_counts.most_common()),
                "labels": sorted(label_counts.keys()),
                "labeled_items": [li.__dict__ for li in labeled_items],
            }
            if src == "普通标注":
                pos_counts = Counter((li.tooth_position or "") for li in labeled_items)
                labels_info["tooth_position_counts"] = dict(pos_counts.most_common())

                # label x tooth_position
                ct: dict[str, dict[str, int]] = {}
                label_pos = defaultdict(Counter)
                for li in labeled_items:
                    label_pos[li.label][li.tooth_position or ""] += 1
                for label, c in label_pos.items():
                    ct[label] = dict(c)
                labels_info["label_x_tooth_position"] = ct

            info["labels"] = labels_info

        if mapping_issues:
            info["mapping_issues"] = mapping_issues
        return info

    by_source = {src: scan_source(src) for src in ["普通标注", "专家标注"]}

    stats: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paths": {"raw_dir": str(raw_dir.relative_to(repo_root))},
        "counts": {
            "total_files": total_files,
            "bin_files": int(ext_counts.get(".bin", 0)),
            "xlsx_files": int(ext_counts.get(".xlsx", 0)),
        },
        "sizes_bytes": {"raw_total": raw_total_size},
        "ext_counts": dict(ext_counts),
        "ccb2_header_version_counts": dict(sorted(ccb_versions.items())),
        "by_source": by_source,
    }

    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_stats_md(out_md, stats)
    print(f"wrote {out_json.relative_to(repo_root)}")
    print(f"wrote {out_md.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
