#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_stats(xs: list[int]) -> str:
    if not xs:
        return "n=0"
    xs2 = sorted(xs)
    return (
        f"n={len(xs2)}, min={xs2[0]}, p25={xs2[int(0.25*(len(xs2)-1))]}, "
        f"median={statistics.median(xs2)}, p75={xs2[int(0.75*(len(xs2)-1))]}, max={xs2[-1]}"
    )


@dataclass(frozen=True)
class CaseSummary:
    case_key: str
    label: str
    source: str
    n_clouds: int
    n_named: int
    n_segmented: int
    min_segmented_points: int | None
    max_segmented_points: int | None


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate converted/raw manifest and summarize extraction heuristics.")
    ap.add_argument("--manifest", type=Path, default=Path("converted/raw/manifest_with_labels.json"))
    ap.add_argument("--segmented-regex", type=str, default=r"segmented$", help="Regex to define 'segmented' clouds.")
    ap.add_argument("--out", type=Path, default=None, help="Optional output markdown path.")
    ap.add_argument("--topk-missing", type=int, default=20, help="Show top-K cases missing segmented clouds.")
    args = ap.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    seg_re = re.compile(str(args.segmented_regex))
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list):
        raise SystemExit("manifest_with_labels.json must be a list")

    rows: list[CaseSummary] = []
    clouds_per_case: list[int] = []
    seg_min_points: list[int] = []
    seg_max_points: list[int] = []
    missing_segmented: list[str] = []

    by_label_segmin: dict[str, list[int]] = defaultdict(list)
    by_source_segmin: dict[str, list[int]] = defaultdict(list)

    for entry in manifest:
        case_key = str(entry.get("input") or "").strip()
        if not case_key:
            continue
        label_info = entry.get("label_info") or {}
        label = str(label_info.get("label") or "未知").strip() or "未知"
        source = str(label_info.get("source") or "(missing)").strip() or "(missing)"

        exported = entry.get("exported_clouds") or []
        n_clouds = int(len(exported))
        clouds_per_case.append(n_clouds)

        seg_points: list[int] = []
        n_named = 0
        for c in exported:
            name = str(c.get("name") or "")
            if name:
                n_named += 1
            if seg_re.search(name):
                try:
                    seg_points.append(int(c.get("points") or 0))
                except Exception:
                    pass

        if seg_points:
            mn = min(seg_points)
            mx = max(seg_points)
            seg_min_points.append(mn)
            seg_max_points.append(mx)
            by_label_segmin[label].append(mn)
            by_source_segmin[source].append(mn)
        else:
            mn = None
            mx = None
            missing_segmented.append(case_key)

        rows.append(
            CaseSummary(
                case_key=case_key,
                label=label,
                source=source,
                n_clouds=n_clouds,
                n_named=n_named,
                n_segmented=len(seg_points),
                min_segmented_points=mn,
                max_segmented_points=mx,
            )
        )

    total_cases = len(rows)
    total_clouds = sum(r.n_clouds for r in rows)
    total_segmented = sum(r.n_segmented for r in rows)

    lines: list[str] = []
    lines.append("# converted/raw 提取质量检查")
    lines.append("")
    lines.append(f"- manifest: `{manifest_path}`")
    lines.append(f"- total_cases: {total_cases}")
    lines.append(f"- total_clouds: {total_clouds}")
    lines.append(f"- total_segmented_clouds: {total_segmented}")
    lines.append(f"- cases_with_no_segmented: {len(missing_segmented)}")
    lines.append("")
    lines.append("## 每个 case 导出的对象数")
    lines.append(f"- exported_clouds per case: {fmt_stats(clouds_per_case)}")
    lines.append("")
    lines.append("## segmented$ 对象点数（每个 case 的最小/最大）")
    lines.append(f"- min(segmented.points) per case: {fmt_stats(seg_min_points)}")
    lines.append(f"- max(segmented.points) per case: {fmt_stats(seg_max_points)}")
    lines.append("")
    lines.append("## 按 label 的 min(segmented.points)（Top 10）")
    for lab, xs in sorted(by_label_segmin.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]:
        lines.append(f"- {lab}: {fmt_stats(xs)}")
    lines.append("")
    lines.append("## 按 source 的 min(segmented.points)")
    for src, xs in sorted(by_source_segmin.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"- {src}: {fmt_stats(xs)}")
    lines.append("")
    if missing_segmented:
        lines.append(f"## 缺失 segmented$ 的样本（Top {int(args.topk_missing)}）")
        for ck in missing_segmented[: int(args.topk_missing)]:
            lines.append(f"- {ck}")
        if len(missing_segmented) > int(args.topk_missing):
            lines.append(f"- ... (还有 {len(missing_segmented) - int(args.topk_missing)} 个)")
        lines.append("")

    out = "\n".join(lines) + "\n"
    if args.out is not None:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
        print(f"[OK] wrote: {out_path}")
    else:
        print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
