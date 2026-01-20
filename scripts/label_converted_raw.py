#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach raw (修复体) labels to converted manifest.")
    parser.add_argument("--raw-stats", default="RAW_DATASET_STATS.json", help="Input raw stats JSON")
    parser.add_argument("--manifest", default="converted/raw/manifest.json", help="Converted manifest JSON")
    parser.add_argument("--out-manifest", default="converted/raw/manifest_with_labels.json", help="Output manifest JSON")
    parser.add_argument("--out-labels-json", default="converted/raw/labels.json", help="Output labels JSON")
    parser.add_argument("--out-labels-csv", default="converted/raw/labels.csv", help="Output labels CSV")
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    raw_stats_path = (repo_root / args.raw_stats).resolve()
    manifest_path = (repo_root / args.manifest).resolve()

    stats: dict[str, Any] = json.loads(raw_stats_path.read_text(encoding="utf-8"))
    manifest: list[dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))

    label_map: dict[str, dict[str, Any]] = {}
    for src, src_info in stats.get("by_source", {}).items():
        items = src_info.get("labels", {}).get("labeled_items", [])
        for item in items:
            raw_path = str(item.get("path", ""))
            if raw_path.startswith("raw/"):
                raw_path = raw_path[4:]
            label_map[raw_path] = {
                "source": item.get("source", src),
                "label": item.get("label"),
                "tooth_position": item.get("tooth_position"),
                "note": item.get("note"),
                "index": item.get("index"),
                "filename": item.get("filename"),
            }

    out_manifest: list[dict[str, Any]] = []
    for entry in manifest:
        input_rel = entry.get("input")
        label_info = label_map.get(input_rel)
        out_entry = dict(entry)
        out_entry["label_info"] = label_info
        out_manifest.append(out_entry)

    out_manifest_path = (repo_root / args.out_manifest).resolve()
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_path.write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    out_labels_json_path = (repo_root / args.out_labels_json).resolve()
    out_labels_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_labels_json_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    out_labels_csv_path = (repo_root / args.out_labels_csv).resolve()
    out_labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_labels_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["input", "source", "label", "tooth_position", "note", "index", "filename"],
        )
        w.writeheader()
        for input_rel, info in sorted(label_map.items()):
            w.writerow({"input": input_rel, **info})

    print(f"wrote {out_manifest_path.relative_to(repo_root)}")
    print(f"wrote {out_labels_json_path.relative_to(repo_root)}")
    print(f"wrote {out_labels_csv_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

