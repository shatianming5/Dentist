#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_patient_split_map(splits_path: Path) -> dict[str, str]:
    obj = read_json(splits_path)
    patient = (obj.get("patient") or {}).get("derived") or {}
    if not patient:
        raise SystemExit(f"No patient-level splits found in: {splits_path}")
    m: dict[str, str] = {}
    for split_name, ids in patient.items():
        for pid in ids:
            for jaw in ["upper", "lower"]:
                m[f"{pid}_{jaw}"] = str(split_name)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="Rewrite processed/teeth3ds_teeth index.jsonl split field using patient-level splits.")
    ap.add_argument("--data-root", type=Path, default=Path("processed/teeth3ds_teeth/v1"))
    ap.add_argument("--splits", type=Path, default=Path("metadata/splits_teeth3ds.json"))
    ap.add_argument("--inplace", action="store_true", help="Overwrite index.jsonl (creates index.jsonl.bak).")
    ap.add_argument("--out", type=Path, default=None, help="Output path when not using --inplace.")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    index_path = data_root / "index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"Missing index.jsonl: {index_path}")

    split_map = build_patient_split_map(args.splits.resolve())
    rows = read_jsonl(index_path)
    updated: list[dict[str, Any]] = []
    changed = 0
    for r in rows:
        ck = str(r.get("case_key") or "")
        new_split = split_map.get(ck, "unknown")
        old_split = str(r.get("split") or "unknown")
        if old_split != new_split:
            changed += 1
        updated.append({**r, "split": new_split, "split_mode": "patient"})

    if args.inplace:
        bak = index_path.with_suffix(index_path.suffix + ".bak")
        if not bak.exists():
            index_path.replace(bak)
        else:
            # If backup already exists, keep it and overwrite index.jsonl.
            pass
        write_jsonl(index_path, updated)
        print(f"[OK] wrote: {index_path} (backup: {bak})")
    else:
        out_path = args.out.resolve() if args.out is not None else (data_root / "index_patient.jsonl")
        write_jsonl(out_path, updated)
        print(f"[OK] wrote: {out_path}")

    print(f"[OK] rows: {len(updated)}, changed_splits: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
