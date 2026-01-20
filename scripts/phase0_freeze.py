#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DirStat:
    files: int
    bytes: int


def dir_stat(root: Path) -> DirStat:
    file_count = 0
    total_bytes = 0
    if not root.exists():
        return DirStat(files=0, bytes=0)
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            file_count += 1
            p = Path(dirpath) / name
            try:
                total_bytes += p.stat().st_size
            except OSError:
                continue
    return DirStat(files=file_count, bytes=total_bytes)


def read_lines(path: Path) -> list[str]:
    lines: list[str] = []
    if not path.exists():
        return lines
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if s:
            lines.append(s)
    return lines


def strip_jaw_suffix(item: str) -> str:
    s = str(item).strip()
    if s.endswith("_upper"):
        return s[: -len("_upper")]
    if s.endswith("_lower"):
        return s[: -len("_lower")]
    return s


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


def stratified_split(
    cases: list[dict[str, Any]],
    seed: int,
    ratios: tuple[float, float, float],
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    train_ratio, val_ratio, test_ratio = ratios
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        by_label[str(c.get(key, "未知"))].append(c)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for _label, group in sorted(by_label.items(), key=lambda kv: kv[0]):
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val : n_train + n_val + n_test])

    for split in splits.values():
        split.sort(key=lambda c: str(c.get("case_key", "")))
    return splits


def count_labels(cases: Iterable[dict[str, Any]], label_key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for c in cases:
        counter[str(c.get(label_key, "未知"))] += 1
    return dict(counter.most_common())


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0: freeze metadata (inventory + splits).")
    ap.add_argument("--root", type=Path, default=Path("."), help="Repo root (default: .)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for splits.")
    ap.add_argument("--raw-train", type=float, default=0.8, help="Raw train ratio.")
    ap.add_argument("--raw-val", type=float, default=0.1, help="Raw val ratio.")
    ap.add_argument("--raw-test", type=float, default=0.1, help="Raw test ratio.")
    ap.add_argument("--teeth3ds-val-ratio", type=float, default=0.1, help="Val ratio from official train.")
    args = ap.parse_args()

    root = args.root.resolve()
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now_iso()

    # ---------- inventory ----------
    disk = shutil.disk_usage(root)
    inv: dict[str, Any] = {
        "generated_at": generated_at,
        "root": str(root),
        "disk_usage": {
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        },
        "directories": {},
        "artifacts": {},
    }

    for name in ["data", "archives", "raw", "converted", "scripts"]:
        st = dir_stat(root / name)
        inv["directories"][name] = {"files": st.files, "bytes": st.bytes}

    artifact_paths = [
        root / "DATASET_STATS.json",
        root / "RAW_DATASET_STATS.json",
        root / "converted" / "raw" / "manifest_with_labels.json",
        root / "converted" / "raw" / "labels.csv",
        root / "scripts" / "convert_ccb2_bin.py",
        root / "scripts" / "label_converted_raw.py",
    ]
    for p in artifact_paths:
        if not p.exists():
            continue
        try:
            inv["artifacts"][str(p.relative_to(root))] = {
                "bytes": p.stat().st_size,
                "sha1": sha1_file(p),
            }
        except OSError:
            continue

    inv_path = meta_dir / "data_inventory.json"
    inv_path.write_text(json.dumps(inv, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- raw case splits ----------
    manifest_path = root / "converted" / "raw" / "manifest_with_labels.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise SystemExit("manifest_with_labels.json must be a list")

    raw_cases: list[dict[str, Any]] = []
    for entry in manifest:
        case_key = str(entry.get("input", "")).strip()
        if not case_key:
            continue
        label_info = entry.get("label_info") or {}
        label_raw = label_info.get("label")
        raw_cases.append(
            {
                "case_key": case_key,
                "source": label_info.get("source"),
                "label_raw": label_raw,
                "label": canonicalize_label(label_raw),
                "tooth_position": label_info.get("tooth_position"),
                "input_bytes": entry.get("input_bytes"),
            }
        )

    raw_cases.sort(key=lambda c: c["case_key"])
    raw_splits = stratified_split(
        raw_cases,
        seed=args.seed,
        ratios=(args.raw_train, args.raw_val, args.raw_test),
        key="label",
    )
    raw_split_out: dict[str, Any] = {
        "generated_at": generated_at,
        "seed": args.seed,
        "ratios": {"train": args.raw_train, "val": args.raw_val, "test": args.raw_test},
        "strategy": "case-level stratified split by canonical label; case_key == manifest[].input",
        "splits": {k: [c["case_key"] for c in v] for k, v in raw_splits.items()},
        "label_distribution": {k: count_labels(v, "label") for k, v in raw_splits.items()},
        "cases": [],
    }
    split_of: dict[str, str] = {}
    for split_name, items in raw_splits.items():
        for c in items:
            split_of[c["case_key"]] = split_name
    for c in raw_cases:
        raw_split_out["cases"].append({**c, "split": split_of.get(c["case_key"], "unknown")})

    (meta_dir / "splits_raw_case.json").write_text(
        json.dumps(raw_split_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # ---------- teeth3ds splits ----------
    teeth3ds_dir = root / "data" / "splits" / "Teeth3DS_train_test_split"
    training_upper = read_lines(teeth3ds_dir / "training_upper.txt")
    testing_upper = read_lines(teeth3ds_dir / "testing_upper.txt")
    training_lower = read_lines(teeth3ds_dir / "training_lower.txt")
    testing_lower = read_lines(teeth3ds_dir / "testing_lower.txt")

    def derived_from_official_train(official_train: list[str]) -> dict[str, list[str]]:
        rng = random.Random(args.seed)
        items = list(official_train)
        rng.shuffle(items)
        n_val = int(len(items) * float(args.teeth3ds_val_ratio))
        val = sorted(items[:n_val])
        train = sorted(items[n_val:])
        return {"train": train, "val": val}

    teeth3ds_out: dict[str, Any] = {
        "generated_at": generated_at,
        "seed": args.seed,
        "official": {
            "upper": {"train": sorted(training_upper), "test": sorted(testing_upper)},
            "lower": {"train": sorted(training_lower), "test": sorted(testing_lower)},
        },
        "derived": {
            "upper": {**derived_from_official_train(training_upper), "test": sorted(testing_upper)},
            "lower": {**derived_from_official_train(training_lower), "test": sorted(testing_lower)},
        },
    }

    # Patient-level split to avoid cross-jaw leakage when training on both jaws.
    upper_train_ids = {strip_jaw_suffix(x) for x in training_upper}
    upper_test_ids = {strip_jaw_suffix(x) for x in testing_upper}
    lower_train_ids = {strip_jaw_suffix(x) for x in training_lower}
    lower_test_ids = {strip_jaw_suffix(x) for x in testing_lower}

    official_test_patients = sorted(upper_test_ids | lower_test_ids)
    official_train_candidates = sorted((upper_train_ids | lower_train_ids) - set(official_test_patients))
    overlap_train_test = sorted((upper_train_ids | lower_train_ids) & (upper_test_ids | lower_test_ids))

    rng_pat = random.Random(args.seed)
    train_patients = list(official_train_candidates)
    rng_pat.shuffle(train_patients)
    n_val = int(len(train_patients) * float(args.teeth3ds_val_ratio))
    patient_val = sorted(train_patients[:n_val])
    patient_train = sorted(train_patients[n_val:])

    teeth3ds_out["patient"] = {
        "strategy": "patient-level: patient_id is case_id without _upper/_lower; "
        "test_patients are any patient appearing in official test (either jaw); "
        "val sampled from remaining train patients",
        "official": {"train": official_train_candidates, "test": official_test_patients},
        "derived": {"train": patient_train, "val": patient_val, "test": official_test_patients},
        "overlap_train_test_patients": overlap_train_test,
        "counts": {
            "upper_train": len(upper_train_ids),
            "upper_test": len(upper_test_ids),
            "lower_train": len(lower_train_ids),
            "lower_test": len(lower_test_ids),
            "patient_train_candidates": len(official_train_candidates),
            "patient_test": len(official_test_patients),
            "patient_overlap_train_test": len(overlap_train_test),
        },
    }

    # Optional: scan available cases (fast directory listing)
    for jaw in ["upper", "lower"]:
        jaw_dir = root / "data" / "teeth3ds" / jaw
        ids: list[str] = []
        if jaw_dir.exists():
            for child in jaw_dir.iterdir():
                if child.is_dir():
                    ids.append(child.name)
        teeth3ds_out.setdefault("available", {})[jaw] = sorted(f"{i}_{jaw}" for i in ids)

    (meta_dir / "splits_teeth3ds.json").write_text(
        json.dumps(teeth3ds_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Wrote: {inv_path}")
    print(f"[OK] Wrote: {meta_dir / 'splits_raw_case.json'}")
    print(f"[OK] Wrote: {meta_dir / 'splits_teeth3ds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
