#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Create stratified K-fold splits for raw cases (case_key == manifest[].input).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--manifest", type=Path, default=Path("converted/raw/manifest_with_labels.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--merge-extraction-to-unknown", action="store_true", help="Map label '拔除' to '未知'.")
    ap.add_argument("--out", type=Path, default=Path("metadata/splits_raw_case_kfold.json"))
    args = ap.parse_args()

    root = args.root.resolve()
    manifest_path = (root / args.manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    k = int(args.k)
    if k < 2:
        raise SystemExit("--k must be >= 2")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise SystemExit("manifest_with_labels.json must be a list")

    cases: list[dict[str, Any]] = []
    for entry in manifest:
        case_key = str(entry.get("input", "")).strip()
        if not case_key:
            continue
        label_info = entry.get("label_info") or {}
        label_raw = label_info.get("label")
        label = canonicalize_label(label_raw)
        if args.merge_extraction_to_unknown and label == "拔除":
            label = "未知"
        cases.append({"case_key": case_key, "label": label, "label_raw": label_raw, "source": label_info.get("source")})

    # Stratified assignment: shuffle each label group then round-robin into folds.
    rng = random.Random(int(args.seed))
    by_label: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        by_label[str(c["label"])].append(str(c["case_key"]))
    for group in by_label.values():
        rng.shuffle(group)

    folds: dict[int, list[str]] = {i: [] for i in range(k)}
    for _lab, group in sorted(by_label.items(), key=lambda kv: kv[0]):
        for i, case_key in enumerate(group):
            folds[i % k].append(case_key)

    for i in range(k):
        folds[i] = sorted(folds[i])

    case_to_fold: dict[str, int] = {}
    for i in range(k):
        for ck in folds[i]:
            case_to_fold[ck] = i

    label_dist: dict[str, dict[str, int]] = {}
    for i in range(k):
        ctr: Counter[str] = Counter()
        for ck in folds[i]:
            lab = next((c["label"] for c in cases if c["case_key"] == ck), "未知")
            ctr[str(lab)] += 1
        label_dist[str(i)] = dict(ctr.most_common())

    out_obj = {
        "generated_at": utc_now_iso(),
        "seed": int(args.seed),
        "k": int(k),
        "strategy": "stratified round-robin by canonical label over manifest_with_labels.json",
        "merge_extraction_to_unknown": bool(args.merge_extraction_to_unknown),
        "folds": {str(i): folds[i] for i in range(k)},
        "case_to_fold": case_to_fold,
        "label_distribution": label_dist,
        "total_cases": len(cases),
    }

    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

