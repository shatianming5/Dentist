#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_contains(path: Path, required: list[str]) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, [f"missing file: {path}"]
    txt = file_text(path)
    missing = [s for s in required if s not in txt]
    return (len(missing) == 0), missing


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # ok|warn|fail
    details: list[str]


def _status(ok: bool, *, warn: bool = False) -> str:
    if ok:
        return "ok"
    return "warn" if warn else "fail"


def audit_teeth3ds_patient_splits(root: Path) -> Check:
    path = root / "metadata/splits_teeth3ds.json"
    if not path.exists():
        return Check("teeth3ds_patient_splits", "fail", [f"missing: {path}"])
    obj = read_json(path)
    patient = (obj.get("patient") or {}).get("derived") or {}
    if not patient:
        return Check("teeth3ds_patient_splits", "fail", ["missing patient.derived in splits_teeth3ds.json"])
    train = set(str(x) for x in (patient.get("train") or []))
    val = set(str(x) for x in (patient.get("val") or []))
    test = set(str(x) for x in (patient.get("test") or []))
    missing_keys = [k for k in ["train", "val", "test"] if not (patient.get(k) or [])]
    if missing_keys:
        return Check("teeth3ds_patient_splits", "fail", [f"missing keys in patient.derived: {missing_keys}"])
    overlap = {
        "train∩val": sorted(train & val),
        "train∩test": sorted(train & test),
        "val∩test": sorted(val & test),
    }
    bad = {k: v for k, v in overlap.items() if v}
    details = [
        f"path: {path}",
        f"patients: train={len(train)} val={len(val)} test={len(test)}",
    ]
    if bad:
        details.append(f"overlap: { {k: len(v) for k, v in bad.items()} }")
        return Check("teeth3ds_patient_splits", "fail", details)
    return Check("teeth3ds_patient_splits", "ok", details)


def audit_teeth3ds_processed_index(root: Path, data_root: Path) -> Check:
    index_path = data_root / "index.jsonl"
    if not index_path.exists():
        return Check("teeth3ds_processed_index", "warn", [f"missing: {index_path}"])

    rows_by_split: dict[str, int] = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    patients_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set(), "unknown": set()}
    bad_split_mode = 0
    for r in iter_jsonl(index_path):
        split = str(r.get("split") or "unknown")
        if split not in rows_by_split:
            split = "unknown"
        rows_by_split[split] += 1
        pid = str(r.get("id_patient") or "")
        if not pid:
            ck = str(r.get("case_key") or "")
            pid = ck.rsplit("_", 1)[0] if "_" in ck else ck
        if pid:
            patients_by_split[split].add(pid)
        if split in {"train", "val", "test"}:
            if str(r.get("split_mode") or "").strip().lower() != "patient":
                bad_split_mode += 1

    train = patients_by_split["train"]
    val = patients_by_split["val"]
    test = patients_by_split["test"]
    details = [
        f"path: {index_path}",
        f"rows: {rows_by_split}",
        f"patients: train={len(train)} val={len(val)} test={len(test)} unknown={len(patients_by_split['unknown'])}",
    ]
    if bad_split_mode:
        details.append(f"bad split_mode (!=patient) rows: {bad_split_mode}")
        return Check("teeth3ds_processed_index", "fail", details)
    if (train & test) or (val & test) or (train & val):
        details.append(
            f"patient_overlap sizes: train∩val={len(train & val)} train∩test={len(train & test)} val∩test={len(val & test)}"
        )
        return Check("teeth3ds_processed_index", "fail", details)
    return Check("teeth3ds_processed_index", "ok", details)


def audit_raw_cls_dataset(root: Path, data_root: Path) -> Check:
    index_path = data_root / "index.jsonl"
    label_map_path = data_root / "label_map.json"
    if not index_path.exists() or not label_map_path.exists():
        missing = [str(p) for p in [index_path, label_map_path] if not p.exists()]
        return Check("raw_cls_dataset", "warn", [f"missing: {', '.join(missing)}"])

    label_to_id = {str(k): int(v) for k, v in read_json(label_map_path).items()}
    labels = sorted(label_to_id.keys(), key=lambda x: label_to_id[x])
    train_counts = {lab: 0 for lab in labels}
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    unknown_labels: dict[str, int] = {}
    for r in iter_jsonl(index_path):
        split = str(r.get("split") or "unknown")
        if split not in split_counts:
            split = "unknown"
        split_counts[split] += 1
        lab = str(r.get("label") or "未知")
        if lab not in label_to_id:
            unknown_labels[lab] = int(unknown_labels.get(lab, 0)) + 1
            continue
        if split == "train":
            train_counts[lab] += 1

    details = [f"path: {data_root}", f"splits: {split_counts}", f"labels: {labels}"]
    if unknown_labels:
        details.append(f"unknown_labels_in_index: {unknown_labels}")
        return Check("raw_cls_dataset", "fail", details)
    zero_train = [lab for lab, n in train_counts.items() if int(n) <= 0]
    if zero_train:
        details.append(f"labels_with_zero_train: {zero_train}")
        return Check("raw_cls_dataset", "fail", details)
    return Check("raw_cls_dataset", "ok", details)


def audit_kfold_file(root: Path, kfold_path: Path) -> Check:
    if not kfold_path.exists():
        return Check("raw_kfold_file", "warn", [f"missing: {kfold_path} (can be generated)"])
    obj = read_json(kfold_path)
    k = int(obj.get("k") or 0)
    case_to_fold = obj.get("case_to_fold") or {}
    folds = obj.get("folds") or {}
    details = [f"path: {kfold_path}", f"k={k}", f"case_to_fold={len(case_to_fold)} folds={len(folds)}"]
    if k < 2:
        return Check("raw_kfold_file", "fail", details + ["invalid k (must be >=2)"])
    if not case_to_fold and not folds:
        return Check("raw_kfold_file", "fail", details + ["missing case_to_fold/folds mapping"])
    return Check("raw_kfold_file", "ok", details)


def audit_code_presence(root: Path) -> list[Check]:
    checks: list[Check] = []

    p3 = root / "scripts/phase3_train_raw_cls_baseline.py"
    ok, missing = check_contains(
        p3,
        [
            'choices=["pointnet", "dgcnn", "meta_mlp", "geom_mlp"]',
            "--balanced-sampler",
            "--label-smoothing",
            'ap.add_argument("--kfold", type=Path, default=None',
            'ap.add_argument("--init-feat", type=Path, default=None',
        ],
    )
    checks.append(Check("phase3_raw_cls_baseline_features", _status(ok), [f"path: {p3}"] + ([f"missing: {missing}"] if not ok else [])))

    p2_ae = root / "scripts/phase2_train_raw_ae.py"
    checks.append(Check("raw_ae_pretrain_script", _status(p2_ae.exists()), [f"path: {p2_ae}"]))

    p2_build = root / "scripts/phase2_build_teeth3ds_teeth.py"
    ok, missing = check_contains(p2_build, ["--pca-align-globalz", "pca_align_globalz"])
    checks.append(Check("pca_align_globalz_option", _status(ok), [f"path: {p2_build}"] + ([f"missing: {missing}"] if not ok else [])))

    p3_p2t = root / "scripts/phase3_train_teeth3ds_prep2target.py"
    ok, missing = check_contains(p3_p2t, ["--cut-mode", 'choices=["z", "plane"]'])
    checks.append(Check("prep2target_cut_mode_plane", _status(ok), [f"path: {p3_p2t}"] + ([f"missing: {missing}"] if not ok else [])))

    p4_constraints = root / "scripts/phase4_train_teeth3ds_prep2target_constraints.py"
    ok, missing = check_contains(
        p4_constraints,
        [
            "--cut-mode",
            "--occlusion-mode",
            'choices=["jaw", "tooth"]',
            "occlusion_min_d_p05",
            "occlusion_contact_ratio",
            'write_json(out_dir / "env.json"',
        ],
    )
    checks.append(Check("constraints_training_features", _status(ok), [f"path: {p4_constraints}"] + ([f"missing: {missing}"] if not ok else [])))

    p4_eval = root / "scripts/phase4_eval_teeth3ds_constraints_run.py"
    ok, missing = check_contains(p4_eval, ["--occlusion-mode", "occlusion_mode", "--cut-mode", "_opp_fdi"])
    checks.append(Check("constraints_eval_features", _status(ok), [f"path: {p4_eval}"] + ([f"missing: {missing}"] if not ok else [])))

    diag = [
        root / "scripts/validate_converted_raw.py",
        root / "scripts/check_teeth3ds_jaw_alignment.py",
        root / "scripts/summarize_raw_cls_runs.py",
        root / "scripts/phase4_summarize_constraints_runs.py",
    ]
    missing_diag = [str(p) for p in diag if not p.exists()]
    checks.append(Check("diagnostic_scripts", _status(len(missing_diag) == 0), ([f"missing: {missing_diag}"] if missing_diag else [f"ok: {len(diag)} scripts"])))

    req = root / "requirements.txt"
    checks.append(Check("requirements_txt", _status(req.exists()), [f"path: {req}"]))

    return checks


def to_markdown(checks: list[Check], *, root: Path) -> str:
    ok_n = sum(1 for c in checks if c.status == "ok")
    warn_n = sum(1 for c in checks if c.status == "warn")
    fail_n = sum(1 for c in checks if c.status == "fail")
    lines: list[str] = []
    lines.append("# Journal audit report")
    lines.append("")
    lines.append(f"- generated_at: {utc_now_iso()}")
    lines.append(f"- root: `{root}`")
    lines.append(f"- python: `{sys.version.split()[0]}`")
    lines.append(f"- platform: `{platform.platform()}`")
    lines.append(f"- summary: ok={ok_n} warn={warn_n} fail={fail_n}")
    lines.append("")
    lines.append("| check | status | notes |")
    lines.append("|---|---|---|")
    for c in checks:
        notes = "<br>".join(c.details) if c.details else ""
        lines.append(f"| `{c.name}` | {c.status} | {notes} |")
    lines.append("")
    if fail_n:
        lines.append("## Next actions")
        lines.append("- Fix the failed items above, then rerun `python3 scripts/journal_audit.py --root . --out JOURNAL_AUDIT.md`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit whether the repo matches the 12-fix journal-readiness checklist.")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--teeth3ds-data-root", type=Path, default=Path("processed/teeth3ds_teeth/v1"))
    ap.add_argument("--raw-cls-root", type=Path, default=Path("processed/raw_cls/v13_main4"))
    ap.add_argument("--kfold", type=Path, default=Path("metadata/splits_raw_case_kfold.json"))
    ap.add_argument("--out", type=Path, default=None, help="Optional markdown output path.")
    args = ap.parse_args()

    root = args.root.resolve()
    teeth3ds_root = (root / args.teeth3ds_data_root).resolve()
    raw_cls_root = (root / args.raw_cls_root).resolve()
    kfold_path = (root / args.kfold).resolve()

    checks: list[Check] = []
    checks.append(audit_teeth3ds_patient_splits(root))
    checks.append(audit_teeth3ds_processed_index(root, teeth3ds_root))
    checks.append(audit_raw_cls_dataset(root, raw_cls_root))
    checks.append(audit_kfold_file(root, kfold_path))
    checks.extend(audit_code_presence(root))

    md = to_markdown(checks, root=root)
    if args.out is not None:
        out_path = args.out.resolve() if args.out.is_absolute() else (root / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"[OK] wrote: {out_path}")
    else:
        print(md, end="")

    if any(c.status == "fail" for c in checks):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
