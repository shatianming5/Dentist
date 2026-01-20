#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # ok|warn|fail
    note: str


def exists_check(path: Path, name: str) -> Check:
    if path.exists():
        return Check(name=name, status="ok", note=f"`{path}`")
    return Check(name=name, status="fail", note=f"missing `{path}`")


def contains_check(path: Path, name: str, needles: list[str]) -> Check:
    if not path.exists():
        return Check(name=name, status="fail", note=f"missing `{path}`")
    txt = read_text(path)
    missing = [s for s in needles if s not in txt]
    if missing:
        return Check(name=name, status="fail", note=f"`{path}` missing: {missing}")
    return Check(name=name, status="ok", note=f"`{path}`")


def to_md(root: Path, checks: list[Check]) -> str:
    ok_n = sum(1 for c in checks if c.status == "ok")
    fail_n = sum(1 for c in checks if c.status == "fail")
    warn_n = sum(1 for c in checks if c.status == "warn")
    lines: list[str] = []
    lines.append("# Paper readiness audit (non-data)")
    lines.append("")
    lines.append(f"- generated_at: {utc_now_iso()}")
    lines.append(f"- root: `{root}`")
    lines.append(f"- summary: ok={ok_n} warn={warn_n} fail={fail_n}")
    lines.append("")
    lines.append("| check | status | note |")
    lines.append("|---|---|---|")
    for c in checks:
        lines.append(f"| `{c.name}` | {c.status} | {c.note} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit non-data paper readiness: protocol/docs/tools for robust reporting.")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("PAPER_AUDIT.md"))
    args = ap.parse_args()

    root = args.root.resolve()
    checks: list[Check] = []

    checks.append(exists_check(root / "PAPER_SCOPE.md", "paper_scope"))
    checks.append(exists_check(root / "PAPER_PROTOCOL.md", "paper_protocol"))
    checks.append(exists_check(root / "environment.yml", "environment_yml"))

    checks.append(exists_check(root / "scripts/run_raw_cls_kfold.py", "kfold_runner"))
    checks.append(exists_check(root / "scripts/raw_cls_bootstrap_ci.py", "bootstrap_ci"))
    checks.append(exists_check(root / "scripts/paper_table_raw_cls.py", "paper_table_raw_cls"))
    checks.append(exists_check(root / "scripts/run_constraints_eval_suite.py", "constraints_eval_suite"))
    checks.append(exists_check(root / "scripts/phase4_summarize_constraints_runs.py", "constraints_summarizer"))

    checks.append(
        contains_check(
            root / "scripts/phase3_train_raw_cls_baseline.py",
            "raw_cls_reporting",
            [
                "--tta",
                "test_by_source_calibration",
                "test_calibration",
                "test_by_tooth_position",
            ],
        )
    )
    checks.append(
        contains_check(
            root / "scripts/phase4_eval_teeth3ds_constraints_run.py",
            "constraints_metrics",
            [
                "occlusion_contact_ratio",
                "occlusion_min_d_p05",
                "occlusion_min_d_p50",
                "occlusion_min_d_p95",
            ],
        )
    )

    out = to_md(root, checks)
    out_path = args.out.resolve() if args.out.is_absolute() else (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")
    print(f"[OK] wrote: {out_path}")

    return 2 if any(c.status == "fail" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
