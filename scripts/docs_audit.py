#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


ID_RE = {
    "C": re.compile(r"\bC\d{4}\b"),
    "P": re.compile(r"\bP\d{4}\b"),
    "M": re.compile(r"\bM\d{4}\b"),
    "E": re.compile(r"\bE\d{4}\b"),
}


@dataclass(frozen=True)
class Finding:
    level: str  # ok|warn|fail
    name: str
    detail: str


def _uniq(ids: list[str]) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    dup: list[str] = []
    uniq: list[str] = []
    for x in ids:
        if x in seen:
            dup.append(x)
            continue
        seen.add(x)
        uniq.append(x)
    return uniq, dup


def extract_ids(text: str, kind: str) -> list[str]:
    return ID_RE[kind].findall(text)


def extract_defined_ids(md: str, prefix: str) -> list[str]:
    """
    Extract IDs defined as top-level checkbox items, e.g.:
      - [ ] C0001: ...
    """
    out: list[str] = []
    pat = re.compile(rf"^- \[(?: |x|X)\] ({re.escape(prefix)}\d{{4}}):", re.MULTILINE)
    for m in pat.finditer(md):
        out.append(m.group(1))
    return out


def extract_claim_evidence_eids(plan_md: str) -> list[str]:
    """
    Extract E#### ids from "Evidence:" lines under claims.
    """
    out: set[str] = set()
    for line in plan_md.splitlines():
        s = line.strip()
        if s.lower().startswith("- evidence:"):
            for eid in ID_RE["E"].findall(s):
                out.add(eid)
    return sorted(out)


def check_required_sections(path: Path, required_headers: list[str]) -> list[Finding]:
    if not path.exists():
        return [Finding("fail", "missing_file", f"missing `{path}`")]
    txt = read_text(path)
    out: list[Finding] = []
    for h in required_headers:
        if h not in txt:
            out.append(Finding("fail", "missing_section", f"`{path}` missing header: {h!r}"))
    if not out:
        out.append(Finding("ok", "sections", f"`{path}` required headers present"))
    return out


def parse_experiment_table(experiment_md: str) -> dict[str, dict[str, object]]:
    """
    Parse the first markdown table under "## Experiment List".
    Returns: EID -> {"smoke": bool, "full": bool, "row": str}
    """
    lines = experiment_md.splitlines()
    start = None
    header = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| ID |"):
            start = i
            header = line
            break
    if start is None or header is None:
        return {}

    cols = [c.strip() for c in header.strip().strip("|").split("|")]
    try:
        smoke_idx = cols.index("Smoke")
        full_idx = cols.index("Full")
        id_idx = cols.index("ID")
    except ValueError:
        return {}

    out: dict[str, dict[str, object]] = {}
    for line in lines[start + 2 :]:
        s = line.strip()
        if not s.startswith("|"):
            break
        cells = [c.strip() for c in s.strip().strip("|").split("|")]
        if len(cells) < max(smoke_idx, full_idx, id_idx) + 1:
            continue
        eid = cells[id_idx]
        if not re.fullmatch(r"E\d{4}", eid):
            continue
        smoke = "[x]" in cells[smoke_idx].lower()
        full = "[x]" in cells[full_idx].lower()
        out[eid] = {"smoke": smoke, "full": full, "row": line}
    return out


def to_md(root: Path, findings: list[Finding]) -> str:
    ok_n = sum(1 for f in findings if f.level == "ok")
    warn_n = sum(1 for f in findings if f.level == "warn")
    fail_n = sum(1 for f in findings if f.level == "fail")
    lines: list[str] = []
    lines.append("# Docs audit (docs-spec)")
    lines.append("")
    lines.append(f"- generated_at: {utc_now_iso()}")
    lines.append(f"- root: `{root}`")
    lines.append(f"- summary: ok={ok_n} warn={warn_n} fail={fail_n}")
    lines.append("")
    lines.append("| level | check | detail |")
    lines.append("|---|---|---|")
    for f in findings:
        lines.append(f"| {f.level} | `{f.name}` | {f.detail} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit docs/{plan,mohu,experiment}.md against docs-spec (minimal).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("docs/DOCS_AUDIT.md"))
    args = ap.parse_args()

    root = args.root.resolve()
    plan_path = root / "docs/plan.md"
    mohu_path = root / "docs/mohu.md"
    exp_path = root / "docs/experiment.md"

    findings: list[Finding] = []

    findings.extend(
        check_required_sections(
            plan_path,
            [
                "# Plan",
                "## Goals",
                "## Claims (C####)",
                "## Plan Items (P####)",
                "## Changelog",
            ],
        )
    )
    findings.extend(
        check_required_sections(
            mohu_path,
            [
                "# Mohu",
                "## 1. Not Implemented",
                "## 2. Ambiguities",
            ],
        )
    )
    findings.extend(
        check_required_sections(
            exp_path,
            [
                "# Experiments",
                "## Summary",
                "## Experiment List",
            ],
        )
    )

    if plan_path.exists():
        plan_txt = read_text(plan_path)
        claim_ids, dup = _uniq(extract_defined_ids(plan_txt, "C"))
        if dup:
            findings.append(Finding("fail", "claim_ids_unique", f"duplicate C IDs in Claims section: {sorted(set(dup))}"))
        else:
            findings.append(Finding("ok", "claim_ids_unique", f"claims={len(claim_ids)}"))

        plan_item_ids, dup_p = _uniq(extract_defined_ids(plan_txt, "P"))
        if dup_p:
            findings.append(
                Finding("fail", "plan_item_ids_unique", f"duplicate P IDs in Plan Items section: {sorted(set(dup_p))}")
            )
        else:
            findings.append(Finding("ok", "plan_item_ids_unique", f"plan_items={len(plan_item_ids)}"))

        evidence_eids = extract_claim_evidence_eids(plan_txt)
    else:
        evidence_eids = []

    if mohu_path.exists():
        mohu_txt = read_text(mohu_path)
        mohu_items, dup_m = _uniq(extract_defined_ids(mohu_txt, "M"))
        if dup_m:
            findings.append(Finding("fail", "mohu_ids_unique", f"duplicate M IDs: {sorted(set(dup_m))}"))
        else:
            findings.append(Finding("ok", "mohu_ids_unique", f"mohu_items={len(mohu_items)}"))

        # Ensure each Mohu item has a Verification line (within its block).
        for mid in mohu_items:
            start_pat = re.compile(rf"^- \[(?: |x|X)\] {re.escape(mid)}:", re.MULTILINE)
            m0 = start_pat.search(mohu_txt)
            if not m0:
                continue
            tail = mohu_txt[m0.start() :]
            block = tail.split("\n\n", 1)[0]
            if "Verification:" not in block:
                findings.append(Finding("fail", "mohu_verification", f"{mid} missing `Verification:`"))
    else:
        findings.append(Finding("fail", "missing_file", f"missing `{mohu_path}`"))

    if exp_path.exists():
        exp_txt = read_text(exp_path)
        exp_rows = parse_experiment_table(exp_txt)
        if not exp_rows:
            findings.append(Finding("fail", "experiment_table", f"failed to parse experiment table in `{exp_path}`"))
        else:
            exp_ids, dup_e = _uniq(sorted(exp_rows.keys()))
            if dup_e:
                findings.append(Finding("fail", "experiment_ids_unique", f"duplicate E IDs: {sorted(set(dup_e))}"))
            else:
                findings.append(Finding("ok", "experiment_ids_unique", f"experiments={len(exp_ids)}"))

            missing = [e for e in evidence_eids if e not in exp_rows]
            if missing:
                findings.append(Finding("fail", "evidence_covered", f"E IDs referenced in plan but missing in experiment.md: {missing}"))
            else:
                findings.append(Finding("ok", "evidence_covered", f"plan evidence E IDs covered: {len(evidence_eids)}"))

    out_md = to_md(root, findings)
    out_path = args.out.resolve() if args.out.is_absolute() else (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_md, encoding="utf-8")
    print(f"[OK] wrote: {out_path}")

    return 2 if any(f.level == "fail" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
