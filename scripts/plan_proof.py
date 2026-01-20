#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Claim:
    cid: str
    text: str
    evidence: list[str]
    proof_rule: str


@dataclass(frozen=True)
class ExpStatus:
    eid: str
    smoke: bool
    full: bool


def parse_experiment_status(exp_md: str) -> dict[str, ExpStatus]:
    lines = exp_md.splitlines()
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

    out: dict[str, ExpStatus] = {}
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
        out[eid] = ExpStatus(eid=eid, smoke=smoke, full=full)
    return out


def parse_claims(plan_md: str) -> list[Claim]:
    lines = plan_md.splitlines()
    claims: list[Claim] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^- \[(?: |x|X)\] (C\d{4}):\s*(.*)$", line.strip())
        if not m:
            i += 1
            continue
        cid = m.group(1)
        text = m.group(2).strip()
        evidence: list[str] = []
        proof_rule = ""
        i += 1
        while i < len(lines) and lines[i].startswith("  - "):
            s = lines[i].strip()
            if s.lower().startswith("- evidence:"):
                evidence = re.findall(r"E\d{4}", s)
            if s.lower().startswith("- proof rule:"):
                proof_rule = s.split(":", 1)[1].strip()
            i += 1
        claims.append(Claim(cid=cid, text=text, evidence=evidence, proof_rule=proof_rule))
    return claims


def expand_token(token: str) -> list[str]:
    t = token.strip()
    if "{val,test}" in t:
        return [t.replace("{val,test}", "val"), t.replace("{val,test}", "test")]
    return [t]


def extract_backtick_tokens(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def glob_any(root: Path, pattern: str) -> list[Path]:
    pat = pattern
    if pat.startswith("./"):
        pat = pat[2:]
    return sorted(root.glob(pat))


def check_raw_cls_metrics_keys(root: Path) -> tuple[bool, str]:
    candidates: list[Path] = []
    for p in (root / "runs/raw_cls_baseline").iterdir():
        if not p.is_dir():
            continue
        mp = p / "metrics.json"
        if mp.exists():
            candidates.append(mp)
    if not candidates:
        return False, "no metrics.json under runs/raw_cls_baseline/"

    need = {"test_by_source_calibration", "test_calibration", "test_by_tooth_position"}
    for mp in candidates:
        try:
            obj = read_json(mp)
        except Exception:
            continue
        missing = [k for k in sorted(need) if k not in obj]
        if not missing:
            return True, f"ok: {mp.relative_to(root)}"
    return False, f"all candidate metrics.json missing keys: {sorted(need)}"


def check_raw_cls_meta_feature_table(root: Path) -> tuple[bool, str]:
    table = root / "paper_tables/raw_cls_table_v13.md"
    if not table.exists():
        return False, "missing paper_tables/raw_cls_table_v13.md"

    want_extra = "scale,log_points,objects_used"
    txt = read_text(table)
    rows = [ln for ln in txt.splitlines() if ln.strip().startswith("|") and "v13_main4" in ln]
    found: dict[str, int] = {}
    for ln in rows:
        cells = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(cells) < 11:
            continue
        # columns: test_macro_f1 | test_acc | n | model | n_points | kfold_k | balanced | label_smoothing | extra_features | tta | dataset
        try:
            n = int(cells[2])
            model = cells[3].lower()
            kfold_k = int(cells[5])
            extra = cells[8]
            dataset = cells[10]
        except Exception:
            continue
        if dataset != "v13_main4" or kfold_k != 5:
            continue
        if model not in {"pointnet", "dgcnn"}:
            continue
        if extra != want_extra:
            continue
        found[model] = n

    missing = [m for m in ["pointnet", "dgcnn"] if m not in found]
    if missing:
        return False, f"missing table rows for extra_features={want_extra}: {missing}"
    bad_n = {m: n for m, n in found.items() if int(n) != 15}
    if bad_n:
        return False, f"table rows must have n=15 (k=5×seeds=3) for extra_features={want_extra}: {bad_n}"
    return True, f"ok: found pointnet/dgcnn rows (n=15) for extra_features={want_extra}"


def check_raw_cls_meta_only_table(root: Path) -> tuple[bool, str]:
    table = root / "paper_tables/raw_cls_table_v13.md"
    if not table.exists():
        return False, "missing paper_tables/raw_cls_table_v13.md"
    want_extra = "scale,log_points,objects_used"
    txt = read_text(table)
    rows = [ln for ln in txt.splitlines() if ln.strip().startswith("|") and "v13_main4" in ln]
    found_n: int | None = None
    for ln in rows:
        cells = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(cells) < 11:
            continue
        try:
            n = int(cells[2])
            model = cells[3].lower()
            kfold_k = int(cells[5])
            extra = cells[8]
            dataset = cells[10]
        except Exception:
            continue
        if dataset != "v13_main4" or kfold_k != 5:
            continue
        if model != "meta_mlp":
            continue
        if extra != want_extra:
            continue
        found_n = int(n)
        break
    if found_n is None:
        return False, f"missing table row for model=meta_mlp extra_features={want_extra}"
    if int(found_n) != 15:
        return False, f"table row must have n=15 (k=5×seeds=3) for meta_mlp extra_features={want_extra}: n={found_n}"
    return True, "ok: found meta_mlp row (n=15)"


def check_raw_cls_geom_mlp_table(root: Path) -> tuple[bool, str]:
    table = root / "paper_tables/raw_cls_table_v13.md"
    if not table.exists():
        return False, "missing paper_tables/raw_cls_table_v13.md"
    txt = read_text(table)
    rows = [ln for ln in txt.splitlines() if ln.strip().startswith("|") and "v13_main4" in ln]
    found_n: int | None = None
    for ln in rows:
        cells = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(cells) < 11:
            continue
        try:
            n = int(cells[2])
            model = cells[3].lower()
            kfold_k = int(cells[5])
            extra = cells[8]
            dataset = cells[10]
        except Exception:
            continue
        if dataset != "v13_main4" or kfold_k != 5:
            continue
        if model != "geom_mlp":
            continue
        if extra not in {"(none)", ""}:
            continue
        found_n = int(n)
        break
    if found_n is None:
        return False, "missing table row for model=geom_mlp extra_features=(none)"
    if int(found_n) != 15:
        return False, f"table row must have n=15 (k=5×seeds=3) for geom_mlp extra_features=(none): n={found_n}"
    return True, "ok: found geom_mlp row (n=15)"


def check_raw_cls_domain_shift_table(root: Path) -> tuple[bool, str]:
    table = root / "paper_tables/raw_cls_domain_shift_table_v13.md"
    if not table.exists():
        return False, "missing paper_tables/raw_cls_domain_shift_table_v13.md"
    lines = read_text(table).splitlines()
    header_idx = None
    header: list[str] = []
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "train_source" in line and "test_source" in line and "model" in line:
            header_idx = i
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            break
    if header_idx is None:
        return False, "failed to locate domain-shift markdown table header"

    def idx(name: str) -> int:
        try:
            return header.index(name)
        except ValueError:
            return -1

    n_idx = idx("n")
    model_idx = idx("model")
    tr_idx = idx("train_source")
    te_idx = idx("test_source")
    ds_idx = idx("dataset")
    if min(n_idx, model_idx, tr_idx, te_idx, ds_idx) < 0:
        return False, f"domain-shift table missing required columns: header={header}"

    # Data rows start after the alignment row.
    got: dict[tuple[str, str, str], int] = {}
    for line in lines[header_idx + 2 :]:
        s = line.strip()
        if not s.startswith("|"):
            break
        cells = [c.strip() for c in s.strip().strip("|").split("|")]
        if len(cells) < len(header):
            continue
        if cells[ds_idx] != "v13_main4":
            continue
        try:
            n = int(cells[n_idx])
        except Exception:
            continue
        model = cells[model_idx].lower()
        tr = cells[tr_idx]
        te = cells[te_idx]
        got[(model, tr, te)] = n

    need = [
        ("pointnet", "普通标注", "专家标注"),
        ("dgcnn", "普通标注", "专家标注"),
        ("pointnet", "专家标注", "普通标注"),
        ("dgcnn", "专家标注", "普通标注"),
    ]
    missing = [k for k in need if k not in got]
    if missing:
        return False, f"domain-shift table missing groups: {missing}"
    bad = {k: got[k] for k in need if int(got[k]) < 3}
    if bad:
        return False, f"domain-shift table requires n>=3 seeds per group: {bad}"
    return True, "ok: found both directions (A↔B) for pointnet/dgcnn with n>=3"


def check_raw_cls_kfold_merged_report(root: Path) -> tuple[bool, str]:
    md = root / "paper_tables/raw_cls_kfold_merged_report_v13_main4.md"
    js = root / "paper_tables/raw_cls_kfold_merged_report_v13_main4.json"
    missing = [p.name for p in [md, js] if not p.exists()]
    if missing:
        return False, f"missing: {missing}"
    try:
        obj = read_json(js)
    except Exception:
        return False, f"failed to parse: {js.relative_to(root)}"
    models: set[str] = set()
    for g in (obj.get("groups") or []):
        if not isinstance(g, dict):
            continue
        key = g.get("key") or {}
        if isinstance(key, dict):
            m = str(key.get("model") or "").lower()
            if m:
                models.add(m)
    need = {"pointnet", "dgcnn"}
    if not need.issubset(models):
        return False, f"report missing models: need={sorted(need)} have={sorted(models)}"
    comps = obj.get("comparisons") or []
    if not isinstance(comps, list) or not comps:
        return False, "report missing comparisons"
    has_f1 = False
    for c in comps:
        if not isinstance(c, dict):
            continue
        r = c.get("result") or {}
        if not isinstance(r, dict):
            continue
        if str(r.get("metric") or "") != "macro_f1_present":
            continue
        ci = r.get("delta_ci95") or {}
        if not (isinstance(ci, dict) and "lo" in ci and "hi" in ci):
            continue
        if "p_two_sided" not in r:
            continue
        has_f1 = True
        break
    if not has_f1:
        return False, "report comparisons missing macro_f1_present paired diff entries"
    return True, "ok"


def check_constraints_summary(root: Path) -> tuple[bool, str]:
    p = root / "paper_tables/constraints_summary.md"
    if not p.exists():
        return False, "missing paper_tables/constraints_summary.md"
    txt = read_text(p)
    if "min_d_p05" not in txt:
        return False, "constraints_summary.md missing min_d_p05 column"
    # Require at least one eval_val.json + eval_test.json in runs dir.
    runs_dir = root / "runs/teeth3ds_prep2target_constraints"
    ev = list(runs_dir.glob("*/eval_val.json"))
    et = list(runs_dir.glob("*/eval_test.json"))
    if not ev or not et:
        return False, "missing eval_val.json/eval_test.json under runs/teeth3ds_prep2target_constraints/"
    return True, "ok"


def check_plan_report_scope(root: Path) -> tuple[bool, str]:
    scope = root / "PAPER_SCOPE.md"
    rep = root / "plan_report.md"
    if not scope.exists():
        return False, "missing PAPER_SCOPE.md"
    if not rep.exists():
        return False, "missing plan_report.md"
    head = "\\n".join(read_text(rep).splitlines()[:5])
    if "PAPER_SCOPE.md" not in head:
        return False, "plan_report.md top note does not reference PAPER_SCOPE.md"
    return True, "ok"


def check_repro_artifacts(root: Path) -> tuple[bool, str]:
    need = [
        root / "docs/experiment.md",
        root / "PAPER_PROTOCOL.md",
        root / "JOURNAL_AUDIT.md",
        root / "environment.yml",
    ]
    missing = [str(p.relative_to(root)) for p in need if not p.exists()]
    if missing:
        return False, f"missing: {missing}"
    # .rd_queue is optional until you start tmux queue; treat as warn in report.
    return True, "ok (note: .rd_queue is created by tmux runner)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Proof-audit docs/plan.md claims against docs/experiment.md and artifacts.")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--plan", type=Path, default=Path("docs/plan.md"))
    ap.add_argument("--experiment", type=Path, default=Path("docs/experiment.md"))
    ap.add_argument("--out", type=Path, default=Path("docs/proof.md"))
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any claim is NOT_PROVED.")
    args = ap.parse_args()

    root = args.root.resolve()
    plan_path = (root / args.plan).resolve() if not args.plan.is_absolute() else args.plan.resolve()
    exp_path = (root / args.experiment).resolve() if not args.experiment.is_absolute() else args.experiment.resolve()

    if not plan_path.exists():
        raise SystemExit(f"Missing plan: {plan_path}")
    if not exp_path.exists():
        raise SystemExit(f"Missing experiment ledger: {exp_path}")

    claims = parse_claims(read_text(plan_path))
    exp_status = parse_experiment_status(read_text(exp_path))

    lines: list[str] = []
    lines.append("# Proof audit")
    lines.append("")
    lines.append(f"- generated_at: {utc_now_iso()}")
    lines.append(f"- root: `{root}`")
    lines.append(f"- plan: `{plan_path.relative_to(root)}`")
    lines.append(f"- experiment: `{exp_path.relative_to(root)}`")

    failed = 0
    proved = 0

    for c in claims:
        lines.append(f"## {c.cid}")
        lines.append(f"- claim: {c.text}")
        lines.append(f"- evidence: {', '.join(c.evidence) if c.evidence else '(missing)'}")

        missing_e = [e for e in c.evidence if e not in exp_status]
        if missing_e:
            failed += 1
            lines.append(f"- status: FAIL (missing experiments in docs/experiment.md: {missing_e})")
            lines.append("")
            continue

        # Default: require all evidence experiments have Full checked.
        not_full = [e for e in c.evidence if not exp_status[e].full]
        status = "NOT_PROVED"
        notes: list[str] = []

        if not_full:
            notes.append(f"evidence not marked Full: {not_full}")

        # Claim-specific artifact checks (based on current docs/plan.md content).
        ok = True
        if c.cid == "C0001":
            table = root / "paper_tables/raw_cls_table_v13.md"
            if not table.exists():
                ok = False
                notes.append("missing paper_tables/raw_cls_table_v13.md")
            ci = sorted((root / "paper_tables").glob("raw_cls_ci_*.json"))
            if not ci:
                ok = False
                notes.append("missing paper_tables/raw_cls_ci_*.json")
            # Optional: validate n=15 for the paper-intended setting.
            if table.exists():
                txt = read_text(table)
                # Find rows where kfold_k=5 and dataset=v13_main4 and model in {pointnet,dgcnn} and tta=8 and label_smoothing=0.100.
                # Only validate the main paper-intended setting (extra_features=(none)) to avoid unrelated ablation rows
                # breaking the claim.
                rows = [ln for ln in txt.splitlines() if ln.strip().startswith("|") and "v13_main4" in ln]
                ok_rows = 0
                for ln in rows:
                    cells = [x.strip() for x in ln.strip().strip("|").split("|")]
                    if len(cells) < 11:
                        continue
                    # columns: test_macro_f1 | test_acc | n | model | n_points | kfold_k | balanced | label_smoothing | extra_features | tta | dataset
                    try:
                        n = int(cells[2])
                        model = cells[3].lower()
                        kfold_k = int(cells[5])
                        balanced = cells[6].lower()
                        ls = cells[7]
                        extra = cells[8]
                        tta = int(cells[9])
                        dataset = cells[10]
                    except Exception:
                        continue
                    if dataset != "v13_main4" or kfold_k != 5:
                        continue
                    if model not in {"pointnet", "dgcnn"}:
                        continue
                    if balanced not in {"yes", "y", "true"}:
                        continue
                    if tta != 8:
                        continue
                    if not ls.startswith("0.100"):
                        continue
                    if extra not in {"(none)", ""}:
                        continue
                    if n < 15:
                        ok = False
                        notes.append(f"table row has n<{15} for {model}: n={n}")
                    ok_rows += 1
                if ok_rows == 0:
                    notes.append("warn: no matching rows found in raw_cls_table_v13.md for the intended setting")

        elif c.cid == "C0002":
            ok2, note2 = check_raw_cls_metrics_keys(root)
            if not ok2:
                ok = False
            notes.append(note2)
        elif c.cid == "C0003":
            ok3, note3 = check_plan_report_scope(root)
            if not ok3:
                ok = False
            notes.append(note3)
        elif c.cid == "C0004":
            ok4, note4 = check_constraints_summary(root)
            if not ok4:
                ok = False
            notes.append(note4)
        elif c.cid == "C0005":
            ok5, note5 = check_repro_artifacts(root)
            if not ok5:
                ok = False
            notes.append(note5)
        elif c.cid == "C0006":
            ok6, note6 = check_raw_cls_meta_feature_table(root)
            if not ok6:
                ok = False
            notes.append(note6)
        elif c.cid == "C0007":
            ok7, note7 = check_raw_cls_kfold_merged_report(root)
            if not ok7:
                ok = False
            notes.append(note7)
        elif c.cid == "C0008":
            ok8, note8 = check_raw_cls_meta_only_table(root)
            if not ok8:
                ok = False
            notes.append(note8)
        elif c.cid == "C0009":
            ok9, note9 = check_raw_cls_geom_mlp_table(root)
            if not ok9:
                ok = False
            notes.append(note9)
        elif c.cid == "C0010":
            ok10, note10 = check_raw_cls_domain_shift_table(root)
            if not ok10:
                ok = False
            notes.append(note10)

        if ok and not not_full:
            status = "PROVED"
            proved += 1
        else:
            failed += 1

        lines.append(f"- status: {status}")
        if c.proof_rule:
            lines.append(f"- proof_rule: {c.proof_rule}")
        if notes:
            lines.append("- notes:")
            for n0 in notes:
                lines.append(f"  - {n0}")
        lines.append("")

    lines.insert(
        6,
        f"- summary: proved={proved} not_proved={failed} (rule: all evidence experiments must be marked Full + artifact checks pass)",
    )
    lines.insert(7, "")

    out_path = args.out.resolve() if args.out.is_absolute() else (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_path}")

    return 2 if (failed and bool(args.strict)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
