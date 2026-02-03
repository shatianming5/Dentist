#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AuditIssue:
    level: str  # "ERROR" | "WARN"
    message: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_inline_code_spans(md: str) -> Iterable[str]:
    # Naive but works for README usage: capture `...` spans.
    for m in re.finditer(r"`([^`\n]+)`", md):
        yield m.group(1).strip()


def _iter_md_links(md: str) -> Iterable[str]:
    # Extract (path) from both links and images: ](path) or ![](path)
    for m in re.finditer(r"\]\(([^)\n]+)\)", md):
        yield m.group(1).strip()


def _clean_path_token(tok: str) -> str:
    s = tok.strip()
    s = s.strip(".,;:!?)\"]'")
    s = s.lstrip("[\"'")
    return s


def _looks_like_repo_path(s: str) -> bool:
    if not s or any(ch.isspace() for ch in s):
        return False
    if s.startswith(("http://", "https://", "mailto:")):
        return False
    if s.startswith("#"):
        return False
    # common repo-relative roots
    return any(
        s.startswith(prefix)
        for prefix in (
            "paper_tables/",
            "scripts/",
            "configs/",
            "runs/",
            "data/",
            "metadata/",
            "archives/",
            "assets/",
        )
    )


def _is_glob_or_placeholder_path(rel: str) -> bool:
    s = str(rel)
    if any(ch in s for ch in ("*", "?", "[", "]")):
        return True
    if "<" in s and ">" in s:
        return True
    if "..." in s or "**" in s:
        return True
    return False


def _is_local_only_path(rel: str) -> bool:
    # Paths that are expected to be missing in a public repo clone.
    s = str(rel)
    if s.startswith("metadata/internal_db/"):
        return True
    if s.startswith(("outputs/", "downloads/")):
        return True
    # We only ship a small guide in runs/; most artifacts are local.
    if s.startswith("runs/") and s != "runs/README.md":
        return True
    return False


def _check_paths_exist(repo_root: Path, readme_md: str) -> list[AuditIssue]:
    issues: list[AuditIssue] = []

    paths: set[str] = set()
    for tok in _iter_inline_code_spans(readme_md):
        p = _clean_path_token(tok)
        if _looks_like_repo_path(p):
            paths.add(p)
    for tok in _iter_md_links(readme_md):
        p = _clean_path_token(tok)
        if _looks_like_repo_path(p):
            paths.add(p)

    for rel in sorted(paths):
        if _is_glob_or_placeholder_path(rel):
            issues.append(AuditIssue("WARN", f"Skip existence check for glob/placeholder: `{rel}`"))
            continue
        p = (repo_root / rel).resolve()
        if not p.exists():
            lvl = "WARN" if _is_local_only_path(rel) else "ERROR"
            issues.append(AuditIssue(lvl, f"Missing referenced path: `{rel}`"))
    return issues


def _check_no_case_uid(readme_md: str) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    # 40-hex SHA / case_uid patterns (we do not want these in public README).
    for m in re.finditer(r"\b[0-9a-f]{40}\b", readme_md):
        issues.append(AuditIssue("ERROR", f"README contains a 40-hex id (possible case_uid/SHA): {m.group(0)}"))
    return issues


def _extract_results_row_metrics(readme_md: str) -> dict[str, float] | None:
    # Expect a snippet like: accuracy=0.6371 / macro_f1=0.6128 / balanced_acc=0.6261 / ece=0.1255 (n=248)
    m = re.search(
        r"accuracy\s*=\s*(?P<acc>[0-9.]+)\s*/\s*macro_f1\s*=\s*(?P<f1>[0-9.]+)\s*/\s*balanced_acc\s*=\s*(?P<bal>[0-9.]+)\s*/\s*ece\s*=\s*(?P<ece>[0-9.]+)\s*\(n\s*=\s*(?P<n>[0-9]+)\)",
        readme_md,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "accuracy": float(m.group("acc")),
        "macro_f1": float(m.group("f1")),
        "balanced_acc": float(m.group("bal")),
        "ece": float(m.group("ece")),
        "n": float(m.group("n")),
    }


def _check_raw_cls_metrics(repo_root: Path, readme_md: str, *, eval_path: str) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    p = (repo_root / eval_path).resolve()
    if not p.exists():
        return [AuditIssue("ERROR", f"Missing raw_cls eval json: `{eval_path}` (needed for README metric check)")]

    got = _extract_results_row_metrics(readme_md)
    if got is None:
        return [AuditIssue("ERROR", "README missing raw_cls metric snippet: accuracy/macro_f1/balanced_acc/ece (n=...)")]

    d = _read_json(p)
    overall = d.get("overall") or {}
    exp = {
        "accuracy": float(overall.get("accuracy", float("nan"))),
        "macro_f1": float(overall.get("macro_f1", float("nan"))),
        "balanced_acc": float(overall.get("balanced_acc", float("nan"))),
        "ece": float(overall.get("ece", float("nan"))),
        "n": float(overall.get("n", float("nan"))),
    }

    def _close(a: float, b: float, tol: float) -> bool:
        return abs(a - b) <= tol

    # We expect README to show rounded values; allow small tolerance.
    tol = 5e-4
    for k in ["accuracy", "macro_f1", "balanced_acc", "ece"]:
        if not _close(float(got[k]), float(exp[k]), tol):
            issues.append(AuditIssue("ERROR", f"raw_cls metric mismatch for {k}: README={got[k]} vs json={exp[k]} ({eval_path})"))
    if int(got["n"]) != int(exp["n"]):
        issues.append(AuditIssue("ERROR", f"raw_cls n mismatch: README={int(got['n'])} vs json={int(exp['n'])} ({eval_path})"))
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit README.md for missing details and broken references.")
    ap.add_argument("--repo-root", type=Path, default=Path("."), help="Repo root.")
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument(
        "--raw-cls-eval",
        type=str,
        default="paper_tables/raw_cls_ensemble_eval_mean_v18_best.json",
        help="Path used to validate raw_cls metrics in README.",
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    readme_path = (root / args.readme).resolve()
    md = _read_text(readme_path)

    issues: list[AuditIssue] = []
    issues.extend(_check_paths_exist(root, md))
    issues.extend(_check_no_case_uid(md))
    issues.extend(_check_raw_cls_metrics(root, md, eval_path=str(args.raw_cls_eval)))

    n_err = sum(1 for i in issues if i.level == "ERROR")
    n_warn = sum(1 for i in issues if i.level == "WARN")
    for i in issues:
        print(f"[{i.level}] {i.message}")

    print(f"[SUMMARY] errors={n_err} warnings={n_warn}")
    raise SystemExit(1 if n_err else 0)


if __name__ == "__main__":
    main()
