from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _run_git(root: Path, args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(root), stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="replace").strip()
        return s or None
    except Exception:
        return None


def get_git_info(root: Path) -> dict[str, Any]:
    return {
        "commit": _run_git(root, ["rev-parse", "HEAD"]),
        "dirty": _run_git(root, ["status", "--porcelain"]) not in {None, ""},
    }

