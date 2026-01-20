from __future__ import annotations

import hashlib
from pathlib import Path


def sha1_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(int(chunk_bytes))
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_int_seed(text: str, base_seed: int = 0) -> int:
    """Stable 32-bit integer seed derived from text (+ optional base seed)."""

    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ int(base_seed)) & 0xFFFFFFFF
