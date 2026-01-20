from __future__ import annotations

import torch


def normalize_device(device: str) -> str:
    d = device.strip().lower()
    if d in {"auto", ""}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d in {"cuda", "cpu"}:
        if d == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return d
    raise ValueError(f"Unsupported device: {device}")

