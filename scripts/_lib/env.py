from __future__ import annotations

import sys
from typing import Any

from _lib.time import utc_now_iso


def get_env_info() -> dict[str, Any]:
    # Lazy imports: avoid importing numpy/torch for scripts that only need env metadata.
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        torch = None  # type: ignore

    out: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "python": sys.version.replace("\n", " "),
        "numpy": getattr(np, "__version__", "unknown") if np is not None else "missing",
        "torch": getattr(torch, "__version__", "unknown") if torch is not None else "missing",
    }
    if torch is not None:
        out.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda": getattr(torch.version, "cuda", None),
                "cudnn": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            }
        )
    return out
