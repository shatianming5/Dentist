from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    # cuDNN benchmark can pick non-deterministic algorithms; avoid surprising behavior.
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark) if not bool(deterministic) else False

