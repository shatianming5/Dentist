from __future__ import annotations

import numpy as np


def rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(a.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty((n,), dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        r = 0.5 * (float(i) + float(j)) + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = r
        i = j + 1
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float | None:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size != a.size:
        return None
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.sqrt(float(np.sum(a * a)) * float(np.sum(b * b))))
    if not np.isfinite(denom) or denom <= 0:
        return None
    r = float(np.sum(a * b) / denom)
    if not np.isfinite(r):
        return None
    return r


def spearmanr(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearsonr(rankdata_avg_ties(x), rankdata_avg_ties(y))

