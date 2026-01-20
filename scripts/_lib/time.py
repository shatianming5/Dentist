from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """UTC timestamp in ISO-8601, e.g. 2026-01-20T05:36:00Z."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_compact_ts() -> str:
    """Compact UTC timestamp, e.g. 20260120_053600."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
