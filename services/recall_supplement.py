"""Thin-pool supplemental-retrieval decision helpers.

When the deduped candidate pool is small even after the boolean compiler +
legacy-planner fallback (an over-specific PICOTT/anchor set starved retrieval),
the pipeline runs one broadened retrieval round and merges it in BEFORE ranking,
so the new candidates inherit every downstream guardrail (safety filter, PICOTT
anchor caps, default-selection, rescue re-add). These are the pure, side-effect-
free decision functions the pipeline uses; the I/O lives in services/pipeline.py.

Enabled by default (no deployment env var required); set
THIN_POOL_SUPPLEMENT_ENABLED=false to disable.
"""
from __future__ import annotations

import os

DEFAULT_MIN_POOL = 8


def thin_pool_settings() -> tuple[bool, int]:
    """(enabled, min_pool) read from env with safe, default-ON values."""
    enabled = os.environ.get("THIN_POOL_SUPPLEMENT_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    try:
        min_pool = int(os.environ.get("THIN_POOL_SUPPLEMENT_MIN_POOL", str(DEFAULT_MIN_POOL)))
    except (TypeError, ValueError):
        min_pool = DEFAULT_MIN_POOL
    return enabled, min_pool


def should_supplement(pool_size: int, *, enabled: bool, min_pool: int) -> bool:
    """True when the deduped pool is below the broaden threshold."""
    try:
        return bool(enabled) and int(pool_size) < int(min_pool)
    except (TypeError, ValueError):
        return False


def sources_to_broaden(sources, fallback_sources) -> list:
    """Sources to run the broadened (legacy-planner) round for: every enabled
    source that was NOT already broadened by the per-source legacy-planner
    fallback (re-running its identical queries would add nothing). Order
    preserved."""
    fb = set(fallback_sources or [])
    return [s for s in (sources or []) if s not in fb]


__all__ = [
    "DEFAULT_MIN_POOL",
    "thin_pool_settings",
    "should_supplement",
    "sources_to_broaden",
]
