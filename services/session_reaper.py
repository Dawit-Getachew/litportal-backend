"""Stale search-session reaper — the approved interim safeguard for job 2
until the ECS worker path lands (register §1, R1 in DATA_RISK_REGISTER.md).

``run_pipeline`` swallows every exception and a hard process death skips even
that, so a session can sit in a transient status forever (observed in prod:
two sessions stuck in ``ranking`` since July). The frontend then shows an
eternal spinner and the question can never be retried. This sweeper flips
sessions that have been in a transient status far longer than the worst-case
pipeline runtime (~33 min) to ``error`` so the UI unblocks and a retry is
possible. The full status-claim + heartbeat design arrives with the ECS
worker, which re-reads final status and lets SQS redelivery retry.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger("clinevidence.session_reaper")

#: Statuses the pipeline/summary paths move through while actively running.
TRANSIENT_STATUSES = (
    "searching",
    "ranking",
    "brokering_access",
    "fetching_oa_full_text",
    "summarizing",
)

REAPED_ERROR_MESSAGE = (
    "This search stalled and was cleaned up automatically. Please retry."
)


def _stale_after_seconds() -> int:
    return int(os.environ.get("SESSION_REAPER_STALE_SECONDS", "5400"))


def _sweep_interval_seconds() -> int:
    return int(os.environ.get("SESSION_REAPER_INTERVAL_SECONDS", "600"))


def _enabled() -> bool:
    return os.environ.get("SESSION_REAPER_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


async def reap_stale_sessions_once(
    db: Any = None, *, now: Optional[datetime] = None
) -> int:
    """Flip stale in-flight sessions to ``error``. Returns sessions reaped."""
    if db is None:
        from core.db import get_db

        db = get_db()
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=_stale_after_seconds())
    result = await db["search_sessions"].update_many(
        {
            "status": {"$in": list(TRANSIENT_STATUSES)},
            "$or": [
                {"updated_at": {"$lt": cutoff}},
                {"updated_at": {"$exists": False}, "created_at": {"$lt": cutoff}},
            ],
        },
        {
            "$set": {
                "status": "error",
                "error_message": REAPED_ERROR_MESSAGE,
                "reaped_at": now,
                "updated_at": now,
            }
        },
    )
    count = int(getattr(result, "modified_count", 0) or 0)
    if count:
        logger.warning(
            "session_reaper reaped %s stale session(s) (stale_after=%ss)",
            count,
            _stale_after_seconds(),
        )
    return count


async def session_reaper_loop(stop_event: asyncio.Event) -> None:
    """Background loop reaping stale sessions until ``stop_event`` is set."""
    if not _enabled():
        logger.info("session_reaper disabled (SESSION_REAPER_ENABLED)")
        return
    logger.info(
        "session_reaper started (interval=%ss, stale_after=%ss)",
        _sweep_interval_seconds(),
        _stale_after_seconds(),
    )
    while not stop_event.is_set():
        try:
            await reap_stale_sessions_once()
        except Exception:  # noqa: BLE001
            logger.exception("session_reaper_iteration_failed")
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=_sweep_interval_seconds()
            )
        except asyncio.TimeoutError:
            pass
    logger.info("session_reaper stopped")
