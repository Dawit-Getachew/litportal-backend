"""LitHub dual-write mirror with a Mongo-backed outbox + background sweeper.

The Mongo analogue of litbridge-backend's ``lithub_sync_outbox`` table. Library
saves/deletes are mirrored into the central LitHub service keyed by the user's
Identity ``sub``. Writes are enqueued to ``lithub_sync_outbox`` first (a fast,
durable local insert) and an immediate best-effort delivery is kicked off; a
60s background sweeper retries with capped exponential backoff and a terminal
``dead`` state. Everything here is gated by ``LITPORTAL_DUAL_WRITE_LITHUB`` and
``LITHUB_BASE_URL`` and is fail-safe — a LitHub outage never breaks a user save.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any, Optional

import httpx
from pymongo import ReturnDocument

from core.config import get_settings
from core.db import Collections, get_db
from models.common import new_id, utcnow

logger = logging.getLogger("litportal.lithub_sync")

_HTTP_TIMEOUT = 8.0
_MAX_ATTEMPTS = 5
# Capped exponential backoff between retries (seconds): 1m, 2m, 4m, 8m, 16m.
_BACKOFF_SECONDS = (60, 120, 240, 480, 960)
_SWEEP_INTERVAL_SECONDS = 60
_SWEEP_BATCH = 50
# A row claimed (status="processing") longer than this is considered stale
# (the worker died mid-flight) and is reclaimed by the next sweep.
_STALE_PROCESSING_SECONDS = 300

# Strong references to fire-and-forget delivery tasks. asyncio only keeps a
# WEAK reference to a running task, so without this set the GC can collect an
# in-flight immediate delivery before it finishes. Tasks self-remove on done.
_BACKGROUND_TASKS: set = set()


def _enabled() -> bool:
    s = get_settings()
    return bool(s.LITPORTAL_DUAL_WRITE_LITHUB and s.LITHUB_BASE_URL)


def _backoff(attempts: int) -> int:
    return _BACKOFF_SECONDS[min(max(attempts - 1, 0), len(_BACKOFF_SECONDS) - 1)]


async def _attempt(op: str, payload: dict[str, Any], client: httpx.AsyncClient) -> None:
    """Perform a single mirror operation against LitHub. Raises on failure."""
    from clients.lithub_client import LitHubClient

    lithub = LitHubClient(http_client=client, settings=get_settings())
    sub = payload["user_id"]
    if op == "save":
        await lithub.internal_save_paper(sub, payload["item"])
    elif op == "delete":
        await lithub.internal_delete_library(
            sub, pmid=payload.get("pmid"), doi=payload.get("doi"),
        )
    else:  # pragma: no cover — defensive
        raise ValueError(f"unknown outbox op: {op}")


async def _enqueue(op: str, payload: dict[str, Any]) -> None:
    """Durably record a mirror op, then kick an immediate best-effort delivery."""
    db = get_db()
    now = utcnow()
    try:
        await db[Collections.LITHUB_SYNC_OUTBOX].insert_one({
            "id": new_id(),
            "op": op,
            "payload": payload,
            "status": "pending",
            "attempts": 0,
            "next_attempt_at": now,
            "last_error": None,
            "created_at": now,
            "updated_at": now,
        })
    except Exception:  # noqa: BLE001
        logger.exception("lithub_outbox_enqueue_failed op=%s", op)
        return
    # Immediate, non-blocking delivery attempt (the 60s sweeper is the backstop).
    try:
        task = asyncio.create_task(run_outbox_sweep_once())
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
    except RuntimeError:
        pass  # no running loop (e.g. called outside an event loop in a test)


async def mirror_save(sub: str, item: dict[str, Any]) -> None:
    """Mirror a library save to LitHub (gated; never raises)."""
    if not _enabled():
        return
    await _enqueue("save", {"user_id": str(sub), "item": item})


async def mirror_delete(sub: str, *, pmid: Optional[str] = None, doi: Optional[str] = None) -> None:
    """Mirror a library delete to LitHub (gated; never raises)."""
    if not _enabled() or not (pmid or doi):
        return
    await _enqueue("delete", {"user_id": str(sub), "pmid": pmid, "doi": doi})


async def list_central(sub: str, *, params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """Best-effort read of the user's central LitHub library. None on any error."""
    if not _enabled():
        return None
    try:
        from clients.lithub_client import LitHubClient

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            lithub = LitHubClient(http_client=client, settings=get_settings())
            return await lithub.internal_list_library(sub, params=params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("lithub_list_central_failed sub=%s error=%s", sub, exc)
        return None


async def _claim_due_row(coll: Any) -> Optional[dict[str, Any]]:
    now = utcnow()
    stale_before = now - timedelta(seconds=_STALE_PROCESSING_SECONDS)
    return await coll.find_one_and_update(
        {
            "$or": [
                {"status": "pending", "next_attempt_at": {"$lte": now}},
                {"status": "processing", "updated_at": {"$lte": stale_before}},
            ],
        },
        {"$set": {"status": "processing", "updated_at": now}},
        sort=[("next_attempt_at", 1)],
        return_document=ReturnDocument.AFTER,
    )


async def run_outbox_sweep_once(limit: int = _SWEEP_BATCH) -> int:
    """Drain due outbox rows once. Returns the count delivered. Never raises."""
    if not _enabled():
        return 0
    coll = get_db()[Collections.LITHUB_SYNC_OUTBOX]
    delivered = 0
    for _ in range(limit):
        try:
            row = await _claim_due_row(coll)
        except Exception:  # noqa: BLE001
            logger.exception("lithub_outbox_claim_failed")
            break
        if row is None:
            break
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                await _attempt(row["op"], row["payload"], client)
            await coll.update_one(
                {"id": row["id"]},
                {"$set": {"status": "done", "last_error": None, "updated_at": utcnow()}},
            )
            delivered += 1
        except Exception as exc:  # noqa: BLE001
            attempts = int(row.get("attempts", 0)) + 1
            now = utcnow()
            if attempts >= _MAX_ATTEMPTS:
                await coll.update_one(
                    {"id": row["id"]},
                    {"$set": {
                        "status": "dead", "attempts": attempts,
                        "last_error": str(exc)[:500], "updated_at": now,
                    }},
                )
                logger.error("lithub_outbox_dead op=%s id=%s error=%s", row["op"], row["id"], exc)
            else:
                await coll.update_one(
                    {"id": row["id"]},
                    {"$set": {
                        "status": "pending", "attempts": attempts,
                        "last_error": str(exc)[:500],
                        "next_attempt_at": now + timedelta(seconds=_backoff(attempts)),
                        "updated_at": now,
                    }},
                )
    return delivered


async def outbox_sweeper_loop(stop_event: asyncio.Event) -> None:
    """Background loop draining the outbox until ``stop_event`` is set."""
    logger.info("lithub_outbox_sweeper started (interval=%ss)", _SWEEP_INTERVAL_SECONDS)
    while not stop_event.is_set():
        try:
            await run_outbox_sweep_once()
        except Exception:  # noqa: BLE001
            logger.exception("lithub_outbox_sweep_iteration_failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_SWEEP_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass
    logger.info("lithub_outbox_sweeper stopped")
