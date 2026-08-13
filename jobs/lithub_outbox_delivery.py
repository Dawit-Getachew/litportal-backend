"""AWS-neutral entry point for ``litportal.lithub_outbox_delivery`` (A1).

The audit called this "the cleanest fit in the entire codebase": the app
already hand-rolled SQS-in-Mongo (`services/lithub_sync.py` — durable rows,
backoff, dead state) because no real queue existed. This worker replaces the
60s polling sweeper's delivery step with an SQS-triggered one; the enqueue
points and the outbox row stay exactly as they are.

The message carries the row id, not the item payload — the worker re-reads the
row, so the message stays small and the payload has one source of truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from jobs.contracts import LitHubOutboxDeliveryMessage, PermanentInputError

logger = logging.getLogger("litportal.jobs.lithub_outbox_delivery")


@dataclass
class RunResult:
    status: str  # succeeded | already_done | superseded
    outbox_id: str
    op: str = ""
    detail: str = ""


async def run(
    body: dict[str, Any],
    *,
    db: Any = None,
    deliver: Any = None,
) -> RunResult:
    """Deliver one outbox row to LitHub. Raises on failure so SQS retries.

    Delivery is at-least-once and safe: LitHub's save is an upsert and its
    delete is idempotent. A row already ``done`` (or superseded by a newer
    opposite op) is ACKed without a second delivery.
    """
    envelope, message = LitHubOutboxDeliveryMessage.parse(body)
    log_ctx = {
        "correlation_id": envelope.correlation_id,
        "outbox_id": message.outbox_id,
        "op": message.op,
    }
    logger.info("lithub_outbox_delivery_started", extra=log_ctx)

    if db is None:
        db = _default_db()

    row = await db["lithub_sync_outbox"].find_one({"id": message.outbox_id})
    if row is None:
        # The row is the source of truth; without it the message can never
        # succeed — straight to the DLQ, tagged.
        raise PermanentInputError(
            f"outbox row {message.outbox_id} does not exist"
        )
    if row.get("status") == "done":
        logger.info("lithub_outbox_delivery_already_done", extra=log_ctx)
        return RunResult(
            status="already_done", outbox_id=message.outbox_id, op=message.op
        )
    if row.get("status") == "superseded":
        logger.info("lithub_outbox_delivery_superseded", extra=log_ctx)
        return RunResult(
            status="superseded", outbox_id=message.outbox_id, op=message.op,
            detail="a newer opposite op replaced this row",
        )

    if deliver is None:
        deliver = _default_deliver
    async with httpx.AsyncClient(timeout=8.0) as client:
        await deliver(row["op"], row["payload"], client)

    await db["lithub_sync_outbox"].update_one(
        {"id": message.outbox_id},
        {"$set": {"status": "done", "last_error": None}},
    )
    logger.info("lithub_outbox_delivery_succeeded", extra=log_ctx)
    return RunResult(status="succeeded", outbox_id=message.outbox_id, op=row["op"])


def _default_db() -> Any:
    from core.db import get_db

    return get_db()


async def _default_deliver(op: str, payload: dict[str, Any], client: Any) -> None:
    from services.lithub_sync import _attempt

    await _attempt(op, payload, client)
