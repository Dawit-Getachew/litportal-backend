"""Phase M4-shadow — admin-only sidecar persistence for the
Full-Question ClinicalQuestionMap Interpreter.

Append-only. TTL 30d (``created_at`` index). One row per pipeline run
that TRIGGERED the M4 interpreter. NEVER clinician-visible. NO raw
prompt or raw completion is ever stored — only the validated candidate,
closed-set trigger/rejection reasons, the deterministic baseline
snapshot, and the would-be compiler projection.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from core.db import Collections, get_db

logger = logging.getLogger("clinevidence.m4_shadow_sidecar")


def _isofmt(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v


async def write_m4_sidecar(
    *,
    session_id: str,
    user_id: Optional[str],
    m4_block: dict[str, Any],
) -> Optional[str]:
    """Persist one M4 shadow-decision sidecar row. Returns the row id,
    or None on failure (non-fatal). Caller guarantees ``m4_block`` is
    already leak-scrubbed (no prompt/completion, closed-set only).
    """
    doc_id = uuid.uuid4().hex
    now_dt = datetime.now(timezone.utc)
    doc = {
        "id": doc_id,
        "search_session_id": session_id,
        "user_id": user_id,
        "mode": m4_block.get("mode"),
        "triggered": bool(m4_block.get("triggered")),
        "trigger_reasons": list(m4_block.get("trigger_reasons") or []),
        "validation_status": m4_block.get("validation_status"),
        "rejection_reasons": list(m4_block.get("rejection_reasons") or []),
        "baseline": m4_block.get("baseline"),
        "candidate": m4_block.get("candidate"),
        "would_be_compiler_input": m4_block.get("would_be_compiler_input"),
        "m4_0_1_projection": m4_block.get("m4_0_1_projection"),
        "accepted": bool(m4_block.get("accepted")),
        "active_retrieval_changed": False,
        "latency_ms": int(m4_block.get("latency_ms") or 0),
        "phase": "M4",
        "created_at": now_dt,
    }
    try:
        await get_db()[Collections.M4_SHADOW_DECISIONS].insert_one(dict(doc))
        return doc_id
    except Exception:  # noqa: BLE001
        logger.exception(
            "m4_shadow_decisions insert failed (non-fatal) session_id=%s",
            session_id,
        )
        return None


async def get_m4_decision_latest(session_id: str) -> Optional[dict]:
    """Return the most-recent M4 sidecar row for ``session_id`` (``_id``
    stripped, ``created_at`` ISO-rendered), or None."""
    db = get_db()
    doc = await db[Collections.M4_SHADOW_DECISIONS].find_one(
        {"search_session_id": session_id},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    if not doc:
        return None
    doc["created_at"] = _isofmt(doc.get("created_at"))
    return doc


async def list_recent_m4_decisions(*, limit: int = 50) -> list[dict]:
    """Return the most-recent N M4 sidecar rows across all sessions."""
    db = get_db()
    cur = (
        db[Collections.M4_SHADOW_DECISIONS]
        .find({}, {"_id": 0})
        .sort([("created_at", -1)])
        .limit(int(limit))
    )
    rows = await cur.to_list(length=int(limit))
    for d in rows:
        d["created_at"] = _isofmt(d.get("created_at"))
    return rows


__all__ = [
    "write_m4_sidecar",
    "get_m4_decision_latest",
    "list_recent_m4_decisions",
]
