"""Audit logging service. Writes append-only audit_events documents."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from core.db import Collections, get_db
from models.audit import AuditEventDoc


class AuditService:
    """Thin wrapper for inserting audit_events docs.

    Designed to be safe to call from request handlers; never raises on failure
    (best-effort). Logs are stored with timezone-aware UTC datetimes.
    """

    async def log(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        search_session_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> str:
        doc = AuditEventDoc(
            event_type=event_type,
            user_id=user_id,
            search_session_id=search_session_id,
            event_payload_json=payload or {},
        )
        doc_dict = doc.model_dump()
        doc_dict["created_at"] = doc_dict["created_at"].astimezone(timezone.utc)
        try:
            await get_db()[Collections.AUDIT_EVENTS].insert_one(doc_dict)
        except Exception:  # noqa: BLE001 - audit must never break the caller
            # In production we'd surface to monitoring; for now swallow.
            pass
        return doc.id


audit_service = AuditService()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
