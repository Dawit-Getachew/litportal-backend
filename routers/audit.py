"""Audit router — convenience re-export (admin-only).

The main admin router already mounts ``/admin/audit-events``; this module
exists per the file-layout requirement and exposes the same listing under
its own router so it can be imported by tests or future read-only auditor
roles. Currently mounted under the admin prefix as well.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from core.db import Collections, get_db
from core.security import require_admin

router = APIRouter(prefix="/admin/audit", tags=["audit"], dependencies=[Depends(require_admin)])


@router.get("/events")
async def list_events(limit: int = Query(default=100, ge=1, le=500)):
    db = get_db()
    return (
        await db[Collections.AUDIT_EVENTS]
        .find({}, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
        .to_list(length=limit)
    )
