"""Cleanup router — admin-only trigger for retention sweep.

Companion to ``POST /api/admin/cleanup`` defined in admin.py. Exists per the
file-layout requirement and exposes a convenience endpoint under the same
admin prefix.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from core.security import require_admin
from services.audit import audit_service
from services.retention_cleanup import retention_cleanup

router = APIRouter(prefix="/admin/cleanup", tags=["cleanup"], dependencies=[Depends(require_admin)])


@router.post("/run")
async def run_cleanup(current_admin: dict = Depends(require_admin)):
    result = await retention_cleanup.cleanup_expired_sessions()
    await audit_service.log(
        "admin_cleanup_triggered",
        user_id=current_admin["id"],
        payload=result,
    )
    return result
