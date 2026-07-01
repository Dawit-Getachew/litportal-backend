"""Source configs router — public (admin only) read-only convenience endpoint.

The main admin router exposes the canonical CRUD-ish endpoints. This module
exists per the file-layout requirement.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from core.db import Collections, get_db
from core.security import require_admin

router = APIRouter(prefix="/admin/source-configs-readonly", tags=["source-configs"], dependencies=[Depends(require_admin)])


@router.get("/")
async def read_all():
    db = get_db()
    return await db[Collections.SOURCE_CONFIGS].find({}, {"_id": 0}).sort("source_name", 1).to_list(length=200)


@router.get("/guideline-sources")
async def read_guideline_sources():
    """Phase K5.0 — admin-only read of the guideline source registry
    (Northstar §12.1 permission model). Read-only; no network."""
    db = get_db()
    return (
        await db[Collections.GUIDELINE_SOURCES]
        .find({}, {"_id": 0})
        .sort("source_name", 1)
        .to_list(length=200)
    )
