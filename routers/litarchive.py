"""B14 — LitArchive router: filterable repository/table view over the user's
saved articles (library + research collections + folders), hydrated from the
canonical article store. Read-only; user-scoped.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from core.security import get_current_user
from services.litarchive_service import litarchive_service

router = APIRouter(prefix="/litarchive", tags=["litarchive"])


@router.get("")
async def list_litarchive(
    source: Optional[str] = Query(default=None),
    journal: Optional[str] = Query(default=None),
    year: Optional[int] = Query(default=None),
    pmid: Optional[str] = Query(default=None),
    doi: Optional[str] = Query(default=None),
    saved_from: Optional[str] = Query(default=None, pattern="^(library|collection|folder)$"),
    collection_id: Optional[str] = Query(default=None),
    folder_id: Optional[str] = Query(default=None),
    full_text_available: Optional[bool] = Query(default=None),
    ai_summary_eligible: Optional[bool] = Query(default=None),
    q: Optional[str] = Query(default=None, max_length=200),
    limit: int = Query(default=100, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """Repository/table view over LitHub-saved content with filters + columns."""
    return await litarchive_service.query(
        current_user["id"],
        source=source,
        journal=journal,
        year=year,
        pmid=pmid,
        doi=doi,
        saved_from=saved_from,
        collection_id=collection_id,
        folder_id=folder_id,
        full_text_available=full_text_available,
        ai_summary_eligible=ai_summary_eligible,
        q=q,
        limit=limit,
        skip=skip,
    )
