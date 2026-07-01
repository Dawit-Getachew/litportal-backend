"""B15 — single-paper full-text finder/import router.

The LitPulse "Full text import" button calls this. Given ONE paper's PMID / DOI
(or, as a fallback, its title), it resolves the paper, finds & ingests its OA
full text (license-gated; link-out otherwise), attaches it to a ready
single-article session the clinician can summarize ("evidence chat") or save.

Mounted under ``/api/full-text-import`` (and ``/api/litportal/full-text-import``
when the LitPulse module mount is enabled). Requires ``get_current_user``.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from core.config import get_settings
from core.security import get_current_user
from services.full_text_import import FullTextImportError, full_text_import_service

router = APIRouter(prefix="/full-text-import", tags=["full-text-import"])


class FullTextImportRequest(BaseModel):
    """At least one of pmid / doi / pmcid / title is required."""

    pmid: Optional[str] = Field(default=None, max_length=20)
    doi: Optional[str] = Field(default=None, max_length=200)
    pmcid: Optional[str] = Field(default=None, max_length=20)
    title: Optional[str] = Field(default=None, max_length=512)
    # Reuse an existing caller-owned session instead of creating an import one.
    session_id: Optional[str] = None
    # Import the top candidate even when the match is low confidence
    # (typically only relevant for title-only lookups).
    accept_low_confidence: bool = False
    # Re-run discovery + OA fetch even if a cached/import session exists.
    force_refresh: bool = False
    # Also save the paper to the user's library (a research collection).
    save: bool = False
    collection_id: Optional[str] = None

    @model_validator(mode="after")
    def _need_one_identifier(self) -> "FullTextImportRequest":
        if not any((
            (self.pmid or "").strip(),
            (self.doi or "").strip(),
            (self.pmcid or "").strip(),
            (self.title or "").strip(),
        )):
            raise ValueError("Provide at least one of: pmid, doi, pmcid, title.")
        return self


@router.post("")
@router.post("/")
async def full_text_import(
    body: FullTextImportRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.FULL_TEXT_IMPORT_ENABLED:
        raise HTTPException(status_code=404, detail="Full-text import is disabled.")

    # Reuse the shared app-level httpx client when present (set in lifespan).
    http_client = getattr(request.app.state, "http_client", None)

    try:
        return await full_text_import_service.resolve(
            user_id=current_user["id"],
            pmid=body.pmid, doi=body.doi, pmcid=body.pmcid, title=body.title,
            session_id=body.session_id,
            accept_low_confidence=body.accept_low_confidence,
            force_refresh=body.force_refresh,
            save=body.save, collection_id=body.collection_id,
            http_client=http_client,
        )
    except FullTextImportError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.code) from exc
