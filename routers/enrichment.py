"""Per-record enrichment endpoint — frontend compatibility.

The integrated LitPulse/LitBridge frontend fetches `GET /api/enrichment/{search_id}/{record_id}`
for each result card to show citation count, open-access status, and a PDF link.
That route existed on the old LitBridge backend; this reimplements a compatible
shape on litportal so the frontend stops getting a burst of 404s.

`record_id` is the frontend's card id — i.e. the ``search_session_articles`` ROW
id (scoped to the session). We map it to the underlying article and derive the
enrichment from data litportal already has (access-broker locations + any stored
citation count). Unknown records return a benign 200 (never a 404 burst).
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from core.db import Collections, get_db
from services.full_text_access_broker import compute_best_access

router = APIRouter(prefix="/enrichment", tags=["enrichment"])


class EnrichmentResponse(BaseModel):
    id: str
    tldr: Optional[str] = None
    citation_count: Optional[int] = None
    oa_status: Optional[str] = None  # open | closed | unknown
    pdf_url: Optional[str] = None


async def _resolve_article_id(db, search_id: str, record_id: str) -> Optional[str]:
    """record_id is normally the search_session_articles ROW id (scoped to the
    session); fall back to treating it as an article id."""
    ssa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"id": record_id, "search_session_id": search_id},
        {"_id": 0, "article_id": 1},
    )
    if ssa and ssa.get("article_id"):
        return ssa["article_id"]
    # Fallbacks: the row id without session scope, or an article id directly.
    ssa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"id": record_id}, {"_id": 0, "article_id": 1},
    )
    if ssa and ssa.get("article_id"):
        return ssa["article_id"]
    art = await db[Collections.ARTICLES].find_one({"id": record_id}, {"_id": 0, "id": 1})
    return art["id"] if art else None


def _citation_count(art: dict[str, Any]) -> Optional[int]:
    for key in ("citation_count", "icite_citation_count"):
        v = art.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return int(v)
    icite = art.get("icite_metrics") or {}
    if isinstance(icite, dict):
        v = icite.get("citation_count")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def _oa_status_and_pdf(art: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, Optional[str]]:
    # Prefer a direct OA PDF, then any OA route, then fall back to full_text_status.
    oa_pdf = next(
        (r for r in rows if r.get("is_oa") and r.get("is_direct_pdf") and r.get("url")), None,
    )
    if oa_pdf:
        return "open", oa_pdf["url"]
    if any(r.get("is_oa") for r in rows):
        viewable = next(
            (r for r in rows if r.get("is_oa") and r.get("url") and r.get("can_view_in_app")), None,
        )
        if viewable:
            return "open", viewable["url"]
        best = compute_best_access(art, rows)
        return "open", (None if best.get("disabled") else best.get("target_url"))
    fts = (art.get("full_text_status") or "").lower()
    if fts in ("full_text_available", "free_full_text_link"):
        best = compute_best_access(art, rows)
        return "open", (None if best.get("disabled") else best.get("target_url"))
    return "closed" if fts == "abstract_only" else "unknown", None


@router.get("/{search_id}/{record_id}", response_model=EnrichmentResponse)
async def get_enrichment(
    search_id: str,
    record_id: str,
) -> EnrichmentResponse:
    # NOTE: intentionally unauthenticated — the frontend fetches enrichment
    # without an Authorization header (matches the old LitBridge endpoint), and
    # the payload is non-sensitive public bibliographic metadata (citation count,
    # OA status, PDF link). No PHI, no user-scoped data.
    db = get_db()
    article_id = await _resolve_article_id(db, search_id, record_id)
    if not article_id:
        # Unknown record — benign 200 so the frontend never sees a 404 burst.
        return EnrichmentResponse(id=record_id, oa_status="unknown")
    art = await db[Collections.ARTICLES].find_one({"id": article_id}, {"_id": 0}) or {}
    rows: list[dict[str, Any]] = []
    async for r in db[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": article_id}, {"_id": 0},
    ):
        rows.append(r)
    oa_status, pdf_url = _oa_status_and_pdf(art, rows)
    return EnrichmentResponse(
        id=record_id,
        tldr=None,
        citation_count=_citation_count(art),
        oa_status=oa_status,
        pdf_url=pdf_url,
    )
