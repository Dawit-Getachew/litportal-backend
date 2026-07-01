"""Phase K4B-light — Known Article Finder router.

Mounted under ``/api/known-article-lookup`` via ``server.py``.

Endpoints:
  - POST  /parse              — pure-function parse, no outbound calls
  - POST  /search             — PHI-guarded full lookup, returns candidates
  - POST  /confirm-access     — resolve access for a single candidate
  - POST  /attach-to-session  — attach confirmed candidate to a session

All routes require ``get_current_user``. No admin scope; clinicians
operate this surface directly.
"""
from __future__ import annotations

from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.security import get_current_user
from core.db import get_db
from services.known_article_lookup import (
    known_article_lookup_service,
    parse_input,
    ACCESS_STATUS_LABELS,
    AI_SUMMARY_TEXT_SOURCE_LABELS,
)
from services.audit import audit_service


router = APIRouter(prefix="/known-article-lookup", tags=["known-article-lookup"])


class ParseRequest(BaseModel):
    raw_text: str = Field(..., max_length=4000)


class SearchRequest(BaseModel):
    raw_text: str = Field(..., max_length=4000)
    session_id: Optional[str] = None


class Candidate(BaseModel):
    """Closed-set candidate payload. Used as request body for confirm + attach."""
    match_confidence: str
    match_reason_codes: list[str] = Field(default_factory=list)
    pmid: Optional[str] = None
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    authors: list[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    publication_types: list[str] = Field(default_factory=list)
    is_retracted: bool = False
    has_expression_of_concern: bool = False
    source: Optional[str] = None


class ConfirmAccessRequest(BaseModel):
    candidate: Candidate


class AttachToSessionRequest(BaseModel):
    session_id: str
    candidate: Candidate
    include_in_evidence: bool = False


@router.post("/parse")
async def parse_endpoint(
    body: ParseRequest,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Lightweight parse-only endpoint. NO outbound calls. Returns the
    closed-set parsed envelope so the UI can show a preview before
    spending a full /search call."""
    parsed = parse_input(body.raw_text)
    return parsed.model_dump()


@router.post("/search")
async def search_endpoint(
    body: SearchRequest,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """PHI-guarded search. Returns blocked envelope OR candidate list."""
    result = await known_article_lookup_service.search(body.raw_text)
    # Audit — closed-set payload only. NO raw text, NO PHI.
    await audit_service.log(
        "known_article_lookup_search",
        user_id=current_user["id"],
        search_session_id=body.session_id,
        payload={
            "phi_blocked": result.get("phi_blocked"),
            "detected_kind": (result.get("parsed") or {}).get("detected_kind"),
            "lookup_source_provenance": result.get("lookup_source_provenance") or [],
            "candidate_count": len(result.get("candidates") or []),
            "top_confidence": (
                (result.get("candidates") or [{}])[0].get("match_confidence")
                if result.get("candidates") else None
            ),
        },
    )
    return result


@router.post("/confirm-access")
async def confirm_access_endpoint(
    body: ConfirmAccessRequest,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Resolve access for a single confirmed candidate."""
    db = get_db()
    panel = await known_article_lookup_service.confirm_access(
        db=db,
        candidate=body.candidate.model_dump(),
        user_id=current_user["id"],
    )
    panel["access_status_label_options"] = list(ACCESS_STATUS_LABELS)
    panel["ai_summary_text_source_label_options"] = list(
        AI_SUMMARY_TEXT_SOURCE_LABELS,
    )
    return panel


@router.post("/attach-to-session")
async def attach_to_session_endpoint(
    body: AttachToSessionRequest,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Attach a confirmed candidate to a search session.

    Enforces the closed-set rule: low-confidence candidates may NOT be
    silently attached. The router gates here so the rule is visible.
    """
    if body.candidate.match_confidence == "low":
        raise HTTPException(
            status_code=400,
            detail=(
                "Low-confidence match requires explicit user confirmation "
                "via the modal — call /attach-to-session only after the "
                "clinician has confirmed the candidate."
            ),
        )
    if body.candidate.is_retracted or body.candidate.has_expression_of_concern:
        # Mirror the card-view safety block. Caller must surface the
        # closed-set message to the clinician.
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot attach retracted or expression-of-concern article. "
                "Open the article details to review the safety flag."
            ),
        )
    db = get_db()
    result = await known_article_lookup_service.attach_to_session(
        db=db,
        session_id=body.session_id,
        user_id=current_user["id"],
        candidate=body.candidate.model_dump(),
        include_in_evidence=bool(body.include_in_evidence),
    )
    if result.get("status") == "session_not_found":
        raise HTTPException(status_code=404, detail="Search session not found.")
    if result.get("status") == "session_blocked_phi":
        raise HTTPException(
            status_code=400,
            detail="Cannot attach to a session that is blocked for PHI.",
        )
    return result
