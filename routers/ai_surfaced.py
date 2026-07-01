"""AI-surfaced evidence shelf (F4 backend) — read-only, preview-gated.

Exposes Agent 3 (EvidenceCurator) output for a search session as a SEPARATE
shelf, never merged into the deterministic ``articles``/``excluded_articles``.
Visible only when a preview/user-visible flag is on. Isolated from the large
search-sessions router on purpose.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from core.config import settings
from core.db import Collections, get_db
from core.security import get_current_user

router = APIRouter(prefix="/search-sessions", tags=["ai-surfaced"])

_ARTICLE_PROJECTION = {
    "_id": 0, "id": 1, "title": 1, "authors_json": 1, "journal": 1,
    "publication_year": 1, "doi": 1, "pmid": 1, "pmcid": 1,
    "study_design": 1, "is_guideline": 1, "full_text_status": 1,
}


@router.get("/{session_id}/ai-surfaced")
async def get_ai_surfaced(session_id: str, current_user: dict = Depends(get_current_user)):
    """Return the AI-surfaced candidate shelf for a session (preview-gated)."""
    db = get_db()
    session = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id}, {"_id": 0, "id": 1, "user_id": 1},
    )
    if not session:
        raise HTTPException(status_code=404, detail="search_session_not_found")
    is_admin = current_user.get("role") == "admin"
    if session.get("user_id") != current_user["id"] and not is_admin:
        raise HTTPException(status_code=404, detail="search_session_not_found")

    visible = bool(
        settings.AI_EVIDENCE_SURFACING_USER_VISIBLE_ALLOWED
        or settings.AI_EVIDENCE_SURFACING_PREVIEW_VISIBLE
        or is_admin
    )
    if not visible:
        return {"session_id": session_id, "visible": False, "items": []}

    rows = await db[Collections.AI_SURFACED_ARTICLES].find(
        {"search_session_id": session_id}, {"_id": 0},
    ).sort("confidence", -1).to_list(length=200)

    article_ids = [r["article_id"] for r in rows]
    art_map: dict[str, dict] = {}
    if article_ids:
        async for a in db[Collections.ARTICLES].find(
            {"id": {"$in": article_ids}}, _ARTICLE_PROJECTION,
        ):
            art_map[a["id"]] = a
        if art_map:
            from services.article_email_redactor import redact_article_emails
            for _aid in list(art_map.keys()):
                art_map[_aid] = redact_article_emails(art_map[_aid])

    items = [
        {
            "article_id": r["article_id"],
            "advisory_label": r.get("advisory_label"),
            "confidence": r.get("confidence"),
            "rationale": r.get("rationale"),
            "original_exclusion_reason": r.get("original_exclusion_reason"),
            "article": art_map.get(r["article_id"]),
        }
        for r in rows
    ]
    return {"session_id": session_id, "visible": True, "count": len(items), "items": items}
