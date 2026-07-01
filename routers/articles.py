"""Articles router — Phase 1 placeholder (read-only, mostly empty)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from core.db import Collections, get_db
from core.security import get_current_user

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("/")
async def list_articles(
    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    # Phase 1: no article ingestion yet, return paginated empty list shape.
    return {"items": [], "limit": limit, "skip": skip, "total": 0}


@router.get("/{article_id}")
async def get_article(article_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    doc = await db[Collections.ARTICLES].find_one({"id": article_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Article not found.")
    # Phase H2-tail Task A — defensive author-email scrub on read so
    # legacy records ingested before the centralized persistence
    # redactor was wired never leak emails through this endpoint.
    from services.article_email_redactor import redact_article_emails
    return redact_article_emails(doc)
