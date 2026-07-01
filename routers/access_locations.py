"""Access locations router — Phase 2C.

`GET /articles/{article_id}/access-locations` — owner-scoped: caller must
have access to the article via a non-deleted/non-blocked-PHI session OR a
folder they own. Otherwise 404 (uniform with not-found to avoid leaking
existence).

`GET /search-sessions/{session_id}/access-locations` — owner-scoped.
Returns rows per-article keyed by article_id, with persisted system rows
PLUS dynamically-generated institutional rows from the current user's
institution settings. Institutional rows are NEVER persisted — they're
built per-request to prevent leaking one user's institution into another
user's article view.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from core.db import Collections, get_db
from core.security import get_current_user
from services.full_text_access_broker import institutional_rows_for_user

router = APIRouter(tags=["access-locations"])


async def _user_has_article(user_id: str, article_id: str) -> bool:
    db = get_db()
    # Path 1: article surfaced via a non-deleted, non-PHI-blocked search session.
    pipeline = [
        {"$match": {"article_id": article_id}},
        {"$lookup": {
            "from": "search_sessions",
            "localField": "search_session_id",
            "foreignField": "id",
            "as": "session",
        }},
        {"$unwind": "$session"},
        {"$match": {
            "session.user_id": user_id,
            "session.deleted_at": None,
            "session.status": {"$ne": "blocked_phi"},
        }},
        {"$limit": 1},
    ]
    cursor = db[Collections.SEARCH_SESSION_ARTICLES].aggregate(pipeline)
    if await cursor.to_list(length=1):
        return True
    # Path 2: article saved in a folder the user owns.
    pipeline2 = [
        {"$match": {"article_id": article_id}},
        {"$lookup": {
            "from": "folders",
            "localField": "folder_id",
            "foreignField": "id",
            "as": "folder",
        }},
        {"$unwind": "$folder"},
        {"$match": {"folder.user_id": user_id}},
        {"$limit": 1},
    ]
    cursor2 = db[Collections.FOLDER_ARTICLES].aggregate(pipeline2)
    return bool(await cursor2.to_list(length=1))


async def _get_user_settings(user_id: str) -> dict | None:
    db = get_db()
    return await db[Collections.USER_INSTITUTION_SETTINGS].find_one(
        {"user_id": user_id}, {"_id": 0}
    )


@router.get("/articles/{article_id}/access-locations")
async def list_article_access_locations(
    article_id: str, current_user: dict = Depends(get_current_user)
):
    db = get_db()
    has_access = await _user_has_article(current_user["id"], article_id)
    if not has_access:
        raise HTTPException(status_code=404, detail="Article not found.")
    raw_rows = await (
        db[Collections.ARTICLE_ACCESS_LOCATIONS]
        .find({"article_id": article_id}, {"_id": 0})
        .to_list(length=200)
    )
    # Phase 2D: user-uploaded full-text access rows are scoped to the owning
    # user. Filter them out for any other user's view.
    rows = [
        r for r in raw_rows
        if r.get("access_category") != "user_uploaded_full_text"
        or r.get("user_id") == current_user["id"]
    ]
    article = await db[Collections.ARTICLES].find_one({"id": article_id}, {"_id": 0})
    settings = await _get_user_settings(current_user["id"])
    if article and settings:
        # Build dynamic institutional rows — never persisted.
        rows.extend(institutional_rows_for_user(article, settings))
    return rows


@router.get("/search-sessions/{session_id}/access-locations")
async def list_session_access_locations(
    session_id: str, current_user: dict = Depends(get_current_user)
):
    db = get_db()
    session = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "user_id": current_user["id"], "deleted_at": None},
        {"_id": 0, "status": 1},
    )
    if not session or session.get("status") == "blocked_phi":
        raise HTTPException(status_code=404, detail="Search session not found.")
    sa_rows = await (
        db[Collections.SEARCH_SESSION_ARTICLES]
        .find({"search_session_id": session_id}, {"_id": 0, "article_id": 1})
        .to_list(length=500)
    )
    article_ids = sorted({r["article_id"] for r in sa_rows})
    if not article_ids:
        return {}
    rows_by_aid: dict[str, list[dict]] = {aid: [] for aid in article_ids}
    cursor = db[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": {"$in": article_ids}}, {"_id": 0}
    )
    async for row in cursor:
        # Phase 2D: user-uploaded full-text rows are per-user. Hide from
        # others. A row written by user A is invisible to user B.
        if (
            row.get("access_category") == "user_uploaded_full_text"
            and row.get("user_id") != current_user["id"]
        ):
            continue
        rows_by_aid.setdefault(row["article_id"], []).append(row)

    settings = await _get_user_settings(current_user["id"])
    if settings:
        async for art in db[Collections.ARTICLES].find(
            {"id": {"$in": article_ids}}, {"_id": 0}
        ):
            inst = institutional_rows_for_user(art, settings)
            if inst:
                rows_by_aid.setdefault(art["id"], []).extend(inst)
    return rows_by_aid
