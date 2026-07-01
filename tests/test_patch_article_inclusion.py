"""Phase 2A.1 — PATCH article inclusion endpoint tests."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.db import Collections, get_db  # noqa: E402


async def _make_session_with_article(client, email: str = "patch@test.com"):
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!", "name": "Patch"},
    )
    token = r.json()["access_token"]
    user_id = r.json()["user"]["id"]
    headers = {"Authorization": f"Bearer {token}"}
    # Directly insert a session + article + session_article row (skip pipeline)
    db = get_db()
    now = datetime.now(timezone.utc)
    sid = f"sess-{email}"
    aid = f"art-{email}"
    sa_id = f"sa-{email}"
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": user_id, "original_question": "test",
        "sanitized_question": "test", "status": "awaiting_selection",
        "pico_json": {}, "expires_at": now, "created_at": now,
        "updated_at": now, "deleted_at": None,
    })
    await db[Collections.ARTICLES].insert_one({
        "id": aid, "canonical_key": f"pmid:{aid}", "title": "T",
        "created_at": now, "updated_at": now,
    })
    await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
        "id": sa_id, "search_session_id": sid, "article_id": aid,
        "relevance_score": 80.0, "evidence_design_score": 80.0,
        "overall_rank": 80, "rank_explanation": "",
        "included_by_default": True, "user_included": True,
        "exclusion_reason": None,
        "created_at": now, "updated_at": now,
    })
    return {"headers": headers, "session_id": sid, "article_id": aid, "sa_id": sa_id, "user_id": user_id}


@pytest.mark.asyncio
async def test_patch_by_canonical_article_id_succeeds(client):
    ctx = await _make_session_with_article(client, "patch1@test.com")
    r = await client.patch(
        f"/api/search-sessions/{ctx['session_id']}/articles/{ctx['article_id']}",
        json={"user_included": False},
        headers=ctx["headers"],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "updated"
    assert body["article_id"] == ctx["article_id"]
    assert body["user_included"] is False


@pytest.mark.asyncio
async def test_patch_by_session_article_row_id_succeeds(client):
    ctx = await _make_session_with_article(client, "patch2@test.com")
    r = await client.patch(
        f"/api/search-sessions/{ctx['session_id']}/articles/{ctx['sa_id']}",
        json={"user_included": False},
        headers=ctx["headers"],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["search_session_article_id"] == ctx["sa_id"]
    assert body["user_included"] is False


@pytest.mark.asyncio
async def test_patch_with_unknown_id_returns_404(client):
    ctx = await _make_session_with_article(client, "patch3@test.com")
    r = await client.patch(
        f"/api/search-sessions/{ctx['session_id']}/articles/no-such-thing",
        json={"user_included": False},
        headers=ctx["headers"],
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_patch_other_users_session_returns_404(client):
    owner = await _make_session_with_article(client, "owner@test.com")
    # Different user
    r = await client.post(
        "/api/auth/signup",
        json={"email": "other@test.com", "password": "Pass1234567!", "name": "Other"},
    )
    other_headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r2 = await client.patch(
        f"/api/search-sessions/{owner['session_id']}/articles/{owner['article_id']}",
        json={"user_included": False},
        headers=other_headers,
    )
    assert r2.status_code == 404


@pytest.mark.asyncio
async def test_patch_unauthenticated_returns_401(client):
    ctx = await _make_session_with_article(client, "patchnoauth@test.com")
    r = await client.patch(
        f"/api/search-sessions/{ctx['session_id']}/articles/{ctx['article_id']}",
        json={"user_included": False},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_patch_response_shape(client):
    ctx = await _make_session_with_article(client, "patchshape@test.com")
    r = await client.patch(
        f"/api/search-sessions/{ctx['session_id']}/articles/{ctx['article_id']}",
        json={"user_included": True},
        headers=ctx["headers"],
    )
    assert r.status_code == 200
    body = r.json()
    for k in ("search_session_article_id", "article_id", "user_included", "status"):
        assert k in body
    assert body["status"] == "updated"
