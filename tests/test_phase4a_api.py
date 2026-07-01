"""Phase 4A — OA fetcher API + access control (Task 11 / G)."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from core.db import Collections, get_db
from tests._phase3_helpers import signup
from tests.fixtures_oa_xml import BIOC_SAMPLE, JATS_SAMPLE, PMC_OA_SVC_OK


async def _seed(client, headers, *, n=3, pmcid_prefix="PMC"):
    me = await client.get("/api/auth/me", headers=headers)
    user_id = me.json()["id"]
    db = get_db()
    sid = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": user_id, "status": "awaiting_selection",
        "original_question": "q", "sanitized_question": "q", "pico_json": {},
        "deleted_at": None, "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    aids = []
    for i in range(n):
        aid = uuid.uuid4().hex
        aids.append(aid)
        await db[Collections.ARTICLES].insert_one({
            "id": aid, "canonical_key": f"id:{aid}", "title": f"T{i}",
            "abstract": "abstract " + "x"*30, "doi": None, "pmid": f"100{i}",
            "pmcid": f"{pmcid_prefix}{i}{uuid.uuid4().hex[:5]}",
            "journal": "J", "publication_year": 2024,
            "article_type_classification": "rct",
            "authors_json": [], "source_ids_json": {"inEPMC": "Y"},
            "source_urls_json": [],
            "is_preprint": False, "is_retracted": False, "is_retraction_notice": False,
            "has_expression_of_concern": False, "publication_types_json": [],
            "full_text_status": "unknown", "created_at": now, "updated_at": now,
        })
        await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
            "id": uuid.uuid4().hex, "search_session_id": sid, "article_id": aid,
            "user_included": True, "included_by_default": True, "exclusion_reason": None,
            "relevance_score": 80, "evidence_design_score": 80,
            "overall_rank": 99 - i, "rank_explanation": "test",
            "query_types_matched": [], "sources_present": ["test"], "flags": [],
            "created_at": now, "updated_at": now,
        })
    return sid, aids


@pytest.mark.asyncio
async def test_fetch_oa_full_text_requires_auth(client):
    r = await client.post("/api/search-sessions/x/articles/y/fetch-oa-full-text", json={})
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_non_owner_returns_404_uniform(client):
    t1, _ = await signup(client, "oa-owner@test.com")
    t2, _ = await signup(client, "oa-other@test.com")
    h1 = {"Authorization": f"Bearer {t1}"}
    h2 = {"Authorization": f"Bearer {t2}"}
    sid, aids = await _seed(client, h1)
    r = await client.post(
        f"/api/search-sessions/{sid}/articles/{aids[0]}/fetch-oa-full-text",
        headers=h2, json={},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_article_not_in_session_returns_404(client):
    t, _ = await signup(client, "oa-notin@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _ = await _seed(client, h)
    r = await client.post(
        f"/api/search-sessions/{sid}/articles/{uuid.uuid4().hex}/fetch-oa-full-text",
        headers=h, json={},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_fetch_disabled_returns_safe_disabled_response(client, monkeypatch):
    monkeypatch.setenv("OA_FETCHER_ENABLED", "false")
    t, _ = await signup(client, "oa-disabled@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, aids = await _seed(client, h)
    r = await client.post(
        f"/api/search-sessions/{sid}/articles/{aids[0]}/fetch-oa-full-text",
        headers=h, json={},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "disabled"


@pytest.mark.asyncio
async def test_max_fetches_per_search_enforced(client, monkeypatch):
    monkeypatch.setenv("OA_MAX_FULL_TEXT_FETCHES_PER_SEARCH", "1")
    t, _ = await signup(client, "oa-cap@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, aids = await _seed(client, h, n=3)
    # Patch the fetcher's per-article call to skip HTTP entirely.
    from services.oa_fetcher import oa_fetcher
    async def fake_fetch(**kwargs):
        return {
            "article_id": kwargs["article_id"], "status": "skipped_ineligible",
            "source": None, "license": None, "license_category": None,
            "can_ingest_for_summary": False, "sections_created": 0,
            "token_count": 0, "evidence_text_level": None,
            "error_code": "no_oa_signal",
        }
    monkeypatch.setattr(oa_fetcher, "fetch_for_article", fake_fetch)
    r = await client.post(
        f"/api/search-sessions/{sid}/fetch-oa-full-text",
        headers=h, json={"article_ids": aids},
    )
    assert r.status_code == 200
    body = r.json()
    cap = sum(1 for x in body["results"] if x["status"] == "skipped_cap_exceeded")
    assert cap == 2  # 3 articles - 1 cap = 2 capped


@pytest.mark.asyncio
async def test_admin_oa_metrics_do_not_expose_full_text(client):
    from tests._phase3_helpers import login
    tok = await login(client, "admin@test.com", "AdminTest123!")
    h = {"Authorization": f"Bearer {tok}"}
    r = await client.get("/api/admin/oa-fetcher-metrics", headers=h)
    assert r.status_code == 200
    body = r.json()
    for k in ("section_text", "full_text", "raw_text", "section_texts"):
        assert k not in body


@pytest.mark.asyncio
async def test_clinician_cannot_read_admin_oa_metrics_403(client):
    t, _ = await signup(client, "oa-no-admin@test.com")
    h = {"Authorization": f"Bearer {t}"}
    r = await client.get("/api/admin/oa-fetcher-metrics", headers=h)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_oa_fetch_status_endpoint_returns_per_article_state(client):
    t, _ = await signup(client, "oa-status@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, aids = await _seed(client, h, n=2)
    r = await client.get(f"/api/search-sessions/{sid}/oa-fetch-status", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert "articles" in body
    assert len(body["articles"]) == 2
    for a in body["articles"]:
        assert "article_id" in a
        assert "fetch_status" in a
        assert "eligible" in a
