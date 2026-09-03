"""Phase 4A — cache + idempotency (Task 11 / E)."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from core.db import Collections, get_db
from services.oa_fetcher import oa_fetcher
from services.oa_fetcher.cache import (
    CACHE_DAYS, lookup_article_cache, write_cache_rows,
)
from tests.fixtures_oa_xml import JATS_SAMPLE


def _now():
    return datetime.now(timezone.utc)


class _MockResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


def _client_returning(*responses):
    cli = AsyncMock(spec=httpx.AsyncClient)
    cli.get = AsyncMock(side_effect=list(responses))
    return cli


async def _seed_article_session(*, user_id="u1", pmcid="PMC1234567"):
    db = get_db()
    sid = uuid.uuid4().hex
    aid = uuid.uuid4().hex
    now = _now()
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": user_id, "status": "awaiting_selection",
        "original_question": "q", "sanitized_question": "q", "pico_json": {},
        "deleted_at": None, "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    await db[Collections.ARTICLES].insert_one({
        "id": aid, "canonical_key": f"pmid:{pmcid}", "title": "Test",
        "abstract": "abstract", "doi": None, "pmid": "1000",
        "pmcid": pmcid, "journal": "J", "publication_year": 2024,
        "article_type_classification": "rct",
        "authors_json": [], "source_ids_json": {}, "source_urls_json": [],
        "is_preprint": False, "is_retracted": False, "is_retraction_notice": False,
        "has_expression_of_concern": False, "publication_types_json": [],
        "full_text_status": "unknown", "created_at": now, "updated_at": now,
    })
    await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
        "id": uuid.uuid4().hex, "search_session_id": sid, "article_id": aid,
        "user_included": True, "included_by_default": True, "exclusion_reason": None,
        "relevance_score": 80, "evidence_design_score": 80,
        "overall_rank": 99, "rank_explanation": "test",
        "query_types_matched": [], "sources_present": ["test"], "flags": [],
        "created_at": now, "updated_at": now,
    })
    return sid, aid


@pytest.mark.asyncio
async def test_cache_hit_does_not_call_network():
    """First fetch warms the cache; second fetch must not call the network."""
    sid, aid = await _seed_article_session()
    db = get_db()
    # Seed a cache row directly so we don't depend on HTTP for the warm-up.
    now = _now()
    await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one({
        "id": uuid.uuid4().hex, "article_id": aid,
        "search_session_id": None, "text_source": "oa_full_text",
        "section_name": "abstract", "section_text": "cached abstract text " * 5,
        "token_count": 12, "include_in_summary": True,
        "evidence_text_level": "oa_full_text", "created_at": now,
        "expires_at": None,
        "oa_cache": True, "oa_source": "europe_pmc_oa",
        "oa_license": "cc-by", "oa_license_category": "commercial_reuse_allowed",
        "oa_fetch_checksum": "abc", "oa_last_fetched_at": now,
        "oa_expires_at": now + timedelta(days=CACHE_DAYS),
    })
    # Pass a client whose .get would raise — proves the cascade is short-circuited.
    cli = AsyncMock(spec=httpx.AsyncClient)
    cli.get = AsyncMock(side_effect=AssertionError("network must not be called on cache hit"))
    res = await oa_fetcher.fetch_for_article(
        article_id=aid, session_id=sid, user_id="u1", client=cli,
    )
    assert res["status"] == "cache_hit"
    cli.get.assert_not_called()


@pytest.mark.asyncio
async def test_cache_expires_per_OA_CACHE_DAYS():
    sid, aid = await _seed_article_session()
    db = get_db()
    now = _now()
    # Insert an expired cache row.
    await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one({
        "id": uuid.uuid4().hex, "article_id": aid,
        "search_session_id": None, "text_source": "oa_full_text",
        "section_name": "abstract", "section_text": "old cached text " * 5,
        "token_count": 10, "include_in_summary": True,
        "evidence_text_level": "oa_full_text", "created_at": now - timedelta(days=60),
        "oa_cache": True, "oa_source": "europe_pmc_oa",
        "oa_license": "cc-by", "oa_license_category": "commercial_reuse_allowed",
        "oa_expires_at": now - timedelta(days=5),  # already expired
    })
    cached = await lookup_article_cache(aid)
    assert cached == []  # expired rows MUST not surface


@pytest.mark.asyncio
async def test_cache_only_when_commercial_reuse_allowed():
    """write_cache_rows must refuse non-commercial-reuse licenses."""
    sid, aid = await _seed_article_session(pmcid="PMC222")
    sections = [{"section_name": "abstract", "section_text": "x" * 60,
                 "token_count": 5, "include_in_summary": True}]
    n_nc = await write_cache_rows(
        article_id=aid, oa_source="europe_pmc_oa",
        oa_license="cc-by-nc", oa_license_category="noncommercial_only",
        sections=sections,
    )
    assert n_nc == 0
    n_ok = await write_cache_rows(
        article_id=aid, oa_source="europe_pmc_oa",
        oa_license="cc-by", oa_license_category="commercial_reuse_allowed",
        sections=sections,
    )
    assert n_ok == 1


@pytest.mark.asyncio
async def test_cache_never_stores_user_upload_text():
    """User-uploaded extracts have text_source='user_uploaded_full_text' and
    are NEVER selected by the OA cache lookup.
    """
    sid, aid = await _seed_article_session(pmcid="PMC333")
    db = get_db()
    now = _now()
    await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one({
        "id": uuid.uuid4().hex, "article_id": aid,
        "search_session_id": None, "user_id": "someone-else",
        "text_source": "user_uploaded_full_text",
        "section_name": "results", "section_text": "user uploaded content " * 5,
        "token_count": 10, "include_in_summary": True,
        "evidence_text_level": "user_uploaded_full_text",
        "created_at": now, "oa_cache": False, "oa_source": None,
    })
    cached = await lookup_article_cache(aid)
    assert cached == []


@pytest.mark.asyncio
async def test_force_refresh_busts_cache():
    sid, aid = await _seed_article_session(pmcid="PMC444")
    db = get_db()
    now = _now()
    await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one({
        "id": uuid.uuid4().hex, "article_id": aid,
        "search_session_id": None, "text_source": "oa_full_text",
        "section_name": "abstract", "section_text": "stale abstract " * 5,
        "token_count": 8, "include_in_summary": True,
        "evidence_text_level": "oa_full_text", "created_at": now,
        "oa_cache": True, "oa_source": "europe_pmc_oa",
        "oa_license": "cc-by", "oa_license_category": "commercial_reuse_allowed",
        "oa_expires_at": now + timedelta(days=10),
    })
    cli = _client_returning(_MockResp(200, JATS_SAMPLE.encode("utf-8")))
    res = await oa_fetcher.fetch_for_article(
        article_id=aid, session_id=sid, user_id="u1",
        force_refresh=True, client=cli,
    )
    assert res["status"] == "fetched"  # not cache_hit
    cli.get.assert_called()  # the network WAS called


@pytest.mark.asyncio
async def test_cache_hit_creates_session_scoped_clones():
    sid, aid = await _seed_article_session(pmcid="PMC555")
    db = get_db()
    now = _now()
    await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one({
        "id": uuid.uuid4().hex, "article_id": aid,
        "search_session_id": None, "text_source": "oa_full_text",
        "section_name": "abstract", "section_text": "cached abstract " * 5,
        "token_count": 12, "include_in_summary": True,
        "evidence_text_level": "oa_full_text", "created_at": now,
        "oa_cache": True, "oa_source": "europe_pmc_oa",
        "oa_license": "cc-by", "oa_license_category": "commercial_reuse_allowed",
        "oa_expires_at": now + timedelta(days=30),
    })
    res = await oa_fetcher.fetch_for_article(
        article_id=aid, session_id=sid, user_id="u1",
    )
    assert res["status"] == "cache_hit"
    # A session-scoped clone row should now exist.
    rows = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
        {"article_id": aid, "search_session_id": sid, "oa_cache": {"$ne": True}},
        {"_id": 0},
    ).to_list(length=10)
    assert len(rows) >= 1
    assert rows[0]["text_source"] == "oa_full_text"
    # Session row expires_at must match session.expires_at (or be set).
    assert rows[0]["expires_at"] is not None


@pytest.mark.asyncio
async def test_session_scoped_clones_expire_with_session():
    """Session-scoped clone's `expires_at` matches its parent session, so
    retention cleanup deletes the clones when the session expires.
    """
    sid, aid = await _seed_article_session(pmcid="PMC666")
    db = get_db()
    now = _now()
    await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one({
        "id": uuid.uuid4().hex, "article_id": aid,
        "search_session_id": None, "text_source": "oa_full_text",
        "section_name": "abstract", "section_text": "cached abstract " * 5,
        "token_count": 12, "include_in_summary": True,
        "evidence_text_level": "oa_full_text", "created_at": now,
        "oa_cache": True, "oa_source": "europe_pmc_oa",
        "oa_license": "cc-by", "oa_license_category": "commercial_reuse_allowed",
        "oa_expires_at": now + timedelta(days=30),
    })
    res = await oa_fetcher.fetch_for_article(
        article_id=aid, session_id=sid, user_id="u1",
    )
    assert res["status"] == "cache_hit"
    # Force-expire session and run retention cleanup.
    past = now - timedelta(days=1)
    await db[Collections.SEARCH_SESSIONS].update_one({"id": sid}, {"$set": {"expires_at": past}})
    from services.retention_cleanup import retention_cleanup
    out = await retention_cleanup.cleanup_expired_sessions()
    assert out["deleted_count"] >= 1
    # The cache row remains; only the session clone is gone.
    cache_rows = await db[Collections.ARTICLE_TEXT_EXTRACTS].count_documents({
        "article_id": aid, "oa_cache": True,
    })
    assert cache_rows >= 1
    session_rows = await db[Collections.ARTICLE_TEXT_EXTRACTS].count_documents({
        "article_id": aid, "search_session_id": sid,
    })
    assert session_rows == 0
