"""Phase 4B — OA retry endpoint + failed-verification guidance tests."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from core.db import Collections, get_db
from services.summarization.prompts import PROMPT_VERSION
from tests._phase3_helpers import signup


async def _seed_session_article(client, headers, *, prior_error_code=None,
                                 prior_attempts=0, prior_retry_offset_s=999):
    me = (await client.get("/api/auth/me", headers=headers)).json()
    db = get_db()
    now = datetime.now(timezone.utc)
    sid = uuid.uuid4().hex
    aid = uuid.uuid4().hex
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": me["id"], "status": "awaiting_selection",
        "original_question": "q", "sanitized_question": "q", "pico_json": {},
        "deleted_at": None, "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    await db[Collections.ARTICLES].insert_one({
        "id": aid, "canonical_key": f"id:{aid}", "title": "T",
        "abstract": "abstract", "doi": None, "pmid": "200001",
        "pmcid": f"PMC{uuid.uuid4().hex[:6]}", "journal": "J",
        "publication_year": 2024, "article_type_classification": "rct",
        "authors_json": [], "source_ids_json": {"inEPMC": "Y"},
        "source_urls_json": [], "is_preprint": False,
        "is_retracted": False, "is_retraction_notice": False,
        "has_expression_of_concern": False,
        "publication_types_json": [], "full_text_status": "unknown",
        "created_at": now, "updated_at": now,
    })
    await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
        "id": uuid.uuid4().hex, "search_session_id": sid, "article_id": aid,
        "user_included": True, "included_by_default": True,
        "exclusion_reason": None, "relevance_score": 80,
        "evidence_design_score": 80, "overall_rank": 1,
        "rank_explanation": "t", "query_types_matched": [],
        "sources_present": [], "flags": [],
        "created_at": now, "updated_at": now,
    })
    if prior_error_code is not None:
        await db[Collections.ARTICLE_ACCESS_LOCATIONS].insert_one({
            "id": uuid.uuid4().hex,
            "article_id": aid,
            "search_session_id": sid,
            "source": "europe_pmc_oa",
            "access_category": "europe_pmc_oa",
            "url": "https://www.ebi.ac.uk/europepmc/article/PMC/",
            "is_direct_pdf": False, "is_oa": False, "requires_login": False,
            "can_view_in_app": True, "can_ingest_for_summary": False,
            "license": None, "license_category": "unknown",
            "version": "fetched", "evidence_text_level": None,
            "oa_fetch_status": "failed",
            "oa_fetch_error_code": prior_error_code,
            "oa_sections_created": 0, "oa_token_count": 0,
            "oa_last_fetched_at": now,
            "oa_retry_attempt_count": prior_attempts,
            "oa_last_retry_at": now - timedelta(seconds=prior_retry_offset_s),
            "last_checked_at": now,
            "created_at": now, "updated_at": now,
        })
    return sid, aid


@pytest.mark.asyncio
async def test_retry_oa_fetch_disabled_returns_safe_payload(client, monkeypatch):
    monkeypatch.setenv("OA_FETCHER_ENABLED", "false")
    t, _ = await signup(client, "phase4b-oa-retry-disabled@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, aid = await _seed_session_article(client, h)
    r = await client.post(
        f"/api/search-sessions/{sid}/articles/{aid}/retry-oa-fetch",
        headers=h, json={},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "disabled"


@pytest.mark.asyncio
async def test_retry_oa_fetch_returns_terminal_for_retracted(client):
    t, _ = await signup(client, "phase4b-oa-retry-term@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, aid = await _seed_session_article(
        client, h, prior_error_code="article_retracted",
    )
    r = await client.post(
        f"/api/search-sessions/{sid}/articles/{aid}/retry-oa-fetch",
        headers=h, json={},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "terminal"
    assert body["error_code"] == "article_retracted"


@pytest.mark.asyncio
async def test_retry_oa_fetch_rate_limited_after_max_attempts(client, monkeypatch):
    monkeypatch.setenv("OA_RETRY_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("OA_RETRY_MIN_INTERVAL_SECONDS", "0")
    t, _ = await signup(client, "phase4b-oa-retry-cap@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, aid = await _seed_session_article(
        client, h, prior_error_code="europe_pmc_failed",
        prior_attempts=2, prior_retry_offset_s=999,
    )
    r = await client.post(
        f"/api/search-sessions/{sid}/articles/{aid}/retry-oa-fetch",
        headers=h, json={},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "rate_limited"
    assert body["error_code"] == "max_retries_exceeded"
    assert body["attempts"] == 2
    assert body["max_attempts"] == 2


@pytest.mark.asyncio
async def test_retry_oa_fetch_too_soon(client, monkeypatch):
    monkeypatch.setenv("OA_RETRY_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("OA_RETRY_MIN_INTERVAL_SECONDS", "60")
    t, _ = await signup(client, "phase4b-oa-retry-soon@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, aid = await _seed_session_article(
        client, h, prior_error_code="europe_pmc_failed",
        prior_attempts=1, prior_retry_offset_s=5,  # last retry 5s ago < 60s
    )
    r = await client.post(
        f"/api/search-sessions/{sid}/articles/{aid}/retry-oa-fetch",
        headers=h, json={},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "rate_limited"
    assert body["error_code"] == "retry_too_soon"
    assert body["retry_after_seconds"] >= 1


# ---------------------------------------------------------------------------
# Failed-verification guidance
# ---------------------------------------------------------------------------


async def _create_failed_summary(client, headers, *, failed_reason):
    me = (await client.get("/api/auth/me", headers=headers)).json()
    db = get_db()
    now = datetime.now(timezone.utc)
    sid = uuid.uuid4().hex
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": me["id"], "status": "awaiting_selection",
        "original_question": "q", "sanitized_question": "q", "pico_json": {},
        "deleted_at": None, "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    await db[Collections.SUMMARIES].insert_one({
        "id": uuid.uuid4().hex,
        "search_session_id": sid,
        "selected_article_ids_json": [],
        "answer_json": {},  # NEVER store raw draft on failure.
        "verification_status": "failed",
        "verifier_notes": "Verifier rejected the draft.",
        "evidence_packet_summary_json": {"abstract_only_count": 1},
        "claim_map_json": [],
        "model_provider": "emergent", "summary_model": "gpt-4o-mini",
        "verifier_model": "gpt-4o-mini", "prompt_version": PROMPT_VERSION,
        "verification_attempts": 2,
        "failed_reason_code": failed_reason,
        "displayed_to_user": False,
        "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    return sid


@pytest.mark.asyncio
async def test_verification_guidance_returns_safe_advice(client):
    t, _ = await signup(client, "phase4b-guidance@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid = await _create_failed_summary(client, h, failed_reason="unsupported_claims")
    r = await client.get(
        f"/api/search-sessions/{sid}/verification-guidance",
        headers=h,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["has_failure"] is True
    assert body["failed_reason_code"] == "unsupported_claims"
    assert isinstance(body["suggestions"], list)
    assert len(body["suggestions"]) >= 1
    # NEVER leak draft / raw text. The phrase "draft" appears in the safe
    # user_message itself ("we could not verify that all claims in the draft
    # summary..."); the suggestions and reason code must NOT include raw
    # extracted content or verifier raw output.
    raw = repr(body).lower()
    assert "section_text" not in raw
    assert "evidence_packet_summary_json" not in raw
    assert "claim_map_json" not in raw
    assert "answer_json" not in raw


@pytest.mark.asyncio
async def test_verification_guidance_no_failure_returns_empty(client):
    t, _ = await signup(client, "phase4b-guidance-clean@test.com")
    h = {"Authorization": f"Bearer {t}"}
    me = (await client.get("/api/auth/me", headers=h)).json()
    db = get_db()
    now = datetime.now(timezone.utc)
    sid = uuid.uuid4().hex
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": me["id"], "status": "awaiting_selection",
        "original_question": "q", "sanitized_question": "q", "pico_json": {},
        "deleted_at": None, "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    r = await client.get(
        f"/api/search-sessions/{sid}/verification-guidance",
        headers=h,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["has_failure"] is False
    assert body["suggestions"] == []


@pytest.mark.asyncio
async def test_get_summary_does_not_leak_failed_draft(client):
    """A failed summary must never be returned via GET /summary."""
    t, _ = await signup(client, "phase4b-no-leak@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid = await _create_failed_summary(client, h, failed_reason="unsupported_claims")
    r = await client.get(f"/api/search-sessions/{sid}/summary", headers=h)
    # Only verified summaries are returned. Failed -> 404.
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Admin dashboard Phase 4B blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_dashboard_includes_phase4b_blocks(client):
    from tests._phase3_helpers import login
    tok = await login(client, "admin@test.com", "AdminTest123!")
    h = {"Authorization": f"Bearer {tok}"}
    r = await client.get("/api/admin/dashboard", headers=h)
    assert r.status_code == 200
    body = r.json()
    # Phase 4B additions.
    assert "job_metrics" in body
    assert "malware_scan_metrics" in body
    assert "counts" in body["job_metrics"]
    msm = body["malware_scan_metrics"]
    assert "enabled" in msm
    assert "mode" in msm
    assert "uploads_allowed" in msm
    assert "scan_result_counts" in msm
