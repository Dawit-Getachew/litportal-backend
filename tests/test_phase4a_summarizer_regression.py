"""Phase 4A — summarizer + verifier regression with OA full text (Task 11 / I)."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from core.db import Collections, get_db
from tests._phase3_helpers import signup, patch_llm_configured
from tests.test_phase3_summary_api import (
    _build_span_ids, _good_draft_for, _verifier_passed,
)


def _now():
    return datetime.now(timezone.utc)


async def _seed_session_with_oa(*, user_id, n=2):
    """Create a session whose selected articles have OA-source extracts."""
    db = get_db()
    sid = uuid.uuid4().hex
    now = _now()
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": user_id, "status": "awaiting_selection",
        "original_question": "q", "sanitized_question": "q", "pico_json": {},
        "deleted_at": None, "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    article_ids = []
    for i in range(n):
        aid = uuid.uuid4().hex
        article_ids.append(aid)
        await db[Collections.ARTICLES].insert_one({
            "id": aid, "canonical_key": f"id:{aid}", "title": f"OA study {i}",
            "abstract": "abs", "doi": f"10.x/{i}", "pmid": f"200{i}",
            "pmcid": f"PMC100{i}", "journal": "NEJM", "publication_year": 2024,
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
            "overall_rank": 99 - i, "rank_explanation": "test",
            "query_types_matched": [], "sources_present": ["test"], "flags": [],
            "created_at": now, "updated_at": now,
        })
        # Pre-seed an oa_full_text extract row in this session.
        await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one({
            "id": uuid.uuid4().hex, "article_id": aid,
            "search_session_id": sid, "text_source": "oa_full_text",
            "section_name": "results",
            "section_text": ("Results section text long enough to survive the "
                             "minimum-length filter and exercise the summarizer "
                             "input path. " * 3),
            "token_count": 40, "include_in_summary": True,
            "evidence_text_level": "oa_full_text", "created_at": now,
        })
    return sid, article_ids


@pytest.mark.asyncio
async def test_summarizer_receives_oa_extracts_when_selected_and_eligible(client, monkeypatch):
    monkeypatch.setenv("OA_FETCHER_ENABLED", "false")  # avoid auto-fetch
    t, _ = await signup(client, "reg-oa1@test.com")
    me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {t}"})
    uid = me.json()["id"]
    sid, article_ids = await _seed_session_with_oa(user_id=uid)
    h = {"Authorization": f"Bearer {t}"}
    with patch_llm_configured() as prov:
        mock = AsyncMock(return_value=None)
        prov.complete_json = mock  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true",
                          headers=h, json={"acknowledge_abstract_only_limitations": True})
    # Summarizer was called with OA extracts visible in the prompt.
    assert mock.call_args_list, "summarizer LLM never called"
    all_user = " ".join(c.kwargs.get("user", "") for c in mock.call_args_list)
    assert "oa_full_text" in all_user
    assert "Results section text" in all_user


@pytest.mark.asyncio
async def test_certainty_can_upgrade_to_moderate_when_oa_full_text_dominant(client, monkeypatch):
    """Mechanical floor does NOT force a downgrade when oa_full_text spans
    dominate. (The floor only downgrades when EVERY span is abstract-level.)"""
    monkeypatch.setenv("OA_FETCHER_ENABLED", "false")
    t, _ = await signup(client, "reg-oa2@test.com")
    me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {t}"})
    uid = me.json()["id"]
    sid, article_ids = await _seed_session_with_oa(user_id=uid)
    h = {"Authorization": f"Bearer {t}"}
    # Build a "moderate certainty" draft that cites the OA spans.
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    draft["certainty"] = "moderate"
    draft["certainty_rationale"] = "Most evidence is OA full text."
    # Update study entries to label the level correctly.
    for s in draft["selected_studies"]:
        s["evidence_text_level"] = "user_uploaded_full_text" if False else "oa_full_text"
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[draft, _verifier_passed()])  # type: ignore[assignment]
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true",
                              headers=h, json={"acknowledge_abstract_only_limitations": True})
    body = r.json()
    assert body["status"] == "passed", body
    assert body["summary"]["certainty"] == "moderate"


@pytest.mark.asyncio
async def test_verifier_still_rejects_unsupported_claims_with_oa_text(client, monkeypatch):
    """Even with OA full text spans, the verifier must reject unsupported claims."""
    monkeypatch.setenv("OA_FETCHER_ENABLED", "false")
    t, _ = await signup(client, "reg-oa3@test.com")
    me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {t}"})
    uid = me.json()["id"]
    sid, article_ids = await _seed_session_with_oa(user_id=uid)
    h = {"Authorization": f"Bearer {t}"}
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    # Inject a fabricated PMID on one selected_studies entry.
    draft["selected_studies"][0]["pmid"] = "9999999"  # different from article record
    # Both attempts get the same broken draft + verifier-fail.
    fail = {"verification_status": "failed", "final_answer_safe_to_show": False,
            "unsupported_claims": [], "incorrect_citations": [],
            "required_revisions": ["bad pmid"], "verifier_notes": "rejected",
            "verified_summary_json": None}
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[draft, fail, draft, fail])  # type: ignore[assignment]
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true",
                              headers=h, json={"acknowledge_abstract_only_limitations": True})
    body = r.json()
    assert body["status"] == "failed"
    assert "summary" not in body  # unverified draft never returned
