"""Phase 3 — generate-summary API endpoint tests (Task 11 / F)."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from core.db import Collections, get_db
from tests._phase3_helpers import (  # noqa: E402
    make_packet_session, signup, good_apixaban_articles, patch_llm_disabled,
    patch_llm_configured,
)


def _good_draft_for(article_ids, span_ids_by_article):
    """Return a canned valid DraftSummary dict referencing only provided spans."""
    a1 = article_ids[0]
    return {
        "clinical_bottom_line": "DOACs reduce stroke vs warfarin (low certainty).",
        "evidence_summary": "Three RCTs reported favorable safety/efficacy.",
        "applicability": "Adults with non-valvular AF.",
        "certainty": "low",
        "certainty_rationale": "Only abstract-level evidence available.",
        "selected_studies": [
            {
                "article_id": aid,
                "citation_label": f"PMID:{span_ids_by_article[aid][0]}",
                "pmid": None, "doi": None, "pmcid": None,
                "title": "t", "journal": "j", "publication_year": 2024,
                "study_type": "rct", "evidence_text_level": "abstract",
                "key_finding": "Favorable.", "limitations": "Industry-funded.",
                "supporting_evidence_span_ids": [span_ids_by_article[aid][0]],
            } for aid in article_ids
        ],
        "claim_map": [
            {
                "claim_id": "c1", "claim_text": "DOACs reduced stroke.",
                "claim_type": "effect",
                "supporting_evidence_span_ids": [span_ids_by_article[a1][0]],
                "supporting_article_ids": [a1],
            }
        ],
        "conflicting_evidence": "",
        "important_caveats": "Abstract-only.",
        "what_was_not_answered": "Long-term mortality.",
    }


def _verifier_passed():
    return {
        "verification_status": "passed", "final_answer_safe_to_show": True,
        "unsupported_claims": [], "incorrect_citations": [],
        "required_revisions": [], "verifier_notes": "ok",
        "verified_summary_json": None,
    }


def _verifier_failed():
    return {
        "verification_status": "failed", "final_answer_safe_to_show": False,
        "unsupported_claims": [{"claim_id": "c1", "claim_text": "x", "reason": "no support"}],
        "incorrect_citations": [],
        "required_revisions": ["Remove unsupported claim c1"],
        "verifier_notes": "rejected", "verified_summary_json": None,
    }


async def _build_span_ids(client, sid, headers):
    """Trigger packet prep so abstract extracts are guaranteed; then read the
    text_extracts rows to know what evidence_span_ids the summarizer will see.
    """
    await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=headers)
    db = get_db()
    rows = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
        {"search_session_id": sid, "include_in_summary": True}, {"_id": 0}
    ).to_list(length=100)
    span_ids: dict[str, list[str]] = {}
    for r in rows:
        sid_str = f"span:{r['id']}:0"
        span_ids.setdefault(r["article_id"], []).append(sid_str)
    return span_ids


@pytest.mark.asyncio
async def test_generate_summary_requires_auth(client):
    r = await client.post("/api/search-sessions/whatever/generate-summary?sync=true", json={})
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_user_cannot_summarize_other_users_session(client):
    t1, _ = await signup(client, "owner1@test.com")
    t2, _ = await signup(client, "other1@test.com")
    h1 = {"Authorization": f"Bearer {t1}"}
    h2 = {"Authorization": f"Bearer {t2}"}
    sid, _, _ = await make_packet_session(client, h1, articles=good_apixaban_articles())
    r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h2, json={})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_zero_selected_articles_rejected_400(client):
    t, _ = await signup(client, "zsel@test.com")
    h = {"Authorization": f"Bearer {t}"}
    arts = good_apixaban_articles()
    for a in arts: a["user_included"] = False
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    with patch_llm_configured():
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    assert r.status_code == 400
    body = r.json()
    assert body["detail"]["reason"] == "no_selected_articles"


@pytest.mark.asyncio
async def test_no_llm_provider_returns_503(client):
    t, _ = await signup(client, "nollm@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, _ = await make_packet_session(client, h, articles=good_apixaban_articles())
    with patch_llm_disabled():
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    assert r.status_code == 503
    body = r.json()
    assert body["reason"] == "llm_not_configured"
    assert "Configure an LLM provider" in body["message"]


@pytest.mark.asyncio
async def test_abstract_only_acknowledgement_required(client):
    t, _ = await signup(client, "absack@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, _ = await make_packet_session(client, h, articles=good_apixaban_articles())
    with patch_llm_configured():
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={})
    assert r.status_code == 400
    body = r.json()
    assert body["detail"]["detail"] == "abstract_only_acknowledgement_required"


@pytest.mark.asyncio
async def test_max_articles_exceeded_rejected(client):
    t, _ = await signup(client, "maxx@test.com")
    h = {"Authorization": f"Bearer {t}"}
    arts = [
        {**good_apixaban_articles()[0], "title": f"Study {i}", "pmid": f"100{i}",
         "doi": f"10.1234/x{i}", "abstract": f"abs {i}"}
        for i in range(11)
    ]
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    with patch_llm_configured():
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    assert r.status_code == 400
    body = r.json()
    assert body["detail"]["reason"] == "max_articles_exceeded"


@pytest.mark.asyncio
async def test_failed_verification_does_not_return_draft(client):
    t, _ = await signup(client, "failv@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    with patch_llm_configured() as prov:
        # Drafter returns a valid draft; verifier always fails.
        draft = _good_draft_for(article_ids, span_ids)
        prov.complete_json = AsyncMock(side_effect=[  # type: ignore[assignment]
            draft, _verifier_failed(),
            draft, _verifier_failed(),
        ])
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert body["user_message"]
    # The unverified draft text/claims must NEVER appear in the response.
    assert "summary" not in body
    assert "DOACs reduce stroke" not in str(body)


@pytest.mark.asyncio
async def test_passed_verification_stores_and_returns_summary(client):
    t, _ = await signup(client, "passv@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[draft, _verifier_passed()])  # type: ignore[assignment]
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "passed"
    assert body["summary"]["clinical_bottom_line"]
    assert body["summary_model"]
    # Persisted.
    db = get_db()
    stored = await db[Collections.SUMMARIES].find_one({"search_session_id": sid})
    assert stored is not None
    assert stored["verification_status"] == "passed"


@pytest.mark.asyncio
async def test_get_summary_returns_only_latest_verified(client):
    t, _ = await signup(client, "getv@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[draft, _verifier_passed()])  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    r = await client.get(f"/api/search-sessions/{sid}/summary", headers=h)
    assert r.status_code == 200
    assert r.json()["verification_status"] == "passed"


@pytest.mark.asyncio
async def test_summary_expires_with_session(client):
    """RetentionCleanup deletes summaries for expired sessions."""
    t, _ = await signup(client, "expir@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[draft, _verifier_passed()])  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    # Force-expire the session.
    db = get_db()
    past = datetime.now(timezone.utc) - timedelta(days=1)
    await db[Collections.SEARCH_SESSIONS].update_one({"id": sid}, {"$set": {"expires_at": past}})
    from services.retention_cleanup import retention_cleanup
    result = await retention_cleanup.cleanup_expired_sessions()
    assert result["deleted_count"] >= 1
    assert result.get("deleted_summaries", 0) >= 1


@pytest.mark.asyncio
async def test_regenerate_summary_creates_new_row(client):
    t, _ = await signup(client, "regen@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[  # type: ignore[assignment]
            draft, _verifier_passed(),
            draft, _verifier_passed(),
        ])
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
        await client.post(f"/api/search-sessions/{sid}/regenerate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    db = get_db()
    count = await db[Collections.SUMMARIES].count_documents({"search_session_id": sid})
    assert count == 2


@pytest.mark.asyncio
async def test_summary_status_endpoint_returns_correct_states(client):
    t, _ = await signup(client, "stat@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())

    r = await client.get(f"/api/search-sessions/{sid}/summary-status", headers=h)
    assert r.status_code == 200
    assert r.json()["status"] == "none"

    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[draft, _verifier_passed()])  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    r = await client.get(f"/api/search-sessions/{sid}/summary-status", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "passed"
    assert body["attempts"] >= 1


@pytest.mark.asyncio
async def test_llm_status_and_readiness_endpoints(client):
    t, _ = await signup(client, "stat2@test.com")
    h = {"Authorization": f"Bearer {t}"}
    # Disabled state.
    with patch_llm_disabled():
        r = await client.get("/api/llm/summary-readiness", headers=h)
        assert r.status_code == 200
        assert r.json()["summarization_enabled"] is False
    # Configured state.
    with patch_llm_configured():
        r = await client.get("/api/llm/summary-readiness", headers=h)
        assert r.status_code == 200
        assert r.json()["summarization_enabled"] is True
    # Admin LLM status — requires admin.
    r = await client.get("/api/admin/llm-status", headers=h)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_llm_status_returns_no_keys(client):
    from tests._phase3_helpers import login
    tok = await login(client, "admin@test.com", "AdminTest123!")
    h = {"Authorization": f"Bearer {tok}"}
    r = await client.get("/api/admin/llm-status", headers=h)
    assert r.status_code == 200
    body = r.json()
    for k in ("openai_api_key", "OPENAI_API_KEY", "emergent_llm_key", "EMERGENT_LLM_KEY", "key", "api_key"):
        assert k not in body
