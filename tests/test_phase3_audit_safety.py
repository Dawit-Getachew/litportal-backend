"""Phase 3 — audit-safe payload tests (Task 11 / H).

The summary pipeline emits closed-set audit events with counts + IDs + reason
codes only. NEVER raw evidence text, NEVER raw prompts, NEVER PHI, NEVER
secrets.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from core.db import Collections, get_db
from tests._phase3_helpers import (  # noqa: E402
    make_packet_session, signup, good_apixaban_articles, patch_llm_configured,
)
from tests.test_phase3_summary_api import (  # noqa: E402
    _build_span_ids, _good_draft_for, _verifier_passed,
)


async def _summary_audit_events(sid: str) -> list[dict]:
    db = get_db()
    rows = await db[Collections.AUDIT_EVENTS].find(
        {"search_session_id": sid, "event_type": {"$regex": r"^summary_"}},
        {"_id": 0},
    ).to_list(length=200)
    return rows


def _dumps(events: list[dict]) -> str:
    return json.dumps(events, default=str)


@pytest.mark.asyncio
async def test_summary_audit_payload_no_raw_evidence_text(client):
    t, _ = await signup(client, "audraw1@test.com")
    h = {"Authorization": f"Bearer {t}"}
    arts = good_apixaban_articles()
    arts[0]["abstract"] = "SECRET_EVIDENCE_PHRASE_1234 about apixaban."
    sid, _, article_ids = await make_packet_session(client, h, articles=arts)
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[draft, _verifier_passed()])  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    events = await _summary_audit_events(sid)
    serialized = _dumps(events)
    assert "SECRET_EVIDENCE_PHRASE_1234" not in serialized
    assert "apixaban" not in serialized.lower(), "audit payload leaked evidence text"


@pytest.mark.asyncio
async def test_summary_audit_payload_no_raw_prompt(client):
    t, _ = await signup(client, "audraw2@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[draft, _verifier_passed()])  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    events = await _summary_audit_events(sid)
    serialized = _dumps(events).lower()
    for marker in ("you are an evidence", "you are a strict medical",
                   "clinical question:", "return json only", "evidence_span_id"):
        assert marker not in serialized, f"audit payload leaked prompt: {marker}"


@pytest.mark.asyncio
async def test_summary_audit_payload_no_phi(client):
    """If a question is PHI-blocked, no summary audit events ever fire."""
    t, _ = await signup(client, "audraw3@test.com")
    h = {"Authorization": f"Bearer {t}"}
    r = await client.post("/api/search-sessions/", headers=h, json={
        "question": "John Doe, DOB 03/14/1979, email j@x.com — best HTN tx?",
    })
    sid = r.json()["session_id"]
    events = await _summary_audit_events(sid)
    assert events == []


@pytest.mark.asyncio
async def test_summary_audit_payload_no_secrets(client):
    """No API key fragment ever appears in audit payloads."""
    t, _ = await signup(client, "audraw4@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[draft, _verifier_passed()])  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    events = await _summary_audit_events(sid)
    serialized = _dumps(events).lower()
    for marker in ("sk-emergent-", "sk-proj-", "sk-", "bearer ", "openai_api_key", "emergent_llm_key"):
        assert marker not in serialized, f"audit payload leaked secret marker: {marker}"
