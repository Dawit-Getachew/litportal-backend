"""Phase 3 — revision cycle tests (Task 11 / E)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.db import Collections, get_db
from tests._phase3_helpers import (  # noqa: E402
    make_packet_session, signup, good_apixaban_articles, patch_llm_configured,
)
from tests.test_phase3_summary_api import (  # noqa: E402
    _build_span_ids, _good_draft_for, _verifier_passed, _verifier_failed,
)


@pytest.mark.asyncio
async def test_first_failure_triggers_one_revision(client):
    t, _ = await signup(client, "rev1@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        # First attempt: drafter -> verifier_failed. Second: drafter -> verifier_passed.
        prov.complete_json = AsyncMock(side_effect=[  # type: ignore[assignment]
            draft, _verifier_failed(),
            draft, _verifier_passed(),
        ])
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "passed"
    assert body["verification_attempts"] == 2


@pytest.mark.asyncio
async def test_second_failure_withholds_answer(client):
    t, _ = await signup(client, "rev2@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[  # type: ignore[assignment]
            draft, _verifier_failed(),
            draft, _verifier_failed(),
        ])
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    body = r.json()
    assert body["status"] == "failed"
    assert "summary" not in body
    # Persisted as failed with reason code.
    db = get_db()
    stored = await db[Collections.SUMMARIES].find_one({"search_session_id": sid})
    assert stored["verification_status"] == "failed"
    assert stored["failed_reason_code"]
    assert stored["displayed_to_user"] is False


@pytest.mark.asyncio
async def test_passed_on_second_attempt_stores_verified_summary(client):
    t, _ = await signup(client, "rev3@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
    span_ids = await _build_span_ids(client, sid, h)
    draft = _good_draft_for(article_ids, span_ids)
    with patch_llm_configured() as prov:
        prov.complete_json = AsyncMock(side_effect=[  # type: ignore[assignment]
            draft, _verifier_failed(),
            draft, _verifier_passed(),
        ])
        r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    assert r.json()["status"] == "passed"
    db = get_db()
    stored = await db[Collections.SUMMARIES].find_one({"search_session_id": sid})
    assert stored["verification_status"] == "passed"
    assert stored["verification_attempts"] == 2


@pytest.mark.asyncio
async def test_revision_count_capped_by_env(client, monkeypatch):
    """MAX_SUMMARY_REVISIONS=0 → at most 1 attempt total."""
    monkeypatch.setenv("MAX_SUMMARY_REVISIONS", "0")
    # Re-import to pick up env (router constant is module-level).
    import importlib
    import routers.summary as sm
    importlib.reload(sm)
    # Re-mount the reloaded router so the FastAPI app routes resolve to the
    # updated module functions.
    from server import app
    for r in list(app.router.routes):
        if getattr(r, "path", "").endswith("/generate-summary?sync=true") or \
           getattr(r, "path", "").endswith("/regenerate-summary?sync=true") or \
           getattr(r, "path", "").endswith("/summary") or \
           getattr(r, "path", "").endswith("/summary-status") or \
           getattr(r, "path", "").endswith("/summary-readiness") or \
           getattr(r, "path", "").endswith("/llm-status"):
            try:
                app.router.routes.remove(r)
            except ValueError:
                pass
    app.include_router(sm.router, prefix="/api")
    try:
        t, _ = await signup(client, "rev4@test.com")
        h = {"Authorization": f"Bearer {t}"}
        sid, _, article_ids = await make_packet_session(client, h, articles=good_apixaban_articles())
        span_ids = await _build_span_ids(client, sid, h)
        draft = _good_draft_for(article_ids, span_ids)
        with patch_llm_configured() as prov:
            prov.complete_json = AsyncMock(side_effect=[draft, _verifier_failed()])  # type: ignore[assignment]
            r = await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
                "acknowledge_abstract_only_limitations": True,
            })
        body = r.json()
        assert body["status"] == "failed"
        assert body["verification_attempts"] == 1
    finally:
        # Restore env + module so subsequent tests use the default MAX_SUMMARY_REVISIONS.
        monkeypatch.undo()
        importlib.reload(sm)
        for r in list(app.router.routes):
            if getattr(r, "path", "").endswith("/generate-summary?sync=true") or \
               getattr(r, "path", "").endswith("/regenerate-summary?sync=true") or \
               getattr(r, "path", "").endswith("/summary") or \
               getattr(r, "path", "").endswith("/summary-status") or \
               getattr(r, "path", "").endswith("/summary-readiness") or \
               getattr(r, "path", "").endswith("/llm-status"):
                try:
                    app.router.routes.remove(r)
                except ValueError:
                    pass
        app.include_router(sm.router, prefix="/api")
