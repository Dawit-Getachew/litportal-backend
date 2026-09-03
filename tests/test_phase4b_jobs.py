"""Phase 4B — durable job + async summary tests."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from core.db import Collections, get_db
from services.jobs import JobService, job_service
from tests._phase3_helpers import (
    good_apixaban_articles, login, make_packet_session, patch_llm_configured, signup,
)


# ---------------------------------------------------------------------------
# JobService unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_job_create_and_get(client):
    svc = JobService()
    user_id = uuid.uuid4().hex
    job = await svc.create(user_id=user_id, job_type="summary_generation",
                            search_session_id="s1", total_steps=4)
    assert job["status"] == "queued"
    fetched = await svc.get(job_id=job["id"], user_id=user_id)
    assert fetched["id"] == job["id"]
    # User isolation
    other = await svc.get(job_id=job["id"], user_id="other")
    assert other is None


@pytest.mark.asyncio
async def test_job_lifecycle_state_transitions(client):
    svc = JobService()
    user_id = uuid.uuid4().hex
    job = await svc.create(user_id=user_id, job_type="oa_fetch")
    await svc.mark_running(job_id=job["id"], stage="fetching", percent=20)
    cur = await svc.get(job_id=job["id"], user_id=user_id)
    assert cur["status"] == "running"
    assert cur["progress_percent"] == 20
    await svc.update_progress(job_id=job["id"], stage="parsing", current_step=2, percent=50)
    cur = await svc.get(job_id=job["id"], user_id=user_id)
    assert cur["progress_stage"] == "parsing"
    await svc.succeed(job_id=job["id"], result_ref={"summary_id": "x", "article_ids": ["a"]})
    cur = await svc.get(job_id=job["id"], user_id=user_id)
    assert cur["status"] == "succeeded"
    assert cur["result_ref_json"]["summary_id"] == "x"


@pytest.mark.asyncio
async def test_job_fail_persists_safe_error_only(client):
    svc = JobService()
    user_id = uuid.uuid4().hex
    job = await svc.create(user_id=user_id, job_type="summary_generation")
    await svc.fail(
        job_id=job["id"], error_code="verification_failed",
        error_message_safe="A safe failure summary.",
        result_ref={"raw_draft_text": "LEAKED CONTENT", "failed_reason_code": "x"},
    )
    cur = await svc.get(job_id=job["id"], user_id=user_id)
    assert cur["status"] == "failed"
    assert cur["error_code"] == "verification_failed"
    # The sanitizer drops any key not on the allow-list.
    assert "raw_draft_text" not in cur["result_ref_json"]
    assert cur["result_ref_json"].get("failed_reason_code") == "x"


@pytest.mark.asyncio
async def test_job_result_ref_sanitizer_drops_unknown_keys(client):
    svc = JobService()
    user_id = uuid.uuid4().hex
    job = await svc.create(user_id=user_id, job_type="summary_generation")
    bad_payload = {
        "summary_id": "ok",
        "raw_text": "PHI: Jane Doe MRN 12345",  # NOT allowed
        "prompt": "system prompt leak",         # NOT allowed
        "doi": "10.x/y",                         # NOT allowed
    }
    await svc.succeed(job_id=job["id"], result_ref=bad_payload)
    cur = await svc.get(job_id=job["id"], user_id=user_id)
    rr = cur["result_ref_json"]
    assert rr == {"summary_id": "ok"}


@pytest.mark.asyncio
async def test_job_cancel_is_idempotent(client):
    svc = JobService()
    user_id = uuid.uuid4().hex
    job = await svc.create(user_id=user_id, job_type="summary_generation")
    a = await svc.cancel(job_id=job["id"], user_id=user_id)
    assert a["status"] == "canceled"
    b = await svc.cancel(job_id=job["id"], user_id=user_id)
    assert b["status"] == "canceled"


@pytest.mark.asyncio
async def test_sweep_stale_jobs(client):
    svc = JobService()
    user_id = uuid.uuid4().hex
    job = await svc.create(user_id=user_id, job_type="summary_generation")
    # Force updated_at far in the past.
    db = get_db()
    await db[Collections.JOBS].update_one(
        {"id": job["id"]},
        {"$set": {
            "status": "running",
            "updated_at": datetime.now(timezone.utc) - timedelta(hours=2),
        }},
    )
    swept = await svc.sweep_stale_jobs()
    assert swept >= 1
    cur = await svc.get(job_id=job["id"], user_id=user_id)
    assert cur["status"] == "failed"
    assert cur["error_code"] == "stale_job"


# ---------------------------------------------------------------------------
# Jobs router
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_job_404_for_non_owner(client):
    t1, _ = await signup(client, "phase4b-jobs-own@test.com")
    t2, _ = await signup(client, "phase4b-jobs-other@test.com")
    h1 = {"Authorization": f"Bearer {t1}"}
    h2 = {"Authorization": f"Bearer {t2}"}
    # Create a job for user 1
    me = (await client.get("/api/auth/me", headers=h1)).json()
    job = await job_service.create(user_id=me["id"], job_type="summary_generation")
    # User 2 cannot read it.
    r = await client.get(f"/api/jobs/{job['id']}", headers=h2)
    assert r.status_code == 404
    # User 1 can.
    r = await client.get(f"/api/jobs/{job['id']}", headers=h1)
    assert r.status_code == 200
    assert r.json()["id"] == job["id"]


@pytest.mark.asyncio
async def test_admin_job_metrics_no_raw_payloads(client):
    tok = await login(client, "admin@test.com", "AdminTest123!")
    h = {"Authorization": f"Bearer {tok}"}
    r = await client.get("/api/admin/job-metrics", headers=h)
    assert r.status_code == 200
    body = r.json()
    # Check expected keys.
    for k in ("counts", "by_job_type", "summary_timeouts", "stale_jobs", "runner_mode"):
        assert k in body
    # Never leak raw text.
    raw = repr(body).lower()
    assert "raw_draft" not in raw
    assert "phi" not in raw or "phi_blocked" in raw  # phi_blocked counter ok


@pytest.mark.asyncio
async def test_clinician_cannot_read_admin_job_metrics(client):
    t, _ = await signup(client, "phase4b-jobs-noadmin@test.com")
    h = {"Authorization": f"Bearer {t}"}
    r = await client.get("/api/admin/job-metrics", headers=h)
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Async summary endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_summary_returns_job_id_immediately(client):
    """Default (no ?sync) returns job_id and 200 immediately."""
    t, _ = await signup(client, "phase4b-summary-async@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _uid, _aids = await make_packet_session(
        client, h, articles=good_apixaban_articles()[:1],
    )

    with patch_llm_configured() as llm:
        # Mock the LLM call so the runner finishes deterministically.
        async def _slow_then_ok(*a, **kw):
            await asyncio.sleep(0.05)
            return None  # No JSON; will cause failure

        llm.complete_json = AsyncMock(side_effect=_slow_then_ok)
        r = await client.post(
            f"/api/search-sessions/{sid}/generate-summary",
            headers=h, json={
                "selected_article_ids": _aids,
                "acknowledge_abstract_only_limitations": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running", "failed", "succeeded")
        job_id = body["job_id"]

        # Poll the job until terminal (INSIDE the patch context so the mock
        # stays installed for the background runner).
        for _ in range(60):
            rr = await client.get(f"/api/jobs/{job_id}", headers=h)
            assert rr.status_code == 200
            s = rr.json()["status"]
            if s in ("succeeded", "failed", "canceled", "expired"):
                break
            await asyncio.sleep(0.1)
        final = (await client.get(f"/api/jobs/{job_id}", headers=h)).json()
    assert final["status"] in ("succeeded", "failed")
    # Failure must NEVER carry raw evidence text — only safe codes.
    if final["status"] == "failed":
        rr = final["result_ref_json"]
        assert "raw_draft" not in rr
        assert "section_text" not in rr


@pytest.mark.asyncio
async def test_sync_mode_still_works(client):
    """Backward compatibility: ?sync=true returns the synchronous response."""
    t, _ = await signup(client, "phase4b-summary-sync@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _uid, _aids = await make_packet_session(
        client, h, articles=good_apixaban_articles()[:1],
    )
    with patch_llm_configured() as llm:
        llm.complete_json = AsyncMock(return_value=None)
        r = await client.post(
            f"/api/search-sessions/{sid}/generate-summary?sync=true",
            headers=h, json={
                "selected_article_ids": _aids,
                "acknowledge_abstract_only_limitations": True,
            },
        )
    assert r.status_code == 200
    body = r.json()
    # In sync mode we get either a passed or failed result directly.
    assert "job_id" not in body
    assert body["status"] in ("passed", "failed")


@pytest.mark.asyncio
async def test_async_summary_dedupes_concurrent_jobs(client):
    t, _ = await signup(client, "phase4b-summary-dedup@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, _uid, _aids = await make_packet_session(
        client, h, articles=good_apixaban_articles()[:1],
    )

    # Make the runner slow so two concurrent posts see the first job in-flight.
    slow_event = asyncio.Event()

    async def _slow(*a, **kw):
        await slow_event.wait()
        return None

    with patch_llm_configured() as llm:
        llm.complete_json = AsyncMock(side_effect=_slow)
        body = {
            "selected_article_ids": _aids,
            "acknowledge_abstract_only_limitations": True,
        }
        r1 = await client.post(
            f"/api/search-sessions/{sid}/generate-summary",
            headers=h, json=body,
        )
        r2 = await client.post(
            f"/api/search-sessions/{sid}/generate-summary",
            headers=h, json=body,
        )
        slow_event.set()
    assert r1.status_code == 200 and r2.status_code == 200
    j1 = r1.json()
    j2 = r2.json()
    assert j1["job_id"] == j2["job_id"]
    assert j2.get("duplicate_suppressed") is True
    # Drain the running task so the next test starts clean.
    from services.jobs import job_service as _js
    me = (await client.get("/api/auth/me", headers=h)).json()
    await _js.cancel(job_id=j1["job_id"], user_id=me["id"])
