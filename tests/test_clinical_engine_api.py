"""W1.4 over HTTP: the human-in-the-loop gate on /api/v2/engine.

Auth + ownership, feature flag, PHI, versioning, conflicts, idempotent confirm,
and "no evidence search before approval" (no search session exists until the
clinician confirms)."""
from __future__ import annotations

import asyncio
import uuid

import pytest

from clinical_engine import store
from clinical_engine.config import settings
from clinical_engine.orchestrator import runner
from core.db import Collections, get_db

pytestmark = pytest.mark.xdist_group("clinical_engine")
BASE = "/api/v2/engine"
PASSWORD = "EngineApi9!Password"


@pytest.fixture(autouse=True)
async def engine_on(client, monkeypatch):
    monkeypatch.setattr(settings, "ENABLED", True)
    monkeypatch.setattr(settings, "RETRIEVAL_ADAPTER", "mock")
    monkeypatch.setattr(settings, "POLL_INTERVAL_S", 0.05)
    monkeypatch.setenv("LLM_PROVIDER", "none")
    await store.ensure_indexes()
    yield
    await runner.shutdown()


async def _signup(client) -> tuple[dict, str]:
    email = f"engine-{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post("/api/auth/signup", json={"email": email, "password": PASSWORD, "name": "Engine Tester"})
    assert r.status_code == 201, r.text
    body = r.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]["id"]


@pytest.fixture
async def auth(client):
    headers, _ = await _signup(client)
    return headers


async def _search_sessions(user_id: str) -> int:
    return await get_db()[Collections.SEARCH_SESSIONS].count_documents({"user_id": user_id})


async def test_requires_auth(client):
    assert (await client.post(f"{BASE}/questions", json={"text": "x"})).status_code == 401


async def test_disabled_engine_hides_routes_but_reports_capabilities(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "ENABLED", False)
    assert (await client.post(f"{BASE}/questions", json={"text": "q"}, headers=auth)).status_code == 404
    caps = (await client.get(f"{BASE}/capabilities")).json()
    assert caps["enabled"] is False
    monkeypatch.setattr(settings, "ENABLED", True)
    caps = (await client.get(f"{BASE}/capabilities")).json()
    assert caps["enabled"] is True and caps["modes"] == ["review_papers", "answer_for_me"]
    # Also available on the LitPortal module mount.
    assert (await client.get("/api/litportal/v2/engine/capabilities")).status_code == 200


async def test_full_human_in_the_loop_flow(client):
    headers, user_id = await _signup(client)
    before = await _search_sessions(user_id)

    # 1. Ask → interpretation, nothing searched yet.
    r = await client.post(f"{BASE}/questions", headers=headers,
                          json={"text": "Impact of corticosteroids on mortality in septic shock"})
    assert r.status_code == 201, r.text
    view = r.json()
    qid = view["question_id"]
    assert view["status"] == "awaiting_confirmation" and view["version"] == 1
    interp = view["interpretation"]
    assert interp["question_type"] == "treatment" and interp["age_group"] == "not_stated"
    assert "adult" not in interp["summary"].lower()
    assert any(c["id"] == "add:age_group" and c["state"] == "placeholder" for c in interp["chips"])
    assert set(view["allowed_actions"]) >= {"confirm_review_papers", "confirm_answer_for_me", "revise", "decline"}
    assert await _search_sessions(user_id) == before  # no evidence search before approval

    # 2. Modify: add an age group chip → new version, material change.
    r = await client.post(f"{BASE}/questions/{qid}/revise", headers=headers,
                          json={"base_version": 1, "edits": [{"op": "add", "role": "age_group", "value": "adult"}]})
    assert r.status_code == 200, r.text
    v2 = r.json()
    assert v2["version"] == 2 and v2["interpretation"]["age_group"] == "adult"
    assert [v["status"] for v in v2["versions"]] == ["superseded", "current"]

    # A no-op edit does not create a version.
    adult_chip = next(c for c in v2["interpretation"]["chips"] if c["role"] == "age_group")
    r = await client.post(f"{BASE}/questions/{qid}/revise", headers=headers,
                          json={"base_version": 2, "edits": [{"op": "set", "element_id": adult_chip["id"],
                                                              "value": "adult"}]})
    assert r.json()["version"] == 2 and r.json().get("unchanged") is True

    # Stale base version is rejected.
    r = await client.post(f"{BASE}/questions/{qid}/revise", headers=headers,
                          json={"base_version": 1, "text": "Hydrocortisone for septic shock"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "stale_version"

    # Confirming an old version is rejected.
    r = await client.post(f"{BASE}/questions/{qid}/confirm", headers=headers,
                          json={"version": 1, "mode": "review_papers"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "stale_version"
    assert await _search_sessions(user_id) == before

    # 3. Accept → approval recorded, run started with a versioned plan.
    r = await client.post(f"{BASE}/questions/{qid}/confirm", headers=headers,
                          json={"version": 2, "mode": "review_papers"})
    assert r.status_code == 200, r.text
    confirmed = r.json()
    assert confirmed["status"] == "confirmed" and confirmed["approval"]["version"] == 2
    assert confirmed["delegation"] is None  # review-papers records no delegation
    run_id = confirmed["run"]["id"]
    # Double-click confirm is idempotent.
    again = await client.post(f"{BASE}/questions/{qid}/confirm", headers=headers,
                              json={"version": 2, "mode": "review_papers"})
    assert again.status_code == 200 and again.json()["run"]["id"] == run_id
    # Other option on the same version is refused.
    other = await client.post(f"{BASE}/questions/{qid}/confirm", headers=headers,
                              json={"version": 2, "mode": "answer_for_me"})
    assert other.status_code == 409

    for _ in range(100):
        run = (await client.get(f"{BASE}/runs/{run_id}", headers=headers)).json()
        if run["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.05)
    assert run["status"] == "completed" and run["evidence_label"] == "fixture"
    assert run["search_plan"]["question_version"] == 2
    assert run["search_plan"]["constraints"]["age_group"] == "adult"

    # 4. Trace is safe: no question text anywhere in it.
    trace = (await client.get(f"{BASE}/questions/{qid}/trace", headers=headers)).json()["events"]
    names = [e["event"] for e in trace]
    for expected in ("question_created", "version_created", "approval_recorded", "run_created", "run_completed"):
        assert expected in names
    assert "corticosteroids" not in str(trace).lower()

    # 5. A later material edit invalidates that run's results.
    r = await client.post(f"{BASE}/questions/{qid}/revise", headers=headers,
                          json={"base_version": 2, "text": "Does hydrocortisone reduce mortality in septic shock?"})
    assert r.status_code == 200 and r.json()["version"] == 3 and r.json()["run"] is None
    old_run = (await client.get(f"{BASE}/runs/{run_id}", headers=headers)).json()
    assert old_run["invalidated_by_version"] == 3


async def test_decline_and_ownership(client):
    alice, _ = await _signup(client)
    bob, _ = await _signup(client)
    qid = (await client.post(f"{BASE}/questions", headers=alice,
                             json={"text": "Does apixaban reduce stroke in atrial fibrillation?"})).json()["question_id"]
    # Bob cannot see, change, confirm or decline Alice's question.
    assert (await client.get(f"{BASE}/questions/{qid}", headers=bob)).status_code == 404
    assert (await client.post(f"{BASE}/questions/{qid}/confirm", headers=bob,
                              json={"version": 1, "mode": "review_papers"})).status_code == 404
    assert (await client.post(f"{BASE}/questions/{qid}/decline", headers=bob, json={})).status_code == 404
    listed = (await client.get(f"{BASE}/questions", headers=bob)).json()["questions"]
    assert all(q["question_id"] != qid for q in listed)

    r = await client.post(f"{BASE}/questions/{qid}/decline", headers=alice, json={"reason": "wrong_interpretation"})
    assert r.status_code == 200 and r.json()["status"] == "declined" and r.json()["allowed_actions"] == []
    assert (await client.post(f"{BASE}/questions/{qid}/confirm", headers=alice,
                              json={"version": 1, "mode": "review_papers"})).status_code == 409
    assert (await client.post(f"{BASE}/questions/{qid}/revise", headers=alice,
                              json={"base_version": 1, "text": "x y z"})).status_code == 409


async def test_blocking_conflict_must_be_fixed_before_confirm(client, auth):
    qid_view = (await client.post(f"{BASE}/questions", headers=auth,
                                  json={"text": "Management of septic shock"})).json()
    qid = qid_view["question_id"]
    # Make it contradictory: the condition as the outcome of a treatment question.
    r = await client.post(f"{BASE}/questions/{qid}/revise", headers=auth, json={"base_version": 1, "edits": [
        {"op": "set_type", "value": "treatment"},
        {"op": "add", "role": "intervention", "value": "hydrocortisone"},
        {"op": "add", "role": "outcome", "value": "septic shock"},
    ]})
    view = r.json()
    assert view["status"] == "needs_clarification" and view["can_confirm"] is False
    assert "condition_as_outcome" in {c["code"] for c in view["interpretation"]["conflicts"]}
    r = await client.post(f"{BASE}/questions/{qid}/confirm", headers=auth, json={"version": 2, "mode": "review_papers"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "blocking_conflicts"
    # Fix it → confirmable again.
    outcome = next(c for c in view["interpretation"]["chips"] if c["role"] == "outcome")
    r = await client.post(f"{BASE}/questions/{qid}/revise", headers=auth, json={"base_version": 2, "edits": [
        {"op": "set", "element_id": outcome["id"], "value": "mortality"}]})
    assert r.json()["can_confirm"] is True and r.json()["status"] == "awaiting_confirmation"


async def test_phi_is_blocked_and_not_stored(client):
    headers, user_id = await _signup(client)
    r = await client.post(f"{BASE}/questions", headers=headers,
                          json={"text": "Should Mr. John Smith, MRN 12345678, get steroids?"})
    assert r.status_code == 201
    view = r.json()
    assert view["status"] == "blocked_phi" and "question_text" not in view
    stored = await get_db()[store.QUESTIONS].find_one({"id": view["question_id"]}, {"_id": 0})
    assert "John" not in str(stored)
    assert await get_db()[store.VERSIONS].count_documents({"question_id": view["question_id"]}) == 0
    # PHI typed into a chip is refused too.
    qid = (await client.post(f"{BASE}/questions", headers=headers,
                             json={"text": "Steroids in septic shock"})).json()["question_id"]
    r = await client.post(f"{BASE}/questions/{qid}/revise", headers=headers, json={
        "base_version": 1, "edits": [{"op": "add", "role": "population", "value": "patient Jane Doe"}]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "phi_detected"


async def test_answer_for_me_records_delegation_and_evidence_set(client, auth):
    qid = (await client.post(f"{BASE}/questions", headers=auth,
                             json={"text": "Does hydrocortisone reduce mortality in septic shock?"})).json()["question_id"]
    view = (await client.post(f"{BASE}/questions/{qid}/confirm", headers=auth,
                              json={"version": 1, "mode": "answer_for_me"})).json()
    assert view["approval"]["mode"] == "answer_for_me" and view["delegation"]["policy"]
    run_id = view["run"]["id"]
    for _ in range(100):
        run = (await client.get(f"{BASE}/runs/{run_id}", headers=auth)).json()
        if run["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.05)
    assert run["status"] == "completed"
    es = (await client.get(f"{BASE}/evidence-sets/{run['links']['evidence_set_id']}", headers=auth)).json()
    assert es["selection_mode"] == "delegated" and es["items"]
    assert all(i["selected_by"] == "policy" for i in es["items"])


async def test_input_validation(client, auth):
    assert (await client.post(f"{BASE}/questions", headers=auth, json={"text": ""})).status_code == 422
    r = await client.post(f"{BASE}/questions", headers=auth, json={"text": "a" * 1500})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "question_too_long"
    qid = (await client.post(f"{BASE}/questions", headers=auth, json={"text": "Steroids in ARDS"})).json()["question_id"]
    r = await client.post(f"{BASE}/questions/{qid}/revise", headers=auth, json={"base_version": 1})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "revise_needs_text_or_edits"
    r = await client.post(f"{BASE}/questions/{qid}/confirm", headers=auth, json={"version": 1, "mode": "nope"})
    assert r.status_code == 422
    assert (await client.get(f"{BASE}/runs/does-not-exist", headers=auth)).status_code == 404
