"""W1.2 — durable orchestration: pause/resume, cancel, stale-result rejection,
restart/resume without duplicate writes, blocked unapproved calls, budgets and
bounded retries. Needs the local test Mongo (see tests/conftest.py)."""
from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

import pytest

from clinical_engine import store
from clinical_engine.config import settings
from clinical_engine.orchestrator import runner
from clinical_engine.orchestrator import tasks as task_mod
from clinical_engine import questions

pytestmark = pytest.mark.xdist_group("clinical_engine")

Q = "Does hydrocortisone reduce 28-day mortality in adults with septic shock?"


@pytest.fixture(autouse=True)
async def engine_env(client, monkeypatch):
    monkeypatch.setattr(settings, "ENABLED", True)
    monkeypatch.setattr(settings, "POLL_INTERVAL_S", 0.05)
    monkeypatch.setattr(settings, "LEASE_SECONDS", 2)
    monkeypatch.setattr(settings, "RETRY_BACKOFF_BASE_S", 0.01)
    monkeypatch.setattr(settings, "RETRIEVAL_ADAPTER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    await store.ensure_indexes()
    yield
    await runner.shutdown()


def user(uid: str = "engine-test-user") -> dict:
    return {"id": uid}


async def wait_run(run_id: str, statuses: set[str], timeout: float = 10.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        run = await store.get_run(run_id)
        if run and run["status"] in statuses:
            return run
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"run {run_id} stuck in {run and run['status']}")
        await asyncio.sleep(0.05)


async def events(run_id: str) -> list[str]:
    return [t["event"] for t in await store.traces(run_id=run_id)]


async def confirmed(mode: str = "review_papers", uid: str = "engine-test-user") -> tuple[dict, dict]:
    view = await questions.create_question(user(uid), Q)
    view = await questions.confirm_question(user(uid), view["question_id"], version=view["version"], mode=mode)
    return view, await store.get_run(view["run"]["id"])


async def test_mock_run_completes_with_plan_and_identity_rejection():
    view, run = await confirmed()
    done = await wait_run(run["id"], {"completed", "failed"})
    assert done["status"] == "completed" and done["result_kind"] == "fixture_demo"
    full = await questions.get_run(user(), run["id"])
    assert [t["key"] for t in full["tasks"]] == ["plan_search", "retrieve", "integrity_check"]
    plan = full["search_plan"]
    assert [lane["lane"] for lane in plan["lanes"]] == ["aggregated", "foundational_primary", "recent_primary"]
    assert plan["constraints"]["age_group"] == "adult"
    integrity = full["integrity"]
    assert integrity["evidence_label"] == "fixture"
    assert sorted(integrity["quarantined_record_ids"]) == ["fx-3", "fx-4"]  # borrowed-PMID conflict
    assert integrity["studies"][0]["registration_ids"] == ["NCT09000001"]  # 2 reports, 1 study
    assert done["usage"]["tool_calls"] == 1


async def test_no_evidence_tool_runs_without_approval():
    """Defence in depth: a run for an unapproved version is blocked at the tool gateway."""
    created = await questions.create_question(user(), Q)
    version = await store.get_version(created["question_id"], 1)
    assert version["approval"] is None
    run = await runner.create_run(version, mode="review_papers", user_id="engine-test-user")
    runner.spawn(run["id"])
    done = await wait_run(run["id"], {"completed", "failed"})
    assert done["status"] == "failed"
    assert done["error"]["detail"]["code"] == "tool_blocked:question_not_approved"
    assert done["usage"]["tool_calls"] == 0 and done["usage"]["blocked_calls"] == 1
    assert "tool_blocked" in await events(run["id"])


async def test_tool_not_on_allowlist_is_blocked(monkeypatch):
    real = task_mod.HANDLERS["plan_search"]

    async def sneaky(ctx):
        await ctx.tools.call("mock.evidence_search", task_mod._mock_search, ctx.cqm)  # not on its allowlist
        return await real(ctx)

    monkeypatch.setitem(task_mod.HANDLERS, "plan_search", sneaky)
    _, run = await confirmed()
    done = await wait_run(run["id"], {"completed", "failed"})
    assert done["error"]["detail"]["code"] == "tool_blocked:not_on_task_allowlist"


async def test_call_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(settings, "RUN_MAX_TOOL_CALLS", 0)
    _, run = await confirmed()
    done = await wait_run(run["id"], {"completed", "failed"})
    assert done["error"]["detail"]["code"] == "budget_exceeded:tool_calls"
    assert done["usage"]["tool_calls"] == 0


async def test_retries_are_bounded(monkeypatch):
    calls = {"n": 0}

    async def flaky(ctx):
        calls["n"] += 1
        raise RuntimeError("upstream down")

    monkeypatch.setitem(task_mod.HANDLERS, "plan_search", flaky)
    _, run = await confirmed()
    done = await wait_run(run["id"], {"completed", "failed"})
    assert calls["n"] == settings.TASK_MAX_ATTEMPTS == 3
    assert done["error"]["detail"]["code"] == "retries_exhausted:RuntimeError"
    assert (await events(run["id"])).count("task_retry_scheduled") == 2


async def test_transient_failure_recovers_within_budget(monkeypatch):
    real = task_mod.HANDLERS["plan_search"]
    calls = {"n": 0}

    async def once_flaky(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise task_mod.RetryableTaskError("temporary")
        return await real(ctx)

    monkeypatch.setitem(task_mod.HANDLERS, "plan_search", once_flaky)
    _, run = await confirmed()
    done = await wait_run(run["id"], {"completed", "failed"})
    assert done["status"] == "completed" and calls["n"] == 2


# ── legacy adapter: pause/resume, cancel, restart/resume ────────────────────


@pytest.fixture
def fake_legacy(monkeypatch):
    """Stub the legacy pipeline: count session creations; the session stays
    'searching' until the test flips it."""
    from core.db import Collections, get_db

    monkeypatch.setattr(settings, "RETRIEVAL_ADAPTER", "legacy_pipeline")
    created: list[str] = []

    async def start(user_id, inputs):
        sid = f"fake-session-{uuid.uuid4().hex}"
        created.append(sid)
        await get_db()[Collections.SEARCH_SESSIONS].insert_one(
            {"id": sid, "user_id": user_id, "status": "searching", "original_question": inputs["question"],
             "deleted_at": None, "created_at": store.now()})
        return sid

    async def set_status(sid, status):
        await get_db()[Collections.SEARCH_SESSIONS].update_one({"id": sid}, {"$set": {"status": status}})

    monkeypatch.setattr(task_mod, "_start_legacy_session", start)
    return created, set_status


async def wait_task(run_id: str, key: str, predicate, timeout: float = 10.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        task = next(t for t in await store.get_tasks(run_id) if t["key"] == key)
        if predicate(task):
            return task
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"task {key} never matched: {task['status']}")
        await asyncio.sleep(0.05)


async def test_pause_and_resume(fake_legacy):
    created, set_status = fake_legacy
    view, run = await confirmed()
    await wait_task(run["id"], "await_retrieval", lambda t: (t.get("checkpoint") or {}).get("polls", 0) >= 1)
    await questions.control_run(user(), run["id"], "pause")
    paused = await wait_run(run["id"], {"paused"})
    assert paused["links"]["search_session_id"] == created[0]
    await asyncio.sleep(0.3)
    polls_while_paused = (await wait_task(run["id"], "await_retrieval", lambda t: True))["checkpoint"]["polls"]
    await asyncio.sleep(0.3)
    assert (await wait_task(run["id"], "await_retrieval", lambda t: True))["checkpoint"]["polls"] == polls_while_paused
    await set_status(created[0], "awaiting_selection")
    await questions.control_run(user(), run["id"], "resume")
    done = await wait_run(run["id"], {"completed", "failed"})
    assert done["status"] == "completed" and done["result_kind"] == "papers_ready"
    assert done["links"]["legacy_status"] == "awaiting_selection"
    evs = await events(run["id"])
    assert "run_paused" in evs and "run_resumed" in evs
    assert len(created) == 1


async def test_cancel_rejects_late_results(fake_legacy, monkeypatch):
    created, _ = fake_legacy
    gate = asyncio.Event()
    real = task_mod.HANDLERS["plan_search"]

    async def slow_plan(ctx):
        await gate.wait()
        return await real(ctx)

    monkeypatch.setitem(task_mod.HANDLERS, "plan_search", slow_plan)
    _, run = await confirmed()
    await wait_task(run["id"], "plan_search", lambda t: t["status"] == "running")
    await questions.control_run(user(), run["id"], "cancel")
    gate.set()
    await asyncio.sleep(0.3)
    final = await store.get_run(run["id"])
    assert final["status"] == "cancelled"
    tasks = {t["key"]: t for t in await store.get_tasks(run["id"])}
    assert tasks["plan_search"]["status"] == "cancelled" and not tasks["plan_search"]["output"]
    assert all(t["status"] == "cancelled" for t in tasks.values())
    assert "stale_result_rejected" in await events(run["id"])
    assert created == []  # nothing searched after cancellation
    with pytest.raises(questions.EngineConflict):
        await questions.control_run(user(), run["id"], "resume")


async def test_restart_resumes_from_checkpoint_without_duplicate_writes(fake_legacy):
    created, set_status = fake_legacy
    _, run = await confirmed()
    await wait_task(run["id"], "await_retrieval", lambda t: (t.get("checkpoint") or {}).get("polls", 0) >= 1)
    # Simulate the process dying: stop the driver, then leave the next task
    # 'running' under a dead worker's expired lease.
    await runner.shutdown()
    from core.db import get_db
    await get_db()[store.TASKS].update_one(
        {"run_id": run["id"], "key": "await_retrieval"},
        {"$set": {"status": "running", "lease_owner": "dead-worker:1",
                  "lease_expires_at": store.now() - timedelta(seconds=5)}},
    )
    await set_status(created[0], "awaiting_selection")
    resumed = await runner.recover_once()
    assert resumed >= 1
    done = await wait_run(run["id"], {"completed", "failed"})
    assert done["status"] == "completed"
    assert len(created) == 1  # the search was not started a second time
    tasks = {t["key"]: t for t in await store.get_tasks(run["id"])}
    assert tasks["retrieve"]["attempts"] == 1
    evs = await events(run["id"])
    assert "run_recovered" in evs
    claims = [t for t in await store.traces(run_id=run["id"]) if t["event"] == "task_claimed"
              and t["details"]["key"] == "await_retrieval"]
    assert claims[-1]["details"]["resumed_from_checkpoint"] is True


async def test_crash_after_session_created_adopts_it(fake_legacy, monkeypatch):
    """Crash between 'legacy session created' and 'checkpoint saved' must not
    start a second search on resume."""
    created, _ = fake_legacy
    from core.db import Collections, get_db

    gate = asyncio.Event()
    real_plan = task_mod.HANDLERS["plan_search"]

    async def held_plan(ctx):
        await gate.wait()
        return await real_plan(ctx)

    monkeypatch.setitem(task_mod.HANDLERS, "plan_search", held_plan)
    view = await questions.create_question(user(), Q)
    await questions.confirm_question(user(), view["question_id"], version=1, mode="review_papers")
    run_id = (await store.get_question(view["question_id"]))["active_run_id"]
    await wait_task(run_id, "plan_search", lambda t: t["status"] == "running")
    await runner.shutdown()  # the process dies here
    monkeypatch.setitem(task_mod.HANDLERS, "plan_search", real_plan)
    # Crash state: retrieve had created a session but not yet saved its checkpoint.
    await get_db()[store.TASKS].update_one({"run_id": run_id, "key": "plan_search"},
                                           {"$set": {"status": "succeeded", "lease_owner": None,
                                                     "output": {"search_plan": {}}}})
    await get_db()[store.TASKS].update_one(
        {"run_id": run_id, "key": "retrieve"},
        {"$set": {"status": "running", "lease_owner": "dead", "lease_expires_at": store.now() - timedelta(seconds=5),
                  "checkpoint": {"phase": "creating", "creating_at": store.now() - timedelta(seconds=11)},
                  "started_at": store.now() - timedelta(seconds=30)}})
    orphan = f"orphan-{run_id}"
    await get_db()[Collections.SEARCH_SESSIONS].insert_one(
        {"id": orphan, "user_id": "engine-test-user", "status": "awaiting_selection", "deleted_at": None,
         "original_question": Q, "created_at": store.now() - timedelta(seconds=10)})
    await runner.recover_once()
    done = await wait_run(run_id, {"completed", "failed"})
    assert done["status"] == "completed"
    assert created == []  # adopted the orphan instead of creating another session
    assert done["links"]["search_session_id"] == orphan


async def test_material_edit_supersedes_running_run(fake_legacy):
    created, _ = fake_legacy
    view, run = await confirmed()
    await wait_task(run["id"], "await_retrieval", lambda t: (t.get("checkpoint") or {}).get("polls", 0) >= 1)
    revised = await questions.revise_question(
        user(), view["question_id"], base_version=1,
        edits=[questions.ChipEdit(op="add", role="setting", value="intensive care unit")],
    )
    assert revised["version"] == 2 and revised["status"] == "awaiting_confirmation"
    old = await wait_run(run["id"], {"superseded"})
    assert old["invalidated_by_version"] == 2
    assert all(t["status"] in ("succeeded", "cancelled") for t in await store.get_tasks(run["id"]))
    # The earlier approval does not carry over to the new version.
    assert revised["approval"] is None and revised["run"] is None


async def test_answer_for_me_builds_a_delegated_evidence_set():
    view, run = await confirmed(mode="answer_for_me")
    done = await wait_run(run["id"], {"completed", "failed"})
    assert done["status"] == "completed"
    v = await store.get_version(view["question_id"], 1)
    assert v["approval"]["mode"] == "answer_for_me"
    assert v["delegation"]["policy"] == task_mod.DELEGATION_POLICY
    assert v["delegation"]["id"] != v["approval"]["id"]
    es = await store.get_evidence_set(done["links"]["evidence_set_id"])
    assert es["selection_mode"] == "delegated" and es["delegation_id"] == v["delegation"]["id"]
    refs = [i["publication_ref"] for i in es["items"]]
    assert "fx-3" not in refs and "fx-4" not in refs  # quarantined records never selected
    assert es["manifest_hash"]
    evs = [t["event"] for t in await store.traces(question_id=view["question_id"])]
    assert evs.index("approval_recorded") < evs.index("delegation_recorded")
