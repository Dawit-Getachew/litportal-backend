"""Mongo persistence for the engine (engine_* collections).

Only plain CRUD, conditional single-document updates and find_one_and_update —
all supported by Amazon DocumentDB (the AWS target) and Atlas alike; no
multi-document transactions, $expr or server-side JS. Every read strips _id.
"""
from __future__ import annotations

import logging
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pymongo import ASCENDING, DESCENDING, ReturnDocument

logger = logging.getLogger("litportal.clinical_engine.store")

QUESTIONS = "engine_questions"
VERSIONS = "engine_question_versions"
RUNS = "engine_runs"
TASKS = "engine_tasks"
TRACES = "engine_traces"
EVIDENCE_SETS = "engine_evidence_sets"
RATE = "engine_rate_counters"
FEEDBACK = "engine_feedback"
ALL_COLLECTIONS = (QUESTIONS, VERSIONS, RUNS, TASKS, TRACES, EVIDENCE_SETS, RATE, FEEDBACK)

# Identifies this process as a lease holder (multi-replica safe).
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
_P = {"_id": 0}


def _db():
    from core.db import get_db

    return get_db()


def now() -> datetime:
    return datetime.now(timezone.utc)


async def ensure_indexes() -> None:
    db = _db()
    specs: list[tuple[str, list[tuple[str, int]], dict[str, Any]]] = [
        (QUESTIONS, [("id", ASCENDING)], {"unique": True, "name": "eq_id"}),
        (QUESTIONS, [("user_id", ASCENDING), ("updated_at", DESCENDING)], {"name": "eq_user_updated"}),
        (VERSIONS, [("id", ASCENDING)], {"unique": True, "name": "eqv_id"}),
        (VERSIONS, [("question_id", ASCENDING), ("version", ASCENDING)], {"unique": True, "name": "eqv_q_version"}),
        (RUNS, [("id", ASCENDING)], {"unique": True, "name": "er_id"}),
        (RUNS, [("question_id", ASCENDING), ("created_at", DESCENDING)], {"name": "er_q_created"}),
        (RUNS, [("status", ASCENDING), ("updated_at", ASCENDING)], {"name": "er_status_updated"}),
        (RUNS, [("links.search_session_id", ASCENDING)], {"name": "er_session"}),
        (TASKS, [("id", ASCENDING)], {"unique": True, "name": "et_id"}),
        (TASKS, [("idempotency_key", ASCENDING)], {"unique": True, "name": "et_idem"}),
        (TASKS, [("run_id", ASCENDING), ("seq", ASCENDING)], {"name": "et_run_seq"}),
        (TRACES, [("question_id", ASCENDING), ("ts", ASCENDING)], {"name": "etr_q_ts"}),
        (TRACES, [("run_id", ASCENDING), ("ts", ASCENDING)], {"name": "etr_run_ts"}),
        (EVIDENCE_SETS, [("id", ASCENDING)], {"unique": True, "name": "ees_id"}),
        (EVIDENCE_SETS, [("run_id", ASCENDING)], {"name": "ees_run"}),
        (RATE, [("key", ASCENDING)], {"unique": True, "name": "erc_key"}),
        (RATE, [("expires_at", ASCENDING)], {"expireAfterSeconds": 0, "name": "erc_ttl"}),
        (FEEDBACK, [("question_id", ASCENDING), ("created_at", DESCENDING)], {"name": "efb_q_created"}),
    ]
    for coll, keys, opts in specs:
        try:
            await db[coll].create_index(keys, **opts)
        except Exception:  # noqa: BLE001 — an index problem must never stop the app
            logger.exception("engine index %s on %s failed", opts.get("name"), coll)


# ── Questions & versions ────────────────────────────────────────────────────


async def insert_question(doc: dict) -> None:
    await _db()[QUESTIONS].insert_one(dict(doc))


async def get_question(question_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    q: dict[str, Any] = {"id": question_id}
    if user_id is not None:
        q["user_id"] = user_id
    return await _db()[QUESTIONS].find_one(q, _P)


async def update_question(question_id: str, set_fields: dict, *, expect: Optional[dict] = None) -> bool:
    flt = {"id": question_id, **(expect or {})}
    res = await _db()[QUESTIONS].update_one(flt, {"$set": {**set_fields, "updated_at": now()}})
    return res.matched_count == 1


async def list_questions(user_id: str, limit: int = 20) -> list[dict]:
    cur = _db()[QUESTIONS].find({"user_id": user_id}, _P).sort("updated_at", DESCENDING).limit(limit)
    return [d async for d in cur]


async def insert_version(doc: dict) -> None:
    await _db()[VERSIONS].insert_one(dict(doc))


async def get_version(question_id: str, version: int) -> Optional[dict]:
    return await _db()[VERSIONS].find_one({"question_id": question_id, "version": version}, _P)


async def get_version_by_id(version_id: str) -> Optional[dict]:
    return await _db()[VERSIONS].find_one({"id": version_id}, _P)


async def list_versions(question_id: str) -> list[dict]:
    cur = _db()[VERSIONS].find({"question_id": question_id}, _P).sort("version", ASCENDING)
    return [d async for d in cur]


async def supersede_versions(question_id: str, keep_version: int) -> None:
    await _db()[VERSIONS].update_many(
        {"question_id": question_id, "version": {"$ne": keep_version}, "status": "current"},
        {"$set": {"status": "superseded"}},
    )


async def record_approval(version_id: str, approval: dict, delegation: Optional[dict],
                          *, search_terms_version: Optional[int] = None) -> bool:
    """Atomic: only a current, not-yet-approved version can be approved once —
    and, when the search terms were reviewed, only while they are still at the
    version the clinician saw."""
    set_fields: dict[str, Any] = {"approval": approval}
    if delegation is not None:
        set_fields["delegation"] = delegation
    flt: dict[str, Any] = {"id": version_id, "status": "current", "approval": None}
    if search_terms_version is not None:
        flt["search_terms.version"] = search_terms_version
    res = await _db()[VERSIONS].update_one(flt, {"$set": set_fields})
    return res.modified_count == 1


async def record_interpretation_approval(version_id: str, approval: dict, search_terms: dict) -> bool:
    """Step 1, atomic: a current, unapproved version gets its interpretation
    approval and first search terms exactly once."""
    res = await _db()[VERSIONS].update_one(
        {"id": version_id, "status": "current", "approval": None, "interpretation_approval": None},
        {"$set": {"interpretation_approval": approval, "search_terms": search_terms}},
    )
    return res.modified_count == 1


async def save_search_terms(version_id: str, search_terms: dict, *, expect_terms_version: int) -> bool:
    """Conditional write: the terms are replaced only if nobody edited them since
    the clinician's view (and the search has not been started)."""
    res = await _db()[VERSIONS].update_one(
        {"id": version_id, "status": "current", "approval": None, "search_terms.version": expect_terms_version},
        {"$set": {"search_terms": search_terms}},
    )
    return res.modified_count == 1


async def insert_feedback(doc: dict) -> None:
    await _db()[FEEDBACK].insert_one(dict(doc))


async def count_interpretation(user_id: str) -> int:
    """Per-user hourly counter (TTL cleaned) for model spend before approval."""
    bucket = now().strftime("%Y%m%d%H")
    doc = await _db()[RATE].find_one_and_update(
        {"key": f"interp:{user_id}:{bucket}"},
        {"$inc": {"count": 1}, "$setOnInsert": {"expires_at": now() + timedelta(hours=2)}},
        upsert=True, return_document=ReturnDocument.AFTER, projection=_P,
    )
    return int((doc or {}).get("count", 1))


# ── Runs & tasks ─────────────────────────────────────────────────────────────


async def insert_run(doc: dict, tasks: list[dict]) -> None:
    # Tasks first: a run is never visible without its plan.
    if tasks:
        await _db()[TASKS].insert_many([dict(t) for t in tasks], ordered=True)
    await _db()[RUNS].insert_one(dict(doc))


async def get_run(run_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    q: dict[str, Any] = {"id": run_id}
    if user_id is not None:
        q["user_id"] = user_id
    return await _db()[RUNS].find_one(q, _P)


async def get_run_for_session(session_id: str, user_id: str) -> Optional[dict]:
    """The owner's engine run that started this search session (latest first)."""
    cur = _db()[RUNS].find({"links.search_session_id": session_id, "user_id": user_id}, _P) \
        .sort("created_at", DESCENDING).limit(1)
    runs = [d async for d in cur]
    return runs[0] if runs else None


async def update_run(run_id: str, set_fields: dict, *, expect: Optional[dict] = None,
                     inc: Optional[dict] = None) -> bool:
    update: dict[str, Any] = {"$set": {**set_fields, "updated_at": now()}}
    if inc:
        update["$inc"] = inc
    res = await _db()[RUNS].update_one({"id": run_id, **(expect or {})}, update)
    return res.matched_count == 1


async def runs_for_question(question_id: str) -> list[dict]:
    cur = _db()[RUNS].find({"question_id": question_id}, _P).sort("created_at", DESCENDING)
    return [d async for d in cur]


async def sessions_owned_by_runs(session_ids: list[str]) -> set[str]:
    if not session_ids:
        return set()
    cur = _db()[RUNS].find({"links.search_session_id": {"$in": session_ids}}, {"_id": 0, "links": 1})
    return {d["links"]["search_session_id"] async for d in cur}


async def recoverable_runs(limit: int = 50) -> list[dict]:
    cur = _db()[RUNS].find({"status": {"$in": ["queued", "running"]}}, _P).sort("updated_at", ASCENDING).limit(limit)
    return [d async for d in cur]


async def consume_budget(run_id: str, field: str, amount: int, limit: int) -> bool:
    """Atomically add *amount* to usage.<field> only if it stays within *limit*."""
    res = await _db()[RUNS].update_one(
        {"id": run_id, f"usage.{field}": {"$lte": limit - amount}},
        {"$inc": {f"usage.{field}": amount}, "$set": {"updated_at": now()}},
    )
    return res.modified_count == 1


async def note_blocked_call(run_id: str) -> None:
    await _db()[RUNS].update_one({"id": run_id}, {"$inc": {"usage.blocked_calls": 1}})


async def get_tasks(run_id: str) -> list[dict]:
    cur = _db()[TASKS].find({"run_id": run_id}, _P).sort("seq", ASCENDING)
    return [d async for d in cur]


async def claim_task(task: dict, lease_seconds: int) -> Optional[dict]:
    """Take the lease on *task* if it is still claimable (pending and due, or
    running with an expired lease — i.e. its previous worker died)."""
    t = now()
    return await _db()[TASKS].find_one_and_update(
        {
            "id": task["id"],
            "$or": [
                {"status": "pending", "$or": [{"not_before": None}, {"not_before": {"$lte": t}}]},
                {"status": "running", "lease_expires_at": {"$lt": t}},
            ],
        },
        {"$set": {
            "status": "running", "lease_owner": WORKER_ID,
            "lease_expires_at": t + timedelta(seconds=lease_seconds),
            "started_at": task.get("started_at") or t, "updated_at": t,
        }},
        return_document=ReturnDocument.AFTER, projection=_P,
    )


async def renew_lease(task_id: str, lease_seconds: int) -> bool:
    res = await _db()[TASKS].update_one(
        {"id": task_id, "status": "running", "lease_owner": WORKER_ID},
        {"$set": {"lease_expires_at": now() + timedelta(seconds=lease_seconds), "updated_at": now()}},
    )
    return res.matched_count == 1


async def finish_task(task_id: str, question_version_id: str, set_fields: dict,
                      inc: Optional[dict] = None) -> bool:
    """Fenced write: only the current lease holder, for the task's own question
    version, while the task is still running. Anything else is a stale result."""
    update: dict[str, Any] = {"$set": {**set_fields, "updated_at": now()}}
    if inc:
        update["$inc"] = inc
    res = await _db()[TASKS].update_one(
        {"id": task_id, "status": "running", "lease_owner": WORKER_ID,
         "question_version_id": question_version_id},
        update,
    )
    return res.modified_count == 1


async def save_checkpoint(task_id: str, checkpoint: dict) -> bool:
    res = await _db()[TASKS].update_one(
        {"id": task_id, "status": "running", "lease_owner": WORKER_ID},
        {"$set": {"checkpoint": checkpoint, "updated_at": now()}},
    )
    return res.modified_count == 1


async def cancel_open_tasks(run_id: str, reason: str) -> int:
    res = await _db()[TASKS].update_many(
        {"run_id": run_id, "status": {"$in": ["pending", "running"]}},
        {"$set": {"status": "cancelled", "error": {"code": reason}, "updated_at": now(),
                  "lease_owner": None, "lease_expires_at": None}},
    )
    return res.modified_count


# ── Traces & evidence sets ──────────────────────────────────────────────────


async def add_trace(doc: dict) -> None:
    try:
        await _db()[TRACES].insert_one(dict(doc))
    except Exception:  # noqa: BLE001 — tracing must never break the workflow
        logger.exception("engine trace write failed")


async def traces(*, question_id: Optional[str] = None, run_id: Optional[str] = None,
                 limit: int = 500) -> list[dict]:
    q: dict[str, Any] = {}
    if question_id:
        q["question_id"] = question_id
    if run_id:
        q["run_id"] = run_id
    cur = _db()[TRACES].find(q, _P).sort("ts", ASCENDING).limit(limit)
    return [d async for d in cur]


async def insert_evidence_set(doc: dict) -> None:
    await _db()[EVIDENCE_SETS].insert_one(dict(doc))


async def get_evidence_set(set_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    q: dict[str, Any] = {"id": set_id}
    if user_id is not None:
        q["user_id"] = user_id
    return await _db()[EVIDENCE_SETS].find_one(q, _P)
