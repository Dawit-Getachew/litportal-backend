"""Durable orchestrator (W1.2): one runtime, state in Mongo, safe to restart.

* A run advances one leased task at a time; a task's lease is renewed while it
  works and expires if its process dies — another driver then reclaims it
  (restart / resume). Handlers are checkpoint-idempotent.
* Every result write is fenced on (lease owner, running status, question
  version); cancellation or supersession flips open tasks to ``cancelled`` so a
  late result is rejected and traced, never written.
* Retries are bounded with backoff; tool blocks, budget breaches and fatal
  errors never retry. Cooperative waits (TaskYield) consume no attempt.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from clinical_engine import store
from clinical_engine.config import settings
from clinical_engine.contracts.run import TERMINAL_RUN_STATUSES, Budget, RunDoc, TraceDoc
from clinical_engine.orchestrator.plan import build_tasks
from clinical_engine.orchestrator.tasks import (
    HANDLERS,
    FatalTaskError,
    RetryableTaskError,
    TaskContext,
    TaskYield,
)
from clinical_engine.orchestrator.tools import BudgetExceeded, ToolBlocked, ToolGateway

logger = logging.getLogger("litportal.clinical_engine.runner")

_driving: dict[str, asyncio.Task[None]] = {}


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def trace(run: dict, event: str, *, task: Optional[dict] = None, actor: str = "system",
                **details: Any) -> None:
    await store.add_trace(TraceDoc(
        question_id=run["question_id"], run_id=run["id"], task_id=task["id"] if task else None,
        event=event, actor=actor, details=details,
    ).model_dump())


# ── Creation and control ────────────────────────────────────────────────────


async def create_run(version: dict, *, mode: str, user_id: str) -> dict:
    run = RunDoc(
        question_id=version["question_id"], question_version_id=version["id"], version=version["version"],
        user_id=user_id, mode=mode,  # type: ignore[arg-type]
        retrieval_adapter="mock" if settings.RETRIEVAL_ADAPTER == "mock" else "legacy_pipeline",
        budget=Budget(max_tool_calls=settings.RUN_MAX_TOOL_CALLS, max_llm_tokens=settings.RUN_MAX_LLM_TOKENS,
                      max_wall_seconds=settings.RUN_MAX_WALL_SECONDS, max_task_attempts=settings.TASK_MAX_ATTEMPTS),
    ).model_dump()
    tasks = build_tasks(run, mode=mode, adapter=run["retrieval_adapter"])
    await store.insert_run(run, tasks)
    await trace(run, "run_created", actor=user_id, mode=mode, version=version["version"],
                plan=[t["key"] for t in tasks], adapter=run["retrieval_adapter"])
    return run


def spawn(run_id: str) -> None:
    existing = _driving.get(run_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(drive(run_id), name=f"engine-run-{run_id}")
    _driving[run_id] = task
    task.add_done_callback(lambda _t, rid=run_id: _driving.pop(rid, None) if _driving.get(rid) is _t else None)


async def pause(run: dict, actor: str) -> bool:
    ok = await store.update_run(run["id"], {"pause_requested": True},
                                expect={"status": {"$in": ["queued", "running"]}})
    if ok:
        await trace(run, "run_pause_requested", actor=actor)
    return ok


async def resume(run: dict, actor: str) -> bool:
    ok = await store.update_run(run["id"], {"status": "running", "pause_requested": False},
                                expect={"status": "paused"})
    if ok:
        await trace(run, "run_resumed", actor=actor)
        spawn(run["id"])
    return ok


async def cancel(run: dict, actor: str, reason: str = "cancelled_by_user") -> bool:
    ok = await store.update_run(
        run["id"], {"status": "cancelled", "cancel_requested": True, "finished_at": store.now(),
                    "error": {"code": reason}},
        expect={"status": {"$nin": list(TERMINAL_RUN_STATUSES)}},
    )
    if ok:
        n = await store.cancel_open_tasks(run["id"], reason)
        await trace(run, "run_cancelled", actor=actor, reason=reason, open_tasks_cancelled=n)
    return ok


async def supersede_question_runs(question_id: str, new_version: int, actor: str) -> int:
    """A material edit invalidates every run of earlier versions."""
    count = 0
    for run in await store.runs_for_question(question_id):
        if run["version"] >= new_version:
            continue
        if run["status"] not in TERMINAL_RUN_STATUSES:
            ok = await store.update_run(
                run["id"], {"status": "superseded", "finished_at": store.now(),
                            "error": {"code": "superseded_by_new_version"}, "invalidated_by_version": new_version},
                expect={"status": {"$nin": list(TERMINAL_RUN_STATUSES)}},
            )
            if ok:
                await store.cancel_open_tasks(run["id"], "superseded_by_new_version")
                await trace(run, "run_superseded", actor=actor, new_version=new_version)
                count += 1
        elif not run.get("invalidated_by_version"):
            await store.update_run(run["id"], {"invalidated_by_version": new_version})
            await trace(run, "run_results_invalidated", actor=actor, new_version=new_version)
            count += 1
    return count


# ── Driver ──────────────────────────────────────────────────────────────────


def _ready(task: dict, by_key: dict[str, dict], now: datetime) -> bool:
    if task["status"] == "running":
        lease = _aware(task.get("lease_expires_at"))
        return lease is not None and lease < now
    if task["status"] != "pending":
        return False
    if any(by_key.get(d, {}).get("status") != "succeeded" for d in task.get("depends_on") or []):
        return False
    nb = _aware(task.get("not_before"))
    return nb is None or nb <= now


async def _finish_run(run: dict, status: str, **fields: Any) -> None:
    ok = await store.update_run(run["id"], {"status": status, "finished_at": store.now(), **fields},
                                expect={"status": {"$nin": list(TERMINAL_RUN_STATUSES)}})
    if ok:
        if status != "completed":
            await store.cancel_open_tasks(run["id"], f"run_{status}")
        await trace(run, f"run_{status}", **{k: v for k, v in fields.items() if "." not in k})


async def drive(run_id: str) -> None:
    try:
        while True:
            run = await store.get_run(run_id)
            if run is None or run["status"] in TERMINAL_RUN_STATUSES or run["status"] == "paused":
                return
            if run.get("cancel_requested"):
                await cancel(run, "system")
                return
            if run.get("pause_requested"):
                if await store.update_run(run_id, {"status": "paused", "pause_requested": False},
                                          expect={"status": {"$in": ["queued", "running"]}}):
                    await trace(run, "run_paused")
                return
            if run["status"] == "queued":
                if await store.update_run(run_id, {"status": "running", "started_at": store.now()},
                                          expect={"status": "queued"}):
                    await trace(run, "run_started")
                continue

            now = store.now()
            started = _aware(run.get("started_at")) or _aware(run["created_at"])
            max_wall = int((run.get("budget") or {}).get("max_wall_seconds", settings.RUN_MAX_WALL_SECONDS))
            if now - started > timedelta(seconds=max_wall):
                await _finish_run(run, "failed", error={"code": "budget_exceeded:wall_seconds"})
                return

            tasks = await store.get_tasks(run_id)
            by_key = {t["key"]: t for t in tasks}
            failed = [t for t in tasks if t["status"] == "failed"]
            if failed:
                await _finish_run(run, "failed", error={"code": "task_failed", "task": failed[0]["key"],
                                                        "detail": failed[0].get("error")})
                return
            if tasks and all(t["status"] in ("succeeded", "skipped") for t in tasks):
                await _complete(run, by_key)
                return

            ready = [t for t in tasks if _ready(t, by_key, now)]
            if not ready:
                live_elsewhere = any(
                    t["status"] == "running" and t.get("lease_owner") != store.WORKER_ID
                    and (_aware(t.get("lease_expires_at")) or now) > now
                    for t in tasks
                )
                if live_elsewhere:
                    return  # another process holds the work; its driver continues
                waits = [(_aware(t["not_before"]) - now).total_seconds() for t in tasks
                         if t["status"] == "pending" and t.get("not_before")]
                await asyncio.sleep(max(0.05, min([settings.POLL_INTERVAL_S, *[w for w in waits if w > 0]])))
                continue

            claimed = await store.claim_task(ready[0], settings.LEASE_SECONDS)
            if claimed is None:
                continue  # lost the race to another driver
            routine_poll = ready[0]["status"] == "pending" and bool((ready[0].get("checkpoint") or {}).get("polls"))
            if not routine_poll:  # first claim, retry, or recovery of a dead worker's task
                await trace(run, "task_claimed", task=claimed, key=claimed["key"],
                            attempt=int(claimed.get("attempts", 0)) + 1,
                            resumed_from_checkpoint=bool(claimed.get("checkpoint")))
            await _execute(run, claimed, by_key)
    except asyncio.CancelledError:
        raise  # shutdown: leases lapse and another driver resumes the run
    except Exception:  # noqa: BLE001 — a driver bug must not kill the process
        logger.exception("engine driver crashed for run %s", run_id)


async def _heartbeat(task_id: str) -> None:
    interval = max(1.0, settings.LEASE_SECONDS / 3)
    while True:
        await asyncio.sleep(interval)
        if not await store.renew_lease(task_id, settings.LEASE_SECONDS):
            return


async def _fail_task(run: dict, task: dict, code: str, message: str = "") -> None:
    ok = await store.finish_task(task["id"], task["question_version_id"], {
        "status": "failed", "error": {"code": code, "message": message[:300]},
        "finished_at": store.now(), "lease_owner": None, "lease_expires_at": None,
    }, inc={"attempts": 1})
    await trace(run, "task_failed" if ok else "stale_result_rejected", task=task, key=task["key"], code=code)


async def _execute(run: dict, task: dict, by_key: dict[str, dict]) -> None:
    version = await store.get_version_by_id(run["question_version_id"])
    outputs = {k: (t.get("output") or {}) for k, t in by_key.items() if t["status"] == "succeeded"}
    ctx = TaskContext(run=run, task=task, version=version or {}, tools=ToolGateway(run, task), outputs=outputs)
    handler = HANDLERS.get(task["type"])
    started = _aware(run.get("started_at")) or store.now()
    max_wall = int((run.get("budget") or {}).get("max_wall_seconds", settings.RUN_MAX_WALL_SECONDS))
    remaining = max(1.0, max_wall - (store.now() - started).total_seconds())
    heartbeat = asyncio.create_task(_heartbeat(task["id"]))
    try:
        if handler is None or version is None:
            raise FatalTaskError("unknown_task_type" if handler is None else "question_version_missing")
        output = await asyncio.wait_for(handler(ctx), timeout=remaining)
    except TaskYield as y:
        ok = await store.finish_task(task["id"], task["question_version_id"], {
            "status": "pending", "not_before": store.now() + timedelta(seconds=y.delay_s),
            "checkpoint": y.checkpoint, "lease_owner": None, "lease_expires_at": None,
        })
        if not ok:
            await trace(run, "stale_result_rejected", task=task, key=task["key"], stage="yield")
        return
    except ToolBlocked as exc:
        await _fail_task(run, task, f"tool_blocked:{exc.reason}", exc.tool)
        return
    except BudgetExceeded as exc:
        await _fail_task(run, task, f"budget_exceeded:{exc.budget}")
        return
    except FatalTaskError as exc:
        await _fail_task(run, task, exc.code, exc.message)
        return
    except asyncio.TimeoutError:
        await _fail_task(run, task, "budget_exceeded:wall_seconds")
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — bounded retry for anything unexpected
        code = exc.code if isinstance(exc, RetryableTaskError) else type(exc).__name__
        attempts = int(task.get("attempts", 0)) + 1
        if attempts < int(task.get("max_attempts", settings.TASK_MAX_ATTEMPTS)):
            delay = min(60.0, settings.RETRY_BACKOFF_BASE_S * 2 ** (attempts - 1))
            ok = await store.finish_task(task["id"], task["question_version_id"], {
                "status": "pending", "not_before": store.now() + timedelta(seconds=delay),
                "lease_owner": None, "lease_expires_at": None, "error": {"code": code},
            }, inc={"attempts": 1})
            await trace(run, "task_retry_scheduled" if ok else "stale_result_rejected", task=task,
                        key=task["key"], code=code, attempt=attempts, delay_s=delay)
        else:
            await _fail_task(run, task, f"retries_exhausted:{code}")
        logger.warning("engine task %s failed (%s), attempt %s", task["key"], code, attempts)
        return
    finally:
        heartbeat.cancel()

    fresh = await store.get_run(run["id"])
    if fresh is None or fresh["status"] != "running":
        await trace(run, "stale_result_rejected", task=task, key=task["key"],
                    run_status=(fresh or {}).get("status"))
        return
    ok = await store.finish_task(task["id"], task["question_version_id"], {
        "status": "succeeded", "output": output, "finished_at": store.now(),
        "lease_owner": None, "lease_expires_at": None, "error": None,
    }, inc={"attempts": 1})
    if not ok:
        await trace(run, "stale_result_rejected", task=task, key=task["key"], stage="commit")
        return
    links = output.get("links") or {}
    if links:
        await store.update_run(run["id"], {f"links.{k}": v for k, v in links.items()})
    await trace(run, "task_succeeded", task=task, key=task["key"])


async def _complete(run: dict, by_key: dict[str, dict]) -> None:
    if run.get("retrieval_adapter") == "mock":
        kind = "fixture_demo"
    elif run["mode"] == "answer_for_me":
        kind = "evidence_set_ready"
    else:
        kind = "papers_ready"
    legacy = (by_key.get("await_retrieval") or {}).get("output") or {}
    extra: dict[str, Any] = {"result_kind": kind}
    if legacy.get("legacy_status"):
        extra["links.legacy_status"] = legacy["legacy_status"]
    await _finish_run(run, "completed", **extra)


# ── Recovery (restart / resume) ─────────────────────────────────────────────


async def recover_once() -> int:
    resumed = 0
    now = store.now()
    for run in await store.recoverable_runs():
        if run["id"] in _driving:
            continue
        tasks = await store.get_tasks(run["id"])
        live_elsewhere = any(
            t["status"] == "running" and t.get("lease_owner") != store.WORKER_ID
            and (_aware(t.get("lease_expires_at")) or now) > now
            for t in tasks
        )
        if live_elsewhere:
            continue
        await trace(run, "run_recovered", worker=store.WORKER_ID)
        spawn(run["id"])
        resumed += 1
    return resumed


async def recovery_loop(stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await recover_once()
        except Exception:  # noqa: BLE001
            logger.exception("engine recovery pass failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.RECOVERY_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


async def shutdown() -> None:
    for task in list(_driving.values()):
        task.cancel()
    for task in list(_driving.values()):
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    _driving.clear()
