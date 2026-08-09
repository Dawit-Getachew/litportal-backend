"""Phase 4B durable job service.

In-process async runner backed by Mongo persistence. Jobs survive backend
process restarts: stale running jobs (no heartbeat newer than
`JOB_STALE_AFTER_MINUTES`) are marked failed by the retention sweep.

Concurrency is enforced with asyncio Semaphores per job_type. The current
runner is single-process; the design surface is ready to swap to Redis/Arq
later (see TODO comments).

NEVER persists raw evidence text, raw prompts, PHI, secrets, or uploaded
bytes — only IDs, codes, and counts.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from core.db import Collections, get_db
from models.jobs import JOB_RESULT_REF_KEYS, JobDoc, JobType

logger = logging.getLogger("clinevidence.jobs")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sanitize_result_ref(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip any payload key not on the closed-set allow-list. Never carries
    raw evidence text / raw prompts / PHI / secrets / uploaded bytes.
    """
    if not isinstance(payload, dict):
        return {}
    return {k: v for k, v in payload.items() if k in JOB_RESULT_REF_KEYS}


# ---- Module-level concurrency primitives -----------------------------------


_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def _semaphore_for(job_type: JobType) -> asyncio.Semaphore:
    if job_type == "summary_generation":
        cap = _env_int("MAX_CONCURRENT_SUMMARY_JOBS", 2)
        key = "summary_generation"
    elif job_type == "oa_fetch":
        cap = _env_int("MAX_CONCURRENT_OA_FETCH_JOBS", 3)
        key = "oa_fetch"
    else:
        cap = 4
        key = job_type
    sem = _SEMAPHORES.get(key)
    if sem is None:
        sem = asyncio.Semaphore(cap)
        _SEMAPHORES[key] = sem
    return sem


_LIVE_TASKS: set[asyncio.Task] = set()


class JobService:
    """Durable Mongo-backed job orchestrator. See module docstring."""

    # ------------------------------------------------------------------ CRUD

    async def create(
        self,
        *,
        user_id: str,
        job_type: JobType,
        search_session_id: Optional[str] = None,
        total_steps: int = 4,
        max_attempts: int = 1,
        expires_at: Optional[datetime] = None,
    ) -> dict:
        db = get_db()
        doc = JobDoc(
            user_id=user_id, job_type=job_type,
            search_session_id=search_session_id,
            total_steps=total_steps, max_attempts=max_attempts,
            expires_at=expires_at,
        ).model_dump()
        await db[Collections.JOBS].insert_one(doc)
        return doc

    async def get(self, *, job_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        db = get_db()
        q: dict[str, Any] = {"id": job_id}
        if user_id is not None:
            q["user_id"] = user_id
        return await db[Collections.JOBS].find_one(q, {"_id": 0})

    async def list_for_session(
        self, *, search_session_id: str, user_id: str,
    ) -> list[dict]:
        db = get_db()
        rows = await db[Collections.JOBS].find(
            {"search_session_id": search_session_id, "user_id": user_id},
            {"_id": 0},
        ).sort("created_at", -1).to_list(length=50)
        return rows

    async def find_running(
        self, *, user_id: str, job_type: JobType,
        search_session_id: Optional[str] = None,
    ) -> Optional[dict]:
        db = get_db()
        q: dict[str, Any] = {
            "user_id": user_id, "job_type": job_type,
            "status": {"$in": ["queued", "running"]},
        }
        if search_session_id is not None:
            q["search_session_id"] = search_session_id
        return await db[Collections.JOBS].find_one(q, {"_id": 0})

    # -------------------------------------------------------------- state ops

    async def mark_running(
        self, *, job_id: str, stage: Optional[str] = None,
        current_step: int = 1, percent: int = 5,
    ) -> None:
        await self._set_state(job_id, status="running",
                              started_at=_utcnow(),
                              progress_stage=stage,
                              current_step=current_step,
                              progress_percent=percent)

    async def update_progress(
        self, *, job_id: str, stage: str,
        current_step: int, percent: int,
    ) -> None:
        await self._set_state(
            job_id, progress_stage=stage,
            current_step=current_step, progress_percent=percent,
        )

    async def succeed(
        self, *, job_id: str, result_ref: Optional[dict[str, Any]] = None,
        stage: str = "complete", percent: int = 100,
    ) -> None:
        await self._set_state(
            job_id, status="succeeded",
            completed_at=_utcnow(),
            progress_stage=stage, progress_percent=percent,
            result_ref_json=_sanitize_result_ref(result_ref or {}),
            error_code="ok",
        )

    async def fail(
        self, *, job_id: str, error_code: str,
        error_message_safe: str,
        result_ref: Optional[dict[str, Any]] = None,
        stage: str = "failed", percent: int = 100,
    ) -> None:
        await self._set_state(
            job_id, status="failed",
            completed_at=_utcnow(),
            error_code=str(error_code)[:64],
            error_message_safe=str(error_message_safe)[:240],
            result_ref_json=_sanitize_result_ref(result_ref or {}),
            progress_stage=stage, progress_percent=percent,
        )

    async def cancel(self, *, job_id: str, user_id: str) -> Optional[dict]:
        db = get_db()
        doc = await db[Collections.JOBS].find_one(
            {"id": job_id, "user_id": user_id}, {"_id": 0}
        )
        if not doc:
            return None
        if doc["status"] in ("succeeded", "failed", "canceled", "expired"):
            return doc  # idempotent
        await db[Collections.JOBS].update_one(
            {"id": job_id, "user_id": user_id},
            {"$set": {
                "status": "canceled", "completed_at": _utcnow(),
                "error_code": "canceled",
                "error_message_safe": "Job canceled by user.",
                "updated_at": _utcnow(),
            }},
        )
        return await db[Collections.JOBS].find_one({"id": job_id}, {"_id": 0})

    async def _set_state(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = _utcnow()
        # Drop None values to avoid clobbering.
        fields = {k: v for k, v in fields.items() if v is not None}
        db = get_db()
        await db[Collections.JOBS].update_one({"id": job_id}, {"$set": fields})

    async def heartbeat(self, *, job_id: str) -> None:
        await self._set_state(job_id)

    # ------------------------------------------------------------ dispatcher

    def dispatch(
        self,
        *,
        job_id: str,
        runner: Callable[[], Awaitable[None]],
        job_type: JobType,
    ) -> None:
        """Spawn the runner as a fire-and-forget asyncio task.

        Each task is bracketed by a semaphore for the job_type and a top-level
        try/except so any uncaught error becomes `failed` with a safe code.
        """
        async def _wrapped() -> None:
            sem = _semaphore_for(job_type)
            async with sem:
                try:
                    await runner()
                except asyncio.CancelledError:
                    await self.fail(
                        job_id=job_id, error_code="canceled",
                        error_message_safe="Job canceled.",
                    )
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("job %s runner crashed", job_id)
                    await self.fail(
                        job_id=job_id, error_code="internal_error",
                        error_message_safe=f"{type(exc).__name__}",
                    )

        task = asyncio.create_task(_wrapped(), name=f"job:{job_id}")
        _LIVE_TASKS.add(task)
        task.add_done_callback(_LIVE_TASKS.discard)

    # ------------------------------------------------------------ stale sweep

    async def sweep_stale_jobs(self) -> int:
        """Mark in-progress jobs whose `updated_at` is older than the stale
        threshold as failed with `error_code=stale_job`.

        Returns the number of jobs marked stale.
        """
        db = get_db()
        threshold = _utcnow() - timedelta(minutes=_env_int("JOB_STALE_AFTER_MINUTES", 15))
        res = await db[Collections.JOBS].update_many(
            {
                "status": {"$in": ["queued", "running"]},
                "updated_at": {"$lt": threshold},
            },
            {"$set": {
                "status": "failed",
                "completed_at": _utcnow(),
                "error_code": "stale_job",
                "error_message_safe": "Job marked stale after no heartbeat.",
                "updated_at": _utcnow(),
            }},
        )
        return int(res.modified_count or 0)


job_service = JobService()
