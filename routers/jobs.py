"""Jobs router — Phase 4B.

Endpoints (all `/api`):
  GET  /jobs/{job_id}
  POST /jobs/{job_id}/cancel
  GET  /search-sessions/{sid}/jobs
  GET  /admin/job-metrics
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from core.db import Collections, get_db
from core.security import get_current_user, require_admin
from services.jobs import job_service

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, current_user: dict = Depends(get_current_user)):
    doc = await job_service.get(job_id=job_id, user_id=current_user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found.")
    return doc


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, current_user: dict = Depends(get_current_user)):
    doc = await job_service.cancel(job_id=job_id, user_id=current_user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found.")
    return doc


@router.get("/search-sessions/{session_id}/jobs")
async def list_session_jobs(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    # Ownership is enforced via user_id filter in JobService.
    return {
        "jobs": await job_service.list_for_session(
            search_session_id=session_id, user_id=current_user["id"],
        ),
    }


@router.get("/admin/job-metrics", dependencies=[Depends(require_admin)])
async def admin_job_metrics():
    """Aggregate counts only. Never returns raw evidence text or payloads."""
    db = get_db()
    counts: dict[str, int] = {}
    for status_ in ("queued", "running", "succeeded", "failed", "canceled", "expired"):
        counts[status_] = await db[Collections.JOBS].count_documents({"status": status_})
    # Avg duration for succeeded/failed jobs (ms).
    pipeline = [
        {"$match": {"status": {"$in": ["succeeded", "failed"]},
                    "started_at": {"$ne": None}, "completed_at": {"$ne": None}}},
        {"$project": {"job_type": 1,
                      "duration_ms": {"$subtract": ["$completed_at", "$started_at"]}}},
        {"$group": {"_id": "$job_type",
                     "avg_ms": {"$avg": "$duration_ms"},
                     "n": {"$sum": 1}}},
    ]
    by_type: dict[str, dict] = {}
    async for row in db[Collections.JOBS].aggregate(pipeline):
        avg = row.get("avg_ms")
        # In Motor avg of subtracted dates returns timedelta milliseconds.
        if hasattr(avg, "total_seconds"):
            avg_ms_val = int(avg.total_seconds() * 1000)
        elif avg is not None:
            avg_ms_val = int(avg)
        else:
            avg_ms_val = None
        by_type[row["_id"]] = {"avg_duration_ms": avg_ms_val, "n": int(row.get("n") or 0)}
    summary_timeouts = await db[Collections.JOBS].count_documents(
        {"job_type": "summary_generation", "error_code": "timeout"}
    )
    summary_llm_not_configured = await db[Collections.JOBS].count_documents(
        {"job_type": "summary_generation", "error_code": "llm_not_configured"}
    )
    summary_verification_failures = await db[Collections.JOBS].count_documents(
        {"job_type": "summary_generation", "error_code": "verification_failed"}
    )
    stale = await db[Collections.JOBS].count_documents({"error_code": "stale_job"})
    duplicate_suppressed = await db[Collections.JOBS].count_documents(
        {"result_ref_json.duplicate_of_job_id": {"$ne": None}}
    )
    runner_mode = (os.environ.get("JOB_RUNNER_MODE") or "in_process").strip().lower()
    return {
        "runner_mode": runner_mode,
        "in_process_warning": (
            "Job runner is in_process: jobs do not survive across multiple "
            "backend replicas. Switch JOB_RUNNER_MODE to an external backend "
            "before scaling out."
        ) if runner_mode == "in_process" else None,
        "counts": counts,
        "by_job_type": by_type,
        "summary_timeouts": summary_timeouts,
        "summary_llm_not_configured": summary_llm_not_configured,
        "summary_verification_failures": summary_verification_failures,
        "stale_jobs": stale,
        "duplicate_jobs_suppressed": duplicate_suppressed,
    }
