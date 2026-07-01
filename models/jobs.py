"""Phase 4B durable job model.

Jobs are user-scoped, stored in Mongo so they survive process restarts.
Payloads NEVER contain raw evidence text, raw prompts, PHI, secrets, or
uploaded PDF bytes — IDs and refs only.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow


JobType = Literal[
    "search_pipeline",
    "oa_fetch",
    "evidence_packet_prepare",
    "summary_generation",
    "pdf_processing",
    "retention_cleanup",
]

JobStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "canceled",
    "expired",
]


# Closed-set error codes for jobs (never raw strings).
JobErrorCode = Literal[
    "ok",
    "validation_failed",
    "phi_blocked_session",
    "no_selected_articles",
    "abstract_only_acknowledgement_required",
    "max_articles_exceeded",
    "no_usable_evidence",
    "session_expired",
    "llm_not_configured",
    "parser_validation_failed",
    "revision_limit_exceeded",
    "verification_failed",
    "timeout",
    "stale_job",
    "canceled",
    "duplicate_suppressed",
    "internal_error",
    "other",
]


class JobDoc(BaseModel):
    """Durable job record. Polled by the frontend for async UX."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    search_session_id: Optional[str] = None
    job_type: JobType
    status: JobStatus = "queued"
    progress_stage: Optional[str] = None
    progress_percent: int = 0
    current_step: int = 0
    total_steps: int = 0
    # IDs and refs only — NEVER raw evidence text, raw prompts, raw bytes, PHI.
    result_ref_json: dict[str, Any] = Field(default_factory=dict)
    error_code: Optional[str] = None
    error_message_safe: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 1
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# Allowed payload-key whitelist (validated by the audit + admin layers).
JOB_RESULT_REF_KEYS: set[str] = {
    "summary_id", "session_id", "article_ids", "verification_attempts",
    "packet_summary_counts", "model_provider", "summary_model", "verifier_model",
    "prompt_version", "failed_reason_code", "user_message_code",
    "sections_created", "token_count", "source", "license_category",
    "oa_fetch_status", "duplicate_of_job_id",
}
