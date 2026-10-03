"""Engine state documents: question threads, versions, approvals, runs, tasks, traces."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from models.common import new_id, utcnow

from clinical_engine.search_terms.contracts import SearchTerms

from .question import ClinicalQuestionMap, IntentConflict

QuestionStatus = Literal[
    "awaiting_confirmation",  # step 1: interpretation shown, waiting for the clinician
    "needs_clarification",    # blocking conflicts — must be edited before approval
    "awaiting_search_terms",  # step 2: interpretation approved; search terms shown for review
    "confirmed",              # an approved version exists; a run was started
    "declined",               # clinician discarded this question
    "blocked_phi",            # PHI detected — nothing stored, nothing searched
]

VersionChange = Literal["initial", "text_revision", "chip_edit"]
RunMode = Literal["review_papers", "answer_for_me"]

RunStatus = Literal[
    "queued", "running", "paused", "completed", "failed", "cancelled", "superseded",
]
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled", "superseded"}
)

TaskStatus = Literal[
    "pending", "running", "succeeded", "failed", "cancelled", "skipped",
]


class InterpretationMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["model", "rules_only", "user_edit"]
    model: Optional[str] = None
    fallback_reason: Optional[str] = None  # why the model was not used
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    rejected_model_fields: list[str] = Field(default_factory=list)


class InterpretationApproval(BaseModel):
    """Step 1: the clinician approved *this exact* interpretation version and moved
    on to reviewing the search terms. Nothing is searched yet."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    user_id: str
    version: int
    material_hash: str
    statement: str
    approved_at: datetime = Field(default_factory=utcnow)


class ApprovalRecord(BaseModel):
    """The clinician confirmed *this exact* interpretation version — and, in the
    two-step flow, this exact version of the search terms — and chose a mode."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    user_id: str
    version: int
    material_hash: str
    mode: RunMode
    statement: str
    search_terms_version: Optional[int] = None
    search_terms_hash: Optional[str] = None
    approved_at: datetime = Field(default_factory=utcnow)


class DelegationRecord(BaseModel):
    """Recorded separately from approval: only exists for "answer for me"."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    user_id: str
    version: int
    scope: Literal["policy_based_evidence_selection"] = "policy_based_evidence_selection"
    policy: str
    statement: str
    delegated_at: datetime = Field(default_factory=utcnow)


class QuestionDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    status: QuestionStatus
    current_version: int = 0
    current_version_id: Optional[str] = None
    confirmed_version: Optional[int] = None
    active_run_id: Optional[str] = None
    decline_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class QuestionVersionDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    question_id: str
    user_id: str
    version: int
    parent_version: Optional[int] = None
    change_kind: VersionChange
    map: ClinicalQuestionMap
    material_hash: str
    status: Literal["current", "superseded"] = "current"
    conflicts: list[IntentConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    interpretation: InterpretationMeta
    interpretation_approval: Optional[InterpretationApproval] = None
    search_terms: Optional[SearchTerms] = None
    approval: Optional[ApprovalRecord] = None
    delegation: Optional[DelegationRecord] = None
    created_at: datetime = Field(default_factory=utcnow)


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tool_calls: int = 20
    max_llm_tokens: int = 20000
    max_wall_seconds: int = 1800
    max_task_attempts: int = 3


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: int = 0
    llm_tokens: int = 0
    blocked_calls: int = 0


class RunDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    question_id: str
    question_version_id: str
    version: int
    user_id: str
    mode: RunMode
    status: RunStatus = "queued"
    plan_version: str = "plan.v1"
    retrieval_adapter: Literal["legacy_pipeline", "mock", "engine"] = "legacy_pipeline"
    progress: dict[str, Any] = Field(default_factory=dict)  # engine adapter: stage, sources, counts
    budget: Budget = Field(default_factory=Budget)
    usage: Usage = Field(default_factory=Usage)
    cancel_requested: bool = False
    pause_requested: bool = False
    links: dict[str, Any] = Field(default_factory=dict)
    result_kind: Optional[str] = None
    error: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class TaskDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    run_id: str
    question_version_id: str
    key: str
    type: str
    seq: int
    depends_on: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)  # allowlist for this task
    status: TaskStatus = "pending"
    attempts: int = 0
    max_attempts: int = 3
    not_before: Optional[datetime] = None
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: Optional[dict[str, Any]] = None
    idempotency_key: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class TraceDoc(BaseModel):
    """Safe audit trace. Never holds question text, prompts or evidence text."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    ts: datetime = Field(default_factory=utcnow)
    question_id: str
    run_id: Optional[str] = None
    task_id: Optional[str] = None
    event: str
    actor: str = "system"
    details: dict[str, Any] = Field(default_factory=dict)


class FeedbackDoc(BaseModel):
    """Was the result helpful? Optional comment (PHI-screened before storage)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    question_id: str
    version: Optional[int] = None
    run_id: Optional[str] = None
    helpful: bool
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
