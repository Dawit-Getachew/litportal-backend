"""Phase 6E — Benchmark gold-set curation models.

Closed-set enums; no free-form status / role / lane fields.

User-supplied text fields (``rationale``, ``reviewer_notes``) are
capped at 200 chars at the model boundary to keep audit payloads
PHI-free.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow


# ============================================================
# Closed-set enums (per Phase 6E spec — binding)
# ============================================================
IdentifierType = Literal[
    "pmid", "doi", "pmcid", "title_exact",
    "official_source_url", "source_record_id",
]
RelevanceRole = Literal[
    "primary_gold",
    "secondary_gold",
    "negative_example",
    "official_guideline",
    "background_reference",
]
ExpectedLane = Literal[
    "diagnosis", "therapy", "guideline", "prevention", "schedule",
    "benefit", "background", "harm", "prognosis", "pharmacology",
    "dosing_protocol",
]
RankingSignal = Literal[
    "must_appear", "should_appear", "must_not_dominate",
    "informational_only",
]
CandidateStatus = Literal[
    "pending_review", "accepted", "rejected", "needs_review",
]
GovernanceStatus = Literal[
    "parse_only", "needs_gold_set", "ready_for_review",
    "active_benchmark", "retired",
]


# ============================================================
# benchmark_gold_candidates collection
# ============================================================
class BenchmarkGoldCandidateDoc(BaseModel):
    """One curation candidate for one benchmark case.

    Idempotency is enforced at the service layer via the natural key
    ``(benchmark_case_id, identifier_type, identifier_value)``. Multiple
    candidates per case are normal — each represents one PMID/DOI/URL
    proposal.
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    benchmark_case_id: str
    identifier_type: IdentifierType
    identifier_value: str = Field(min_length=1, max_length=500)
    relevance_role: RelevanceRole
    expected_lane: Optional[ExpectedLane] = None
    ranking_signal: list[RankingSignal] = Field(default_factory=list)
    rationale: Optional[str] = Field(default=None, max_length=200)
    reviewer_notes: Optional[str] = Field(default=None, max_length=200)
    status: CandidateStatus = "pending_review"
    domain_trust: Optional[str] = None  # ``trusted`` | ``untrusted`` | None
    domain_host: Optional[str] = None   # host portion only — never path/query

    created_by_user_id: str
    accepted_by_user_id: Optional[str] = None
    rejected_by_user_id: Optional[str] = None
    accepted_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class BenchmarkGoldCandidateCreate(BaseModel):
    benchmark_case_id: str
    identifier_type: IdentifierType
    identifier_value: str = Field(min_length=1, max_length=500)
    relevance_role: RelevanceRole
    expected_lane: Optional[ExpectedLane] = None
    ranking_signal: list[RankingSignal] = Field(default_factory=list)
    rationale: Optional[str] = Field(default=None, max_length=200)


class BenchmarkGoldCandidateUpdate(BaseModel):
    """Used by the accept/reject PATCH path."""
    status: Optional[CandidateStatus] = None
    reviewer_notes: Optional[str] = Field(default=None, max_length=200)


# ============================================================
# Seed import shapes (Task 4)
# ============================================================
class SeedCandidate(BaseModel):
    identifier_type: IdentifierType
    identifier_value: str = Field(min_length=1, max_length=500)
    relevance_role: RelevanceRole
    expected_lane: Optional[ExpectedLane] = None
    ranking_signal: list[RankingSignal] = Field(default_factory=list)
    rationale: Optional[str] = Field(default=None, max_length=200)


class SeedCase(BaseModel):
    benchmark_case_id: str
    candidates: list[SeedCandidate] = Field(default_factory=list)


class BenchmarkGoldSeedImport(BaseModel):
    """Top-level shape for ``POST /api/admin/benchmark-gold/import``."""
    schema_version: Literal["phase6e_v1"] = "phase6e_v1"
    cases: list[SeedCase] = Field(default_factory=list)


# ============================================================
# Case activation / retirement
# ============================================================
class BenchmarkCaseActivateRequest(BaseModel):
    activate_reason: Optional[str] = Field(default=None, max_length=200)


class BenchmarkCaseRetireRequest(BaseModel):
    retire_reason: Optional[str] = Field(default=None, max_length=200)
