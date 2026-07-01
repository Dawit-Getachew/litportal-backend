"""Search session, queries, session_articles, and summary schemas."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow

SessionStatus = Literal[
    "draft",
    "blocked_phi",
    "searching",
    "ranking",
    "brokering_access",
    "fetching_oa_full_text",
    "awaiting_selection",
    # Phase 6F closeout — terminal status emitted when
    # PrimaryClinicalEntityAnchorService returns
    # ``anchor_failure_reason="no_primary_clinical_entity"`` after
    # repair attempts. NO PubMed / Europe PMC / OpenAlex / scholar-like
    # expansion / LLM call has been made. The session carries
    # ``clarification_message_text`` and the UI surfaces a
    # ``search-clarification-banner`` with a "Refine question" CTA.
    "awaiting_clinical_subject",
    "summarizing",
    "complete",
    "error",
]


def default_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=30)


class SearchSessionDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    # NOTE: original_question may be None when status="blocked_phi" — the raw
    # text MUST NOT be persisted if PHI was detected. See routers/search_sessions.
    original_question: Optional[str] = None
    sanitized_question: Optional[str] = None
    topic_label: Optional[str] = None
    clinical_question_type: Optional[str] = None
    pico_json: dict = Field(default_factory=dict)
    status: SessionStatus = "draft"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime = Field(default_factory=default_expiry)
    deleted_at: Optional[datetime] = None
    # Phase 2A additions
    evaluator_json: Optional[dict] = None
    error_message: Optional[str] = None
    # Phase 2B additions
    retrieved_by_sources: list[str] = Field(default_factory=list)
    # Phase L1 — explicit clinician evidence-lens + population-focus
    # preferences (soft-boost / shadow-first). Stored per session;
    # never learned or personalized.
    evidence_lens: str = "clinically_direct"
    population_focus: str = "any"
    preference_mode: str = "soft_boost"
    preference_provenance: str = "user_explicit_preference"
    l1_preference_diagnostics_json: Optional[dict] = None


class EvidenceLensUpdate(BaseModel):
    """Phase L1 — per-search preference update body."""

    evidence_lens: Optional[str] = None
    population_focus: Optional[str] = None


class SearchSessionCreate(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    # Phase L1 — optional explicit clinician preferences at creation.
    evidence_lens: Optional[str] = None
    population_focus: Optional[str] = None


class SearchSessionResponse(BaseModel):
    id: str
    user_id: str
    original_question: Optional[str] = None
    sanitized_question: Optional[str] = None
    topic_label: Optional[str] = None
    clinical_question_type: Optional[str] = None
    pico_json: dict = Field(default_factory=dict)
    # Phase 5A — richer PICOTT structure + spelling corrections. The legacy
    # ``pico_json`` is preserved for BC.
    picott_json: dict = Field(default_factory=dict)
    spelling_corrections: list[dict] = Field(default_factory=list)
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    deleted_at: Optional[datetime] = None
    # Computed extras
    reasons: list[str] = Field(default_factory=list)
    next_step: Optional[str] = None
    evaluator_json: Optional[dict] = None
    error_message: Optional[str] = None
    # Phase 5D.1 — new fields exposed on GET. Both nullable so the
    # legacy / pre-Phase-5D.1 sessions stay valid.
    interpreted_question_display_json: Optional[dict] = None
    default_selection_summary: Optional[dict] = None
    # Phase 6F closeout (IMPL-2) — when status==``awaiting_clinical_subject``
    # the API surfaces ``clarification_message`` so the UI banner can
    # render the closed-set CLARIFICATION_MESSAGE. Empty / None for any
    # other status.
    clarification_message: Optional[str] = None
    # Phase 6F closeout (PROOF E) — efficiency instrumentation panel.
    # Closed-set numeric / boolean diagnostics computed at GET time
    # from persisted session state. NO PHI / NO question text. Used
    # by the transparency panel + admin views.
    session_diagnostics_json: Optional[dict] = None
    # Phase L1 — explicit evidence-lens / population-focus preferences.
    # Surfaced on every GET so the review-page controls render the
    # current state. ``l1_preference_diagnostics`` is the closed-set
    # soft-boost/shadow result block (no PHI / no free text).
    evidence_lens: str = "clinically_direct"
    population_focus: str = "any"
    preference_mode: str = "soft_boost"
    l1_preference_diagnostics: Optional[dict] = None
    # Phase L1.0.1 — production UI visibility gate. Backend is the source
    # of truth; the frontend renders the L1 controls only when this is
    # true (always true in preview/dev; false in production unless
    # explicitly enabled).
    l1_controls_visible: bool = True


class SearchQueryDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    search_session_id: str
    source: str
    query_type: str
    query_string: str
    rationale: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    # Phase 6B.1 — planner provenance + concept-block metadata. These
    # fields are populated by ``services/pipeline.py`` on every insert
    # so the transparency panel + admin diagnostics + tests can read
    # them without joining other collections.
    query_family: Optional[str] = None
    planner_source: Optional[str] = None  # closed set: boolean_compiler |
                                            # legacy_planner | rescue |
                                            # scholar_like_expansion |
                                            # manual_debug
    concept_blocks_used_json: dict = Field(default_factory=dict)
    mandatory_anchors_json: list = Field(default_factory=list)
    optional_blocks_json: list = Field(default_factory=list)
    role_weights_json: Optional[dict] = None
    is_primary_strategy: bool = False
    is_fallback_strategy: bool = False
    fallback_reason: Optional[str] = None


class SearchSessionArticleDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    search_session_id: str
    article_id: str
    relevance_score: float = 0.0
    evidence_design_score: float = 0.0
    overall_rank: int = 0
    rank_explanation: Optional[str] = None
    included_by_default: bool = False
    user_included: bool = False
    exclusion_reason: Optional[str] = None
    # Phase 2A: which planner query variant(s) surfaced this article
    query_types_matched: list[str] = Field(default_factory=list)
    # Phase 2B: which sources retrieved this article (e.g. ["europe_pmc","pubmed"])
    sources_present: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


VerificationStatus = Literal["pending", "running", "passed", "failed"]

FailedReasonCode = Literal[
    "no_selected_articles",
    "no_usable_evidence",
    "llm_not_configured",
    "unsupported_claims",
    "incorrect_citations",
    "fabricated_identifiers",
    "overstated_certainty",
    "revision_limit_exceeded",
    "parser_validation_failed",
    "timeout",
    "abstract_only_acknowledgement_required",
    "max_articles_exceeded",
    "session_expired",
    "phi_blocked_session",
    "other",
]


class SummaryDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    search_session_id: str
    selected_article_ids_json: list[str] = Field(default_factory=list)
    answer_json: dict[str, Any] = Field(default_factory=dict)
    verification_status: VerificationStatus = "pending"
    verifier_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    # Phase 3 additions
    evidence_packet_summary_json: dict[str, Any] = Field(default_factory=dict)
    claim_map_json: list[dict[str, Any]] = Field(default_factory=list)
    model_provider: Optional[str] = None
    summary_model: Optional[str] = None
    verifier_model: Optional[str] = None
    prompt_version: Optional[str] = None
    verification_attempts: int = 0
    failed_reason_code: Optional[FailedReasonCode] = None
    displayed_to_user: bool = False
    expires_at: Optional[datetime] = None

