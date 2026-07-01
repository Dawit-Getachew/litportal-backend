"""Phase K4 — evidence_packets Pydantic models.

Closed-set schema for the clinician-visible branch-aware evidence
packet. Versioned via ``evidence_packet_version`` (semver string).
Strictly excludes: abstracts, full text, uploaded text, prompts,
completions, credentials, API keys, JWTs, env values, PHI strings.
Title / journal / year / source IDs allowed.

NO LLM. NO retrieval. Display-only transparency on top of the
already-persisted ``search_session_articles`` + ``articles`` +
``search_queries`` collections.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow


EVIDENCE_PACKET_VERSION = "k5.1.0"

# Closed-set per-article access eligibility (mirrors articles.full_text_status).
AccessEligibility = Literal[
    "full_text_available",
    "free_full_text_link",
    "abstract_only",
    "unknown",
]

# Closed-set per-article AI-ingestion eligibility for future summary.
AIIngestionEligibility = Literal[
    "eligible",
    "requires_upload",
    "not_eligible",
    "unknown",
]

# Closed-set selection state.
SelectedState = Literal[
    "selected_by_default",
    "selected_by_user",
    "unselected",
]

# Closed-set citation-verification readiness.
CitationVerificationReadiness = Literal[
    "ready",
    "missing_doi",
    "missing_pmid",
    "not_applicable",
]

# Phase K5.1 — closed-set per-article guideline-lane provenance label.
# Only "pubmed_indexed_guideline" exists today (official-society connectors
# are deferred to K5.2+). ``None`` = not a guideline-lane article / lane
# not surfaced for this session.
GuidelineLaneProvenance = Literal["pubmed_indexed_guideline"]


class EvidencePacketArticle(BaseModel):
    """Closed-set per-article record inside a branch."""

    model_config = ConfigDict(extra="ignore")

    article_id: str
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    source: Optional[str] = None

    branch_rank: int = 0
    global_rank: int = 0

    # Display-only R-series transparency
    directness_labels: list[str] = Field(default_factory=list)
    directness_label_human_strings: list[dict[str, str]] = Field(
        default_factory=list,
    )
    is_mixed_signal: bool = False

    source_provenance: list[str] = Field(default_factory=list)
    access_eligibility: AccessEligibility = "unknown"
    ai_ingestion_eligibility: AIIngestionEligibility = "unknown"
    selected_state: SelectedState = "unselected"
    summary_eligibility: bool = False
    citation_verification_readiness: CitationVerificationReadiness = (
        "not_applicable"
    )

    # Phase K5.1 — PubMed guideline-lane provenance (display-only).
    guideline_lane: Optional[GuidelineLaneProvenance] = None


class EvidencePacketBranch(BaseModel):
    """Closed-set per-branch record."""

    model_config = ConfigDict(extra="ignore")

    branch_id: str
    framework: str = ""
    intent: str = ""
    branch_summary: str = ""
    query_terms_used: list[str] = Field(default_factory=list)
    articles: list[EvidencePacketArticle] = Field(default_factory=list)

    # Phase K5.1 — guideline-lane surfacing (display-only). Active only for
    # guideline / management / protocol / stabilization / monitoring /
    # prevention / screening / diagnostic-workup questions AND when the
    # branch contains at least one PubMed-indexed guideline article.
    guideline_lane_active: bool = False
    guideline_lane_label: Optional[str] = None
    guideline_lane_tooltip: Optional[str] = None


class EvidencePacketProvenance(BaseModel):
    """Closed-set provenance block."""

    model_config = ConfigDict(extra="ignore")

    generator: Literal["k4_evidence_packet_builder"] = (
        "k4_evidence_packet_builder"
    )
    retrieval_run_id: Optional[str] = None
    r_series_labels_source: Literal[
        "computed_at_packet_build_time",
        "pulled_from_sidecar",
    ] = "computed_at_packet_build_time"
    r_series_labels_versions: dict[str, str] = Field(default_factory=dict)


class EvidencePacketDoc(BaseModel):
    """Closed-set top-level packet doc — one per search_session_id."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    search_session_id: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    evidence_packet_version: str = EVIDENCE_PACKET_VERSION
    generated_at: datetime = Field(default_factory=utcnow)
    provenance: EvidencePacketProvenance = Field(
        default_factory=EvidencePacketProvenance,
    )
    question: Optional[str] = None  # sanitized snapshot only
    clinical_question_map_snapshot: dict[str, Any] = Field(default_factory=dict)
    branches: list[EvidencePacketBranch] = Field(default_factory=list)
    # Phase L1 — closed-set clinician preference metadata (evidence lens
    # + population focus). Display-only; soft-boost/shadow, never a hard
    # filter, never overrides safety caps. Empty dict when no preference
    # was applied.
    preference_metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "EVIDENCE_PACKET_VERSION",
    "AccessEligibility",
    "AIIngestionEligibility",
    "SelectedState",
    "CitationVerificationReadiness",
    "GuidelineLaneProvenance",
    "EvidencePacketArticle",
    "EvidencePacketBranch",
    "EvidencePacketProvenance",
    "EvidencePacketDoc",
]
