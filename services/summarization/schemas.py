"""Pydantic schemas for the Phase 3 grounded summarizer + verifier.

These models are the SINGLE source of truth for the LLM output shapes. The
JSON-mode response is parsed strictly here; anything that does not validate
is rejected (failed_reason_code="parser_validation_failed").
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EvidenceTextLevel = Literal["abstract", "oa_full_text", "user_uploaded_full_text"]
Certainty = Literal["high", "moderate", "low", "very_low", "insufficient"]
ClaimType = Literal[
    "bottom_line",
    "effect",
    "population",
    "intervention",
    "comparator",
    "outcome",
    "limitation",
    "certainty",
    "caveat",
    "other",
]


class SelectedStudyEntry(BaseModel):
    """Per-study card on the verified summary view."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    article_id: str
    citation_label: str
    pmid: Optional[str] = None
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    title: str
    journal: Optional[str] = None
    publication_year: Optional[int | str] = None
    study_type: str = "unspecified"
    evidence_text_level: EvidenceTextLevel
    key_finding: str
    limitations: str
    supporting_evidence_span_ids: list[str] = Field(min_length=1)

    @field_validator("supporting_evidence_span_ids")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("supporting_evidence_span_ids must be non-empty")
        return v


class ClaimMapEntry(BaseModel):
    """One claim in the summary, tied to one or more evidence spans."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    claim_id: str
    claim_text: str
    claim_type: ClaimType = "other"
    supporting_evidence_span_ids: list[str] = Field(min_length=1)
    supporting_article_ids: list[str] = Field(default_factory=list)

    @field_validator("supporting_evidence_span_ids")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("supporting_evidence_span_ids must be non-empty")
        return v


class DraftSummary(BaseModel):
    """The structured output produced by the summarizer LLM."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    clinical_bottom_line: str
    evidence_summary: str
    applicability: str
    certainty: Certainty
    certainty_rationale: str
    selected_studies: list[SelectedStudyEntry] = Field(default_factory=list)
    claim_map: list[ClaimMapEntry] = Field(default_factory=list)
    conflicting_evidence: str = ""
    important_caveats: str = ""
    what_was_not_answered: str = ""


# ----------------------------------------------------------------------------
# Verifier output schema (Task 5)
# ----------------------------------------------------------------------------


class UnsupportedClaim(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    claim_id: str
    claim_text: str
    reason: str


class IncorrectCitation(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    claim_id: str
    evidence_span_id: str
    reason: str


class VerifierResult(BaseModel):
    """Raw LLM verifier output. The mechanical floor (verifier.py) overrides
    this when deterministic checks find violations even if the LLM said pass.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    verification_status: Literal["passed", "failed"]
    final_answer_safe_to_show: bool = False
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    incorrect_citations: list[IncorrectCitation] = Field(default_factory=list)
    required_revisions: list[str] = Field(default_factory=list)
    verifier_notes: str = ""
    verified_summary_json: Optional[dict] = None
