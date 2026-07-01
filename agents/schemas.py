"""Closed-set Pydantic output schemas for the three LitPortal AI agents.

All models use ``extra="ignore"`` and self-clamping validators so a sloppy LLM
response degrades gracefully (coerced to safe defaults) instead of discarding
the whole result. No field accepts a free-form identifier the model could
fabricate (PMIDs/DOIs) — candidates are referenced by integer index only.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.pico import CLINICAL_QUESTION_TYPES

_ENTITY_KINDS = {
    "drug", "drug_class", "condition", "procedure", "test",
    "population", "outcome", "other",
}
_ADVISORY_LABELS = {
    "direct_candidate", "borderline_candidate", "indirect_candidate", "keep_excluded",
}


def _clamp_conf(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _str_list(v: Any, *, cap: int = 25, item_max: int = 120) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v:
        s = str(item).strip()
        if s:
            out.append(s[:item_max])
        if len(out) >= cap:
            break
    return out


class Entity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(default="", max_length=160)
    kind: str = "other"

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, v: Any) -> str:
        s = str(v or "").strip().lower().replace(" ", "_")
        return s if s in _ENTITY_KINDS else "other"


# ── Agent 1: query categorizer ──────────────────────────────────────
class QueryCategoryResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    normalized_query: str = Field(default="", max_length=600)
    question_type_hint: str = "other"
    entities: list[Entity] = Field(default_factory=list)
    enriched_terms: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("question_type_hint", mode="before")
    @classmethod
    def _coerce_type(cls, v: Any) -> str:
        s = str(v or "").strip().lower()
        return s if s in CLINICAL_QUESTION_TYPES else "other"

    @field_validator("enriched_terms", "reason_codes", mode="before")
    @classmethod
    def _coerce_lists(cls, v: Any) -> list[str]:
        return _str_list(v)

    @field_validator("entities", mode="before")
    @classmethod
    def _cap_entities(cls, v: Any) -> Any:
        return v[:25] if isinstance(v, list) else []

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        return _clamp_conf(v)


# ── Agent 2: question understanding (ContextAugment / B3) ────────────
class QuestionUnderstanding(BaseModel):
    model_config = ConfigDict(extra="ignore")
    intent_hint: str = Field(default="", max_length=120)
    broad_question_likely: bool = False
    review_or_guideline_preferred: bool = False
    entities: list[Entity] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    fused_slots: list[str] = Field(default_factory=list)
    optional_terms: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("missing_slots", "fused_slots", "optional_terms", "reason_codes", mode="before")
    @classmethod
    def _coerce_lists(cls, v: Any) -> list[str]:
        return _str_list(v)

    @field_validator("entities", mode="before")
    @classmethod
    def _cap_entities(cls, v: Any) -> Any:
        return v[:25] if isinstance(v, list) else []

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        return _clamp_conf(v)


# ── Agent 3: evidence curator (surfacing + ranker optimization) ──────
class SurfacedCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # Index into the candidate list the agent was given (NOT a PMID/DOI — the
    # caller maps the index back to the real article id; the model cannot
    # fabricate identifiers).
    index: int = -1
    advisory_label: str = "keep_excluded"
    confidence: float = 0.0
    rationale: str = Field(default="", max_length=300)

    @field_validator("advisory_label", mode="before")
    @classmethod
    def _coerce_label(cls, v: Any) -> str:
        s = str(v or "").strip().lower()
        return s if s in _ADVISORY_LABELS else "keep_excluded"

    @field_validator("index", mode="before")
    @classmethod
    def _coerce_index(cls, v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        return _clamp_conf(v)


class EvidenceCurationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    surfaced: list[SurfacedCandidate] = Field(default_factory=list)
    broader_guideline_topics: list[str] = Field(default_factory=list)
    parallel_ai_search_recommended: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("broader_guideline_topics", "reason_codes", mode="before")
    @classmethod
    def _coerce_lists(cls, v: Any) -> list[str]:
        return _str_list(v)

    @field_validator("surfaced", mode="before")
    @classmethod
    def _cap_surfaced(cls, v: Any) -> Any:
        return v[:60] if isinstance(v, list) else []

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        return _clamp_conf(v)
