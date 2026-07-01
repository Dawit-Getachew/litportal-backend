"""Phase 5A — PICOTT (PICO + Time horizon + Type-of-question) Pydantic model.

Coexists with the legacy ``models/pico.ParsedPICO``. The parser populates
BOTH shapes per session for backward compatibility with the existing UI,
summarizer, and ranker, while the new search relevance pipeline uses this
richer PICOTT shape.

Legacy↔new ``type_of_question`` mapping is in
``services/study_type_router.py`` (single source of truth).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

QuestionScope = Literal["background", "foreground", "hybrid", "unclear"]

# New, richer type-of-question enum. Coexists with the legacy 11-value
# ClinicalQuestionType. The two are bidirectionally mapped in
# ``services/study_type_router.py``.
TypeOfQuestion = Literal[
    "therapy",
    "diagnosis",
    "prognosis",
    "etiology_harm",
    "prevention",
    "clinical_exam",
    "cost",
    "pharmacology",
    "dosing_protocol",
    "guideline",
    "epidemiology",
    "background",
    "other",
]

TYPE_OF_QUESTION_VALUES: tuple[str, ...] = (
    "therapy", "diagnosis", "prognosis", "etiology_harm",
    "prevention", "clinical_exam", "cost", "pharmacology",
    "dosing_protocol", "guideline", "epidemiology", "background", "other",
)

# Roles a must-have/should-have/ranking concept can play. Used by the
# ranker to apply per-role anchor gates (e.g., missing intervention →
# capped at 30 for a dosing question).
ConceptRole = Literal[
    "intervention",           # drug, procedure, exposure, index test
    "comparator",             # comparator drug/procedure/reference standard
    "condition",              # target disease/condition
    "population",             # population/sub-population (geriatric, ICU, etc.)
    "outcome",                # primary/secondary outcome or target diagnosis
    "study_design",           # rct, cohort, etc.
    "context",                # dosing words: dose, infusion, mg/kg, mcg/kg/min
    "setting",                # ICU, outpatient, ED, inpatient
    "time_horizon",
    "other",
]


class SpellingCorrection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    original: str
    normalized: str
    confidence: float = 1.0
    reason: str = ""


class PICOBlocks(BaseModel):
    """Concept blocks. Each list contains short, canonical phrases."""

    model_config = ConfigDict(extra="ignore")
    population: list[str] = Field(default_factory=list)
    intervention_or_exposure_or_index_test: list[str] = Field(default_factory=list)
    comparator_or_reference_standard: list[str] = Field(default_factory=list)
    outcomes_or_target_diagnosis: list[str] = Field(default_factory=list)
    time_horizon: list[str] = Field(default_factory=list)
    setting: list[str] = Field(default_factory=list)


class MustHaveConcept(BaseModel):
    model_config = ConfigDict(extra="ignore")
    concept: str
    role: ConceptRole
    synonyms: list[str] = Field(default_factory=list)
    # If true, an article missing this concept (and its synonyms) cannot be
    # high-relevance and gets the ``missing_relevance_cap`` ceiling applied.
    required_for_high_relevance: bool = True
    # Hard-ceiling cap on the article's final relevance score when this
    # concept is absent from title + abstract + ingestible OA text.
    missing_relevance_cap: int = 30


class ShouldHaveConcept(BaseModel):
    model_config = ConfigDict(extra="ignore")
    concept: str
    role: ConceptRole
    synonyms: list[str] = Field(default_factory=list)
    # Soft signal: absent should-have concepts do NOT cap, they only reduce
    # the soft relevance score.
    boost_weight: int = 2


class RankingConcept(BaseModel):
    model_config = ConfigDict(extra="ignore")
    concept: str
    role: ConceptRole = "context"
    synonyms: list[str] = Field(default_factory=list)
    # Bonus added to relevance when matched. Cap-aware: never overrides caps.
    boost_weight: int = 5


class NegativeConcept(BaseModel):
    model_config = ConfigDict(extra="ignore")
    concept: str
    penalty_reason: str
    # If any of these concepts ARE present in the article, the penalty is
    # waived (e.g., "CTA" doesn't trigger a penalty if "milrinone" is also
    # present in a milrinone/SAH dosing query — the article actually treats
    # both topics).
    unless_concepts_present: list[str] = Field(default_factory=list)
    # Hard-ceiling penalty cap. When the negative concept fires the article
    # cannot score above this. Defaults to 50.
    penalty_cap: int = 50


class PICOTT(BaseModel):
    """Phase 5A — Parsed PICOTT representation.

    Stored on ``search_sessions.picott_json``. The legacy ``ParsedPICO`` is
    still stored on ``search_sessions.pico_json`` for BC.
    """

    model_config = ConfigDict(extra="ignore")

    question_scope: QuestionScope = "foreground"
    normalized_question: str = ""
    original_question: str = ""
    spelling_corrections: list[SpellingCorrection] = Field(default_factory=list)
    pico: PICOBlocks = Field(default_factory=PICOBlocks)

    type_of_question: TypeOfQuestion = "other"
    question_subtype: Optional[str] = None
    specialty: str = ""

    # Preferred + acceptable study types from the StudyTypeRouter.
    ideal_study_types: list[str] = Field(default_factory=list)

    must_have_concepts: list[MustHaveConcept] = Field(default_factory=list)
    should_have_concepts: list[ShouldHaveConcept] = Field(default_factory=list)
    ranking_concepts: list[RankingConcept] = Field(default_factory=list)
    negative_or_penalty_concepts: list[NegativeConcept] = Field(default_factory=list)

    # PICO completeness flags — e.g., {"population": "missing", "comparator": "implicit"}.
    missing_elements: dict[str, str] = Field(default_factory=dict)

    # Two short user-facing strings.
    interpretation_for_user: str = ""
    rationale: str = ""

    # 0.0 — 1.0; the parser's self-reported confidence in the structured
    # interpretation (used by the UI to render a "review interpretation" hint
    # when low).
    confidence: float = 0.0

    # Phase 5B — Clinical concept translation outputs (see
    # services/clinical_concept_translation.py).
    translated_question: str = ""
    concept_translations: list[dict] = Field(default_factory=list)
    latent_concepts: list[str] = Field(default_factory=list)
    translation_warnings: list[str] = Field(default_factory=list)
