"""Pydantic model for the parsed clinical question.

The 11-value clinical_question_type Literal is the single source of truth and
will reject any non-canonical value at validation time. The parser must
canonicalize before constructing this model.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ClinicalQuestionType = Literal[
    "treatment",
    "diagnosis",
    "prognosis",
    "harm",
    "screening",
    "prevention",
    "etiology",
    "guideline",
    "epidemiology",
    "pharmacology",
    "other",
]

CLINICAL_QUESTION_TYPES: tuple[str, ...] = (
    "treatment", "diagnosis", "prognosis", "harm", "screening",
    "prevention", "etiology", "guideline", "epidemiology",
    "pharmacology", "other",
)


class ParsedPICO(BaseModel):
    model_config = ConfigDict(extra="ignore")

    normalized_question: str
    clinical_question_type: ClinicalQuestionType
    specialty: str = ""
    population: str = ""
    intervention_or_exposure: str = ""
    comparator: str = ""
    outcomes: list[str] = Field(default_factory=list)
    setting: str = ""
    time_horizon: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    drug_names_and_classes: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    rationale: str = ""
