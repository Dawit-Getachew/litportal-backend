"""ClinicalQuestionMap — the one authoritative interpretation of a question (W1.3/W1.4).

Every element records where it came from. Only elements the clinician stated
(``user_text``) or set themselves (``user_edited``) are active by default;
anything the model inferred is an inactive *suggestion* until the clinician
accepts it, so nothing — in particular an age group — is ever applied silently.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from models.common import new_id

SCHEMA_VERSION = "cqm.v1"

QuestionType = Literal[
    "treatment",            # does an intervention work (PICO)
    "prevention",           # does an intervention prevent a condition (PICO)
    "harm",                 # adverse effects of an exposure/intervention (PECO)
    "etiology_risk",        # does an exposure raise risk (PECO / PEO)
    "diagnosis",            # test accuracy (PIRD)
    "screening",            # should a population be screened (PICO)
    "prognosis",            # likely course / prognostic factors (PFO)
    "epidemiology",         # incidence / prevalence (CoCoPop)
    "clinical_management",  # how to manage a condition, no single intervention
    "background",           # definitions, mechanisms
    "other",
]

Framework = Literal["PICO", "PECO", "PEO", "PIRD", "PFO", "CoCoPop", "open"]

FRAMEWORK_FOR_TYPE: dict[str, str] = {
    "treatment": "PICO",
    "prevention": "PICO",
    "screening": "PICO",
    "harm": "PECO",
    "etiology_risk": "PEO",
    "diagnosis": "PIRD",
    "prognosis": "PFO",
    "epidemiology": "CoCoPop",
    "clinical_management": "open",
    "background": "open",
    "other": "open",
}

ElementRole = Literal[
    "condition",            # the clinical problem the population has / target condition
    "population",           # population descriptor beyond the condition (e.g. mechanically ventilated)
    "age_group",
    "sex",
    "pregnancy",
    "intervention",
    "exposure",
    "index_test",
    "comparator",
    "reference_standard",
    "prognostic_factor",
    "outcome",
    "setting",
    "time_frame",
    "study_design",
    "publication_window",
]

AgeGroup = Literal[
    "neonate", "infant", "child", "adolescent", "pediatric",
    "adult", "older_adult", "all_ages",
]
AGE_GROUPS: tuple[str, ...] = (
    "neonate", "infant", "child", "adolescent", "pediatric", "adult", "older_adult", "all_ages",
)

ElementSource = Literal[
    "user_text",    # stated in the question (span recorded)
    "user_edited",  # set or changed by the clinician through a chip
    "ai_inferred",  # suggested by the model, NOT stated — inactive until accepted
]

OutcomeImportance = Literal["critical", "important", "unspecified"]

# Roles that restrict what evidence is eligible. An inferred value for these
# must never become active without the clinician accepting it.
RESTRICTIVE_ROLES: frozenset[str] = frozenset(
    {"age_group", "sex", "pregnancy", "setting", "publication_window", "study_design"}
)

# Roles whose change alters what is being asked (a "material" edit).
MATERIAL_ROLES: frozenset[str] = frozenset(
    {
        "condition", "population", "age_group", "sex", "pregnancy", "intervention",
        "exposure", "index_test", "comparator", "reference_standard",
        "prognostic_factor", "outcome", "setting", "time_frame", "publication_window",
        "study_design",
    }
)


class QuestionElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    role: ElementRole
    value: str = Field(min_length=1, max_length=200)
    source: ElementSource
    active: bool = True
    # Supporting substring of the question for ``user_text`` elements.
    span_text: Optional[str] = Field(default=None, max_length=300)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: Optional[OutcomeImportance] = None  # outcomes only
    negated: bool = False  # e.g. "excluding children"


class IntentConflict(BaseModel):
    """A contradiction in the interpretation. Blocking conflicts stop approval."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["blocking", "warning"]
    message: str
    roles: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)


class ClinicalQuestionMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["cqm.v1"] = SCHEMA_VERSION
    question_text: str = Field(min_length=1, max_length=2000)
    question_type: QuestionType
    question_type_source: ElementSource = "ai_inferred"
    framework: Framework
    elements: list[QuestionElement] = Field(default_factory=list)
    # One short sentence shown to the clinician. Built from active elements only.
    summary: str = ""
    ambiguities: list[str] = Field(default_factory=list)

    def active(self, role: str | None = None) -> list[QuestionElement]:
        return [
            e for e in self.elements
            if e.active and not e.negated and (role is None or e.role == role)
        ]

    def suggestions(self) -> list[QuestionElement]:
        return [e for e in self.elements if not e.active]

    def age_group(self) -> str:
        """The active age group, or ``not_stated`` — never an implied default."""
        groups = self.active("age_group")
        return groups[0].value if groups else "not_stated"


def material_fingerprint(cqm: ClinicalQuestionMap) -> dict:
    """The parts of a map that define *what is being asked*."""
    return {
        "question_type": cqm.question_type,
        "framework": cqm.framework,
        "elements": sorted(
            (
                e.role,
                " ".join(e.value.lower().split()),
                bool(e.negated),
                e.importance or "",
            )
            for e in cqm.elements
            if e.active and e.role in MATERIAL_ROLES
        ),
    }


def material_hash(cqm: ClinicalQuestionMap) -> str:
    raw = json.dumps(material_fingerprint(cqm), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def framework_for(question_type: str) -> str:
    return FRAMEWORK_FOR_TYPE.get(question_type, "open")
