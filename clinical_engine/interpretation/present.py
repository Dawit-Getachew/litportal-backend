"""What the clinician sees and edits: the one-line interpretation and chips."""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from clinical_engine.contracts.question import (
    AGE_GROUPS,
    ClinicalQuestionMap,
    QuestionElement,
    framework_for,
)
from clinical_engine.interpretation.model_reading import QUESTION_TYPES, ROLES
from clinical_engine.interpretation.rules import _SETTINGS  # shared setting lexicon


class EngineInputError(ValueError):
    """A client-side problem with an edit or question (maps to HTTP 400/422)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


TYPE_LABELS = {
    "treatment": "Treatment effect", "prevention": "Prevention", "harm": "Harms / adverse effects",
    "etiology_risk": "Risk / cause", "diagnosis": "Diagnostic accuracy", "screening": "Screening",
    "prognosis": "Prognosis", "epidemiology": "How common (incidence / prevalence)",
    "clinical_management": "Clinical management", "background": "Background", "other": "Other",
}
AGE_LABELS = {
    "neonate": "Neonates", "infant": "Infants", "child": "Children", "adolescent": "Adolescents",
    "pediatric": "Pediatric (under 18)", "adult": "Adults", "older_adult": "Older adults (65+)",
    "all_ages": "All ages",
}
_AGE_NOUNS = {
    "neonate": "neonates", "infant": "infants", "child": "children", "adolescent": "adolescents",
    "pediatric": "children", "adult": "adults", "older_adult": "older adults", "all_ages": "patients",
}
_ROLE_LABELS = {
    "condition": "Condition", "population": "Population", "age_group": "Age group", "sex": "Sex",
    "pregnancy": "Pregnancy", "intervention": "Intervention", "exposure": "Exposure",
    "index_test": "Test", "comparator": "Compared with", "reference_standard": "Reference standard",
    "prognostic_factor": "Prognostic factor", "outcome": "Outcome", "setting": "Setting",
    "time_frame": "Time frame", "study_design": "Study design", "publication_window": "Publication dates",
}
# Chip order in the card.
ROLE_ORDER = [
    "intervention", "exposure", "index_test", "comparator", "reference_standard", "outcome",
    "prognostic_factor", "condition", "population", "age_group", "sex", "pregnancy", "setting",
    "time_frame", "study_design", "publication_window",
]
# Types where the patients' condition may also be the outcome of interest.
_CONDITION_IS_OUTCOME_OK = {"prevention", "harm", "etiology_risk", "screening", "epidemiology"}


def role_label(role: str, question_type: str) -> str:
    if role == "intervention" and question_type in ("treatment", "prevention", "clinical_management"):
        return "Treatment"
    if role == "condition" and question_type == "diagnosis":
        return "Target condition"
    return _ROLE_LABELS.get(role, role.replace("_", " ").title())


class ChipOption(BaseModel):
    value: str
    label: str


class Chip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    label: str
    value: Optional[str] = None
    display: str
    source: Literal["user_text", "user_edited", "ai_inferred", "none"]
    state: Literal["active", "suggestion", "excluded", "placeholder"]
    importance: Optional[str] = None
    options: list[ChipOption] = Field(default_factory=list)


def _display(e: QuestionElement) -> str:
    if e.role == "age_group":
        return AGE_LABELS.get(e.value, e.value)
    return e.value


def chips(cqm: ClinicalQuestionMap) -> list[Chip]:
    out = [Chip(
        id="question_type", role="question_type", label="Question type", value=cqm.question_type,
        display=TYPE_LABELS.get(cqm.question_type, cqm.question_type),
        source=cqm.question_type_source, state="active",
        options=[ChipOption(value=t, label=TYPE_LABELS[t]) for t in QUESTION_TYPES],
    )]
    ordered = sorted(cqm.elements, key=lambda e: (ROLE_ORDER.index(e.role) if e.role in ROLE_ORDER else 99))
    has_age = False
    for e in ordered:
        state = "excluded" if e.negated else ("active" if e.active else "suggestion")
        if e.role == "age_group" and e.active and not e.negated:
            has_age = True
        out.append(Chip(
            id=e.id, role=e.role, label=role_label(e.role, cqm.question_type), value=e.value,
            display=("Not " if e.negated else "") + _display(e),
            source=e.source, state=state, importance=e.importance,
            options=[ChipOption(value=a, label=AGE_LABELS[a]) for a in AGE_GROUPS] if e.role == "age_group" else [],
        ))
    if not has_age:
        out.append(Chip(
            id="add:age_group", role="age_group", label="Age group", value=None,
            display="Not specified", source="none", state="placeholder",
            options=[ChipOption(value=a, label=AGE_LABELS[a]) for a in AGE_GROUPS],
        ))
    return out


def _join(values: list[str]) -> str:
    values = [v for v in values if v]
    if len(values) <= 1:
        return "".join(values)
    return ", ".join(values[:-1]) + " and " + values[-1]


_PEOPLE_NOUN = re.compile(
    r"(?i)\b(patients?|people|persons?|adults?|children|child|infants?|neonates?|newborns?|women|men|"
    r"adolescents?|smokers|individuals|residents|workers|athletes|survivors|elderly|recipients|users)\b"
)
_AGE_WORD = re.compile(r"(?i)\b(adults?|child|children|infants?|neonat\w*|newborns?|elderly|older|"
                       r"p(?:a)?ediatric|adolescents?|teen\w*|geriatric)\b")
_SCREEN_FOR = re.compile(r"(?i)^screening\s+for\s+(?P<target>.+)$")
_SCREEN_WORD = re.compile(r"(?i)^screening\s+(?:with|by|using|via)\s+|\s+screening$")


_SETTING_PHRASES = {
    "intensive care unit": "in the intensive care unit", "emergency department": "in the emergency department",
    "primary care": "in primary care", "outpatient": "in outpatient care", "inpatient": "in hospital",
    "prehospital": "in the prehospital setting", "perioperative": "in the perioperative period",
    "nursing home": "in nursing homes",
}
_SETTING_PATTERNS = dict(_SETTINGS)


def population_phrase(
    cqm: ClinicalQuestionMap, *, with_conditions: bool = True, exclude: frozenset[str] = frozenset(),
) -> str:
    ages = cqm.active("age_group")
    age_noun = _AGE_NOUNS.get(ages[0].value) if ages else None
    outcomes = {e.value.lower() for e in cqm.active("outcome")}
    conditions = [e.value for e in cqm.active("condition")
                  if not (cqm.question_type in _CONDITION_IS_OUTCOME_OK and e.value.lower() in outcomes)
                  and e.value.lower() not in exclude]
    descs = [e.value for e in cqm.active("population")]
    if descs:
        head = _join(descs)
        if not _PEOPLE_NOUN.search(head):
            head = f"{head} {age_noun or 'patients'}"
        elif age_noun and not _AGE_WORD.search(head):
            head = f"{head} ({age_noun})"
    else:
        head = age_noun or "patients"
    for e in cqm.active("pregnancy"):
        head = f"{head} ({e.value})"
    shown = [c for c in conditions if c.lower() not in head.lower()] if with_conditions else []
    joiner = "and" if " with " in f" {head.lower()} " else "with"
    phrase = f"{head} {joiner} {_join(shown)}" if shown else head
    excluded = [_display(e).lower() for e in cqm.elements if e.negated]
    if excluded:
        phrase += f" (excluding {_join(excluded)})"
    # Skip a setting the patient description already says ("ICU patients", "hospitalized adults").
    settings = [e.value for e in cqm.active("setting")
                if not (e.value in _SETTING_PATTERNS and re.search(_SETTING_PATTERNS[e.value], phrase, re.I))]
    if len(settings) == 1:
        phrase += " " + _SETTING_PHRASES.get(settings[0].lower(), f"in the {settings[0]}")
    elif settings:
        phrase += f" ({_join(settings)})"
    return phrase


def build_summary(cqm: ClinicalQuestionMap) -> str:
    t = cqm.question_type
    ints = _join([e.value for e in cqm.active("intervention")])
    exps = _join([e.value for e in cqm.active("exposure")]) or ints
    tests = _join([e.value for e in cqm.active("index_test")])
    comps = _join([e.value for e in cqm.active("comparator")])
    outs = _join([e.value for e in cqm.active("outcome")])
    conds = _join([e.value for e in cqm.active("condition")])
    factors = _join([e.value for e in cqm.active("prognostic_factor")])
    refs = _join([e.value for e in cqm.active("reference_standard")])
    pop = population_phrase(cqm)
    vs = f" compared with {comps}" if comps else ""
    time = _join([e.value for e in cqm.active("time_frame")
                  if e.value.lower() not in outs.lower()])
    time_part = f" ({time})" if time else ""
    if t == "treatment":
        s = f"Effect of {ints or 'the treatment'}{vs} on {outs or 'clinical outcomes'}{time_part} in {pop}"
    elif t == "prevention":
        s = f"Prevention of {outs or conds or 'the outcome'}{time_part} with {ints or 'the intervention'}{vs} in {pop}"
    elif t == "harm":
        s = f"Harms of {exps or 'the exposure'}{vs}" + (f" — {outs}" if outs else "") + f" in {pop}"
    elif t == "etiology_risk":
        s = f"Association between {exps or 'the exposure'} and {outs or conds or 'the outcome'} in {pop}"
    elif t == "diagnosis":
        s = f"Accuracy of {tests or 'the test'} for diagnosing {conds or 'the condition'}"
        s += f" in {population_phrase(cqm, with_conditions=False)}"
        if refs:
            s += f" (reference standard: {refs})"
    elif t == "screening":
        # "screening for diabetic retinopathy" names what is screened for, not the test.
        targets, tools = [], []
        for e in cqm.active("index_test"):
            m = _SCREEN_FOR.match(e.value)
            if m:
                targets.append(m.group("target"))
            elif (tool := _SCREEN_WORD.sub("", e.value).strip()) and tool.lower() != "screening":
                tools.append(tool)
        screened = frozenset(v.lower() for v in targets)
        gains = _join([e.value for e in cqm.active("outcome") if e.value.lower() not in screened])
        s = f"Whether screening {population_phrase(cqm, exclude=screened)}"
        s += (f" for {_join(targets)}" if targets else "") + (f" with {_join(tools)}" if tools else "")
        s += f" improves {gains or 'outcomes'}"
    elif t == "prognosis":
        s = "Prognosis" + (f" ({outs})" if outs else "") + f" of {pop}" + (f" by {factors}" if factors else "")
    elif t == "epidemiology":
        s = f"How common {conds or outs or 'the condition'} is in {population_phrase(cqm, with_conditions=False)}"
    elif t == "clinical_management":
        s = f"How to manage {conds or 'the condition'}"
        if ints:
            s += f" with {ints}"
        if cqm.active("age_group") or cqm.active("population") or cqm.active("setting"):
            s += f" in {population_phrase(cqm, with_conditions=False)}"
    elif t == "background":
        s = f"Background on {conds or ints or exps or tests or cqm.question_text.rstrip('?')}"
    else:
        s = cqm.question_text.rstrip("?")
    s = " ".join(s.split())
    return (s[0].upper() + s[1:] + ".")[:300]


# ── Edits ────────────────────────────────────────────────────────────────────


class ChipEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["set", "add", "remove", "accept", "set_type", "set_importance", "exclude"]
    element_id: Optional[str] = None
    role: Optional[str] = None
    value: Optional[str] = Field(default=None, max_length=120)
    importance: Optional[Literal["critical", "important", "unspecified"]] = None


def _clean_value(role: str, value: Optional[str]) -> str:
    v = " ".join((value or "").split()).strip(" .,;:")
    if not v:
        raise EngineInputError("empty_value", "A chip value cannot be empty.")
    if role == "age_group" and v not in AGE_GROUPS:
        raise EngineInputError("invalid_age_group", f"Age group must be one of {', '.join(AGE_GROUPS)}.")
    return v


def apply_edits(cqm: ClinicalQuestionMap, edits: list[ChipEdit]) -> ClinicalQuestionMap:
    if not edits:
        raise EngineInputError("no_edits", "No edits were provided.")
    if len(edits) > 30:
        raise EngineInputError("too_many_edits", "Too many edits in one request.")
    new = cqm.model_copy(deep=True)
    by_id = {e.id: e for e in new.elements}
    for edit in edits:
        if edit.op == "set_type":
            if edit.value not in QUESTION_TYPES:
                raise EngineInputError("invalid_question_type", "Unknown question type.")
            new.question_type = edit.value  # type: ignore[assignment]
            new.question_type_source = "user_edited"
            new.framework = framework_for(edit.value)  # type: ignore[assignment]
            continue
        if edit.op == "add":
            if edit.role not in ROLES:
                raise EngineInputError("invalid_role", "Unknown chip role.")
            value = _clean_value(edit.role, edit.value)
            if edit.role == "age_group":
                new.elements = [e for e in new.elements if not (e.role == "age_group" and not e.negated)]
            if any(e.role == edit.role and e.value.lower() == value.lower() and e.active for e in new.elements):
                continue
            element = QuestionElement(role=edit.role, value=value, source="user_edited", active=True)
            new.elements.append(element)
            by_id[element.id] = element
            continue
        element = by_id.get(edit.element_id or "")
        if element is None or element not in new.elements:
            raise EngineInputError("unknown_element", "That chip no longer exists — refresh and try again.")
        if edit.op == "remove":
            new.elements = [e for e in new.elements if e.id != element.id]
        elif edit.op == "accept":
            element.active = True
            element.source = "user_edited"
        elif edit.op == "exclude":
            element.negated = True
            element.active = True
            element.source = "user_edited"
        elif edit.op == "set":
            element.value = _clean_value(element.role, edit.value)
            element.source = "user_edited"
            element.active = True
            element.span_text = None
        elif edit.op == "set_importance":
            if element.role != "outcome":
                raise EngineInputError("importance_outcome_only", "Importance applies to outcomes only.")
            element.importance = edit.importance
            element.source = "user_edited"
    new.summary = build_summary(new)
    return new
