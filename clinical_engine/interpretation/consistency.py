"""Intent-consistency checks. Blocking conflicts stop approval until the
clinician edits the interpretation — a contradictory reading fails safely
instead of silently driving a search."""
from __future__ import annotations

from clinical_engine.contracts.question import ClinicalQuestionMap, IntentConflict

# What a question type needs before evidence can be searched for it.
REQUIRED_ROLES: dict[str, list[tuple[str, ...]]] = {
    "treatment": [("intervention",), ("condition", "population")],
    "prevention": [("intervention",), ("condition", "population", "outcome")],
    "harm": [("exposure", "intervention")],
    "etiology_risk": [("exposure", "intervention", "prognostic_factor"), ("outcome", "condition")],
    "diagnosis": [("index_test",), ("condition",)],
    "screening": [("condition", "population", "index_test")],
    "prognosis": [("condition", "population")],
    "epidemiology": [("condition",)],
    "clinical_management": [("condition",)],
    "background": [("condition", "intervention", "exposure", "index_test", "population")],
    "other": [],
}
_ROLE_LABELS = {
    "intervention": "the treatment or intervention", "condition": "the condition or patient group",
    "population": "the patient group", "outcome": "the outcome", "exposure": "the exposure",
    "index_test": "the test being evaluated", "prognostic_factor": "the prognostic factor",
}
# Types where the condition may legitimately also be the outcome of interest.
_CONDITION_CAN_BE_OUTCOME = {"prevention", "harm", "etiology_risk", "screening", "epidemiology"}
_INCOMPATIBLE_AGES = [
    ({"neonate", "infant", "child", "pediatric", "adolescent"}, {"adult", "older_adult"}),
]


def _norm(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def check(cqm: ClinicalQuestionMap) -> tuple[list[IntentConflict], list[str]]:
    conflicts: list[IntentConflict] = []
    warnings: list[str] = []
    active = cqm.active()

    for alternatives in REQUIRED_ROLES.get(cqm.question_type, []):
        if not any(e.role in alternatives for e in active):
            label = _ROLE_LABELS.get(alternatives[0], alternatives[0])
            conflicts.append(IntentConflict(
                code="missing_core_element", severity="blocking", roles=list(alternatives),
                message=f"This reads as a {cqm.question_type.replace('_', ' ')} question but "
                        f"{label} is missing. Add it, or change the question type.",
            ))

    interventions = {_norm(e.value): e for e in active if e.role in ("intervention", "exposure", "index_test")}
    for comp in cqm.active("comparator"):
        if _norm(comp.value) in interventions:
            conflicts.append(IntentConflict(
                code="intervention_equals_comparator", severity="blocking",
                roles=["intervention", "comparator"],
                element_ids=[comp.id, interventions[_norm(comp.value)].id],
                message=f"'{comp.value}' is both the intervention and the comparator.",
            ))

    if cqm.question_type not in _CONDITION_CAN_BE_OUTCOME:
        conditions = {_norm(e.value): e for e in cqm.active("condition")}
        for outcome in cqm.active("outcome"):
            if _norm(outcome.value) in conditions:
                conflicts.append(IntentConflict(
                    code="condition_as_outcome", severity="blocking", roles=["condition", "outcome"],
                    element_ids=[outcome.id, conditions[_norm(outcome.value)].id],
                    message=f"'{outcome.value}' is listed as both the patients' condition and the outcome. "
                            "Remove one or name the outcome you care about (e.g. mortality).",
                ))

    ages = {e.value for e in cqm.active("age_group")}
    for young, old in _INCOMPATIBLE_AGES:
        if ages & young and ages & old:
            conflicts.append(IntentConflict(
                code="contradictory_age_groups", severity="blocking", roles=["age_group"],
                message=f"Age groups {sorted(ages)} do not overlap. Keep the one you mean.",
            ))
    negated_ages = {e.value for e in cqm.elements if e.role == "age_group" and e.negated}
    if ages & negated_ages:
        conflicts.append(IntentConflict(
            code="age_group_included_and_excluded", severity="blocking", roles=["age_group"],
            message=f"Age group {sorted(ages & negated_ages)} is both included and excluded.",
        ))

    if not cqm.active("age_group"):
        warnings.append(
            "No age group stated — results will not be limited to adults or children. "
            "Add one if it matters."
        )
    if cqm.question_type in ("other", "background") and not conflicts:
        warnings.append("This reads as a general question; answers may favour reviews and guidelines.")
    return conflicts, warnings
