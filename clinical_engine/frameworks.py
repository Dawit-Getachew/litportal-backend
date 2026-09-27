"""Clinical Question Map V2: the clinician's intent picks the framework (view),
and the framework decides which fields the interpretation card emphasizes.

Source: "Clinical Question Map - V2" (new-scope/new-comments). One row per intent:

  Treatment or prevention effects     PICO (timing/setting when useful)
  Exposure, cause, or harm            PEO/PECO (comparator when relevant)
  Test accuracy                       Diagnostic accuracy (+ reference standard, threshold)
  Diagnostic workup                   Presentation, conditions considered, strategy, decision
  Expected future course              Overall prognosis
  Prognostic factor                   PFO
  Prediction model                    Prediction-model map
  Experience or qualitative evidence  PICo/SPIDER
  Broad evidence mapping              PCC
  Prevalence or incidence             CoCoPop
  Guideline or management question    PIPOH-style view
  Mechanisms, monitoring, complications, or screening
                                      Operational map — only the roles relevant to the intent
  Health services or implementation   SPICE/CIMO
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clinical_engine.contracts.question import ClinicalQuestionMap

_PATIENT_ROLES = ("condition", "population", "age_group", "sex", "pregnancy")
_PEOPLE_ROLES = ("population", "age_group", "sex", "pregnancy")


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    roles: tuple[str, ...]        # element roles shown in this field
    add_roles: tuple[str, ...]    # roles a clinician can add here (first is the default)
    emphasized: bool = True       # shown even when empty


@dataclass(frozen=True)
class Intent:
    id: str
    label: str
    types: tuple[str, ...]


INTENTS: tuple[Intent, ...] = (
    Intent("treatment_prevention", "Treatment or prevention effects", ("treatment", "prevention")),
    Intent("exposure_harm", "Exposure, cause or harm", ("harm", "etiology_risk")),
    Intent("test_accuracy", "Test accuracy", ("diagnosis",)),
    Intent("diagnostic_workup", "Diagnostic workup", ("diagnostic_workup",)),
    Intent("future_course", "Expected future course", ("prognosis",)),
    Intent("prognostic_factor", "Prognostic factor", ("prognostic_factor",)),
    Intent("prediction_model", "Prediction model", ("prediction_model",)),
    Intent("experience", "Experience or qualitative evidence", ("qualitative",)),
    Intent("evidence_mapping", "Broad evidence mapping", ("evidence_map",)),
    Intent("frequency", "Prevalence or incidence", ("epidemiology",)),
    Intent("guideline_management", "Guideline or management question", ("clinical_management",)),
    Intent("operational", "Mechanisms, monitoring, complications or screening",
           ("background", "monitoring", "complications", "screening")),
    Intent("health_services", "Health services or implementation", ("health_services",)),
    Intent("other", "Other", ("other",)),
)
INTENT_FOR_TYPE: dict[str, Intent] = {t: i for i in INTENTS for t in i.types}

TYPE_LABELS: dict[str, str] = {
    "treatment": "Treatment effect", "prevention": "Prevention",
    "harm": "Harms / adverse effects", "etiology_risk": "Risk / cause",
    "diagnosis": "Diagnostic accuracy", "diagnostic_workup": "Diagnostic workup",
    "screening": "Screening", "prognosis": "Overall prognosis", "prognostic_factor": "Prognostic factor",
    "prediction_model": "Prediction model", "qualitative": "Patient experience / qualitative",
    "evidence_map": "Evidence map / scoping", "epidemiology": "How common (incidence / prevalence)",
    "clinical_management": "Guideline / management", "monitoring": "Monitoring",
    "complications": "Complications", "health_services": "Health services / implementation",
    "background": "Mechanism / background", "other": "Other",
}

FRAMEWORK_LABELS: dict[str, str] = {
    "PICO": "PICO", "PEO": "PEO", "PECO": "PECO", "PIRD": "Diagnostic accuracy",
    "DxWorkup": "Diagnostic workup", "Prognosis": "Overall prognosis", "PFO": "PFO",
    "PredictionModel": "Prediction-model map", "PICo": "PICo / SPIDER", "PCC": "PCC",
    "CoCoPop": "CoCoPop", "PIPOH": "PIPOH-style", "Operational": "Operational map",
    "SPICE": "SPICE / CIMO", "open": "Open",
}


def _f(key: str, label: str, roles: tuple[str, ...], add: tuple[str, ...] | None = None,
       emphasized: bool = True) -> FieldSpec:
    return FieldSpec(key, label, roles, add if add is not None else roles[:1], emphasized)


_TIMING = _f("timing", "Timing", ("time_frame",), emphasized=False)
_SETTING = _f("setting", "Setting", ("setting",), emphasized=False)
_DESIGN = _f("design", "Study design", ("study_design",), emphasized=False)
_DATES = _f("dates", "Publication dates", ("publication_window",), emphasized=False)

FRAMEWORK_FIELDS: dict[str, tuple[FieldSpec, ...]] = {
    "PICO": (
        _f("patients", "Patients", _PATIENT_ROLES, ("condition", "population")),
        _f("intervention", "Treatment", ("intervention",)),
        _f("comparison", "Comparison", ("comparator",)),
        _f("outcomes", "Outcomes", ("outcome",)),
        _TIMING, _SETTING,
    ),
    "PEO": (
        _f("population", "Population", _PATIENT_ROLES, ("condition", "population")),
        _f("exposure", "Exposure", ("exposure", "intervention"), ("exposure",)),
        _f("outcome", "Outcome", ("outcome",)),
        _f("comparison", "Comparator", ("comparator",), emphasized=False),
        _TIMING, _SETTING,
    ),
    "PECO": (
        _f("population", "Population", _PATIENT_ROLES, ("condition", "population")),
        _f("exposure", "Exposure", ("exposure", "intervention"), ("exposure",)),
        _f("comparison", "Comparator", ("comparator",)),
        _f("outcome", "Outcome", ("outcome",)),
        _TIMING, _SETTING,
    ),
    "PIRD": (
        _f("patients", "Patients", _PEOPLE_ROLES + ("presentation",), ("population", "presentation")),
        _f("index_test", "Index test", ("index_test",)),
        _f("target", "Target condition", ("condition",)),
        _f("reference", "Reference standard", ("reference_standard",), emphasized=False),
        _f("threshold", "Threshold", ("threshold",), emphasized=False),
        _SETTING,
    ),
    "DxWorkup": (
        _f("presentation", "Presentation", ("presentation",) + _PEOPLE_ROLES, ("presentation", "population")),
        _f("considered", "Conditions under consideration", ("condition",)),
        _f("strategy", "Diagnostic strategy", ("index_test",)),
        _f("decision", "Intended decision", ("decision",)),
        _SETTING,
    ),
    "Prognosis": (
        _f("population", "Population", _PATIENT_ROLES, ("condition", "population")),
        _f("start", "Starting point", ("starting_point",)),
        _f("outcome", "Future outcome", ("outcome",)),
        _f("horizon", "Time horizon", ("time_frame",)),
        _SETTING,
    ),
    "PFO": (
        _f("population", "Population", _PATIENT_ROLES, ("condition", "population")),
        _f("factor", "Prognostic factor", ("prognostic_factor",)),
        _f("reference", "Reference level", ("reference_level",)),
        _f("outcome", "Outcome", ("outcome",)),
        _f("horizon", "Horizon", ("time_frame",)),
    ),
    "PredictionModel": (
        _f("moment", "Prediction moment", ("prediction_moment",)),
        _f("predictors", "Model / predictors", ("prognostic_factor", "index_test"), ("prognostic_factor",)),
        _f("outcome", "Outcome", ("outcome",)),
        _f("horizon", "Horizon", ("time_frame",)),
        _f("phase", "Model phase", ("model_phase",)),
        _f("population", "Population", _PATIENT_ROLES, ("condition", "population"), emphasized=False),
    ),
    "PICo": (
        _f("population", "Population / sample", _PATIENT_ROLES, ("population", "condition")),
        _f("phenomenon", "Phenomenon of interest", ("phenomenon", "intervention", "exposure"), ("phenomenon",)),
        _f("context", "Context", ("setting",)),
        _f("design", "Design", ("study_design",), emphasized=False),
        _f("evaluation", "Evaluation", ("outcome",), emphasized=False),
    ),
    "PCC": (
        _f("population", "Population", _PATIENT_ROLES, ("population", "condition")),
        _f("concept", "Concept", ("concept", "intervention", "exposure"), ("concept",)),
        _f("context", "Context", ("setting",)),
    ),
    "CoCoPop": (
        _f("condition", "Condition", ("condition",)),
        _f("population", "Population", _PEOPLE_ROLES),
        _f("context", "Context", ("setting",)),
        _f("measure", "Frequency measure", ("frequency_measure",)),
        _f("period", "Time period", ("time_frame",)),
    ),
    "PIPOH": (
        _f("population", "Population", _PATIENT_ROLES, ("condition", "population")),
        _f("domain", "Management domain", ("intervention",)),
        _f("setting", "Setting", ("setting",)),
        _f("users", "Intended users", ("intended_users",)),
        _f("jurisdiction", "Jurisdiction", ("jurisdiction",)),
    ),
    "SPICE": (
        _f("setting", "Setting / context", ("setting",)),
        _f("perspective", "Perspective / mechanism", ("perspective",) + _PATIENT_ROLES, ("perspective", "population")),
        _f("intervention", "Intervention", ("intervention", "exposure"), ("intervention",)),
        _f("comparison", "Comparison", ("comparator",), emphasized=False),
        _f("evaluation", "Evaluation / outcomes", ("outcome",)),
    ),
    "open": (
        _f("patients", "Patients", _PATIENT_ROLES, ("condition", "population")),
        _f("topic", "Treatment, exposure or test", ("intervention", "exposure", "index_test"), ("intervention",)),
        _f("outcome", "Outcome", ("outcome",), emphasized=False),
        _SETTING,
    ),
}

# "Show only the roles relevant to that intent; do not force treatment PICO."
OPERATIONAL_FIELDS: dict[str, tuple[FieldSpec, ...]] = {
    "background": (
        _f("topic", "Condition or topic", _PATIENT_ROLES, ("condition",)),
        _f("agent", "Treatment or exposure", ("intervention", "exposure"), ("intervention",), emphasized=False),
        _f("aspect", "Aspect of interest", ("concept", "outcome"), ("concept",)),
    ),
    "monitoring": (
        _f("patients", "Patients", _PATIENT_ROLES, ("condition", "population")),
        _f("monitored", "Treatment being monitored", ("intervention",), emphasized=False),
        _f("test", "Monitoring test", ("index_test",)),
        _f("outcome", "Outcome", ("outcome",)),
        _TIMING,
    ),
    "complications": (
        _f("patients", "Patients", _PATIENT_ROLES, ("condition", "population")),
        _f("procedure", "Procedure or treatment", ("intervention", "exposure"), ("intervention",)),
        _f("complication", "Complication", ("outcome",)),
        _TIMING,
    ),
    "screening": (
        _f("population", "Population", _PEOPLE_ROLES + ("condition",), ("population",)),
        _f("test", "Screening test", ("index_test",)),
        _f("target", "Target condition", ("condition",), emphasized=False),
        _f("outcomes", "Outcomes", ("outcome",)),
        _SETTING,
    ),
}

# Role labels for chips (the field label says where the chip sits).
ROLE_LABELS: dict[str, str] = {
    "condition": "Condition", "population": "Population", "age_group": "Age group", "sex": "Sex",
    "pregnancy": "Pregnancy", "intervention": "Intervention", "exposure": "Exposure",
    "index_test": "Test", "comparator": "Compared with", "reference_standard": "Reference standard",
    "prognostic_factor": "Prognostic factor", "outcome": "Outcome", "setting": "Setting",
    "time_frame": "Time frame", "study_design": "Study design", "publication_window": "Publication dates",
    "presentation": "Presentation", "threshold": "Threshold", "decision": "Intended decision",
    "starting_point": "Starting point", "reference_level": "Reference level",
    "prediction_moment": "Prediction moment", "model_phase": "Model phase",
    "phenomenon": "Phenomenon of interest", "concept": "Concept", "frequency_measure": "Frequency measure",
    "intended_users": "Intended users", "jurisdiction": "Jurisdiction", "perspective": "Perspective",
}


def fields_for(framework: str, question_type: str) -> tuple[FieldSpec, ...]:
    if framework == "Operational":
        specs = OPERATIONAL_FIELDS.get(question_type, OPERATIONAL_FIELDS["background"])
    else:
        specs = FRAMEWORK_FIELDS.get(framework, FRAMEWORK_FIELDS["open"])
    # Study design and publication dates refine any question (shown only when set).
    covered = {r for spec in specs for r in spec.roles}
    return specs + tuple(extra for extra in (_DESIGN, _DATES) if extra.roles[0] not in covered)


def field_for_role(framework: str, question_type: str, role: str) -> FieldSpec | None:
    for spec in fields_for(framework, question_type):
        if role in spec.roles:
            return spec
    return None


def role_label(role: str, framework: str, question_type: str) -> str:
    if role == "intervention" and question_type in ("treatment", "prevention"):
        return "Treatment" if question_type == "treatment" else "Intervention"
    if role == "condition" and framework == "PIRD":
        return "Target condition"
    spec = field_for_role(framework, question_type, role)
    if spec is not None and spec.roles[0] == role and role not in ("condition", "population"):
        return spec.label
    return ROLE_LABELS.get(role, role.replace("_", " ").title())


def field_view(cqm: ClinicalQuestionMap, *, age_placeholder: bool) -> list[dict[str, Any]]:
    """The framework's fields in order, each with the ids of the chips that belong
    to it (chip id == element id; ``add:age_group`` is the age placeholder).
    Elements whose role the framework does not show land in "Other details", so
    nothing the clinician stated is ever hidden."""
    placed: set[str] = set()
    out: list[dict[str, Any]] = []
    age_pending = age_placeholder
    for spec in fields_for(cqm.framework, cqm.question_type):
        ids = [e.id for e in cqm.elements if e.role in spec.roles and e.id not in placed]
        placed.update(ids)
        if age_pending and "age_group" in spec.roles:
            ids.append("add:age_group")
            age_pending = False
        out.append({
            "key": spec.key, "label": spec.label, "emphasized": spec.emphasized,
            "roles": list(spec.roles), "add_roles": list(spec.add_roles), "chip_ids": ids,
        })
    leftovers = [e.id for e in cqm.elements if e.id not in placed]
    if age_pending:
        leftovers.append("add:age_group")
    if leftovers:
        out.append({"key": "other", "label": "Other details", "emphasized": False,
                    "roles": [], "add_roles": [], "chip_ids": leftovers})
    return out


def type_options() -> list[dict[str, str]]:
    """Question-type choices grouped by V2 intent, with the framework each uses."""
    from clinical_engine.contracts.question import FRAMEWORK_FOR_TYPE

    return [
        {"value": t, "label": TYPE_LABELS[t], "intent": i.label,
         "framework": FRAMEWORK_LABELS.get(FRAMEWORK_FOR_TYPE.get(t, "open"), "Open")}
        for i in INTENTS for t in i.types
    ]
