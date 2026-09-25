"""interpret(): question text → one ClinicalQuestionMap (+ conflicts, warnings, meta).

PHI preflight first (nothing is interpreted, stored or sent to a model if it
trips). Then the deterministic reading, then the model reading (if configured
and within budget) verified against the text, merged conservatively.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from clinical_engine.config import settings
from clinical_engine.contracts.question import ClinicalQuestionMap, IntentConflict, QuestionElement, framework_for
from clinical_engine.contracts.run import InterpretationMeta
from clinical_engine.gateway import ModelGateway
from clinical_engine.interpretation import rules
from clinical_engine.interpretation.consistency import check
from clinical_engine.interpretation.model_reading import SYSTEM_PROMPT, ModelReading, validate_model_output
from clinical_engine.interpretation.present import EngineInputError, build_summary

# HIPAA treats ages over 89 as identifying; the shared guard does not flag them.
_AGE_OVER_89 = re.compile(r"\b(?:9\d|1[0-2]\d)[- ]?(?:years?|yrs?|yo)[- ]?(?:old)?\b|\baged? (?:9\d|1[0-2]\d)\b", re.I)
_CORE_ROLES = ("condition", "population", "intervention", "exposure", "index_test", "comparator",
               "reference_standard", "outcome", "prognostic_factor")
_RULE_ONLY_ROLES = ("setting", "time_frame", "publication_window")
_OUTCOME_IS_THE_CONDITION = {"prevention", "etiology_risk", "screening", "epidemiology", "harm"}


@dataclass
class PHIResult:
    blocked: bool
    reason_codes: list[str] = field(default_factory=list)


def phi_preflight(text: str) -> PHIResult:
    from services.phi_guard import phi_guard  # reuse-as-is (see reuse register)

    result = phi_guard.screen(text)
    codes = list(result.reason_codes)
    if _AGE_OVER_89.search(text):
        codes.append("age_over_89_detected")
    return PHIResult(blocked=bool(codes), reason_codes=codes)


@dataclass
class Interpretation:
    map: ClinicalQuestionMap
    conflicts: list[IntentConflict]
    warnings: list[str]
    meta: InterpretationMeta


def _key(e: QuestionElement) -> tuple[str, str]:
    return e.role, " ".join(e.value.lower().split())


def merge(reading: rules.RulesReading, model: Optional[ModelReading]) -> tuple[str, list[QuestionElement], list[str]]:
    """Model elements first; the deterministic reading fills gaps and is
    authoritative for age (it detects negation) and for setting/time spans."""
    if model is None or (model.question_type is None and not model.elements):
        return reading.question_type, list(reading.elements), []

    ambiguities = list(model.ambiguities)
    qtype = model.question_type or reading.question_type
    if qtype in ("background", "other") and reading.question_type not in ("background", "other"):
        ambiguities.append(f"Question type was unclear; using '{reading.question_type}'.")
        qtype = reading.question_type

    if qtype in ("treatment", "prevention") and not any(
        e.active and e.role == "intervention" for e in model.elements
    ) and not any(e.role == "intervention" for e in reading.elements):
        # "What is the first-line treatment for X?" names no intervention to evaluate.
        ambiguities.append("No specific intervention named; reading this as a management question.")
        qtype = "clinical_management"

    if qtype in ("diagnosis", "screening") and not any(e.active and e.role == "index_test" for e in model.elements):
        for e in model.elements:
            if e.role == "intervention" and e.active:
                e.role = "index_test"  # the model named the test as an "intervention"

    rule_ages = [e for e in reading.elements if e.role == "age_group"]
    merged = [e for e in model.elements if not (rule_ages and e.role == "age_group")]
    merged.extend(rule_ages)
    active_roles = {e.role for e in merged if e.active}
    for e in reading.elements:
        if e.role in _RULE_ONLY_ROLES and e.role not in active_roles:
            merged.append(e)
        elif e.role in _CORE_ROLES and e.role not in active_roles:
            merged.append(e)
            active_roles.add(e.role)

    seen: set[tuple[str, str]] = set()
    unique: list[QuestionElement] = []
    # Prefer active/stated over suggestions when both exist.
    for e in sorted(merged, key=lambda x: (not x.active, x.source != "user_text")):
        k = _key(e)
        if k in seen:
            continue
        seen.add(k)
        unique.append(e)
    if qtype in _OUTCOME_IS_THE_CONDITION:
        # "Does X prevent delirium?": delirium is the outcome; the patients do not have it yet.
        outcomes = {_key(e)[1] for e in unique if e.role == "outcome" and e.active}
        unique = [e for e in unique if not (e.role == "condition" and _key(e)[1] in outcomes)]
    return qtype, unique, ambiguities


def _user_prompt(question: str) -> str:
    return "Question:\n" + json.dumps(question)


async def interpret(
    question: str, *, gateway: Optional[ModelGateway] = None, allow_model: bool = True,
) -> Interpretation:
    text = " ".join((question or "").split())
    if not text:
        raise EngineInputError("empty_question", "Type a clinical question.")
    if len(text) > settings.MAX_QUESTION_CHARS:
        raise EngineInputError(
            "question_too_long", f"Keep the question under {settings.MAX_QUESTION_CHARS} characters.",
        )

    reading = rules.read(text)
    model_reading: Optional[ModelReading] = None
    meta = InterpretationMeta(method="rules_only")
    gateway = gateway if gateway is not None else ModelGateway()
    if not allow_model:
        meta.fallback_reason = "model_budget_exhausted"
    elif not gateway.is_configured():
        meta.fallback_reason = "model_not_configured"
    else:
        result = await gateway.complete_json(
            SYSTEM_PROMPT, _user_prompt(text),
            max_tokens=settings.INTERPRETER_MAX_TOKENS, timeout_s=settings.INTERPRETER_TIMEOUT_S,
        )
        meta.model = result.model
        meta.tokens_in, meta.tokens_out, meta.latency_ms = result.tokens_in, result.tokens_out, result.latency_ms
        if result.data is None:
            meta.fallback_reason = f"model_{result.error}"
        else:
            model_reading = validate_model_output(text, result.data)
            meta.rejected_model_fields = model_reading.rejected[:20]
            if model_reading.question_type is None and not model_reading.elements:
                meta.fallback_reason = "model_output_rejected"
                model_reading = None
            else:
                meta.method = "model"

    qtype, elements, ambiguities = merge(reading, model_reading)
    cqm = ClinicalQuestionMap(
        question_text=text,
        question_type=qtype,  # type: ignore[arg-type]
        question_type_source="ai_inferred",
        framework=framework_for(qtype),  # type: ignore[arg-type]
        elements=elements,
        ambiguities=ambiguities[:4],
    )
    cqm.summary = build_summary(cqm)
    conflicts, warnings = check(cqm)
    return Interpretation(map=cqm, conflicts=conflicts, warnings=warnings, meta=meta)
