"""AI Agent 2 — Question Understanding / ContextAugment (scope B3).

Runs after the deterministic parse + clinical-context repair, BEFORE the Boolean
compiler. It produces a structured *understanding* of the clinical question
(intent, whether it's broad, whether a review/guideline is preferred, entities,
missing/fused PICO slots, optional terms) and stores it as diagnostics.

Governance: STORAGE-ONLY. It does NOT feed the planner, compiler, ranker, or
selection — deterministic retrieval/ranking is byte-unchanged. Feature-flagged
(default OFF), fail-closed. No source calls, no citations.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agents.base import run_json_agent, store_agent_decision
from agents.schemas import QuestionUnderstanding
from core.config import get_settings
from services.audit import audit_service

logger = logging.getLogger("litportal.agents.question_interpreter")

PROMPT_VERSION = "qunderstand-v1"

_SYSTEM = (
    "You are a clinical-question INTERPRETER for a medical evidence search engine. "
    "You DO NOT answer the question, retrieve sources, write a search query, "
    "rank or select articles, or invent any PMIDs/DOIs/citations. You produce a "
    "READ-ONLY understanding of the question to help reviewers. Return STRICT JSON only.\n\n"
    "Determine: intent_hint (short phrase), broad_question_likely (bool — true if "
    "the question is broad/overview rather than a specific PICO), "
    "review_or_guideline_preferred (bool — true if a systematic review or guideline "
    "would best answer it), entities, missing_slots (PICO elements the user left "
    "implicit, e.g. comparator/outcome), fused_slots (distinct concepts the user "
    "merged), optional_terms (synonyms that could broaden recall), reason_codes, "
    "and confidence in [0,1]. Never include patient identifiers."
)

_SCHEMA_HINT = {
    "intent_hint": "string",
    "broad_question_likely": False,
    "review_or_guideline_preferred": False,
    "entities": [{"text": "string", "kind": "condition|drug|procedure|test|population|outcome|other"}],
    "missing_slots": ["string"],
    "fused_slots": ["string"],
    "optional_terms": ["string"],
    "reason_codes": ["string"],
    "confidence": 0.0,
}


def _picott_summary(picott_payload: Optional[dict[str, Any]]) -> str:
    """Compact, closed-field summary of the repaired PICOTT for the prompt."""
    if not isinstance(picott_payload, dict):
        return "(none)"
    keys = ("population", "intervention", "comparator", "outcome", "timing", "study_type")
    parts = []
    for k in keys:
        v = picott_payload.get(k)
        if v:
            parts.append(f"{k}={str(v)[:120]}")
    return "; ".join(parts) if parts else "(sparse)"


async def maybe_interpret(
    session_id: str,
    user_id: str,
    sanitized_question: str,
    parsed: dict[str, Any],
    picott_payload: Optional[dict[str, Any]],
) -> None:
    """Run the understanding agent (if enabled) and store diagnostics. Never raises.

    Does NOT mutate ``parsed`` or ``picott_payload`` — diagnostics only.
    """
    settings = get_settings()
    if not settings.AI_CONTEXT_AUGMENT_ENABLED:
        return

    user_prompt = (
        f"Clinical question:\n{sanitized_question}\n\n"
        f"Deterministic type: {parsed.get('clinical_question_type') or 'other'}\n"
        f"Deterministic PICOTT: {_picott_summary(picott_payload)}\n"
        "Interpret the question now."
    )
    model, reason = await run_json_agent(
        system=_SYSTEM,
        user=user_prompt,
        schema_hint=_SCHEMA_HINT,
        validator=QuestionUnderstanding.model_validate,
        max_tokens=800,
        session_id=session_id,
    )

    await store_agent_decision(
        agent="question_understanding",
        session_id=session_id,
        user_id=user_id,
        decision=(model.model_dump() if model is not None else None),
        applied=False,  # diagnostics only — never applied to retrieval
        failure_reason=reason,
    )
    await audit_service.log(
        "ai_question_understanding_decision",
        user_id=user_id,
        search_session_id=session_id,
        payload={
            "produced": model is not None,
            "broad_question_likely": (model.broad_question_likely if model else None),
            "review_or_guideline_preferred": (model.review_or_guideline_preferred if model else None),
            "confidence": (model.confidence if model else None),
            "failure_reason": reason,
            "prompt_version": PROMPT_VERSION,
        },
    )
