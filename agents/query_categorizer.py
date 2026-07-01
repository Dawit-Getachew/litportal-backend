"""AI Agent 1 — Query Categorizer / Enricher (pre-deterministic-parser).

Problem it solves: the deterministic parser's ``_classify_type`` is keyword-gated
— a drug/clinical question that lacks the expected trigger words falls through to
``"other"``, which then mis-routes the whole search. This agent runs right after
the deterministic parse and, ONLY when the deterministic type came back weak
(``"other"``) and the agent is confident, refines the ``clinical_question_type``
so downstream type-aware logic categorizes it correctly. It also stashes
advisory entities + enriched terms as diagnostics (NOT auto-injected into the
Boolean query — that stays deterministic).

Governance: advisory only. Deterministic parser still runs and still wins unless
its own answer was the catch-all ``"other"``. Feature-flagged (default OFF),
fail-closed (any failure leaves the deterministic parse untouched). No source
calls, no citations.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.base import run_json_agent, store_agent_decision
from agents.schemas import QueryCategoryResult
from core.config import get_settings
from models.pico import CLINICAL_QUESTION_TYPES
from services.audit import audit_service

logger = logging.getLogger("litportal.agents.query_categorizer")

PROMPT_VERSION = "qcat-v1"

_SYSTEM = (
    "You are a clinical-question CATEGORIZER for a medical evidence search engine. "
    "You DO NOT answer the question, retrieve sources, write a search query, or "
    "invent any PMIDs/DOIs/citations. You ONLY classify and extract structure so a "
    "downstream deterministic parser categorizes the question correctly. "
    "Return STRICT JSON only.\n\n"
    "Classify the question into exactly one question_type_hint from this closed set: "
    + ", ".join(CLINICAL_QUESTION_TYPES) + ". "
    "Extract clinical entities (drug, drug_class, condition, procedure, test, "
    "population, outcome). Provide a few enriched_terms (synonyms / canonical "
    "names) that make the clinical intent explicit. Set confidence in [0,1] for "
    "your type classification. Use reason_codes (short snake_case tags) to explain "
    "the classification. Never include patient identifiers."
)

_SCHEMA_HINT = {
    "normalized_query": "string",
    "question_type_hint": "one of: " + "|".join(CLINICAL_QUESTION_TYPES),
    "entities": [{"text": "string", "kind": "drug|drug_class|condition|procedure|test|population|outcome|other"}],
    "enriched_terms": ["string"],
    "reason_codes": ["string"],
    "confidence": 0.0,
}


async def maybe_categorize(
    session_id: str,
    user_id: str,
    sanitized_question: str,
    parsed: dict[str, Any],
) -> None:
    """Run the categorizer (if enabled) and refine ``parsed`` in place.

    ``parsed`` is the dict returned by clinical_question_parser.parse(). On a
    high-confidence result where the deterministic type was the catch-all
    ``"other"``, this sets ``parsed["clinical_question_type"]`` to the agent's
    hint; otherwise it only attaches advisory diagnostics. Never raises.
    """
    settings = get_settings()
    if not settings.AI_QUERY_CATEGORIZER_ENABLED:
        return

    det_type = (parsed.get("clinical_question_type") or "other")
    user_prompt = (
        f"Clinical question:\n{sanitized_question}\n\n"
        f"Deterministic preliminary type (may be wrong): {det_type}\n"
        "Categorize the question now."
    )
    model, reason = await run_json_agent(
        system=_SYSTEM,
        user=user_prompt,
        schema_hint=_SCHEMA_HINT,
        validator=QueryCategoryResult.model_validate,
        max_tokens=700,
        session_id=session_id,
    )

    applied = False
    if model is not None:
        # Refine ONLY when the deterministic parser fell through to "other"
        # and the agent is confident — never override a confident deterministic type.
        if (
            det_type == "other"
            and model.question_type_hint != "other"
            and model.confidence >= settings.AI_QUERY_CATEGORIZER_MIN_CONFIDENCE
        ):
            parsed["clinical_question_type"] = model.question_type_hint
            applied = True
        # Advisory enrichment for diagnostics + (optional, later) planner use.
        parsed["ai_query_categorizer"] = {
            "question_type_hint": model.question_type_hint,
            "entities": [e.model_dump() for e in model.entities],
            "enriched_terms": model.enriched_terms,
            "reason_codes": model.reason_codes,
            "confidence": model.confidence,
            "deterministic_type": det_type,
            "applied": applied,
            "prompt_version": PROMPT_VERSION,
        }

    await store_agent_decision(
        agent="query_categorizer",
        session_id=session_id,
        user_id=user_id,
        decision=(model.model_dump() if model is not None else None),
        applied=applied,
        failure_reason=reason,
    )
    await audit_service.log(
        "ai_query_categorizer_decision",
        user_id=user_id,
        search_session_id=session_id,
        payload={
            "applied": applied,
            "deterministic_type": det_type,
            "hint": (model.question_type_hint if model else None),
            "confidence": (model.confidence if model else None),
            "failure_reason": reason,
            "prompt_version": PROMPT_VERSION,
        },
    )
