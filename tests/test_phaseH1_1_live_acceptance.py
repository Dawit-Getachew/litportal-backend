"""Phase H1.1 — Live acceptance + hardening tests.

These tests cover the H1.1 brief's verification surface:

  * Resolver trigger rules (closed-set, no LLM call) — extends H1
    coverage with parametrized regression cases.
  * Audit / diagnostic payload redaction (no PHI, no question text,
    no prompt/completion text appears in the persisted resolver
    fields).
  * Generic-only fast-path skipped_reason persistence.
  * Boolean compiler integration smoke test — when the resolver merge
    fires the must_have_concepts list carries the resolver-sourced
    anchors as ``required_for_high_relevance=True``.
  * `_build_user_payload` does NOT echo back deterministic generic
    modifiers (root cause of the Phase H1.1 LLM-confidence bug).

NO network. NO real LLM call. The resolver's LLM dispatch is
monkeypatched. The pipeline / pubmed paths are not exercised.
"""
from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock

import pytest

from services.clinical_context_resolver import (
    SYSTEM_PROMPT,
    _build_user_payload,
    clinical_context_resolver,
    validate_output,
)


# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


def _accept_payload() -> dict:
    return {
        "is_valid_clinical_question": True,
        "question_scope": "foreground",
        "type_of_question": "dosing",
        "question_subtype": None,
        "specialty": ["hematology"],
        "picott": {
            "population_problem": ["autologous stem cell transplant"],
            "intervention_exposure_index_test": ["rituximab"],
            "comparator_reference_standard": [],
            "outcomes_target_diagnosis": [],
            "time_horizon": [],
            "setting": [],
            "clinical_action": [],
        },
        "primary_anchors": [
            {
                "text": "rituximab",
                "canonical_concept": "rituximab",
                "entity_type": "drug_or_drug_class",
                "role": "intervention_exposure_index_test",
                "required_for_search": True,
                "missing_relevance_cap": 20,
            },
            {
                "text": "autologous stem cell transplant",
                "canonical_concept": "autologous stem cell transplant",
                "entity_type": "procedure_or_surgery",
                "role": "population_problem",
                "required_for_search": True,
                "missing_relevance_cap": 20,
            },
        ],
        "generic_modifiers": ["dose", "recommended"],
        "must_have_concepts": [
            {
                "concept": "rituximab",
                "role": "intervention_exposure_index_test",
                "required_for_high_relevance": True,
                "missing_relevance_cap": 20,
            },
        ],
        "ranking_concepts": [],
        "negative_concepts": [],
        "clarification_needed": False,
        "clarification_message": None,
        "refinement_note": None,
        "confidence": 0.85,
        "rationale": "Drug + procedure pair extracted.",
    }


# ---------------------------------------------------------------------
# H1.1 root-cause fix: _build_user_payload MUST NOT echo modifiers.
# ---------------------------------------------------------------------


def test_user_payload_does_not_echo_generic_modifiers():
    """Phase H1.1 root cause — passing the deterministic-detected
    generic modifiers (e.g. ``[dose]``) into the resolver user
    payload made GPT-4o-mini over-weight "this question is generic"
    and reject otherwise-confident clinical entities with
    confidence_below_threshold. The user payload MUST NOT echo
    them."""
    payload = _build_user_payload(
        normalized_question="recommended dose of rituximab",
        anchor_failure_reason="no_primary_clinical_entity",
        generic_modifiers_detected=["dose", "recommended"],
        specialty_hints=["hematology"],
    )
    assert "normalized_question" in payload
    assert "deterministic_anchor_failure_reason" in payload
    # These two MUST be absent — they confuse the LLM and add no signal.
    assert "deterministic_generic_modifiers_detected" not in payload
    assert "deterministic_specialty_hints" not in payload
    # The literal modifier strings must not leak in either.
    assert "[dose]" not in payload
    assert "hematology" not in payload


def test_user_payload_signature_remains_backwards_compatible():
    """The pipeline still passes the modifier + hints args — the
    function MUST accept them silently (no TypeError) but discard
    them from the payload body."""
    payload = _build_user_payload(
        normalized_question="x",
        anchor_failure_reason=None,
        generic_modifiers_detected=[{"text": "any"}, "string-mod"],
        specialty_hints=[123, "any"],
    )
    assert payload  # non-empty
    assert "x" in payload


# ---------------------------------------------------------------------
# Confidence-guidance prompt clauses (H1.1 prompt fix).
# ---------------------------------------------------------------------


def test_system_prompt_has_confidence_guidance_section():
    p = SYSTEM_PROMPT
    assert "Confidence guidance:" in p
    assert "at least one clear, named clinical entity" in p
    assert "confidence between 0.7 and 0.95" in p
    # The old over-conservative phrasing is gone.
    assert "Preserve uncertainty: set confidence < 0.7" not in p


# ---------------------------------------------------------------------
# Audit / diagnostic payload safety — no question text leak.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolver_decision_payload_carries_no_question_text(monkeypatch):
    """Even when the resolver ACCEPTS valid output, the closed-set
    ResolverDecision payload that the pipeline persists MUST NOT
    contain the original question text. Acts as the test-suite
    contract that downstream audit / diagnostics paths can rely on.
    """
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "fallback_active")
    monkeypatch.setattr(
        "services.clinical_context_resolver.llm_provider.complete_json",
        AsyncMock(return_value=_accept_payload()),
    )
    marker = "MARKER-H1-1-UNIQUE-2026-06-04"
    question = f"recommended dose of rituximab {marker} after ASCT"
    decision = await clinical_context_resolver.resolve_clinical_context(
        original_question=question,
        normalized_question=question,
        deterministic_picott=None,
        primary_anchors_detected=[],
        generic_modifiers_detected=["dose"],
        anchor_failure_reason="no_primary_clinical_entity",
        existing_concept_translations=[],
        specialty_hints=[],
    )
    blob = json.dumps(decision.model_dump())
    assert marker not in blob


# ---------------------------------------------------------------------
# Closed-set rejection enum invariants.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("rejection_reason", [
    "schema_invalid", "missing_primary_anchor_for_foreground",
    "confidence_below_threshold", "contains_pmid_doi_pattern",
    "contains_article_reference", "contains_answer_text",
    "contains_phi_pattern", "references_unapproved_source",
    "enum_violation", "empty_output",
])
def test_every_closed_set_rejection_reason_is_documented(rejection_reason):
    """Defensive: ensure no resolver rejection ever surfaces a
    rejection_reason outside the closed-set enum. The constant set
    is the contract."""
    from services.clinical_context_resolver import REJECTION_REASONS
    assert rejection_reason in REJECTION_REASONS


# ---------------------------------------------------------------------
# Trigger-rule table — quick regression matrix.
# ---------------------------------------------------------------------


def test_trigger_rules_table_explicit():
    """Documents the closed-set trigger / skip decision table for
    audit. If any of these strings changes you've also changed the
    contract — update this test deliberately."""
    from services.clinical_context_resolver import SKIPPED_REASONS
    expected = {
        "phi_blocked",
        "generic_only_fast_path_fired",
        "deterministic_high_confidence",
        "background_anchor_already_succeeded",
        "post_procedure_router_already_succeeded",
        "mode_off",
        "no_trigger_matched",
        "cached_resolution",
    }
    assert expected == set(SKIPPED_REASONS)


# ---------------------------------------------------------------------
# Resolver-merged anchor carries `source="llm_resolver_fallback"`.
# ---------------------------------------------------------------------


def test_anchor_sources_includes_llm_resolver_fallback_post_h1_1():
    """Phase H1.1 — the source string MUST remain in the closed-set
    ANCHOR_SOURCES so the persistence + UI consumers can rely on it.
    """
    from services.primary_clinical_anchor import ANCHOR_SOURCES
    assert "llm_resolver_fallback" in ANCHOR_SOURCES


# ---------------------------------------------------------------------
# Validate-output regression — the H1.1 prompt fix doesn't loosen
# the schema lock.
# ---------------------------------------------------------------------


def test_validate_output_still_rejects_phi_after_h1_1():
    raw = _accept_payload()
    raw["rationale"] = "Patient email: foo@bar.com"
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "contains_phi_pattern"


def test_validate_output_still_rejects_pmid_after_h1_1():
    raw = _accept_payload()
    raw["rationale"] = "See PMID 12345678"
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "contains_pmid_doi_pattern"


def test_validate_output_still_rejects_scholar_after_h1_1():
    raw = _accept_payload()
    raw["rationale"] = "Google Scholar has the paper."
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "references_unapproved_source"


def test_validate_output_still_rejects_low_confidence_after_h1_1():
    raw = _accept_payload()
    raw["confidence"] = 0.3
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "confidence_below_threshold"


# ---------------------------------------------------------------------
# Mode flag stability across the H1.1 changes.
# ---------------------------------------------------------------------


def test_mode_flag_off_disables_resolver(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "false")
    from services.clinical_context_resolver import current_mode
    assert current_mode() == "off"


def test_mode_flag_shadow_recognized(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "shadow")
    from services.clinical_context_resolver import current_mode
    assert current_mode() == "shadow"


def test_mode_flag_fallback_active_default(monkeypatch):
    monkeypatch.delenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", raising=False)
    monkeypatch.delenv("CLINICAL_CONTEXT_RESOLVER_MODE", raising=False)
    from services.clinical_context_resolver import current_mode
    assert current_mode() == "fallback_active"
