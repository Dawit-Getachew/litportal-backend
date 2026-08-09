"""Phase H1 — Clinical Context Resolver tests.

These tests are deterministic / unit-level. They exercise:

  * The closed-set Pydantic schema + scrubbing pass in
    ``services.clinical_context_resolver.validate_output``.
  * The closed-set rejection enum (every rejection_reason is covered).
  * The closed-set mode flag (off / shadow / fallback_active).
  * The system prompt invariants (no-cite / no-answer / no-scholar).
  * The trigger / skip rules baked into the pipeline (the
    deterministic high-confidence cases NEVER invoke the resolver;
    the 5 closed-set generic-only questions NEVER invoke the
    resolver; the 5 Phase 6G.4P procedure reproducers use the
    deterministic path).

The resolver is exercised via the public ``resolve_clinical_context``
coroutine with the ``llm_provider`` monkeypatched to return a closed
fixture — no real LLM call leaves the test process.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from services.clinical_context_resolver import (
    REJECTION_REASONS,
    SKIPPED_REASONS,
    SYSTEM_PROMPT,
    ClinicalContextResolverOutput,
    clinical_context_resolver,
    current_mode,
    validate_output,
)


# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


def _good_output() -> dict:
    """Return a closed-set valid resolver output dict."""
    return {
        "is_valid_clinical_question": True,
        "question_scope": "foreground",
        "type_of_question": "therapy",
        "question_subtype": "post_procedure_management",
        "specialty": ["cardiology"],
        "picott": {
            "population_problem": ["aortic valve replacement"],
            "intervention_exposure_index_test": ["anticoagulation"],
            "comparator_reference_standard": [],
            "outcomes_target_diagnosis": ["thromboembolism", "bleeding"],
            "time_horizon": [],
            "setting": [],
            "clinical_action": [],
        },
        "primary_anchors": [
            {
                "text": "aortic valve replacement",
                "canonical_concept": "aortic valve replacement",
                "entity_type": "procedure_or_surgery",
                "role": "population_problem",
                "required_for_search": True,
                "missing_relevance_cap": 20,
            },
            {
                "text": "anticoagulation",
                "canonical_concept": "anticoagulation",
                "entity_type": "drug_or_drug_class",
                "role": "intervention_exposure_index_test",
                "required_for_search": True,
                "missing_relevance_cap": 30,
            },
        ],
        "generic_modifiers": ["recommendations"],
        "must_have_concepts": [
            {
                "concept": "aortic valve replacement",
                "role": "population_problem",
                "required_for_high_relevance": True,
                "missing_relevance_cap": 20,
            },
            {
                "concept": "anticoagulation",
                "role": "intervention_exposure_index_test",
                "required_for_high_relevance": True,
                "missing_relevance_cap": 30,
            },
        ],
        "ranking_concepts": ["guidelines", "warfarin", "DOAC"],
        "negative_concepts": [],
        "clarification_needed": False,
        "clarification_message": None,
        "refinement_note": None,
        "confidence": 0.85,
        "rationale": "Procedure + management pair detected.",
    }


# ---------------------------------------------------------------------
# Schema invariants.
# ---------------------------------------------------------------------


def test_system_prompt_contains_required_safety_clauses():
    p = SYSTEM_PROMPT
    required = [
        "You are not answering the medical question.",
        "You are only extracting structured clinical search interpretation.",
        "Return JSON only",
        "Do not cite articles. Do not include PMIDs, DOIs",
        "Do not recommend treatment",
        "Do not reference Google Scholar",
        "Do not invent MeSH terms",
        "If the question lacks a clinical subject",
        # Phase H1.1 — old "Preserve uncertainty" line was replaced
        # with the more discriminating "Confidence guidance" section.
        # Either phrasing is acceptable; assert the new section is
        # present so future drift is caught.
        "Confidence guidance",
        # Phase J-tail — additive structured guidance sections that
        # MUST remain present. The brief mandates these as
        # non-negotiable closed-set clauses.
        "Confidence calibration",
        "Shape A",
        "Shape B",
        "Non-negotiable clarification cases",
        # Confidence-calibration semantic anchor: the prompt MUST
        # clarify that missing comparator / outcome / time horizon
        # MUST NOT auto-push confidence below 0.5.
        "Missing comparator, unspecified outcome",
    ]
    for clause in required:
        assert clause in p, clause


def test_closed_set_rejection_enum():
    assert {
        "schema_invalid",
        "missing_primary_anchor_for_foreground",
        "confidence_below_threshold",
        "contains_pmid_doi_pattern",
        "contains_article_reference",
        "contains_answer_text",
        "contains_phi_pattern",
        "references_unapproved_source",
        "enum_violation",
        "empty_output",
    } == set(REJECTION_REASONS)


def test_closed_set_skipped_enum():
    assert {
        "phi_blocked",
        "generic_only_fast_path_fired",
        "deterministic_high_confidence",
        "background_anchor_already_succeeded",
        "post_procedure_router_already_succeeded",
        "mode_off",
        "no_trigger_matched",
        "cached_resolution",
    } == set(SKIPPED_REASONS)


# ---------------------------------------------------------------------
# validate_output — closed-set scrubbing pass.
# ---------------------------------------------------------------------


def test_validate_output_accepts_valid():
    out, rej = validate_output(_good_output())
    assert rej is None
    assert isinstance(out, ClinicalContextResolverOutput)


def test_validate_output_rejects_schema_invalid():
    bad = {"foo": "bar"}
    out, rej = validate_output(bad)
    assert out is None
    assert rej == "schema_invalid"


def test_validate_output_rejects_extra_keys():
    raw = _good_output()
    raw["unknown_extra_field"] = "naughty"
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "schema_invalid"


def test_validate_output_rejects_pmid_pattern():
    raw = _good_output()
    raw["rationale"] = "See PMID: 12345678 for details"
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "contains_pmid_doi_pattern"


def test_validate_output_rejects_bare_long_digit_run():
    raw = _good_output()
    raw["ranking_concepts"] = ["12345678"]
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "contains_pmid_doi_pattern"


def test_validate_output_rejects_doi_pattern():
    raw = _good_output()
    raw["rationale"] = "Source: 10.1056/NEJMra2102255"
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "contains_pmid_doi_pattern"


def test_validate_output_rejects_article_reference_pattern():
    raw = _good_output()
    raw["rationale"] = "See NCT04567890 enrollment data"
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "contains_article_reference"


def test_validate_output_rejects_unapproved_source():
    raw = _good_output()
    raw["rationale"] = "Google Scholar has relevant content"
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "references_unapproved_source"


@pytest.mark.parametrize("ph,token", [
    ("email", "patient@example.com"),
    ("phone", "(555) 123-4567"),
    ("ssn", "123-45-6789"),
    ("mrn", "MRN: 1234567"),
    ("date", "03/14/1979"),
])
def test_validate_output_rejects_phi_patterns(ph, token):
    raw = _good_output()
    raw["rationale"] = f"Pt {token} present"
    out, rej = validate_output(raw)
    assert out is None, ph
    assert rej == "contains_phi_pattern", ph


def test_validate_output_rejects_long_rationale():
    raw = _good_output()
    raw["rationale"] = "x" * 250
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "contains_answer_text"


def test_validate_output_rejects_answer_indicator_in_list():
    raw = _good_output()
    raw["ranking_concepts"] = ["warfarin is recommended"]
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "contains_answer_text"


def test_validate_output_rejects_low_confidence():
    raw = _good_output()
    raw["confidence"] = 0.3
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "confidence_below_threshold"


def test_validate_output_rejects_foreground_without_anchors():
    raw = _good_output()
    raw["primary_anchors"] = []
    raw["clarification_needed"] = False
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "missing_primary_anchor_for_foreground"


def test_validate_output_rejects_empty_output():
    raw = _good_output()
    raw["primary_anchors"] = []
    raw["must_have_concepts"] = []
    raw["clarification_needed"] = False
    raw["question_scope"] = "background"  # avoid foreground reject
    out, rej = validate_output(raw)
    assert out is None
    assert rej == "empty_output"


def test_validate_output_accepts_clarification_branch():
    raw = _good_output()
    raw["primary_anchors"] = []
    raw["must_have_concepts"] = []
    raw["clarification_needed"] = True
    raw["clarification_message"] = "Please specify a clinical subject."
    raw["question_scope"] = "background"
    out, rej = validate_output(raw)
    assert rej is None
    assert out is not None
    assert out.clarification_needed is True


def test_validate_output_rejects_enum_violation():
    raw = _good_output()
    raw["primary_anchors"][0]["entity_type"] = "not_a_real_type"
    out, rej = validate_output(raw)
    assert out is None
    # Pydantic Literal violation → schema_invalid path.
    assert rej == "schema_invalid"


def test_validate_output_rejects_none():
    out, rej = validate_output(None)
    assert out is None
    assert rej == "empty_output"


def test_validate_output_rejects_list_at_top():
    out, rej = validate_output(["not", "a", "dict"])
    assert out is None
    assert rej == "schema_invalid"


# ---------------------------------------------------------------------
# Mode flag.
# ---------------------------------------------------------------------


def test_mode_default_is_fallback_active(monkeypatch):
    monkeypatch.delenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", raising=False)
    monkeypatch.delenv("CLINICAL_CONTEXT_RESOLVER_MODE", raising=False)
    assert current_mode() == "fallback_active"


def test_mode_off_when_disabled(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "false")
    assert current_mode() == "off"


def test_mode_shadow(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "shadow")
    assert current_mode() == "shadow"


def test_mode_unknown_falls_back_to_fallback_active(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "nuclear")
    assert current_mode() == "fallback_active"


# ---------------------------------------------------------------------
# resolve_clinical_context — service-level integration with mocked LLM.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolver_off_mode_returns_immediately(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "false")
    decision = await clinical_context_resolver.resolve_clinical_context(
        original_question="what about rituximab post stem cell transplant?",
        normalized_question="what about rituximab post stem cell transplant?",
        deterministic_picott=None,
        primary_anchors_detected=[],
        generic_modifiers_detected=[],
        anchor_failure_reason="no_primary_clinical_entity",
        existing_concept_translations=[],
        specialty_hints=[],
    )
    assert decision.accepted is False
    assert decision.mode == "off"
    assert decision.output is None


@pytest.mark.asyncio
async def test_resolver_accepts_valid_llm_output(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "fallback_active")
    fake = AsyncMock(return_value=_good_output())
    monkeypatch.setattr(
        "services.clinical_context_resolver.llm_provider.complete_json",
        fake,
    )
    decision = await clinical_context_resolver.resolve_clinical_context(
        original_question="anticoagulation after aortic valve surgery",
        normalized_question="anticoagulation after aortic valve surgery",
        deterministic_picott=None,
        primary_anchors_detected=[],
        generic_modifiers_detected=[],
        anchor_failure_reason="no_primary_clinical_entity",
        existing_concept_translations=[],
        specialty_hints=[],
    )
    assert decision.accepted is True
    assert decision.output is not None
    assert decision.mode == "fallback_active"
    assert decision.output.primary_anchors[0].canonical_concept == "aortic valve replacement"
    fake.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolver_rejects_phi_in_output(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "fallback_active")
    bad = _good_output()
    bad["rationale"] = "Pt email patient@example.com — see record"
    monkeypatch.setattr(
        "services.clinical_context_resolver.llm_provider.complete_json",
        AsyncMock(return_value=bad),
    )
    decision = await clinical_context_resolver.resolve_clinical_context(
        original_question="some question",
        normalized_question="some question",
        deterministic_picott=None,
        primary_anchors_detected=[],
        generic_modifiers_detected=[],
        anchor_failure_reason="no_primary_clinical_entity",
        existing_concept_translations=[],
        specialty_hints=[],
    )
    assert decision.accepted is False
    assert decision.rejection_reason == "contains_phi_pattern"


@pytest.mark.asyncio
async def test_resolver_rejects_pmid_in_output(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "fallback_active")
    bad = _good_output()
    bad["rationale"] = "See PMID:12345678"
    monkeypatch.setattr(
        "services.clinical_context_resolver.llm_provider.complete_json",
        AsyncMock(return_value=bad),
    )
    decision = await clinical_context_resolver.resolve_clinical_context(
        original_question="x",
        normalized_question="x",
        deterministic_picott=None,
        primary_anchors_detected=[],
        generic_modifiers_detected=[],
        anchor_failure_reason="no_primary_clinical_entity",
        existing_concept_translations=[],
        specialty_hints=[],
    )
    assert decision.accepted is False
    assert decision.rejection_reason == "contains_pmid_doi_pattern"


@pytest.mark.asyncio
async def test_resolver_rejects_unapproved_source(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "fallback_active")
    bad = _good_output()
    bad["rationale"] = "Google Scholar has the consensus paper."
    monkeypatch.setattr(
        "services.clinical_context_resolver.llm_provider.complete_json",
        AsyncMock(return_value=bad),
    )
    decision = await clinical_context_resolver.resolve_clinical_context(
        original_question="x",
        normalized_question="x",
        deterministic_picott=None,
        primary_anchors_detected=[],
        generic_modifiers_detected=[],
        anchor_failure_reason="no_primary_clinical_entity",
        existing_concept_translations=[],
        specialty_hints=[],
    )
    assert decision.accepted is False
    assert decision.rejection_reason == "references_unapproved_source"


@pytest.mark.asyncio
async def test_resolver_handles_llm_returning_none(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "fallback_active")
    monkeypatch.setattr(
        "services.clinical_context_resolver.llm_provider.complete_json",
        AsyncMock(return_value=None),
    )
    decision = await clinical_context_resolver.resolve_clinical_context(
        original_question="x",
        normalized_question="x",
        deterministic_picott=None,
        primary_anchors_detected=[],
        generic_modifiers_detected=[],
        anchor_failure_reason="no_primary_clinical_entity",
        existing_concept_translations=[],
        specialty_hints=[],
    )
    assert decision.accepted is False
    assert decision.rejection_reason == "empty_output"


# ---------------------------------------------------------------------
# Trigger / skip — deterministic path coverage.
# ---------------------------------------------------------------------


def test_phase6f2_closed_set_generic_only_unchanged():
    """The 5 Phase 6F.2 fast-path questions still hit
    ``is_generic_only`` deterministically — the resolver path must not
    fire for them."""
    from services.primary_clinical_anchor import (
        primary_clinical_anchor_service as svc,
    )
    for q in [
        "What are the side effects?",
        "What is first-line treatment?",
        "What is the dose?",
        "What is the recommended age?",
        "What is diagnostic accuracy?",
    ]:
        assert svc.is_generic_only(q) is True, q


def test_phase6g1_anticoagulation_still_routes_to_drug_overview():
    """Phase 6G.1 invariant: "What is anticoagulation?" still
    succeeds deterministically (anchor present) — the resolver must
    not be triggered for this case."""
    from services.primary_clinical_anchor import (
        primary_clinical_anchor_service as svc,
    )
    out = svc.derive(
        picott={},
        ontology_mappings=[],
        normalized_question="What is anticoagulation?",
    )
    # is_generic_only must be False so the pipeline runs the
    # background-overview path (NOT the resolver).
    assert svc.is_generic_only("What is anticoagulation?") is False
    # A primary anchor exists; the resolver is not triggered because
    # ``anchor_failure_reason`` is None.
    anchors = out["primary_anchors"]
    assert len(anchors) >= 1
    assert out["anchor_failure_reason"] is None


@pytest.mark.parametrize("q", [
    "what are the recommendations for anticoagulation after aortic valve surgery?",
    "what antiplatelet therapy is recommended after PCI?",
    "what VTE prophylaxis is recommended after hip replacement?",
    "when should anticoagulation be restarted after colonoscopy with polypectomy?",
    "what immunosuppression is used after kidney transplant?",
])
def test_phase6g4p_reproducers_use_deterministic_path(q):
    """The 5 Phase 6G.4P reproducers must reach awaiting_selection
    via the deterministic post-procedure router — resolver is
    SKIPPED because ``anchor_failure_reason`` is None."""
    from services.primary_clinical_anchor import (
        primary_clinical_anchor_service as svc,
    )
    out = svc.derive(
        picott={},
        ontology_mappings=[],
        normalized_question=q,
    )
    assert out["anchor_failure_reason"] is None, q
    et = {a["entity_type"] for a in out["primary_anchors"]}
    assert "procedure_or_surgery" in et, (q, et)
    # The pipeline only calls the resolver when
    # ``anchor_failure_reason == "no_primary_clinical_entity"``.
    # The deterministic path produces a non-None primary_anchor list
    # for these queries so the resolver branch is not taken.


# ---------------------------------------------------------------------
# Closed-set source: ANCHOR_SOURCES includes the new resolver source.
# ---------------------------------------------------------------------


def test_anchor_sources_includes_llm_resolver_fallback():
    from services.primary_clinical_anchor import ANCHOR_SOURCES
    assert "llm_resolver_fallback" in ANCHOR_SOURCES


# ---------------------------------------------------------------------
# Diagnostics stage allowlist.
# ---------------------------------------------------------------------


def test_diagnostics_stage_clinical_context_resolver_added():
    from services.diagnostics import STAGE_NAMES
    assert "clinical_context_resolver" in STAGE_NAMES
    # Total = 18 stages. Phase H1 added clinical_context_resolver (17th);
    # Phase L1 subsequently added l1_evidence_lens (18th). Both are legitimate
    # closed-set pipeline stages — this count assertion just trailed the L1 add.
    assert len(STAGE_NAMES) == 18
    assert "l1_evidence_lens" in STAGE_NAMES


def test_diagnostics_stage_can_be_opened_via_context_manager():
    from services.diagnostics import SessionDiagnostics
    d = SessionDiagnostics()
    with d.stage("clinical_context_resolver"):
        pass
    assert "clinical_context_resolver" in d.latency_by_stage
    assert d.latency_by_stage["clinical_context_resolver"] >= 0


# ---------------------------------------------------------------------
# Closed-set audit-payload safety contract.
#
# We don't directly test audit_events here (those land in MongoDB at
# runtime). Instead we assert the closed-set contract: the resolver's
# returned ``ResolverDecision`` carries NO question text, NO prompt,
# NO completion. The pipeline only consumes counters + closed-set
# strings.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolver_decision_carries_no_question_text(monkeypatch):
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_CONTEXT_RESOLVER_MODE", "fallback_active")
    memorable_question = "MARKER-UNIQUE-PHRASE-1234567890 special question"
    monkeypatch.setattr(
        "services.clinical_context_resolver.llm_provider.complete_json",
        AsyncMock(return_value=_good_output()),
    )
    decision = await clinical_context_resolver.resolve_clinical_context(
        original_question=memorable_question,
        normalized_question=memorable_question,
        deterministic_picott=None,
        primary_anchors_detected=[],
        generic_modifiers_detected=[],
        anchor_failure_reason="no_primary_clinical_entity",
        existing_concept_translations=[],
        specialty_hints=[],
    )
    blob = json.dumps(decision.model_dump())
    assert "MARKER-UNIQUE-PHRASE" not in blob
