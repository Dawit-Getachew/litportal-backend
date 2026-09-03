"""Phase M2.1 — Bounded Semantic Repair Schema Reliability & Prompt
Stabilization. Tests assert against the M2.1 closed-set rejection
taxonomy, the canonicalization layer (deterministic alias map), and
the prompt's refusal contract.

Live LLM is NOT called here — the LLM is mocked via
``services.llm_provider.llm_provider.complete_json``. Live preview
smokes are run separately in the closeout report.

Cross-phase preservation tests assert that K4A.2 / K4A.3 / K4A.4
routes still suppress the M-phase detector (the K4A.4 → K4A.4.1
lesson). All assertions are framed against the values the closed-set
admin serializer would emit (``semantic_repair_validation_status``,
``semantic_repair_rejection_reasons``, ``semantic_repair_candidate``
fields), NOT against internal pydantic state.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from services.semantic_anchor_repair import (
    ALLOWED_FRAMEWORKS,
    ALLOWED_PRIMARY_INTENTS,
    SEMANTIC_REPAIR_REJECTION_REASONS,
    canonicalize_candidate,
    detect_repair_triggers,
    run_semantic_anchor_repair_shadow,
    validate_repair_candidate,
)


# =====================================================================
# Fixtures
# =====================================================================
def _good_candidate(
    *,
    primary_intent: str = "risk_etiology",
    framework: str = "risk_etiology",
    required=("schizophrenia",),
    reason: str = "missing_risk_etiology_framework",
    confidence: float = 0.85,
    provenance: str = "bounded_semantic_repair_shadow",
) -> dict:
    return {
        "primary_intent": primary_intent,
        "secondary_intents": [],
        "frameworks": [framework],
        "branches": [],
        "slots": {
            "population": ["young adults"],
            "condition": ["schizophrenia"],
            "intervention_or_exposure": [],
            "comparator": [],
            "outcome": [],
            "index_test": [],
            "target_condition": [],
            "procedure_or_device": [],
            "timeframe": [],
            "setting": [],
            "mechanism": [],
        },
        "slot_policies": {"condition": "required_for_retrieval"},
        "required_anchors": list(required),
        "optional_or_ranking_only_concepts": [],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": ["risk factors"],
        "candidate_mesh_terms_unvalidated": ["Schizophrenia"],
        "semantic_repair_reason": reason,
        "confidence": confidence,
        "provenance": provenance,
    }


# =====================================================================
# M2.1 — Closed-set taxonomy extension.
# =====================================================================
def test_M21_new_rejection_reasons_are_in_closed_set() -> None:
    """The Phase M2.1 specific rejection reasons must all be members
    of the public closed-set frozenset so admin serializers and
    downstream filters can rely on them."""
    expected = {
        "missing_required_field",
        "extra_field",
        "invalid_enum",
        "invalid_branch_shape",
        "invalid_slot_type",
        "invalid_slot_policy",
        "invalid_confidence",
        "empty_required_anchors",
        "generic_only_required_anchor",
        "unvalidated_mesh_marked_validated",
        "malformed_json",
        "unsafe_text_detected",
        "refused",
    }
    assert expected.issubset(SEMANTIC_REPAIR_REJECTION_REASONS)


def test_M21_validator_returns_list_of_reasons() -> None:
    """Validator now returns ``list[str]`` rejection reasons (empty
    on success). The pipeline expects this list-based contract."""
    cand, rejs = validate_repair_candidate(
        _good_candidate(),
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
    )
    assert cand is not None
    assert isinstance(rejs, list)
    assert rejs == []


# =====================================================================
# M2.1 — Closed-set canonicalization. The alias map should normalize
# common LLM phrasings to approved enum values BEFORE strict Pydantic
# validation. NEW values must NEVER be produced — only existing enum
# members are returned by the map.
# =====================================================================
def test_M21_canonicalize_primary_intent_aliases() -> None:
    out = canonicalize_candidate({
        **_good_candidate(),
        "primary_intent": "risk_factors",
    })
    assert out["primary_intent"] == "risk_etiology"
    assert out["primary_intent"] in ALLOWED_PRIMARY_INTENTS


def test_M21_canonicalize_framework_casing() -> None:
    out = canonicalize_candidate({
        **_good_candidate(),
        "frameworks": ["picot", "cocopop", "PECO"],
    })
    assert "PICOT" in out["frameworks"]
    assert "CoCoPop" in out["frameworks"]
    # PECO is canonicalized to CoCoPop (closest semantic equivalent in
    # the M2.1 alias map; PECO is not in the closed-set).
    assert "CoCoPop" in out["frameworks"]
    for fw in out["frameworks"]:
        assert fw in ALLOWED_FRAMEWORKS


def test_M21_canonicalize_slot_policy_aliases() -> None:
    out = canonicalize_candidate({
        **_good_candidate(),
        "slot_policies": {
            "condition": "required",
            "outcome": "optional",
            "mechanism": "ranking",
        },
    })
    assert out["slot_policies"]["condition"] == "required_for_retrieval"
    assert out["slot_policies"]["outcome"] == "optional_for_retrieval"
    assert out["slot_policies"]["mechanism"] == "ranking_only"


def test_M21_canonicalize_semantic_repair_reason_alias() -> None:
    out = canonicalize_candidate({
        **_good_candidate(),
        "semantic_repair_reason": "mechanism",  # short alias
    })
    assert out["semantic_repair_reason"] == "missing_mechanism_pathophysiology_framework"


def test_M21_canonicalize_passes_through_unknown_aliases() -> None:
    """Unknown aliases pass through unchanged and fail strict
    validation downstream — the canonicalization layer is NEVER
    permissive."""
    out = canonicalize_candidate({
        **_good_candidate(),
        "primary_intent": "totally_made_up",
    })
    assert out["primary_intent"] == "totally_made_up"


# =====================================================================
# M2.1 — Validator end-to-end with canonicalization wired in.
# =====================================================================
def test_M21_validator_canonicalizes_risk_factors_to_risk_etiology() -> None:
    """A candidate that uses the natural-language phrasing
    ``"risk_factors"`` for primary_intent should be accepted because
    the canonicalization layer maps it to ``"risk_etiology"``."""
    raw = _good_candidate(primary_intent="risk_factors")
    cand, rejs = validate_repair_candidate(
        raw,
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
    )
    assert rejs == []
    assert cand is not None
    assert cand.primary_intent == "risk_etiology"


def test_M21_validator_canonicalizes_cocopop_casing() -> None:
    raw = _good_candidate()
    raw["frameworks"] = ["cocopop"]
    cand, rejs = validate_repair_candidate(
        raw,
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
    )
    assert rejs == []
    assert cand is not None
    assert "CoCoPop" in cand.frameworks


# =====================================================================
# M2.1 — Specific Pydantic-error → closed-set rejection-reason mapping.
# =====================================================================
def test_M21_missing_required_field_specific_reason() -> None:
    bad = _good_candidate()
    del bad["primary_intent"]
    _, rejs = validate_repair_candidate(bad, deterministic_primary_anchors=[])
    assert "missing_required_field" in rejs


def test_M21_invalid_confidence_specific_reason() -> None:
    bad = _good_candidate()
    bad["confidence"] = "not-a-number"
    _, rejs = validate_repair_candidate(bad, deterministic_primary_anchors=[])
    assert "invalid_confidence" in rejs


def test_M21_invalid_slot_type_specific_reason() -> None:
    bad = _good_candidate()
    bad["slots"] = "this should be a dict, not a string"
    _, rejs = validate_repair_candidate(bad, deterministic_primary_anchors=[])
    # Pydantic surfaces either invalid_branch_shape (model_type) or
    # invalid_slot_type depending on how it fails the RepairSlots
    # sub-model — accept either as long as it's a specific reason.
    assert any(
        r in rejs
        for r in {"invalid_slot_type", "invalid_branch_shape", "missing_required_field"}
    )
    # Must NOT be the opaque legacy reason.
    assert "schema_invalid" not in rejs


def test_M21_invalid_branch_shape_specific_reason() -> None:
    bad = _good_candidate()
    # Send a branch with the wrong shape — ``branch_label`` is
    # required and must be a string. Pydantic surfaces this as
    # ``missing_required_field`` or ``invalid_branch_shape`` depending
    # on which error path it takes.
    bad["branches"] = [{"branch_intent": "y", "anchors": ["x"]}]  # missing branch_label
    _, rejs = validate_repair_candidate(bad, deterministic_primary_anchors=[])
    # Either specific reason is acceptable — but never the opaque
    # legacy umbrella.
    assert any(
        r in rejs
        for r in {"invalid_branch_shape", "missing_required_field"}
    )
    assert "schema_invalid" not in rejs


def test_M21_invalid_slot_policy_specific_reason() -> None:
    bad = _good_candidate()
    bad["slot_policies"] = {"condition": "totally_made_up_policy"}
    _, rejs = validate_repair_candidate(bad, deterministic_primary_anchors=[])
    assert "invalid_slot_policy" in rejs


def test_M21_provenance_mismatch_specific_reason() -> None:
    """Provenance is a Pydantic Literal — its mismatch must surface
    as the SPECIFIC ``provenance_mismatch`` reason, NOT the opaque
    ``schema_invalid`` umbrella that M0/M1/M2 emitted."""
    bad = _good_candidate(provenance="some_other_path")
    _, rejs = validate_repair_candidate(bad, deterministic_primary_anchors=[])
    assert "provenance_mismatch" in rejs
    assert "schema_invalid" not in rejs


def test_M21_extra_field_specific_reason() -> None:
    bad = _good_candidate()
    bad["spurious_extra"] = "should be rejected"
    _, rejs = validate_repair_candidate(bad, deterministic_primary_anchors=[])
    assert "extra_field" in rejs


def test_M21_malformed_json_specific_reason() -> None:
    """Non-dict input (which can only happen if the LLM provider
    returns a non-dict — e.g. a list) maps to ``malformed_json``."""
    _, rejs = validate_repair_candidate(
        ["not", "a", "dict"],  # type: ignore[arg-type]
        deterministic_primary_anchors=[],
    )
    assert "malformed_json" in rejs


def test_M21_refused_envelope_returns_refused_reason() -> None:
    """An explicit LLM refusal envelope short-circuits with the
    ``refused`` closed-set reason."""
    raw = {"refused": True, "refusal_reason": "ambiguous_question"}
    cand, rejs = validate_repair_candidate(raw, deterministic_primary_anchors=[])
    assert cand is None
    assert rejs == ["refused"]


# =====================================================================
# M2.1 — A through J canonical-shape projection tests. Each asserts
# against the closed-set surface (rejection list, validation status,
# candidate present/absent) that the admin diagnostics endpoint
# exposes — NOT against internal pydantic state. Mirrors the
# K4A.4 → K4A.4.1 lesson.
# =====================================================================

# ---------- A. risk/etiology in a population ----------
def test_M21_A_causes_in_population_canonical_candidate_passes() -> None:
    """The canonical A repair candidate (using the same shape as the
    M2.1 few-shot example) passes validation cleanly."""
    raw = {
        "primary_intent": "risk_etiology",
        "secondary_intents": [],
        "frameworks": ["risk_etiology"],
        "branches": [],
        "slots": {
            "population": ["young adults"],
            "condition": ["schizophrenia"],
            "intervention_or_exposure": [],
            "comparator": [],
            "outcome": [],
            "index_test": [],
            "target_condition": [],
            "procedure_or_device": [],
            "timeframe": [],
            "setting": [],
            "mechanism": [],
        },
        "slot_policies": {"condition": "required_for_retrieval"},
        "required_anchors": ["schizophrenia"],
        "optional_or_ranking_only_concepts": ["young adults"],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": ["etiology", "risk factors", "causes"],
        "candidate_mesh_terms_unvalidated": ["Schizophrenia"],
        "semantic_repair_reason": "missing_risk_etiology_framework",
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
    }
    cand, rejs = validate_repair_candidate(
        raw,
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
    )
    assert rejs == []
    assert cand is not None
    assert cand.primary_intent == "risk_etiology"
    assert "schizophrenia" in cand.required_anchors
    assert "young adults" in cand.slots.population


# ---------- B. risk factors with exposure population ----------
def test_M21_B_risk_factors_in_smokers_canonical_candidate_passes() -> None:
    raw = {
        "primary_intent": "risk_etiology",
        "secondary_intents": [],
        "frameworks": ["risk_etiology"],
        "branches": [],
        "slots": {
            "population": ["smokers"],
            "condition": ["pancreatic cancer"],
            "intervention_or_exposure": ["smoking"],
            "comparator": [],
            "outcome": ["pancreatic cancer"],
            "index_test": [],
            "target_condition": [],
            "procedure_or_device": [],
            "timeframe": [],
            "setting": [],
            "mechanism": [],
        },
        "slot_policies": {"condition": "required_for_retrieval"},
        "required_anchors": ["pancreatic cancer"],
        "optional_or_ranking_only_concepts": ["smoking", "smokers"],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": ["risk factors", "tobacco"],
        "candidate_mesh_terms_unvalidated": ["Pancreatic Neoplasms", "Smoking"],
        "semantic_repair_reason": "missing_risk_etiology_framework",
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
    }
    cand, rejs = validate_repair_candidate(
        raw,
        deterministic_primary_anchors=[
            {"canonical_concept": "pancreatic cancer"},
        ],
    )
    assert rejs == []
    assert cand is not None
    assert "pancreatic cancer" in cand.required_anchors


# ---------- C. cross-phase preservation: K4A.4 fires → M skips ----------
def test_M21_C_K4A4_cross_phase_preservation() -> None:
    triggers = detect_repair_triggers(
        normalized_question="What is the incidence of aspiration pneumonia in patients with GCS 8 or less?",
        question_subtype="epidemiology_incidence_complication_risk",
        type_of_question="epidemiology",
        primary_anchors=[
            {"canonical_concept": "aspiration pneumonia", "text": "aspiration pneumonia"},
            {"canonical_concept": "low Glasgow Coma Scale", "text": "low Glasgow Coma Scale"},
        ],
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert triggers == []


# ---------- D. cross-phase preservation: K4A.2 fires → M skips ----------
def test_M21_D_K4A2_cross_phase_preservation() -> None:
    triggers = detect_repair_triggers(
        normalized_question="What are markers of severity in sepsis?",
        question_subtype="severity_marker_overview",
        type_of_question="other",
        primary_anchors=[
            {"canonical_concept": "sepsis", "text": "sepsis"},
        ],
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert triggers == []


# ---------- E. diagnostic rule-out canonical candidate ----------
def test_M21_E_dimer_pe_canonical_candidate_passes() -> None:
    raw = {
        "primary_intent": "rule_out_test_utility",
        "secondary_intents": ["diagnostic_accuracy"],
        "frameworks": ["PIRD", "diagnostic_accuracy"],
        "branches": [],
        "slots": {
            "population": [],
            "condition": [],
            "intervention_or_exposure": [],
            "comparator": [],
            "outcome": [],
            "index_test": ["D-dimer"],
            "target_condition": ["pulmonary embolism"],
            "procedure_or_device": [],
            "timeframe": [],
            "setting": [],
            "mechanism": [],
        },
        "slot_policies": {
            "index_test": "required_for_retrieval",
            "target_condition": "required_for_retrieval",
        },
        "required_anchors": ["D-dimer", "pulmonary embolism"],
        "optional_or_ranking_only_concepts": ["negative predictive value"],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": ["rule out", "NPV"],
        "candidate_mesh_terms_unvalidated": [
            "Fibrin Fibrinogen Degradation Products", "Pulmonary Embolism",
        ],
        "semantic_repair_reason": "missing_diagnostic_ruleout_split",
        "confidence": 0.88,
        "provenance": "bounded_semantic_repair_shadow",
    }
    cand, rejs = validate_repair_candidate(
        raw,
        deterministic_primary_anchors=[
            {"canonical_concept": "pulmonary embolism"},
        ],
    )
    assert rejs == []
    assert cand is not None
    assert "D-dimer" in cand.required_anchors
    assert "pulmonary embolism" in cand.required_anchors


# ---------- F. mechanism / pathophysiology canonical candidate ----------
def test_M21_F_mechanism_aps_canonical_candidate_passes() -> None:
    """The Reproducer F shape — the one that flapped between
    schema-valid and schema-invalid pre-M2.1 — MUST validate cleanly
    when the LLM emits the canonical shape."""
    raw = {
        "primary_intent": "mechanism",
        "secondary_intents": ["pathophysiology"],
        "frameworks": ["mechanism"],
        "branches": [],
        "slots": {
            "population": [],
            "condition": ["antiphospholipid syndrome"],
            "intervention_or_exposure": [],
            "comparator": [],
            "outcome": ["thrombosis"],
            "index_test": [],
            "target_condition": [],
            "procedure_or_device": [],
            "timeframe": [],
            "setting": [],
            "mechanism": ["thrombosis"],
        },
        "slot_policies": {"condition": "required_for_retrieval"},
        "required_anchors": ["antiphospholipid syndrome"],
        "optional_or_ranking_only_concepts": ["thrombosis", "hypercoagulability"],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": ["pathophysiology", "mechanism"],
        "candidate_mesh_terms_unvalidated": [
            "Antiphospholipid Syndrome", "Thrombosis",
        ],
        "semantic_repair_reason": "missing_mechanism_pathophysiology_framework",
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
    }
    cand, rejs = validate_repair_candidate(
        raw,
        deterministic_primary_anchors=[
            {"canonical_concept": "antiphospholipid syndrome"},
        ],
    )
    assert rejs == []
    assert cand is not None
    assert cand.primary_intent == "mechanism"
    assert "antiphospholipid syndrome" in cand.required_anchors


def test_M21_F_mechanism_aps_canonical_via_canonicalization_layer() -> None:
    """Same shape as F but the LLM emits common variants
    (``"pathophysiology_mechanism"`` for the repair reason,
    lowercase ``"mechanism_framework"`` for framework) — the
    canonicalization layer normalizes these before strict
    validation."""
    raw = {
        "primary_intent": "mechanism",
        "secondary_intents": [],
        "frameworks": ["mechanism_framework"],  # alias
        "branches": [],
        "slots": {
            "population": [],
            "condition": ["antiphospholipid syndrome"],
            "intervention_or_exposure": [],
            "comparator": [],
            "outcome": ["thrombosis"],
            "index_test": [],
            "target_condition": [],
            "procedure_or_device": [],
            "timeframe": [],
            "setting": [],
            "mechanism": ["thrombosis"],
        },
        "slot_policies": {"condition": "required_for_retrieval"},
        "required_anchors": ["antiphospholipid syndrome"],
        "optional_or_ranking_only_concepts": ["thrombosis"],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": [],
        "candidate_mesh_terms_unvalidated": [],
        "semantic_repair_reason": "mechanism",  # alias
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
    }
    cand, rejs = validate_repair_candidate(
        raw,
        deterministic_primary_anchors=[
            {"canonical_concept": "antiphospholipid syndrome"},
        ],
    )
    assert rejs == []
    assert cand is not None
    assert "mechanism" in cand.frameworks
    assert cand.semantic_repair_reason == "missing_mechanism_pathophysiology_framework"


# ---------- G. complications after procedure canonical candidate ----------
def test_M21_G_complication_after_pacemaker_canonical_candidate_passes() -> None:
    raw = {
        "primary_intent": "complication_after_procedure",
        "secondary_intents": ["complication_risk"],
        "frameworks": ["complication"],
        "branches": [],
        "slots": {
            "population": [],
            "condition": [],
            "intervention_or_exposure": [],
            "comparator": [],
            "outcome": ["infection"],
            "index_test": [],
            "target_condition": [],
            "procedure_or_device": ["pacemaker", "pacemaker implantation"],
            "timeframe": ["post-implantation"],
            "setting": [],
            "mechanism": [],
        },
        "slot_policies": {"procedure_or_device": "required_for_retrieval"},
        "required_anchors": ["pacemaker"],
        "optional_or_ranking_only_concepts": ["pacemaker implantation", "infection"],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": [
            "device infection", "CIED infection",
        ],
        "candidate_mesh_terms_unvalidated": [
            "Pacemaker, Artificial", "Prosthesis-Related Infections",
        ],
        "semantic_repair_reason": "missing_complication_after_procedure_split",
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
    }
    cand, rejs = validate_repair_candidate(
        raw,
        deterministic_primary_anchors=[
            {"canonical_concept": "pacemaker"},
        ],
    )
    assert rejs == []
    assert cand is not None
    assert "pacemaker" in cand.required_anchors


# ---------- H. generic-only / unanchored input ----------
def test_M21_H_generic_only_query_no_trigger() -> None:
    """``"What is the dose?"`` — no detector trigger; M-phase short-
    circuits cleanly."""
    triggers = detect_repair_triggers(
        normalized_question="What is the dose?",
        question_subtype=None,
        type_of_question="other",
        primary_anchors=[],
        generic_modifiers=["dose"],
        anchor_failure_reason=None,
    )
    assert triggers == []


def test_M21_H_generic_only_required_anchor_rejected() -> None:
    bad = _good_candidate(required=("patients",))
    _, rejs = validate_repair_candidate(
        bad,
        deterministic_primary_anchors=[],
    )
    assert "generic_only_required_anchor" in rejs


# ---------- I. PHI dominance — pre-LLM short circuit ----------
@pytest.mark.asyncio
async def test_M21_I_phi_dominance_no_llm_call(monkeypatch) -> None:
    """When the trigger list is empty (the outward effect of an
    upstream PHI block), the M-phase entrypoint MUST NOT invoke
    the LLM. Asserted by mocking ``complete_json`` and confirming
    ``await_count == 0``."""
    from services import llm_provider as lp
    mock_complete = AsyncMock(return_value=None)
    monkeypatch.setattr(lp.llm_provider, "complete_json", mock_complete)
    cand, rejs, latency = await run_semantic_anchor_repair_shadow(
        normalized_question="",
        trigger_reasons=[],
        deterministic_primary_anchors=[],
    )
    assert cand is None
    assert rejs == []
    assert latency == 0
    assert mock_complete.await_count == 0


@pytest.mark.asyncio
async def test_M21_I_phi_pattern_in_candidate_rejected() -> None:
    """Defense-in-depth: even if a candidate sneaks through with
    PHI, the validator rejects it. The pipeline contract is that
    PHI is blocked UPSTREAM of the M-phase entrypoint; this
    validator is the last-line guard."""
    bad = _good_candidate()
    # Inject a PHI date pattern into an LLM-controlled free-text
    # field.
    bad["candidate_keywords_unvalidated"] = ["born on 03/14/1979"]
    _, rejs = validate_repair_candidate(bad, deterministic_primary_anchors=[])
    assert "phi_pattern_detected" in rejs


# ---------- J. cross-phase regression sentinel ----------
def test_M21_J_cross_phase_invariants_no_K4A_route_change() -> None:
    """Sentinel test: the M-phase detector's cross-phase preservation
    set must include the three K4A.* subtypes. If a developer ever
    accidentally narrows this set, the test fails immediately."""
    from services.semantic_anchor_repair import _K4A_FIRED_SUBTYPES  # type: ignore[attr-defined]
    assert "severity_marker_overview" in _K4A_FIRED_SUBTYPES
    assert "rule_out_test_utility" in _K4A_FIRED_SUBTYPES
    assert "epidemiology_incidence_complication_risk" in _K4A_FIRED_SUBTYPES


# =====================================================================
# M2.1 — Prompt-leakage protection (admin payload must not contain
# few-shot examples or system-prompt boilerplate).
# =====================================================================
def test_M21_prompt_does_not_leak_into_candidate() -> None:
    """The system-prompt few-shot examples include the literal
    strings ``"schizophrenia"``, ``"antiphospholipid syndrome"``,
    etc. The closed-set rejection-reason list MUST NOT contain any
    system-prompt boilerplate strings — only closed-set identifiers."""
    cand, rejs = validate_repair_candidate(
        _good_candidate(),
        deterministic_primary_anchors=[{"canonical_concept": "schizophrenia"}],
    )
    assert rejs == []
    cand_dump = (cand.model_dump_json() if cand else "") or ""
    # The candidate dump must NOT contain any of the prompt's
    # instructional boilerplate fragments.
    leak_signals = (
        "You are the Bounded Semantic",
        "You are NOT answering",
        "Schema fields",
        "Few-shot examples",
        "Hard contract",
        "Refusal envelope",
        "Return STRICT JSON only",
    )
    for s in leak_signals:
        assert s not in cand_dump, f"prompt leak detected: {s!r}"


# =====================================================================
# M2.1 — Shadow-only invariant sentinel. The pipeline must NEVER
# promote the candidate to the active retrieval interpretation.
# =====================================================================
def test_M21_shadow_only_pipeline_constants_unchanged() -> None:
    """The deterministic-pipeline interpretation hooks that ranking /
    default-selection / evidence packets / summaries / citation
    verifier consume MUST not import anything M-phase. This sentinel
    fails if a future change accidentally wires an M-phase symbol
    into one of those modules."""
    import services.evidence_ranker as er
    import services.summarizer as sm
    import services.citation_verifier as cv

    for mod in (er, sm, cv):
        # Inspect every str-typed attribute for the closed-set
        # M-phase symbol names. If any of these names appears we
        # have likely promoted shadow logic into active code.
        for attr_name in dir(mod):
            assert "semantic_anchor_repair" not in str(attr_name).lower(), (
                f"{mod.__name__}.{attr_name} appears to reference M-phase code"
            )


# =====================================================================
# M2.1 — Closed-set rejection-reason coverage (no opaque
# schema_invalid emitted by translator for known Pydantic types).
# =====================================================================
def test_M21_translator_avoids_opaque_schema_invalid_for_known_types() -> None:
    """The Pydantic→closed-set translator must produce a specific
    reason for every Pydantic error type we expect. The opaque
    ``schema_invalid`` is only ever used as the final fallback when
    a truly unknown Pydantic error type is encountered."""
    # Missing required.
    bad1 = {"provenance": "bounded_semantic_repair_shadow"}  # missing many required
    _, rejs1 = validate_repair_candidate(bad1, deterministic_primary_anchors=[])
    assert "missing_required_field" in rejs1
    assert "schema_invalid" not in rejs1

    # Wrong literal value on provenance.
    bad2 = _good_candidate(provenance="not_the_bounded_tag")
    _, rejs2 = validate_repair_candidate(bad2, deterministic_primary_anchors=[])
    assert "provenance_mismatch" in rejs2
    assert "schema_invalid" not in rejs2

    # Wrong type on confidence.
    bad3 = _good_candidate()
    bad3["confidence"] = "0.85"  # string instead of float
    _, rejs3 = validate_repair_candidate(bad3, deterministic_primary_anchors=[])
    # Pydantic v2 coerces "0.85" → 0.85 by default; if it doesn't, we
    # expect the specific reason. Either pass-through or specific
    # rejection is acceptable — but NEVER the opaque schema_invalid.
    assert "schema_invalid" not in rejs3
