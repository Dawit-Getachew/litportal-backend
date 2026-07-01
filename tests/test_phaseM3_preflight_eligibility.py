"""Phase M3-preflight — eligibility decision unit tests.

DETERMINISTIC. NO LLM. NO RETRIEVAL.

Asserts against the closed-set
``EligibilityDecision.to_diagnostics()`` payload — i.e., what the
admin diagnostics endpoint surfaces. Each test injects a synthetic
candidate + config so the underlying ``is_semantic_repair_active_eligible``
function can be exercised without standing up the full pipeline.
"""
from __future__ import annotations

from services.semantic_repair_active_config import (
    SemanticRepairActiveConfig,
)
from services.semantic_repair_active_eligibility import (
    ELIGIBILITY_BLOCK_REASONS,
    ELIGIBILITY_PASS_REASONS,
    EligibilityDecision,
    is_semantic_repair_active_eligible,
)


def _default_cfg(**overrides) -> SemanticRepairActiveConfig:
    fields = dict(
        enabled=False,
        env_allowlist=("dev", "preview"),
        shapes_allowlist=(
            "etiology_risk",
            "pathophysiology_mechanism",
            "procedure_complication",
        ),
        max_latency_ms=5000,
        kill_switch=True,
        prod_allow_active=False,
        current_env="preview",
    )
    fields.update(overrides)
    return SemanticRepairActiveConfig(**fields)


def _good_candidate() -> dict:
    return {
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
        "optional_or_ranking_only_concepts": [],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": [],
        "candidate_mesh_terms_unvalidated": ["Schizophrenia"],
        "semantic_repair_reason": "missing_risk_etiology_framework",
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
    }


# =====================================================================
# Closed-set taxonomy sanity.
# =====================================================================
def test_M3_pass_and_block_reasons_are_distinct_closed_sets() -> None:
    assert ELIGIBILITY_PASS_REASONS.isdisjoint(ELIGIBILITY_BLOCK_REASONS)
    assert len(ELIGIBILITY_PASS_REASONS) >= 13
    assert len(ELIGIBILITY_BLOCK_REASONS) >= 13


def test_M3_decision_returns_closed_set_strings_only() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=True,
        candidate_rejection_reasons=[],
        candidate_dict=_good_candidate(),
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert isinstance(dec, EligibilityDecision)
    for r in dec.pass_reasons:
        assert r in ELIGIBILITY_PASS_REASONS, f"non-closed-set pass: {r!r}"
    for r in dec.block_reasons:
        assert r in ELIGIBILITY_BLOCK_REASONS, f"non-closed-set block: {r!r}"


# =====================================================================
# In M3-preflight, EVERY call returns eligible=False.
# =====================================================================
def test_M3_preflight_eligible_is_always_false_with_default_config() -> None:
    cfg = _default_cfg()  # enabled=false, kill_switch=true
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=True,
        candidate_rejection_reasons=[],
        candidate_dict=_good_candidate(),
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert dec.eligible is False
    assert "active_flag_disabled" in dec.block_reasons
    assert "kill_switch_engaged" in dec.block_reasons
    # Clinical gates passed → would-be-eligible is True.
    assert dec.would_be_eligible_if_flags_enabled is True


def test_M3_preflight_would_be_eligible_true_for_canonical_A_F_G_shapes() -> None:
    cfg = _default_cfg()
    for reason, intent, anchor in [
        ("missing_risk_etiology_framework", "risk_etiology", "schizophrenia"),
        ("missing_mechanism_pathophysiology_framework", "mechanism", "antiphospholipid syndrome"),
        ("missing_complication_after_procedure_split", "complication_after_procedure", "pacemaker"),
    ]:
        cand = _good_candidate()
        cand["primary_intent"] = intent
        cand["required_anchors"] = [anchor]
        cand["semantic_repair_reason"] = reason
        dec = is_semantic_repair_active_eligible(
            session_blocked_phi=False,
            detector_triggered=True,
            trigger_reasons=[reason],
            candidate_validated=True,
            candidate_rejection_reasons=[],
            candidate_dict=cand,
            deterministic_primary_anchors=[{"canonical_concept": anchor}],
            question_subtype=None,
            latency_ms=2700,
            config=cfg,
        )
        assert dec.eligible is False  # always blocked by ops flags
        assert dec.would_be_eligible_if_flags_enabled is True, f"{reason} should be clinically eligible"


# =====================================================================
# Clinical block reasons.
# =====================================================================
def test_M3_blocked_when_phi_blocked() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=True,
        detector_triggered=False,
        trigger_reasons=[],
        candidate_validated=False,
        candidate_rejection_reasons=[],
        candidate_dict=None,
        deterministic_primary_anchors=[],
        question_subtype=None,
        latency_ms=0,
        config=cfg,
    )
    assert "phi_blocked" in dec.block_reasons
    assert dec.eligible is False
    assert dec.would_be_eligible_if_flags_enabled is False


def test_M3_blocked_when_detector_did_not_trigger() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=False,
        trigger_reasons=[],
        candidate_validated=False,
        candidate_rejection_reasons=[],
        candidate_dict=None,
        deterministic_primary_anchors=[],
        question_subtype=None,
        latency_ms=0,
        config=cfg,
    )
    assert "detector_did_not_trigger" in dec.block_reasons
    assert dec.eligible is False
    assert dec.would_be_eligible_if_flags_enabled is False


def test_M3_blocked_when_candidate_rejected_schema_invalid() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=False,
        candidate_rejection_reasons=["missing_required_field"],
        candidate_dict=None,
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert "candidate_schema_invalid" in dec.block_reasons
    assert dec.would_be_eligible_if_flags_enabled is False


def test_M3_blocked_when_candidate_unsafe_or_refused() -> None:
    cfg = _default_cfg()
    for rej in [
        ["refused"],
        ["phi_pattern_detected"],
        ["article_reference_detected"],
        ["answer_text_detected"],
        ["unsafe_text_detected"],
        ["unapproved_source_referenced"],
    ]:
        dec = is_semantic_repair_active_eligible(
            session_blocked_phi=False,
            detector_triggered=True,
            trigger_reasons=["missing_risk_etiology_framework"],
            candidate_validated=False,
            candidate_rejection_reasons=rej,
            candidate_dict=None,
            deterministic_primary_anchors=[],
            question_subtype=None,
            latency_ms=2700,
            config=cfg,
        )
        assert "candidate_refused_or_unsafe" in dec.block_reasons


def test_M3_blocked_when_generic_only_required_anchor() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=False,
        candidate_rejection_reasons=["generic_only_required_anchor"],
        candidate_dict=None,
        deterministic_primary_anchors=[],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert "generic_only_required_anchor" in dec.block_reasons


def test_M3_blocked_when_central_anchor_removed() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=False,
        candidate_rejection_reasons=["would_remove_central_anchor"],
        candidate_dict=None,
        deterministic_primary_anchors=[
            {"canonical_concept": "pacemaker"},
        ],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert "central_anchor_removed" in dec.block_reasons


def test_M3_blocked_when_unvalidated_mesh_marked_validated() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=False,
        candidate_rejection_reasons=["unvalidated_mesh_marked_validated"],
        candidate_dict=None,
        deterministic_primary_anchors=[],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert "unvalidated_mesh_treated_as_validated" in dec.block_reasons


def test_M3_blocked_when_route_owned_by_k4a_specialty() -> None:
    """K4A.* routes own their shapes — M3 must NEVER promote on
    these sessions. Cross-phase preservation gate."""
    cfg = _default_cfg()
    for subtype in [
        "severity_marker_overview",
        "rule_out_test_utility",
        "epidemiology_incidence_complication_risk",
    ]:
        dec = is_semantic_repair_active_eligible(
            session_blocked_phi=False,
            detector_triggered=True,
            trigger_reasons=["missing_risk_etiology_framework"],
            candidate_validated=True,
            candidate_rejection_reasons=[],
            candidate_dict=_good_candidate(),
            deterministic_primary_anchors=[
                {"canonical_concept": "schizophrenia"},
            ],
            question_subtype=subtype,
            latency_ms=2700,
            config=cfg,
        )
        assert "route_owned_by_k4a_specialty" in dec.block_reasons, (
            f"K4A.* subtype {subtype} must block M3 active eligibility"
        )
        assert dec.would_be_eligible_if_flags_enabled is False


def test_M3_blocked_when_latency_above_threshold() -> None:
    cfg = _default_cfg(max_latency_ms=3000)
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=True,
        candidate_rejection_reasons=[],
        candidate_dict=_good_candidate(),
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
        question_subtype=None,
        latency_ms=4500,  # exceeds 3000
        config=cfg,
    )
    assert "latency_above_threshold" in dec.block_reasons
    # Latency is an OPERATIONAL knob — clinical eligibility (the
    # would_be answer) does NOT depend on it. Operators can tune
    # ``SEMANTIC_REPAIR_ACTIVE_MAX_LATENCY_MS`` upward without
    # changing clinical safety, so latency must NOT gate the
    # would_be_eligible_if_flags_enabled answer.
    assert dec.would_be_eligible_if_flags_enabled is True


def test_M3_blocked_when_shape_not_in_allowlist() -> None:
    """``prognosis`` is NOT in the default 3-shape allowlist. The
    candidate is otherwise clean — it must still be blocked because
    M3 has not been benchmarked on this shape."""
    cfg = _default_cfg()  # default allowlist excludes prognosis
    cand = _good_candidate()
    cand["primary_intent"] = "prognosis"
    cand["semantic_repair_reason"] = "missing_prognosis_framework"
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_prognosis_framework"],
        candidate_validated=True,
        candidate_rejection_reasons=[],
        candidate_dict=cand,
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert "shape_not_in_allowlist" in dec.block_reasons


def test_M3_blocked_when_shape_explicitly_added_to_allowlist() -> None:
    """Inverse — when the operator adds ``prognosis`` to the
    allowlist, the shape gate passes."""
    cfg = _default_cfg(
        shapes_allowlist=(
            "etiology_risk", "pathophysiology_mechanism",
            "procedure_complication", "prognosis",
        ),
    )
    cand = _good_candidate()
    cand["primary_intent"] = "prognosis"
    cand["semantic_repair_reason"] = "missing_prognosis_framework"
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_prognosis_framework"],
        candidate_validated=True,
        candidate_rejection_reasons=[],
        candidate_dict=cand,
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert "active_shape_in_allowlist" in dec.pass_reasons
    assert "shape_not_in_allowlist" not in dec.block_reasons


# =====================================================================
# Operational gates.
# =====================================================================
def test_M3_blocked_when_production_env_and_prod_allow_off() -> None:
    cfg = SemanticRepairActiveConfig(
        enabled=True, kill_switch=False,
        env_allowlist=("dev", "preview", "production"),
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=False,
        current_env="production",
    )
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=True,
        candidate_rejection_reasons=[],
        candidate_dict=_good_candidate(),
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert "production_env_fail_closed" in dec.block_reasons
    assert dec.eligible is False


def test_M3_eligible_only_when_all_flags_aligned() -> None:
    cfg = SemanticRepairActiveConfig(
        enabled=True, kill_switch=False,
        env_allowlist=("dev", "preview", "production"),
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=True,
        current_env="production",
    )
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=True,
        candidate_rejection_reasons=[],
        candidate_dict=_good_candidate(),
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia"},
        ],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    assert dec.eligible is True
    assert dec.would_be_eligible_if_flags_enabled is True


# =====================================================================
# to_diagnostics — closed-set admin payload shape.
# =====================================================================
def test_M3_to_diagnostics_emits_closed_set_keys() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=True,
        trigger_reasons=["missing_risk_etiology_framework"],
        candidate_validated=True,
        candidate_rejection_reasons=[],
        candidate_dict=_good_candidate(),
        deterministic_primary_anchors=[{"canonical_concept": "schizophrenia"}],
        question_subtype=None,
        latency_ms=2700,
        config=cfg,
    )
    payload = dec.to_diagnostics()
    assert set(payload.keys()) == {
        "eligible",
        "pass_reasons",
        "block_reasons",
        "would_be_eligible_if_flags_enabled",
        "shape",
    }
    assert isinstance(payload["pass_reasons"], list)
    assert isinstance(payload["block_reasons"], list)
    assert payload["shape"] == "etiology_risk"
