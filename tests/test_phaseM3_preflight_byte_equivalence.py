"""Phase M3-preflight — K4A.* protected-route byte-equivalence
regression net.

Asserts that the M3-preflight code path (eligibility + structural
projection) DOES NOT change anything that flows into the active
deterministic compiler for sessions routed through K4A.2 / K4A.3 /
K4A.4 or for generic / PHI sessions. The unit-level proof:

  * The M-phase detector returns ``[]`` for the K4A.*-routed shapes
    (already proven in M0/M1/M2 + M2.1; restated here for the
    regression net).
  * Given empty triggers, the M-phase block writes a no-op
    ``semantic_repair_block``: ``triggered=false``, no candidate,
    no projection. The M3-preflight eligibility decision can still
    be computed against this empty block, and it MUST return
    ``would_be_eligible_if_flags_enabled=false`` so the operator
    cannot ever flip M3 active for these sessions.
  * The would-be projection module is never called for these
    sessions (no candidate → no projection).

Live byte-equivalence of Boolean query strings is asserted in the
M3-preflight closeout report against the live preview sessions.
"""
from __future__ import annotations

from services.semantic_anchor_repair import detect_repair_triggers
from services.semantic_repair_active_config import (
    SemanticRepairActiveConfig,
)
from services.semantic_repair_active_eligibility import (
    is_semantic_repair_active_eligible,
)


def _default_cfg() -> SemanticRepairActiveConfig:
    return SemanticRepairActiveConfig(
        enabled=False, kill_switch=True,
        env_allowlist=("dev", "preview"),
        shapes_allowlist=(
            "etiology_risk", "pathophysiology_mechanism",
            "procedure_complication",
        ),
        max_latency_ms=5000, prod_allow_active=False,
        current_env="preview",
    )


# ---------------------------------------------------------------
# BE-1 — K4A.2 severity-marker
# ---------------------------------------------------------------
def test_BE1_K4A2_severity_marker_detector_silent() -> None:
    triggers = detect_repair_triggers(
        normalized_question="What are the markers of severity of illness in acute pancreatitis?",
        question_subtype="severity_marker_overview",
        type_of_question="other",
        primary_anchors=[{"canonical_concept": "acute pancreatitis"}],
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert triggers == []


def test_BE1_K4A2_eligibility_blocks_with_route_owned_by_k4a_specialty() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=False,
        trigger_reasons=[],
        candidate_validated=False,
        candidate_rejection_reasons=[],
        candidate_dict=None,
        deterministic_primary_anchors=[{"canonical_concept": "acute pancreatitis"}],
        question_subtype="severity_marker_overview",
        latency_ms=0,
        config=cfg,
    )
    assert dec.eligible is False
    assert dec.would_be_eligible_if_flags_enabled is False
    assert "route_owned_by_k4a_specialty" in dec.block_reasons
    assert "detector_did_not_trigger" in dec.block_reasons


# ---------------------------------------------------------------
# BE-2 — K4A.4 incidence-aspiration-pneumonia-GCS
# ---------------------------------------------------------------
def test_BE2_K4A4_aspiration_gcs_detector_silent() -> None:
    triggers = detect_repair_triggers(
        normalized_question="What is the incidence of aspiration pneumonia in patients with GCS 8 or less?",
        question_subtype="epidemiology_incidence_complication_risk",
        type_of_question="epidemiology",
        primary_anchors=[
            {"canonical_concept": "aspiration pneumonia"},
            {"canonical_concept": "low Glasgow Coma Scale"},
        ],
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert triggers == []


def test_BE2_K4A4_eligibility_blocks_with_route_owned() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=False,
        trigger_reasons=[],
        candidate_validated=False,
        candidate_rejection_reasons=[],
        candidate_dict=None,
        deterministic_primary_anchors=[
            {"canonical_concept": "aspiration pneumonia"},
        ],
        question_subtype="epidemiology_incidence_complication_risk",
        latency_ms=0,
        config=cfg,
    )
    assert dec.eligible is False
    assert dec.would_be_eligible_if_flags_enabled is False
    assert "route_owned_by_k4a_specialty" in dec.block_reasons


# ---------------------------------------------------------------
# BE-3 — K4A.3 MRSA rule-out
# ---------------------------------------------------------------
def test_BE3_K4A3_mrsa_rule_out_detector_silent() -> None:
    triggers = detect_repair_triggers(
        normalized_question="Does a negative mrsa nasal swab rule out mrsa pneumonia?",
        question_subtype="rule_out_test_utility",
        type_of_question="diagnosis",
        primary_anchors=[{"canonical_concept": "mrsa pneumonia"}],
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert triggers == []


def test_BE3_K4A3_eligibility_blocks_with_route_owned() -> None:
    cfg = _default_cfg()
    dec = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=False,
        trigger_reasons=[],
        candidate_validated=False,
        candidate_rejection_reasons=[],
        candidate_dict=None,
        deterministic_primary_anchors=[
            {"canonical_concept": "mrsa pneumonia"},
        ],
        question_subtype="rule_out_test_utility",
        latency_ms=0,
        config=cfg,
    )
    assert dec.eligible is False
    assert "route_owned_by_k4a_specialty" in dec.block_reasons


# ---------------------------------------------------------------
# BE-4 — generic clarification (no clinical subject)
# ---------------------------------------------------------------
def test_BE4_generic_dose_detector_silent() -> None:
    triggers = detect_repair_triggers(
        normalized_question="What is the dose?",
        question_subtype=None,
        type_of_question="other",
        primary_anchors=[],
        generic_modifiers=["dose"],
        anchor_failure_reason="no_primary_clinical_entity",
    )
    assert triggers == []


def test_BE4_generic_dose_eligibility_blocks() -> None:
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
    assert dec.eligible is False
    assert dec.would_be_eligible_if_flags_enabled is False
    assert "detector_did_not_trigger" in dec.block_reasons


# ---------------------------------------------------------------
# BE-5 — PHI dominance (PHI guard wins).
# ---------------------------------------------------------------
def test_BE5_phi_dominance_eligibility_blocks_unconditionally() -> None:
    """The pipeline never reaches M-phase when the PHI guard fires,
    but the eligibility decision is callable with
    ``session_blocked_phi=True`` for defense in depth — it MUST
    return ``eligible=false`` AND
    ``would_be_eligible_if_flags_enabled=false`` regardless of
    flags."""
    cfg = SemanticRepairActiveConfig(
        enabled=True, kill_switch=False,
        env_allowlist=("dev", "preview", "production"),
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=True,
        current_env="production",  # all operational flags open
    )
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
    assert dec.eligible is False
    assert dec.would_be_eligible_if_flags_enabled is False
    assert "phi_blocked" in dec.block_reasons


# ---------------------------------------------------------------
# Aggregate sentinel — proves none of the protected routes can ever
# become eligible regardless of flag combinations.
# ---------------------------------------------------------------
def test_BE_aggregate_protected_routes_never_would_be_eligible() -> None:
    """Sentinel: even with EVERY operational flag flipped to its
    most permissive value, the K4A.* protected routes + PHI must
    NEVER reach ``would_be_eligible_if_flags_enabled=true``. This
    is the strongest assertion we can make for the regression
    net: not even an operator with full write access to the env
    can promote these sessions."""
    permissive_cfg = SemanticRepairActiveConfig(
        enabled=True, kill_switch=False,
        env_allowlist=("dev", "preview", "production"),
        shapes_allowlist=(
            "etiology_risk", "pathophysiology_mechanism",
            "procedure_complication", "prognosis",
            "epidemiology", "severity_marker", "diagnostic_ruleout",
        ),
        max_latency_ms=999999,
        prod_allow_active=True,
        current_env="production",
    )
    for blocked_subtype in [
        "severity_marker_overview",
        "rule_out_test_utility",
        "epidemiology_incidence_complication_risk",
    ]:
        dec = is_semantic_repair_active_eligible(
            session_blocked_phi=False,
            detector_triggered=False,  # detector returns [] on K4A.* per cross-phase preservation
            trigger_reasons=[],
            candidate_validated=False,
            candidate_rejection_reasons=[],
            candidate_dict=None,
            deterministic_primary_anchors=[
                {"canonical_concept": "test concept"},
            ],
            question_subtype=blocked_subtype,
            latency_ms=10,
            config=permissive_cfg,
        )
        assert dec.would_be_eligible_if_flags_enabled is False, (
            f"K4A.* subtype {blocked_subtype} must NEVER be eligible "
            f"even with permissive flags"
        )
