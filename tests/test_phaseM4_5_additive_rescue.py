"""Phase M4.5 — Additive, Diversity-Preserving M4 Rescue Families tests.

Deterministic (NO LLM, NO network). Proves that for genuinely degraded
literal-anchor baselines M4 now PRESERVES the baseline families and ADDS
validated rescue families (final strategy = baseline + m4_rescue), so the
M4.4 diversity guard no longer over-blocks Case 1 / Case 7 — WITHOUT
re-introducing the over-broadening harm (Cases 2/4/5/6 still protected),
and with K4A / PHI / generic / candidate-not-accepted precedences intact.
"""
from __future__ import annotations

from services.m4_specificity_gate import (
    APP_ADDITIVE_RESCUE,
    APP_BASELINE_PRESERVED,
    APP_REPLACEMENT_BLOCKED,
    BLK_BASELINE_PRECISION_VETO,
    BLK_BIOMARKER_OR_STAGE_DEMOTED,
    BLK_HIGH_PRECISION_SPECIFICITY_WOULD_DECREASE,
    BLK_MECHANISM_CONCEPT_DROPPED,
    BLK_OUTCOME_DROPPED,
    BLK_PROGNOSTIC_FACTOR_DROPPED,
    BLK_ROUTE_OWNED_BY_K4A_SPECIALTY,
    FINAL_BLOCKED_K4A,
    FINAL_M4_APPLIED,
    M4_APPLICATION_MODES,
    M4_FINAL_QUERY_STRATEGIES,
    STRAT_BASELINE_ONLY,
    STRAT_BASELINE_PLUS_M4_RESCUE,
    run_specificity_gate,
)


def _gate(**kw):
    base = dict(
        normalized_question="",
        baseline_type_of_question=None,
        baseline_question_subtype=None,
        baseline_must_have_concepts=[],
        baseline_literal_anchors=[],
        m4_candidate={},
        baseline_degraded=True,
        baseline_degradation_reasons=["full_literal_phrase_required_anchor"],
        baseline_query_family_count=6,
        m4_query_family_count=4,
        route_owned_by_k4a_specialty=False,
        candidate_accepted=True,
        upstream_timed_out=False,
        additive_mode=True,
        m4_rescue_family_names=["m4_active_b_required_AND", "m4_active_b_required_AND_with_outcome"],
    )
    base.update(kw)
    return run_specificity_gate(**base)


# ---------------------------------------------------------------------
# Group A — recovered additive rescues (the M4.5 goal).
# ---------------------------------------------------------------------
def test_case1_schizophrenia_additive_rescue_applies_despite_fewer_families():
    res = _gate(
        normalized_question="what are the causes of schizophrenia in young adults",
        baseline_type_of_question="background",
        baseline_question_subtype="disease_overview",
        baseline_literal_anchors=["causes of schizophrenia in young adults"],
        m4_candidate={
            "primary_intent": "etiology",
            "clinical_problem": ["schizophrenia"],
            "population": ["young adults"],
            "required_anchors": ["schizophrenia"],
        },
        baseline_query_family_count=6,   # baseline richer than M4
        m4_query_family_count=2,
    )
    assert res["allow"] is True
    assert res["final_decision"] == FINAL_M4_APPLIED
    assert res["m4_application_mode"] == APP_ADDITIVE_RESCUE
    assert res["baseline_families_preserved"] is True
    assert res["final_query_strategy"] == STRAT_BASELINE_PLUS_M4_RESCUE
    assert res["query_family_diversity_before"] == 6
    assert res["query_family_diversity_after"] == 8
    assert res["diversity_delta"] == 2
    assert res["added_m4_rescue_families"]


def test_case7_cholestasis_additive_rescue_applies_when_preserved():
    res = _gate(
        normalized_question="what is the mechanism of pruritus in cholestasis",
        baseline_type_of_question="background",
        baseline_question_subtype="disease_overview",
        baseline_literal_anchors=["mechanism of pruritus in cholestasis"],
        m4_candidate={
            "primary_intent": "etiology",
            "clinical_problem": ["cholestasis"],
            "outcome": ["pruritus"],
            "mechanism": ["pathophysiology"],
            "required_anchors": ["cholestasis"],
        },
        baseline_query_family_count=5,
        m4_query_family_count=3,
    )
    assert res["allow"] is True
    assert res["m4_application_mode"] == APP_ADDITIVE_RESCUE
    assert res["final_query_strategy"] == STRAT_BASELINE_PLUS_M4_RESCUE


# ---------------------------------------------------------------------
# Group B — over-broadening still protected in additive mode.
# ---------------------------------------------------------------------
def test_case6_troponin_sepsis_still_blocked_additive():
    res = _gate(
        normalized_question="does troponin elevation predict mortality in sepsis",
        baseline_type_of_question="prognostic_factor",
        m4_candidate={
            "primary_intent": "prognostic_factor",
            "clinical_problem": ["sepsis"], "index_test": ["troponin"],
            "outcome": ["mortality"], "required_anchors": ["sepsis"],
        },
    )
    assert res["allow"] is False
    assert res["m4_application_mode"] == APP_BASELINE_PRESERVED
    assert res["final_query_strategy"] == STRAT_BASELINE_ONLY
    assert BLK_PROGNOSTIC_FACTOR_DROPPED in res["m4_specificity_block_reasons"]
    assert BLK_OUTCOME_DROPPED in res["m4_specificity_block_reasons"]


def test_case5_ipf_still_blocked_by_precision_veto_additive():
    res = _gate(
        normalized_question="what is the prognosis of idiopathic pulmonary fibrosis",
        baseline_type_of_question="background",
        baseline_question_subtype="disease_overview",
        baseline_literal_anchors=["prognosis of idiopathic pulmonary fibrosis"],
        m4_candidate={
            "primary_intent": "prognosis_natural_history",
            "clinical_problem": ["idiopathic pulmonary fibrosis"],
            "outcome": ["overall survival", "mortality"],
            "required_anchors": ["idiopathic pulmonary fibrosis"],
        },
    )
    assert res["allow"] is False
    assert BLK_BASELINE_PRECISION_VETO in res["m4_specificity_block_reasons"]
    assert res["final_query_strategy"] == STRAT_BASELINE_ONLY


def test_case4_her2_still_blocked_additive():
    res = _gate(
        normalized_question="what is the prognosis of her2-positive stage 2 breast cancer",
        baseline_type_of_question="prognosis_natural_history",
        m4_candidate={
            "primary_intent": "prognosis_natural_history",
            "clinical_problem": ["breast cancer"], "biomarker": ["HER2-positive"],
            "stage": ["stage II"], "outcome": ["overall survival"],
            "required_anchors": ["breast cancer"],
        },
    )
    assert res["allow"] is False
    assert BLK_BIOMARKER_OR_STAGE_DEMOTED in res["m4_specificity_block_reasons"]


def test_case2_aps_broad_still_blocked_additive():
    res = _gate(
        normalized_question="what is the mechanism of thrombosis in antiphospholipid syndrome",
        baseline_type_of_question="background",
        baseline_question_subtype="disease_overview",
        baseline_literal_anchors=["mechanism of thrombosis in antiphospholipid syndrome"],
        m4_candidate={
            "primary_intent": "etiology",
            "clinical_problem": ["antiphospholipid syndrome"],
            "outcome": ["thrombosis"],
            "required_anchors": ["antiphospholipid syndrome"],
        },
    )
    assert res["allow"] is False
    assert BLK_MECHANISM_CONCEPT_DROPPED in res["m4_specificity_block_reasons"]


# ---------------------------------------------------------------------
# Group C — degenerate broad-only rescue + precedences.
# ---------------------------------------------------------------------
def test_degenerate_broad_only_rescue_blocked():
    """Multi-concept question whose rescue collapses to a single broad
    disease-only family → replacement_blocked."""
    res = _gate(
        normalized_question="does troponin elevation predict mortality in sepsis",
        baseline_type_of_question="prognostic_factor",
        m4_candidate={
            "primary_intent": "prognostic_factor",
            "clinical_problem": ["sepsis"], "index_test": ["troponin"],
            "outcome": ["mortality"],
            "required_anchors": ["sepsis", "troponin", "mortality"],
        },
        m4_query_family_count=1,
        m4_rescue_family_names=["m4_active_b_required_AND"],
    )
    assert res["allow"] is False
    assert BLK_HIGH_PRECISION_SPECIFICITY_WOULD_DECREASE in res["m4_specificity_block_reasons"]
    assert res["m4_application_mode"] == APP_REPLACEMENT_BLOCKED


def test_k4a_route_still_dominates_additive():
    res = _gate(
        normalized_question="markers of severity in acute pancreatitis",
        baseline_question_subtype="severity_marker_overview",
        m4_candidate={"clinical_problem": ["acute pancreatitis"], "required_anchors": ["acute pancreatitis"]},
        route_owned_by_k4a_specialty=True,
    )
    assert res["allow"] is False
    assert res["final_decision"] == FINAL_BLOCKED_K4A
    assert BLK_ROUTE_OWNED_BY_K4A_SPECIALTY in res["m4_specificity_block_reasons"]
    assert res["m4_application_mode"] == APP_BASELINE_PRESERVED


def test_candidate_not_accepted_baseline_only_additive():
    res = _gate(candidate_accepted=False, m4_candidate={})
    assert res["allow"] is False
    assert res["final_query_strategy"] == STRAT_BASELINE_ONLY
    assert res["baseline_families_preserved"] is True


def test_closed_set_contract_modes_and_strategies():
    res = _gate(
        normalized_question="what are the causes of schizophrenia in young adults",
        m4_candidate={"clinical_problem": ["schizophrenia"], "required_anchors": ["schizophrenia"]},
    )
    assert res["m4_application_mode"] in M4_APPLICATION_MODES
    assert res["final_query_strategy"] in M4_FINAL_QUERY_STRATEGIES
