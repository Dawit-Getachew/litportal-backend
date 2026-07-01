"""Phase M4.4 — Specificity Preservation / Do-No-Harm Gate unit tests.

Deterministic (NO LLM, NO network). Forced M4 candidates modeled on the
preview physician adjudication B-arms. Proves that broadening which drops
or demotes central clinical concepts is BLOCKED (baseline preserved),
that genuine literal-anchor rescues which preserve every central concept
are ALLOWED, and that the K4A protected-route + candidate-not-accepted +
timeout precedences dominate.
"""
from __future__ import annotations

import pytest

from services.m4_specificity_gate import (
    FAM_BIOMARKER_STAGE_PROGNOSIS,
    FAM_ETIOLOGY,
    FAM_MECHANISM,
    FAM_PROGNOSTIC_FACTOR,
    FINAL_BASELINE_PRESERVED,
    FINAL_BLOCKED_K4A,
    FINAL_M4_APPLIED,
    BLK_BASELINE_PRECISION_VETO,
    BLK_BIOMARKER_OR_STAGE_DEMOTED,
    BLK_CANDIDATE_NOT_ACCEPTED,
    BLK_MECHANISM_CONCEPT_DROPPED,
    BLK_OUTCOME_DROPPED,
    BLK_PROGNOSTIC_FACTOR_DROPPED,
    BLK_ROUTE_OWNED_BY_K4A_SPECIALTY,
    M4_SPECIFICITY_BLOCK_REASONS,
    M4_SPECIFICITY_PASS_REASONS,
    decompose_literal_anchor,
    detect_question_family,
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
        baseline_query_family_count=4,
        m4_query_family_count=4,
        route_owned_by_k4a_specialty=False,
        candidate_accepted=True,
        upstream_timed_out=False,
    )
    base.update(kw)
    return run_specificity_gate(**base)


# ---------------------------------------------------------------------
# Group A — family detection + literal decomposition.
# ---------------------------------------------------------------------
def test_decompose_single_concept_prognosis_phrase():
    dec = decompose_literal_anchor("prognosis of idiopathic pulmonary fibrosis")
    assert dec["intent_removed"] is True
    assert dec["clinical_fragments"] == ["idiopathic pulmonary fibrosis"]


def test_decompose_two_concept_mechanism_phrase():
    dec = decompose_literal_anchor("mechanism of thrombosis in antiphospholipid syndrome")
    assert dec["intent_removed"] is True
    assert dec["clinical_fragments"] == ["thrombosis", "antiphospholipid syndrome"]


def test_decompose_etiology_with_population():
    dec = decompose_literal_anchor("causes of schizophrenia in young adults")
    assert dec["intent_removed"] is True
    # young adults is population-only and excluded from clinical fragments
    assert dec["clinical_fragments"] == ["schizophrenia"]


def test_family_prognostic_factor():
    fam = detect_question_family(
        normalized_question="does troponin elevation predict mortality in sepsis",
        baseline_type_of_question="prognostic_factor",
        baseline_question_subtype=None,
        candidate={"index_test": ["troponin"], "outcome": ["mortality"]},
    )
    assert fam == FAM_PROGNOSTIC_FACTOR


def test_family_biomarker_stage_prognosis():
    fam = detect_question_family(
        normalized_question="what is the prognosis of her2-positive stage 2 breast cancer",
        baseline_type_of_question="prognosis_natural_history",
        baseline_question_subtype=None,
        candidate={"biomarker": ["HER2-positive"], "stage": ["stage II"]},
    )
    assert fam == FAM_BIOMARKER_STAGE_PROGNOSIS


def test_family_mechanism_and_etiology():
    assert detect_question_family(
        normalized_question="what is the mechanism of thrombosis in antiphospholipid syndrome",
        baseline_type_of_question="background", baseline_question_subtype="disease_overview",
        candidate={"clinical_problem": ["antiphospholipid syndrome"]},
    ) == FAM_MECHANISM
    assert detect_question_family(
        normalized_question="what are the causes of schizophrenia in young adults",
        baseline_type_of_question="background", baseline_question_subtype="disease_overview",
        candidate={"clinical_problem": ["schizophrenia"]},
    ) == FAM_ETIOLOGY


# ---------------------------------------------------------------------
# Group B — MUST-BLOCK over-broadening (the adjudication regressions).
# ---------------------------------------------------------------------
def test_case6_troponin_sepsis_broad_blocked():
    """prognostic_factor: reducing to sepsis-only drops troponin + mortality."""
    res = _gate(
        normalized_question="does troponin elevation predict mortality in sepsis",
        baseline_type_of_question="prognostic_factor",
        baseline_must_have_concepts=[
            {"concept": "mortality", "role": "outcome"},
            {"concept": "Sepsis", "role": "other"},
            {"concept": "troponin elevation", "role": "index_test"},
        ],
        m4_candidate={
            "primary_intent": "prognostic_factor",
            "clinical_problem": ["sepsis"],
            "index_test": ["troponin"],
            "outcome": ["mortality"],
            "required_anchors": ["sepsis"],
        },
    )
    assert res["allow"] is False
    assert res["final_decision"] == FINAL_BASELINE_PRESERVED
    assert BLK_PROGNOSTIC_FACTOR_DROPPED in res["m4_specificity_block_reasons"]
    assert BLK_OUTCOME_DROPPED in res["m4_specificity_block_reasons"]


def test_case5_ipf_prognosis_broad_blocked_by_precision_veto():
    res = _gate(
        normalized_question="what is the prognosis of idiopathic pulmonary fibrosis",
        baseline_type_of_question="background",
        baseline_question_subtype="disease_overview",
        baseline_must_have_concepts=[
            {"concept": "prognosis of idiopathic pulmonary fibrosis", "role": "condition"},
        ],
        baseline_literal_anchors=["prognosis of idiopathic pulmonary fibrosis"],
        m4_candidate={
            "primary_intent": "prognosis_natural_history",
            "clinical_problem": ["idiopathic pulmonary fibrosis"],
            "outcome": ["overall survival", "mortality"],
            "required_anchors": ["idiopathic pulmonary fibrosis"],
        },
    )
    assert res["allow"] is False
    assert res["baseline_precision_veto"] is True
    assert BLK_BASELINE_PRECISION_VETO in res["m4_specificity_block_reasons"]


def test_case4_her2_stage_prognosis_broad_blocked():
    res = _gate(
        normalized_question="what is the prognosis of her2-positive stage 2 breast cancer",
        baseline_type_of_question="prognosis_natural_history",
        baseline_must_have_concepts=[
            {"concept": "HER2-positive stage 2 breast cancer", "role": "condition"},
            {"concept": "prognosis", "role": "outcome"},
        ],
        baseline_literal_anchors=["HER2-positive stage 2 breast cancer"],
        m4_candidate={
            "primary_intent": "prognosis_natural_history",
            "clinical_problem": ["breast cancer"],
            "biomarker": ["HER2-positive"],
            "stage": ["stage II"],
            "outcome": ["overall survival", "recurrence", "mortality"],
            "required_anchors": ["breast cancer"],
        },
    )
    assert res["allow"] is False
    assert res["question_family"] == FAM_BIOMARKER_STAGE_PROGNOSIS
    assert BLK_BIOMARKER_OR_STAGE_DEMOTED in res["m4_specificity_block_reasons"]


def test_case2_aps_mechanism_broad_blocked():
    res = _gate(
        normalized_question="what is the mechanism of thrombosis in antiphospholipid syndrome",
        baseline_type_of_question="background",
        baseline_question_subtype="disease_overview",
        baseline_must_have_concepts=[
            {"concept": "mechanism of thrombosis in antiphospholipid syndrome", "role": "condition"},
        ],
        baseline_literal_anchors=["mechanism of thrombosis in antiphospholipid syndrome"],
        m4_candidate={
            "primary_intent": "etiology",
            "clinical_problem": ["antiphospholipid syndrome"],
            "outcome": ["thrombosis"],
            "required_anchors": ["antiphospholipid syndrome"],
        },
    )
    assert res["allow"] is False
    assert res["question_family"] == FAM_MECHANISM
    assert BLK_MECHANISM_CONCEPT_DROPPED in res["m4_specificity_block_reasons"]


def test_case7_cholestasis_mechanism_broad_blocked():
    """cholestasis-only (drops pruritus + mechanism) must block."""
    res = _gate(
        normalized_question="what is the mechanism of pruritus in cholestasis",
        baseline_type_of_question="background",
        baseline_question_subtype="disease_overview",
        baseline_literal_anchors=["mechanism of pruritus in cholestasis"],
        m4_candidate={
            "primary_intent": "etiology",
            "clinical_problem": ["cholestasis"],
            "required_anchors": ["cholestasis"],
        },
    )
    assert res["allow"] is False
    assert BLK_MECHANISM_CONCEPT_DROPPED in res["m4_specificity_block_reasons"]


# ---------------------------------------------------------------------
# Group C — MUST-ALLOW genuine rescues (central concepts preserved).
# ---------------------------------------------------------------------
def test_case1_schizophrenia_etiology_allowed():
    res = _gate(
        normalized_question="what are the causes of schizophrenia in young adults",
        baseline_type_of_question="background",
        baseline_question_subtype="disease_overview",
        baseline_must_have_concepts=[
            {"concept": "causes of schizophrenia in young adults", "role": "condition"},
        ],
        baseline_literal_anchors=["causes of schizophrenia in young adults"],
        m4_candidate={
            "primary_intent": "etiology",
            "clinical_problem": ["schizophrenia"],
            "population": ["young adults"],
            "required_anchors": ["schizophrenia"],
        },
        baseline_query_family_count=3,
        m4_query_family_count=4,
    )
    assert res["allow"] is True
    assert res["final_decision"] == FINAL_M4_APPLIED
    assert res["question_family"] == FAM_ETIOLOGY


def test_case7_cholestasis_mechanism_allowed_when_preserved():
    """cholestasis + pruritus + mechanism term preserved → allow."""
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
    )
    assert res["allow"] is True
    assert res["final_decision"] == FINAL_M4_APPLIED


# ---------------------------------------------------------------------
# Group D — dominant precedences.
# ---------------------------------------------------------------------
def test_k4a_route_dominates():
    res = _gate(
        normalized_question="what are the markers of severity in acute pancreatitis",
        baseline_question_subtype="severity_marker_overview",
        m4_candidate={"clinical_problem": ["acute pancreatitis"], "required_anchors": ["acute pancreatitis"]},
        route_owned_by_k4a_specialty=True,
    )
    assert res["allow"] is False
    assert res["final_decision"] == FINAL_BLOCKED_K4A
    assert BLK_ROUTE_OWNED_BY_K4A_SPECIALTY in res["m4_specificity_block_reasons"]


def test_candidate_not_accepted_preserves_baseline():
    res = _gate(candidate_accepted=False, m4_candidate={})
    assert res["allow"] is False
    assert BLK_CANDIDATE_NOT_ACCEPTED in res["m4_specificity_block_reasons"]


def test_timeout_preserves_baseline():
    res = _gate(
        upstream_timed_out=True,
        m4_candidate={"clinical_problem": ["sepsis"], "required_anchors": ["sepsis"]},
    )
    assert res["allow"] is False


def test_baseline_not_degraded_preserves_baseline():
    res = _gate(
        normalized_question="what are the causes of schizophrenia",
        m4_candidate={"clinical_problem": ["schizophrenia"], "required_anchors": ["schizophrenia"]},
        baseline_degraded=False,
        baseline_degradation_reasons=[],
    )
    assert res["allow"] is False


def test_additive_mode_diversity_decrease_does_not_block():
    """Phase M4.5: in additive mode the baseline is preserved and rescue
    families are ADDED, so a smaller M4 family count is NOT a diversity
    block."""
    res = _gate(
        normalized_question="what are the causes of schizophrenia",
        m4_candidate={"clinical_problem": ["schizophrenia"], "required_anchors": ["schizophrenia"]},
        baseline_query_family_count=6,
        m4_query_family_count=2,
    )
    assert res["allow"] is True
    assert res["m4_application_mode"] == "additive_rescue"
    assert res["final_query_strategy"] == "baseline_plus_m4_rescue"
    assert res["query_family_diversity_after"] == 8
    assert res["diversity_delta"] == 2


def test_replacement_mode_diversity_decrease_blocks():
    """Legacy replacement mode (additive_mode=False) still blocks a
    diversity decrease."""
    res = _gate(
        normalized_question="what are the causes of schizophrenia",
        m4_candidate={"clinical_problem": ["schizophrenia"], "required_anchors": ["schizophrenia"]},
        baseline_query_family_count=6,
        m4_query_family_count=2,
        additive_mode=False,
    )
    assert res["allow"] is False


# ---------------------------------------------------------------------
# Group E — closed-set contract: reasons stay within the closed sets.
# ---------------------------------------------------------------------
@pytest.mark.parametrize("res", [
    _gate(
        normalized_question="does troponin elevation predict mortality in sepsis",
        baseline_type_of_question="prognostic_factor",
        m4_candidate={
            "primary_intent": "prognostic_factor",
            "clinical_problem": ["sepsis"], "index_test": ["troponin"],
            "outcome": ["mortality"], "required_anchors": ["sepsis"],
        },
    ),
    _gate(
        normalized_question="what are the causes of schizophrenia in young adults",
        m4_candidate={"clinical_problem": ["schizophrenia"], "required_anchors": ["schizophrenia"]},
        baseline_query_family_count=3, m4_query_family_count=4,
    ),
])
def test_reasons_within_closed_sets(res):
    for r in res["m4_specificity_block_reasons"]:
        assert r in M4_SPECIFICITY_BLOCK_REASONS, r
    for r in res["m4_specificity_pass_reasons"]:
        assert r in M4_SPECIFICITY_PASS_REASONS, r
