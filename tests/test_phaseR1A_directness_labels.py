"""Phase R1A — Part 8 — directness label catalog test sweep.

Pure unit tests against ``services/r1a_directness_labels.py``.
Deterministic. NO LLM. NO Mongo. NO network.
"""
from __future__ import annotations

import pytest

from services.r1a_directness_labels import (
    PATHO_NEGATIVE_LABELS,
    PATHO_POSITIVE_LABELS,
    PROC_NEGATIVE_LABELS,
    PROC_POSITIVE_LABELS,
    R1A_ALL_LABELS,
    SHADOW_LABEL_WEIGHTS,
    compute_shadow_score,
    detect_labels_for_shape,
    detect_patho_mechanism_labels,
    detect_procedure_complication_labels,
)


# =====================================================================
# Closed-set sanity — label sets + weight table.
# =====================================================================
def test_R1A_label_sets_are_disjoint_pos_neg_pathophysiology():
    pos = set(PATHO_POSITIVE_LABELS)
    neg = set(PATHO_NEGATIVE_LABELS)
    assert pos.isdisjoint(neg), "pathophysiology pos+neg label sets must be disjoint"


def test_R1A_label_sets_are_disjoint_pos_neg_procedure_complication():
    pos = set(PROC_POSITIVE_LABELS)
    neg = set(PROC_NEGATIVE_LABELS)
    assert pos.isdisjoint(neg)


def test_R1A_weight_table_exhausts_label_universe():
    assert set(SHADOW_LABEL_WEIGHTS.keys()) == set(R1A_ALL_LABELS)


def test_R1A_positive_weights_are_positive_and_negative_weights_are_negative():
    for lab, w in SHADOW_LABEL_WEIGHTS.items():
        is_pos = (
            lab in PATHO_POSITIVE_LABELS
            or lab in PROC_POSITIVE_LABELS
        )
        is_neg = (
            lab in PATHO_NEGATIVE_LABELS
            or lab in PROC_NEGATIVE_LABELS
        )
        assert is_pos != is_neg, f"{lab} must be exactly one of pos/neg"
        if is_pos:
            assert w > 0, f"positive label {lab} must have positive weight"
        else:
            assert w < 0, f"negative label {lab} must have negative weight"


# =====================================================================
# pathophysiology_mechanism — positive detections.
# =====================================================================
def test_R1A_patho_direct_positive_hit():
    article = {
        "title": (
            "Mechanisms of pruritus in cholestasis: understanding and "
            "treating the itch."
        ),
        "abstract": (
            "Lysophosphatidic acid and autotaxin contribute to the "
            "pathophysiology of cholestatic pruritus."
        ),
        "publication_types_json": ["Review"],
    }
    labels = detect_patho_mechanism_labels(article)
    assert "pathophysiology_mechanism_direct" in labels
    assert "mechanism_review" in labels
    assert "mechanism_with_named_process" in labels


def test_R1A_patho_treatment_is_indirect_negative():
    article = {
        "title": "Randomized controlled trial of IBAT inhibitor for cholestatic pruritus",
        "abstract": "Double blind placebo controlled trial of an IBAT inhibitor as treatment.",
        "publication_types_json": ["Randomized Controlled Trial"],
    }
    labels = detect_patho_mechanism_labels(article)
    assert "treatment_modality_indirect_for_mechanism" in labels


def test_R1A_patho_no_mechanism_focus_negative():
    article = {
        "title": "Approach to cholestatic liver disease",
        "abstract": "Overview of common bile-duct disorders for the general clinician.",
        "publication_types_json": [],
    }
    labels = detect_patho_mechanism_labels(article)
    assert "missing_mechanism_focus" in labels


# =====================================================================
# procedure_complication_risk — high-precision rule + label coverage.
# =====================================================================
def test_R1A_proc_high_precision_requires_both_anchors_positive_hit():
    article = {
        "title": "Incidence of pneumothorax after central venous catheterization in the ICU",
        "abstract": (
            "Prospective cohort of central line placement complications including "
            "pneumothorax in critical care units."
        ),
        "publication_types_json": ["Prospective Cohort Study"],
    }
    labels = detect_procedure_complication_labels(article)
    assert "procedure_complication_direct" in labels
    assert "procedure_complication_incidence" in labels
    assert "procedure_complication_cohort" in labels


def test_R1A_proc_complication_missing_procedure_anchor_emits_negative():
    article = {
        "title": "Pneumothorax in critical care: a review",
        "abstract": "Discussion of pneumothorax management without specific procedural exposure.",
        "publication_types_json": ["Review"],
    }
    labels = detect_procedure_complication_labels(article)
    # No procedure anchor → "procedure_missing_exposure".
    assert "procedure_missing_exposure" in labels
    # Direct positive label should NOT fire (high-precision rule).
    assert "procedure_complication_direct" not in labels


def test_R1A_proc_imaging_followup_is_indirect_negative():
    article = {
        "title": "Post-procedure imaging utility for line confirmation",
        "abstract": "Routine chest radiograph after central line placement.",
        "publication_types_json": [],
    }
    labels = detect_procedure_complication_labels(article)
    assert "diagnostic_followup_indirect_for_complication_risk" in labels


# =====================================================================
# detect_labels_for_shape dispatcher.
# =====================================================================
@pytest.mark.parametrize("shape,detector_called", [
    ("pathophysiology_mechanism", True),
    ("procedure_complication_risk", True),
    ("procedure_complication", True),
    ("unknown_shape", False),
    ("", False),
])
def test_R1A_dispatcher_routes_to_correct_detector(shape, detector_called):
    article = {
        "title": "Some title with mechanism and pathophysiology language",
        "abstract": "abstract content",
        "publication_types_json": [],
    }
    out = detect_labels_for_shape(shape, article)
    if detector_called:
        assert isinstance(out, list)
    else:
        assert out == []


# =====================================================================
# compute_shadow_score — deterministic, positive labels raise score,
# negative labels lower it.
# =====================================================================
def test_R1A_shadow_score_positive_labels_raise_score():
    base = compute_shadow_score(base_score=10.0, labels=[])
    boosted = compute_shadow_score(
        base_score=10.0,
        labels=["pathophysiology_mechanism_direct", "mechanism_review"],
    )
    assert boosted > base


def test_R1A_shadow_score_negative_labels_lower_score():
    base = compute_shadow_score(base_score=10.0, labels=[])
    lowered = compute_shadow_score(
        base_score=10.0,
        labels=["treatment_modality_indirect_for_mechanism", "missing_mechanism_focus"],
    )
    assert lowered < base


def test_R1A_shadow_score_unknown_label_is_no_op():
    out = compute_shadow_score(base_score=5.0, labels=["totally_made_up_label"])
    assert out == 5.0
