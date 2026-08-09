"""Phase R1C — anchor-label catalogue + detector unit tests.

Pure tests, no I/O. Validate the closed-set procedure-outcome
labels + mechanism-target labels and their deterministic
detection against synthetic article payloads.
"""
from __future__ import annotations

import pytest

from services.r1c_anchor_labels import (
    CASE_ANCHOR_CONFIG,
    MECHANISM_TARGET_LABELS,
    PROCEDURE_OUTCOME_LABELS,
    R1C_ALL_LABELS,
    R1C_LABEL_MULTIPLIERS,
    compute_r1c_multiplier,
    detect_mechanism_target_labels,
    detect_procedure_outcome_labels,
    detect_r1c_labels,
)


# =====================================================================
# Closed-set catalogues.
# =====================================================================
def test_R1C_procedure_outcome_labels_closed_set():
    assert set(PROCEDURE_OUTCOME_LABELS) == {
        "procedure_complication_outcome_exact",
        "procedure_complication_outcome_missing",
        "procedure_complication_wrong_outcome",
        "procedure_complication_general_not_outcome_specific",
    }


def test_R1C_mechanism_target_labels_closed_set():
    assert set(MECHANISM_TARGET_LABELS) == {
        "mechanism_target_exact",
        "mechanism_target_missing",
        "disease_only_indirect_for_mechanism",
        "treatment_mentions_mechanism_but_not_primary",
        "mechanism_review_with_treatment_discussion",
    }


def test_R1C_all_labels_is_union():
    assert R1C_ALL_LABELS == frozenset(
        list(PROCEDURE_OUTCOME_LABELS) + list(MECHANISM_TARGET_LABELS)
    )


def test_R1C_multiplier_table_covers_all_labels():
    assert set(R1C_LABEL_MULTIPLIERS.keys()) == set(R1C_ALL_LABELS)
    for lab, m in R1C_LABEL_MULTIPLIERS.items():
        # All multipliers MUST be ≤ 1.0 (flag-only or deprioritize).
        assert m <= 1.0
        assert m >= 0.0


def test_R1C_case_anchor_config_closed_set():
    assert set(CASE_ANCHOR_CONFIG.keys()) == {
        "r1a-case-A-cholestatic-pruritus",
        "r1a-case-B-pcos-insulin-resistance",
        "r1a-case-C-amyloid-diastolic-dysfunction",
        "r1a-case-D-cvc-pneumothorax",
    }


# =====================================================================
# Procedure-complication outcome detection — Case D.
# =====================================================================
def test_R1C_procedure_outcome_exact_when_both_anchors_present():
    article = {
        "title": "Incidence of pneumothorax after central venous catheterization",
        "abstract": "Prospective cohort of central venous catheter pneumothorax events.",
    }
    labels = detect_procedure_outcome_labels(
        case_id="r1a-case-D-cvc-pneumothorax", article=article,
    )
    assert labels == ["procedure_complication_outcome_exact"]


def test_R1C_procedure_wrong_outcome_when_dvt_or_bsi():
    article = {
        "title": "Deep vein thrombosis after central venous catheter placement",
        "abstract": "Risk of DVT in cancer patients with a CVC.",
    }
    labels = detect_procedure_outcome_labels(
        case_id="r1a-case-D-cvc-pneumothorax", article=article,
    )
    assert labels == ["procedure_complication_wrong_outcome"]


def test_R1C_procedure_wrong_outcome_bloodstream_infection():
    article = {
        "title": "Central line-associated bloodstream infection epidemiology",
        "abstract": "CLABSI rates in ICU patients with central venous catheter.",
    }
    labels = detect_procedure_outcome_labels(
        case_id="r1a-case-D-cvc-pneumothorax", article=article,
    )
    assert labels == ["procedure_complication_wrong_outcome"]


def test_R1C_procedure_outcome_missing_when_procedure_only():
    article = {
        "title": "Central venous catheter placement techniques",
        "abstract": "Bedside ultrasound for CVC placement; technical considerations.",
    }
    labels = detect_procedure_outcome_labels(
        case_id="r1a-case-D-cvc-pneumothorax", article=article,
    )
    # No outcome term and no wrong outcome → outcome_missing
    # (no "overview"/"review"/"general" markers so not general).
    assert labels == ["procedure_complication_outcome_missing"]


def test_R1C_procedure_general_not_outcome_specific():
    article = {
        "title": "Overview of central venous catheter complications",
        "abstract": "General review of CVC complications.",
    }
    labels = detect_procedure_outcome_labels(
        case_id="r1a-case-D-cvc-pneumothorax", article=article,
    )
    assert labels == [
        "procedure_complication_general_not_outcome_specific",
    ]


def test_R1C_procedure_returns_empty_for_non_procedure_cases():
    article = {"title": "Mechanism of itch", "abstract": "pruritus"}
    labels = detect_procedure_outcome_labels(
        case_id="r1a-case-A-cholestatic-pruritus", article=article,
    )
    assert labels == []


# =====================================================================
# Mechanism target detection — Cases A/B/C.
# =====================================================================
def test_R1C_mechanism_target_exact_for_A():
    article = {
        "title": "Mechanism of pruritus in cholestatic liver disease",
        "abstract": "Autotaxin pathway and lysophosphatidic acid mediate the itch.",
    }
    labels = detect_mechanism_target_labels(
        case_id="r1a-case-A-cholestatic-pruritus", article=article,
    )
    assert "mechanism_target_exact" in labels
    # Not a review + no treatment → no mixed-signal flag.
    assert "mechanism_review_with_treatment_discussion" not in labels


def test_R1C_mechanism_review_with_treatment_discussion_flag():
    article = {
        "title": "Review of cholestatic pruritus: mechanisms and itch pathway",
        "abstract": (
            "Comprehensive review of pathophysiology including treatment "
            "trials of IBAT inhibitors and bile acid sequestrants."
        ),
    }
    labels = detect_mechanism_target_labels(
        case_id="r1a-case-A-cholestatic-pruritus", article=article,
    )
    assert "mechanism_target_exact" in labels
    assert "mechanism_review_with_treatment_discussion" in labels


def test_R1C_disease_only_indirect_when_no_mechanism_target():
    article = {
        "title": "Approach to cholestatic liver disease",
        "abstract": "General overview of common bile-duct disorders.",
    }
    labels = detect_mechanism_target_labels(
        case_id="r1a-case-A-cholestatic-pruritus", article=article,
    )
    # No mechanism target + no treatment focus.
    assert labels == ["disease_only_indirect_for_mechanism"]


def test_R1C_treatment_not_primary_when_treatment_in_disease_paper():
    article = {
        "title": "Randomized controlled trial of rifampicin in cholestasis",
        "abstract": (
            "Double-blind placebo-controlled trial of rifampicin "
            "pharmacotherapy in cholestatic patients."
        ),
    }
    labels = detect_mechanism_target_labels(
        case_id="r1a-case-A-cholestatic-pruritus", article=article,
    )
    assert labels == ["treatment_mentions_mechanism_but_not_primary"]


def test_R1C_mechanism_target_missing_when_mech_term_no_disease():
    article = {
        "title": "Pruritus and itch pathway in psoriasis",
        "abstract": "Itch mechanism in dermatology.",
    }
    labels = detect_mechanism_target_labels(
        case_id="r1a-case-A-cholestatic-pruritus", article=article,
    )
    assert labels == ["mechanism_target_missing"]


# =====================================================================
# Per-case detection — Case C (amyloid + diastolic dysfunction).
# =====================================================================
def test_R1C_caseC_mechanism_target_exact():
    article = {
        "title": "Mechanism of impaired relaxation in cardiac amyloidosis",
        "abstract": "Myocardial stiffness and diastolic dysfunction in ATTR-CM.",
    }
    labels = detect_mechanism_target_labels(
        case_id="r1a-case-C-amyloid-diastolic-dysfunction", article=article,
    )
    assert "mechanism_target_exact" in labels


def test_R1C_caseC_disease_only_indirect():
    article = {
        "title": "Cardiac amyloidosis: a clinical review",
        "abstract": "Overview of transthyretin amyloid clinical features.",
    }
    labels = detect_mechanism_target_labels(
        case_id="r1a-case-C-amyloid-diastolic-dysfunction", article=article,
    )
    assert labels == ["disease_only_indirect_for_mechanism"]


# =====================================================================
# Multiplier helper — deterministic product.
# =====================================================================
def test_R1C_multiplier_product_of_present_labels():
    m = compute_r1c_multiplier([
        "mechanism_target_missing",   # 0.5
        "treatment_mentions_mechanism_but_not_primary",  # 0.4
    ])
    assert m == pytest.approx(0.5 * 0.4, rel=1e-4)


def test_R1C_multiplier_unknown_label_is_one():
    m = compute_r1c_multiplier(["totally_made_up_label"])
    assert m == 1.0


def test_R1C_multiplier_mechanism_target_exact_is_one():
    # ``exact`` carries no penalty — multiplier 1.0.
    assert compute_r1c_multiplier(["mechanism_target_exact"]) == 1.0


def test_R1C_multiplier_review_with_treatment_is_flag_only():
    assert compute_r1c_multiplier(
        ["mechanism_target_exact", "mechanism_review_with_treatment_discussion"]
    ) == 1.0


# =====================================================================
# Shape dispatcher.
# =====================================================================
def test_R1C_detect_r1c_labels_dispatches_by_shape():
    article = {
        "title": "Incidence of pneumothorax after central venous catheter",
        "abstract": "pneumothorax following CVC.",
    }
    out = detect_r1c_labels(
        case_id="r1a-case-D-cvc-pneumothorax",
        article=article,
        shape="procedure_complication_risk",
    )
    assert "procedure_complication_outcome_exact" in out


def test_R1C_detect_r1c_labels_returns_empty_for_non_eligible_shape():
    out = detect_r1c_labels(
        case_id="r1a-case-A-cholestatic-pruritus",
        article={"title": "x", "abstract": "y"},
        shape="therapy",
    )
    assert out == []
