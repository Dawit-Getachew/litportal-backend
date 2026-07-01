"""Phase 6G.3 — Background overview routing + aspect-specific review
ranking tests.

Covers:
  * Subtype override: "What is psoriasis?" → disease_overview (not
    treatment_sequencing); "first-line treatment for psoriasis"
    still → treatment_sequencing.
  * Comparison-trigger detection (used by the query compiler to allow
    AND across independent disease anchors).
  * Conjoined-background detector ("What is X and what is Y?").
  * Overview directness classifier — closed-set, deterministic.
  * Ranker directness boost + caps for disease_overview subtype.
"""
from __future__ import annotations

import pytest

from services.overview_directness_classifier import (
    classify_overview_directness,
)
from services.picott_ranker import (
    _DIRECTNESS_BOOSTS,
    _DIRECTNESS_CAPS_OVERVIEW,
    _DIRECTNESS_CAP_REASONS,
    CAP_REASONS,
)
from services.primary_clinical_anchor import (
    _question_has_treatment_trigger,
    detect_conjoined_background_questions,
    primary_clinical_anchor_service as svc,
    question_has_comparison_trigger,
)


# ---------------------------------------------------------------------
# Section A — subtype override for "What is X?" without treatment
# ---------------------------------------------------------------------


def _picott_with_psoriasis_anchor():
    return {
        "must_have_concepts": [{
            "concept": "psoriasis", "role": "condition",
            "synonyms": ["plaque psoriasis"],
            "required_for_high_relevance": True,
        }],
        "specialty": "dermatology",
        "question_subtype": "",
    }


def test_what_is_psoriasis_overrides_to_disease_overview():
    out = svc.derive(
        picott=_picott_with_psoriasis_anchor(),
        ontology_mappings=[],
        normalized_question="What is psoriasis?",
    )
    overrides = out["picott_overrides"]
    assert overrides.get("question_subtype") == "disease_overview"
    assert overrides.get("specialty") == "dermatology"
    # Existing 6G.1 invariants.
    assert overrides.get("type_of_question") == "background"
    assert overrides.get("question_scope") == "background"


def test_first_line_treatment_for_psoriasis_NOT_overridden():
    """Treatment trigger present → DO NOT force disease_overview."""
    out = svc.derive(
        picott=_picott_with_psoriasis_anchor(),
        ontology_mappings=[],
        normalized_question="first-line treatment for psoriasis",
    )
    overrides = out["picott_overrides"]
    # Either empty (no background pattern matched) or the pack hint
    # passed through (NOT forced to disease_overview).
    assert overrides.get("question_subtype") != "disease_overview"


def test_treatment_trigger_detection():
    for q in (
        "first-line treatment for psoriasis",
        "What is the first-line treatment?",
        "second-line therapy",
        "treatment algorithm for acne",
        "dose of methotrexate",
        "dosing protocol",
        "side effects of biologics",
    ):
        assert _question_has_treatment_trigger(q) is True, q
    for q in (
        "What is psoriasis?",
        "Tell me about acne vulgaris",
        "Define HIV",
        "What is ARDS?",
    ):
        assert _question_has_treatment_trigger(q) is False, q


def test_comparison_trigger_detection():
    for q in (
        "Compare psoriasis and acne vulgaris",
        "psoriasis vs acne vulgaris",
        "association between acne and psoriasis",
        "Is acne associated with psoriasis?",
        "comorbidity of psoriasis",
        "difference between rosacea and acne",
    ):
        assert question_has_comparison_trigger(q) is True, q
    for q in (
        "What is psoriasis?",
        "What is psoriasis and what is acne vulgaris?",
        "treatment of acne",
    ):
        assert question_has_comparison_trigger(q) is False, q


def test_conjoined_background_detector():
    assert detect_conjoined_background_questions(
        "What is psoriasis and what is acne vulgaris?",
    ) == ["psoriasis", "acne vulgaris"]
    assert detect_conjoined_background_questions(
        "Define HIV and define AIDS",
    ) == ["HIV", "AIDS"]
    # Negatives.
    assert detect_conjoined_background_questions(
        "What is psoriasis?",
    ) is None
    assert detect_conjoined_background_questions(
        "Compare psoriasis and acne vulgaris",
    ) is None
    assert detect_conjoined_background_questions(
        "first-line treatment for psoriasis",
    ) is None


# ---------------------------------------------------------------------
# Section B — overview directness classifier
# ---------------------------------------------------------------------


def _art(title="", abstract="", pub_types=None):
    return {
        "title": title,
        "abstract": abstract,
        "publication_types_json": list(pub_types or []),
    }


@pytest.mark.parametrize(
    "title, abstract, anchor, expected",
    [
        # General-overview: bare disease title + overview keyword.
        ("Acne vulgaris: a clinical review", "", "acne vulgaris",
         "general_overview"),
        ("Psoriasis: overview", "", "psoriasis", "general_overview"),
        # Disease-seminar: bare disease title only.
        ("Acne Vulgaris", "", "acne vulgaris", "disease_seminar"),
        ("Psoriasis", "", "psoriasis", "disease_seminar"),
        # Environmental / lifestyle.
        ("Environmental pollution and acne vulgaris", "", "acne vulgaris",
         "environmental_or_lifestyle_specific"),
        ("Mediterranean diet in psoriasis", "", "psoriasis",
         "environmental_or_lifestyle_specific"),
        ("Skin microbiome and acne", "", "acne",
         "environmental_or_lifestyle_specific"),
        # Multi-disease umbrella.
        ("Stem cells in inflammatory skin diseases", "", "psoriasis",
         "multi_disease_review"),
        ("Microbiome and skin diseases", "", "acne",
         "multi_disease_review"),
        # Pathogenesis-only.
        ("IL-17 in psoriasis pathogenesis", "", "psoriasis",
         "pathogenesis_specific_review"),
        # Treatment-specific.
        ("Biologics in psoriasis", "", "psoriasis",
         "treatment_specific_review"),
        ("Isotretinoin for acne", "", "acne",
         "treatment_specific_review"),
        # Aspect-specific (QoL / cost).
        ("Cost-effectiveness in acne management", "", "acne",
         "aspect_specific_review"),
        # Wrong-disease.
        ("Hidradenitis suppurativa: a clinical review",
         "Discussion of hidradenitis suppurativa pathology",
         "acne vulgaris", "wrong_disease"),
        # Unrelated / empty.
        ("", "", "acne vulgaris", "unrelated"),
    ],
)
def test_directness_classifier(title, abstract, anchor, expected):
    """Closed-set classifier verdicts for representative articles."""
    art = _art(title=title, abstract=abstract)
    assert classify_overview_directness(art, anchor) == expected


def test_broad_clinical_review_via_abstract_domains():
    art = _art(
        title="Comprehensive view of acne",
        abstract=(
            "We discuss epidemiology, pathogenesis, classification, "
            "diagnosis, management, and complications of acne vulgaris."
        ),
    )
    assert classify_overview_directness(art, "acne vulgaris") in (
        "general_overview", "broad_clinical_review",
    )


# ---------------------------------------------------------------------
# Section C — ranker directness boost + cap lookups
# ---------------------------------------------------------------------


def test_directness_boost_table():
    assert _DIRECTNESS_BOOSTS["general_overview"] == 25
    assert _DIRECTNESS_BOOSTS["guideline_overview"] == 20
    assert _DIRECTNESS_BOOSTS["disease_seminar"] == 20
    assert _DIRECTNESS_BOOSTS["broad_clinical_review"] == 15
    assert _DIRECTNESS_BOOSTS["aspect_specific_review"] == 0
    assert _DIRECTNESS_BOOSTS["multi_disease_review"] == 0


def test_directness_cap_table_for_disease_overview():
    assert _DIRECTNESS_CAPS_OVERVIEW["general_overview"] is None
    assert _DIRECTNESS_CAPS_OVERVIEW["guideline_overview"] is None
    assert _DIRECTNESS_CAPS_OVERVIEW["disease_seminar"] is None
    assert _DIRECTNESS_CAPS_OVERVIEW["aspect_specific_review"] == 65
    assert _DIRECTNESS_CAPS_OVERVIEW["treatment_specific_review"] == 65
    assert (
        _DIRECTNESS_CAPS_OVERVIEW["environmental_or_lifestyle_specific"]
        == 55
    )
    assert _DIRECTNESS_CAPS_OVERVIEW["multi_disease_review"] == 70


def test_cap_reasons_token_set():
    """All new directness cap reasons must be in the closed CAP_REASONS
    set so the audit / UI surfaces never log an unknown token."""
    for cr in _DIRECTNESS_CAP_REASONS.values():
        if cr is None:
            continue
        assert cr in CAP_REASONS, f"missing cap reason: {cr}"


# ---------------------------------------------------------------------
# Section D — generic-only fast-path preservation (regression)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What are the side effects?",
        "What is first-line treatment?",
        "What is the dose?",
        "What is the recommended age?",
        "What is diagnostic accuracy?",
    ],
)
def test_generic_only_fast_path_preserved(question):
    """Phase 6F.1 fast-path must STILL short-circuit on these 5
    closed-set generic-only queries after the Phase 6G.3 changes."""
    assert svc.is_generic_only(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is psoriasis?",
        "What is acne vulgaris?",
        "What is ARDS?",
        "What is septic shock?",
    ],
)
def test_background_definition_NOT_generic_only_post_6g3(question):
    """The questions we WANT to repair must NOT trip the
    is_generic_only fast-path."""
    assert svc.is_generic_only(question) is False
