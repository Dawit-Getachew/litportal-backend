"""Phase K4A.4.1 — Regression test for the live-preview symptoms
that the 42 K4A.4 tests in ``test_phaseK4A4_incidence_aspiration_gcs``
did NOT catch.

The bug, as observed by e1_tester on the live preview for the
reproducer ``"What is the incidence of aspiration pneumonia in
patients with GCS 8 or less?"``:

  1. The interpretation panel showed outcome = ``"incidence"`` instead
     of ``"aspiration pneumonia"``. Root cause: the K4A.4 pipeline
     override was injecting ``{concept: "incidence", role: "outcome"}``
     as a must_have_concept which became a primary_clinical_anchor
     with ``entity_type=biomarker_or_lab`` + ``role=outcome``. The
     ``picott_term_mapping_service._anchor_to_role`` mapped that to
     the ``outcomes_target_diagnosis`` bucket — which is what the UI
     "Outcome" panel renders. Meanwhile ``aspiration pneumonia``
     stayed in the ``population_problem`` bucket and was rendered
     under "Population".

  2. ``patients`` was rendered under "Clinical action" / intent.
     Root cause: it lived in
     ``primary_clinical_anchors_json.generic_modifiers`` and
     ``picott_term_mapping_service._modifier_to_role("patients")``
     fell through the lexical hint guards and returned
     ``"clinical_action"`` (the default).

Coverage gap: the existing K4A.4 tests asserted against the internal
``picott_payload`` state (must_have_concepts, pico) but NOT against
the downstream ``picott_term_mapping_json`` projection that the UI
actually consumes. K4A.4.1 closes that gap.

These tests run the real ``build_picott_term_mapping`` against a
minimal session_doc that mirrors what the live K4A.4 pipeline persists.
"""
from __future__ import annotations

import pytest

from services.picott_term_mapping_service import build_picott_term_mapping
from services.picott_term_mapping_service import (
    _modifier_to_role,
    _DISPLAY_ONLY_GENERIC_TERMS,
)


def _k4a4_session_doc() -> dict:
    """A minimal session_doc that mirrors the live K4A.4 pipeline
    state for the reproducer query, AFTER the K4A.4 pipeline override
    has fired. Mirrors the live session JSON snapshot captured during
    the K4A.4.1 investigation.
    """
    return {
        "picott_json": {
            "question_subtype": "epidemiology_incidence_complication_risk",
            "type_of_question": "epidemiology",
            "incidence_outcome_anchor": "aspiration pneumonia",
            "incidence_risk_context_anchor": "low Glasgow Coma Scale",
            "pico": {
                "population": [
                    "low Glasgow Coma Scale", "Glasgow Coma Scale",
                    "GCS", "low GCS", "coma", "comatose",
                ],
                "outcomes_or_target_diagnosis": [
                    "aspiration pneumonia", "aspiration pneumonitis",
                    "incidence", "prevalence", "risk",
                ],
            },
            "must_have_concepts": [
                {
                    "concept": "aspiration pneumonia",
                    "role": "condition",
                    "synonyms": ["aspiration pneumonitis"],
                    "required_for_high_relevance": True,
                },
                {
                    "concept": "low Glasgow Coma Scale",
                    "role": "exposure",
                    "synonyms": ["Glasgow Coma Scale", "GCS"],
                    "required_for_high_relevance": True,
                },
            ],
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [
                {
                    "canonical_concept": "aspiration pneumonia",
                    "text": "aspiration pneumonia",
                    "entity_type": "disease_or_problem",
                    "role": "population",
                    "required_for_search": True,
                    "synonyms": ["aspiration pneumonitis"],
                    "confidence": 0.9,
                },
                {
                    "canonical_concept": "low Glasgow Coma Scale",
                    "text": "low Glasgow Coma Scale",
                    "entity_type": "exposure",
                    "role": "exposure",
                    "required_for_search": True,
                    "synonyms": ["Glasgow Coma Scale", "GCS", "coma"],
                    "confidence": 0.9,
                },
            ],
            "generic_modifiers": [
                "patients",  # ← the K4A.4.1 culprit; must be dropped
            ],
            "anchor_failure_reason": None,
        },
        "role_repairs_json": [],
        "ontology_lifts_json": [],
        "multi_intent_json": {},
    }


# =====================================================================
# REGRESSION GUARD #1 — outcome anchor in projected mapping must be
# aspiration pneumonia, NOT incidence.
# =====================================================================
def test_k4a4_projection_outcome_is_aspiration_pneumonia_not_incidence() -> None:
    mapping = build_picott_term_mapping(_k4a4_session_doc())
    outcomes = mapping.fields["outcomes_target_diagnosis"]
    concepts = [e.canonical_concept.lower() for e in outcomes]
    assert "aspiration pneumonia" in concepts, (
        f"K4A.4.1 regression: outcomes_target_diagnosis must include "
        f"'aspiration pneumonia'. Got: {concepts}"
    )
    assert "incidence" not in concepts, (
        f"K4A.4.1 regression: outcomes_target_diagnosis must NOT include "
        f"'incidence' (it is the method, not the outcome). Got: {concepts}"
    )


# =====================================================================
# REGRESSION GUARD #2 — generic ``patients`` is NEVER surfaced as a
# clinical_action / intent entry in the projected mapping.
# =====================================================================
def test_k4a4_projection_patients_never_clinical_action() -> None:
    mapping = build_picott_term_mapping(_k4a4_session_doc())
    actions = mapping.fields["clinical_action"]
    concepts = [e.canonical_concept.lower() for e in actions]
    assert "patients" not in concepts, (
        f"K4A.4.1 regression: clinical_action must NOT contain "
        f"'patients' (it is a generic population label, not a "
        f"clinical action). Got: {concepts}"
    )
    assert "patient" not in concepts, (
        f"K4A.4.1 regression: clinical_action must NOT contain "
        f"'patient'. Got: {concepts}"
    )


# =====================================================================
# REGRESSION GUARD #3 — the population_problem bucket should still
# carry the exposure / risk-context anchor (low Glasgow Coma Scale).
# =====================================================================
def test_k4a4_projection_low_gcs_present_in_intervention_exposure() -> None:
    mapping = build_picott_term_mapping(_k4a4_session_doc())
    iei = mapping.fields["intervention_exposure_index_test"]
    concepts = [e.canonical_concept.lower() for e in iei]
    assert "low glasgow coma scale" in concepts, (
        f"K4A.4.1 regression: 'low Glasgow Coma Scale' must appear in "
        f"the intervention/exposure bucket. Got: {concepts}"
    )


# =====================================================================
# REGRESSION GUARD #4 — _modifier_to_role contract for closed-set
# generic population labels.
# =====================================================================
@pytest.mark.parametrize("term", [
    "patients", "patient",
    "adults", "adult",
    "child", "children",
    "infant", "infants",
    "person", "people",
    "individual", "individuals",
    "subject", "subjects",
    "case", "cases",
    "population", "populations",
])
def test_modifier_to_role_drops_generic_population_terms(term: str) -> None:
    assert _modifier_to_role(term) == "_drop_", (
        f"K4A.4.1 regression: '{term}' must be dropped from the "
        f"picott_term_mapping projection, not routed to "
        f"clinical_action. Got: {_modifier_to_role(term)!r}"
    )


def test_modifier_to_role_still_routes_outcome_hints() -> None:
    # Sanity — the K4A.4.1 sentinel must NOT swallow legitimate
    # outcome hints.
    assert _modifier_to_role("adverse effects") == "outcomes_target_diagnosis"
    assert _modifier_to_role("side effects") == "outcomes_target_diagnosis"


def test_modifier_to_role_still_routes_action_hints() -> None:
    # Sanity — legitimate clinical-action hints still route to
    # clinical_action.
    assert _modifier_to_role("first-line treatment") == "clinical_action"
    assert _modifier_to_role("dose") == "clinical_action"


def test_display_only_generic_terms_closed_set_non_empty() -> None:
    assert "patient" in _DISPLAY_ONLY_GENERIC_TERMS
    assert "patients" in _DISPLAY_ONLY_GENERIC_TERMS
    assert "adults" in _DISPLAY_ONLY_GENERIC_TERMS
    # Closed-set sanity — at least these 10+ classes are covered.
    assert len(_DISPLAY_ONLY_GENERIC_TERMS) >= 10


# =====================================================================
# REGRESSION GUARD #5 — non-K4A.4 question_subtypes must NOT be
# affected by the K4A.4-subtype-aware re-routing in _anchor_to_role.
# =====================================================================
def test_disease_anchor_in_non_k4a4_subtype_stays_in_population() -> None:
    """When the question_subtype is NOT
    ``epidemiology_incidence_complication_risk``, a disease anchor
    must remain in the ``population_problem`` bucket (no cross-phase
    leak from the K4A.4.1 re-route)."""
    session_doc = {
        "picott_json": {
            "question_subtype": "disease_overview",
            "type_of_question": "background_overview",
            "must_have_concepts": [],
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [{
                "canonical_concept": "psoriasis",
                "text": "psoriasis",
                "entity_type": "disease_or_problem",
                "role": "population",
                "required_for_search": True,
                "synonyms": [],
                "confidence": 0.9,
            }],
            "generic_modifiers": [],
            "anchor_failure_reason": None,
        },
        "role_repairs_json": [],
        "ontology_lifts_json": [],
        "multi_intent_json": {},
    }
    mapping = build_picott_term_mapping(session_doc)
    pp = [e.canonical_concept.lower()
          for e in mapping.fields["population_problem"]]
    od = [e.canonical_concept.lower()
          for e in mapping.fields["outcomes_target_diagnosis"]]
    assert "psoriasis" in pp, (
        f"Non-K4A.4 disease anchor must stay in population_problem. "
        f"Got pp={pp}, od={od}"
    )
    assert "psoriasis" not in od, (
        f"Non-K4A.4 disease anchor must NOT leak into "
        f"outcomes_target_diagnosis. Got od={od}"
    )
