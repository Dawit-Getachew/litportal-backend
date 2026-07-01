"""Phase M4.1.6 — read-side M4 display projection tests.

Covers the pure builder (``build_m4_interpreted_display``), the intent
humanizer, and the router helper (``_m4_applied_candidate``) that decides
whether a session's PICOTT card should render the M4 structured
interpretation instead of the degraded baseline.

Display-only behavior: no retrieval / ranking / default-selection logic
is exercised here.
"""
from __future__ import annotations

from services.interpreted_question_display import (
    NOT_SPECIFIED,
    build_m4_interpreted_display,
)
from routers.search_sessions import _m4_applied_candidate


# --- Representative APPLIED M4 candidate (colon-cancer prognosis). ----
_COLON_CANCER_CANDIDATE = {
    "primary_intent": "prognosis_natural_history",
    "secondary_intents": ["prognostic_factor"],
    "clinical_problem": ["colon cancer"],
    "stage": ["stage I"],
    "biomarker": [],
    "population": ["patients with stage I colon cancer"],
    "intervention": [],
    "exposure": [],
    "comparator": [],
    "outcome": [
        "overall survival",
        "disease-free survival",
        "recurrence",
        "mortality",
    ],
    "required_anchors": ["colon cancer"],
}


def test_builder_maps_core_m4_fields():
    base = {
        "specialty": "Oncology",
        "question_scope": "Foreground",
        "ideal_study_types": ["Cohort study"],
    }
    d = build_m4_interpreted_display(_COLON_CANCER_CANDIDATE, base)
    assert d["source"] == "m4"
    assert d["interpretation_source"] == "m4"
    assert d["has_picott"] is True
    assert d["type_of_question"] == "Prognosis / natural history"
    assert d["type_of_question_raw"] == "prognosis_natural_history"
    assert d["question_subtype"] == "Prognostic factor"
    assert d["population"] == ["patients with stage I colon cancer"]
    assert d["outcomes_or_target_diagnosis"] == [
        "overall survival",
        "disease-free survival",
        "recurrence",
        "mortality",
    ]
    # Empty M4 lists render the literal sentinel, not blank.
    assert d["intervention_or_exposure_or_index_test"] == [NOT_SPECIFIED]
    assert d["comparator_or_reference_standard"] == [NOT_SPECIFIED]
    # required_anchors become must-have chips.
    assert d["must_have_concepts"] == [
        {
            "concept": "colon cancer",
            "role": "condition",
            "synonyms": [],
            "required_for_high_relevance": True,
        }
    ]


def test_builder_preserves_base_badges():
    """Specialty / scope / ideal_study_types are NOT carried by the M4
    candidate and must survive from the base display."""
    base = {
        "specialty": "Oncology",
        "question_scope": "Foreground",
        "ideal_study_types": ["Cohort study"],
    }
    d = build_m4_interpreted_display(_COLON_CANCER_CANDIDATE, base)
    assert d["specialty"] == "Oncology"
    assert d["question_scope"] == "Foreground"
    assert d["ideal_study_types"] == ["Cohort study"]


def test_builder_population_falls_back_to_clinical_problem():
    cand = dict(_COLON_CANCER_CANDIDATE)
    cand["population"] = []
    d = build_m4_interpreted_display(cand, {})
    assert d["population"] == ["colon cancer"]


def test_builder_dedupes_intervention_sources_case_insensitive():
    cand = {
        "primary_intent": "therapy",
        "intervention": ["Apixaban"],
        "medication": ["apixaban"],
        "procedure": ["PCI"],
        "required_anchors": ["apixaban"],
    }
    d = build_m4_interpreted_display(cand, {})
    assert d["intervention_or_exposure_or_index_test"] == ["Apixaban", "PCI"]


def test_builder_unknown_intent_titlecases():
    d = build_m4_interpreted_display({"primary_intent": "some_new_intent"}, {})
    assert d["type_of_question"] == "Some new intent"


def test_builder_other_intent_is_not_specified():
    d = build_m4_interpreted_display({"primary_intent": "other"}, {})
    assert d["type_of_question"] == NOT_SPECIFIED


def test_builder_handles_non_dict_inputs():
    d = build_m4_interpreted_display(None, None)
    assert d["source"] == "m4"
    assert d["type_of_question"] == NOT_SPECIFIED
    assert d["population"] == [NOT_SPECIFIED]


# --- Router helper: which sessions get the M4 projection. ------------

def _doc_with_m4(applied: bool, candidate: dict | None) -> dict:
    return {
        "session_diagnostics_json": {
            "m4_active": {"applied": applied},
        },
        "pico_json": {
            "session_diagnostics_json": {
                "m4_full_question_interpreter": {"candidate": candidate},
            },
        },
    }


def test_helper_returns_candidate_when_applied():
    doc = _doc_with_m4(True, _COLON_CANCER_CANDIDATE)
    assert _m4_applied_candidate(doc) == _COLON_CANCER_CANDIDATE


def test_helper_returns_none_when_not_applied():
    doc = _doc_with_m4(False, _COLON_CANCER_CANDIDATE)
    assert _m4_applied_candidate(doc) is None


def test_helper_returns_none_when_no_candidate():
    doc = _doc_with_m4(True, None)
    assert _m4_applied_candidate(doc) is None


def test_helper_falls_back_to_pico_json_applied_flag():
    """Some sessions carry the applied flag only on the pico_json copy."""
    doc = {
        "session_diagnostics_json": {},
        "pico_json": {
            "session_diagnostics_json": {
                "m4_active": {"applied": True},
                "m4_full_question_interpreter": {
                    "candidate": _COLON_CANCER_CANDIDATE,
                },
            },
        },
    }
    assert _m4_applied_candidate(doc) == _COLON_CANCER_CANDIDATE


def test_helper_legacy_doc_returns_none():
    assert _m4_applied_candidate({}) is None
    assert _m4_applied_candidate({"pico_json": {}}) is None


# --- M4 PICOTT term-mapping projection. -------------------------------

from services.picott_term_mapping_service import build_m4_picott_term_mapping


def test_term_mapping_maps_m4_candidate():
    cand = dict(_COLON_CANCER_CANDIDATE)
    cand["candidate_synonyms_unvalidated"] = ["colon carcinoma", "colorectal cancer"]
    cand["optional_or_ranking_only_terms"] = ["prognosis", "survival"]
    cand["confidence"] = 0.9
    base = {
        "question_scope": "background",
        "specialty": "oncology",
        "query_blocks": [
            {"source": "pubmed", "block_name": "condition", "terms": ["colon cancer"]},
        ],
        "subdomain": None,
        "refinement_note": None,
        "post_procedure_routing": {},
    }
    m = build_m4_picott_term_mapping(cand, base)
    d = m.model_dump(mode="json")
    assert d["type_of_question"] == "prognosis_natural_history"
    assert d["question_subtype"] == "prognostic_factor"
    assert d["specialty"] == "oncology"
    # Clean anchor — NOT the baseline fused literal.
    assert d["primary_clinical_anchors"] == [
        {
            "text": "colon cancer",
            "canonical_concept": "colon cancer",
            "entity_type": "disease_or_problem",
            "role": "population",
            "confidence": 0.9,
            "required_for_search": True,
        }
    ]
    pop = [e["canonical_concept"] for e in d["fields"]["population_problem"]]
    assert pop == ["patients with stage I colon cancer"]
    outcomes = [e["canonical_concept"] for e in d["fields"]["outcomes_target_diagnosis"]]
    assert outcomes == [
        "overall survival", "disease-free survival", "recurrence", "mortality",
    ]
    # The legacy "definition / disease overview" placeholder must be gone.
    assert d["fields"]["clinical_action"] == []
    assert d["must_have_concepts"] == [
        {
            "concept": "colon cancer",
            "role": "condition",
            "required_for_high_relevance": True,
            "synonyms": ["colon carcinoma", "colorectal cancer"],
        }
    ]
    assert d["completeness"]["safe_to_search"] is True
    # Query blocks (which reflect the M4-swapped queries) are preserved.
    assert d["query_blocks"] == base["query_blocks"]


def test_term_mapping_preserves_base_specialty_and_scope():
    m = build_m4_picott_term_mapping(
        _COLON_CANCER_CANDIDATE,
        {"specialty": "neurology", "question_scope": "foreground"},
    )
    d = m.model_dump(mode="json")
    assert d["specialty"] == "neurology"
    assert d["question_scope"] == "foreground"


def test_term_mapping_handles_missing_base():
    m = build_m4_picott_term_mapping(_COLON_CANCER_CANDIDATE, None)
    d = m.model_dump(mode="json")
    assert d["type_of_question"] == "prognosis_natural_history"
    assert d["query_blocks"] == []
    assert d["question_scope"] == "foreground"

