"""Phase M4.6 — deterministic prognosis/stage cancer retrieval rescue tests."""
from __future__ import annotations

import pytest

from services.prognosis_stage_cancer_rescue import (
    apply_rescue,
    extract_components,
    is_prognosis_stage_cancer_question,
)

POSITIVE = [
    ("What is the prognosis of stage 3 prostate cancer?", "prostate cancer", "stage 3"),
    ("What is the prognosis for stage III prostate cancer?", "prostate cancer", "stage 3"),
    ("What is the survival rate for stage II colon cancer?", "colon cancer", "stage 2"),
    ("What is the recurrence risk after stage I colon cancer?", "colon cancer", "stage 1"),
    ("What is the prognosis for HER2 positive stage 2 breast cancer?", "breast cancer", "stage 2"),
]

NEGATIVE = [
    "What is colon cancer?",
    "What is the dose?",
    "Mr James presented with new onset headache. What is the differential diagnosis?",
    "What is safe dose of dexmedetomidine (Precedex) for agitation in patients who are not intubated?",
    "What are the markers of severity in acute pancreatitis?",
    "Does a negative MRSA nasal swab rule out MRSA pneumonia?",
    "What is the incidence of aspiration pneumonia in patients with GCS 8 or less?",
    "What are the guidelines for colorectal cancer screening in average-risk adults?",
]


@pytest.mark.parametrize("q,cancer,stage", POSITIVE)
def test_extract_components_positive(q, cancer, stage):
    comp = extract_components(q)
    assert comp is not None
    assert comp["cancer"] == cancer
    assert comp["stage"] is not None
    assert comp["stage"]["label"] == stage


@pytest.mark.parametrize("q", [p[0] for p in POSITIVE])
def test_is_prognosis_question_true(q):
    assert is_prognosis_stage_cancer_question(q) is True


@pytest.mark.parametrize("q", NEGATIVE)
def test_is_prognosis_question_false(q):
    assert is_prognosis_stage_cancer_question(q) is False


def test_apply_rescue_rewrites_prostate():
    # Simulates the observed BAD parse: glued full-phrase condition concept.
    picott = {
        "type_of_question": "background",
        "question_subtype": "disease_overview",
        "question_scope": "background",
        "confidence": 0.4,
        "must_have_concepts": [
            {"concept": "prognosis of stage 3 prostate cancer", "role": "condition",
             "synonyms": ["prognosis of stage 3 prostate cancer"],
             "required_for_high_relevance": True},
        ],
        "pico": {"population": [], "outcomes_or_target_diagnosis": []},
    }
    diag = apply_rescue(picott, "What is the prognosis of stage 3 prostate cancer?")
    assert diag["applied"] is True
    assert diag["cancer"] == "prostate cancer"
    assert picott["type_of_question"] == "prognosis"
    assert picott["question_subtype"] == "stage_specific_prognosis"
    assert picott["question_scope"] == "foreground"
    # condition anchor is now the cancer, not the glued phrase
    conditions = [m for m in picott["must_have_concepts"] if m.get("role") == "condition"]
    assert len(conditions) == 1
    assert conditions[0]["concept"] == "prostate cancer"
    assert conditions[0]["required_for_high_relevance"] is True
    # glued full-phrase concept removed
    assert all("prognosis of" not in (m["concept"].lower())
               for m in picott["must_have_concepts"])
    # stage is a ranking concept, NOT a must_have
    assert all("stage" not in (m["concept"].lower())
               for m in picott["must_have_concepts"])
    rank = [r["concept"].lower() for r in picott.get("ranking_concepts") or []]
    assert any("stage" in r for r in rank)
    # outcomes ride as should_have + legacy pico outcomes
    sh = [s["concept"].lower() for s in picott.get("should_have_concepts") or []]
    assert "overall survival" in sh and "prognosis" in sh
    assert "prostate cancer" in [p.lower() for p in picott["pico"]["population"]]
    assert "overall survival" in [o.lower() for o in picott["pico"]["outcomes_or_target_diagnosis"]]
    assert picott["confidence"] >= 0.85


def test_apply_rescue_breast_her2_strips_biomarker_and_stage():
    picott = {"must_have_concepts": [
        {"concept": "her2 positive stage 2 breast cancer", "role": "condition"}]}
    diag = apply_rescue(picott, "What is the prognosis for HER2 positive stage 2 breast cancer?")
    assert diag["applied"] is True
    assert diag["cancer"] == "breast cancer"
    cond = [m for m in picott["must_have_concepts"] if m.get("role") == "condition"]
    assert cond[0]["concept"] == "breast cancer"


def test_apply_rescue_colon_synonyms():
    picott = {"must_have_concepts": []}
    apply_rescue(picott, "What is the survival rate for stage II colon cancer?")
    cond = [m for m in picott["must_have_concepts"] if m.get("role") == "condition"][0]
    syns = [s.lower() for s in cond["synonyms"]]
    assert "colorectal" in " ".join(syns)


@pytest.mark.parametrize("q", NEGATIVE)
def test_apply_rescue_negative_no_change(q):
    picott = {"type_of_question": "background", "must_have_concepts": []}
    diag = apply_rescue(picott, q)
    assert diag["applied"] is False
    assert picott["type_of_question"] == "background"


def test_apply_rescue_skips_prior_phase_route():
    picott = {"question_subtype": "severity_marker_overview", "must_have_concepts": []}
    diag = apply_rescue(picott, "What is the prognosis of stage 3 prostate cancer?")
    assert diag["applied"] is False
    assert diag["reason"] == "prior_phase_owns_route"
