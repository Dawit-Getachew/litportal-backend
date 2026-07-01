"""Phase RankPrecision-1 — central condition anchor precision gate tests."""
from __future__ import annotations

import pytest

from services.central_condition_anchor_gate import evaluate, _extract_title_sites
from services.picott_ranker import _check_concepts

PROSTATE_PICOTT = {
    "type_of_question": "prognosis",
    "question_subtype": "stage_specific_prognosis",
    "prognosis_stage_cancer_condition_anchor": "prostate cancer",
    "must_have_concepts": [
        {"concept": "prostate cancer", "role": "condition",
         "synonyms": ["prostate carcinoma", "prostatic neoplasm"],
         "required_for_high_relevance": True},
    ],
}


def _picott_for(condition):
    return {"type_of_question": "prognosis",
            "prognosis_stage_cancer_condition_anchor": condition,
            "must_have_concepts": [{"concept": condition, "role": "condition"}]}


@pytest.mark.parametrize("title", [
    "Tumour budding: a promising parameter in colorectal cancer.",
    "LncRNA PCAT6 is a predictor of poor prognosis of colorectal cancer.",
    "Healthcare Research in Renal Cell Cancer: 7-Year Data.",
    "Molecular Pathogenesis of Colorectal Cancer.",
])
def test_offdisease_title_fires_wrong_condition(title):
    v = evaluate(title, PROSTATE_PICOTT)
    assert v["fires"] is True
    assert v["reason"] == "wrong_condition"
    assert v["target_site"] == "prostate"


@pytest.mark.parametrize("title", [
    "Focal Therapy in Localized Prostate Cancer: a systematic review.",
    "Five-year survival of patients with late-stage prostate cancer.",
    "Active surveillance vs treatment for prostate carcinoma.",
])
def test_target_in_title_does_not_fire(title):
    assert evaluate(title, PROSTATE_PICOTT)["fires"] is False


def test_multicancer_registry_no_specific_site_not_fired():
    # No single competing cancer in the title → not wrong_condition.
    v = evaluate("Five-year cancer survival by stage at diagnosis in Canada", PROSTATE_PICOTT)
    assert v["fires"] is False
    assert v["title_sites"] == []


def test_both_target_and_other_in_title_not_fired():
    v = evaluate("Comparative survival of prostate and colorectal cancer", PROSTATE_PICOTT)
    assert v["fires"] is False  # target present in title


def test_standalone_cancer_term_fires():
    v = evaluate("Prognostic factors in metastatic melanoma", PROSTATE_PICOTT)
    assert v["fires"] is True and v["target_site"] == "prostate"


def test_non_cancer_question_never_fires():
    picott = {"type_of_question": "prognosis",
              "must_have_concepts": [{"concept": "idiopathic pulmonary fibrosis", "role": "condition"}]}
    assert evaluate("Survival in colorectal cancer", picott)["fires"] is False


def test_colon_target_blocks_prostate_titled():
    picott = _picott_for("colon cancer")
    assert evaluate("Outcomes in metastatic prostate cancer", picott)["fires"] is True
    assert evaluate("Tumour budding in colorectal cancer", picott)["fires"] is False


def test_extract_title_sites():
    assert _extract_title_sites("Renal Cell Cancer registry") == {"kidney"}
    assert _extract_title_sites("Tumour budding in colorectal cancer") == {"colorectal"}
    assert _extract_title_sites("Five-year cancer survival by stage") == set()
    assert _extract_title_sites("Metastatic melanoma prognosis") == {"skin"}


def test_check_concepts_integration_offdisease_capped():
    # Off-disease title, target only incidental in abstract → cap<=30 + wrong_condition.
    cc = _check_concepts(
        title="Tumour budding: a promising parameter in colorectal cancer.",
        abstract="Tumour budding predicts prognosis; similar findings reported in prostate cancer.",
        ingest_full_text="",
        picott=PROSTATE_PICOTT,
    )
    assert cc.cap <= 30
    assert "wrong_condition" in cc.cap_reasons


def test_check_concepts_integration_ontarget_not_capped():
    cc = _check_concepts(
        title="Focal therapy in localized prostate cancer.",
        abstract="Outcomes and overall survival in men with prostate cancer.",
        ingest_full_text="",
        picott=PROSTATE_PICOTT,
    )
    assert "wrong_condition" not in cc.cap_reasons
