"""Phase 2A.1 — PICO extraction + qtype canonicalization tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.clinical_question_parser import (  # noqa: E402
    ClinicalQuestionParserService,
    _canonicalize_qtype,
)
from models.pico import ParsedPICO  # noqa: E402


@pytest.mark.asyncio
async def test_apixaban_warfarin_treatment_pico():
    q = ("In adults with nonvalvular atrial fibrillation, how does apixaban "
         "compare with warfarin for stroke prevention and major bleeding?")
    out = await ClinicalQuestionParserService().parse(q)
    assert out["clinical_question_type"] == "treatment"
    assert "atrial fibrillation" in out["population"].lower()
    assert "adults" in out["population"].lower()
    assert out["intervention_or_exposure"].lower() == "apixaban"
    assert out["comparator"].lower() == "warfarin"
    out_lower = [o.lower() for o in out["outcomes"]]
    assert any("stroke prevention" in o for o in out_lower), out["outcomes"]
    assert any("major bleeding" in o for o in out_lower), out["outcomes"]
    drugs_lower = [d.lower() for d in out["drug_names_and_classes"]]
    assert "apixaban" in drugs_lower and "warfarin" in drugs_lower
    kc_lower = [k.lower() for k in out["key_concepts"]]
    assert any("atrial fibrillation" in k for k in kc_lower)


@pytest.mark.asyncio
async def test_sglt2_hfpef_pico():
    q = ("In adults with heart failure with preserved ejection fraction, do "
         "SGLT2 inhibitors reduce hospitalization compared with placebo?")
    out = await ClinicalQuestionParserService().parse(q)
    assert "heart failure with preserved ejection fraction" in out["population"].lower()
    assert "SGLT2 inhibitors" in out["intervention_or_exposure"] or \
        "sglt2" in out["intervention_or_exposure"].lower()
    assert out["comparator"].lower() == "placebo"
    out_lower = [o.lower() for o in out["outcomes"]]
    assert any("hospitalization" in o for o in out_lower)


@pytest.mark.asyncio
async def test_ddimer_pe_diagnosis_pico():
    q = "What is the diagnostic accuracy of D-dimer for pulmonary embolism in low-risk adults?"
    out = await ClinicalQuestionParserService().parse(q)
    assert out["clinical_question_type"] == "diagnosis"
    assert "low-risk adults" in out["population"].lower()
    assert "D-dimer" in out["intervention_or_exposure"] or \
        "d-dimer" in out["intervention_or_exposure"].lower()
    assert out["comparator"] in ("", "reference standard")
    out_lower = [o.lower() for o in out["outcomes"]]
    assert any("diagnostic accuracy" in o for o in out_lower)


@pytest.mark.asyncio
async def test_nsaid_harm_pico():
    q = ("In patients on long-term NSAIDs, what is the risk of major "
         "gastrointestinal bleeding compared with no NSAID use?")
    out = await ClinicalQuestionParserService().parse(q)
    assert out["clinical_question_type"] == "harm"
    pop_lower = out["population"].lower()
    assert "long-term nsaids" in pop_lower or "nsaids" in pop_lower
    intv_lower = out["intervention_or_exposure"].lower()
    assert ("long-term nsaids" in intv_lower or "nsaids" in intv_lower)
    assert "no nsaid use" in out["comparator"].lower()
    out_lower = [o.lower() for o in out["outcomes"]]
    assert any("gastrointestinal bleeding" in o for o in out_lower)


def test_qtype_canonicalization():
    assert _canonicalize_qtype("management") == "treatment"
    assert _canonicalize_qtype("Therapy") == "treatment"
    assert _canonicalize_qtype("drug_comparison") == "treatment"
    assert _canonicalize_qtype("drug-comparison") == "treatment"
    assert _canonicalize_qtype("diagnostic") == "diagnosis"
    assert _canonicalize_qtype("safety") == "harm"
    assert _canonicalize_qtype("risk_factor") == "etiology"
    assert _canonicalize_qtype("recommendation") == "guideline"
    assert _canonicalize_qtype("prevalence") == "epidemiology"
    assert _canonicalize_qtype("random_string") == "other"
    assert _canonicalize_qtype("") == "other"
    # identities
    for v in ("treatment", "diagnosis", "prognosis", "harm", "screening",
              "prevention", "etiology", "guideline", "epidemiology",
              "pharmacology", "other"):
        assert _canonicalize_qtype(v) == v


def test_pydantic_rejects_invalid_qtype():
    base = {
        "normalized_question": "x", "clinical_question_type": "banana",
        "specialty": "", "population": "", "intervention_or_exposure": "",
        "comparator": "", "outcomes": [], "setting": "", "time_horizon": "",
        "key_concepts": [], "mesh_terms": [], "synonyms": [],
        "drug_names_and_classes": [], "exclusions": [], "rationale": "",
    }
    with pytest.raises(Exception):
        ParsedPICO(**base)
    # valid value passes
    base["clinical_question_type"] = "treatment"
    ParsedPICO(**base)
