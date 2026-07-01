"""Phase 2A — deterministic clinical question parser tests (no LLM)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure LLM is disabled for these tests.
os.environ["LLM_PROVIDER"] = "none"

from services.clinical_question_parser import ClinicalQuestionParserService  # noqa: E402
from services.llm_provider import llm_provider  # noqa: E402


@pytest.mark.asyncio
async def test_treatment_question_classified():
    svc = ClinicalQuestionParserService()
    out = await svc.parse("Is empagliflozin compared to placebo effective for HFpEF?")
    assert out["clinical_question_type"] == "treatment"
    assert out["mesh_terms"] == []  # never hallucinated
    assert "empagliflozin" in out["drug_names_and_classes"]
    assert "SGLT2 inhibitors" in out["drug_names_and_classes"]
    assert "deterministic_fallback" in out["rationale"]


@pytest.mark.asyncio
async def test_diagnosis_question_classified():
    out = await ClinicalQuestionParserService().parse(
        "What is the diagnostic accuracy of D-dimer for pulmonary embolism?"
    )
    assert out["clinical_question_type"] == "diagnosis"


@pytest.mark.asyncio
async def test_harm_question_classified():
    out = await ClinicalQuestionParserService().parse(
        "Risk of bleeding with apixaban vs warfarin in elderly patients"
    )
    assert out["clinical_question_type"] == "harm"
    assert "apixaban" in out["drug_names_and_classes"]


@pytest.mark.asyncio
async def test_prognosis_question_classified():
    out = await ClinicalQuestionParserService().parse(
        "What is the 5-year mortality after STEMI with primary PCI?"
    )
    assert out["clinical_question_type"] == "prognosis"


@pytest.mark.asyncio
async def test_guideline_question_classified():
    out = await ClinicalQuestionParserService().parse(
        "What does the guideline recommend for atrial fibrillation rate vs rhythm control?"
    )
    assert out["clinical_question_type"] == "guideline"


@pytest.mark.asyncio
async def test_screening_question_classified():
    out = await ClinicalQuestionParserService().parse(
        "Screening for colorectal cancer in average-risk adults"
    )
    assert out["clinical_question_type"] == "screening"


@pytest.mark.asyncio
async def test_no_llm_provider_called_when_unconfigured():
    """If LLM is not configured, complete_json must never be invoked."""
    # Force-disable the provider for the duration of this test so the assertion
    # is meaningful regardless of the runtime env (Phase 3 ships with the
    # Emergent gateway enabled by default).
    with patch.object(llm_provider, "is_configured", return_value=False), \
         patch.object(llm_provider, "complete_json") as mock_call:
        await ClinicalQuestionParserService().parse(
            "Best evidence for SGLT2 inhibitors in HFpEF?"
        )
        mock_call.assert_not_called()


@pytest.mark.asyncio
async def test_synonyms_expanded_for_known_acronyms():
    out = await ClinicalQuestionParserService().parse(
        "AF anticoagulation in elderly with HTN"
    )
    syns = out["synonyms"]
    # at least one known expansion present
    assert any("atrial fibrillation" in s or "hypertension" in s for s in syns)
