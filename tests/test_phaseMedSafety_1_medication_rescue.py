"""Phase MedSafety-1 — medication dosing/safety subject detection rescue tests.

Validates:
  * Deterministic medication scanner (generic / brand / safe-typo) detection.
  * Dosing/safety intent signal detection.
  * Anchor shape (drug_or_drug_class / drug_pack).
  * Lexicon contains the failing-case drugs.
  * The rescued picott (question_subtype=drug_safety) + drug anchor routes to
    the ``drug_safety`` intent + ``PECO`` framework via the K1 builder.
  * Generic-only / non-medication questions detect NO medication.
"""
from __future__ import annotations

import pytest

from services.medication_subject_rescue import (
    build_medication_anchor,
    detect_medication,
    is_dosing_safety_question,
)
from services.pipeline import (
    _k1_build_clinical_question_map,
    _k1_framework_for_intent,
    _k1_intent_for_picott,
)


# =====================================================================
# 1. Scanner — positives (a medication IS present).
# =====================================================================
@pytest.mark.parametrize("q,generic", [
    ("What is safe dose of dexmedetomidine (Precedex) for agitation in patients who are not intubated?", "dexmedetomidine"),
    ("What is the dosing of Precedex for procedural sedation in non-intubated adults?", "dexmedetomidine"),
    ("What are adverse effects of dexmedetomidine in ICU patients?", "dexmedetomidine"),
    ("Is dexmedetomidine safe for agitation in non-intubated patients?", "dexmedetomidine"),
    ("What is the dose of quetiapine for agitation?", "quetiapine"),
    ("What is the dose of Seroquel XR?", "quetiapine"),
    ("dexmedotamidine dosing in the ICU", "dexmedetomidine"),  # safe typo
    ("dose of atorvastatin for primary prevention", "atorvastatin"),  # rxnorm fallback
])
def test_medsafety1_scanner_detects_medication(q, generic):
    m = detect_medication(q)
    assert m is not None, q
    assert m["generic"] == generic


# =====================================================================
# 2. Scanner — negatives (NO medication present → must clarify).
# =====================================================================
@pytest.mark.parametrize("q", [
    "What is the dose?",
    "What is the safe dose?",
    "Dose for agitation?",
    "What is the mechanism of pruritus in cholestatic liver disease?",
    "What are the markers of severity in acute pancreatitis?",
    "How should suspected pulmonary embolism be evaluated?",
])
def test_medsafety1_scanner_no_false_positive(q):
    assert detect_medication(q) is None, q


# =====================================================================
# 3. Dosing/safety signal + anchor shape + brand synonym.
# =====================================================================
def test_medsafety1_dosing_safety_signal():
    assert is_dosing_safety_question("What is safe dose of dexmedetomidine?") is True
    assert is_dosing_safety_question("adverse effects of dexmedetomidine") is True
    assert is_dosing_safety_question("epidemiology of sepsis") is False


def test_medsafety1_anchor_shape_and_brand_synonym():
    m = detect_medication("safe dose of dexmedetomidine (Precedex)")
    anchor = build_medication_anchor(m)
    assert anchor["entity_type"] == "drug_or_drug_class"
    assert anchor["role"] == "intervention"
    assert anchor["source"] == "drug_pack"
    assert anchor["canonical_concept"] == "dexmedetomidine"
    assert "Precedex" in anchor["synonyms"]   # brand synonym captured
    assert anchor["required_for_search"] is True


def test_medsafety1_lexicon_has_failing_case_drugs():
    # Brand resolves to generic.
    assert detect_medication("Precedex infusion")["generic"] == "dexmedetomidine"
    assert detect_medication("dexmedetomidine titration")["generic"] == "dexmedetomidine"


# =====================================================================
# 4. K1 routing — rescued picott routes to drug_safety / PECO.
# =====================================================================
def _rescued_anchor():
    return build_medication_anchor(detect_medication("safe dose of dexmedetomidine (Precedex)"))


def test_medsafety1_intent_routes_to_drug_safety():
    picott = {"question_subtype": "drug_safety", "type_of_question": "other"}
    anchors_out = {"primary_anchors": [_rescued_anchor()], "anchor_failure_reason": None}
    assert _k1_intent_for_picott(picott, anchors_out) == "drug_safety"
    assert _k1_framework_for_intent("drug_safety") == "PECO"


def test_medsafety1_cqm_build_captures_drug_safety_intent():
    picott = {
        "question_subtype": "drug_safety",
        "type_of_question": "other",
        "must_have_concepts": [{
            "concept": "dexmedetomidine", "role": "intervention",
            "synonyms": ["Precedex"], "required_for_high_relevance": True,
        }],
    }
    anchors_out = {
        "primary_anchors": [_rescued_anchor()],
        "anchor_failure_reason": None,
        "medication_subject_rescue_applied": True,
    }
    cqm = _k1_build_clinical_question_map(
        normalized_question="safe dose of dexmedetomidine precedex for agitation in non intubated patients",
        original_question="x",
        picott_payload=picott,
        anchors_out=anchors_out,
        resolver_diag={"called": False, "accepted": False},
        signal_summary={},
        clarification_only=False,
    )
    assert cqm is not None
    assert cqm.get("primary_intent") == "drug_safety"
    assert "PECO" in (cqm.get("frameworks") or [])
    assert cqm.get("clarification_required") is False
    assert cqm.get("interpretation_provenance") == "deterministic_concept_pack"
