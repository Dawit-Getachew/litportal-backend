"""Phase M4.1.3 — No-Primary-Entity Rescue Path tests (deterministic)."""
from __future__ import annotations

import pytest

from services.m4_active_pipeline import (
    validate_no_primary_rescue_candidate,
    is_m4_no_primary_rescue_eligible,
    RESCUE_NO_PRIMARY_MEANINGFUL,
    RESCUE_REJECT_NO_REQUIRED_ANCHOR,
    RESCUE_REJECT_GENERIC_ANCHOR,
    RESCUE_REJECT_FULL_LITERAL_ANCHOR,
    RESCUE_REJECT_NO_CLINICAL_ENTITY,
    RESCUE_REJECT_NOT_ACCEPTED,
    RESCUE_REJECT_PROJECTION_FLAGGED,
    RESCUE_REJECT_LOW_CONFIDENCE,
)


def _clean_cand(**ov):
    base = {
        "primary_intent": "prognosis_natural_history",
        "clinical_problem": ["colon cancer"],
        "stage": ["stage II"], "biomarker": [], "population": [],
        "medication": [], "procedure": [], "index_test": [],
        "outcome": ["survival"], "mechanism": [],
        "required_anchors": ["colon cancer"],
        "confidence": 0.9,
    }
    base.update(ov)
    return base


# ---- validate_no_primary_rescue_candidate ----

def test_rescue_accepts_clean_candidate():
    ok, reason = validate_no_primary_rescue_candidate(_clean_cand())
    assert ok is True
    assert reason == RESCUE_NO_PRIMARY_MEANINGFUL


def test_rescue_rejects_empty_required():
    ok, reason = validate_no_primary_rescue_candidate(_clean_cand(required_anchors=[]))
    assert ok is False
    assert reason == RESCUE_REJECT_NO_REQUIRED_ANCHOR


@pytest.mark.parametrize("generic", ["dose", "treatment", "risk", "prognosis", "mechanism"])
def test_rescue_rejects_generic_anchor(generic):
    ok, reason = validate_no_primary_rescue_candidate(
        _clean_cand(required_anchors=[generic], clinical_problem=[generic]))
    assert ok is False
    assert reason == RESCUE_REJECT_GENERIC_ANCHOR


def test_rescue_rejects_full_literal_anchor():
    ok, reason = validate_no_primary_rescue_candidate(
        _clean_cand(required_anchors=["survival rate for stage two colon cancer patients"]))
    assert ok is False
    assert reason == RESCUE_REJECT_FULL_LITERAL_ANCHOR


def test_rescue_rejects_no_clinical_entity():
    ok, reason = validate_no_primary_rescue_candidate(
        _clean_cand(required_anchors=["foobar"], clinical_problem=[],
                    medication=[], procedure=[], index_test=[]))
    assert ok is False
    assert reason == RESCUE_REJECT_NO_CLINICAL_ENTITY


def test_rescue_rejects_low_confidence():
    ok, reason = validate_no_primary_rescue_candidate(_clean_cand(confidence=0.2))
    assert ok is False
    assert reason == RESCUE_REJECT_LOW_CONFIDENCE


def test_rescue_accepts_medication_entity():
    ok, reason = validate_no_primary_rescue_candidate(
        _clean_cand(clinical_problem=[], required_anchors=["warfarin"],
                    medication=["warfarin"], outcome=["bleeding"]))
    assert ok is True
    assert reason == RESCUE_NO_PRIMARY_MEANINGFUL


# ---- is_m4_no_primary_rescue_eligible ----

def _block(accepted=True, proj="passed", cand=None):
    return {
        "accepted": accepted,
        "candidate": cand if cand is not None else _clean_cand(),
        "m4_0_1_projection": {"projection_validation_status": proj},
    }


def test_eligible_accepted_clean():
    ok, reason = is_m4_no_primary_rescue_eligible(_block())
    assert ok is True
    assert reason == RESCUE_NO_PRIMARY_MEANINGFUL


def test_eligible_not_accepted():
    ok, reason = is_m4_no_primary_rescue_eligible(_block(accepted=False))
    assert ok is False
    assert reason == RESCUE_REJECT_NOT_ACCEPTED


def test_eligible_projection_flagged():
    ok, reason = is_m4_no_primary_rescue_eligible(_block(proj="flagged"))
    assert ok is False
    assert reason == RESCUE_REJECT_PROJECTION_FLAGGED


def test_eligible_generic_question_blocked():
    # "What is the dose?" style — generic anchor, no clinical entity.
    cand = _clean_cand(required_anchors=["dose"], clinical_problem=[],
                       medication=[], procedure=[], index_test=[])
    ok, reason = is_m4_no_primary_rescue_eligible(_block(cand=cand))
    assert ok is False
    assert reason in (RESCUE_REJECT_GENERIC_ANCHOR,)
