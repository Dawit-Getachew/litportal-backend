"""Phase 5D — regression test for the PICOTT concurrency corruption bug.

Root cause (pre-fix): ``ClinicalQuestionParserService.parse`` stored the raw
question on ``self._picott_raw_question`` (the module-level singleton),
then awaited an LLM call, then read the attribute back in
``_attach_picott``. Two concurrent ``parse()`` calls would clobber each
other's raw question, producing PICOTT objects keyed to the WRONG
question. Symptoms: AMI session ended up with diabetes PICOTT,
``question_scope="background"``, ``specialty="primary_care"``,
``must_have_concepts=[]``.

This test runs 3 concurrent ``parse()`` calls with very different
questions and asserts each result carries its OWN normalized question
and its OWN PICOTT must-have concepts.

To make the test reproduce the bug deterministically regardless of
whether an LLM is configured in the test env, we patch
``llm_provider.complete_json`` to await a short ``asyncio.sleep``
(this is the await window where the corruption occurred). The patch
returns ``None`` so the parser falls through to the deterministic
path — the corruption surface is the singleton ``self.*`` attribute,
not the LLM output itself.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.clinical_question_parser import clinical_question_parser  # noqa: E402


_AMI_Q = (
    "What is the optimal transfusion threshold in patients with acute "
    "myocardial infarction and anemia?"
)
_DIABETES_Q = "Tell me about diabetes."
_MILRINONE_Q = (
    "What is the dose of milrinone in subarachnoid hemorrhage with "
    "cerebral vasospasm?"
)


async def _slow_llm(system, q, schema):
    """Simulates a real LLM call so concurrent ``parse()`` invocations
    interleave their post-LLM continuation. Returns ``None`` so the
    parser falls back to its deterministic path — exactly the same
    code path the post-LLM session takes."""
    await asyncio.sleep(0.05)
    return None


def _must_have_concepts(parsed: dict) -> set[str]:
    picott = parsed.get("picott_json") or {}
    return {
        (mh.get("concept") or "").lower()
        for mh in (picott.get("must_have_concepts") or [])
    }


@pytest.mark.asyncio
async def test_parse_is_safe_under_concurrent_calls():
    """Three concurrent parse() calls must each return their own PICOTT."""
    with patch(
        "services.clinical_question_parser.llm_provider.is_configured",
        return_value=True,
    ), patch(
        "services.clinical_question_parser.llm_provider.complete_json",
        side_effect=_slow_llm,
    ):
        results = await asyncio.gather(
            clinical_question_parser.parse(_AMI_Q),
            clinical_question_parser.parse(_DIABETES_Q),
            clinical_question_parser.parse(_MILRINONE_Q),
        )
    ami_out, diabetes_out, milrinone_out = results

    # Each PICOTT sub-payload must carry its OWN normalized question —
    # this is the field the tester saw clobbered.
    ami_picott = ami_out.get("picott_json") or {}
    diabetes_picott = diabetes_out.get("picott_json") or {}
    milrinone_picott = milrinone_out.get("picott_json") or {}

    ami_picott_nq = (ami_picott.get("normalized_question") or "").lower()
    diabetes_picott_nq = (diabetes_picott.get("normalized_question") or "").lower()
    milrinone_picott_nq = (milrinone_picott.get("normalized_question") or "").lower()

    assert "myocardial" in ami_picott_nq or "transfusion" in ami_picott_nq or \
        "anemia" in ami_picott_nq, \
        f"AMI picott_json got corrupted question: {ami_picott_nq!r}"
    assert "diabetes" in diabetes_picott_nq, \
        f"Diabetes picott_json got corrupted question: {diabetes_picott_nq!r}"
    assert "milrinone" in milrinone_picott_nq or \
        "subarachnoid" in milrinone_picott_nq, \
        f"Milrinone picott_json got corrupted question: {milrinone_picott_nq!r}"

    # No cross-contamination: the three normalized_question values
    # must all differ.
    assert ami_picott_nq != diabetes_picott_nq, \
        f"AMI/diabetes picott normalized_question collided: {ami_picott_nq!r}"
    assert ami_picott_nq != milrinone_picott_nq, \
        f"AMI/milrinone picott normalized_question collided: {ami_picott_nq!r}"
    assert diabetes_picott_nq != milrinone_picott_nq, \
        f"diabetes/milrinone picott normalized_question collided: {diabetes_picott_nq!r}"

    # Each must-have concept set must reflect its own question's anchors.
    ami_mh = _must_have_concepts(ami_out)
    milrinone_mh = _must_have_concepts(milrinone_out)
    assert any(
        token in (" ".join(ami_mh))
        for token in ("myocardial", "anemia", "transfusion", "infarction")
    ), f"AMI must-haves missing anchors: {ami_mh!r}"
    assert any(
        token in (" ".join(milrinone_mh))
        for token in ("milrinone", "subarachnoid")
    ), f"Milrinone must-haves missing anchors: {milrinone_mh!r}"
    # The two sets must NOT be equal (= no cross-contamination).
    assert ami_mh != milrinone_mh, \
        f"AMI and milrinone must-have concepts collided: {ami_mh!r} == {milrinone_mh!r}"


@pytest.mark.asyncio
async def test_parse_no_instance_state_leak():
    """Defensive: the parser singleton must not carry any per-call
    attribute (like ``_picott_raw_question``) across parse() calls."""
    with patch(
        "services.clinical_question_parser.llm_provider.is_configured",
        return_value=True,
    ), patch(
        "services.clinical_question_parser.llm_provider.complete_json",
        side_effect=_slow_llm,
    ):
        a_q = "What is the diagnostic accuracy of D-dimer for pulmonary embolism?"
        b_q = "Effect of metformin on cardiovascular outcomes."
        a_out, b_out = await asyncio.gather(
            clinical_question_parser.parse(a_q),
            clinical_question_parser.parse(b_q),
        )
    a_picott = (a_out.get("picott_json") or {}).get("normalized_question", "").lower()
    b_picott = (b_out.get("picott_json") or {}).get("normalized_question", "").lower()
    assert "d-dimer" in a_picott or "pulmonary embolism" in a_picott or \
           "diagnostic accuracy" in a_picott, \
        f"A picott corrupted: {a_picott!r}"
    assert "metformin" in b_picott or "cardiovascular" in b_picott, \
        f"B picott corrupted: {b_picott!r}"


@pytest.mark.asyncio
async def test_parse_five_concurrent_no_corruption():
    """Stress test: 5 concurrent parse() calls. Each must return its own
    normalized_question."""
    questions = [
        "What is the evidence for SGLT2 inhibitors in HFpEF?",
        "Diagnostic accuracy of troponin for acute coronary syndrome.",
        "Side effects of statins in elderly patients.",
        "Effect of CPAP on cardiovascular outcomes in OSA.",
        "Should we use albumin for sepsis resuscitation?",
    ]
    with patch(
        "services.clinical_question_parser.llm_provider.is_configured",
        return_value=True,
    ), patch(
        "services.clinical_question_parser.llm_provider.complete_json",
        side_effect=_slow_llm,
    ):
        results = await asyncio.gather(*(
            clinical_question_parser.parse(q) for q in questions
        ))

    # Each result's picott_json normalized_question must START with the
    # corresponding input's first significant token (signal-only, not
    # exact equality — spelling normalizer may have edited it).
    input_signals = [
        ["sglt2", "hfpef"],
        ["troponin", "coronary", "acs"],
        ["statin"],
        ["cpap", "osa"],
        ["albumin", "sepsis"],
    ]
    for q, signals, r in zip(questions, input_signals, results):
        picott_nq = (
            (r.get("picott_json") or {}).get("normalized_question") or ""
        ).lower()
        assert any(s in picott_nq for s in signals), (
            f"Concurrent corruption: input={q!r} picott_nq={picott_nq!r} "
            f"signals={signals!r}"
        )
