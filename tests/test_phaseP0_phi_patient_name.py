"""Phase P0 — PHI guard honorific / patient-name-like pattern fix.

Closes the gap where a named patient vignette WITHOUT an honorific
(``Patient John Doe has ...``) was not blocked. The honorific forms
(``Mr James``, ``Mrs Smith``, ``Ms Johnson``) were already covered by
the R1A name detector; this phase adds the label form and an explicit
``patient named X`` marker form, while preserving the closed-set reason
code (``name_title_detected``) and avoiding over-blocking common
clinical phrases (``Patient Reported Outcomes`` etc.).

Safety contract validated here:
  * Patient-name-like vignettes block (``is_blocked=True``).
  * Closed-set reason code only (``name_title_detected``).
  * Raw name text is redacted out of ``sanitized_question``.
  * Citation/literature context still does NOT false-positive.
  * Common ``Patient <Phrase>`` clinical terms do NOT false-positive.
"""
from __future__ import annotations

import pytest

from services.phi_guard import REASON_CODES, phi_guard


# =====================================================================
# Acceptance — patient-name-like vignettes MUST block.
# =====================================================================
@pytest.mark.parametrize("text", [
    "Mr James presented with new onset headache. What is the differential diagnosis?",
    "Mrs Smith is a 70-year-old with chest pain. What is the differential?",
    "Ms Johnson has fever and cough. What is the workup?",
    "Patient John Doe has abdominal pain. What is the differential?",
])
def test_P0_patient_name_vignettes_block(text):
    result = phi_guard.screen(text)
    assert result.is_blocked, f"should block patient vignette: {text!r}"
    assert "name_title_detected" in result.reason_codes


# =====================================================================
# Acceptance — citation / literature context MUST NOT block.
# =====================================================================
@pytest.mark.parametrize("text", [
    "Smith et al. JAMA 2024 sepsis trial",
    "What does the Smith 2024 trial say about sepsis?",
    "How is Smith-Lemli-Opitz syndrome diagnosed?",
    "What is James Parkinson's disease history?",
])
def test_P0_citation_and_eponym_context_pass(text):
    result = phi_guard.screen(text)
    assert not result.is_blocked, (
        f"literature/eponym context should pass: {text!r} → {result.reason_codes}"
    )


# =====================================================================
# False-positive guard — common ``Patient <Phrase>`` clinical terms
# MUST NOT block (no honorific, but a real clinical concept).
# =====================================================================
@pytest.mark.parametrize("text", [
    "What are Patient Reported Outcomes in oncology trials?",
    "patient reported outcomes in oncology trials",
    "Patient Health Questionnaire (PHQ-9) for depression screening",
    "Patient Global Assessment in rheumatoid arthritis",
    "Patient Safety Indicators in hospitalized adults",
    "Patient Centered Medical Home outcomes",
    "Patient Derived Xenograft models in cancer",
    "Patient Selection criteria for TAVR",
    "How does patient adherence affect statin outcomes?",
])
def test_P0_clinical_patient_phrases_do_not_false_positive(text):
    result = phi_guard.screen(text)
    assert "name_title_detected" not in result.reason_codes, (
        f"clinical phrase should not trip name detector: {text!r} → {result.reason_codes}"
    )


# =====================================================================
# Explicit naming markers MUST block.
# =====================================================================
@pytest.mark.parametrize("text", [
    "The patient named John Doe presented with chest pain",
    "Patient's name is Jane Smith, she has a fever",
    "patient called Robert Brown has abdominal pain",
])
def test_P0_explicit_named_marker_blocks(text):
    result = phi_guard.screen(text)
    assert result.is_blocked, f"explicit named marker should block: {text!r}"
    assert "name_title_detected" in result.reason_codes


# =====================================================================
# No-leak — the detected name is redacted out of the sanitized text,
# and only closed-set reason codes are emitted.
# =====================================================================
def test_P0_patient_name_redacted_and_codes_closed_set():
    result = phi_guard.screen("Patient John Doe has abdominal pain.")
    assert "[REDACTED]" in result.sanitized_question
    assert "John" not in result.sanitized_question
    assert "Doe" not in result.sanitized_question
    for code in result.reason_codes:
        assert code in REASON_CODES, f"unexpected reason code: {code}"
