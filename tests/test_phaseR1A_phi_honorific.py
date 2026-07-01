"""Phase R1A — Part 2B — patient-name-like PHI policy tests.

Validates:
  * Honorific WITHOUT period blocks (``Mr James``).
  * Honorific WITH period still blocks (``Mr. James``).
  * Citation context ALLOWS (``Smith et al.``).
  * Lowercase running text does NOT trip (``mr james``).
  * Author + journal context allows (citation pattern).
"""
from __future__ import annotations

import pytest

from services.phi_guard import phi_guard


# =====================================================================
# Honorific WITHOUT period — patient name pattern.
# =====================================================================
@pytest.mark.parametrize("text", [
    "Mr James presented to the ED with chest pain",
    "Mrs Smith reports nausea since yesterday",
    "Mr James Wilson with new-onset AF",
    "Dr Roberts is the consulting cardiologist",
])
def test_R1A_phi_name_without_period_blocks(text):
    result = phi_guard.screen(text)
    assert result.is_blocked, f"should block patient name: {text!r}"
    assert "name_title_detected" in result.reason_codes


# =====================================================================
# Honorific WITH period — still blocks.
# =====================================================================
@pytest.mark.parametrize("text", [
    "Mr. James presented to the ED",
    "Mrs. Smith with chest pain",
])
def test_R1A_phi_name_with_period_blocks(text):
    result = phi_guard.screen(text)
    assert result.is_blocked, f"should block: {text!r}"
    assert "name_title_detected" in result.reason_codes


# =====================================================================
# Citation contexts — allowed (literature reference, NOT PHI).
# =====================================================================
@pytest.mark.parametrize("text", [
    "Dr. Smith et al. JAMA 2024 reported",
    "Beuers et al. (2023) described the pathway",
    "Mr Roberts et al. 2022 found",
    "according to Smith JAMA 2024, the rate is",
])
def test_R1A_phi_name_in_citation_context_allows(text):
    result = phi_guard.screen(text)
    assert "name_title_detected" not in result.reason_codes, (
        f"citation context should not trigger name_title_detected: "
        f"{text!r} → reasons={result.reason_codes}"
    )


# =====================================================================
# Lowercase running text — does NOT trip.
# =====================================================================
@pytest.mark.parametrize("text", [
    "the patient is a mr who has chest pain",
    "discussion of mr findings on imaging",
])
def test_R1A_phi_name_lowercase_does_not_trip(text):
    result = phi_guard.screen(text)
    # Lowercase honorifics shouldn't trigger the name detector at all.
    assert "name_title_detected" not in result.reason_codes


# =====================================================================
# Clean clinical question — no false positives.
# =====================================================================
@pytest.mark.parametrize("text", [
    "What is the evidence for SGLT2 inhibitors in HFpEF?",
    "Mechanism of pruritus in cholestatic liver disease",
    "Risk of pneumothorax after central line placement",
])
def test_R1A_phi_clean_clinical_questions_pass(text):
    result = phi_guard.screen(text)
    assert not result.is_blocked, f"clean question should pass: {text!r} → {result.reason_codes}"


# =====================================================================
# Trailing comma form — Mr, James — still blocked.
# =====================================================================
def test_R1A_phi_trailing_comma_form_blocks():
    result = phi_guard.screen("Mr, James presented with chest pain")
    assert "name_title_detected" in result.reason_codes
