"""PHI guard unit tests — pure, no DB."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.phi_guard import PHIGuardService, REASON_CODES, USER_PHI_MESSAGE  # noqa: E402


def test_clean_question_passes():
    res = PHIGuardService().screen("What is the evidence for SGLT2 inhibitors in HFpEF?")
    assert res.is_blocked is False
    assert res.reason_codes == []
    assert "[REDACTED]" not in res.sanitized_question


def test_email_is_blocked():
    res = PHIGuardService().screen("Contact me at john.doe@example.com please")
    assert res.is_blocked is True
    assert "email_detected" in res.reason_codes


def test_phone_is_blocked():
    res = PHIGuardService().screen("Call (415) 555-1212 about the case")
    assert res.is_blocked is True
    assert "phone_detected" in res.reason_codes


def test_url_is_blocked():
    res = PHIGuardService().screen("See http://patient-records.example.com/x for context")
    assert res.is_blocked is True
    assert "url_detected" in res.reason_codes


def test_ssn_is_blocked():
    res = PHIGuardService().screen("Patient SSN 123-45-6789 has HTN")
    assert res.is_blocked is True
    assert "ssn_detected" in res.reason_codes


def test_mrn_prefix_is_blocked():
    res = PHIGuardService().screen("Pt MRN: 12345 has new diagnosis of HFpEF")
    assert res.is_blocked is True
    assert "mrn_pattern_detected" in res.reason_codes


def test_long_digit_run_is_blocked():
    res = PHIGuardService().screen("ID 1234567890 — best tx?")
    assert res.is_blocked is True
    assert "long_digit_sequence_detected" in res.reason_codes


def test_standalone_year_is_not_blocked():
    res = PHIGuardService().screen("Per the 2024 update of the ACC/AHA guideline")
    assert res.is_blocked is False, res.reason_codes


def test_numeric_dob_is_blocked():
    res = PHIGuardService().screen("DOB 03/14/1979 — best tx for HTN?")
    assert res.is_blocked is True
    assert "exact_date_detected" in res.reason_codes


def test_iso_date_is_blocked():
    res = PHIGuardService().screen("Encounter 2024-01-05 outcome?")
    assert res.is_blocked is True
    assert "exact_date_detected" in res.reason_codes


def test_named_date_is_blocked():
    res = PHIGuardService().screen("Admitted Jan 5, 2024 with new SOB")
    assert res.is_blocked is True
    assert "exact_date_detected" in res.reason_codes


def test_personal_name_is_blocked():
    res = PHIGuardService().screen("Dr. Smith recommended starting metformin")
    assert res.is_blocked is True
    assert "name_title_detected" in res.reason_codes


def test_street_address_is_blocked():
    res = PHIGuardService().screen("Lives at 1234 Elm Street and reports chest pain")
    assert res.is_blocked is True
    assert "street_address_detected" in res.reason_codes


def test_combined_phi_acceptance_string_is_blocked():
    q = "45M with MRN 12345678, DOB 03/14/1979, email john@x.com — best tx for HTN?"
    res = PHIGuardService().screen(q)
    assert res.is_blocked is True
    assert "mrn_pattern_detected" in res.reason_codes or "long_digit_sequence_detected" in res.reason_codes
    assert "exact_date_detected" in res.reason_codes
    assert "email_detected" in res.reason_codes
    assert "[REDACTED]" in res.sanitized_question


def test_reason_codes_are_from_fixed_enum():
    res = PHIGuardService().screen("Mrs. Jane Doe at 999 Main St, DOB 1/2/1980, MRN 9876543")
    assert res.is_blocked is True
    for code in res.reason_codes:
        assert code in REASON_CODES, f"unexpected reason code: {code}"


def test_user_phi_message_is_fixed_string():
    assert USER_PHI_MESSAGE.startswith("Your question appears to include patient identifiers")
