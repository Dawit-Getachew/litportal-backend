"""Phase R1A — Part 2A — leak-scan banner state contract.

Validates the closed-set 4-state banner classifier that replaces
the ambiguous "0 redacted (leak scan failed)" UI wording.

Pure unit tests against ``services/m3_2_banner_state.py``. No I/O.
"""
from __future__ import annotations

import pytest

from services.m3_2_banner_state import (
    LEAK_SCAN_BANNER_STATES,
    LEAK_SCAN_ROW_MARKER,
    banner_copy_for_state,
    banner_envelope,
    classify_banner_state,
)


# =====================================================================
# Closed-set state machine — exhaustive.
# =====================================================================
def test_R1A_banner_states_are_closed_set():
    assert LEAK_SCAN_BANNER_STATES == frozenset({
        "leak_scan_passed",
        "leak_scan_failed_with_redactions",
        "leak_scan_unavailable",
    })


def test_R1A_banner_row_marker_is_distinct_from_states():
    # ``row_level_redacted`` is a PER-ROW marker, NOT a banner state.
    assert LEAK_SCAN_ROW_MARKER == "row_level_redacted"
    assert LEAK_SCAN_ROW_MARKER not in LEAK_SCAN_BANNER_STATES


# =====================================================================
# classify_banner_state — closed-set precedence.
# =====================================================================
@pytest.mark.parametrize("total,redacted,unscanned,expected", [
    # All passed — default safe state.
    (10, 0, 0, "leak_scan_passed"),
    (0, 0, 0, "leak_scan_passed"),
    # Redactions present — failure-with-redactions wins.
    (10, 1, 0, "leak_scan_failed_with_redactions"),
    (10, 5, 3, "leak_scan_failed_with_redactions"),
    # Unscanned only — unavailable.
    (10, 0, 2, "leak_scan_unavailable"),
    # Redactions take precedence over unscanned.
    (10, 2, 2, "leak_scan_failed_with_redactions"),
])
def test_R1A_classify_banner_state(total, redacted, unscanned, expected):
    out = classify_banner_state(
        total_rows=total,
        redacted_count=redacted,
        unscanned_count=unscanned,
    )
    assert out == expected


# =====================================================================
# banner_copy_for_state — must NEVER read as failure when no failure.
# =====================================================================
def test_R1A_banner_copy_passed_state_does_not_say_failed():
    copy = banner_copy_for_state(
        "leak_scan_passed",
        total_rows=10, redacted_count=0,
    )
    # The original bug was wording like "0 redacted (leak scan failed)".
    # The new wording must NEVER say "failed" when nothing failed.
    assert "failed" not in copy.lower()
    assert "passed" in copy.lower()


def test_R1A_banner_copy_failed_state_includes_redaction_count():
    copy = banner_copy_for_state(
        "leak_scan_failed_with_redactions",
        total_rows=20, redacted_count=3,
    )
    assert "3" in copy
    assert "redact" in copy.lower()


def test_R1A_banner_copy_unavailable_state_explains_legacy():
    copy = banner_copy_for_state(
        "leak_scan_unavailable",
        total_rows=15, redacted_count=0, unscanned_count=5,
    )
    assert "unavailable" in copy.lower()
    assert "5" in copy


# =====================================================================
# banner_envelope — closed-set envelope contract.
# =====================================================================
def test_R1A_banner_envelope_passed_state_contract():
    env = banner_envelope(
        total_rows=10, redacted_count=0, unscanned_count=0,
    )
    assert env["state"] == "leak_scan_passed"
    assert env["total_rows"] == 10
    assert env["redacted_count"] == 0
    assert env["unscanned_count"] == 0
    assert "failed" not in env["copy"].lower()


def test_R1A_banner_envelope_failure_state_contract():
    env = banner_envelope(
        total_rows=10, redacted_count=2, unscanned_count=0,
    )
    assert env["state"] == "leak_scan_failed_with_redactions"
    assert env["redacted_count"] == 2
    assert "2" in env["copy"]


def test_R1A_banner_envelope_keys_are_closed_set():
    env = banner_envelope(
        total_rows=10, redacted_count=0, unscanned_count=0,
    )
    assert set(env.keys()) == {
        "state", "copy", "total_rows", "redacted_count", "unscanned_count",
    }
