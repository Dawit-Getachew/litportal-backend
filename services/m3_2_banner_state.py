# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M3.2 active-repair banner state.
# M3.2 is dormant + superseded by the M4 interpretation chip. NOT used in the
# forward clinician path (M3.2 never applies now). Retained for the read-only
# admin surface + service-level unit tests. Do not re-wire into the forward path.

"""Phase R1A — Part 2A — closed-set leak-scan state mapping.

DETERMINISTIC. NO LLM. NO PHI. NO MONGO READ at module load.

Replaces the ambiguous "0 redacted (leak scan failed)" admin banner
wording with a closed-set state machine. Four possible states:

  * ``leak_scan_passed`` — all rows in the scan passed; nothing was
    redacted; this is the safe state. Displayed as "All rows passed
    leak scan (0 redactions)".
  * ``leak_scan_failed_with_redactions`` — one or more rows had
    fields scrubbed pre-write; count is exposed to the operator.
    Displayed as "{N} row(s) had fields redacted by leak scan".
  * ``leak_scan_unavailable`` — scanner did not run (e.g., older
    pre-R1A rows that pre-date the scanner). Displayed as "Leak
    scan unavailable for {N} legacy row(s)".
  * ``row_level_redacted`` — per-row marker on the table; the
    sidecar row itself returns a ``{row_redacted: true,
    redaction_reason}`` shape and the table renders a "Row
    redacted by leak scan" pill instead of the full row.

The closed-set states are mutually exclusive at the BANNER level;
``row_level_redacted`` is a PER-ROW marker that can co-exist with
any banner state.
"""
from __future__ import annotations

from typing import Any

# Closed-set state keys.
LEAK_SCAN_BANNER_STATES: frozenset[str] = frozenset({
    "leak_scan_passed",
    "leak_scan_failed_with_redactions",
    "leak_scan_unavailable",
})

# Per-row marker key (separate from banner states).
LEAK_SCAN_ROW_MARKER: str = "row_level_redacted"


def classify_banner_state(
    *,
    total_rows: int,
    redacted_count: int,
    unscanned_count: int = 0,
) -> str:
    """Pure classifier. Returns one of the closed-set banner states.

    Precedence (deterministic):
      1. ``redacted_count > 0`` → ``leak_scan_failed_with_redactions``
      2. ``unscanned_count > 0`` → ``leak_scan_unavailable``
      3. otherwise → ``leak_scan_passed`` (default safe state)
    """
    if redacted_count and redacted_count > 0:
        return "leak_scan_failed_with_redactions"
    if unscanned_count and unscanned_count > 0:
        return "leak_scan_unavailable"
    return "leak_scan_passed"


def banner_copy_for_state(
    state: str,
    *,
    total_rows: int,
    redacted_count: int = 0,
    unscanned_count: int = 0,
) -> str:
    """Closed-set wording function. Pure, deterministic. Used by the
    admin UI and by the response shape in
    ``GET /api/admin/m3_2/adjudication-export`` so the API consumer
    doesn't have to interpret booleans.

    Returns a string that NEVER reads as failure when no failure
    occurred (the original "0 redacted (leak scan failed)" bug).
    """
    if state == "leak_scan_failed_with_redactions":
        return (
            f"{total_rows} row(s) returned — "
            f"{int(redacted_count)} redacted by leak scan"
        )
    if state == "leak_scan_unavailable":
        return (
            f"{total_rows} row(s) returned — "
            f"leak scan unavailable for {int(unscanned_count)} legacy row(s)"
        )
    # Default safe state.
    return f"{total_rows} row(s) returned — all passed leak scan"


def banner_envelope(
    *,
    total_rows: int,
    redacted_count: int,
    unscanned_count: int = 0,
) -> dict[str, Any]:
    """Closed-set envelope for the admin API response. Includes the
    state key, the human-readable copy, and the raw counts so the
    UI can render verbosely without re-classifying.
    """
    state = classify_banner_state(
        total_rows=total_rows,
        redacted_count=redacted_count,
        unscanned_count=unscanned_count,
    )
    return {
        "state": state,
        "copy": banner_copy_for_state(
            state,
            total_rows=total_rows,
            redacted_count=redacted_count,
            unscanned_count=unscanned_count,
        ),
        "total_rows": int(total_rows),
        "redacted_count": int(redacted_count or 0),
        "unscanned_count": int(unscanned_count or 0),
    }


__all__ = [
    "LEAK_SCAN_BANNER_STATES",
    "LEAK_SCAN_ROW_MARKER",
    "classify_banner_state",
    "banner_copy_for_state",
    "banner_envelope",
]
