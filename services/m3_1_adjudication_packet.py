# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M3.1/M3.2 adjudication-packet
# builder for the read-only admin adjudication/export surface. Superseded by the
# M4 forward path. NOT invoked in the clinician pipeline. Retained for the
# read-only admin surface + service-level unit tests. Do not extend.

"""Phase M3.1 — Clinical adjudication packet helper.

Single source of truth for the A-arm section of the human-readable
adjudication packet. By construction, the packet code path and the
admin GET path both flow through ``serialize_a_arm_from_session``
in ``services.m3_1_benchmark``, so the packet and the admin UI can
never drift apart.

This module exists so:
  * the packet-builder script under ``scripts/m3_1/`` (which writes
    the Markdown packet) can import a stable, in-repo function,
  * the unit test ``test_phaseM3_1_1_a_arm_field_mapping`` can
    assert byte-identical A-arm output across the two callers, and
  * any future packet/UI consumer is forced to go through the
    canonical serializer.

NEVER imports or re-implements field lookups for ``picott_payload``
or ``queries_by_family``. Those fields do not exist on the real
``search_sessions`` schema; reading them was the M3.1.1 bug.
"""
from __future__ import annotations

from typing import Any

from services.m3_1_benchmark import serialize_a_arm_from_session


def packet_a_arm_section(session_doc: dict | None) -> dict[str, Any]:
    """Return the A-arm projection used by the clinical adjudication
    packet for a given ``search_sessions`` document.

    This is a thin wrapper over the canonical serializer so the test
    ``test_M3_1_1_packet_and_admin_get_share_serializer`` can prove
    that both call paths produce byte-identical A-arm output.
    """
    return serialize_a_arm_from_session(session_doc)


__all__ = ["packet_a_arm_section"]
