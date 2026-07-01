"""Phase M3.2 Code-Review patch tests — leak-scan hard gate +
clinician GET projection restriction.

These tests cover the two NON-BLOCKING findings raised during the
M3.2 code review:

  1. The leak scanner over the candidate-derived projections is now
     a HARD gate. If a leak signal fires, ``applied`` flips back to
     False, the chip is removed, and fallback_status is set to
     ``fallback_to_a_arm``.
  2. The clinician-facing ``GET /api/search-sessions/{id}``
     ``m3_2_active_repair`` block exposes ONLY ``{applied, shape,
     chip}``. Operator internals (``decision``,
     ``kill_switch_state``, ``timeout_status``, ``fallback_status``,
     ``route_owned_by_k4a_specialty``, ``sidecar_doc_id``) MUST NOT
     appear in the clinician response.
"""
from __future__ import annotations

import pytest

from routers.search_sessions import _clinician_m3_2_block
from services.semantic_repair_active_config import (
    SemanticRepairActiveConfig,
)
from services.semantic_repair_active_pipeline import (
    get_session_decision_latest,
    run_m3_2_active_decision,
)


def _allowed_cfg(**overrides):
    fields = dict(
        enabled=True,
        env_allowlist=("dev", "preview"),
        shapes_allowlist=(
            "etiology_risk",
            "pathophysiology_mechanism",
            "procedure_complication",
        ),
        max_latency_ms=3500,
        kill_switch=False,
        prod_allow_active=False,
        current_env="preview",
    )
    fields.update(overrides)
    return SemanticRepairActiveConfig(**fields)


def _candidate_with_leak_in_slot():
    # Inject a credential-shaped leak signal into the slots payload
    # (defense-in-depth — the M2.1 validator wouldn't let this
    # through in real traffic, but we exercise the gate explicitly).
    return {
        "primary_intent": "risk_etiology",
        "frameworks": ["risk_etiology"],
        "required_anchors": ["schizophrenia"],
        "slots": {
            "population": ["young adults"],
            # Bearer-prefixed leak signal — `_LEAK_SIGNALS` covers this.
            "condition": ["Bearer eyJhbGciOiJIUzI1NiJ9.fake"],
        },
        "semantic_repair_reason": "missing_risk_etiology_framework",
        "confidence": 0.85,
        "branches": [{"branch_id": "b1"}],
    }


def _shadow_block(candidate):
    return {
        "semantic_repair_triggered": True,
        "semantic_repair_trigger_reasons": [
            "missing_risk_etiology_framework",
        ],
        "semantic_repair_candidate": candidate,
        "semantic_repair_validation_status": "passed",
        "semantic_repair_rejection_reasons": [],
        "semantic_repair_diff": {},
        "semantic_repair_latency_ms": 50,
    }


# =====================================================================
# Leak-scan hard gate — applied=False on leak detection.
# =====================================================================
@pytest.mark.asyncio
async def test_M3_2_leak_signal_in_candidate_forces_a_arm_fallback() -> None:
    out = await run_m3_2_active_decision(
        session_id="sess-leak-001",
        user_id="user-x",
        semantic_repair_block=_shadow_block(_candidate_with_leak_in_slot()),
        picott_payload={
            "question_subtype": "etiology_risk_overview",
            "type_of_question": "background",
            "framework_list_json": ["background"],
        },
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia",
             "required_for_search": True},
        ],
        sources=["pubmed", "europe_pmc"],
        a_arm_cqm={"primary_intent": "background"},
        a_arm_boolean_queries=["schizophrenia"],
        config=_allowed_cfg(),
        mongo_kill_engaged=False,
    )
    assert out["applied"] is False, (
        "Leak signal in candidate slots must force fallback "
        "(applied=False) even after eligibility passed."
    )
    assert out["fallback_status"] == "fallback_to_a_arm"
    assert out["chip"] is None, "Chip must NOT render on leak fallback"
    assert out["m_repaired_cqm"] is None
    assert out["m_repaired_compiler_input"] is None
    assert out["m_repaired_boolean_queries"] is None
    assert out["m_repaired_cb_records"] is None
    # The sidecar safety_flags.leak_scan_passed should be False.
    assert out["safety_flags"]["leak_scan_passed"] is False
    # Sidecar persisted with the scrubbed (None) m-artifacts.
    side = await get_session_decision_latest("sess-leak-001")
    assert side is not None
    assert side["semantic_repair_active_applied"] is False
    assert side["m_repaired_cqm"] is None
    assert side["m_repaired_boolean_queries"] is None
    assert side["fallback_status"] == "fallback_to_a_arm"
    assert side["safety_flags"]["leak_scan_passed"] is False


# =====================================================================
# Clinician projection — only {applied, shape, chip}.
# =====================================================================
def test_M3_2_clinician_block_strips_operator_internals() -> None:
    persisted = {
        "applied": True,
        "decision": "eligible",
        "shape": "etiology_risk",
        "timeout_status": "none",
        "fallback_status": "none",
        "latency_ms": 12,
        "kill_switch_state": {
            "env_kill_switch_engaged": False,
            "mongo_kill_switch_engaged": False,
            "combined_engaged": False,
            "fail_closed_default": True,
        },
        "chip": {
            "primary_text": "Clinical interpretation refined",
            "shape_machine": "etiology_risk",
            "shape_human": "etiology/risk",
            "tooltip_text": "...etiology/risk...",
        },
        "route_owned_by_k4a_specialty": False,
        "sidecar_doc_id": "deadbeef" * 4,
        "phase": "M3.2",
    }
    block = _clinician_m3_2_block(persisted)
    assert block is not None
    assert set(block.keys()) == {"applied", "shape", "chip"}
    # Critical: operator internals MUST be absent.
    for forbidden in (
        "decision",
        "timeout_status",
        "fallback_status",
        "latency_ms",
        "kill_switch_state",
        "route_owned_by_k4a_specialty",
        "sidecar_doc_id",
        "phase",
    ):
        assert forbidden not in block, (
            f"Clinician block MUST NOT expose operator internal "
            f"`{forbidden}` — leaks operator state to clinicians."
        )
    assert block["applied"] is True
    assert block["shape"] == "etiology_risk"
    assert block["chip"]["primary_text"] == "Clinical interpretation refined"


def test_M3_2_clinician_block_returns_none_on_applied_false() -> None:
    persisted = {
        "applied": False,
        "decision": "ineligible_kill_switch_engaged",
        "shape": "etiology_risk",
        "kill_switch_state": {
            "env_kill_switch_engaged": True,
            "mongo_kill_switch_engaged": True,
            "combined_engaged": True,
            "fail_closed_default": True,
        },
        "chip": None,
        "route_owned_by_k4a_specialty": False,
    }
    assert _clinician_m3_2_block(persisted) is None


def test_M3_2_clinician_block_returns_none_on_k4a_excluded() -> None:
    persisted = {
        "applied": False,
        "decision": "ineligible_route_owned_by_k4a_specialty",
        "shape": "etiology_risk",
        "route_owned_by_k4a_specialty": True,
        "chip": None,
    }
    assert _clinician_m3_2_block(persisted) is None


def test_M3_2_clinician_block_handles_none_and_malformed() -> None:
    assert _clinician_m3_2_block(None) is None
    assert _clinician_m3_2_block({}) is None
    assert _clinician_m3_2_block("not-a-dict") is None
    assert _clinician_m3_2_block([1, 2, 3]) is None
    # Missing 'applied' → treated as not applied.
    assert _clinician_m3_2_block({"shape": "etiology_risk"}) is None
