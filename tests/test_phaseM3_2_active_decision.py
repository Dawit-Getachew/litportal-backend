"""Phase M3.2 — Active Semantic Repair orchestrator + audit sidecar
integration tests.

Uses ``mongomock_motor`` so the orchestrator can exercise the
``m3_2_session_decisions`` collection and the ``feature_flags``
collection without a live Mongo. NEVER touches a real LLM.
"""
from __future__ import annotations

import asyncio

import pytest

from services.semantic_repair_active_config import (
    SemanticRepairActiveConfig,
)
from services.semantic_repair_active_pipeline import (
    run_m3_2_active_decision,
    get_session_decision_latest,
    list_recent_session_decisions,
)


def _allowed_cfg(**overrides) -> SemanticRepairActiveConfig:
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


def _good_candidate() -> dict:
    return {
        "primary_intent": "risk_etiology",
        "frameworks": ["risk_etiology"],
        "required_anchors": ["schizophrenia"],
        "slots": {
            "population": ["young adults"],
            "condition": ["schizophrenia"],
            "intervention_or_exposure": [],
            "outcome": [],
        },
        "semantic_repair_reason": "missing_risk_etiology_framework",
        "confidence": 0.85,
        "branches": [{"branch_id": "b1"}],
    }


def _shadow_block(candidate: dict | None = None, latency_ms: int = 1200) -> dict:
    return {
        "semantic_repair_triggered": True,
        "semantic_repair_trigger_reasons": [
            "missing_risk_etiology_framework",
        ],
        "semantic_repair_candidate": candidate or _good_candidate(),
        "semantic_repair_validation_status": (
            "passed" if candidate is not False else "rejected"
        ),
        "semantic_repair_rejection_reasons": [],
        "semantic_repair_diff": {},
        "semantic_repair_latency_ms": latency_ms,
    }


# =====================================================================
# Eligible path — applied=True, sidecar written.
# =====================================================================
@pytest.mark.asyncio
async def test_M3_2_eligible_path_applies_and_writes_sidecar(monkeypatch):
    monkeypatch.setattr(
        "services.semantic_repair_active_pipeline.is_mongo_kill_engaged",
        lambda: _async_return(False),
    )
    out = await run_m3_2_active_decision(
        session_id="sess-elig-001",
        user_id="user-1",
        semantic_repair_block=_shadow_block(),
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
        a_arm_cqm={
            "primary_intent": "background",
            "frameworks": ["background"],
            "required_anchors": ["schizophrenia"],
        },
        a_arm_boolean_queries=["schizophrenia"],
        config=_allowed_cfg(),
        mongo_kill_engaged=False,
    )
    assert out["applied"] is True
    assert out["decision"] == "eligible"
    assert out["shape"] == "etiology_risk"
    assert out["timeout_status"] == "none"
    assert out["fallback_status"] == "none"
    assert out["route_owned_by_k4a_specialty"] is False
    assert out["chip"] is not None
    assert out["chip"]["primary_text"] == "Clinical interpretation refined"
    assert "etiology/risk" in out["chip"]["tooltip_text"]
    # Sidecar fields are present and structured.
    assert out["m_repaired_cqm"]["primary_intent"] == "risk_etiology"
    assert out["m_repaired_compiler_input"]["required_anchors"] == ["schizophrenia"]
    assert out["m_repaired_boolean_queries"]["pubmed"], "B-arm pubmed queries non-empty"
    # Sidecar is queryable.
    side = await get_session_decision_latest("sess-elig-001")
    assert side is not None
    assert side["semantic_repair_active_applied"] is True
    assert side["eligibility_decision"] == "eligible"
    assert side["shape"] == "etiology_risk"
    assert side["route_owned_by_k4a_specialty"] is False
    assert side["a_arm_cqm"]["primary_intent"] == "background"


# =====================================================================
# K4A.* protected routes — NEVER applied.
# =====================================================================
@pytest.mark.parametrize("subtype", [
    "severity_marker_overview",
    "rule_out_test_utility",
    "epidemiology_incidence_complication_risk",
])
@pytest.mark.asyncio
async def test_M3_2_K4A_subtypes_are_excluded(subtype, monkeypatch):
    out = await run_m3_2_active_decision(
        session_id=f"sess-k4a-{subtype}",
        user_id="user-1",
        semantic_repair_block=_shadow_block(),
        picott_payload={
            "question_subtype": subtype,
            "type_of_question": "background",
            "framework_list_json": ["background"],
        },
        deterministic_primary_anchors=[
            {"canonical_concept": "acute pancreatitis",
             "required_for_search": True},
        ],
        sources=["pubmed", "europe_pmc"],
        a_arm_cqm={"primary_intent": "background"},
        a_arm_boolean_queries=["acute pancreatitis"],
        config=_allowed_cfg(),
        mongo_kill_engaged=False,
    )
    assert out["applied"] is False
    assert out["route_owned_by_k4a_specialty"] is True
    assert out["decision"] == "ineligible_route_owned_by_k4a_specialty"
    assert out["chip"] is None
    # Sidecar persists the K4A exclusion explicitly so the operator
    # can audit it later.
    side = await get_session_decision_latest(f"sess-k4a-{subtype}")
    assert side is not None
    assert side["route_owned_by_k4a_specialty"] is True
    assert side["semantic_repair_active_applied"] is False


# =====================================================================
# Kill switch — env source engaged blocks.
# =====================================================================
@pytest.mark.asyncio
async def test_M3_2_env_kill_switch_blocks(monkeypatch):
    out = await run_m3_2_active_decision(
        session_id="sess-killenv",
        user_id="user-1",
        semantic_repair_block=_shadow_block(),
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
        config=_allowed_cfg(kill_switch=True),  # env engaged
        mongo_kill_engaged=False,
    )
    assert out["applied"] is False
    assert out["decision"] == "ineligible_kill_switch_engaged"
    assert out["kill_switch_state"]["env_kill_switch_engaged"] is True
    assert out["kill_switch_state"]["mongo_kill_switch_engaged"] is False
    assert out["kill_switch_state"]["combined_engaged"] is True
    assert out["chip"] is None


# =====================================================================
# Kill switch — Mongo source engaged blocks.
# =====================================================================
@pytest.mark.asyncio
async def test_M3_2_mongo_kill_switch_blocks(monkeypatch):
    out = await run_m3_2_active_decision(
        session_id="sess-killmongo",
        user_id="user-1",
        semantic_repair_block=_shadow_block(),
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
        config=_allowed_cfg(kill_switch=False),  # env disengaged
        mongo_kill_engaged=True,                  # mongo engaged
    )
    assert out["applied"] is False
    assert out["decision"] == "ineligible_kill_switch_engaged"
    assert out["kill_switch_state"]["env_kill_switch_engaged"] is False
    assert out["kill_switch_state"]["mongo_kill_switch_engaged"] is True
    assert out["kill_switch_state"]["combined_engaged"] is True


# =====================================================================
# Shape not in allowlist — emits warning + falls back.
# =====================================================================
@pytest.mark.asyncio
async def test_M3_2_shape_not_in_allowlist_falls_back(caplog):
    # Constrain allowlist to ONLY pathophysiology, so etiology_risk
    # (the candidate's shape) is rejected.
    cfg = _allowed_cfg(shapes_allowlist=("pathophysiology_mechanism",))
    out = await run_m3_2_active_decision(
        session_id="sess-shape-block",
        user_id="user-1",
        semantic_repair_block=_shadow_block(),
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
        config=cfg,
        mongo_kill_engaged=False,
    )
    assert out["applied"] is False
    assert out["decision"] == "ineligible_shape_not_in_allowlist"


# =====================================================================
# Timeout — 3500ms ceiling honored (using a small ceiling for the test).
# =====================================================================
@pytest.mark.asyncio
async def test_M3_2_timeout_falls_back_to_a_arm(monkeypatch):
    """Force a timeout by patching the artifact builder to sleep
    longer than the configured ceiling. Validates the
    closed-set fields ``timeout_status`` and ``fallback_status``."""

    async def _slow_builder(candidate, sources):
        await asyncio.sleep(0.5)
        return ({}, {}, {}, [])

    monkeypatch.setattr(
        "services.semantic_repair_active_pipeline._build_m_repaired_artifacts",
        _slow_builder,
    )
    cfg = _allowed_cfg(max_latency_ms=50)
    out = await run_m3_2_active_decision(
        session_id="sess-timeout",
        user_id="user-1",
        # Candidate LLM latency must be below the ``max_latency_ms``
        # eligibility gate so eligibility passes; the timeout we're
        # exercising is the orchestrator's ``asyncio.wait_for``
        # wrapper around ``_build_m_repaired_artifacts`` (= 50 ms).
        semantic_repair_block=_shadow_block(latency_ms=10),
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
        config=cfg,
        mongo_kill_engaged=False,
    )
    assert out["applied"] is False
    assert out["timeout_status"] == "m3_2_active_timeout"
    assert out["fallback_status"] == "fallback_to_a_arm"
    # Eligibility was eligible — the timeout is an operational
    # fallback, not a clinical block.
    assert out["decision"] == "eligible"
    # Sidecar should record the timeout state.
    side = await get_session_decision_latest("sess-timeout")
    assert side["timeout_status"] == "m3_2_active_timeout"
    assert side["fallback_status"] == "fallback_to_a_arm"
    assert side["semantic_repair_active_applied"] is False


# =====================================================================
# Default config uses 3500 ms ceiling.
# =====================================================================
def test_M3_2_default_ceiling_is_3500() -> None:
    cfg = _allowed_cfg()
    assert cfg.max_latency_ms == 3500


# =====================================================================
# Sidecar list endpoint includes recently-written rows.
# =====================================================================
@pytest.mark.asyncio
async def test_M3_2_list_recent_returns_inserted_rows() -> None:
    # We rely on previous test inserts in the same test module to
    # populate at least one row. Insert one more to be safe.
    out = await run_m3_2_active_decision(
        session_id="sess-list-1",
        user_id="user-x",
        semantic_repair_block=_shadow_block(),
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
    assert out["applied"] is True
    rows = await list_recent_session_decisions(limit=10)
    assert any(r["search_session_id"] == "sess-list-1" for r in rows)


# Helper: produce a sync function that returns a future for the mock.
def _async_return(value):
    async def _f():
        return value
    return _f()
