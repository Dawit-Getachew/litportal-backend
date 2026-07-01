"""Phase M4.1 — Explicit K4A Protected-Route Exclusion Gate for M4 active
query rewriting.

Central acceptance test: a K4A-owned route receives an OTHERWISE VALID,
ACCEPTED, ELIGIBLE M4 candidate and M4 STILL blocks with the closed-set
decision ``route_owned_by_k4a_specialty`` (applied=False, no rewrite).
Naturally-rejected candidates are NOT sufficient — these tests force a
valid candidate to prove the gate is explicit and dominant.

DETERMINISTIC. NO real LLM. Uses the conftest mongomock DB for the
sidecar + feature-flag collections.
"""
from __future__ import annotations

import pytest

from services.m4_active_config import M4ActiveConfig
from services.m4_active_pipeline import (
    ACTIVE_ROUTE_IDS,
    K4A_OWNED_SUBTYPES,
    M4_ACTIVE_DECISIONS,
    _decide,
    resolve_active_route,
    run_m4_active_decision,
)


# ---------------------------------------------------------------------
# Fixtures — a VALID, accepted, projection-passed M4 candidate.
# ---------------------------------------------------------------------

def _valid_candidate() -> dict:
    return {
        "primary_intent": "background_overview",
        "frameworks": ["background"],
        "clinical_problem": ["acute pancreatitis"],
        "required_anchors": ["acute pancreatitis"],
        "outcome": ["severity"],
        "optional_or_ranking_only_terms": ["markers of severity"],
        "clarification_required": False,
        "confidence": 0.9,
        "slots": {"condition": ["acute pancreatitis"], "outcome": ["severity"]},
    }


def _m4_block(*, accepted: bool = True, projection_status: str = "passed") -> dict:
    cand = _valid_candidate()
    proj = {
        "projected_compiler_input": {"required_anchors": ["acute pancreatitis"]},
        "projection_validation_status": projection_status,
        "projection_validation_flags": [],
    }
    return {
        "mode": "shadow",
        "triggered": True,
        "trigger_reasons": ["full_literal_phrase"],
        "validation_status": "passed" if accepted else "rejected",
        "accepted": accepted,
        "candidate": cand if accepted else None,
        "m4_0_1_projection": proj if accepted else {},
        "timed_out": False,
        "upstream_latency_ms": 120,
    }


def _cfg(**overrides) -> M4ActiveConfig:
    fields = dict(
        enabled=True,
        env_allowlist=("dev", "preview", "development"),
        max_latency_ms=4000,
        kill_switch=False,
        prod_allow_active=False,
        current_env="preview",
    )
    fields.update(overrides)
    return M4ActiveConfig(**fields)


def _picott(subtype: str) -> dict:
    return {
        "type_of_question": "background",
        "question_subtype": subtype,
        "framework_list_json": ["background"],
        "specialty": "general",
    }


async def _run(subtype, **overrides):
    base = dict(
        session_id=f"m4-k4a-{subtype}",
        user_id="u1",
        m4_block=_m4_block(),
        picott_payload=_picott(subtype),
        sources=["pubmed", "europe_pmc"],
        a_arm_cqm={"primary_intent": "background", "required_anchors": ["x"]},
        m3_2_applied=False,
        m3_2_kill_combined_engaged=True,   # M3.2 dormant
        baseline_degraded=True,            # a real defect (would normally promote)
        baseline_degradation_reasons=["full_literal_phrase_required_anchor"],
        baseline_query_family_count=1,
        a_arm_boolean_queries=["baseline q1"],
        config=_cfg(),
        mongo_kill_engaged=False,          # kill DISENGAGED → otherwise eligible
    )
    base.update(overrides)
    return await run_m4_active_decision(**base)


# ---------------------------------------------------------------------
# resolve_active_route — deterministic closed-set mapping.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("subtype,route_id", [
    ("severity_marker_overview", "k4a_2_severity_marker_overview"),
    ("rule_out_test_utility", "k4a_3_rule_out_diagnostic_accuracy"),
    ("epidemiology_incidence_complication_risk", "k4a_4_incidence_risk_epidemiology"),
])
def test_resolve_route_k4a_owned(subtype, route_id):
    out = resolve_active_route(question_subtype=subtype)
    assert out["route_owned_by_k4a_specialty"] is True
    assert out["active_route_id"] == route_id
    assert out["protected_route_family"] == "k4a"
    assert out["active_route_id"] in ACTIVE_ROUTE_IDS


def test_resolve_route_non_k4a_default_and_unknown():
    assert resolve_active_route(question_subtype="mechanism")["active_route_id"] == "default"
    assert resolve_active_route(question_subtype="mechanism")["route_owned_by_k4a_specialty"] is False
    assert resolve_active_route(question_subtype=None)["active_route_id"] == "unknown"
    assert resolve_active_route(question_subtype="")["active_route_id"] == "unknown"


def test_resolve_route_hint_defense_in_depth():
    out = resolve_active_route(question_subtype="mechanism", route_owned_hint=True)
    assert out["route_owned_by_k4a_specialty"] is True
    assert out["active_route_id"] == "unknown"
    assert out["protected_route_family"] == "k4a"


def test_k4a_owned_subtypes_match_canonical_set():
    assert K4A_OWNED_SUBTYPES == frozenset({
        "severity_marker_overview",
        "rule_out_test_utility",
        "epidemiology_incidence_complication_risk",
    })


# ---------------------------------------------------------------------
# _decide — route ownership DOMINATES every other verdict.
# ---------------------------------------------------------------------

def test_decide_route_owned_dominates_eligible():
    d = _decide(
        candidate={"required_anchors": ["x"]}, accepted=True,
        projection_status="passed", required_anchors=["x"], cfg=_cfg(),
        combined_kill=False, m3_2_applied=False, m3_2_dormant=True,
        route_owned_by_k4a_specialty=True,
    )
    assert d == "route_owned_by_k4a_specialty"


def test_decide_route_owned_dominates_even_with_kill_and_prod():
    # Even with kill engaged + production hard-off, route ownership wins.
    d = _decide(
        candidate={"required_anchors": ["x"]}, accepted=True,
        projection_status="passed", required_anchors=["x"],
        cfg=_cfg(current_env="production", prod_allow_active=False),
        combined_kill=True, m3_2_applied=False, m3_2_dormant=True,
        route_owned_by_k4a_specialty=True,
    )
    assert d == "route_owned_by_k4a_specialty"


def test_decide_default_false_unchanged():
    # No route ownership → normal eligible path preserved (BC).
    d = _decide(
        candidate={"required_anchors": ["x"]}, accepted=True,
        projection_status="passed", required_anchors=["x"], cfg=_cfg(),
        combined_kill=False, m3_2_applied=False, m3_2_dormant=True,
    )
    assert d == "eligible"


# ---------------------------------------------------------------------
# CENTRAL ACCEPTANCE — forced-valid candidate on K4A routes still blocks.
# ---------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("subtype,route_id", [
    ("severity_marker_overview", "k4a_2_severity_marker_overview"),       # K4A.2
    ("rule_out_test_utility", "k4a_3_rule_out_diagnostic_accuracy"),      # K4A.3
    ("epidemiology_incidence_complication_risk", "k4a_4_incidence_risk_epidemiology"),  # K4A.4
])
async def test_forced_valid_candidate_blocked_on_k4a_route(subtype, route_id):
    out = await _run(subtype)
    assert out["applied"] is False
    assert out["active_query_rewritten"] is False
    assert out["decision"] == "route_owned_by_k4a_specialty"
    assert "route_owned_by_k4a_specialty" in out["m4_block_reasons"]
    assert out["active_route_id"] == route_id
    assert out["protected_route_family"] == "k4a"
    assert out["route_owned_by_k4a_specialty"] is True
    assert out["chip"] is None
    assert not out["cb_records"]
    assert out["decision"] in M4_ACTIVE_DECISIONS


@pytest.mark.asyncio
async def test_non_k4a_eligible_candidate_still_applies():
    # Control: identical VALID candidate on a NON-K4A route must apply
    # (no regression to the M4-Prod-1 dress-rehearsal behavior).
    out = await _run("etiology_risk")
    assert out["applied"] is True
    assert out["active_query_rewritten"] is True
    assert out["decision"] == "eligible"
    assert out["route_owned_by_k4a_specialty"] is False
    assert out["active_route_id"] == "default"
    assert out["chip"]["primary_text"] == "Clinical question interpretation refined"
    assert out["m4_block_reasons"] == []


@pytest.mark.asyncio
async def test_route_hint_blocks_even_non_k4a_subtype():
    out = await _run("etiology_risk", route_owned_hint=True)
    assert out["applied"] is False
    assert out["decision"] == "route_owned_by_k4a_specialty"
    assert out["protected_route_family"] == "k4a"


@pytest.mark.asyncio
async def test_diagnostics_fields_present_and_closed_set():
    out = await _run("severity_marker_overview")
    for k in (
        "active_route_id", "protected_route_family",
        "route_owned_by_k4a_specialty", "m4_block_reasons",
        "active_query_rewritten",
    ):
        assert k in out, f"missing diagnostic field {k}"
    assert out["active_route_id"] in ACTIVE_ROUTE_IDS
    assert out["decision"] in M4_ACTIVE_DECISIONS
