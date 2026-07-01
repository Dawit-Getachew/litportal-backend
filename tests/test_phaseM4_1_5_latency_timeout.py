"""Phase M4.1.5 — end-to-end M4 latency instrumentation + hard timeout.

Covers:
  * closed-set decision ``m4_latency_timeout`` membership,
  * end-to-end latency total = upstream + promotion-decision,
  * closed-set NUMERIC latency breakdown (no PHI/text keys),
  * a hard-timeout upstream signal NEVER promotes M4 (no partial
    retrieval), even on an otherwise-eligible degraded baseline,
  * applied eligible cases record a NONZERO total latency when the
    upstream bounded-LLM time is supplied.

NEVER touches a real LLM. Uses the conftest mongomock DB.
"""
from __future__ import annotations

import pytest

from services.m4_active_config import M4ActiveConfig
from services.m4_active_pipeline import (
    M4_ACTIVE_DECISIONS,
    run_m4_active_decision,
)


# ---- fixtures (mirror test_phaseM4_1_active.py) ----

def _accepted_candidate() -> dict:
    """Phase M4.4-clean etiology rescue: preserves every central concept
    so the do-no-harm gate ALLOWS promotion (the prior colon-cancer
    stage-prognosis fixture was an over-broadening now correctly
    blocked)."""
    return {
        "primary_intent": "etiology",
        "frameworks": ["etiology"],
        "clinical_problem": ["pulmonary embolism"],
        "stage": [],
        "biomarker": [],
        "population": ["postoperative patients"],
        "exposure": ["recent surgery"],
        "intervention": [],
        "procedure": [],
        "medication": [],
        "index_test": [],
        "outcome": [],
        "mechanism": [],
        "comparator": [],
        "required_anchors": ["pulmonary embolism"],
        "optional_or_ranking_only_terms": [],
        "candidate_synonyms_unvalidated": [],
        "candidate_mesh_terms_unvalidated": [],
        "clarification_required": False,
        "confidence": 0.9,
        "repair_reason": "long_literal_phrase_anchor",
        "provenance": "m4_full_question_interpreter_shadow",
    }


def _m4_block(accepted: bool = True) -> dict:
    cand = _accepted_candidate()
    proj = {
        "projected_compiler_input": {"required_anchors": ["pulmonary embolism"]},
        "projection_validation_status": "passed",
        "projection_validation_flags": [],
    }
    return {
        "mode": "shadow",
        "triggered": True,
        "trigger_reasons": ["long_literal_phrase_anchor"],
        "validation_status": "passed" if accepted else "rejected",
        "accepted": accepted,
        "candidate": cand if accepted else None,
        "m4_0_1_projection": proj if accepted else {},
    }


def _cfg(**ov) -> M4ActiveConfig:
    f = dict(enabled=True, env_allowlist=("dev", "preview", "development"),
             max_latency_ms=4000, kill_switch=False, prod_allow_active=False,
             current_env="preview")
    f.update(ov)
    return M4ActiveConfig(**f)


def _picott() -> dict:
    return {"type_of_question": "background", "question_subtype": "disease_overview",
            "framework_list_json": ["background"], "specialty": "oncology"}


async def _run(**ov):
    base = dict(
        session_id="m4-1-5-sess", user_id="user-1", m4_block=_m4_block(),
        picott_payload=_picott(), sources=["pubmed", "europe_pmc"],
        a_arm_cqm={"primary_intent": "background", "required_anchors": ["x"]},
        m3_2_applied=False, m3_2_kill_combined_engaged=True,
        baseline_degraded=True,
        baseline_degradation_reasons=["full_literal_phrase_required_anchor"],
        baseline_query_family_count=3, a_arm_boolean_queries=["q1"],
        config=_cfg(), mongo_kill_engaged=False,
    )
    base.update(ov)
    return await run_m4_active_decision(**base)


# ---- tests ----

def test_timeout_decision_is_closed_set():
    assert "m4_latency_timeout" in M4_ACTIVE_DECISIONS


@pytest.mark.asyncio
async def test_eligible_records_nonzero_total_latency():
    out = await _run(
        session_id="m4-1-5-elig",
        upstream_latency_ms=850,
        upstream_latency_breakdown={
            "trigger_detect_ms": 1, "llm_candidate_ms": 800,
            "validation_ms": 40, "would_be_projection_ms": 5,
            "compiler_projection_ms": 4,
        },
    )
    assert out["applied"] is True
    assert out["decision"] == "eligible"
    # total = upstream + promotion-decision >= upstream (nonzero).
    assert out["m4_total_latency_ms"] >= 850
    assert out["m4_total_latency_ms"] > 0
    br = out["latency_breakdown"]
    # closed-set NUMERIC keys only.
    assert set(br.keys()) == {
        "trigger_detect_ms", "llm_candidate_ms", "validation_ms",
        "would_be_projection_ms", "compiler_projection_ms",
        "promotion_decision_ms", "total_ms",
    }
    assert all(isinstance(v, int) for v in br.values())
    assert br["total_ms"] == out["m4_total_latency_ms"]
    assert br["llm_candidate_ms"] == 800
    # total == upstream + promotion-decision.
    assert br["total_ms"] == 850 + br["promotion_decision_ms"]


@pytest.mark.asyncio
async def test_timeout_never_promotes_even_when_eligible():
    out = await _run(
        session_id="m4-1-5-timeout",
        upstream_timed_out=True,
        upstream_latency_ms=4000,
        upstream_latency_breakdown={"llm_candidate_ms": 4000},
    )
    assert out["applied"] is False
    assert out["decision"] == "m4_latency_timeout"
    assert out["timed_out"] is True
    assert out["chip"] is None
    assert out["cb_records"] == []
    # latency still recorded (numeric).
    assert out["m4_total_latency_ms"] >= 4000


@pytest.mark.asyncio
async def test_no_upstream_latency_defaults_to_zero_breakdown():
    out = await _run(session_id="m4-1-5-noup")
    # No upstream supplied → breakdown sub-phases default to 0, total is
    # just the (tiny) promotion-decision time.
    br = out["latency_breakdown"]
    assert br["llm_candidate_ms"] == 0
    assert br["total_ms"] == br["promotion_decision_ms"]
    assert out["m4_total_latency_ms"] == br["promotion_decision_ms"]
