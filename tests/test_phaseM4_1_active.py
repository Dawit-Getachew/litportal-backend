"""Phase M4.1 — Preview-only Active M4 promotion tests.

Covers the config (fail-closed + env-driven), the candidate→compiler
mapping, the cb_record compiler, the closed-set eligibility decision,
the startup guard, the Mongo kill switch, and the async orchestrator
(applied / fallback / kill / production / M3.2-conflict paths).

NEVER touches a real LLM. Uses the conftest mongomock DB for the
sidecar + feature-flag collections.
"""
from __future__ import annotations

import pytest

from services.m4_active_config import M4ActiveConfig, load_m4_active_config
from services.m4_active_guard import verify_m4_active_startup
from services.m4_active_pipeline import (
    M4_ACTIVE_CHIP_PRIMARY_TEXT,
    compile_m4_active_cb_records,
    m4_active_chip_payload,
    m4_candidate_to_compiler_candidate,
    run_m4_active_decision,
)
from services.m4_active_feature_flags import (
    M4_ACTIVE_KILL_SWITCH_KEY,
    is_m4_mongo_kill_engaged,
    read_m4_kill_switch_state,
    set_m4_mongo_kill_switch_state,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

def _accepted_candidate() -> dict:
    return {
        "primary_intent": "prognosis_natural_history",
        "secondary_intents": ["prognostic_factor"],
        "frameworks": ["PFO", "prognosis"],
        "clinical_problem": ["colon cancer"],
        "stage": ["stage I"],
        "biomarker": [],
        "population": ["patients with stage I colon cancer"],
        "exposure": [],
        "intervention": [],
        "procedure": [],
        "medication": [],
        "index_test": [],
        "outcome": ["overall survival", "recurrence"],
        "setting": [],
        "timeframe": [],
        "mechanism": [],
        "comparator": [],
        "required_anchors": ["colon cancer"],
        "optional_or_ranking_only_terms": ["prognosis", "stage I"],
        "candidate_synonyms_unvalidated": ["colorectal cancer"],
        "candidate_mesh_terms_unvalidated": ["Colonic Neoplasms"],
        "clarification_required": False,
        "confidence": 0.9,
        "repair_reason": "long_literal_phrase_anchor",
        "provenance": "m4_full_question_interpreter_shadow",
    }

def _clean_rescue_candidate() -> dict:
    """Etiology rescue that PRESERVES every central concept (passes the
    Phase M4.4 specificity gate): clean disease anchor + named exposure
    kept, no biomarker/stage to demote, no central concept dropped."""
    return {
        "primary_intent": "etiology",
        "secondary_intents": [],
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
        "setting": [],
        "timeframe": [],
        "mechanism": [],
        "comparator": [],
        "required_anchors": ["pulmonary embolism"],
        "optional_or_ranking_only_terms": [],
        "candidate_synonyms_unvalidated": [],
        "candidate_mesh_terms_unvalidated": [],
        "clarification_required": False,
        "confidence": 0.9,
        "repair_reason": "full_literal_phrase_anchor",
        "provenance": "m4_full_question_interpreter_shadow",
    }




def _m4_block(*, accepted: bool = True, projection_status: str = "passed") -> dict:
    cand = _clean_rescue_candidate()
    proj = {
        "projected_compiler_input": {"required_anchors": ["pulmonary embolism"]},
        "projection_validation_status": projection_status,
        "projection_validation_flags": [] if projection_status == "passed" else ["stage_dropped"],
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


def _picott() -> dict:
    return {
        "type_of_question": "background",
        "question_subtype": "disease_overview",
        "framework_list_json": ["background"],
        "specialty": "oncology",
    }


def _async_return(value):
    async def _f():
        return value
    return _f()


# ---------------------------------------------------------------------
# Config — fail-closed defaults + env-driven.
# ---------------------------------------------------------------------

def test_config_defaults_are_fail_closed():
    cfg = load_m4_active_config(env={})
    assert cfg.enabled is False
    assert cfg.kill_switch is True
    assert cfg.kill_switch_engaged is True
    assert cfg.effective_enabled is False


def test_config_preview_enabled_disengaged_is_effective():
    cfg = load_m4_active_config(env={
        "M4_ACTIVE_ENABLED": "true",
        "M4_ACTIVE_KILL_SWITCH": "false",
        "APP_ENV": "preview",
    })
    assert cfg.effective_enabled is True


def test_config_production_hard_off_without_prod_allow():
    cfg = load_m4_active_config(env={
        "M4_ACTIVE_ENABLED": "true",
        "M4_ACTIVE_KILL_SWITCH": "false",
        "APP_ENV": "production",
    })
    assert cfg.is_production_env is True
    assert cfg.effective_enabled is False


# ---------------------------------------------------------------------
# Candidate → compiler mapping.
# ---------------------------------------------------------------------

def test_candidate_to_compiler_maps_slots_and_folds_stage_biomarker():
    cand = _accepted_candidate()
    cand["biomarker"] = ["HER2-positive"]
    cc = m4_candidate_to_compiler_candidate(cand)
    assert cc["required_anchors"] == ["colon cancer"]
    # stage + biomarker folded into population precision context.
    pop = [p.lower() for p in cc["slots"]["population"]]
    assert "stage i" in pop
    assert "her2-positive" in pop
    assert "colon cancer" in [c.lower() for c in cc["slots"]["condition"]]
    assert "overall survival" in [o.lower() for o in cc["slots"]["outcome"]]


def test_compile_cb_records_tags_planner_source_and_maps_sources():
    recs = compile_m4_active_cb_records(
        _accepted_candidate(), ["europe_pmc", "pubmed"],
    )
    assert recs, "expected at least one cb record"
    assert all(r["planner_source"] == "m4_active" for r in recs)
    assert all(r["query_family"].startswith("m4_active_") for r in recs)
    srcs = {r["source"] for r in recs}
    assert srcs <= {"pubmed", "europe_pmc"}
    # No fused literal full-phrase anchor leaks into the query.
    assert all("prognosis of stage" not in r["query_string"].lower() for r in recs)


def test_chip_payload_exact_string():
    chip = m4_active_chip_payload()
    assert chip["primary_text"] == "Clinical question interpretation refined"
    assert M4_ACTIVE_CHIP_PRIMARY_TEXT == "Clinical question interpretation refined"
    # No LLM specifics in the tooltip.
    low = chip["tooltip_text"].lower()
    assert "llm" not in low and "prompt" not in low and "model" not in low


# ---------------------------------------------------------------------
# Startup guard.
# ---------------------------------------------------------------------

def test_guard_non_production_is_noop():
    verify_m4_active_startup(is_production=False, env={
        "M4_ACTIVE_ENABLED": "true", "M4_ACTIVE_KILL_SWITCH": "false",
    })  # no raise


def test_guard_production_default_safe_is_noop():
    verify_m4_active_startup(is_production=True, env={})  # defaults safe


def test_guard_production_enabled_without_prod_allow_raises():
    with pytest.raises(RuntimeError):
        verify_m4_active_startup(is_production=True, env={
            "M4_ACTIVE_ENABLED": "true",
            "M4_ACTIVE_KILL_SWITCH": "false",
        })


def test_guard_production_prod_allow_without_allowlist_raises():
    with pytest.raises(RuntimeError):
        verify_m4_active_startup(is_production=True, env={
            "M4_ACTIVE_ENABLED": "true",
            "M4_ACTIVE_KILL_SWITCH": "false",
            "M4_ACTIVE_PROD_ALLOW_ACTIVE": "true",
            "M4_ACTIVE_ENV_ALLOWLIST": "dev,preview",
        })


# ---------------------------------------------------------------------
# Mongo kill switch (fail-closed + round trip).
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mongo_kill_default_engaged_then_toggle():
    # Default (no row) → engaged.
    assert await is_m4_mongo_kill_engaged() is True
    await set_m4_mongo_kill_switch_state(engaged=False, actor_user_id="admin-1")
    assert await is_m4_mongo_kill_engaged() is False
    state = await read_m4_kill_switch_state(_cfg(kill_switch=False))
    assert state["mongo_kill_switch_engaged"] is False
    assert state["combined_engaged"] is False
    assert state["flag_key"] == M4_ACTIVE_KILL_SWITCH_KEY
    # Re-engage.
    await set_m4_mongo_kill_switch_state(engaged=True, actor_user_id="admin-1")
    assert await is_m4_mongo_kill_engaged() is True


# ---------------------------------------------------------------------
# Orchestrator — eligible apply, kill, production, M3.2 conflict.
# ---------------------------------------------------------------------

async def _run(**overrides):
    base = dict(
        session_id="m4a-sess",
        user_id="user-1",
        m4_block=_m4_block(),
        picott_payload=_picott(),
        sources=["pubmed", "europe_pmc"],
        a_arm_cqm={"primary_intent": "background", "required_anchors": ["x"]},
        m3_2_applied=False,
        m3_2_kill_combined_engaged=True,  # M3.2 dormant
        baseline_degraded=True,  # M4.1.1: positive cases have a real defect
        baseline_degradation_reasons=["full_literal_phrase_required_anchor"],
        baseline_query_family_count=3,
        a_arm_boolean_queries=["baseline q1", "baseline q2"],
        config=_cfg(),
        mongo_kill_engaged=False,
    )
    base.update(overrides)
    return await run_m4_active_decision(**base)


@pytest.mark.asyncio
async def test_eligible_applies_and_records():
    out = await _run(session_id="m4a-elig")
    assert out["applied"] is True
    assert out["decision"] == "eligible"
    assert out["chip"]["primary_text"] == "Clinical question interpretation refined"
    assert out["cb_records"], "expected promoted cb records"
    assert all(r["planner_source"] == "m4_active" for r in out["cb_records"])
    assert out["kill_switch_state"]["combined_engaged"] is False


@pytest.mark.asyncio
async def test_candidate_not_accepted_blocks():
    out = await _run(session_id="m4a-noacc", m4_block=_m4_block(accepted=False))
    assert out["applied"] is False
    assert out["decision"] == "ineligible_candidate_not_accepted"
    assert out["chip"] is None


@pytest.mark.asyncio
async def test_projection_flagged_blocks():
    out = await _run(
        session_id="m4a-flag",
        m4_block=_m4_block(projection_status="flagged"),
    )
    assert out["applied"] is False
    assert out["decision"] == "ineligible_projection_flagged"


@pytest.mark.asyncio
async def test_kill_switch_engaged_blocks():
    out = await _run(
        session_id="m4a-kill",
        config=_cfg(kill_switch=True),
        mongo_kill_engaged=True,
    )
    assert out["applied"] is False
    assert out["decision"] == "ineligible_kill_switch_engaged"
    assert out["chip"] is None


@pytest.mark.asyncio
async def test_production_hard_off_blocks():
    out = await _run(
        session_id="m4a-prod",
        config=_cfg(current_env="production"),
    )
    assert out["applied"] is False
    assert out["decision"] == "ineligible_production_hard_off"


@pytest.mark.asyncio
async def test_master_disabled_blocks():
    out = await _run(session_id="m4a-dis", config=_cfg(enabled=False))
    assert out["applied"] is False
    assert out["decision"] == "ineligible_master_disabled"


@pytest.mark.asyncio
async def test_env_not_allowlisted_blocks():
    out = await _run(
        session_id="m4a-env",
        config=_cfg(env_allowlist=("dev",), current_env="qa"),
    )
    assert out["applied"] is False
    assert out["decision"] == "ineligible_env_not_allowlisted"


@pytest.mark.asyncio
async def test_m3_2_applied_blocks_double_engine():
    out = await _run(session_id="m4a-m32a", m3_2_applied=True)
    assert out["applied"] is False
    assert out["decision"] == "m3_2_already_applied_or_not_dormant"


@pytest.mark.asyncio
async def test_m3_2_not_dormant_blocks_double_engine():
    out = await _run(
        session_id="m4a-m32nd",
        m3_2_applied=False,
        m3_2_kill_combined_engaged=False,  # M3.2 live → not dormant
    )
    assert out["applied"] is False
    assert out["decision"] == "m3_2_already_applied_or_not_dormant"


@pytest.mark.asyncio
async def test_m3_2_unknown_state_blocks_fail_closed():
    out = await _run(
        session_id="m4a-m32unk",
        m3_2_applied=False,
        m3_2_kill_combined_engaged=None,  # unknown → treat as not dormant
    )
    assert out["applied"] is False
    assert out["decision"] == "m3_2_already_applied_or_not_dormant"
