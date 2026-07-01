"""Phase M4.1.1 — Active Promotion Eligibility Guard for Degraded
Baselines Only.

Covers the deterministic baseline-degradation assessment and the
promotion guard inside ``run_m4_active_decision`` that:
  * STILL applies M4 active for genuinely degraded baselines, and
  * BLOCKS promotion for already-clean baselines (the B8 fix), and
  * BLOCKS query-family diversity narrowing when no baseline defect.
"""
from __future__ import annotations

import pytest

from services.m4_active_config import M4ActiveConfig
from services.m4_active_pipeline import (
    assess_baseline_degradation,
    run_m4_active_decision,
    DEG_FULL_LITERAL_PHRASE,
    DEG_INTENT_WORD_FUSED,
    DEG_STAGE_BIOMARKER_POP_FUSED,
)


# ---------------------------------------------------------------------
# assess_baseline_degradation — closed-set, deterministic.
# ---------------------------------------------------------------------

def _assess(anchors, fw=None, intent=None, conf=None, cand=None):
    return assess_baseline_degradation(
        baseline_required_anchors=anchors,
        baseline_framework=fw or ["PFO"],
        baseline_intent=intent,
        baseline_confidence=conf,
        m4_candidate=cand or {"primary_intent": "prognosis_natural_history",
                              "clinical_problem": ["colon cancer"],
                              "required_anchors": ["colon cancer"]},
    )


# Degraded positive cases (must be flagged).
@pytest.mark.parametrize("anchor,expect_reason", [
    ("prognosis of stage 1 colon cancer", DEG_FULL_LITERAL_PHRASE),
    ("prognosis of idiopathic pulmonary fibrosis", DEG_INTENT_WORD_FUSED),
    ("Stage II Colon Cancer", DEG_STAGE_BIOMARKER_POP_FUSED),
    ("HER2 positive stage 2 breast cancer", DEG_STAGE_BIOMARKER_POP_FUSED),
    ("causes of schizophrenia in young adults", DEG_INTENT_WORD_FUSED),
])
def test_degraded_anchors_flagged(anchor, expect_reason):
    res = _assess([anchor])
    assert res["baseline_degraded"] is True
    assert expect_reason in res["reasons"]


def test_p3_recurrence_risk_and_stage_flagged():
    res = _assess(["stage I colon cancer", "recurrence risk"])
    assert res["baseline_degraded"] is True
    assert DEG_STAGE_BIOMARKER_POP_FUSED in res["reasons"]


# Clean baselines (must NOT be flagged) — the B8 family.
@pytest.mark.parametrize("anchors", [
    ["cholestatic liver disease", "pruritus"],            # B8
    ["smoking", "pancreatic cancer"],                      # B7
    ["central line placement", "pneumothorax"],            # B9
    ["colon cancer"],                                      # plain entity
])
def test_clean_anchors_not_flagged(anchors):
    res = _assess(anchors, cand={"primary_intent": "mechanism",
                                 "clinical_problem": anchors[:1],
                                 "required_anchors": anchors[:1]})
    assert res["baseline_degraded"] is False
    assert res["reasons"] == []


def test_b8_mechanism_framework_match_is_not_misforce():
    # concept_mechanism baseline + mechanism M4 intent => no misforce.
    res = assess_baseline_degradation(
        baseline_required_anchors=["cholestatic liver disease", "pruritus"],
        baseline_framework=["concept_mechanism"],
        baseline_intent="mechanism",
        baseline_confidence=0.8,
        m4_candidate={"primary_intent": "mechanism",
                      "clinical_problem": ["cholestatic liver disease"],
                      "required_anchors": ["cholestatic liver disease"]},
    )
    assert res["baseline_degraded"] is False


def test_overview_misforce_with_stronger_intent_flagged():
    res = assess_baseline_degradation(
        baseline_required_anchors=["colon cancer"],
        baseline_framework=["disease_overview"],
        baseline_intent="background",
        baseline_confidence=0.9,
        m4_candidate={"primary_intent": "prognosis_natural_history",
                      "clinical_problem": ["colon cancer"],
                      "required_anchors": ["colon cancer"]},
    )
    assert res["baseline_degraded"] is True
    assert "what_is_forced_overview_contradicts_m4_intent" in res["reasons"]


def test_no_baseline_anchor_but_m4_recovers_flagged():
    res = assess_baseline_degradation(
        baseline_required_anchors=[],
        baseline_framework=["PFO"],
        baseline_intent=None,
        baseline_confidence=None,
        m4_candidate={"primary_intent": "etiology",
                      "clinical_problem": ["sepsis"],
                      "required_anchors": ["sepsis"]},
    )
    assert res["baseline_degraded"] is True
    assert "no_clean_primary_entity_in_baseline" in res["reasons"]


# ---------------------------------------------------------------------
# Promotion guard inside run_m4_active_decision.
# ---------------------------------------------------------------------

def _m4_block_accepted():
    cand = {
        "primary_intent": "mechanism",
        "frameworks": ["mechanism"],
        "clinical_problem": ["cholestatic liver disease"],
        "stage": [], "biomarker": [], "population": [],
        "intervention": [], "exposure": [], "procedure": [], "medication": [],
        "index_test": [], "outcome": [], "mechanism": ["pruritus"],
        "required_anchors": ["cholestatic liver disease"],
        "optional_or_ranking_only_terms": [],
    }
    return {
        "mode": "shadow", "triggered": True,
        "trigger_reasons": ["what_is_forced_disease_overview"],
        "validation_status": "passed", "accepted": True,
        "candidate": cand,
        "m4_0_1_projection": {
            "projected_compiler_input": {"required_anchors": ["cholestatic liver disease"]},
            "projection_validation_status": "passed",
        },
    }


def _cfg():
    return M4ActiveConfig(enabled=True, env_allowlist=("preview", "dev"),
                          max_latency_ms=4000, kill_switch=False,
                          prod_allow_active=False, current_env="preview")


async def _run_guard(**ov):
    base = dict(
        session_id="m411", user_id="u1",
        m4_block=_m4_block_accepted(),
        picott_payload={"type_of_question": "mechanism",
                        "framework_list_json": ["concept_mechanism"]},
        sources=["pubmed", "europe_pmc"],
        a_arm_cqm={"primary_intent": "mechanism"},
        m3_2_applied=False, m3_2_kill_combined_engaged=True,
        baseline_degraded=False,  # B8: clean baseline
        baseline_degradation_reasons=[],
        baseline_query_family_count=7,
        a_arm_boolean_queries=["b1", "b2"],
        config=_cfg(), mongo_kill_engaged=False,
    )
    base.update(ov)
    return await run_m4_active_decision(**base)


@pytest.mark.asyncio
async def test_clean_baseline_blocks_with_diversity_reason():
    out = await _run_guard(session_id="m411-b8")
    assert out["applied"] is False
    assert out["decision"] == (
        "query_family_diversity_would_decrease_without_baseline_defect"
    )
    assert out["chip"] is None
    assert out["cb_records"] == []
    assert out["baseline_degraded"] is False


@pytest.mark.asyncio
async def test_clean_baseline_equal_families_blocks_already_clean():
    # M4 yields ~2 families; set baseline count below that so the
    # diversity comparison does not trigger → baseline_already_clean.
    out = await _run_guard(session_id="m411-clean", baseline_query_family_count=1)
    assert out["applied"] is False
    assert out["decision"] == "baseline_already_clean"


@pytest.mark.asyncio
async def test_degraded_baseline_still_applies():
    out = await _run_guard(
        session_id="m411-deg",
        baseline_degraded=True,
        baseline_degradation_reasons=["full_literal_phrase_required_anchor"],
        # M4 mechanism candidate yields 1 core family; keep the baseline
        # count at/below that so the M4.4 diversity check is satisfied.
        baseline_query_family_count=1,
    )
    assert out["applied"] is True
    assert out["decision"] == "eligible"
    assert out["chip"]["primary_text"] == "Clinical question interpretation refined"
    assert out["baseline_degraded"] is True
    assert out["m4_query_family_count"] >= 1
