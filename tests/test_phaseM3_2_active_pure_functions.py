"""Phase M3.2 — Active Semantic Repair pure-function tests.

DETERMINISTIC. NO LLM. NO RETRIEVAL. NO MONGO.

Validates the pure projection helpers (``chip_payload_for``,
``m_repaired_cqm_from_candidate``,
``m_repaired_compiler_input_from_candidate``,
``compile_m_repaired_cb_records``) and the closed-set decision
enum mappings. These tests do NOT touch Mongo and do NOT exercise
the orchestrator (the integration paths are covered in
``test_phaseM3_2_active_decision.py``).
"""
from __future__ import annotations

from services.semantic_repair_active_pipeline import (
    M3_2_CHIP_PRIMARY_TEXT,
    M3_2_ELIGIBILITY_DECISIONS,
    M3_2_FALLBACK_STATUSES,
    M3_2_SHAPE_HUMAN_LABELS,
    M3_2_TIMEOUT_STATUSES,
    chip_payload_for,
    compile_m_repaired_cb_records,
    m_repaired_compiler_input_from_candidate,
    m_repaired_cqm_from_candidate,
)


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
            "mechanism": [],
        },
        "semantic_repair_reason": "missing_risk_etiology_framework",
        "confidence": 0.85,
        "branches": [{"branch_id": "b1"}],
    }


# ---------------------------------------------------------------------
# Closed-set sanity.
# ---------------------------------------------------------------------
def test_M3_2_closed_sets_are_distinct() -> None:
    assert M3_2_ELIGIBILITY_DECISIONS.isdisjoint(M3_2_TIMEOUT_STATUSES)
    assert M3_2_ELIGIBILITY_DECISIONS.isdisjoint(M3_2_FALLBACK_STATUSES)
    assert "eligible" in M3_2_ELIGIBILITY_DECISIONS
    assert "ineligible_kill_switch_engaged" in M3_2_ELIGIBILITY_DECISIONS
    assert "ineligible_route_owned_by_k4a_specialty" in M3_2_ELIGIBILITY_DECISIONS
    assert "ineligible_shape_not_in_allowlist" in M3_2_ELIGIBILITY_DECISIONS
    assert "fallback_to_a_arm" in M3_2_FALLBACK_STATUSES
    assert "m3_2_active_timeout" in M3_2_TIMEOUT_STATUSES


def test_M3_2_chip_primary_text_is_shape_agnostic() -> None:
    assert M3_2_CHIP_PRIMARY_TEXT == "Clinical interpretation refined"


def test_M3_2_shape_human_labels_cover_default_allowlist() -> None:
    # Mirrors the closed-set hard-coded shape allowlist in
    # ``semantic_repair_active_config._DEFAULT_SHAPE_ALLOWLIST``.
    assert set(M3_2_SHAPE_HUMAN_LABELS.keys()) == {
        "etiology_risk",
        "pathophysiology_mechanism",
        "procedure_complication",
    }


# ---------------------------------------------------------------------
# chip_payload_for
# ---------------------------------------------------------------------
def test_M3_2_chip_payload_etiology_risk_renders_shape() -> None:
    c = chip_payload_for("etiology_risk")
    assert c["primary_text"] == "Clinical interpretation refined"
    assert c["shape_machine"] == "etiology_risk"
    assert c["shape_human"] == "etiology/risk"
    assert "etiology/risk" in c["tooltip_text"]
    assert "Source retrieval is unchanged" in c["tooltip_text"]


def test_M3_2_chip_payload_unknown_shape_renders_generic_tooltip() -> None:
    c = chip_payload_for("severity_marker")
    assert c["primary_text"] == "Clinical interpretation refined"
    assert c["shape_machine"] == "severity_marker"
    assert c["shape_human"] is None
    # Tooltip falls back to a generic phrasing.
    assert "(" not in c["tooltip_text"]  # no parenthetical shape label


def test_M3_2_chip_payload_none_shape_yields_generic() -> None:
    c = chip_payload_for(None)
    assert c["shape_human"] is None
    assert "Clinical interpretation refined" == c["primary_text"]


# ---------------------------------------------------------------------
# m_repaired_cqm_from_candidate / compiler_input
# ---------------------------------------------------------------------
def test_M3_2_m_repaired_cqm_projects_closed_set_fields() -> None:
    cand = _good_candidate()
    cqm = m_repaired_cqm_from_candidate(cand)
    assert cqm["primary_intent"] == "risk_etiology"
    assert cqm["frameworks"] == ["risk_etiology"]
    assert cqm["required_anchors"] == ["schizophrenia"]
    assert cqm["branch_count"] == 1
    assert cqm["semantic_repair_reason"] == "missing_risk_etiology_framework"
    assert cqm["confidence"] == 0.85


def test_M3_2_m_repaired_cqm_handles_none() -> None:
    cqm = m_repaired_cqm_from_candidate(None)
    assert cqm["primary_intent"] is None
    assert cqm["frameworks"] == []
    assert cqm["required_anchors"] == []
    assert cqm["branch_count"] == 0


def test_M3_2_m_repaired_compiler_input_includes_slots() -> None:
    inp = m_repaired_compiler_input_from_candidate(_good_candidate())
    assert inp["primary_intent"] == "risk_etiology"
    assert inp["required_anchors"] == ["schizophrenia"]
    assert inp["slots"]["population"] == ["young adults"]
    assert inp["semantic_repair_reason"] == "missing_risk_etiology_framework"


# ---------------------------------------------------------------------
# compile_m_repaired_cb_records — adapter
# ---------------------------------------------------------------------
def test_M3_2_cb_records_carry_planner_source_semantic_repair_active() -> None:
    records = compile_m_repaired_cb_records(
        _good_candidate(),
        sources=["pubmed", "europe_pmc"],
    )
    assert len(records) >= 1
    for rec in records:
        assert rec["planner_source"] == "semantic_repair_active"
        assert rec["is_primary_strategy"] is True
        assert rec["is_fallback_strategy"] is False
        assert rec["rationale"] == "phase_m3_2_active_semantic_repair"
        assert rec["query_family"].startswith("semantic_repair_")
        assert rec["query_string"]
        assert rec["source"] in {"pubmed", "europe_pmc"}


def test_M3_2_cb_records_respect_sources_filter() -> None:
    # If only pubmed is enabled, no europe_pmc records emit.
    records = compile_m_repaired_cb_records(
        _good_candidate(), sources=["pubmed"],
    )
    sources_seen = {r["source"] for r in records}
    assert sources_seen <= {"pubmed"}
    # And the queries are PubMed-shaped (contain ``[tiab]``).
    assert all("[tiab]" in r["query_string"] for r in records)


def test_M3_2_cb_records_empty_when_no_required_anchors() -> None:
    cand = dict(_good_candidate())
    cand["required_anchors"] = []
    cand["slots"] = {}  # also strip slot fallbacks
    records = compile_m_repaired_cb_records(
        cand, sources=["pubmed", "europe_pmc"],
    )
    assert records == []


def test_M3_2_cb_records_query_string_never_contains_literal_full_phrase() -> None:
    """Defense in depth — the B-arm compiler from m3_1_benchmark
    must never emit a literal multi-word phrase from a fictional
    full-question anchor. This is verified by inspecting the
    queries built only from validator-accepted ``required_anchors``
    + ``slots`` — never from the raw normalized_question."""
    cand = dict(_good_candidate())
    cand["required_anchors"] = ["thrombosis"]
    cand["slots"] = {
        "population": ["older adults"],
        "intervention_or_exposure": ["aspirin"],
    }
    records = compile_m_repaired_cb_records(
        cand, sources=["pubmed", "europe_pmc"],
    )
    bad_phrase = "what are the causes of"
    for r in records:
        assert bad_phrase not in r["query_string"].lower()
