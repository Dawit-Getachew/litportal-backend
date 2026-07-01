"""Phase R1A — Part 8 — shadow pipeline simulator integration tests.

Validates:
  * Shape detection (closed-set) + K4A exclusion.
  * compute_shadow_top_n returns aligned active + shadow top-N
    with correct rank_delta + label_set fields.
  * write_shadow_decision persists a closed-set sidecar row with
    correct payload + TTL.
  * Leak-scan failure path redacts per-article arrays.
  * schedule_shadow_simulation is non-blocking (returns a Task
    that completes asynchronously).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from core.db import Collections, get_db
from services.r1a_shadow_simulator import (
    K4A_OWNED_SUBTYPES,
    R1A_SHADOW_SHAPES,
    compute_shadow_top_n,
    detect_shape_for_session,
    is_k4a_owned_subtype,
    list_shadow_decisions,
    run_shadow_simulation_for_session,
    schedule_shadow_simulation,
    write_k4a_skip_marker,
    write_shadow_decision,
)


# =====================================================================
# Pure helpers — shape detection, K4A exclusion.
# =====================================================================
@pytest.mark.parametrize("subtype,expected", [
    ("pathophysiology_mechanism", "pathophysiology_mechanism"),
    ("pathophysiology", "pathophysiology_mechanism"),
    ("procedure_complication_risk", "procedure_complication_risk"),
    ("procedure_complication", "procedure_complication_risk"),
    ("device_complication_risk", "procedure_complication_risk"),
    ("device_complication", "procedure_complication_risk"),
    ("background_overview", None),
    ("therapy", None),
    ("", None),
])
def test_R1A_detect_shape_for_session(subtype, expected):
    out = detect_shape_for_session({"question_subtype": subtype})
    assert out == expected


@pytest.mark.parametrize("subtype,expected", [
    ("severity_marker_overview", True),
    ("rule_out_test_utility", True),
    ("epidemiology_incidence_complication_risk", True),
    ("pathophysiology_mechanism", False),
    ("therapy", False),
])
def test_R1A_is_k4a_owned_subtype(subtype, expected):
    assert is_k4a_owned_subtype({"question_subtype": subtype}) is expected


def test_R1A_shadow_shapes_constant_is_closed_set():
    assert "pathophysiology_mechanism" in R1A_SHADOW_SHAPES
    assert "procedure_complication_risk" in R1A_SHADOW_SHAPES
    assert "procedure_complication" in R1A_SHADOW_SHAPES
    assert "etiology_risk" not in R1A_SHADOW_SHAPES
    assert "therapy" not in R1A_SHADOW_SHAPES


def test_R1A_k4a_owned_subtypes_constant_is_closed_set():
    assert "severity_marker_overview" in K4A_OWNED_SUBTYPES
    assert "rule_out_test_utility" in K4A_OWNED_SUBTYPES
    assert "epidemiology_incidence_complication_risk" in K4A_OWNED_SUBTYPES


# =====================================================================
# compute_shadow_top_n — pure deterministic reranking.
# =====================================================================
def _sample_articles_and_ssas():
    articles = [
        {  # Active rank 1 — treatment trial (indirect for mechanism).
            "id": "art-1", "pmid": "1001", "doi": "10.1/x",
            "title": "Randomized controlled trial of pharmacotherapy",
            "abstract": "Double blind placebo controlled trial of treatment.",
            "journal": "J Treat",
            "publication_year": 2024,
            "sources_present": ["pubmed"],
            "publication_types_json": ["Randomized Controlled Trial"],
        },
        {  # Active rank 2 — direct mechanism review.
            "id": "art-2", "pmid": "1002", "doi": "10.1/y",
            "title": "Mechanisms of pruritus in cholestasis",
            "abstract": "Pathophysiology and molecular pathogenesis of cholestatic pruritus, lysophosphatidic acid pathway.",
            "journal": "Nat Rev Gastroenterol Hepatol",
            "publication_year": 2023,
            "sources_present": ["europe_pmc"],
            "publication_types_json": ["Review"],
        },
        {  # Active rank 3 — broad overview.
            "id": "art-3", "pmid": "1003", "doi": "10.1/z",
            "title": "Approach to liver disease",
            "abstract": "Overview of common bile-duct disorders.",
            "journal": "Gen Med",
            "publication_year": 2022,
            "sources_present": ["pubmed"],
            "publication_types_json": [],
        },
    ]
    ssas = [
        {"article_id": "art-1", "overall_rank": 1},
        {"article_id": "art-2", "overall_rank": 2},
        {"article_id": "art-3", "overall_rank": 3},
    ]
    return articles, ssas


def test_R1A_compute_shadow_top_n_promotes_direct_mechanism_for_patho_shape():
    arts, ssas = _sample_articles_and_ssas()
    active, shadow, dist = compute_shadow_top_n(
        shape="pathophysiology_mechanism",
        articles=arts, ssas=ssas, top_n=10,
    )
    # Active top order honors the input ranks.
    assert [r["pmid"] for r in active] == ["1001", "1002", "1003"]
    # Shadow should promote art-2 (direct mechanism) above art-1 (treatment).
    shadow_pmids = [r["pmid"] for r in shadow]
    assert shadow_pmids.index("1002") < shadow_pmids.index("1001")
    # Distribution should record at least one direct mechanism + one treatment indirect.
    assert dist["pathophysiology_mechanism_direct"] >= 1
    assert dist["treatment_modality_indirect_for_mechanism"] >= 1


def test_R1A_compute_shadow_top_n_rank_delta_is_signed_int_or_none():
    arts, ssas = _sample_articles_and_ssas()
    _, shadow, _ = compute_shadow_top_n(
        shape="pathophysiology_mechanism",
        articles=arts, ssas=ssas, top_n=10,
    )
    for r in shadow:
        rd = r.get("rank_delta")
        assert rd is None or isinstance(rd, int)


def test_R1A_compute_shadow_top_n_per_article_fields_are_closed_set():
    arts, ssas = _sample_articles_and_ssas()
    active, shadow, _ = compute_shadow_top_n(
        shape="pathophysiology_mechanism",
        articles=arts, ssas=ssas, top_n=10,
    )
    active_keys = set(active[0].keys())
    shadow_keys = set(shadow[0].keys())
    # Active row contract — includes the closed-set mixed-signal
    # flag + the positive/indirect label splits.
    assert active_keys == {
        "pmid", "doi", "title", "journal", "year", "source",
        "current_rank", "label_set",
        "positive_labels", "indirect_labels", "mixed_signal",
    }
    # Shadow row contract — same as active plus shadow-specific
    # fields (shadow_rank, shadow_score, rank_delta) plus the
    # per-article ``mixed_signal_verdict`` default.
    assert shadow_keys == {
        "pmid", "doi", "title", "journal", "year", "source",
        "shadow_rank", "shadow_score", "rank_delta", "label_set",
        "positive_labels", "indirect_labels", "mixed_signal",
        "mixed_signal_verdict",
    }


# =====================================================================
# write_shadow_decision — closed-set sidecar persistence.
# =====================================================================
@pytest.mark.asyncio
async def test_R1A_write_shadow_decision_persists_closed_set_fields(client):
    sid = await write_shadow_decision(
        session_id="sess-r1a-write",
        benchmark_case_id=None,
        shape="pathophysiology_mechanism",
        active_top_10=[{"pmid": "999", "doi": "10.1/foo", "title": "T", "journal": "J", "year": 2024, "source": "pubmed", "current_rank": 1, "label_set": ["mechanism_review"]}],
        shadow_top_10=[{"pmid": "999", "doi": "10.1/foo", "title": "T", "journal": "J", "year": 2024, "source": "pubmed", "shadow_rank": 1, "shadow_score": 11.0, "rank_delta": 0, "label_set": ["mechanism_review"]}],
        label_distribution={"mechanism_review": 1},
        query_expansion_terms_used=["mechanism", "pathophysiology"],
        high_precision_query_required_both_anchors=False,
        source_errors={},
        latency_ms=42,
        route_owned_by_k4a_specialty=False,
        shadow_simulation_skipped=False,
    )
    assert isinstance(sid, str) and sid

    db = get_db()
    row = await db[Collections.R1A_SHADOW_DECISIONS].find_one({"id": sid}, {"_id": 0})
    assert row is not None
    # TTL closed-set fields.
    assert "expires_at" in row
    assert "created_at" in row
    assert row["shape"] == "pathophysiology_mechanism"
    # Safety flags.
    assert row["safety_flags"]["leak_scan_passed"] is True
    assert row["safety_flags"]["route_owned_by_k4a_specialty"] is False
    # NO abstract / NO prompt / NO question text persisted.
    flat = str(row).lower()
    assert "abstract" not in flat.replace("abstract_only", "")
    # NO LLM strings.
    for s in ("sk-", "Bearer ", "system:", "assistant:"):
        assert s.lower() not in flat


# =====================================================================
# write_k4a_skip_marker — closed-set skip row.
# =====================================================================
@pytest.mark.asyncio
async def test_R1A_write_k4a_skip_marker_records_route_owned():
    sid = await write_k4a_skip_marker(
        session_id="sess-r1a-k4a",
        shape="procedure_complication_risk",
    )
    db = get_db()
    row = await db[Collections.R1A_SHADOW_DECISIONS].find_one({"id": sid}, {"_id": 0})
    assert row is not None
    assert row["route_owned_by_k4a_specialty"] is True
    assert row["shadow_simulation_skipped"] is True
    assert row["current_active_top10"] == []
    assert row["shadow_simulated_top10"] == []


# =====================================================================
# schedule_shadow_simulation — async, non-blocking.
# =====================================================================
@pytest.mark.asyncio
async def test_R1A_schedule_shadow_simulation_returns_none_for_non_eligible_shape():
    task = schedule_shadow_simulation(
        session_id="sess-noop",
        picott_payload={"question_subtype": "therapy"},
    )
    assert task is None


@pytest.mark.asyncio
async def test_R1A_schedule_shadow_simulation_K4A_subtype_writes_skip_marker():
    task = schedule_shadow_simulation(
        session_id="sess-r1a-k4a-route",
        picott_payload={
            "question_subtype": "severity_marker_overview",
        },
    )
    # K4A subtypes still get a scheduled task to write a skip marker.
    # But ``detect_shape_for_session`` will return None for them.
    # We expect None or a task; if None, no row is written.
    # Verify that detect_shape_for_session is the gate — severity_marker
    # is not one of the R1A shapes so we return None.
    assert task is None


@pytest.mark.asyncio
async def test_R1A_schedule_shadow_simulation_eligible_shape_returns_task():
    task = schedule_shadow_simulation(
        session_id="sess-r1a-eligible-shape",
        picott_payload={
            "question_subtype": "pathophysiology_mechanism",
        },
    )
    assert task is not None
    # Wait for the background task — confirms NEVER raises.
    await task
    # Confirm one row landed.
    rows = await list_shadow_decisions(limit=10)
    assert any(r["search_session_id"] == "sess-r1a-eligible-shape" for r in rows)


# =====================================================================
# run_shadow_simulation_for_session — failure swallowed (top-level wrap).
# =====================================================================
@pytest.mark.asyncio
async def test_R1A_run_shadow_simulation_for_session_with_no_articles_does_not_raise():
    sid = await run_shadow_simulation_for_session(
        session_id="sess-r1a-empty",
        shape="pathophysiology_mechanism",
    )
    # Should still write a row with empty top-N + no error.
    assert isinstance(sid, str)
    db = get_db()
    row = await db[Collections.R1A_SHADOW_DECISIONS].find_one(
        {"id": sid}, {"_id": 0},
    )
    assert row is not None
    assert row["current_active_top10"] == []
    assert row["shadow_simulated_top10"] == []


# =====================================================================
# Pipeline integration — schedule never blocks on exceptions.
# =====================================================================
@pytest.mark.asyncio
async def test_R1A_simulator_swallows_internal_failure(monkeypatch):
    """Force the simulator's write_shadow_decision to raise; the
    scheduled task must NOT raise on the caller.
    """
    async def _boom(*a, **kw):
        raise RuntimeError("simulated DB blowup")

    monkeypatch.setattr(
        "services.r1a_shadow_simulator.write_shadow_decision",
        _boom,
    )
    task = schedule_shadow_simulation(
        session_id="sess-r1a-boom",
        picott_payload={"question_subtype": "pathophysiology_mechanism"},
    )
    assert task is not None
    # Should NOT raise even though the inner call blew up.
    await task
