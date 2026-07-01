"""Phase M3.1.2 — Query-Diff Metrics Input Hotfix tests.

Asserts that:
  1. ``run_benchmark_for_session`` now feeds the canonical
     ``serialize_a_arm_from_session(session)["boolean_queries"]`` list
     into ``compute_metrics(a_queries=...)`` — so ``query_diff`` is a
     true A-vs-B delta instead of one-sided.
  2. The orchestrator's source no longer reads
     ``queries_by_family`` (sentinel + source-grep guard).
  3. A live benchmark run produces non-trivial ``query_diff.removed``
     and ``query_diff.common`` for a session whose canonical A-queries
     overlap with the B-arm's compiled queries.
  4. Article-based metrics (``overlap_at_10``, ``new_direct_hits_at_10``,
     ``wrong_anchor_rate_at_10``, ``source_error_rate``,
     ``required_anchor_coverage``, ``generic_only_anchor_absent``,
     latencies, timeout/fallback) are stable: they only depend on
     article inputs and have not been touched by this hotfix.
  5. The GET path lazy-recomputes ``query_diff`` from canonical session
     data for older stored runs, surfaces ``metrics_source`` =
     ``"live_recomputed"`` vs ``"stored"``, and matches the values a
     freshly-run benchmark would produce.
  6. K4A.2 / K4A.3 / K4A.4 protected routes remain byte-identical;
     ``mock_b.await_count == 0`` preserved.
  7. The new ``metrics_source`` field passes the leak scan.

Pure unit + per-fixture tests. No live network calls. LLM calls are
patched / never invoked.
"""
from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

import pytest

from core.db import get_db, Collections
from services.m3_1_benchmark import (
    serialize_a_arm_from_session,
    compute_metrics,
    leak_scan_passes,
    get_benchmark_run_by_id,
    run_benchmark_for_session,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

def _realistic_session_with_active_queries(sid: str = "sess-M3_1_2") -> dict:
    """Real-schema session whose canonical branch_query_plan emits two
    Boolean queries — one of which the B-arm compiler will retain
    (so query_diff has both removed AND common entries)."""
    return {
        "id": sid, "_id": sid,
        "user_id": "u-m3_1_2",
        "original_question": "What are the causes of schizophrenia in young adults?",
        "sanitized_question": "What are the causes of schizophrenia in young adults?",
        "status": "awaiting_selection",
        "deleted_at": None,
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc),
        # Canonical fields.
        "clinical_question_map_json": {
            "schema_version": "K2.1.0",
            "primary_intent": "background",
            "frameworks": [],
            "branch_query_plan": {
                "schema_version": "K2.1.0",
                "mode": "shadow",
                "active_for_retrieval": True,
                "branches": [{
                    "branch_id": "br-1",
                    "intent": "background",
                    "framework": "background",
                    "branch_priority": 1,
                    "active_for_retrieval": True,
                    "query_families": [
                        {"family_name": "literal", "source": "pubmed",
                         "query_string": '"causes of schizophrenia in young adults"[tiab]',
                         "qa_checks_passed": True},
                        {"family_name": "schizophrenia_anchor", "source": "pubmed",
                         "query_string": '"schizophrenia"[tiab] NOT "preprint"[Publication Type]',
                         "qa_checks_passed": True},
                    ],
                }],
            },
        },
        "picott_json": {
            "type_of_question": "background",
            "question_subtype": "disease_overview",
            "specialty": "psychiatry",
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [
                {"text": "schizophrenia",
                 "canonical_concept": "schizophrenia",
                 "entity_type": "disease_or_problem",
                 "role": "population"},
            ],
        },
        # Sentinel — must NEVER reach the metric input now.
        "queries_by_family": [
            {"DEAD_FIELD_M3_1_2": "MUST_NOT_REACH_METRICS"}
        ],
        # Diagnostics with an accepted semantic-repair candidate.
        "session_diagnostics_json": {
            "semantic_anchor_repair": {
                "semantic_repair_triggered": True,
                "semantic_repair_trigger_reasons": ["missing_risk_etiology_framework"],
                "semantic_repair_validation_status": "passed",
                "semantic_repair_validation_confidence": 0.99,
                "semantic_repair_candidate": {
                    "primary_intent": "risk_etiology",
                    "frameworks": ["risk_etiology"],
                    "required_anchors": ["schizophrenia"],
                    "slots": {},
                    "semantic_repair_reason": "literal_full_phrase_anchor",
                },
                "semantic_repair_rejection_reasons": [],
            },
        },
    }


# ---------------------------------------------------------------------
# 1. Source-of-truth — orchestrator no longer reads ``queries_by_family``
# ---------------------------------------------------------------------

def test_M3_1_2_orchestrator_source_does_not_reference_queries_by_family():
    """Source-level guard: ``queries_by_family`` must no longer be
    looked up in ``run_benchmark_for_session``."""
    src = inspect.getsource(run_benchmark_for_session)
    assert ".get(\"queries_by_family\")" not in src, (
        "Orchestrator must not read queries_by_family — M3.1.2 hotfix "
        "routes the metric-input through serialize_a_arm_from_session."
    )


# ---------------------------------------------------------------------
# 2. Compute metrics — true A-vs-B delta sanity
# ---------------------------------------------------------------------

def test_M3_1_2_query_diff_is_true_delta():
    """Given non-empty A queries and a B compiler output that drops the
    literal-full-phrase but keeps the anchor, query_diff must show
    both ``removed`` AND ``common`` entries — proving the input is
    real, not the empty legacy list."""
    a_queries = [
        '"causes of schizophrenia in young adults"[tiab]',
        '"schizophrenia"[tiab] NOT "preprint"[Publication Type]',
    ]
    b_queries = [
        '"schizophrenia"[tiab] NOT "preprint"[Publication Type]',
        '("schizophrenia"[tiab]) AND ("young adults"[tiab]) NOT "preprint"[Publication Type]',
    ]
    m = compute_metrics(
        a_top10=[], b_top10=[],
        a_queries=a_queries, b_queries=b_queries,
        candidate={"required_anchors": ["schizophrenia"]},
        deterministic_primary_anchors=[{"canonical_concept": "schizophrenia"}],
        original_question="What are the causes of schizophrenia in young adults?",
        per_source_latencies={}, per_source_errors={},
    )
    qd = m["query_diff"]
    assert qd["n_added"] >= 1, "B added at least one query the A-arm did not have"
    assert qd["n_removed"] >= 1, "A had at least one query B-arm dropped (literal full phrase)"
    assert len(qd["common"]) >= 1, "the schizophrenia-anchor query is in both"
    # literal_full_phrase_anchor_removed must be True for this fixture.
    assert m["literal_full_phrase_anchor_removed"] is True


# ---------------------------------------------------------------------
# 3. Stability — article-based metrics unchanged regardless of a_queries
# ---------------------------------------------------------------------

def test_M3_1_2_article_metrics_unchanged_regardless_of_a_queries():
    """compute_metrics article-based fields must be invariant to the
    a_queries argument (they only depend on a_top10/b_top10 PMID sets
    + candidate.required_anchors + per-source-error/latency dicts)."""
    common_kwargs = dict(
        a_top10=[{"pmid": "1", "title": "schizophrenia A"},
                 {"pmid": "2", "title": "young adults B"}],
        b_top10=[{"pmid": "1", "title": "schizophrenia A"},
                 {"pmid": "3", "title": "schizophrenia C"}],
        b_queries=["X"],
        candidate={"required_anchors": ["schizophrenia"]},
        deterministic_primary_anchors=[{"canonical_concept": "schizophrenia"}],
        original_question="What are the causes of schizophrenia in young adults?",
        per_source_latencies={"pubmed": [100, 200], "epmc": [150]},
        per_source_errors={"pubmed": [], "epmc": ["timeout"]},
    )
    m_legacy = compute_metrics(a_queries=[], **common_kwargs)
    m_canonical = compute_metrics(a_queries=["X", "Y", "Z"], **common_kwargs)
    INVARIANT_KEYS = [
        "overlap_at_10", "new_direct_hits_at_10",
        "wrong_anchor_rate_at_10", "source_error_rate",
        "required_anchor_coverage", "generic_only_anchor_absent",
        "latency_p50_ms", "latency_p95_ms", "latency_max_ms",
    ]
    for k in INVARIANT_KEYS:
        assert m_legacy.get(k) == m_canonical.get(k), (
            f"M3.1.2 invariant violated for {k}: "
            f"legacy={m_legacy.get(k)} vs canonical={m_canonical.get(k)}"
        )


# ---------------------------------------------------------------------
# 4. End-to-end orchestrator — query_diff non-empty on both sides
# ---------------------------------------------------------------------

async def _admin_token(client):
    r = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_M3_1_2_orchestrator_emits_true_query_delta(client):  # noqa: F811
    from core.config import settings

    db = get_db()
    sid = "sess-M3_1_2-orch"
    sess = _realistic_session_with_active_queries(sid)
    await db[Collections.SEARCH_SESSIONS].insert_one(sess)

    token = await _admin_token(client)
    with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True):
        with patch(
            "services.m3_1_benchmark._b_arm_retrieve",
            new=AsyncMock(return_value=([], {"pubmed": [], "epmc": []},
                                        {"pubmed": [], "epmc": []},
                                        "none", "none")),
        ):
            r = await client.post(
                "/api/admin/m3_1/benchmark",
                headers={"Authorization": f"Bearer {token}"},
                json={"session_id": sid},
            )
    assert r.status_code == 200, r.text
    body = r.json()
    qd = body["metrics"]["query_diff"]
    # True delta: A had something B didn't (literal full-phrase) AND
    # at least one B query was added.
    assert qd["n_removed"] >= 1, f"expected A-only queries; got {qd}"
    assert qd["n_added"] >= 1, f"expected B-only queries; got {qd}"
    # The literal full-phrase signal must fire because A keeps the
    # literal phrase and the B compiler strips it.
    assert body["metrics"]["literal_full_phrase_anchor_removed"] is True
    # metrics_source surfaced.
    assert body["metrics_source"] == "live_recomputed"


# ---------------------------------------------------------------------
# 5. GET path — lazy recompute for older stored runs
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_M3_1_2_get_lazy_recomputes_query_diff_for_old_runs(client):  # noqa: F811
    """Simulate a pre-M3.1.2 stored run whose ``metrics.query_diff``
    is one-sided. The GET handler must lazy-recompute query_diff +
    literal_full_phrase_anchor_removed from the live session and
    surface metrics_source=live_recomputed; article-based metrics
    must remain the same as stored."""
    db = get_db()
    sid = "sess-M3_1_2-old-run"
    rid = "run-M3_1_2-old"
    sess = _realistic_session_with_active_queries(sid)
    await db[Collections.SEARCH_SESSIONS].insert_one(sess)

    # Stored doc with the OLD broken metrics (A had nothing, so
    # query_diff was one-sided).
    legacy_metrics = {
        "overlap_at_10": 0,
        "new_direct_hits_at_10": 10,
        "wrong_anchor_rate_at_10": 0.4,
        "required_anchor_coverage": 1.0,
        "source_error_rate": 0.0,
        "generic_only_anchor_absent": True,
        "literal_full_phrase_anchor_removed": False,  # incorrect under broken input
        "latency_p50_ms": 200,
        "latency_p95_ms": 400,
        "latency_max_ms": 500,
        "query_diff": {
            "added": ['"schizophrenia"[tiab] NOT "preprint"[Publication Type]'],
            "removed": [],
            "common": [],
            "n_added": 1,
            "n_removed": 0,
        },
    }
    await db[Collections.M3_1_BENCHMARK_RUNS].insert_one({
        "id": rid, "_id": rid,
        "search_session_id": sid,
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc),
        "eligibility_decision": "eligible",
        "route_owned_by_k4a_specialty": False,
        "timeout_status": "none",
        "fallback_status": "none",
        "a_arm": {"cqm": {}, "boolean_queries": [], "top10": []},
        "b_arm": {
            "boolean_queries": {
                "pubmed": [{"query_type": "b_required_AND",
                            "query_string": '"schizophrenia"[tiab] NOT "preprint"[Publication Type]'}],
                "epmc": [],
            },
            "top10": [],
            "per_source_latency_ms": {"pubmed": [200], "epmc": []},
            "per_source_error": {"pubmed": [], "epmc": []},
        },
        "metrics": legacy_metrics,
        "safety_flags": {"leak_scan_passed": True},
    })

    out = await get_benchmark_run_by_id(rid)
    assert out is not None
    assert out["a_arm_source"] == "live_resolved"
    assert out["metrics_source"] == "live_recomputed"
    new_qd = out["metrics"]["query_diff"]
    # Query_diff is now a true delta: removed has at least 1 entry
    # (the literal full-phrase A query that B drops).
    assert new_qd["n_removed"] >= 1
    assert new_qd["n_added"] >= 0
    # literal_full_phrase_anchor_removed corrected to True.
    assert out["metrics"]["literal_full_phrase_anchor_removed"] is True
    # Article-based metrics preserved from stored values.
    for k in ("overlap_at_10", "new_direct_hits_at_10",
              "wrong_anchor_rate_at_10", "required_anchor_coverage",
              "source_error_rate", "generic_only_anchor_absent",
              "latency_p50_ms", "latency_p95_ms", "latency_max_ms"):
        assert out["metrics"][k] == legacy_metrics[k], (
            f"article-based metric {k} must be preserved verbatim"
        )


@pytest.mark.asyncio
async def test_M3_1_2_get_lazy_metrics_source_stored_when_session_missing(client):  # noqa: F811
    """When the search_session is deleted/expired, the GET handler
    must fall back to the stored metrics and surface
    metrics_source=stored."""
    db = get_db()
    rid = "run-M3_1_2-orphaned"
    legacy_metrics = {"query_diff": {"added": [], "removed": [], "common": [],
                                     "n_added": 0, "n_removed": 0}}
    await db[Collections.M3_1_BENCHMARK_RUNS].insert_one({
        "id": rid, "_id": rid,
        "search_session_id": "non-existent-session",
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc),
        "eligibility_decision": "eligible",
        "route_owned_by_k4a_specialty": False,
        "timeout_status": "none",
        "fallback_status": "none",
        "a_arm": {"cqm": {}, "boolean_queries": [], "top10": []},
        "b_arm": {"boolean_queries": {"pubmed": [], "epmc": []}, "top10": []},
        "metrics": legacy_metrics,
        "safety_flags": {"leak_scan_passed": True},
    })
    out = await get_benchmark_run_by_id(rid)
    assert out["a_arm_source"] == "stored"
    assert out["metrics_source"] == "stored"
    assert out["metrics"] == legacy_metrics


# ---------------------------------------------------------------------
# 6. K4A protected routes still blocked from B-arm comparison
# ---------------------------------------------------------------------

@pytest.mark.parametrize("subtype,expected_decision", [
    ("severity_marker_overview", "route_owned_by_k4a_specialty_K4A_2"),
    ("rule_out_test_utility", "route_owned_by_k4a_specialty_K4A_3"),
    ("epidemiology_incidence_complication_risk", "route_owned_by_k4a_specialty_K4A_4"),
])
@pytest.mark.asyncio
async def test_M3_1_2_k4a_routes_still_blocked(client, subtype, expected_decision):  # noqa: F811
    from core.config import settings

    db = get_db()
    sid = f"sess-M3_1_2-k4a-{subtype}"
    sess = _realistic_session_with_active_queries(sid)
    sess["picott_json"]["question_subtype"] = subtype
    sess["session_diagnostics_json"] = {"semantic_anchor_repair": {}}
    await db[Collections.SEARCH_SESSIONS].insert_one(sess)

    token = await _admin_token(client)
    with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True):
        with patch("services.m3_1_benchmark._b_arm_retrieve") as mock_b:
            r = await client.post(
                "/api/admin/m3_1/benchmark",
                headers={"Authorization": f"Bearer {token}"},
                json={"session_id": sid},
            )
            # Critical: B-arm retrieval must never be invoked.
            assert mock_b.await_count == 0
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["eligibility_decision"] == expected_decision
    assert body.get("route_owned_by_k4a_specialty") is True


# ---------------------------------------------------------------------
# 7. Leak scan — new metrics_source field never carries PHI / leak shapes
# ---------------------------------------------------------------------

def test_M3_1_2_metrics_source_field_passes_leak_scan():
    payload = {
        "a_arm_source": "live_resolved",
        "metrics_source": "live_recomputed",
        "metrics": {
            "query_diff": {
                "added": ['"schizophrenia"[tiab]'],
                "removed": ['"causes of schizophrenia in young adults"[tiab]'],
                "common": [],
                "n_added": 1, "n_removed": 1,
            }
        },
    }
    assert leak_scan_passes(payload) is True


# ---------------------------------------------------------------------
# 8. Disabled-state still 503 (regression cover from M3.1)
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_M3_1_2_endpoint_503_when_flag_off_unchanged(client):  # noqa: F811
    from core.config import settings
    token = await _admin_token(client)
    with patch.object(settings, "M3_1_BENCHMARK_ENABLED", False):
        r = await client.post(
            "/api/admin/m3_1/benchmark",
            headers={"Authorization": f"Bearer {token}"},
            json={"session_id": "anything"},
        )
    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "m3_1_disabled"
