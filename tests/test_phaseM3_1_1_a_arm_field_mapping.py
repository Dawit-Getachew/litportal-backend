"""Phase M3.1.1 — A-arm field mapping hotfix tests.

Asserts that:
  1. ``serialize_a_arm_from_session`` reads from the canonical
     ``search_sessions`` schema (``clinical_question_map_json``,
     ``clinical_question_map_json.branch_query_plan``, ``picott_json``,
     ``primary_clinical_anchors_json``) and ignores the dead
     ``picott_payload`` / ``queries_by_family`` fields completely.
  2. Both the packet-builder code path and the admin GET path call the
     same canonical serializer and produce byte-identical A-arm
     payloads for the same input session.
  3. A live benchmark run against a session populated with the real
     schema fields produces a non-empty A-arm CQM and a non-empty
     branch_query_plan in the API response.
  4. The lazy-resolved GET response carries ``a_arm_source ==
     "live_resolved"`` when the session is reachable, and
     ``"stored"`` when the session document is missing/expired.
  5. The K4A.2 / K4A.3 / K4A.4 route-owned-by-specialty decisions
     remain byte-identical (closed-set enum, B-arm not invoked).
  6. The new payload contains no PHI / prompt / completion /
     credential / full-text / evidence-text leak shapes.

Pure-function tests; no live network calls; LLM calls are
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
    get_benchmark_run_by_id,
    get_recent_benchmark_runs,
    leak_scan_passes,
)
from services.m3_1_adjudication_packet import packet_a_arm_section


# ---------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------

def _realistic_session_doc(sid: str = "sess-M3_1_1-A") -> dict:
    """A representative session doc using the REAL schema fields."""
    return {
        "id": sid,
        "_id": sid,
        "user_id": "user-m3_1_1",
        "clinical_question_map_json": {
            "schema_version": "K2.1.0",
            "primary_intent": "differential_diagnosis",
            "frameworks": ["problem_context_differential"],
            "branch_query_plan": {
                "schema_version": "K2.1.0",
                "mode": "shadow",
                "active_for_retrieval": True,
                "branches": [
                    {
                        "branch_id": "br-1",
                        "intent": "differential_diagnosis",
                        "framework": "problem_context_differential",
                        "branch_priority": 1,
                        "active_for_retrieval": True,
                        "query_families": [
                            {
                                "family_name": "high_recall",
                                "source": "pubmed",
                                "query_string": '"foo"[tiab] AND "bar"[tiab]',
                                "qa_checks_passed": True,
                            },
                            {
                                "family_name": "high_recall",
                                "source": "europe_pmc",
                                "query_string": 'TITLE_ABS:"foo" AND TITLE_ABS:"bar"',
                                "qa_checks_passed": True,
                            },
                        ],
                    }
                ],
            },
        },
        "picott_json": {
            "type_of_question": "background",
            "question_subtype": "disease_overview",
            "specialty": "psychiatry",
            "must_have_concepts": [{"concept": "foo"}],
            "ranking_concepts": [{"concept": "bar"}],
            "negative_or_penalty_concepts": [],
            "translated_question": "(deterministic translation)",
            "interpretation_for_user": "(deterministic interpretation)",
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [
                {
                    "text": "foo",
                    "canonical_concept": "foo",
                    "entity_type": "disease_or_problem",
                    "role": "population",
                },
                {
                    "text": "bar",
                    "canonical_concept": "bar",
                    "entity_type": "condition",
                    "role": "outcome",
                },
            ]
        },
        # SENTINEL values for the dead fields — must NEVER reach the
        # serializer output. The unit-test below asserts on this.
        "picott_payload": {"DEAD_FIELD_PICOTT_PAYLOAD": "MUST_NOT_APPEAR"},
        "queries_by_family": [
            {"DEAD_FIELD_QUERIES_BY_FAMILY": "MUST_NOT_APPEAR"}
        ],
    }


# ---------------------------------------------------------------------
# 1. Serializer unit tests
# ---------------------------------------------------------------------

def test_M3_1_1_serializer_populates_canonical_fields():
    sess = _realistic_session_doc()
    out = serialize_a_arm_from_session(sess)
    # CQM
    assert out["cqm"]["primary_intent"] == "differential_diagnosis"
    assert out["cqm"]["question_subtype"] == "disease_overview"
    assert out["cqm"]["specialty"] == "psychiatry"
    assert out["cqm"]["frameworks"] == ["problem_context_differential"]
    assert out["cqm"]["required_anchors"] == ["foo", "bar"]
    # Branch query plan
    bqp = out["branch_query_plan"]
    assert bqp["schema_version"] == "K2.1.0"
    assert bqp["mode"] == "shadow"
    assert bqp["active_for_retrieval"] is True
    assert len(bqp["branches"]) == 1
    fams = bqp["branches"][0]["query_families"]
    assert len(fams) == 2
    assert {f["source"] for f in fams} == {"pubmed", "europe_pmc"}
    # Flat list for legacy callers
    assert out["boolean_queries"] == sorted({
        '"foo"[tiab] AND "bar"[tiab]',
        'TITLE_ABS:"foo" AND TITLE_ABS:"bar"',
    })
    # PICOTT view
    assert out["picott_view"]["type_of_question"] == "background"
    assert out["picott_view"]["specialty"] == "psychiatry"
    # Anchors
    assert [a["concept"] for a in out["anchors"]["primary_anchors"]] == ["foo", "bar"]


def test_M3_1_1_serializer_does_not_reference_dead_fields_in_source():
    """Source-level guard: ``picott_payload`` / ``queries_by_family``
    must never be referenced in the serializer body."""
    src = inspect.getsource(serialize_a_arm_from_session)
    # Allow them in comments only (docstring already calls them out).
    assert ".get(\"picott_payload\")" not in src
    assert ".get(\"queries_by_family\")" not in src


def test_M3_1_1_serializer_ignores_dead_fields_at_runtime():
    """Even if the session has ``picott_payload`` and
    ``queries_by_family`` populated with sentinel garbage, the
    serializer output must contain none of that content."""
    sess = _realistic_session_doc()
    out = serialize_a_arm_from_session(sess)
    s = json.dumps(out, default=str)
    assert "DEAD_FIELD_PICOTT_PAYLOAD" not in s
    assert "DEAD_FIELD_QUERIES_BY_FAMILY" not in s
    assert "MUST_NOT_APPEAR" not in s


def test_M3_1_1_serializer_returns_safe_shape_on_empty_input():
    out = serialize_a_arm_from_session(None)
    assert out["cqm"]["primary_intent"] is None
    assert out["branch_query_plan"]["branches"] == []
    assert out["boolean_queries"] == []
    assert out["picott_view"]["type_of_question"] is None
    assert out["anchors"]["primary_anchors"] == []


# ---------------------------------------------------------------------
# 2. Shared-serializer parity test
# ---------------------------------------------------------------------

def test_M3_1_1_packet_and_admin_get_share_serializer():
    """Packet builder and admin GET path must produce byte-identical
    A-arm output for the same input session."""
    sess = _realistic_session_doc()
    packet_out = packet_a_arm_section(sess)
    admin_get_out = serialize_a_arm_from_session(sess)
    assert json.dumps(packet_out, sort_keys=True, default=str) == \
        json.dumps(admin_get_out, sort_keys=True, default=str)


# ---------------------------------------------------------------------
# 3. Lazy-resolve behavior at GET
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_M3_1_1_lazy_resolve_at_get_marks_live_resolved(client):  # noqa: F811
    db = get_db()
    sid = "sess-M3_1_1-lazy-live"
    rid = "run-M3_1_1-lazy-live"
    sess = _realistic_session_doc(sid)
    await db[Collections.SEARCH_SESSIONS].insert_one(sess)
    # Insert a benchmark run whose stored a_arm has the OLD broken
    # empty shape (simulating a pre-hotfix doc).
    await db[Collections.M3_1_BENCHMARK_RUNS].insert_one({
        "id": rid,
        "_id": rid,
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc),
        "search_session_id": sid,
        "eligibility_decision": "eligible",
        "route_owned_by_k4a_specialty": False,
        "timeout_status": "none",
        "fallback_status": "none",
        "a_arm": {
            "cqm": {"primary_intent": None, "frameworks": [],
                    "required_anchors": []},
            "boolean_queries": [],
            "top10": [
                {"pmid": "111", "title": "stored title", "journal": "J",
                 "year": "2025", "source": "pubmed"},
            ],
        },
        "b_arm": {"top10": [], "boolean_queries": {"pubmed": [], "epmc": []}},
        "metrics": {},
        "safety_flags": {"leak_scan_passed": True},
    })

    out = await get_benchmark_run_by_id(rid)
    assert out is not None
    assert out["a_arm_source"] == "live_resolved"
    assert out["a_arm"]["cqm"]["primary_intent"] == "differential_diagnosis"
    assert out["a_arm"]["cqm"]["required_anchors"] == ["foo", "bar"]
    # Branch query plan now populated.
    bqp = out["a_arm"]["branch_query_plan"]
    assert len(bqp["branches"]) == 1
    # Stored top10 preserved (we never re-run retrieval at GET).
    assert out["a_arm"]["top10"][0]["pmid"] == "111"
    # Legacy stored metric-input queries surfaced under an auditable key.
    assert out["a_arm"]["metric_input_queries"] == []


@pytest.mark.asyncio
async def test_M3_1_1_lazy_resolve_falls_back_to_stored_when_session_missing(client):  # noqa: F811
    db = get_db()
    rid = "run-M3_1_1-lazy-stored"
    await db[Collections.M3_1_BENCHMARK_RUNS].insert_one({
        "id": rid,
        "_id": rid,
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc),
        "search_session_id": "session-that-does-not-exist",
        "eligibility_decision": "eligible",
        "route_owned_by_k4a_specialty": False,
        "timeout_status": "none",
        "fallback_status": "none",
        "a_arm": {
            "cqm": {"primary_intent": "legacy_intent", "frameworks": [],
                    "required_anchors": []},
            "boolean_queries": ["legacy_q"],
            "top10": [],
        },
        "b_arm": {"top10": [], "boolean_queries": {"pubmed": [], "epmc": []}},
        "metrics": {},
        "safety_flags": {"leak_scan_passed": True},
    })
    out = await get_benchmark_run_by_id(rid)
    assert out is not None
    assert out["a_arm_source"] == "stored"
    # Stored content preserved verbatim
    assert out["a_arm"]["cqm"]["primary_intent"] == "legacy_intent"
    assert out["a_arm"]["boolean_queries"] == ["legacy_q"]


@pytest.mark.asyncio
async def test_M3_1_1_lazy_resolve_in_list_endpoint(client):  # noqa: F811
    db = get_db()
    sid = "sess-M3_1_1-list"
    rid = "run-M3_1_1-list"
    sess = _realistic_session_doc(sid)
    await db[Collections.SEARCH_SESSIONS].insert_one(sess)
    await db[Collections.M3_1_BENCHMARK_RUNS].insert_one({
        "id": rid,
        "_id": rid,
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc),
        "search_session_id": sid,
        "eligibility_decision": "eligible",
        "route_owned_by_k4a_specialty": False,
        "timeout_status": "none",
        "fallback_status": "none",
        "a_arm": {"cqm": {}, "boolean_queries": [], "top10": []},
        "b_arm": {"top10": [], "boolean_queries": {"pubmed": [], "epmc": []}},
        "metrics": {},
        "safety_flags": {"leak_scan_passed": True},
    })

    runs = await get_recent_benchmark_runs(limit=50)
    match = next((r for r in runs if r.get("id") == rid), None)
    assert match is not None, "newly inserted run must be in list"
    assert match["a_arm_source"] == "live_resolved"
    assert match["a_arm"]["cqm"]["primary_intent"] == "differential_diagnosis"


# ---------------------------------------------------------------------
# 4. Endpoint regression — non-empty A-arm via the admin HTTP route
# ---------------------------------------------------------------------

async def _admin_token(client):  # noqa: F811
    r = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_M3_1_1_endpoint_returns_non_empty_a_arm_after_hotfix(client):  # noqa: F811
    """Insert a synthetic session that already passes the eligibility
    flow (PHI-clean, semantic-repair candidate validator-accepted).
    POST /api/admin/m3_1/benchmark and assert the response carries the
    populated A-arm CQM + branch_query_plan from the real schema."""
    from core.config import settings

    db = get_db()
    sid = "sess-M3_1_1-endpoint"
    sess = _realistic_session_doc(sid)
    # Make the session look eligible to the M3.1 service.
    sess["status"] = "awaiting_selection"
    sess["original_question"] = "deterministic synthetic question"
    sess["sanitized_question"] = "deterministic synthetic question"
    sess["session_diagnostics_json"] = {
        "semantic_anchor_repair": {
            "semantic_repair_triggered": True,
            "semantic_repair_trigger_reasons": ["literal_full_phrase_anchor"],
            "semantic_repair_validation_status": "passed",
            "semantic_repair_validation_confidence": 0.99,
            "semantic_repair_candidate": {
                "primary_intent": "risk_etiology",
                "frameworks": ["risk_etiology"],
                "required_anchors": ["foo", "bar"],
                "slots": {},
                "semantic_repair_reason": "literal_full_phrase_anchor",
            },
            "semantic_repair_rejection_reasons": [],
        },
    }
    await db[Collections.SEARCH_SESSIONS].insert_one(sess)

    token = await _admin_token(client)
    with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True):
        # Patch B-arm retrieval so we don't make live PubMed / EPMC calls.
        with patch("services.m3_1_benchmark._b_arm_retrieve",
                   new=AsyncMock(return_value=([], {"pubmed": [], "epmc": []},
                                               {"pubmed": [], "epmc": []},
                                               "none", "none"))):
            r = await client.post(
                "/api/admin/m3_1/benchmark",
                headers={"Authorization": f"Bearer {token}"},
                json={"session_id": sid},
            )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["eligibility_decision"] == "eligible"
    a = body["a_arm"]
    # NON-empty CQM proves the hotfix landed.
    assert a["cqm"]["primary_intent"] == "differential_diagnosis"
    assert a["cqm"]["required_anchors"] == ["foo", "bar"]
    assert a["cqm"]["specialty"] == "psychiatry"
    # NON-empty branch_query_plan with 2 query families.
    bqp = a["branch_query_plan"]
    assert len(bqp["branches"]) == 1
    assert len(bqp["branches"][0]["query_families"]) == 2
    # Flat list legacy field populated.
    assert len(a["boolean_queries"]) == 2
    # Lazy-resolve marker.
    assert body["a_arm_source"] == "live_resolved"


# ---------------------------------------------------------------------
# 5. K4A protected routes — byte-identical decision after the hotfix
# ---------------------------------------------------------------------

@pytest.mark.parametrize("subtype,expected_decision", [
    ("severity_marker_overview", "route_owned_by_k4a_specialty_K4A_2"),
    ("rule_out_test_utility", "route_owned_by_k4a_specialty_K4A_3"),
    ("epidemiology_incidence_complication_risk", "route_owned_by_k4a_specialty_K4A_4"),
])
@pytest.mark.asyncio
async def test_M3_1_1_k4a_routes_still_blocked_after_hotfix(client, subtype, expected_decision):  # noqa: F811
    from core.config import settings

    db = get_db()
    sid = f"sess-M3_1_1-k4a-{subtype}"
    sess = _realistic_session_doc(sid)
    sess["status"] = "awaiting_selection"
    sess["original_question"] = "deterministic synthetic question"
    sess["sanitized_question"] = "deterministic synthetic question"
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
# 6. Leak scan — new fields must not introduce any leak shapes
# ---------------------------------------------------------------------

def test_M3_1_1_serializer_output_passes_leak_scan():
    sess = _realistic_session_doc()
    out = serialize_a_arm_from_session(sess)
    # leak_scan_passes returns True if no PHI / prompt / credential
    # shapes are present in any string node of the document.
    assert leak_scan_passes(out) is True


# ---------------------------------------------------------------------
# 7. Disabled-state guard unchanged
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_M3_1_1_endpoint_503_unchanged_when_flag_off(client):  # noqa: F811
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
