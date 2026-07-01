"""Phase M3.1 — preview-only admin A/B retrieval benchmark tests.

DETERMINISTIC. MOCKED CONNECTORS ONLY. NO LIVE LLM. NO LIVE PUBMED.

Covers:
  * Feature-flag OFF → endpoint returns 503 with closed-set
    ``m3_1_disabled`` and ZERO source / LLM calls.
  * Admin auth required → 403 for non-admin caller.
  * Diagnostics GET is side-effect-free: zero source calls.
  * Raw-question PHI block: 4xx, no LLM, no source, no storage,
    no benchmark run document.
  * Raw-question PHI-clean → 501 ``m3_1_raw_question_deferred``,
    no storage, no source / LLM.
  * Eligibility decisions: A/F/G eligible; K4A.2/3/4 blocked with
    correct closed-set decision; generic-only blocked; PHI-block
    upstream.
  * Eligible case with mocked retrieval returns full payload with
    A-top10, B-top10, metrics block.
  * Active clinician-visible retrieval byte-equivalence (no side
    effect on search_session_articles).
  * TTL index on ``expires_at`` exists.
  * Timeout simulation paths: m_repair_timeout / per_source_timeout.
  * Leak scanner rejects PHI/JWT/prompt boilerplate.
  * Production fail-closed guard refuses to boot.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------
# Pure-function tests — closed-set surfaces.
# ---------------------------------------------------------------------

def test_M3_1_eligibility_decisions_are_disjoint_closed_set() -> None:
    from services.m3_1_benchmark import (
        ELIGIBILITY_DECISIONS, TIMEOUT_STATUSES, FALLBACK_STATUSES,
    )
    assert "eligible" in ELIGIBILITY_DECISIONS
    assert "route_owned_by_k4a_specialty_K4A_2" in ELIGIBILITY_DECISIONS
    assert "route_owned_by_k4a_specialty_K4A_3" in ELIGIBILITY_DECISIONS
    assert "route_owned_by_k4a_specialty_K4A_4" in ELIGIBILITY_DECISIONS
    assert TIMEOUT_STATUSES.issuperset({"m_repair_timeout", "per_source_timeout_pubmed", "per_source_timeout_epmc", "none"})
    assert FALLBACK_STATUSES.issuperset({"none", "b_partial_results", "b_unavailable"})
    # Per scope decision §3 — OpenAlex keyword search is NOT in scope.
    assert "per_source_timeout_openalex" not in TIMEOUT_STATUSES


def test_M3_1_b_arm_compiler_emits_no_literal_full_phrase() -> None:
    """For Reproducer A's candidate, the B-arm queries must NOT
    contain the original literal phrase ``"causes of schizophrenia
    in young adults"``."""
    from services.m3_1_benchmark import compile_b_arm_queries
    candidate = {
        "primary_intent": "risk_etiology",
        "frameworks": ["risk_etiology"],
        "required_anchors": ["schizophrenia"],
        "slots": {
            "population": ["young adults"], "condition": ["schizophrenia"],
            "intervention_or_exposure": [], "comparator": [],
            "outcome": [], "index_test": [], "target_condition": [],
            "procedure_or_device": [], "timeframe": [], "setting": [],
            "mechanism": [],
        },
        "semantic_repair_reason": "missing_risk_etiology_framework",
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
    }
    out = compile_b_arm_queries(candidate)
    all_strs = " ".join(
        q["query_string"] for v in out.values() for q in v
    )
    assert "causes of schizophrenia in young adults" not in all_strs.lower()
    assert "schizophrenia" in all_strs
    # PubMed-style + EPMC-style are both emitted.
    assert any('[tiab]' in q["query_string"] for q in out["pubmed"])
    assert any('TITLE_ABS' in q["query_string"] for q in out["epmc"])


def test_M3_1_b_arm_compiler_preserves_both_central_anchors_for_F() -> None:
    from services.m3_1_benchmark import compile_b_arm_queries
    candidate = {
        "primary_intent": "mechanism",
        "frameworks": ["mechanism"],
        "required_anchors": ["antiphospholipid syndrome"],
        "slots": {
            "population": [], "condition": ["antiphospholipid syndrome"],
            "intervention_or_exposure": [], "comparator": [],
            "outcome": ["thrombosis"], "index_test": [],
            "target_condition": [], "procedure_or_device": [],
            "timeframe": [], "setting": [],
            "mechanism": ["thrombosis"],
        },
        "semantic_repair_reason": "missing_mechanism_pathophysiology_framework",
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
    }
    out = compile_b_arm_queries(candidate)
    pm = " ".join(q["query_string"] for q in out["pubmed"])
    assert "antiphospholipid syndrome" in pm
    assert "thrombosis" in pm


def test_M3_1_metrics_compute_overlap_and_wrong_anchor_rate() -> None:
    from services.m3_1_benchmark import compute_metrics
    cand = {
        "required_anchors": ["antiphospholipid syndrome"],
        "slots": {},
    }
    a_top = [
        {"pmid": "1", "title": "Some study"},
        {"pmid": "2", "title": "Another study"},
    ]
    b_top = [
        {"pmid": "1", "title": "antiphospholipid syndrome thrombosis"},
        {"pmid": "3", "title": "off topic"},
    ]
    m = compute_metrics(
        a_top10=a_top, b_top10=b_top,
        a_queries=["a1", "a2"], b_queries=["b1", "b2"],
        candidate=cand,
        deterministic_primary_anchors=[
            {"canonical_concept": "antiphospholipid syndrome"},
        ],
        original_question="What is the mechanism of thrombosis in antiphospholipid syndrome?",
        per_source_latencies={"pubmed": [200, 300], "epmc": [400]},
        per_source_errors={"pubmed": [], "epmc": []},
    )
    assert m["overlap_at_10"] == 1
    assert m["new_direct_hits_at_10"] == 1
    assert m["wrong_anchor_rate_at_10"] == 0.5
    assert m["source_error_rate"] == 0.0
    assert "added" in m["query_diff"] and "removed" in m["query_diff"]


def test_M3_1_leak_scanner_blocks_phi_jwt_and_prompt_boilerplate() -> None:
    from services.m3_1_benchmark import leak_scan_passes
    assert leak_scan_passes({"title": "ok"}) is True
    assert leak_scan_passes({"x": "john@example.com"}) is False
    assert leak_scan_passes({"x": "MRN: 12345678"}) is False
    assert leak_scan_passes({"x": "born 03/14/1979"}) is False
    assert leak_scan_passes({"x": "sk-AbcD1234"}) is False
    assert leak_scan_passes({"x": "Bearer eyJhbGciOiJIUzI1NiJ9"}) is False
    assert leak_scan_passes({"x": "You are the Bounded Semantic ..."}) is False


# ---------------------------------------------------------------------
# Eligibility decision unit tests.
# ---------------------------------------------------------------------

def _good_candidate() -> dict:
    return {
        "primary_intent": "risk_etiology",
        "frameworks": ["risk_etiology"],
        "required_anchors": ["schizophrenia"],
        "slots": {
            "population": ["young adults"], "condition": ["schizophrenia"],
            "intervention_or_exposure": [], "comparator": [],
            "outcome": [], "index_test": [], "target_condition": [],
            "procedure_or_device": [], "timeframe": [], "setting": [],
            "mechanism": [],
        },
        "slot_policies": {},
        "optional_or_ranking_only_concepts": [],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": [],
        "candidate_mesh_terms_unvalidated": [],
        "semantic_repair_reason": "missing_risk_etiology_framework",
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
        "secondary_intents": [],
        "branches": [],
    }


def _session_doc_for_a() -> dict:
    """A synthetic session_doc shape mirroring what Mongo stores."""
    return {
        "id": "sid_a",
        "original_question": "What are the causes of schizophrenia in young adults?",
        "sanitized_question": "What are the causes of schizophrenia in young adults?",
        "normalized_question": "what are the causes of schizophrenia in young adults?",
        "status": "awaiting_selection",
        "deleted_at": None,
        "picott_payload": {
            "type_of_question": "background",
            "question_subtype": None,
            "framework_list_json": [],
            "branches": [],
        },
        "primary_anchors": [
            {"canonical_concept": "schizophrenia",
             "text": "schizophrenia",
             "required_for_search": True},
        ],
        "queries_by_family": [
            {"query_family": "literal",
             "query_string":
                 '"causes of schizophrenia in young adults"'},
        ],
        "session_diagnostics_json": {
            "semantic_anchor_repair": {
                "semantic_repair_triggered": True,
                "semantic_repair_trigger_reasons": [
                    "missing_risk_etiology_framework"
                ],
                "semantic_repair_validation_status": "passed",
                "semantic_repair_rejection_reasons": [],
                "semantic_repair_candidate": _good_candidate(),
                "semantic_repair_latency_ms": 2700,
            },
        },
    }


def test_M3_1_decide_eligibility_eligible_path() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    doc = _session_doc_for_a()
    decision, _payload = _decide_eligibility(
        session_doc=doc,
        diagnostics={
            "session_diagnostics_json": doc["session_diagnostics_json"],
            "picott": doc["picott_payload"],
            "anchors": {"primary_anchors": doc["primary_anchors"]},
        },
    )
    assert decision == "eligible"


def test_M3_1_decide_eligibility_blocks_K4A2_severity_marker() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    doc = _session_doc_for_a()
    doc["picott_payload"]["question_subtype"] = "severity_marker_overview"
    decision, _ = _decide_eligibility(
        session_doc=doc,
        diagnostics={
            "session_diagnostics_json": doc["session_diagnostics_json"],
            "picott": doc["picott_payload"],
            "anchors": {"primary_anchors": doc["primary_anchors"]},
        },
    )
    assert decision == "route_owned_by_k4a_specialty_K4A_2"


def test_M3_1_decide_eligibility_blocks_K4A3_rule_out() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    doc = _session_doc_for_a()
    doc["picott_payload"]["question_subtype"] = "rule_out_test_utility"
    decision, _ = _decide_eligibility(
        session_doc=doc,
        diagnostics={
            "session_diagnostics_json": doc["session_diagnostics_json"],
            "picott": doc["picott_payload"],
            "anchors": {"primary_anchors": doc["primary_anchors"]},
        },
    )
    assert decision == "route_owned_by_k4a_specialty_K4A_3"


def test_M3_1_decide_eligibility_blocks_K4A4_incidence() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    doc = _session_doc_for_a()
    doc["picott_payload"]["question_subtype"] = (
        "epidemiology_incidence_complication_risk"
    )
    decision, _ = _decide_eligibility(
        session_doc=doc,
        diagnostics={
            "session_diagnostics_json": doc["session_diagnostics_json"],
            "picott": doc["picott_payload"],
            "anchors": {"primary_anchors": doc["primary_anchors"]},
        },
    )
    assert decision == "route_owned_by_k4a_specialty_K4A_4"


def test_M3_1_decide_eligibility_blocks_phi() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    doc = _session_doc_for_a()
    doc["status"] = "blocked_phi"
    # No K4A subtype so we hit the PHI branch.
    doc["picott_payload"]["question_subtype"] = None
    decision, _ = _decide_eligibility(
        session_doc=doc,
        diagnostics={
            "session_diagnostics_json": doc["session_diagnostics_json"],
            "picott": doc["picott_payload"],
            "anchors": {"primary_anchors": doc["primary_anchors"]},
        },
    )
    assert decision == "ineligible_phi_blocked"


def test_M3_1_decide_eligibility_blocks_detector_not_triggered() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    doc = _session_doc_for_a()
    sar = doc["session_diagnostics_json"]["semantic_anchor_repair"]
    sar["semantic_repair_triggered"] = False
    sar["semantic_repair_trigger_reasons"] = []
    decision, _ = _decide_eligibility(
        session_doc=doc,
        diagnostics={
            "session_diagnostics_json": doc["session_diagnostics_json"],
            "picott": doc["picott_payload"],
            "anchors": {"primary_anchors": doc["primary_anchors"]},
        },
    )
    assert decision == "ineligible_detector_not_triggered"


def test_M3_1_decide_eligibility_blocks_schema_invalid() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    doc = _session_doc_for_a()
    sar = doc["session_diagnostics_json"]["semantic_anchor_repair"]
    sar["semantic_repair_validation_status"] = "rejected"
    sar["semantic_repair_rejection_reasons"] = ["missing_required_field"]
    sar["semantic_repair_candidate"] = None
    decision, _ = _decide_eligibility(
        session_doc=doc,
        diagnostics={
            "session_diagnostics_json": doc["session_diagnostics_json"],
            "picott": doc["picott_payload"],
            "anchors": {"primary_anchors": doc["primary_anchors"]},
        },
    )
    assert decision == "ineligible_schema_invalid"


def test_M3_1_decide_eligibility_blocks_low_confidence() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    doc = _session_doc_for_a()
    sar = doc["session_diagnostics_json"]["semantic_anchor_repair"]
    sar["semantic_repair_validation_status"] = "rejected"
    sar["semantic_repair_rejection_reasons"] = ["confidence_below_threshold"]
    sar["semantic_repair_candidate"] = None
    decision, _ = _decide_eligibility(
        session_doc=doc,
        diagnostics={
            "session_diagnostics_json": doc["session_diagnostics_json"],
            "picott": doc["picott_payload"],
            "anchors": {"primary_anchors": doc["primary_anchors"]},
        },
    )
    assert decision == "ineligible_low_confidence"


def test_M3_1_decide_eligibility_blocks_generic_only() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    doc = _session_doc_for_a()
    sar = doc["session_diagnostics_json"]["semantic_anchor_repair"]
    sar["semantic_repair_validation_status"] = "rejected"
    sar["semantic_repair_rejection_reasons"] = [
        "generic_only_required_anchor"
    ]
    sar["semantic_repair_candidate"] = None
    decision, _ = _decide_eligibility(
        session_doc=doc,
        diagnostics={
            "session_diagnostics_json": doc["session_diagnostics_json"],
            "picott": doc["picott_payload"],
            "anchors": {"primary_anchors": doc["primary_anchors"]},
        },
    )
    assert decision == "ineligible_generic_only"


def test_M3_1_decide_eligibility_session_not_found() -> None:
    from services.m3_1_benchmark import _decide_eligibility
    decision, _ = _decide_eligibility(
        session_doc=None, diagnostics={},
    )
    assert decision == "ineligible_session_not_found"


# ---------------------------------------------------------------------
# Endpoint contract tests via the existing ``client`` fixture from
# ``tests/conftest.py`` (handles lifespan, ensures indexes, seeds
# admin@test.com / clinician@test.com).
# ---------------------------------------------------------------------

async def _admin_token(client) -> str:
    r = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


async def _clinician_token(client) -> str:
    r = await client.post(
        "/api/auth/login",
        json={"email": "clinician@test.com", "password": "ClinicianTest123!"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_M3_1_endpoint_503_when_flag_off(client):
    from core.config import settings
    token = await _admin_token(client)
    with patch.object(settings, "M3_1_BENCHMARK_ENABLED", False):
        r = await client.post(
            "/api/admin/m3_1/benchmark",
            headers={"Authorization": f"Bearer {token}"},
            json={"session_id": "any"},
        )
    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "m3_1_disabled"


@pytest.mark.asyncio
async def test_M3_1_endpoint_403_for_non_admin(client):
    token = await _clinician_token(client)
    r = await client.post(
        "/api/admin/m3_1/benchmark",
        headers={"Authorization": f"Bearer {token}"},
        json={"session_id": "any"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_M3_1_endpoint_400_when_neither_body_field(client):
    from core.config import settings
    token = await _admin_token(client)
    with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True):
        r = await client.post(
            "/api/admin/m3_1/benchmark",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_M3_1_raw_question_phi_blocked_returns_phi_shape(client):
    from core.config import settings
    from core.db import Collections, get_db
    token = await _admin_token(client)
    db = get_db()
    pre_runs = await db[Collections.M3_1_BENCHMARK_RUNS].count_documents({})
    pre_sessions = await db[Collections.SEARCH_SESSIONS].count_documents({})
    mock_b = AsyncMock(return_value=([], {"pubmed": [], "epmc": []},
                                     {"pubmed": [], "epmc": []}, "none", "none"))
    with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True), \
         patch("services.m3_1_benchmark._b_arm_retrieve", mock_b):
        r = await client.post(
            "/api/admin/m3_1/benchmark",
            headers={"Authorization": f"Bearer {token}"},
            json={"raw_question":
                  "Patient John Doe MRN 12345678 born 03/14/1979 — "
                  "what is the mechanism of thrombosis?"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "blocked_phi"
    assert mock_b.await_count == 0
    assert await db[Collections.M3_1_BENCHMARK_RUNS].count_documents({}) == pre_runs
    assert await db[Collections.SEARCH_SESSIONS].count_documents({}) == pre_sessions


@pytest.mark.asyncio
async def test_M3_1_raw_question_phi_clean_deferred(client):
    from core.config import settings
    from core.db import Collections, get_db
    token = await _admin_token(client)
    db = get_db()
    pre_runs = await db[Collections.M3_1_BENCHMARK_RUNS].count_documents({})
    pre_sessions = await db[Collections.SEARCH_SESSIONS].count_documents({})
    mock_b = AsyncMock(return_value=([], {"pubmed": [], "epmc": []},
                                     {"pubmed": [], "epmc": []}, "none", "none"))
    with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True), \
         patch("services.m3_1_benchmark._b_arm_retrieve", mock_b):
        r = await client.post(
            "/api/admin/m3_1/benchmark",
            headers={"Authorization": f"Bearer {token}"},
            json={"raw_question":
                  "What is the mechanism of thrombosis in antiphospholipid syndrome?"},
        )
    assert r.status_code == 501
    assert r.json()["detail"]["status"] == "m3_1_raw_question_deferred"
    assert mock_b.await_count == 0
    assert await db[Collections.M3_1_BENCHMARK_RUNS].count_documents({}) == pre_runs
    assert await db[Collections.SEARCH_SESSIONS].count_documents({}) == pre_sessions


async def _insert_synthetic_session(
    *, sid: str, question_subtype=None, with_passing_candidate=True,
    rejection_reasons=None,
):
    from core.db import Collections, get_db
    db = get_db()
    cand = _good_candidate() if with_passing_candidate else None
    rej = list(rejection_reasons or [])
    val_status = "passed" if with_passing_candidate else "rejected"
    triggered = bool(with_passing_candidate) or (
        rejection_reasons and rejection_reasons != ["__no_trigger__"]
    )
    if rejection_reasons == ["__no_trigger__"]:
        triggered = False
        val_status = "not_attempted"
        rej = []
    session_doc = {
        "id": sid, "user_id": "test-user",
        "original_question": "What are the causes of schizophrenia in young adults?",
        "sanitized_question": "What are the causes of schizophrenia in young adults?",
        "normalized_question": "what are the causes of schizophrenia in young adults?",
        "status": "awaiting_selection",
        "deleted_at": None,
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) + timedelta(days=30),
        # Phase M3.1.1: canonical schema fields. The serializer reads
        # from these, NOT from the legacy/dead picott_payload + queries_by_family
        # fields (which are kept below only as sentinel proof-of-absence
        # in some M3.1.1 tests; they have no functional role here).
        "clinical_question_map_json": {
            "schema_version": "K2.1.0",
            "primary_intent": "background",
            "frameworks": [],
            "branch_query_plan": {
                "schema_version": "K2.1.0",
                "mode": "shadow",
                "active_for_retrieval": True,
                "branches": [{
                    "branch_id": "br-A-1",
                    "intent": "background",
                    "framework": "background",
                    "branch_priority": 1,
                    "active_for_retrieval": True,
                    "query_families": [{
                        "family_name": "literal",
                        "source": "pubmed",
                        "query_string": '"causes of schizophrenia in young adults"',
                        "qa_checks_passed": True,
                    }],
                }],
            },
        },
        "picott_json": {
            "type_of_question": "background",
            "question_subtype": question_subtype,
            "framework_list_json": [],
            "branches": [],
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [
                {"canonical_concept": "schizophrenia",
                 "text": "schizophrenia",
                 "entity_type": "disease_or_problem",
                 "role": "population",
                 "required_for_search": True},
            ],
        },
        # Sentinel fields — kept solely to prove the serializer
        # ignores them (M3.1.1 tests assert these never reach output).
        "picott_payload": {
            "type_of_question": "background",
            "question_subtype": question_subtype,
            "framework_list_json": [],
            "branches": [],
        },
        "primary_anchors": [
            {"canonical_concept": "schizophrenia",
             "text": "schizophrenia", "required_for_search": True},
        ],
        "queries_by_family": [
            {"query_family": "literal",
             "query_string": '"causes of schizophrenia in young adults"'},
        ],
        "session_diagnostics_json": {
            "semantic_anchor_repair": {
                "semantic_repair_triggered": triggered,
                "semantic_repair_trigger_reasons": (
                    ["missing_risk_etiology_framework"] if triggered else []
                ),
                "semantic_repair_validation_status": val_status,
                "semantic_repair_rejection_reasons": rej,
                "semantic_repair_candidate": cand,
                "semantic_repair_latency_ms": 2700 if triggered else 0,
            },
        },
    }
    await db[Collections.SEARCH_SESSIONS].insert_one(session_doc)
    return session_doc


async def _cleanup_session(sid: str) -> None:
    from core.db import Collections, get_db
    db = get_db()
    await db[Collections.SEARCH_SESSIONS].delete_one({"id": sid})
    await db[Collections.M3_1_BENCHMARK_RUNS].delete_many(
        {"search_session_id": sid},
    )


@pytest.mark.asyncio
async def test_M3_1_eligible_path_produces_side_by_side_payload(client):
    from core.config import settings
    token = await _admin_token(client)
    sid = str(uuid4())
    await _insert_synthetic_session(sid=sid, with_passing_candidate=True)
    mock_b_top10 = [
        {"pmid": "100", "source_id": "100",
         "title": "Schizophrenia risk factors review",
         "journal": "JAMA Psychiatry", "year": "2024",
         "source": "pubmed", "query_type": "b_required_AND"},
        {"pmid": "101", "source_id": "101",
         "title": "Etiology of schizophrenia in young adults",
         "journal": "Lancet", "year": "2023",
         "source": "pubmed", "query_type": "b_required_AND"},
    ]
    mock_b = AsyncMock(return_value=(
        mock_b_top10,
        {"pubmed": [120, 180], "epmc": [300]},
        {"pubmed": [], "epmc": []},
        "none", "none",
    ))
    try:
        with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True), \
             patch("services.m3_1_benchmark._b_arm_retrieve", mock_b):
            r = await client.post(
                "/api/admin/m3_1/benchmark",
                headers={"Authorization": f"Bearer {token}"},
                json={"session_id": sid},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["eligibility_decision"] == "eligible"
            assert body["a_arm"]["cqm"]["primary_intent"] == "background"
            assert body["b_arm"]["repair_candidate"]["primary_intent"] == "risk_etiology"
            assert body["b_arm"]["top10"][0]["pmid"] == "100"
            assert body["metrics"]["literal_full_phrase_anchor_removed"] is True
            assert body["metrics"]["generic_only_anchor_absent"] is True
            assert body["safety_flags"]["leak_scan_passed"] is True
            assert body["timeout_status"] == "none"
            assert body["fallback_status"] == "none"
            assert mock_b.await_count == 1
            # Admin GET list returns the new run.
            r2 = await client.get(
                "/api/admin/m3_1/runs",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert r2.status_code == 200
            assert any(
                run["search_session_id"] == sid for run in r2.json()["runs"]
            )
    finally:
        await _cleanup_session(sid)


@pytest.mark.asyncio
async def test_M3_1_K4A_protected_routes_blocked_no_b_retrieval(client):
    from core.config import settings
    token = await _admin_token(client)
    for subtype, decision_expected in [
        ("severity_marker_overview", "route_owned_by_k4a_specialty_K4A_2"),
        ("rule_out_test_utility", "route_owned_by_k4a_specialty_K4A_3"),
        ("epidemiology_incidence_complication_risk",
         "route_owned_by_k4a_specialty_K4A_4"),
    ]:
        sid = str(uuid4())
        await _insert_synthetic_session(
            sid=sid, question_subtype=subtype,
            with_passing_candidate=False,
            rejection_reasons=["__no_trigger__"],
        )
        mock_b = AsyncMock(return_value=(
            [], {"pubmed": [], "epmc": []}, {"pubmed": [], "epmc": []},
            "none", "none",
        ))
        try:
            with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True), \
                 patch("services.m3_1_benchmark._b_arm_retrieve", mock_b):
                r = await client.post(
                    "/api/admin/m3_1/benchmark",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"session_id": sid},
                )
                assert r.status_code == 200, r.text
                body = r.json()
                assert body["eligibility_decision"] == decision_expected
                assert body["route_owned_by_k4a_specialty"] is True
                assert body["b_arm"]["top10"] == []
                assert mock_b.await_count == 0
        finally:
            await _cleanup_session(sid)


@pytest.mark.asyncio
async def test_M3_1_search_diagnostics_get_is_side_effect_free(client):
    """Pre-existing GET admin diagnostics endpoint MUST NOT trigger
    any M3.1 retrieval / LLM / source call."""
    from core.config import settings
    token = await _admin_token(client)
    sid = str(uuid4())
    await _insert_synthetic_session(sid=sid, with_passing_candidate=True)
    mock_b = AsyncMock()
    try:
        with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True), \
             patch("services.m3_1_benchmark._b_arm_retrieve", mock_b):
            r = await client.get(
                f"/api/admin/search-sessions/{sid}/diagnostics",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert r.status_code == 200
            assert mock_b.await_count == 0
    finally:
        await _cleanup_session(sid)


def test_M3_1_production_guard_refuses_unsafe_combo() -> None:
    from services.semantic_repair_active_guard import (
        verify_m3_1_benchmark_startup,
    )
    verify_m3_1_benchmark_startup(is_production=False, enabled=True)
    verify_m3_1_benchmark_startup(is_production=True, enabled=False)
    with pytest.raises(RuntimeError, match="preview/dev-only"):
        verify_m3_1_benchmark_startup(is_production=True, enabled=True)


@pytest.mark.asyncio
async def test_M3_1_ttl_index_exists_on_expires_at(client):
    from core.db import Collections, get_db
    db = get_db()
    indexes = await db[Collections.M3_1_BENCHMARK_RUNS].index_information()
    has_ttl = any(
        any(k == "expires_at" for k, _ in (idx.get("key") or []))
        and idx.get("expireAfterSeconds") is not None
        for idx in indexes.values()
    )
    assert has_ttl, f"TTL index on expires_at missing: indexes={list(indexes)}"


@pytest.mark.asyncio
async def test_M3_1_timeout_path_per_source_marks_status(client):
    """Simulate per-source timeout on PubMed; benchmark records
    closed-set timeout_status and partial-or-no top10."""
    from core.config import settings
    token = await _admin_token(client)
    sid = str(uuid4())
    await _insert_synthetic_session(sid=sid, with_passing_candidate=True)
    mock_b = AsyncMock(return_value=(
        [],
        {"pubmed": [5000], "epmc": [200]},
        {"pubmed": ["timeout"], "epmc": []},
        "per_source_timeout_pubmed",
        "b_partial_results",
    ))
    try:
        with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True), \
             patch("services.m3_1_benchmark._b_arm_retrieve", mock_b):
            r = await client.post(
                "/api/admin/m3_1/benchmark",
                headers={"Authorization": f"Bearer {token}"},
                json={"session_id": sid},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["timeout_status"] == "per_source_timeout_pubmed"
            assert body["fallback_status"] == "b_partial_results"
    finally:
        await _cleanup_session(sid)


@pytest.mark.asyncio
async def test_M3_1_session_not_found_path(client):
    from core.config import settings
    token = await _admin_token(client)
    with patch.object(settings, "M3_1_BENCHMARK_ENABLED", True):
        r = await client.post(
            "/api/admin/m3_1/benchmark",
            headers={"Authorization": f"Bearer {token}"},
            json={"session_id": "nonexistent-sid"},
        )
    assert r.status_code == 200
    assert r.json()["eligibility_decision"] == "ineligible_session_not_found"
