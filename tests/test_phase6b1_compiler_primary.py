"""Phase 6B.1 — BooleanQueryCompiler is the PRIMARY query producer.

Covers:
  - compiler queries run FIRST (assert created_at order in pipeline trace)
  - legacy planner does NOT run when compiler valid + retrieval sufficient
  - legacy planner DOES run when compiler returns zero queries for a source
  - legacy planner DOES run when high-precision retrieval returns zero
  - planner_source / is_primary_strategy / is_fallback_strategy persist
  - planner_strategy summary lands on the session evaluator_json
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


# ---------- canned Europe PMC record used as retrieval payload ----------
_EPMC_HP_RECORD = {
    "id": "12345678",
    "source": "MED",
    "title": "RCT of empagliflozin vs placebo in HFpEF",
    "abstractText": (
        "Multicenter randomized controlled trial comparing empagliflozin "
        "and placebo in patients with heart failure with preserved ejection "
        "fraction (HFpEF). Outcomes: composite cardiovascular death + "
        "heart-failure hospitalization."
    ),
    "journalInfo": {"journal": {"title": "NEJM"}, "yearOfPublication": "2021"},
    "pubYear": "2021",
    "firstPublicationDate": "2021-08-27",
    "doi": "10.1056/NEJMoa2107038",
    "pmid": "12345678",
    "pmcid": "PMC8500551",
    "pubTypeList": {"pubType": ["Journal Article",
                                  "Randomized Controlled Trial"]},
    "authorList": {"author": [{"fullName": "Anker SD", "lastName": "Anker"}]},
    "fullTextUrlList": {"fullTextUrl": [
        {"url": "https://www.nejm.org/doi/full/10.1056/NEJMoa2107038",
         "availabilityCode": "OA"}
    ]},
    "inEPMC": "Y",
}

_EPMC_HR_RECORD = {
    "id": "99999999",
    "source": "MED",
    "title": "Broad recall hit",
    "abstractText": "Generic abstract",
    "journalInfo": {"journal": {"title": "Lancet"}, "yearOfPublication": "2022"},
    "pubYear": "2022",
    "pubTypeList": {"pubType": ["Journal Article"]},
    "authorList": {"author": []},
    "fullTextUrlList": {"fullTextUrl": []},
}


async def _signup(client, email: str) -> dict:
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!",
              "name": email.split("@")[0]},
    )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _wait_ready(client, sid: str, headers: dict, max_iter: int = 200):
    for _ in range(max_iter):
        await asyncio.sleep(0.2)
        sr = await client.get(
            f"/api/search-sessions/{sid}/status", headers=headers,
        )
        if sr.json().get("articles_ready") or sr.json().get("status") == "error":
            return sr.json()
    return sr.json()  # noqa: F821


async def _disable_pubmed():
    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": False}},
    )


async def _restore_pubmed():
    """Re-enable pubmed so subsequent tests that rely on the default
    enabled state are not broken."""
    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": True}},
    )


import pytest_asyncio  # noqa: E402


@pytest_asyncio.fixture(autouse=True)
async def _pubmed_state_guard():
    """Snapshot pubmed enabled state, then restore after each test."""
    yield
    await _restore_pubmed()


# ---------------------------------------------------------------------------
# Test 1 — Compiler runs FIRST + legacy does NOT run when retrieval is OK.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_compiler_primary_and_no_legacy_when_hp_sufficient(client):
    """Mock retrieval so the compiler's ``cb_high_precision`` family
    returns a hit. Legacy planner MUST NOT run."""
    await _disable_pubmed()
    headers = await _signup(client, "p6b1a@test.com")

    async def fake_retrieve(source, queries, **kwargs):
        # Return one hit attributed to a compiler family so the
        # high-precision-hit check counts it. Tag qtype with the FIRST
        # compiler plan we received so the pipeline sees a primary hit.
        if not queries:
            return []
        compiler_qtypes = [
            q["query_type"] for q in queries
            if q["query_type"].startswith("cb_high_precision")
        ]
        qt = compiler_qtypes[0] if compiler_qtypes else queries[0]["query_type"]
        return [(qt, _EPMC_HP_RECORD)]

    with patch("services.pipeline.literature_retrieval.retrieve",
                new=fake_retrieve):
        r = await client.post(
            "/api/search-sessions/",
            json={"question": "Is empagliflozin effective in HFpEF?"},
            headers=headers,
        )
        assert r.status_code == 200
        sid = r.json()["id"]
        await _wait_ready(client, sid, headers)

    trans = (await client.get(
        f"/api/search-sessions/{sid}/transparency", headers=headers,
    )).json()
    queries = trans["queries"]
    # Sort by created_at to assert compiler-first ordering.
    sorted_qs = sorted(queries, key=lambda q: q["created_at"])
    primary = [q for q in sorted_qs
                if q.get("planner_source") == "boolean_compiler"]
    legacy = [q for q in sorted_qs
                if q.get("planner_source") == "legacy_planner"]
    assert primary, "expected at least one compiler-emitted query"
    assert legacy == [], \
        f"legacy planner unexpectedly ran (compiler should have been sufficient): {len(legacy)}"
    # All primary queries persisted with is_primary_strategy=True
    assert all(q.get("is_primary_strategy") is True for q in primary)
    assert all(q.get("is_fallback_strategy") is False for q in primary)
    assert all(q.get("planner_source") == "boolean_compiler" for q in primary)
    # planner_strategy summary lands on evaluator_json
    sess = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    ps = (sess.get("evaluator_json") or {}).get("planner_strategy")
    assert ps, f"planner_strategy summary missing: {ps}"
    assert ps["primary_planner"] == "boolean_compiler"
    assert ps["legacy_planner_invoked"] is False
    assert ps["legacy_planner_sources"] == []


# ---------------------------------------------------------------------------
# Test 2 — Legacy planner DOES run when high-precision retrieval returns zero.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_legacy_fires_when_high_precision_returns_zero(client):
    """Mock retrieval so it returns hits attributed only to non-
    high-precision compiler families (e.g. ``cb_broad_recall``). The
    pipeline must trip criterion (b) high_precision_returned_zero and
    invoke the legacy planner as fallback."""
    await _disable_pubmed()
    headers = await _signup(client, "p6b1b@test.com")

    async def fake_retrieve(source, queries, **kwargs):
        # Only return hits attributed to broad_recall — no cb_high_*,
        # no cb_mesh_keyword, no cb_clinical_queries. This forces the
        # high_precision_returned_zero fallback.
        return [("cb_broad_recall", _EPMC_HR_RECORD)]

    with patch("services.pipeline.literature_retrieval.retrieve",
                new=fake_retrieve):
        r = await client.post(
            "/api/search-sessions/",
            json={"question": "Is empagliflozin effective in HFpEF?"},
            headers=headers,
        )
        assert r.status_code == 200
        sid = r.json()["id"]
        await _wait_ready(client, sid, headers)

    trans = (await client.get(
        f"/api/search-sessions/{sid}/transparency", headers=headers,
    )).json()
    queries = trans["queries"]
    sorted_qs = sorted(queries, key=lambda q: q["created_at"])
    primary = [q for q in sorted_qs
                if q.get("planner_source") == "boolean_compiler"]
    legacy = [q for q in sorted_qs
                if q.get("planner_source") == "legacy_planner"]
    assert primary, "expected compiler queries to be emitted first"
    assert legacy, \
        "legacy planner should have fired when cb_high_precision returned zero"
    # Compiler-first ordering: every compiler query has created_at <= every
    # legacy query.
    last_primary_ts = max(q["created_at"] for q in primary)
    first_legacy_ts = min(q["created_at"] for q in legacy)
    assert last_primary_ts <= first_legacy_ts, \
        f"compiler not first: last_primary={last_primary_ts} first_legacy={first_legacy_ts}"
    # Legacy rows tagged correctly.
    assert all(q.get("is_primary_strategy") is False for q in legacy)
    assert all(q.get("is_fallback_strategy") is True for q in legacy)
    fb_reasons = {q.get("fallback_reason") for q in legacy}
    assert "high_precision_returned_zero" in fb_reasons, fb_reasons
    # Evaluator summary reflects the fallback.
    sess = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    ps = (sess.get("evaluator_json") or {}).get("planner_strategy")
    assert ps["legacy_planner_invoked"] is True
    assert "europe_pmc" in ps["legacy_planner_sources"]
    assert "high_precision_returned_zero" in (
        ps["fallback_reasons_by_source"].get("europe_pmc") or []
    )


# ---------------------------------------------------------------------------
# Test 3 — Legacy planner DOES run when compiler emits zero queries.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_legacy_fires_when_compiler_yields_zero_queries(client):
    """Force the compiler to emit zero queries by stubbing both compiler
    services to return []. Pipeline must fall back to the legacy
    planner with fallback_reason=compiler_yielded_zero_queries."""
    await _disable_pubmed()
    headers = await _signup(client, "p6b1c@test.com")

    async def fake_retrieve(source, queries, **kwargs):
        # Legacy planner emits at most high_recall etc. — just return
        # one record so the pipeline completes.
        return [("high_recall", _EPMC_HR_RECORD)]

    # Patch the compiler + concept-block builder to emit zero queries.
    with patch("services.pipeline.literature_retrieval.retrieve",
                new=fake_retrieve), \
         patch("services.pipeline.concept_block_query_builder.build",
                return_value=[]), \
         patch("services.pipeline.boolean_query_compiler.compile",
                return_value=[]):
        r = await client.post(
            "/api/search-sessions/",
            json={"question": "Is empagliflozin effective in HFpEF?"},
            headers=headers,
        )
        assert r.status_code == 200
        sid = r.json()["id"]
        await _wait_ready(client, sid, headers)

    trans = (await client.get(
        f"/api/search-sessions/{sid}/transparency", headers=headers,
    )).json()
    queries = trans["queries"]
    primary = [q for q in queries
                if q.get("planner_source") == "boolean_compiler"]
    legacy = [q for q in queries
                if q.get("planner_source") == "legacy_planner"]
    assert primary == [], \
        f"expected 0 primary compiler queries when compiler is stubbed empty, got {len(primary)}"
    assert legacy, "expected legacy planner to fire as fallback"
    fb_reasons = {q.get("fallback_reason") for q in legacy}
    assert "compiler_yielded_zero_queries" in fb_reasons, fb_reasons
    sess = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    ps = (sess.get("evaluator_json") or {}).get("planner_strategy")
    assert ps["legacy_planner_invoked"] is True
    assert "europe_pmc" in ps["legacy_planner_sources"]
    assert "compiler_yielded_zero_queries" in (
        ps["fallback_reasons_by_source"].get("europe_pmc") or []
    )


# ---------------------------------------------------------------------------
# Test 4 — Admin /search-sessions/{id}/planner-counts returns the breakdown.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_admin_planner_counts_endpoint(client):
    """Admin endpoint returns planner_source counts per session."""
    # Run a session that DOES NOT trigger fallback (test 1 path).
    await _disable_pubmed()
    headers = await _signup(client, "p6b1d@test.com")
    async def fake_retrieve(source, queries, **kwargs):
        if not queries:
            return []
        compiler_qtypes = [
            q["query_type"] for q in queries
            if q["query_type"].startswith("cb_high_precision")
        ]
        qt = compiler_qtypes[0] if compiler_qtypes else queries[0]["query_type"]
        return [(qt, _EPMC_HP_RECORD)]
    with patch("services.pipeline.literature_retrieval.retrieve",
                new=fake_retrieve):
        r = await client.post(
            "/api/search-sessions/",
            json={"question": "Is empagliflozin effective in HFpEF?"},
            headers=headers,
        )
        sid = r.json()["id"]
        await _wait_ready(client, sid, headers)

    # Promote a new user to admin to exercise the endpoint.
    admin_signup = await client.post(
        "/api/auth/signup",
        json={"email": "p6b1_admin@test.com", "password": "Pass1234567!",
              "name": "P6B1 Admin"},
    )
    assert admin_signup.status_code in (200, 201)
    from core.db import Collections, get_db
    db = get_db()
    await db[Collections.USERS].update_one(
        {"email": "p6b1_admin@test.com"}, {"$set": {"role": "admin"}},
    )
    re_login = await client.post(
        "/api/auth/login",
        json={"email": "p6b1_admin@test.com", "password": "Pass1234567!"},
    )
    admin_h = {"Authorization": f"Bearer {re_login.json()['access_token']}"}

    counts = (await client.get(
        f"/api/admin/search-sessions/{sid}/planner-counts",
        headers=admin_h,
    )).json()
    assert counts["session_id"] == sid
    assert counts["total_queries"] >= 1
    assert counts["primary_query_count"] >= 1
    assert counts["fallback_query_count"] == 0
    assert "boolean_compiler" in counts["counts_by_planner_source"]
    assert "legacy_planner" not in counts["counts_by_planner_source"]
    # Per-source breakdown is non-empty for the compiler.
    assert "europe_pmc" in (
        counts["counts_by_planner_and_source"].get("boolean_compiler") or {}
    )
    # planner_strategy summary is also returned on the endpoint.
    assert counts["planner_strategy"]["primary_planner"] == "boolean_compiler"
    assert counts["planner_strategy"]["legacy_planner_invoked"] is False
