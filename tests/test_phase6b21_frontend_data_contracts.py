"""Phase 6B.2.1 — Frontend data contract tests.

These tests assert the BACKEND contracts that the new frontend wiring
depends on:
  (1) ``articles[]`` entries on the session GET response carry
      ``role_match_breakdown_json`` + ``role_match_score`` directly
      on the entry-level object (so WhyRankedHere can read them off
      ``sessionArticle``).
  (2) ``/transparency`` for a gibberish session returns
      ``planner_source="legacy_planner"`` and ``is_fallback_strategy=true``
      queries.
  (3) ``default_selection_summary.reason_each_selected`` entries match the
      article_ids that are actually returned by the session GET response.
  (4) Stale ``default_selection_summary`` is reconciled at GET time when
      one of the selected article_ids is no longer in ``articles[]``.
"""
from __future__ import annotations

import asyncio

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _signup(client, email: str) -> dict:
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!",
              "name": email.split("@")[0]},
    )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _wait_ready(client, sid: str, headers: dict, max_iter: int = 200):
    last = None
    for _ in range(max_iter):
        await asyncio.sleep(0.2)
        sr = await client.get(
            f"/api/search-sessions/{sid}/status", headers=headers,
        )
        last = sr.json()
        if last.get("articles_ready") or last.get("status") == "error":
            return last
    return last


async def _disable_pubmed():
    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": False}},
    )


async def _restore_pubmed():
    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": True}},
    )


import pytest_asyncio  # noqa: E402


@pytest_asyncio.fixture(autouse=True)
async def _pubmed_state_guard():
    yield
    await _restore_pubmed()


_EPMC_RECORD = {
    "id": "12345678",
    "source": "MED",
    "title": "RCT of semaglutide on LDL cholesterol in type 2 diabetes",
    "abstractText": (
        "Randomized controlled trial of semaglutide vs placebo in patients "
        "with type 2 diabetes evaluating LDL cholesterol change at 24 weeks."
    ),
    "journalInfo": {"journal": {"title": "NEJM"},
                     "yearOfPublication": "2024"},
    "pubYear": "2024",
    "firstPublicationDate": "2024-03-12",
    "doi": "10.1056/NEJMtest",
    "pmid": "12345678",
    "pmcid": "PMC9999999",
    "pubTypeList": {"pubType": ["Journal Article",
                                  "Randomized Controlled Trial"]},
    "authorList": {"author": [{"fullName": "Smith J", "lastName": "Smith"}]},
    "fullTextUrlList": {"fullTextUrl": []},
    "inEPMC": "Y",
}


# ---------------------------------------------------------------------------
# Test 1 — session GET response surfaces role_match data on each entry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_session_article_response_includes_role_match_breakdown_json(
    client, monkeypatch,
):
    """The session GET response's ``articles[i]`` items (entry-level,
    NOT the nested ``article`` bibliographic record) must carry
    ``role_match_breakdown_json`` and ``role_match_score`` so
    WhyRankedHere can render the role-match chips off ``sessionArticle``.
    """
    await _disable_pubmed()
    headers = await _signup(client, "p6b21a@test.com")

    async def fake_retrieve(source, queries, **kwargs):
        if not queries:
            return []
        qt = next(
            (q["query_type"] for q in queries
             if q["query_type"].startswith("cb_high_precision")),
            queries[0]["query_type"],
        )
        return [(qt, _EPMC_RECORD)]

    monkeypatch.setattr(
        "services.pipeline.literature_retrieval.retrieve", fake_retrieve,
    )
    r = await client.post(
        "/api/search-sessions/",
        json={"question": "What is the effect of semaglutide on LDL in "
                          "type 2 diabetes?"},
        headers=headers,
    )
    sid = r.json()["id"]
    await _wait_ready(client, sid, headers)
    sess = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    arts = sess.get("articles") or []
    assert arts, "expected at least one ranked article"
    entry = arts[0]
    # The keys live at the entry level (search_session_articles row),
    # NOT nested under entry["article"].
    assert "role_match_breakdown_json" in entry, list(entry.keys())
    assert "role_match_score" in entry, list(entry.keys())
    assert isinstance(entry["role_match_breakdown_json"], list)
    assert isinstance(entry["role_match_score"], int)


# ---------------------------------------------------------------------------
# Test 2 — /transparency returns fallback queries on gibberish sessions
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transparency_endpoint_returns_fallback_queries_for_gibberish_session(
    client, monkeypatch,
):
    """Compiler-zero-queries → PICOTT has anchors but the compiler can't
    materialize them → legacy planner fires → transparency endpoint
    returns rows with ``planner_source='legacy_planner'`` AND
    ``is_fallback_strategy=true``.

    Phase 6F closeout (IMPL-2) — gibberish input now triggers the
    ``awaiting_clinical_subject`` clarification gate BEFORE the
    compiler runs, so this test exercises a real clinical question
    (milrinone / SAH dosing) and monkey-patches the compiler to
    return zero queries. The legacy_planner fallback still fires for
    valid-anchor sessions whose compiler can't materialize the
    queries.
    """
    await _disable_pubmed()
    headers = await _signup(client, "p6b21b@test.com")

    async def fake_retrieve(source, queries, **kwargs):
        return [("high_recall", _EPMC_RECORD)] if queries else []

    monkeypatch.setattr(
        "services.pipeline.literature_retrieval.retrieve", fake_retrieve,
    )
    # Bypass the compiler so we deterministically trigger
    # ``compiler_yielded_zero_queries``.
    monkeypatch.setattr(
        "services.pipeline.concept_block_query_builder.build",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "services.pipeline.boolean_query_compiler.compile",
        lambda *args, **kwargs: [],
    )
    r = await client.post(
        "/api/search-sessions/",
        json={"question": "What is the dose of milrinone for subarachnoid hemorrhage?"},
        headers=headers,
    )
    sid = r.json()["id"]
    await _wait_ready(client, sid, headers)

    trans = (await client.get(
        f"/api/search-sessions/{sid}/transparency", headers=headers,
    )).json()
    queries = trans.get("queries") or []
    legacy = [q for q in queries
                if q.get("planner_source") == "legacy_planner"
                and q.get("is_fallback_strategy") is True]
    assert legacy, (
        f"expected at least one legacy_planner / is_fallback_strategy=True query, "
        f"got {[(q.get('planner_source'), q.get('is_fallback_strategy')) for q in queries]}"
    )
    # Every legacy row carries ``fallback_reason``.
    assert all(q.get("fallback_reason") for q in legacy)


# ---------------------------------------------------------------------------
# Test 3 — default_selection_summary IDs match the articles list (fresh session)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_default_selection_summary_consistent_with_articles_list(
    client, monkeypatch,
):
    """For a freshly created session, every ``article_id`` in
    ``default_selection_summary.reason_each_selected`` MUST be present
    in the GET response's ``articles[]`` list — i.e. the GET response
    is internally consistent."""
    await _disable_pubmed()
    headers = await _signup(client, "p6b21c@test.com")

    async def fake_retrieve(source, queries, **kwargs):
        if not queries:
            return []
        qt = next(
            (q["query_type"] for q in queries
             if q["query_type"].startswith("cb_high_precision")),
            queries[0]["query_type"],
        )
        return [(qt, _EPMC_RECORD)]

    monkeypatch.setattr(
        "services.pipeline.literature_retrieval.retrieve", fake_retrieve,
    )
    r = await client.post(
        "/api/search-sessions/",
        json={"question": "Effect of semaglutide on LDL in type 2 diabetes?"},
        headers=headers,
    )
    sid = r.json()["id"]
    await _wait_ready(client, sid, headers)
    sess = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    arts = sess.get("articles") or []
    included_ids = {a["article_id"] for a in arts}
    ds = sess.get("default_selection_summary") or {}
    for entry in ds.get("reason_each_selected") or []:
        assert entry["article_id"] in included_ids, (
            f"default_selection_summary references article_id "
            f"{entry['article_id']!r} that is not in articles[] "
            f"(internal inconsistency)."
        )


# ---------------------------------------------------------------------------
# Test 4 — stale default_selection_summary is reconciled at GET time
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_default_selection_summary_rebuilt_when_articles_dropped(
    client, monkeypatch,
):
    """Build a session with selected articles, manually flip one of them
    to ``exclusion_reason="low_relevance"`` in the
    ``search_session_articles`` collection (simulating a stale legacy
    session), then GET the session and assert ``selected_count`` was
    reconciled downward."""
    from core.db import Collections, get_db

    await _disable_pubmed()
    headers = await _signup(client, "p6b21d@test.com")

    async def fake_retrieve(source, queries, **kwargs):
        if not queries:
            return []
        qt = next(
            (q["query_type"] for q in queries
             if q["query_type"].startswith("cb_high_precision")),
            queries[0]["query_type"],
        )
        return [(qt, _EPMC_RECORD)]

    monkeypatch.setattr(
        "services.pipeline.literature_retrieval.retrieve", fake_retrieve,
    )
    r = await client.post(
        "/api/search-sessions/",
        json={"question": "Effect of semaglutide on LDL in type 2 diabetes?"},
        headers=headers,
    )
    sid = r.json()["id"]
    await _wait_ready(client, sid, headers)
    sess = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    ds = sess.get("default_selection_summary") or {}
    assert ds.get("selected_count", 0) >= 1
    original_count = ds["selected_count"]
    # Pick one selected article and manually exclude it (simulating
    # a stale pre-6B.2.1 session inconsistency).
    target_id = (ds.get("reason_each_selected") or [None])[0]["article_id"]
    db = get_db()
    await db[Collections.SEARCH_SESSION_ARTICLES].update_one(
        {"search_session_id": sid, "article_id": target_id},
        {"$set": {"exclusion_reason": "low_relevance",
                   "included_by_default": False}},
    )
    # GET again. The reconciliation helper must drop the dropped id
    # from reason_each_selected and decrement selected_count.
    sess2 = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    ds2 = sess2.get("default_selection_summary") or {}
    assert ds2.get("stale_summary_reconciled") is True, ds2
    assert ds2.get("selected_count") == original_count - 1, ds2
    surviving_ids = {e.get("article_id")
                      for e in ds2.get("reason_each_selected") or []}
    assert target_id not in surviving_ids
