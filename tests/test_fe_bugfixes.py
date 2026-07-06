"""Regression tests for the two integrated-frontend bugs:

  Issue 2 — GET /search-sessions/{id}/status progress must be MONOTONIC and cover
            every pipeline status (the old map returned 0 for brokering_access /
            fetching_oa_full_text, making the bar jump 60% -> 0% and the FE
            poll-loop never finish).
  Issue 1 — POST /library/items must resolve a paper that is NOT a local article
            (a central LitHub "lithub:<id>" / bare paper id, or a PMID/DOI, or
            provided metadata) instead of always returning article_not_found.

Uses the session-scoped ``client`` fixture + seeded clinician. asyncio_mode=auto.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest_asyncio

from core.db import Collections, get_db


async def _login(client) -> dict[str, str]:
    r = await client.post("/api/auth/login", json={
        "email": "clinician@test.com", "password": "ClinicianTest123!"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest_asyncio.fixture()
async def auth(client):
    return await _login(client)


async def _me_id(client, auth) -> str:
    return (await client.get("/api/auth/me", headers=auth)).json()["id"]


# ════════════════════════════════════════════════════════════════════
# Issue 2 — status progress is monotonic + covers all phases
# ════════════════════════════════════════════════════════════════════
_PHASE_ORDER = [
    ("searching", 25),
    ("ranking", 45),
    ("brokering_access", 65),
    ("fetching_oa_full_text", 85),
    ("summarizing", 92),
    ("awaiting_selection", 100),
]


async def _seed_session(db, uid, status):
    sid = "fx-" + uuid.uuid4().hex[:10]
    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": uid, "status": status,
        "original_question": "q", "sanitized_question": "q",
        "deleted_at": None, "created_at": now, "updated_at": now,
    })
    return sid


async def test_status_progress_is_monotonic_across_phases(client, auth):
    db = get_db()
    uid = await _me_id(client, auth)
    last = -1
    for status, expected in _PHASE_ORDER:
        sid = await _seed_session(db, uid, status)
        r = await client.get(f"/api/search-sessions/{sid}/status", headers=auth)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["progress"] == expected, f"{status} -> {body['progress']} (want {expected})"
        # never regresses along the real pipeline order (the reported 60->0 bug).
        assert body["progress"] >= last, f"progress regressed at {status}"
        last = body["progress"]


async def test_status_mid_pipeline_never_zero(client, auth):
    """brokering_access / fetching_oa_full_text used to map to 0% — the bug."""
    db = get_db()
    uid = await _me_id(client, auth)
    for status in ("brokering_access", "fetching_oa_full_text", "summarizing"):
        sid = await _seed_session(db, uid, status)
        r = await client.get(f"/api/search-sessions/{sid}/status", headers=auth)
        assert r.json()["progress"] > 0, f"{status} still maps to 0%"


async def test_status_terminal_states_report_100_and_done(client, auth):
    db = get_db()
    uid = await _me_id(client, auth)
    for status in ("awaiting_selection", "complete", "error",
                   "blocked_phi", "awaiting_clinical_subject"):
        sid = await _seed_session(db, uid, status)
        r = await client.get(f"/api/search-sessions/{sid}/status", headers=auth)
        body = r.json()
        assert body["progress"] == 100, f"{status} progress != 100"
        assert body["done"] is True, f"{status} not marked done"


# ════════════════════════════════════════════════════════════════════
# Issue 1 — library save resolves non-local article ids
# ════════════════════════════════════════════════════════════════════
async def _seed_article(db, *, pmid=None, doi=None, doi_norm=None):
    aid = "fx-art-" + uuid.uuid4().hex[:10]
    now = datetime.now(timezone.utc)
    await db[Collections.ARTICLES].insert_one({
        "id": aid, "canonical_key": aid, "title": "Seeded paper",
        "journal": "NEJM", "publication_year": 2024,
        "pmid": pmid, "doi": doi, "doi_normalized": doi_norm,
        "sources_present": ["pubmed"], "full_text_status": "abstract_only",
        "created_at": now, "updated_at": now,
    })
    return aid


async def test_save_local_article_id_still_works(client, auth):
    db = get_db()
    aid = await _seed_article(db)
    r = await client.post("/api/library/items", json={"article_id": aid}, headers=auth)
    assert r.status_code == 201, r.text
    assert r.json()["library_item"]["article_id"] == aid


async def test_save_resolves_by_pmid_when_id_not_local(client, auth):
    db = get_db()
    pmid = "39" + uuid.uuid4().hex[:6]
    aid = await _seed_article(db, pmid=pmid)
    # article_id is a bogus non-local id; pmid metadata resolves to the seeded art.
    r = await client.post("/api/library/items", json={
        "article_id": f"lithub:{uuid.uuid4()}", "pmid": pmid}, headers=auth)
    assert r.status_code == 201, r.text
    assert r.json()["library_item"]["article_id"] == aid  # resolved to the local one


async def test_save_bare_pmid_as_article_id(client, auth):
    db = get_db()
    pmid = str(uuid.uuid4().int)[:8]  # a realistic all-numeric PMID
    aid = await _seed_article(db, pmid=pmid)
    r = await client.post("/api/library/items", json={"article_id": pmid}, headers=auth)
    assert r.status_code == 201, r.text
    assert r.json()["library_item"]["article_id"] == aid


async def test_save_creates_from_metadata_when_unknown(client, auth):
    doi = f"10.9999/{uuid.uuid4().hex[:8]}"
    r = await client.post("/api/library/items", json={
        "article_id": f"lithub:{uuid.uuid4()}",
        "doi": doi, "title": "A brand-new paper", "journal": "JAMA",
        "publication_year": 2025, "authors": ["Doe J"]}, headers=auth)
    assert r.status_code == 201, r.text
    li = r.json()["library_item"]
    assert li["article_id"] and not li["article_id"].startswith("lithub:")
    # a local article was created and is now saved.
    db = get_db()
    art = await db[Collections.ARTICLES].find_one({"id": li["article_id"]}, {"_id": 0})
    assert art and art["doi_normalized"] == doi.lower()


async def test_save_unresolvable_id_still_404(client, auth):
    """No metadata + a non-local, non-identifier id → graceful article_not_found."""
    r = await client.post("/api/library/items", json={
        "article_id": "totally-bogus-" + uuid.uuid4().hex}, headers=auth)
    assert r.status_code == 404
    assert r.json()["detail"] == "article_not_found"
