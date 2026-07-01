"""Integration tests for the B15 single-paper full-text import API.

The resolver (PubMed/Crossref), OA access broker, and OA fetcher each make
outbound calls, so they are stubbed here; everything else — session
create/reuse, attach, article enrichment, status computation, governance
gates, and the optional save — runs for real against the test Mongo.

Uses the session-scoped ``client`` fixture + seeded clinician. asyncio_mode=auto.
"""
from __future__ import annotations

import uuid

import pytest_asyncio

from core.db import Collections, get_db
import services.full_text_import as fti


async def _login(client) -> dict[str, str]:
    r = await client.post(
        "/api/auth/login",
        json={"email": "clinician@test.com", "password": "ClinicianTest123!"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest_asyncio.fixture()
async def auth(client):
    return await _login(client)


def _candidate(*, confidence="high", source="pubmed", pmid=None, doi=None,
               retracted=False, eoc=False, title="Empagliflozin in HFpEF"):
    return {
        "match_confidence": confidence,
        "match_reason_codes": ["exact_pmid_match"] if pmid else ["exact_title_year_match"],
        "pmid": pmid, "doi": doi, "pmcid": None,
        "title": title, "journal": "NEJM", "publication_year": 2024,
        "authors": ["Anker SD"], "abstract": "RCT abstract: SGLT2i reduced HF events.",
        "publication_types": ["Randomized Controlled Trial"],
        "is_retracted": retracted, "has_expression_of_concern": eoc,
        "source": source,
    }


def _patch_pipeline(monkeypatch, *, candidate, oa_status, can_ingest,
                    broker_rows=None):
    """Stub the three outbound-call dependencies."""
    async def fake_search(raw_text):  # noqa: ANN001
        if candidate is None:
            return {"phi_blocked": False, "candidates": [],
                    "lookup_source_provenance": ["pubmed"]}
        return {"phi_blocked": False, "candidates": [candidate],
                "lookup_source_provenance": [candidate["source"]]}

    async def fake_broker(session_id, *, client=None, replace_existing=True):  # noqa: ANN001
        if broker_rows:
            db = get_db()
            sa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
                {"search_session_id": session_id}, {"_id": 0, "article_id": 1})
            for row in broker_rows:
                row = {**row, "id": uuid.uuid4().hex, "article_id": sa["article_id"]}
                await db[Collections.ARTICLE_ACCESS_LOCATIONS].insert_one(row)
        return {"articles_processed": 1}

    async def fake_oa(*, article_id, session_id, user_id, force_refresh=False, client=None):  # noqa: ANN001
        return {
            "article_id": article_id, "status": oa_status,
            "source": "pmc_oa" if oa_status in ("fetched", "cache_hit") else None,
            "license": "cc-by" if can_ingest else "unknown",
            "license_category": "commercial_reuse_allowed" if can_ingest else "unknown",
            "can_ingest_for_summary": can_ingest,
            "sections_created": 6 if can_ingest else 0,
            "token_count": 4200 if can_ingest else 0,
            "evidence_text_level": "oa_full_text" if can_ingest else "abstract",
            "error_code": None if can_ingest else "license_unknown",
        }

    monkeypatch.setattr(fti.known_article_lookup_service, "search", fake_search)
    monkeypatch.setattr(fti.full_text_access_broker, "broker_for_session", fake_broker)
    monkeypatch.setattr(fti.oa_fetcher, "fetch_for_article", fake_oa)


async def test_import_full_text_ingested_and_chat_ready(client, auth, monkeypatch):
    pmid = "39" + uuid.uuid4().hex[:6]
    cand = _candidate(pmid=pmid, doi=f"10.1056/{uuid.uuid4().hex[:6]}")
    _patch_pipeline(monkeypatch, candidate=cand, oa_status="fetched", can_ingest=True)

    r = await client.post("/api/full-text-import",
                          json={"pmid": pmid, "save": True}, headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "full_text_ingested"
    assert body["full_text"]["ingested_for_ai"] is True
    assert body["chat"]["ready"] is True
    assert body["chat"]["generate_summary_url"].endswith("/generate-summary")
    assert body["session_id"] and body["article_id"]
    # save landed in the default "LitPortal Imports" collection.
    assert body["saved"]["collection_name"] == "LitPortal Imports"
    assert body["saved"]["added_count"] == 1

    db = get_db()
    # session is an import session, awaiting_selection.
    sess = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": body["session_id"]}, {"_id": 0})
    assert sess["import_origin"] == "full_text_import"
    assert sess["status"] == "awaiting_selection"
    assert sess["imported_article_id"] == body["article_id"]
    # article attached + user_included, and enriched with sources_present.
    ssa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"search_session_id": body["session_id"], "article_id": body["article_id"]},
        {"_id": 0})
    assert ssa["user_included"] is True
    art = await db[Collections.ARTICLES].find_one(
        {"id": body["article_id"]}, {"_id": 0})
    assert art["sources_present"] == ["pubmed"]
    assert art.get("doi_normalized")


async def test_import_idempotent_reuses_session(client, auth, monkeypatch):
    pmid = "39" + uuid.uuid4().hex[:6]
    cand = _candidate(pmid=pmid)
    _patch_pipeline(monkeypatch, candidate=cand, oa_status="fetched", can_ingest=True)

    r1 = await client.post("/api/full-text-import", json={"pmid": pmid}, headers=auth)
    r2 = await client.post("/api/full-text-import", json={"pmid": pmid}, headers=auth)
    assert r1.status_code == r2.status_code == 200
    # Same paper → same import session reused (no duplicate sessions).
    assert r1.json()["session_id"] == r2.json()["session_id"]
    assert r2.json()["was_duplicate"] is True


async def test_import_link_out_only_when_license_blocks_ingest(client, auth, monkeypatch):
    doi = f"10.9999/{uuid.uuid4().hex[:6]}"
    cand = _candidate(doi=doi, pmid="39" + uuid.uuid4().hex[:6])
    # OA fetch skipped on license; broker found a publisher DOI link-out.
    _patch_pipeline(
        monkeypatch, candidate=cand, oa_status="skipped_license", can_ingest=False,
        broker_rows=[{
            "source": "article", "access_category": "publisher_doi_link",
            "url": f"https://doi.org/{doi}", "is_oa": False, "is_direct_pdf": False,
            "requires_login": False, "can_view_in_app": False,
            "can_ingest_for_summary": False, "license": None,
            "evidence_text_level": "metadata_only",
        }],
    )
    r = await client.post("/api/full-text-import", json={"doi": doi}, headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "full_text_link_out"
    assert body["full_text"]["ingested_for_ai"] is False
    # the human still gets a "Get Full Text" / "Open DOI" route.
    assert body["full_text"]["best_access"]["target_url"]
    # abstract is present, so chat (over abstract) is still available.
    assert body["chat"]["ready"] is True


async def test_low_confidence_needs_confirmation(client, auth, monkeypatch):
    cand = _candidate(confidence="low", title="A vague title fragment")
    _patch_pipeline(monkeypatch, candidate=cand, oa_status="fetched", can_ingest=True)
    r = await client.post("/api/full-text-import",
                          json={"title": "A vague title fragment"}, headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "needs_confirmation"
    assert body["session_id"] is None
    assert len(body["candidates"]) >= 1
    # opting in imports the top candidate.
    r2 = await client.post(
        "/api/full-text-import",
        json={"title": "A vague title fragment", "accept_low_confidence": True},
        headers=auth)
    assert r2.json()["status"] == "full_text_ingested"


async def test_retracted_is_blocked(client, auth, monkeypatch):
    cand = _candidate(pmid="39" + uuid.uuid4().hex[:6], retracted=True)
    _patch_pipeline(monkeypatch, candidate=cand, oa_status="fetched", can_ingest=True)
    r = await client.post("/api/full-text-import",
                          json={"pmid": cand["pmid"]}, headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "blocked_safety"
    assert body["reason"] == "article_retracted"
    assert body["session_id"] is None


async def test_not_found(client, auth, monkeypatch):
    _patch_pipeline(monkeypatch, candidate=None, oa_status="fetched", can_ingest=True)
    r = await client.post("/api/full-text-import",
                          json={"pmid": "0000000"}, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "not_found"


async def test_requires_an_identifier(client, auth):
    r = await client.post("/api/full-text-import", json={}, headers=auth)
    assert r.status_code == 422


async def test_requires_auth(client):
    r = await client.post("/api/full-text-import", json={"pmid": "123456"})
    assert r.status_code in (401, 403)
