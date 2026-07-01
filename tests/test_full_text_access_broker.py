"""Phase 2C — FullTextAccessBrokerService unit tests."""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.full_text_access_broker import (  # noqa: E402
    FullTextAccessBrokerService,
    _ingestible,
    _normalize_license,
)
from core.db import Collections, get_db  # noqa: E402


def _art(article_id: str = None, **overrides) -> dict:
    base = {
        "id": article_id or uuid.uuid4().hex,
        "title": "Sample article",
        "abstract": "An abstract.",
        "doi": None,
        "pmid": None,
        "pmcid": None,
        "source_ids_json": {},
        "source_urls_json": [],
        "full_text_urls_json": [],
        "publication_year": 2021,
        "license": None,
    }
    base.update(overrides)
    return base


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _RouteClient:
    """Simple URL-substring matcher; raises if an unexpected URL is hit."""

    def __init__(self, routes: dict, strict: bool = True):
        self._routes = routes
        self._strict = strict

    async def get(self, url, params=None, headers=None):
        for needle, resp in self._routes.items():
            if needle in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        if self._strict:
            raise RuntimeError(f"unexpected URL: {url}")
        return _FakeResponse({}, status=404)

    async def aclose(self):
        pass


async def _persist_article(art):
    db = get_db()
    await db[Collections.ARTICLES].update_one(
        {"id": art["id"]}, {"$set": art}, upsert=True
    )


async def _link_to_session(article_id):
    db = get_db()
    session_id = uuid.uuid4().hex
    # Minimal session row so broker_for_session can find articles.
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": session_id,
        "user_id": "test-user",
        "status": "ranking",
        "original_question": "x",
        "sanitized_question": "x",
        "deleted_at": None,
    })
    await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
        "id": uuid.uuid4().hex,
        "search_session_id": session_id,
        "article_id": article_id,
    })
    return session_id


async def _wipe_for(article_id):
    db = get_db()
    await db[Collections.ARTICLE_ACCESS_LOCATIONS].delete_many({"article_id": article_id})


@pytest.mark.asyncio
async def test_pmcid_creates_pmc_oa_row():
    """PMCID → automatic pmc_oa row, no external call."""
    art = _art(pmcid="PMC1234567", title="t", abstract="a")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = False
    broker.crossref_enabled = False
    broker.ss_enabled = False
    await broker.broker_for_session(sid, client=_RouteClient({}, strict=True))
    rows = await get_db()[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": art["id"]}, {"_id": 0}
    ).to_list(length=20)
    assert any(r["access_category"] == "pmc_oa" and r["is_oa"] for r in rows)


@pytest.mark.asyncio
async def test_europe_pmc_oa_indicator_creates_row():
    art = _art(
        pmid="999",
        source_ids_json={"inEPMC": True},
        source_urls_json=["https://europepmc.org/article/MED/999"],
    )
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = False
    broker.crossref_enabled = False
    broker.ss_enabled = False
    await broker.broker_for_session(sid, client=_RouteClient({}, strict=True))
    rows = await get_db()[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": art["id"]}, {"_id": 0}
    ).to_list(length=20)
    assert any(r["access_category"] == "europe_pmc_oa" for r in rows)


@pytest.mark.asyncio
async def test_unpaywall_best_oa_location_creates_row():
    unpay_resp = {
        "best_oa_location": {
            "url_for_pdf": "https://oa.example/x.pdf",
            "host_type": "publisher",
            "license": "cc-by",
            "version": "publishedVersion",
        }
    }
    art = _art(doi="10.1056/NEJMoa2107038")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = True
    broker.unpaywall_email = "test@x.com"
    broker.crossref_enabled = False
    broker.ss_enabled = False
    client = _RouteClient({"api.unpaywall.org": _FakeResponse(unpay_resp)})
    await broker.broker_for_session(sid, client=client)
    rows = await get_db()[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": art["id"]}, {"_id": 0}
    ).to_list(length=20)
    unpay_row = next(r for r in rows if r["source"] == "unpaywall")
    assert unpay_row["access_category"] == "oa_pdf"
    assert unpay_row["is_direct_pdf"]
    assert unpay_row["license"] == "cc-by"
    assert unpay_row["can_ingest_for_summary"] is True


@pytest.mark.asyncio
async def test_unpaywall_missing_email_disables_source_no_failure(monkeypatch):
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.delenv("UNPAYWALL_ENABLED", raising=False)
    art = _art(doi="10.1/abc")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    # No Unpaywall call should happen (strict client would crash).
    broker.crossref_enabled = False
    broker.ss_enabled = False
    summary = await broker.broker_for_session(sid, client=_RouteClient({}, strict=True))
    assert summary["sources_status"]["unpaywall"]["status"] == "not_configured"


@pytest.mark.asyncio
async def test_unpaywall_failure_is_audited_not_raised():
    art = _art(doi="10.1/fail")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = True
    broker.unpaywall_email = "x@x.com"
    broker.crossref_enabled = False
    broker.ss_enabled = False
    # Connector raises on unpaywall.
    routes = {"api.unpaywall.org": httpx.HTTPError("simulated outage")}
    summary = await broker.broker_for_session(sid, client=_RouteClient(routes))
    assert summary["sources_status"]["unpaywall"]["status"] == "error"


@pytest.mark.asyncio
async def test_semantic_scholar_oa_pdf_creates_row():
    ss_resp = {
        "openAccessPdf": {"url": "https://ss.example/paper.pdf"},
        "isOpenAccess": True,
        "externalIds": {"DOI": "10.1/ss"},
    }
    art = _art(doi="10.1/ss")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = False
    broker.crossref_enabled = False
    broker.ss_enabled = True
    broker.ss_api_key = ""
    client = _RouteClient({"api.semanticscholar.org": _FakeResponse(ss_resp)})
    await broker.broker_for_session(sid, client=client)
    rows = await get_db()[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": art["id"]}, {"_id": 0}
    ).to_list(length=20)
    ss_row = next(r for r in rows if r["source"] == "semantic_scholar"
                  and r["access_category"] == "oa_pdf")
    assert ss_row["url"] == "https://ss.example/paper.pdf"
    # No license returned from S2 → can_ingest_for_summary must be False.
    assert ss_row["can_ingest_for_summary"] is False


@pytest.mark.asyncio
async def test_crossref_disabled_returns_no_calls(monkeypatch):
    monkeypatch.delenv("CROSSREF_MAILTO", raising=False)
    monkeypatch.delenv("CROSSREF_ENABLED", raising=False)
    art = _art(doi="10.1/cross")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = False
    broker.ss_enabled = False
    # Strict client: any unexpected URL crashes — Crossref must not be called.
    summary = await broker.broker_for_session(sid, client=_RouteClient({}, strict=True))
    assert summary["sources_status"]["crossref"]["status"] == "not_configured"


@pytest.mark.asyncio
async def test_crossref_text_mining_link_marked_non_ingestible_when_license_unknown():
    cr_resp = {
        "message": {
            "DOI": "10.1/xref",
            "license": [],  # no license info
            "link": [
                {
                    "URL": "https://publisher.example/text-mining/x.pdf",
                    "intended-application": "text-mining",
                }
            ],
        }
    }
    art = _art(doi="10.1/xref")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = False
    broker.crossref_enabled = True
    broker.crossref_mailto = "test@x.com"
    broker.ss_enabled = False
    client = _RouteClient({"api.crossref.org": _FakeResponse(cr_resp)})
    await broker.broker_for_session(sid, client=client)
    rows = await get_db()[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": art["id"]}, {"_id": 0}
    ).to_list(length=20)
    tm_row = next(
        r for r in rows
        if r["source"] == "crossref" and "text-mining" in (r["url"] or "")
    )
    assert tm_row["can_ingest_for_summary"] is False


@pytest.mark.asyncio
async def test_license_unknown_results_in_can_ingest_false():
    # _ingestible should be False for unknown license even on pmc_oa category.
    assert _ingestible(is_oa=True, license_=None, access_category="pmc_oa") is False
    assert _ingestible(is_oa=True, license_="unknown", access_category="pmc_oa") is False


@pytest.mark.asyncio
async def test_subscription_link_out_results_in_can_ingest_false():
    # Institutional category must never be ingestible regardless of OA.
    assert _ingestible(
        is_oa=False, license_="cc-by", access_category="institutional_openathens"
    ) is False
    assert _ingestible(
        is_oa=True, license_="cc-by", access_category="publisher_doi_link"
    ) is False


@pytest.mark.asyncio
async def test_abstract_only_fallback_when_no_access_routes():
    art = _art(title="Solo", abstract="Some abstract present.")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = False
    broker.crossref_enabled = False
    broker.ss_enabled = False
    await broker.broker_for_session(sid, client=_RouteClient({}, strict=True))
    rows = await get_db()[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": art["id"]}, {"_id": 0}
    ).to_list(length=20)
    assert any(r["access_category"] == "abstract_only" for r in rows)


@pytest.mark.asyncio
async def test_unknown_when_no_data():
    art = _art(title="", abstract="")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = False
    broker.crossref_enabled = False
    broker.ss_enabled = False
    await broker.broker_for_session(sid, client=_RouteClient({}, strict=True))
    rows = await get_db()[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": art["id"]}, {"_id": 0}
    ).to_list(length=20)
    assert any(r["access_category"] == "unknown" for r in rows)


@pytest.mark.asyncio
async def test_publisher_doi_link_added_when_doi_only():
    art = _art(doi="10.5555/onlydoi")
    await _persist_article(art)
    sid = await _link_to_session(art["id"])
    broker = FullTextAccessBrokerService()
    broker.unpaywall_enabled = False
    broker.crossref_enabled = False
    broker.ss_enabled = False
    await broker.broker_for_session(sid, client=_RouteClient({}, strict=True))
    rows = await get_db()[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"article_id": art["id"]}, {"_id": 0}
    ).to_list(length=20)
    doi_row = next(r for r in rows if r["access_category"] == "publisher_doi_link")
    assert doi_row["url"] == "https://doi.org/10.5555/onlydoi"
    assert doi_row["can_ingest_for_summary"] is False
    assert doi_row["requires_login"] is False


def test_license_normalization():
    assert _normalize_license("CC-BY") == "cc-by"
    assert _normalize_license("https://creativecommons.org/licenses/by/4.0/") == "cc-by-4.0" or \
           _normalize_license("https://creativecommons.org/licenses/by/4.0/") == "cc-by"
    assert _normalize_license("cc0") == "cc0"
    assert _normalize_license("publicdomain") == "public-domain"
    assert _normalize_license(None) is None
