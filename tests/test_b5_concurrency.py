"""B5 — bounded-concurrency retrieval, proven order-preserving.

The per-query fetch loops in the PubMed and Europe PMC connectors now run
concurrently. These tests prove the change is behavior-preserving: the returned
records appear in the EXACT same order as the former sequential loop (so
downstream dedup/ranking is byte-identical), errors/empties are handled the same
way, and the work actually overlaps (bounded by a semaphore). Pure unit tests —
no app boot, no DB, no network.
"""
from __future__ import annotations

import asyncio

import pytest

from services.audit import audit_service
from services.connectors.pubmed import PubMedConnector, pubmed_connector
from services.literature_retrieval import (
    LiteratureRetrievalService,
    literature_retrieval,
)
import services.literature_retrieval as lit_mod


@pytest.fixture(autouse=True)
def _no_audit_db(monkeypatch):
    """Keep these unit tests network-free: stub the audit logger (error paths
    would otherwise insert into Mongo)."""
    async def _noop(*args, **kwargs):
        return None
    monkeypatch.setattr(audit_service, "log", _noop)


async def test_pubmed_search_preserves_order_and_overlaps(monkeypatch):
    monkeypatch.setenv("PUBMED_MAX_CONCURRENT_QUERIES", "5")
    conc = {"cur": 0, "max": 0}

    async def fake_esearch(self, client, query):  # noqa: ANN001
        conc["cur"] += 1
        conc["max"] = max(conc["max"], conc["cur"])
        await asyncio.sleep(0.05)  # force overlap window
        conc["cur"] -= 1
        if query == "boom":
            raise RuntimeError("boom")
        return {"qa": ["1", "2"], "qc": ["3"]}[query]

    async def fake_fetch_full(self, client, pmids):  # noqa: ANN001
        return [{"pmid": p} for p in pmids]

    def fake_normalize(self, raw):  # noqa: ANN001
        return {"pmid": raw["pmid"]}

    async def fake_status(self, success, message=None):  # noqa: ANN001
        return None

    monkeypatch.setattr(PubMedConnector, "_esearch", fake_esearch)
    monkeypatch.setattr(PubMedConnector, "_fetch_full", fake_fetch_full)
    monkeypatch.setattr(PubMedConnector, "_normalize", fake_normalize)
    monkeypatch.setattr(PubMedConnector, "_update_status", fake_status)

    queries = [
        {"query_type": "A", "query_string": "qa"},
        {"query_type": "B", "query_string": ""},      # skipped (no status change)
        {"query_type": "C", "query_string": "qc"},
        {"query_type": "D", "query_string": "boom"},   # errors → no records
    ]
    out = await pubmed_connector.search(queries, client="dummy")

    # Byte-identical to the sequential loop: query order, then record order.
    assert [(qt, r["pmid"]) for qt, r in out] == [("A", "1"), ("A", "2"), ("C", "3")]
    # The queries genuinely ran concurrently (would be 1 if still sequential).
    assert conc["max"] >= 2


async def test_europe_pmc_preserves_order_and_overlaps(monkeypatch):
    monkeypatch.setenv("EUROPEPMC_MAX_CONCURRENT_QUERIES", "5")
    conc = {"cur": 0, "max": 0}

    async def fake_do_one(self, client, query_string):  # noqa: ANN001
        conc["cur"] += 1
        conc["max"] = max(conc["max"], conc["cur"])
        await asyncio.sleep(0.05)
        conc["cur"] -= 1
        if query_string == "boom":
            raise RuntimeError("boom")
        return {"qa": [{"id": "e1"}, {"id": "e2"}], "qc": [{"id": "e3"}]}[query_string]

    async def fake_status(name, success, message=None):  # noqa: ANN001
        return None

    monkeypatch.setattr(LiteratureRetrievalService, "_do_one", fake_do_one)
    monkeypatch.setattr(lit_mod, "_update_source_status", fake_status)

    queries = [
        {"query_type": "A", "query_string": "qa"},
        {"query_type": "B", "query_string": ""},
        {"query_type": "C", "query_string": "qc"},
        {"query_type": "D", "query_string": "boom"},
    ]
    out = await literature_retrieval.search_europe_pmc(queries, client="dummy")

    assert [(qt, r["id"]) for qt, r in out] == [("A", "e1"), ("A", "e2"), ("C", "e3")]
    assert conc["max"] >= 2


async def test_pubmed_all_failed_reports_not_ok(monkeypatch):
    """If every query errors, success status stays False (matches old loop)."""
    captured = {}

    async def fake_esearch(self, client, query):  # noqa: ANN001
        raise RuntimeError("down")

    async def fake_status(self, success, message=None):  # noqa: ANN001
        captured["success"] = success

    monkeypatch.setattr(PubMedConnector, "_esearch", fake_esearch)
    monkeypatch.setattr(PubMedConnector, "_update_status", fake_status)

    out = await pubmed_connector.search(
        [{"query_type": "A", "query_string": "x"}], client="dummy")
    assert out == []
    assert captured["success"] is False
