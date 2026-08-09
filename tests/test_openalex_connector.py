"""Phase 2B — OpenAlex enrichment tests."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.connectors.openalex import OpenAlexConnector  # noqa: E402


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


def _oa_work(doi="10.1056/nejmoa2107038", citation=1234, oa_url="https://oa.example/x.pdf",
             is_oa=True, type_field="article", title=None, year=None,
             pmid="34449189", pmcid="PMC8500551"):
    return {
        "id": "https://openalex.org/W1234",
        "ids": {
            "openalex": "https://openalex.org/W1234",
            "doi": f"https://doi.org/{doi}",
            "pmid": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
            "pmcid": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}",
        },
        "title": title or "Empagliflozin in HFpEF",
        "display_name": title or "Empagliflozin in HFpEF",
        "publication_year": year or 2021,
        "host_venue": {"display_name": "NEJM"},
        "primary_location": {"source": {"display_name": "NEJM"}},
        "best_oa_location": {"landing_page_url": oa_url, "pdf_url": oa_url},
        "open_access": {"is_oa": is_oa},
        "cited_by_count": citation,
        "type": type_field,
        "authorships": [{"author": {"display_name": "Anker SD"}}],
    }


class _RouteClient:
    def __init__(self, route_map):
        self._routes = route_map

    async def get(self, url, params=None):
        for needle, resp in self._routes.items():
            if needle in url:
                return resp
        return _FakeResponse({"results": []}, status=404)

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_openalex_doi_lookup_enriches():
    art = {
        "canonical_key": "pmid:34449189",
        "doi": "10.1056/NEJMoa2107038",
        "pmid": "34449189",
        "title": "Empagliflozin in HFpEF",
        "abstract": "Abstract",
        "full_text_status": "abstract_only",
        "publication_year": 2021,
    }
    client = _RouteClient({"works/doi:": _FakeResponse(_oa_work())})
    conn = OpenAlexConnector()
    stats = await conn.enrich_many([art], client=client)
    assert stats["enrichments_applied"] == 1
    assert art["enriched_by_openalex"] is True
    assert art["citation_count"] == 1234
    # OA + is_oa → promote unknown/abstract_only → free_full_text_link
    assert art["full_text_status"] == "free_full_text_link"
    assert any("oa.example" in u for u in art["source_urls_json"])
    assert art["source_ids_json"]["openalex"] == "https://openalex.org/W1234"


@pytest.mark.asyncio
async def test_openalex_never_promotes_to_full_text_available():
    """OpenAlex must NOT upgrade status above free_full_text_link."""
    art = {
        "canonical_key": "pmid:1",
        "doi": "10.1/x",
        "title": "T",
        "abstract": "A",
        "full_text_status": "free_full_text_link",
        "publication_year": 2020,
    }
    client = _RouteClient({"works/doi:": _FakeResponse(_oa_work())})
    await OpenAlexConnector().enrich_many([art], client=client)
    assert art["full_text_status"] == "free_full_text_link"

    art2 = {
        "canonical_key": "pmid:2",
        "doi": "10.1/y",
        "title": "T",
        "abstract": "A",
        "full_text_status": "full_text_available",
        "publication_year": 2020,
    }
    await OpenAlexConnector().enrich_many([art2], client=client)
    assert art2["full_text_status"] == "full_text_available"


@pytest.mark.asyncio
async def test_openalex_falls_back_to_pmid_then_pmcid():
    """No DOI → PMID → PMCID."""
    art = {
        "canonical_key": "pmid:1",
        "doi": None,
        "pmid": "1111",
        "pmcid": "PMC9999",
        "title": "T", "abstract": "A", "publication_year": 2021,
    }
    # No DOI route. PMID route succeeds.
    client = _RouteClient({"works/pmid:": _FakeResponse(_oa_work(citation=42))})
    stats = await OpenAlexConnector().enrich_many([art], client=client)
    assert stats["enrichments_applied"] == 1
    assert art["citation_count"] == 42


@pytest.mark.asyncio
async def test_openalex_title_search_requires_similarity_and_year():
    """Title fallback must reject low-similarity or wrong-year matches."""
    art = {
        "canonical_key": "title_year_journal_firstauthor:foo_2021_x_y",
        "title": "Empagliflozin in HFpEF",
        "abstract": "A", "publication_year": 2021,
        "doi": None, "pmid": None, "pmcid": None,
    }
    # First simulate a low-similarity hit.
    bad = {
        "results": [{
            "id": "https://openalex.org/W1",
            "title": "Completely unrelated study about cats",
            "display_name": "cats",
            "publication_year": 2021,
            "ids": {},
            "open_access": {"is_oa": False},
        }]
    }
    client = _RouteClient({"works?": _FakeResponse(bad)})
    stats = await OpenAlexConnector().enrich_many([art], client=client)
    assert stats["enrichments_applied"] == 0

    # Now a high-similarity match with year off by 5 — must be rejected.
    art2 = dict(art)
    art2["canonical_key"] = "x"
    far_year = {
        "results": [{
            "id": "https://openalex.org/W2",
            "title": "Empagliflozin in HFpEF",
            "display_name": "Empagliflozin in HFpEF",
            "publication_year": 2010,
            "ids": {},
            "open_access": {"is_oa": False},
        }]
    }
    client2 = _RouteClient({"works?": _FakeResponse(far_year)})
    stats2 = await OpenAlexConnector().enrich_many([art2], client=client2)
    assert stats2["enrichments_applied"] == 0


@pytest.mark.asyncio
async def test_openalex_does_not_overwrite_safety_flags():
    art = {
        "canonical_key": "pmid:1",
        "doi": "10.1/x",
        "title": "T", "abstract": "A", "publication_year": 2020,
        "is_retracted": True,
        "has_expression_of_concern": True,
        "is_preprint": True,
    }
    # OpenAlex returns clean type=article, but we must NOT reset these flags.
    client = _RouteClient({"works/doi:": _FakeResponse(_oa_work(type_field="article"))})
    await OpenAlexConnector().enrich_many([art], client=client)
    assert art["is_retracted"] is True
    assert art["has_expression_of_concern"] is True
    assert art["is_preprint"] is True


@pytest.mark.asyncio
async def test_openalex_marks_preprint_to_true_only():
    art = {
        "canonical_key": "x", "doi": "10.1/p",
        "title": "T", "abstract": "A", "publication_year": 2024,
        "is_preprint": False,
    }
    client = _RouteClient({"works/doi:": _FakeResponse(_oa_work(type_field="preprint"))})
    await OpenAlexConnector().enrich_many([art], client=client)
    assert art["is_preprint"] is True
    assert "openalex_marked_preprint" in (art.get("flags") or [])


@pytest.mark.asyncio
async def test_openalex_respects_max_enrichments_cap(monkeypatch):
    monkeypatch.setenv("OPENALEX_MAX_ENRICHMENTS_PER_SEARCH", "2")
    conn = OpenAlexConnector()
    # The cap is read at __init__; rebuild the connector after env change.
    arts = [
        {"canonical_key": f"k{i}", "doi": f"10.1/{i}",
         "title": "T", "abstract": "A", "publication_year": 2021,
         "_pre_rank_score": 100 - i}
        for i in range(5)
    ]
    client = _RouteClient({"works/doi:": _FakeResponse(_oa_work())})
    stats = await conn.enrich_many(arts, client=client)
    # Only top 2 (by _pre_rank_score) should be enriched.
    assert stats["enrichments_attempted"] == 2
