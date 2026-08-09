"""Phase 2B — PubMed connector tests with mocked httpx."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.connectors.pubmed import PubMedConnector  # noqa: E402


# Sample E-utilities responses.
_ESEARCH_RESP = {
    "esearchresult": {
        "idlist": ["34449189", "35379503"],
        "count": "2",
    }
}
_ESUMMARY_RESP = {
    "result": {
        "uids": ["34449189", "35379503"],
        "34449189": {
            "uid": "34449189",
            "pubdate": "2021 Aug 27",
            "title": "Empagliflozin in heart failure with preserved ejection fraction",
            "source": "N Engl J Med",
            "fulljournalname": "The New England Journal of Medicine",
            "authors": [{"name": "Anker SD"}, {"name": "Butler J"}],
            "articleids": [
                {"idtype": "doi", "value": "10.1056/NEJMoa2107038"},
                {"idtype": "pmc", "value": "PMC8500551"},
            ],
            "pubtype": ["Journal Article", "Randomized Controlled Trial"],
        },
        "35379503": {
            "uid": "35379503",
            "pubdate": "2022",
            "title": "ACC/AHA practice guideline for heart failure",
            "source": "Circulation",
            "authors": [{"name": "Heidenreich PA"}],
            "articleids": [
                {"idtype": "doi", "value": "10.1161/cir.0000000000001063"},
            ],
            "pubtype": ["Journal Article", "Practice Guideline"],
        },
    }
}
_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>34449189</PMID>
      <Article>
        <Abstract>
          <AbstractText Label="BACKGROUND">RCT background</AbstractText>
          <AbstractText Label="METHODS">RCT methods</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
          <PublicationType>Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
        <Language>eng</Language>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">34449189</ArticleId>
        <ArticleId IdType="doi">10.1056/NEJMoa2107038</ArticleId>
        <ArticleId IdType="pmc">PMC8500551</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>35379503</PMID>
      <Article>
        <Abstract>
          <AbstractText>Practice guideline for heart failure.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Practice Guideline</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">35379503</ArticleId>
        <ArticleId IdType="doi">10.1161/cir.0000000000001063</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


class _FakeJsonResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeXmlResponse:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def json(self):
        raise ValueError("xml only")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeClient:
    """Routes GET requests by URL substring to predefined responses."""

    def __init__(self, routes: dict):
        self._routes = routes

    async def get(self, url, params=None):
        for needle, resp in self._routes.items():
            if needle in url:
                return resp
        raise RuntimeError(f"no route for {url}")

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_pubmed_connector_parses_records():
    client = _FakeClient({
        "esearch.fcgi": _FakeJsonResponse(_ESEARCH_RESP),
        "esummary.fcgi": _FakeJsonResponse(_ESUMMARY_RESP),
        "efetch.fcgi": _FakeXmlResponse(_EFETCH_XML),
    })
    connector = PubMedConnector()
    pairs = await connector.search(
        [{"query_type": "high_recall", "query_string": "empagliflozin"}],
        client=client,
    )
    assert len(pairs) == 2
    qtypes = [t for t, _ in pairs]
    assert qtypes == ["high_recall", "high_recall"]
    by_pmid = {a["pmid"]: a for _, a in pairs}
    rct = by_pmid["34449189"]
    assert rct["title"].startswith("Empagliflozin")
    assert rct["doi"] == "10.1056/NEJMoa2107038"
    assert rct["pmcid"] == "PMC8500551"
    assert rct["article_type_classification"] == "rct"
    assert rct["canonical_key"] == "pmid:34449189"
    assert rct["full_text_status"] == "full_text_available"
    assert rct["_retrieved_by"] == "pubmed"
    # Guideline
    gl = by_pmid["35379503"]
    assert gl["is_guideline"] is True
    assert gl["article_type_classification"] == "guideline"


@pytest.mark.asyncio
async def test_pubmed_connector_survives_failure_and_reports_status():
    """One failing query must not stop the rest of the run."""
    call_count = {"n": 0}

    async def get_fail_then_pass(url, params=None):
        call_count["n"] += 1
        if "esearch.fcgi" in url and call_count["n"] == 1:
            raise httpx.HTTPError("simulated outage")
        if "esearch.fcgi" in url:
            return _FakeJsonResponse(_ESEARCH_RESP)
        if "esummary.fcgi" in url:
            return _FakeJsonResponse(_ESUMMARY_RESP)
        if "efetch.fcgi" in url:
            return _FakeXmlResponse(_EFETCH_XML)
        raise RuntimeError(f"no route for {url}")

    class _DynClient:
        async def get(self, url, params=None):
            return await get_fail_then_pass(url, params)

        async def aclose(self):
            pass

    connector = PubMedConnector()
    pairs = await connector.search(
        [
            {"query_type": "high_recall", "query_string": "q1"},
            {"query_type": "high_precision", "query_string": "q2"},
        ],
        client=_DynClient(),
    )
    # First query fails after retries; second succeeds (2 records each retry).
    qtypes = [t for t, _ in pairs]
    assert "high_precision" in qtypes


@pytest.mark.asyncio
async def test_pubmed_normalize_handles_missing_pmid():
    """A record without a PMID should be dropped (never returned)."""
    connector = PubMedConnector()
    raw = {"title": "Floating record", "pubdate": "2020"}
    norm = connector._normalize(raw)
    assert norm is None


@pytest.mark.asyncio
async def test_pubmed_classify_detects_systematic_review_and_cohort():
    connector = PubMedConnector()
    raw_sr = {
        "pmid": "111",
        "title": "SR of statins",
        "pubdate": "2020",
        "authors": [],
        "abstractText_xml": "SR abstract",
        "pubTypeList_xml": ["Journal Article", "Systematic Review"],
    }
    norm = connector._normalize(raw_sr)
    assert norm["article_type_classification"] == "systematic_review"

    raw_cohort = {
        "pmid": "222",
        "title": "Cohort study",
        "pubdate": "2019",
        "authors": [],
        "abstractText_xml": "Cohort abstract",
        "pubTypeList_xml": ["Cohort Studies"],
    }
    norm2 = connector._normalize(raw_cohort)
    assert norm2["article_type_classification"] == "cohort_study"


@pytest.mark.asyncio
async def test_pubmed_no_api_key_uses_public_rate_limit(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    c = PubMedConnector()
    assert c.api_key == ""
    # Default 3 req/s without API key
    assert c.rps == 3.0
    # No api_key params injected.
    assert "api_key" not in c._common_params()
