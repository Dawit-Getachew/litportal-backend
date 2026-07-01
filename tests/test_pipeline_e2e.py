"""Phase 2A — Europe PMC connector + end-to-end pipeline w/ mocked httpx."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.literature_retrieval import LiteratureRetrievalService  # noqa: E402


# Canned EuropePMC response shape (one RCT).
_OK_RESPONSE = {
    "resultList": {
        "result": [
            {
                "id": "34449189",
                "source": "MED",
                "title": "Empagliflozin in heart failure with preserved ejection fraction",
                "abstractText": "RCT of empagliflozin vs placebo in HFpEF (EMPEROR-Preserved).",
                "journalInfo": {"journal": {"title": "NEJM"}, "yearOfPublication": "2021"},
                "pubYear": "2021",
                "firstPublicationDate": "2021-08-27",
                "doi": "10.1056/NEJMoa2107038",
                "pmid": "34449189",
                "pmcid": "PMC8500551",
                "pubTypeList": {"pubType": ["Journal Article", "Randomized Controlled Trial"]},
                "authorList": {"author": [{"fullName": "Anker SD", "lastName": "Anker"}]},
                "fullTextUrlList": {"fullTextUrl": [
                    {"url": "https://www.nejm.org/doi/full/10.1056/NEJMoa2107038", "availabilityCode": "OA"}
                ]},
                "inEPMC": "Y",
            }
        ]
    }
}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeClient:
    def __init__(self, response_or_seq):
        # accepts either a single _FakeResponse, a list to iterate, or a callable
        self._calls = 0
        self._resp = response_or_seq

    async def get(self, url, params=None):
        self._calls += 1
        if isinstance(self._resp, list):
            return self._resp[(self._calls - 1) % len(self._resp)]
        if callable(self._resp):
            return self._resp(self._calls, url, params)
        return self._resp

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_europe_pmc_connector_parses_ok_response():
    fake = _FakeClient(_FakeResponse(_OK_RESPONSE))
    pairs = await LiteratureRetrievalService().search_europe_pmc(
        [{"query_type": "high_recall", "query_string": "anything NOT SRC:PPR"}],
        client=fake,
    )
    assert len(pairs) == 1
    qtype, rec = pairs[0]
    assert qtype == "high_recall"
    assert rec["pmid"] == "34449189"


@pytest.mark.asyncio
async def test_europe_pmc_connector_survives_one_failing_query():
    def _resp(call, url, params):
        if call == 1:
            raise httpx.HTTPError("simulated outage")
        return _FakeResponse(_OK_RESPONSE)
    fake = _FakeClient(_resp)
    pairs = await LiteratureRetrievalService().search_europe_pmc(
        [
            {"query_type": "high_recall", "query_string": "q1"},
            {"query_type": "high_precision", "query_string": "q2"},
        ],
        client=fake,
    )
    # First query fails (after retries), second succeeds
    qtypes = [t for t, _ in pairs]
    assert "high_precision" in qtypes


# -------------- end-to-end pipeline (mock httpx) --------------
@pytest.mark.asyncio
async def test_pipeline_e2e_ranks_articles(client):
    # Make sure the test user exists & has token
    r = await client.post(
        "/api/auth/signup",
        json={"email": "e2e@test.com", "password": "Pass1234567!", "name": "E2E"},
    )
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Patch the retrieval to return canned data — pipeline runs in-process.
    async def _fake_retrieve(source, queries, **kwargs):
        return [
            ("high_precision", _OK_RESPONSE["resultList"]["result"][0]),
            ("high_recall", {
                "id": "999",
                "source": "PPR",  # preprint — should be excluded
                "title": "A preprint about empagliflozin",
                "abstractText": "preprint abstract",
                "pubYear": "2024",
                "pubTypeList": {"pubType": ["Preprint"]},
                "authorList": {"author": []},
                "journalInfo": {"journal": {"title": "bioRxiv"}, "yearOfPublication": "2024"},
                "fullTextUrlList": {"fullTextUrl": []},
            }),
            ("guideline_focused", {
                "id": "555",
                "source": "MED",
                "title": "ACC/AHA practice guideline for heart failure",
                "abstractText": "Practice guideline.",
                "pubYear": "2022",
                "pubTypeList": {"pubType": ["Practice Guideline"]},
                "authorList": {"author": [{"fullName": "Heidenreich PA", "lastName": "Heidenreich"}]},
                "journalInfo": {"journal": {"title": "Circulation"}, "yearOfPublication": "2022"},
                "doi": "10.1161/cir.0000000000001063",
                "pmid": "35379503",
                "fullTextUrlList": {"fullTextUrl": []},
            }),
        ]

    with patch("services.pipeline.literature_retrieval.retrieve", new=_fake_retrieve):
        r2 = await client.post(
            "/api/search-sessions/",
            json={"question": "Is empagliflozin compared to placebo effective in HFpEF?"},
            headers=headers,
        )
        assert r2.status_code == 200
        body = r2.json()
        sid = body["id"]
        assert body["status"] == "searching"

        # Wait for the background pipeline to complete (poll).
        import asyncio
        for _ in range(150):
            await asyncio.sleep(0.2)
            sr = await client.get(f"/api/search-sessions/{sid}/status", headers=headers)
            if sr.json().get("articles_ready"):
                break
        status = sr.json()
        assert status["status"] == "awaiting_selection", status
        assert status["counts"]["retrieved"] >= 2

    # Full session response includes ranked articles + evaluator
    full = (await client.get(f"/api/search-sessions/{sid}", headers=headers)).json()
    assert full["status"] == "awaiting_selection"
    titles = [a["article"]["title"] for a in full["articles"]]
    # The guideline OR the RCT should be top-ranked; preprint MUST be excluded.
    assert any("practice guideline" in t.lower() or "empagliflozin" in t.lower() for t in titles)
    excluded_titles = [a["article"]["title"] for a in full["excluded_articles"]]
    assert any("preprint" in t.lower() for t in excluded_titles)
    assert full["evaluator_json"] is not None

    # Transparency endpoint
    trans = (await client.get(f"/api/search-sessions/{sid}/transparency", headers=headers)).json()
    assert "europe_pmc" in trans["sources_searched"]
    # Phase 2B defaults: pubmed is enabled out of the box, so the planner
    # emits 4 europe_pmc + 4 pubmed = 8 legacy plans. Phase 5A may add
    # extra concept-block queries on top — assert the legacy 8.
    assert "pubmed" in trans["sources_searched"]
    legacy_families = {"high_recall", "high_precision", "evidence_filtered", "guideline_focused"}
    legacy_qs = [q for q in trans["queries"] if q.get("query_type") in legacy_families]
    assert len(legacy_qs) == 8

    # PATCH article inclusion
    first_aid = full["articles"][0]["article_id"]
    patch_r = await client.patch(
        f"/api/search-sessions/{sid}/articles/{first_aid}",
        json={"user_included": False},
        headers=headers,
    )
    assert patch_r.status_code == 200
    assert patch_r.json()["user_included"] is False


@pytest.mark.asyncio
async def test_phi_block_does_not_trigger_europe_pmc(client):
    """Critical safety test: PHI input must not produce any HTTP call."""
    r = await client.post(
        "/api/auth/signup",
        json={"email": "phipipe@test.com", "password": "Pass1234567!", "name": "PHI"},
    )
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    with patch("services.literature_retrieval.literature_retrieval.retrieve", new=AsyncMock()) as mock_retrieve, \
         patch.object(__import__("services.llm_provider", fromlist=["llm_provider"]).llm_provider,
                      "complete_json", new=AsyncMock(return_value=None)) as mock_llm:
        r2 = await client.post(
            "/api/search-sessions/",
            json={"question": "Mrs. Jane Doe 1234 Main St DOB 01/02/1980 — best HTN tx?"},
            headers=headers,
        )
        assert r2.status_code == 200
        assert r2.json()["status"] == "blocked_phi"
        mock_retrieve.assert_not_called()
        mock_llm.assert_not_called()
