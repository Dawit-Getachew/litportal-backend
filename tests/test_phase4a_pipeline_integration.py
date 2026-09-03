"""Phase 4A — pipeline integration + no-LLM-leak (Task 11 / F)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from tests._phase3_helpers import (
    signup, make_packet_session, good_apixaban_articles, patch_llm_configured,
)
from tests.fixtures_oa_xml import JATS_SAMPLE


# Phase 6F closeout (IMPL-4) — pin to the ``live_api`` xdist group so
# this file's TestClient + MongoDB startup-event setup cannot race
# against the live-API regression files under ``pytest -n auto``.
pytestmark = pytest.mark.xdist_group(name="live_api")


class _MockResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


def _good_jats_client():
    cli = AsyncMock(spec=httpx.AsyncClient)
    cli.get = AsyncMock(return_value=_MockResp(200, JATS_SAMPLE.encode("utf-8")))
    return cli


@pytest.mark.asyncio
async def test_prepare_evidence_packet_fetches_missing_selected(client, monkeypatch):
    """Inline OA fetch runs on prepare-evidence-packet for selected articles
    that don't yet have OA extracts."""
    monkeypatch.setenv("OA_FETCHER_ENABLED", "true")
    t, _ = await signup(client, "pipe1@test.com")
    h = {"Authorization": f"Bearer {t}"}
    # Give every article a PMCID so eligibility passes.
    arts = good_apixaban_articles()
    for i, a in enumerate(arts):
        a["pmcid"] = f"PMC100{i}"
    sid, _, article_ids = await make_packet_session(client, h, articles=arts)
    # Patch oa_fetcher to record calls without going to network.
    from services.oa_fetcher import oa_fetcher as oa
    seen = {"calls": []}

    async def fake(*, article_id, session_id, user_id, force_refresh=False, client=None):
        seen["calls"].append(article_id)
        return {
            "article_id": article_id, "status": "skipped_ineligible",
            "source": None, "license": None, "license_category": None,
            "can_ingest_for_summary": False, "sections_created": 0,
            "token_count": 0, "evidence_text_level": None,
            "error_code": "no_oa_signal",
        }
    monkeypatch.setattr(oa, "fetch_for_article", fake)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    assert r.status_code == 200
    # The packet builder should have invoked the fetcher for each selected
    # eligible article.
    assert set(seen["calls"]) == set(article_ids)


@pytest.mark.asyncio
async def test_oa_fetch_failure_falls_back_to_abstract_in_packet(client, monkeypatch):
    monkeypatch.setenv("OA_FETCHER_ENABLED", "true")
    t, _ = await signup(client, "pipe2@test.com")
    h = {"Authorization": f"Bearer {t}"}
    arts = good_apixaban_articles()
    for i, a in enumerate(arts):
        a["pmcid"] = f"PMC200{i}"
    sid, _, article_ids = await make_packet_session(client, h, articles=arts)
    from services.oa_fetcher import oa_fetcher as oa
    async def fail_fetch(**kwargs):
        return {
            "article_id": kwargs["article_id"], "status": "failed",
            "source": "europe_pmc_oa", "license": None, "license_category": "unknown",
            "can_ingest_for_summary": False, "sections_created": 0,
            "token_count": 0, "evidence_text_level": None,
            "error_code": "europe_pmc_failed",
        }
    monkeypatch.setattr(oa, "fetch_for_article", fail_fetch)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    body = r.json()
    # Should still be ready (abstracts available).
    assert body["ready_for_summary"] is True
    assert body["failed_oa_fetch_count"] >= 1
    assert isinstance(body["oa_fetch_warnings"], list)
    assert len(body["oa_fetch_warnings"]) >= 1


@pytest.mark.asyncio
async def test_evidence_packet_counts_include_failed_oa_fetch_count(client, monkeypatch):
    monkeypatch.setenv("OA_FETCHER_ENABLED", "true")
    t, _ = await signup(client, "pipe3@test.com")
    h = {"Authorization": f"Bearer {t}"}
    arts = good_apixaban_articles()
    for i, a in enumerate(arts):
        a["pmcid"] = f"PMC300{i}"
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    body = r.json()
    assert "failed_oa_fetch_count" in body
    assert "oa_fetch_warnings" in body
    assert "oa_full_text_count" in body
    assert "user_uploaded_full_text_count" in body
    assert "abstract_only_count" in body
    assert "metadata_only_count" in body


@pytest.mark.asyncio
async def test_pipeline_does_not_error_if_oa_fetcher_disabled(client, monkeypatch):
    monkeypatch.setenv("OA_FETCHER_ENABLED", "false")
    t, _ = await signup(client, "pipe4@test.com")
    h = {"Authorization": f"Bearer {t}"}
    arts = good_apixaban_articles()
    for i, a in enumerate(arts):
        a["pmcid"] = f"PMC400{i}"
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["failed_oa_fetch_count"] == 0
    assert body["oa_fetch_warnings"] == []


@pytest.mark.asyncio
async def test_oa_fetch_does_not_call_llm(client, monkeypatch):
    """Whole OA fetch path must never invoke the LLM provider."""
    monkeypatch.setenv("OA_FETCHER_ENABLED", "true")
    t, _ = await signup(client, "pipe5@test.com")
    h = {"Authorization": f"Bearer {t}"}
    arts = good_apixaban_articles()
    for i, a in enumerate(arts):
        a["pmcid"] = f"PMC500{i}"
    sid, _, article_ids = await make_packet_session(client, h, articles=arts)
    # Patch LLMProvider.complete_json to raise on any call.
    llm_mock = AsyncMock(side_effect=AssertionError("LLM must not be called from OA fetcher"))
    with patch_llm_configured() as prov:
        prov.complete_json = llm_mock  # type: ignore[assignment]
        # Patch oa_fetcher.fetch_for_article to just count (network not needed).
        from services.oa_fetcher import oa_fetcher as oa
        async def fake(**kwargs):
            return {
                "article_id": kwargs["article_id"], "status": "skipped_ineligible",
                "source": None, "license": None, "license_category": None,
                "can_ingest_for_summary": False, "sections_created": 0,
                "token_count": 0, "evidence_text_level": None,
                "error_code": "no_oa_signal",
            }
        monkeypatch.setattr(oa, "fetch_for_article", fake)
        await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_phi_blocked_session_does_not_call_oa_fetcher(client, monkeypatch):
    """A PHI-blocked clinical question must not reach the OA fetcher."""
    monkeypatch.setenv("OA_FETCHER_ENABLED", "true")
    t, _ = await signup(client, "pipe6@test.com")
    h = {"Authorization": f"Bearer {t}"}
    from services.oa_fetcher import oa_fetcher as oa
    spy = AsyncMock(return_value={
        "article_id": "x", "status": "skipped_ineligible",
        "source": None, "license": None, "license_category": None,
        "can_ingest_for_summary": False, "sections_created": 0,
        "token_count": 0, "evidence_text_level": None,
        "error_code": "no_oa_signal",
    })
    monkeypatch.setattr(oa, "fetch_for_article", spy)
    monkeypatch.setattr(oa, "fetch_for_articles", AsyncMock(return_value={
        "results": [], "summary": {"fetched": 0}
    }))
    r = await client.post("/api/search-sessions/", headers=h, json={
        "question": "Patient MRN 12345678 email a@b.com — best HTN tx?",
    })
    body = r.json()
    assert body["status"] == "blocked_phi"
    spy.assert_not_called()
    oa.fetch_for_articles.assert_not_called()


@pytest.mark.asyncio
async def test_low_relevance_articles_not_auto_fetched(client, monkeypatch):
    """The packet integration only fetches selected (user_included=true)
    articles — unselected ones with `included_by_default=false` (or toggled
    off) MUST NOT be fetched."""
    monkeypatch.setenv("OA_FETCHER_ENABLED", "true")
    t, _ = await signup(client, "pipe7@test.com")
    h = {"Authorization": f"Bearer {t}"}
    arts = good_apixaban_articles()
    for i, a in enumerate(arts):
        a["pmcid"] = f"PMC700{i}"
    # Mark the last article as not user-included.
    arts[2]["user_included"] = False
    arts[2]["included_by_default"] = False
    sid, _, article_ids = await make_packet_session(client, h, articles=arts)
    from services.oa_fetcher import oa_fetcher as oa
    seen = []
    async def fake(**kwargs):
        seen.append(kwargs["article_id"])
        return {
            "article_id": kwargs["article_id"], "status": "skipped_ineligible",
            "source": None, "license": None, "license_category": None,
            "can_ingest_for_summary": False, "sections_created": 0,
            "token_count": 0, "evidence_text_level": None,
            "error_code": "no_oa_signal",
        }
    monkeypatch.setattr(oa, "fetch_for_article", fake)
    await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    assert article_ids[2] not in seen, "unselected article was fetched"
    assert article_ids[0] in seen
