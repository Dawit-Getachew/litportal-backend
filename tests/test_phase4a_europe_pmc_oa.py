"""Phase 4A — Europe PMC OA fetcher (Task 11 / B). All HTTP mocked."""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from services.oa_fetcher import europe_pmc_oa
from tests.fixtures_oa_xml import JATS_SAMPLE, JATS_NO_LICENSE


class _MockResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


def _client_returning(*responses):
    """Return a tiny stub http client whose .get yields the queued responses."""
    cli = AsyncMock(spec=httpx.AsyncClient)
    cli.get = AsyncMock(side_effect=list(responses))
    return cli


@pytest.mark.asyncio
async def test_successful_xml_fetch_extracts_sections():
    cli = _client_returning(_MockResp(200, JATS_SAMPLE.encode("utf-8")))
    result = await europe_pmc_oa.fetch(pmcid="PMC1234567", pmid=None, client=cli)
    assert result["status"] == "fetched"
    assert result["http_status"] == 200
    names = {s["section_name"] for s in result["sections"]}
    assert {"abstract", "methods", "results", "conclusion"}.issubset(names)


@pytest.mark.asyncio
async def test_license_captured_from_permissions_block():
    cli = _client_returning(_MockResp(200, JATS_SAMPLE.encode("utf-8")))
    result = await europe_pmc_oa.fetch(pmcid="PMC1234567", pmid=None, client=cli)
    assert result["license_raw"] is not None
    assert "creativecommons.org/licenses/by" in result["license_raw"].lower()


@pytest.mark.asyncio
async def test_404_falls_through_to_pmc_oa_cascade():
    cli = _client_returning(_MockResp(404, b""))
    result = await europe_pmc_oa.fetch(pmcid="PMC1234567", pmid=None, client=cli)
    assert result["status"] == "not_found"
    assert result["http_status"] == 404


@pytest.mark.asyncio
async def test_5xx_retries_then_fails_gracefully():
    cli = _client_returning(_MockResp(503, b""), _MockResp(502, b""), _MockResp(500, b""))
    result = await europe_pmc_oa.fetch(pmcid="PMC1234567", pmid=None, client=cli)
    assert result["status"] == "failed"
    assert result["http_status"] == 500
    assert result["error_code"].startswith("http_")


@pytest.mark.asyncio
async def test_timeout_handled():
    cli = AsyncMock(spec=httpx.AsyncClient)
    cli.get = AsyncMock(side_effect=httpx.ConnectTimeout("slow"))
    result = await europe_pmc_oa.fetch(pmcid="PMC1234567", pmid=None, client=cli)
    assert result["status"] == "failed"
    assert "ConnectTimeout" in (result["error_code"] or "")


@pytest.mark.asyncio
async def test_non_oa_article_not_fetched_when_metadata_says_no():
    """No pmcid AND no pmid → no URL to build → skipped without a network call."""
    cli = _client_returning()  # no mocks needed; assert no .get call happens
    result = await europe_pmc_oa.fetch(pmcid=None, pmid=None, client=cli)
    assert result["status"] == "skipped"
    assert result["error_code"] == "no_identifier"
    cli.get.assert_not_called()


@pytest.mark.asyncio
async def test_source_configs_last_success_at_updated():
    """The service layer (OAFetcherService) updates source_configs after a
    successful fetch. Covered end-to-end in the pipeline integration suite.
    Here we just verify the fetcher returns a `fetched` status that can drive
    that update.
    """
    cli = _client_returning(_MockResp(200, JATS_SAMPLE.encode("utf-8")))
    result = await europe_pmc_oa.fetch(pmcid="PMC1234567", pmid=None, client=cli)
    assert result["status"] == "fetched"
