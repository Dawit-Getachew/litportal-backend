"""Phase 4A — PMC OA Web Service + BioC fetcher (Task 11 / C)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from services.oa_fetcher import pmc_oa
from tests.fixtures_oa_xml import (
    BIOC_SAMPLE, PMC_OA_SVC_NOT_OA, PMC_OA_SVC_OK, PMC_OA_SVC_UNKNOWN_LIC,
)


class _MockResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


def _client_returning(*responses):
    cli = AsyncMock(spec=httpx.AsyncClient)
    cli.get = AsyncMock(side_effect=list(responses))
    return cli


@pytest.mark.asyncio
async def test_pmc_oa_web_service_id_not_open_access_skipped():
    cli = _client_returning(_MockResp(200, PMC_OA_SVC_NOT_OA.encode("utf-8")))
    result = await pmc_oa.fetch(pmcid="PMC9999999", client=cli)
    assert result["status"] == "not_oa"
    # Must NOT issue a follow-up BioC fetch.
    assert cli.get.call_count == 1


@pytest.mark.asyncio
async def test_pmc_oa_web_service_returns_resources_then_bioc_succeeds():
    cli = _client_returning(
        _MockResp(200, PMC_OA_SVC_OK.encode("utf-8")),
        _MockResp(200, BIOC_SAMPLE.encode("utf-8")),
    )
    result = await pmc_oa.fetch(pmcid="PMC1234567", client=cli)
    assert result["status"] == "fetched"
    assert cli.get.call_count == 2
    assert any(s["section_name"] == "abstract" for s in result["sections"])


@pytest.mark.asyncio
async def test_bioc_parsed_into_sections():
    cli = _client_returning(
        _MockResp(200, PMC_OA_SVC_OK.encode("utf-8")),
        _MockResp(200, BIOC_SAMPLE.encode("utf-8")),
    )
    result = await pmc_oa.fetch(pmcid="PMC1234567", client=cli)
    names = {s["section_name"] for s in result["sections"]}
    assert {"abstract", "methods", "results", "discussion", "conclusion"}.issubset(names)


@pytest.mark.asyncio
async def test_license_captured_from_oa_service():
    cli = _client_returning(
        _MockResp(200, PMC_OA_SVC_OK.encode("utf-8")),
        _MockResp(200, BIOC_SAMPLE.encode("utf-8")),
    )
    result = await pmc_oa.fetch(pmcid="PMC1234567", client=cli)
    # The BioC license <infon> wins when present, but the OA service's value
    # would also pass through (both are CC BY in the fixtures).
    assert result["license_raw"] is not None
    assert "by" in result["license_raw"].lower()


@pytest.mark.asyncio
async def test_unknown_license_skipped():
    """OA service returned no license metadata AND BioC has no license infon."""
    BIOC_NO_LIC = """<?xml version="1.0"?>
    <collection><document><id>PMC2</id>
      <passage><infon key="type">abstract</infon><infon key="section_type">abstract</infon>
        <text>Background. Methods. Results. Conclusion. This abstract has been padded so its length exceeds the minimum chunk filter threshold.</text>
      </passage>
    </document></collection>"""
    cli = _client_returning(
        _MockResp(200, PMC_OA_SVC_UNKNOWN_LIC.encode("utf-8")),
        _MockResp(200, BIOC_NO_LIC.encode("utf-8")),
    )
    result = await pmc_oa.fetch(pmcid="PMC2", client=cli)
    # Fetcher returns the text + license_raw=None; the gate (license_gate) is
    # what classifies as "unknown" and refuses ingest. Here we only assert
    # the fetcher emits None for license_raw.
    assert result["status"] == "fetched"
    assert result["license_raw"] is None


@pytest.mark.asyncio
async def test_bioc_download_failure_handled():
    cli = _client_returning(
        _MockResp(200, PMC_OA_SVC_OK.encode("utf-8")),
        _MockResp(500, b""),  # repeated 500s
        _MockResp(500, b""),
        _MockResp(500, b""),
    )
    result = await pmc_oa.fetch(pmcid="PMC1234567", client=cli)
    assert result["status"] == "failed"
    assert result["error_code"].startswith("bioc_http_") or result["error_code"].startswith("http_")


@pytest.mark.asyncio
async def test_source_configs_last_error_at_updated_on_failure():
    """The service layer updates source_configs after a fetch attempt. The
    end-to-end behavior is covered in test_phase4a_pipeline_integration."""
    cli = _client_returning(_MockResp(404, b""))
    result = await pmc_oa.fetch(pmcid="PMC1234567", client=cli)
    assert result["status"] == "failed"
