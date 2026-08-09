"""Europe PMC fullTextXML fetcher.

Endpoint: https://www.ebi.ac.uk/europepmc/webservices/rest/{SOURCE}/{ID}/fullTextXML

Returns JATS XML for PMC OA articles (and for some MED-source articles when
Europe PMC has indexed the full text). HTTP 404 means "Europe PMC does not
have full text for this id" — we propagate that as a *cascade* signal,
not a hard failure.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from xml.etree import ElementTree as ET

import httpx

from .section_parser import (
    extract_jats_license,
    parse_jats,
)

logger = logging.getLogger("clinevidence.oa_fetcher.europe_pmc_oa")

_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePMCResult(dict):
    """Plain dict result so we can audit-log without exposing raw XML."""


def _build_url(*, pmcid: Optional[str], pmid: Optional[str]) -> Optional[str]:
    if pmcid:
        pmcid_clean = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid.lstrip('Pp').lstrip('Mm').lstrip('Cc')}"
        return f"{_BASE}/PMC/{pmcid_clean}/fullTextXML"
    if pmid:
        return f"{_BASE}/MED/{pmid}/fullTextXML"
    return None


async def fetch(
    *,
    pmcid: Optional[str],
    pmid: Optional[str],
    timeout_s: float = 20.0,
    client: Optional[httpx.AsyncClient] = None,
) -> dict:
    """Fetch + parse Europe PMC fullTextXML.

    Returns:
      {
        "status": "fetched|not_found|failed|skipped",
        "sections": [ParsedSection.model_dump(), ...],
        "license_raw": str | None,
        "error_code": str | None,
        "http_status": int | None,
      }
    """
    url = _build_url(pmcid=pmcid, pmid=pmid)
    if not url:
        return {"status": "skipped", "sections": [], "license_raw": None,
                "error_code": "no_identifier", "http_status": None}

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout_s, headers={
            "User-Agent": "ClinEvidence/4.0 (mailto:research@clinevidence.local)",
            "Accept": "application/xml,text/xml",
        })
    try:
        last_err = None
        for attempt in range(3):
            try:
                r = await client.get(url, timeout=timeout_s)
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_err = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return {"status": "failed", "sections": [], "license_raw": None,
                        "error_code": f"network_{type(exc).__name__}", "http_status": None}
            if r.status_code == 404:
                return {"status": "not_found", "sections": [], "license_raw": None,
                        "error_code": None, "http_status": 404}
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"http_{r.status_code}"
                if attempt < 2:
                    await asyncio.sleep(0.6 * (2 ** attempt))
                    continue
                return {"status": "failed", "sections": [], "license_raw": None,
                        "error_code": f"http_{r.status_code}", "http_status": r.status_code}
            if r.status_code != 200:
                return {"status": "failed", "sections": [], "license_raw": None,
                        "error_code": f"http_{r.status_code}", "http_status": r.status_code}
            # Parse JATS XML.
            try:
                root = ET.fromstring(r.content)
            except ET.ParseError:
                return {"status": "failed", "sections": [], "license_raw": None,
                        "error_code": "xml_parse_error", "http_status": 200}
            license_raw = extract_jats_license(root)
            sections = parse_jats(root)
            if not sections:
                return {"status": "failed", "sections": [], "license_raw": license_raw,
                        "error_code": "empty_sections", "http_status": 200}
            return {
                "status": "fetched",
                "sections": [s.model_dump() for s in sections],
                "license_raw": license_raw,
                "error_code": None,
                "http_status": 200,
            }
        return {"status": "failed", "sections": [], "license_raw": None,
                "error_code": f"retry_exhausted:{last_err}", "http_status": None}
    finally:
        if own_client:
            await client.aclose()
