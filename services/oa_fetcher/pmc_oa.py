"""PMC OA Web Service + BioC fetcher.

Two-stage:
  1. PMC OA Web Service `oa.fcgi?id={PMCID}` confirms whether the article is
     in the PMC OA subset (i.e. permissively licensed full text is allowed
     to be retrieved). The XML response also typically carries a `license`
     attribute on the `<record>` element.
  2. PMC BioC `BioC_xml/{PMCID}/unicode` returns clean section-marked XML.

We never call BioC for articles the OA service refuses. We never scrape
publisher pages.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from xml.etree import ElementTree as ET

import httpx

from .section_parser import (
    extract_bioc_license,
    parse_bioc,
)

logger = logging.getLogger("clinevidence.oa_fetcher.pmc_oa")


_OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
_BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmcid}/unicode"


def _normalize_pmcid(raw: str) -> str:
    s = (raw or "").strip().upper()
    if not s:
        return s
    if not s.startswith("PMC"):
        s = "PMC" + s
    return s


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout_s: float,
    params: Optional[dict] = None,
) -> tuple[Optional[httpx.Response], Optional[str]]:
    last_err: Optional[str] = None
    for attempt in range(3):
        try:
            r = await client.get(url, params=params, timeout=timeout_s)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            last_err = f"network_{type(exc).__name__}"
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            return None, last_err
        if r.status_code in (429, 500, 502, 503, 504):
            last_err = f"http_{r.status_code}"
            if attempt < 2:
                await asyncio.sleep(0.6 * (2 ** attempt))
                continue
            return r, last_err
        return r, None
    return None, last_err


async def fetch(
    *,
    pmcid: str,
    timeout_s: float = 20.0,
    client: Optional[httpx.AsyncClient] = None,
) -> dict:
    """Fetch + parse PMC BioC for an OA article.

    Returns:
      {
        "status": "fetched|not_oa|failed",
        "sections": [...],
        "license_raw": str | None,
        "error_code": str | None,
        "http_status": int | None,
      }
    """
    pmcid = _normalize_pmcid(pmcid)
    if not pmcid:
        return {"status": "failed", "sections": [], "license_raw": None,
                "error_code": "no_pmcid", "http_status": None}

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout_s, headers={
            "User-Agent": "ClinEvidence/4.0 (mailto:research@clinevidence.local)",
            "Accept": "application/xml,text/xml",
        })
    try:
        # ---- Step 1: PMC OA Web Service ----
        r, err = await _get_with_retry(
            client, _OA_SERVICE, timeout_s=timeout_s, params={"id": pmcid},
        )
        if r is None:
            return {"status": "failed", "sections": [], "license_raw": None,
                    "error_code": err or "oa_service_unreachable", "http_status": None}
        if r.status_code != 200:
            return {"status": "failed", "sections": [], "license_raw": None,
                    "error_code": f"oa_service_http_{r.status_code}", "http_status": r.status_code}

        license_raw: Optional[str] = None
        try:
            oa_root = ET.fromstring(r.content)
            err_el = oa_root.find(".//error")
            if err_el is not None and (err_el.get("code") or "").lower().startswith("idis"):
                return {"status": "not_oa", "sections": [], "license_raw": None,
                        "error_code": err_el.get("code") or "idIsNotOpenAccess",
                        "http_status": 200}
            rec = oa_root.find(".//record")
            if rec is not None:
                license_raw = rec.get("license") or rec.get("citation")
        except ET.ParseError:
            return {"status": "failed", "sections": [], "license_raw": None,
                    "error_code": "oa_service_parse_error", "http_status": r.status_code}

        # ---- Step 2: BioC ----
        bioc_url = _BIOC.format(pmcid=pmcid)
        r2, err2 = await _get_with_retry(client, bioc_url, timeout_s=timeout_s)
        if r2 is None:
            return {"status": "failed", "sections": [], "license_raw": license_raw,
                    "error_code": err2 or "bioc_unreachable", "http_status": None}
        if r2.status_code != 200:
            return {"status": "failed", "sections": [], "license_raw": license_raw,
                    "error_code": f"bioc_http_{r2.status_code}", "http_status": r2.status_code}
        try:
            bioc_root = ET.fromstring(r2.content)
        except ET.ParseError:
            return {"status": "failed", "sections": [], "license_raw": license_raw,
                    "error_code": "bioc_parse_error", "http_status": 200}
        bioc_license = extract_bioc_license(bioc_root)
        # Prefer BioC license if present; else fall back to OA service value.
        final_license = bioc_license or license_raw
        sections = parse_bioc(bioc_root)
        if not sections:
            return {"status": "failed", "sections": [], "license_raw": final_license,
                    "error_code": "empty_sections", "http_status": 200}
        return {
            "status": "fetched",
            "sections": [s.model_dump() for s in sections],
            "license_raw": final_license,
            "error_code": None,
            "http_status": 200,
        }
    finally:
        if own_client:
            await client.aclose()
