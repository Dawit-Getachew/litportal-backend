"""Phase K5.2A — MyHealthfinder (HHS / ODPHP) official-source connector.

DETERMINISTIC HTTP. NO API KEY. NO secret. Metadata + link-out ONLY.
NO full-text ingestion. NO AI ingestion. NO scraping. NO LLM.

Source: U.S. Department of Health & Human Services, Office of Disease
Prevention and Health Promotion (ODPHP) — MyHealthfinder API v3
(``https://odphp.health.gov/myhealthfinder/api/v3/topicsearch.json``).
Public-domain U.S. government work; no key required. Returns evidence-based
prevention / screening recommendation TOPICS with a title, a link-out URL
(``AccessibleVersion``), and a short description. We surface metadata +
link-out only and NEVER ingest the page body.

Closed-set provenance: every item is tagged
``provenance="official_source_metadata_link_out"`` and
``source="myhealthfinder_hhs"`` — explicitly DISTINCT from the
PubMed-indexed guideline lane (K5.1).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger("clinevidence.myhealthfinder")

MYHF_SOURCE_ID = "myhealthfinder_hhs"
MYHF_ORGANIZATION = "HHS / ODPHP (MyHealthfinder)"
MYHF_PROVENANCE = "official_source_metadata_link_out"
MYHF_BASE_URL = "https://odphp.health.gov/myhealthfinder/api/v3/topicsearch.json"

# Closed-set connector status codes (no free text / no raw payloads).
MYHF_STATUS = (
    "ok", "empty", "timeout", "http_error", "transient_unavailable",
    "parse_error", "no_keyword",
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MAX_SNIPPET = 320
_MAX_ITEMS = 8


def _clean(text: Any, *, limit: int = _MAX_SNIPPET) -> str:
    s = "" if text is None else str(text)
    s = _TAG_RE.sub(" ", s)           # strip any HTML tags
    s = _WS_RE.sub(" ", s).strip()
    if len(s) > limit:
        s = s[:limit].rstrip() + "…"
    return s


def build_keyword(*, primary_anchor: Optional[str], question: Optional[str]) -> str:
    """Deterministic keyword: prefer the primary clinical anchor concept;
    else a trimmed clinical phrase from the (already PHI-screened) question.
    Pure / no network."""
    a = (primary_anchor or "").strip()
    if a:
        return a[:80]
    q = (question or "").strip()
    # Drop common interrogative lead-ins; keep the topical tail.
    q = re.sub(
        r"^(?:what|how|when|is|are|should|does|do|can)\b[\s\w]{0,24}?\b"
        r"(?:guidelines?|recommendation|screening|prevention|for|about|of)\b",
        "", q, flags=re.IGNORECASE,
    ).strip(" ?.")
    return (q or (question or "").strip(" ?."))[:80]


def _result(status: str, items: list[dict[str, Any]], keyword: str) -> dict[str, Any]:
    return {
        "source": MYHF_SOURCE_ID,
        "organization": MYHF_ORGANIZATION,
        "provenance": MYHF_PROVENANCE,
        "status": status,
        "keyword": keyword,
        "ai_ingestion": False,           # invariant — metadata/link-out only
        "results": items,
    }


class MyHealthfinderConnector:
    """Deterministic no-key official-source connector. Metadata/link-out only."""

    def __init__(self, *, base_url: str = MYHF_BASE_URL, timeout: float = 8.0):
        self.base_url = base_url
        self.timeout = timeout

    async def search(
        self, keyword: str, *, lang: str = "en", limit: int = _MAX_ITEMS,
    ) -> dict[str, Any]:
        kw = (keyword or "").strip()
        if not kw:
            return _result("no_keyword", [], kw)
        params = {"keyword": kw, "lang": lang}
        # 1 retry on transient network/5xx.
        last_status = "transient_unavailable"
        for attempt in (1, 2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.get(
                        self.base_url, params=params,
                        headers={"Accept": "application/json"},
                    )
                if resp.status_code >= 500:
                    last_status = "transient_unavailable"
                    continue
                if resp.status_code != 200:
                    return _result("http_error", [], kw)
                data = resp.json()
                items = self._parse(data, limit=limit)
                return _result("ok" if items else "empty", items, kw)
            except httpx.TimeoutException:
                last_status = "timeout"
            except httpx.HTTPError:
                last_status = "transient_unavailable"
            except (ValueError, KeyError, TypeError):
                return _result("parse_error", [], kw)
            except Exception:  # noqa: BLE001 — never raise into the request path
                logger.exception("myhealthfinder unexpected error")
                return _result("transient_unavailable", [], kw)
        return _result(last_status, [], kw)

    @staticmethod
    def _parse(data: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        result = (data or {}).get("Result") or {}
        resources = (result.get("Resources") or {}).get("Resource") or []
        out: list[dict[str, Any]] = []
        for r in resources[: max(0, limit)]:
            if not isinstance(r, dict):
                continue
            title = _clean(r.get("Title") or r.get("MyHFTitle"), limit=160)
            url = str(r.get("AccessibleVersion") or r.get("HealthfinderUrl") or "").strip()
            if not title or not url:
                continue
            out.append({
                "title": title,
                "url": url,                         # link-out only
                "snippet": _clean(r.get("MyHFDescription")),
                "categories": _clean(r.get("Categories"), limit=120),
                "source": MYHF_SOURCE_ID,
                "organization": MYHF_ORGANIZATION,
                "provenance": MYHF_PROVENANCE,
                "ai_ingestion_eligible": False,     # invariant
                "full_text_available": False,       # invariant — no ingestion
            })
        return out


myhealthfinder_connector = MyHealthfinderConnector()


__all__ = [
    "MYHF_SOURCE_ID",
    "MYHF_ORGANIZATION",
    "MYHF_PROVENANCE",
    "MYHF_STATUS",
    "MyHealthfinderConnector",
    "myhealthfinder_connector",
    "build_keyword",
]
