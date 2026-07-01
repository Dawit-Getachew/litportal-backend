"""OpenAlex enrichment connector (Phase 2B).

Enrichment-only. Never the primary ranking source. Never overwrites
authoritative safety flags. Never promotes abstract_only → full_text_available.

Lookup priority:
  1. /works/doi:<doi>
  2. /works/pmid:<pmid>
  3. /works/pmcid:<pmcid>
  4. title search (only accept similarity >= 0.90 AND year within ±1)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Optional

import httpx

from core.db import Collections, get_db
from services.audit import audit_service

logger = logging.getLogger("clinevidence.openalex")

BASE = "https://api.openalex.org"
DEFAULT_TIMEOUT_S = 10.0
MAX_RETRIES = 2
SIMILARITY_THRESHOLD = 0.90


def _strip_key_from_message(msg: str) -> str:
    return re.sub(r"(api_key=)[^&\s]+", r"\1<redacted>", msg or "")


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()


class OpenAlexConnector:
    def __init__(self) -> None:
        self.mailto = os.environ.get("OPENALEX_MAILTO", "").strip()
        self.api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
        self.max_enrichments = int(os.environ.get("OPENALEX_MAX_ENRICHMENTS_PER_SEARCH", "50"))
        self.timeout_s = float(os.environ.get("OPENALEX_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))

    def _params(self) -> dict[str, str]:
        p: dict[str, str] = {}
        if self.mailto:
            p["mailto"] = self.mailto
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    async def enrich_many(
        self,
        articles: list[dict[str, Any]],
        *,
        client: Optional[httpx.AsyncClient] = None,
        search_session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Mutate `articles` in place with enrichments. Returns stats dict."""
        ranked = sorted(articles, key=lambda a: a.get("_pre_rank_score", 0), reverse=True)
        targets = ranked[: self.max_enrichments]
        owns = client is None
        if owns:
            client = httpx.AsyncClient(timeout=self.timeout_s)
        # B5 — enrich the top-N articles concurrently (bounded). Each ``_apply``
        # mutates a distinct article dict, so concurrency is safe; gather
        # preserves target order, so the aggregated counters and "last error"
        # match the former sequential loop. The lookup HTTP layer already
        # backs off on 429, and a semaphore caps the fan-out.
        max_conc = max(1, int(os.environ.get("OPENALEX_MAX_CONCURRENT_ENRICHMENTS", "5")))
        sem = asyncio.Semaphore(max_conc)

        async def _enrich_one(art: dict[str, Any]) -> tuple[str, Optional[str]]:
            """Returns (outcome, error_message). outcome: applied|ok|error."""
            async with sem:
                try:
                    enrichment = await self._lookup_one(client, art)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("openalex lookup failed: %s", exc)
                    await audit_service.log(
                        "source_query_error",
                        search_session_id=search_session_id,
                        payload={"source": "openalex", "truncated_error": _strip_key_from_message(str(exc))[:300]},
                    )
                    return "error", _strip_key_from_message(str(exc))[:300]
                if enrichment:
                    self._apply(art, enrichment)
                    return "applied", None
                return "ok", None

        attempted = len(targets)
        applied = 0
        status = "ok"
        error_message: str | None = None
        try:
            results = await asyncio.gather(*(_enrich_one(a) for a in targets))
            for outcome, msg in results:  # target order → "last error" matches
                if outcome == "applied":
                    applied += 1
                elif outcome == "error":
                    status = "error"
                    error_message = msg
        finally:
            if owns:
                await client.aclose()
        await self._update_status(success=(status == "ok"), message=error_message)
        return {
            "enrichments_attempted": attempted,
            "enrichments_applied": applied,
            "status": status,
            "error_message": error_message,
        }

    async def _update_status(self, success: bool, message: str | None = None) -> None:
        now = datetime.now(timezone.utc)
        update = {"updated_at": now}
        if success:
            update["last_success_at"] = now
            update["status_message"] = "Enrichment-only metadata source."
        else:
            update["last_error_at"] = now
            update["last_error_message"] = (message or "openalex error")[:300]
            update["status_message"] = "Enrichment-only metadata source. Last error logged."
        try:
            await get_db()[Collections.SOURCE_CONFIGS].update_one(
                {"source_name": "openalex"}, {"$set": update}
            )
        except Exception:  # noqa: BLE001
            pass

    async def _get_with_retry(
        self, client: httpx.AsyncClient, url: str, params: dict | None = None
    ) -> httpx.Response | None:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = await client.get(url, params=params or {})
                if r.status_code == 404:
                    return None
                if r.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                r.raise_for_status()
                return r
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt >= MAX_RETRIES:
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt))
        if last_exc:
            raise last_exc
        return None

    async def _lookup_one(
        self, client: httpx.AsyncClient, art: dict[str, Any]
    ) -> dict[str, Any] | None:
        params = self._params()
        # 1) DOI
        if art.get("doi"):
            r = await self._get_with_retry(client, f"{BASE}/works/doi:{art['doi']}", params)
            if r is not None:
                return r.json()
        # 2) PMID
        if art.get("pmid"):
            r = await self._get_with_retry(client, f"{BASE}/works/pmid:{art['pmid']}", params)
            if r is not None:
                return r.json()
        # 3) PMCID (OpenAlex supports pmcid:PMCxxxxxxx)
        if art.get("pmcid"):
            r = await self._get_with_retry(client, f"{BASE}/works/pmcid:{art['pmcid']}", params)
            if r is not None:
                return r.json()
        # 4) Title fallback
        title = art.get("title") or ""
        if title:
            search_params = {**params, "filter": f"title.search:{title[:120]}", "per_page": "3"}
            r = await self._get_with_retry(client, f"{BASE}/works", search_params)
            if r is not None:
                body = r.json()
                results = body.get("results") or []
                norm_title = _norm_title(title)
                src_year = art.get("publication_year")
                best: dict[str, Any] | None = None
                best_sim = 0.0
                for cand in results:
                    cand_title = _norm_title(cand.get("title") or cand.get("display_name") or "")
                    sim = SequenceMatcher(a=norm_title, b=cand_title).ratio()
                    cand_year = cand.get("publication_year")
                    year_ok = (
                        src_year is None
                        or cand_year is None
                        or abs(int(cand_year) - int(src_year)) <= 1
                    )
                    if sim > best_sim and sim >= SIMILARITY_THRESHOLD and year_ok:
                        best = cand
                        best_sim = sim
                if best:
                    return best
        return None

    def _apply(self, art: dict[str, Any], oa: dict[str, Any]) -> None:
        # IDs
        ids = oa.get("ids") or {}
        if not art.get("doi") and ids.get("doi"):
            doi_raw = str(ids["doi"]).replace("https://doi.org/", "")
            art["doi"] = doi_raw
        if not art.get("pmid") and ids.get("pmid"):
            art["pmid"] = str(ids["pmid"]).replace("https://pubmed.ncbi.nlm.nih.gov/", "").rstrip("/")
        if not art.get("pmcid") and ids.get("pmcid"):
            art["pmcid"] = str(ids["pmcid"]).replace("https://www.ncbi.nlm.nih.gov/pmc/articles/", "").rstrip("/")

        # Authors
        if not art.get("authors_json"):
            auths = []
            for a in (oa.get("authorships") or [])[:50]:
                au = (a.get("author") or {})
                disp = au.get("display_name") or ""
                if disp:
                    auths.append({"fullName": disp, "lastName": disp.split(" ")[-1]})
            if auths:
                art["authors_json"] = auths

        # Journal / venue
        if not art.get("journal"):
            host = (oa.get("host_venue") or oa.get("primary_location") or {}).get("display_name")
            if not host:
                src = (oa.get("primary_location") or {}).get("source") or {}
                host = src.get("display_name")
            if host:
                art["journal"] = host

        # Citation count
        cited = oa.get("cited_by_count")
        if isinstance(cited, int):
            art["citation_count"] = cited
            sids = art.setdefault("source_ids_json", {})
            sids["openalex_cited_by_count"] = cited

        # OA URL — only `free_full_text_link`, NEVER promote to full_text_available.
        oa_loc = oa.get("best_oa_location") or {}
        oa_url = oa_loc.get("landing_page_url") or oa_loc.get("pdf_url")
        is_oa = bool(oa.get("open_access", {}).get("is_oa"))
        if oa_url and is_oa:
            urls = art.setdefault("source_urls_json", [])
            if oa_url not in urls:
                urls.append(oa_url)
            current = art.get("full_text_status")
            if current in (None, "", "unknown", "abstract_only"):
                # never promote past free_full_text_link
                if current != "full_text_available":
                    art["full_text_status"] = "free_full_text_link"

        # Work ID
        if oa.get("id"):
            sids = art.setdefault("source_ids_json", {})
            sids["openalex"] = oa["id"]
        art["enriched_by_openalex"] = True

        # Preprint detection (only set TO true, never reset)
        type_field = (oa.get("type") or "").lower()
        if "preprint" in type_field and not art.get("is_preprint"):
            art["is_preprint"] = True
            flags = art.setdefault("flags", [])
            if "openalex_marked_preprint" not in flags:
                flags.append("openalex_marked_preprint")

        # SAFETY: never overwrite is_retracted, has_expression_of_concern, is_preprint=true.
        # (We only set TO true above; never to false.)


openalex_connector = OpenAlexConnector()
