"""LiteratureRetrievalService — multi-source dispatch (Phase 2B).

Dispatches planned queries to the right connector. Each connector returns
``(query_type, normalized_article_dict)`` tuples so the rest of the
pipeline operates on a uniform shape regardless of source.

* europe_pmc  → calls Europe PMC REST search and returns RAW Europe PMC
  records (the pipeline still calls `metadata_normalizer.normalize` on
  these). This preserves Phase 2A behaviour and tests.
* pubmed      → calls NCBI E-utilities; the connector ALREADY normalizes
  the records, so the pipeline must skip metadata_normalizer for these.

Per-query try/except — one failing query MUST NOT crash the pipeline.
Updates ``source_configs`` with last_success/error timestamps.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from core.db import Collections, get_db
from services.audit import audit_service
from services.connectors.pubmed import pubmed_connector

logger = logging.getLogger("clinevidence.retrieval")

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
DEFAULT_TIMEOUT_S = 10.0
MAX_RETRIES = 2

RawRecord = dict[str, Any]


async def _update_source_status(
    source_name: str,
    *,
    success: bool,
    message: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc)
    update: dict[str, Any] = {"updated_at": now}
    if success:
        update["last_success_at"] = now
        update["status_message"] = message or "ok"
    else:
        update["last_error_at"] = now
        update["last_error_message"] = message or "error"
        update["status_message"] = message or "error"
    try:
        await get_db()[Collections.SOURCE_CONFIGS].update_one(
            {"source_name": source_name},
            {"$set": update},
        )
    except Exception:  # noqa: BLE001 — never crash the pipeline on status update
        logger.exception("source_configs status update failed for %s", source_name)


class LiteratureRetrievalService:
    """Source-dispatching retrieval. Phase 2B: europe_pmc + pubmed."""

    def __init__(self) -> None:
        self.page_size = int(os.environ.get("EUROPEPMC_PAGE_SIZE", "25"))
        self.timeout_s = float(os.environ.get("EUROPEPMC_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))

    async def retrieve(
        self,
        source: str,
        queries: list[dict[str, Any]],
        *,
        client: Optional[httpx.AsyncClient] = None,
        search_session_id: Optional[str] = None,
    ) -> list[tuple[str, RawRecord]]:
        """Dispatch retrieval by source name.

        Europe PMC returns RAW records (require normalizer).
        PubMed returns already-normalized records (skip normalizer).
        """
        if source == "europe_pmc":
            return await self.search_europe_pmc(
                queries, client=client, search_session_id=search_session_id
            )
        if source == "pubmed":
            return await pubmed_connector.search(
                queries, client=client, search_session_id=search_session_id
            )
        raise NotImplementedError(f"source '{source}' not supported")

    async def search_europe_pmc(
        self,
        queries: list[dict[str, Any]],
        *,
        client: Optional[httpx.AsyncClient] = None,
        search_session_id: Optional[str] = None,
    ) -> list[tuple[str, RawRecord]]:
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout_s)
        # B5 — run the per-query fetches concurrently (bounded) but reassemble
        # in the ORIGINAL query/record order, so ``out`` is byte-identical to
        # the former sequential loop and downstream dedup/ranking is unchanged.
        max_conc = max(1, int(os.environ.get("EUROPEPMC_MAX_CONCURRENT_QUERIES", "5")))
        sem = asyncio.Semaphore(max_conc)

        async def _run_one(q: dict[str, Any]) -> tuple[str, list[RawRecord], bool]:
            qtype = q.get("query_type", "unknown")
            qstring = q.get("query_string", "")
            if not qstring:
                return qtype, [], False
            async with sem:
                try:
                    records = await self._do_one(client, qstring)
                    return qtype, list(records), True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("europe_pmc query failed (%s): %s", qtype, exc)
                    await audit_service.log(
                        "source_query_error",
                        search_session_id=search_session_id,
                        payload={
                            "source": "europe_pmc",
                            "query_type": qtype,
                            "error": str(exc)[:500],
                        },
                    )
                    return qtype, [], False

        out: list[tuple[str, RawRecord]] = []
        any_success = False
        try:
            per_query = await asyncio.gather(*(_run_one(q) for q in queries))
            for qtype, records, ok in per_query:
                if ok:
                    any_success = True
                for r in records:
                    out.append((qtype, r))
        finally:
            if owns_client:
                await client.aclose()

        if any_success:
            await _update_source_status(
                "europe_pmc",
                success=True,
                message=f"retrieved {len(out)} records across queries",
            )
        else:
            await _update_source_status(
                "europe_pmc",
                success=False,
                message="all queries failed",
            )
        return out

    async def _do_one(self, client: httpx.AsyncClient, query_string: str) -> list[RawRecord]:
        params = {
            "query": query_string,
            "format": "json",
            "resultType": "core",
            "pageSize": self.page_size,
        }
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = await client.get(EUROPE_PMC_URL, params=params)
                r.raise_for_status()
                data = r.json()
                results = (data.get("resultList") or {}).get("result") or []
                return list(results)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= MAX_RETRIES:
                    raise
                await asyncio.sleep(0.5 * (2**attempt))
        if last_exc is not None:
            raise last_exc
        return []


literature_retrieval = LiteratureRetrievalService()
