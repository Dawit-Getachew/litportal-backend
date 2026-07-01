"""NIH iCite enrichment connector (Phase ImpactRank-1 — SHADOW ONLY).

Article-level citation-impact metrics by PMID from the NIH Office of
Portfolio Analysis iCite API. Enrichment-only: NEVER a primary ranking
source, NEVER overrides clinical directness / safety caps / lane
relevance, NEVER promotes full-text status, NEVER generates identifiers.

Endpoint:  https://icite.od.nih.gov/api/pubs?pmids=<csv up to 200>
Fields kept (article-level, deterministic): relative_citation_ratio,
nih_percentile, apt, citation_count, citations_per_year, cited_by_clin,
is_clinical, human, animal, molecular_cellular, provisional, year,
journal, doi, title.

Metrics are cached per-PMID with a timestamp; a missing metric NEVER
penalizes the article.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger("clinevidence.icite")

BASE = "https://icite.od.nih.gov/api/pubs"
DEFAULT_TIMEOUT_S = 12.0
MAX_RETRIES = 2
BATCH_SIZE = 200
CACHE_COLLECTION = "icite_metrics_cache"

# Closed-set article-level fields we keep (no abstracts / no full text).
_KEEP_FIELDS = (
    "pmid", "year", "title", "journal", "doi",
    "relative_citation_ratio", "nih_percentile", "apt",
    "citation_count", "citations_per_year", "cited_by_clin",
    "is_clinical", "human", "animal", "molecular_cellular", "provisional",
)


def _user_agent() -> str:
    mailto = os.environ.get("ICITE_MAILTO", os.environ.get("OPENALEX_MAILTO", "")).strip()
    suffix = f" (mailto:{mailto})" if mailto else ""
    return f"ClinEvidence-ImpactRank/1.0{suffix}"


def _cache_ttl_days() -> int:
    try:
        return max(1, int(os.environ.get("ICITE_CACHE_TTL_DAYS", "30")))
    except ValueError:
        return 30


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only the closed-set article-level fields + derive a numeric
    ``cited_by_clin_count`` from iCite's space-delimited PMID string."""
    rec = {k: raw.get(k) for k in _KEEP_FIELDS}
    cbc = raw.get("cited_by_clin")
    if isinstance(cbc, str):
        rec["cited_by_clin_count"] = len([p for p in cbc.split() if p.strip()])
    elif isinstance(cbc, (list, tuple)):
        rec["cited_by_clin_count"] = len(cbc)
    elif isinstance(cbc, (int, float)):
        rec["cited_by_clin_count"] = int(cbc)
    else:
        rec["cited_by_clin_count"] = 0
    if rec.get("pmid") is not None:
        rec["pmid"] = str(rec["pmid"])
    return rec


class IciteConnector:
    """Deterministic, batched, cached iCite client."""

    def __init__(self) -> None:
        self.timeout_s = float(os.environ.get("ICITE_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))
        self.batch_size = int(os.environ.get("ICITE_BATCH_SIZE", str(BATCH_SIZE)))

    async def _get_with_retry(
        self, client: httpx.AsyncClient, pmids_csv: str
    ) -> Optional[httpx.Response]:
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = await client.get(
                    BASE,
                    params={"pmids": pmids_csv, "legacy": "true"},
                    headers={"User-Agent": _user_agent()},
                )
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

    async def _load_cache(self, db, pmids: list[str]) -> dict[str, dict]:
        if db is None or not pmids:
            return {}
        cutoff = datetime.now(timezone.utc) - timedelta(days=_cache_ttl_days())
        out: dict[str, dict] = {}
        try:
            async for doc in db[CACHE_COLLECTION].find(
                {"pmid": {"$in": pmids}}, {"_id": 0}
            ):
                fetched = doc.get("fetched_at")
                if isinstance(fetched, str):
                    try:
                        fetched = datetime.fromisoformat(fetched)
                    except ValueError:
                        fetched = None
                if fetched and fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                if fetched and fetched >= cutoff and isinstance(doc.get("metrics"), dict):
                    out[str(doc["pmid"])] = doc["metrics"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("icite cache read failed (non-fatal): %s", exc)
        return out

    async def _store_cache(self, db, metrics: dict[str, dict]) -> None:
        if db is None or not metrics:
            return
        now = datetime.now(timezone.utc)
        try:
            for pmid, m in metrics.items():
                await db[CACHE_COLLECTION].update_one(
                    {"pmid": str(pmid)},
                    {"$set": {"pmid": str(pmid), "metrics": m, "fetched_at": now}},
                    upsert=True,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("icite cache write failed (non-fatal): %s", exc)

    async def fetch_metrics(
        self,
        pmids: list[str],
        *,
        client: Optional[httpx.AsyncClient] = None,
        db: Any = None,
    ) -> dict[str, Any]:
        """Return ``{"metrics": {pmid: {...}}, "stats": {...}}``. Never
        raises; a fetch error leaves the affected PMIDs simply unmetered."""
        clean = sorted({str(p).strip() for p in (pmids or []) if str(p).strip().isdigit()})
        metrics: dict[str, dict] = {}
        stats = {
            "requested": len(clean), "from_cache": 0, "fetched": 0,
            "batches": 0, "status": "ok", "error_message": None,
        }
        if not clean:
            return {"metrics": metrics, "stats": stats}

        cached = await self._load_cache(db, clean)
        metrics.update(cached)
        stats["from_cache"] = len(cached)
        to_fetch = [p for p in clean if p not in cached]

        owns = client is None
        if owns and to_fetch:
            client = httpx.AsyncClient(timeout=self.timeout_s)
        newly: dict[str, dict] = {}
        try:
            for i in range(0, len(to_fetch), self.batch_size):
                batch = to_fetch[i:i + self.batch_size]
                stats["batches"] += 1
                try:
                    r = await self._get_with_retry(client, ",".join(batch))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("icite batch failed (non-fatal): %s", exc)
                    stats["status"] = "error"
                    stats["error_message"] = str(exc)[:200]
                    continue
                if r is None:
                    continue
                for raw in (r.json() or {}).get("data") or []:
                    rec = _normalize_record(raw)
                    if rec.get("pmid"):
                        newly[rec["pmid"]] = rec
        finally:
            if owns and client is not None:
                await client.aclose()

        await self._store_cache(db, newly)
        metrics.update(newly)
        stats["fetched"] = len(newly)
        return {"metrics": metrics, "stats": stats}


icite_connector = IciteConnector()

__all__ = ["IciteConnector", "icite_connector", "BATCH_SIZE", "CACHE_COLLECTION"]
