"""Phase 4A OA full-text fetcher orchestrator.

For each (article, session) tuple, runs the cascade:
  1. Article-level cache lookup.
  2. Europe PMC `fullTextXML`.
  3. PMC OA Web Service → BioC (only when 1 + 2 yielded nothing AND PMCID exists).

Persists `article_text_extracts` rows + updates `article_access_locations`
for ingestible licenses. Emits closed-set audit events:
  oa_fetch_attempt | oa_fetch_success | oa_fetch_cache_hit |
  oa_fetch_skipped_ineligible | oa_fetch_skipped_license | oa_fetch_failed |
  oa_fetch_disabled.

NEVER calls an LLM. NEVER scrapes publisher HTML. NEVER persists raw text in
audit payloads. NEVER follows institutional / subscription URLs.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from core.db import Collections, get_db
from services.audit import audit_service

from .cache import (
    clone_cache_to_session,
    lookup_article_cache,
    write_cache_rows,
    write_session_rows,
)
from .license_gate import (
    determine_ingest_permission,
    is_article_eligible,
)
from . import europe_pmc_oa
from . import pmc_oa

logger = logging.getLogger("clinevidence.oa_fetcher")


# --- env helpers ------------------------------------------------------------


def _env_flag(key: str, default: bool) -> bool:
    raw = (os.environ.get(key) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Closed-set error codes that may surface in api responses / audit payloads.
ERROR_CODES = {
    "disabled", "no_identifier", "article_retracted",
    "article_retraction_notice", "article_is_preprint", "no_oa_signal",
    "license_skipped", "license_unknown", "license_noncommercial",
    "license_restricted", "europe_pmc_not_found", "pmc_oa_not_oa",
    "europe_pmc_failed", "pmc_oa_failed", "empty_sections",
    "cap_exceeded", "owner_check_failed", "other",
}


class OAFetcherService:
    """Stateless service. All state lives in Mongo collections."""

    async def fetch_for_article(
        self,
        *,
        article_id: str,
        session_id: str,
        user_id: str,
        force_refresh: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """Run the OA fetch cascade for one article in the context of one session.

        Returns a per-article result dict (used by the router and the
        pipeline). Always returns; never raises out of this method.
        """
        if not _env_flag("OA_FETCHER_ENABLED", True):
            return self._disabled_result(article_id)

        db = get_db()
        article = await db[Collections.ARTICLES].find_one({"id": article_id}, {"_id": 0})
        if article is None:
            return {
                "article_id": article_id, "status": "failed",
                "source": None, "license": None, "license_category": None,
                "can_ingest_for_summary": False, "sections_created": 0,
                "token_count": 0, "evidence_text_level": None,
                "error_code": "article_not_found",
            }

        eligible, reason = is_article_eligible(article)
        await audit_service.log(
            "oa_fetch_attempt",
            user_id=user_id, search_session_id=session_id,
            payload={"article_id": article_id},
        )
        if not eligible:
            await audit_service.log(
                "oa_fetch_skipped_ineligible",
                user_id=user_id, search_session_id=session_id,
                payload={"article_id": article_id, "error_code": reason},
            )
            await self._upsert_access_location(
                article_id=article_id, session_id=session_id,
                source=None, license_=None, license_category="unknown",
                can_ingest=False, status="skipped_ineligible",
                error_code=reason, sections_created=0, token_count=0,
            )
            return {
                "article_id": article_id, "status": "skipped_ineligible",
                "source": None, "license": None, "license_category": None,
                "can_ingest_for_summary": False, "sections_created": 0,
                "token_count": 0, "evidence_text_level": None,
                "error_code": reason,
            }

        # Cache lookup (unless force_refresh).
        if not force_refresh:
            cached = await lookup_article_cache(article_id)
            if cached:
                # Determine common metadata from the first row.
                sample = cached[0]
                session_doc = await db[Collections.SEARCH_SESSIONS].find_one(
                    {"id": session_id}, {"_id": 0, "expires_at": 1}
                )
                created = await clone_cache_to_session(
                    article_id=article_id,
                    search_session_id=session_id,
                    session_expires_at=(session_doc or {}).get("expires_at"),
                )
                token_count = sum(int(r.get("token_count") or 0) for r in cached)
                await audit_service.log(
                    "oa_fetch_cache_hit",
                    user_id=user_id, search_session_id=session_id,
                    payload={
                        "article_id": article_id,
                        "source": sample.get("oa_source"),
                        "sections_count": created,
                        "license_category": sample.get("oa_license_category"),
                    },
                )
                await self._upsert_access_location(
                    article_id=article_id, session_id=session_id,
                    source=sample.get("oa_source"),
                    license_=sample.get("oa_license"),
                    license_category=sample.get("oa_license_category"),
                    can_ingest=True,
                    status="cache_hit",
                    error_code=None,
                    sections_created=created,
                    token_count=token_count,
                )
                return {
                    "article_id": article_id, "status": "cache_hit",
                    "source": sample.get("oa_source"),
                    "license": sample.get("oa_license"),
                    "license_category": sample.get("oa_license_category"),
                    "can_ingest_for_summary": True,
                    "sections_created": created,
                    "token_count": token_count,
                    "evidence_text_level": "oa_full_text",
                    "error_code": None,
                }

        # Cascade: Europe PMC → PMC OA.
        pmcid = (article.get("pmcid") or "").strip()
        pmid = (article.get("pmid") or "").strip()
        t0 = time.time()
        owned_client = client is None
        if owned_client:
            client = httpx.AsyncClient(timeout=_env_int("OA_FETCH_TIMEOUT_SECONDS", 20))

        try:
            epmc = await europe_pmc_oa.fetch(
                pmcid=pmcid or None, pmid=pmid or None,
                timeout_s=float(_env_int("OA_FETCH_TIMEOUT_SECONDS", 20)),
                client=client,
            )
            source_used = "europe_pmc_oa"
            result = epmc
            if epmc["status"] in {"not_found", "failed", "skipped"} and pmcid:
                pmc = await pmc_oa.fetch(
                    pmcid=pmcid,
                    timeout_s=float(_env_int("OA_FETCH_TIMEOUT_SECONDS", 20)),
                    client=client,
                )
                if pmc["status"] == "fetched":
                    result = pmc
                    source_used = "pmc_oa"
                elif pmc["status"] == "not_oa":
                    # If europepmc came back not_found and pmc says not_oa,
                    # neither source can serve us. Honor whichever has more signal.
                    if epmc["status"] == "not_found":
                        result = pmc
                        source_used = "pmc_oa"
        finally:
            await self._update_source_configs_status(
                source=source_used if 'source_used' in locals() else None,
                ok=('result' in locals() and result.get("status") == "fetched"),
                error=(result.get("error_code") if 'result' in locals() else None),
            )
            if owned_client:
                await client.aclose()
        latency_ms = int((time.time() - t0) * 1000)

        if result["status"] != "fetched":
            err_code = (
                "europe_pmc_not_found" if epmc["status"] == "not_found" and source_used == "europe_pmc_oa"
                else "pmc_oa_not_oa" if result["status"] == "not_oa"
                else f"{source_used}_failed"
            )
            await audit_service.log(
                "oa_fetch_failed",
                user_id=user_id, search_session_id=session_id,
                payload={
                    "article_id": article_id, "source": source_used,
                    "error_code": err_code, "latency_ms": latency_ms,
                },
            )
            await self._upsert_access_location(
                article_id=article_id, session_id=session_id,
                source=source_used, license_=None,
                license_category="unknown", can_ingest=False,
                status="failed", error_code=err_code,
                sections_created=0, token_count=0,
            )
            return {
                "article_id": article_id, "status": "failed",
                "source": source_used,
                "license": None, "license_category": None,
                "can_ingest_for_summary": False,
                "sections_created": 0, "token_count": 0,
                "evidence_text_level": None, "error_code": err_code,
            }

        # License gate.
        permission = determine_ingest_permission(
            source=source_used, license=result.get("license_raw"),
            access_category=source_used, metadata=None,
        )
        license_norm = permission["license"]
        category = permission["license_category"]
        can_ingest = bool(permission["can_ingest_for_summary"])

        if not can_ingest:
            reason_code = {
                "noncommercial_only": "license_noncommercial",
                "restricted": "license_restricted",
                "unknown": "license_unknown",
                "other": "license_skipped",
                "commercial_reuse_allowed": "license_skipped",
            }.get(category, "license_skipped")
            await audit_service.log(
                "oa_fetch_skipped_license",
                user_id=user_id, search_session_id=session_id,
                payload={
                    "article_id": article_id, "source": source_used,
                    "license_category": category, "error_code": reason_code,
                    "sections_count": 0, "latency_ms": latency_ms,
                },
            )
            await self._upsert_access_location(
                article_id=article_id, session_id=session_id,
                source=source_used, license_=license_norm,
                license_category=category, can_ingest=False,
                status="skipped_license", error_code=reason_code,
                sections_created=0, token_count=0,
            )
            return {
                "article_id": article_id, "status": "skipped_license",
                "source": source_used, "license": license_norm,
                "license_category": category,
                "can_ingest_for_summary": False,
                "sections_created": 0, "token_count": 0,
                "evidence_text_level": None, "error_code": reason_code,
            }

        # Persist. Cache if commercially reuse-allowed; always create session rows.
        sections = result["sections"]
        token_count = sum(int(s.get("token_count") or 0) for s in sections)
        session_doc = await db[Collections.SEARCH_SESSIONS].find_one(
            {"id": session_id}, {"_id": 0, "expires_at": 1}
        )
        sess_expires = (session_doc or {}).get("expires_at")

        wrote_session = 0
        if category == "commercial_reuse_allowed":
            await write_cache_rows(
                article_id=article_id, oa_source=source_used,
                oa_license=license_norm, oa_license_category=category,
                sections=sections,
            )
            wrote_session = await clone_cache_to_session(
                article_id=article_id, search_session_id=session_id,
                session_expires_at=sess_expires,
            )
        else:
            wrote_session = await write_session_rows(
                article_id=article_id, search_session_id=session_id,
                session_expires_at=sess_expires, oa_source=source_used,
                oa_license=license_norm, oa_license_category=category,
                sections=sections,
            )

        await audit_service.log(
            "oa_fetch_success",
            user_id=user_id, search_session_id=session_id,
            payload={
                "article_id": article_id, "source": source_used,
                "license_category": category,
                "sections_count": wrote_session,
                "token_count": token_count,
                "latency_ms": latency_ms,
            },
        )
        await self._upsert_access_location(
            article_id=article_id, session_id=session_id,
            source=source_used, license_=license_norm,
            license_category=category, can_ingest=True,
            status="fetched", error_code=None,
            sections_created=wrote_session, token_count=token_count,
        )
        return {
            "article_id": article_id, "status": "fetched",
            "source": source_used, "license": license_norm,
            "license_category": category,
            "can_ingest_for_summary": True,
            "sections_created": wrote_session, "token_count": token_count,
            "evidence_text_level": "oa_full_text", "error_code": None,
        }

    async def fetch_for_articles(
        self,
        *,
        article_ids: list[str],
        session_id: str,
        user_id: str,
        force_refresh: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """Bulk fetch with the per-call cap.

        B5/perf — the first ``cap`` articles are fetched CONCURRENTLY (bounded by
        a semaphore) instead of one-at-a-time. Each ``fetch_for_article`` writes
        only its own article's extracts + access-location row, so concurrency is
        safe; results are reassembled in the original order. This is a major cut
        to the ``fetching_oa_full_text`` phase latency.
        """
        cap = _env_int("OA_MAX_FULL_TEXT_FETCHES_PER_SEARCH", 10)
        max_conc = max(1, _env_int("OA_MAX_CONCURRENT_FETCHES", 6))
        sem = asyncio.Semaphore(max_conc)
        to_fetch = article_ids[:cap]
        overflow = article_ids[cap:]

        async def _one(aid: str) -> dict[str, Any]:
            async with sem:
                return await self.fetch_for_article(
                    article_id=aid, session_id=session_id, user_id=user_id,
                    force_refresh=force_refresh, client=client,
                )

        fetched = (
            await asyncio.gather(*(_one(a) for a in to_fetch), return_exceptions=True)
            if to_fetch
            else []
        )
        # fetch_for_article is documented never-raises; if that contract is ever
        # broken, one bad article must not void the whole batch's writes.
        results: list[dict[str, Any]] = []
        for aid, r in zip(to_fetch, fetched):
            if isinstance(r, BaseException):
                logger.error("oa_fetch_unexpected_exception article=%s: %r", aid, r)
                results.append({
                    "article_id": aid, "status": "failed",
                    "source": None, "license": None, "license_category": None,
                    "can_ingest_for_summary": False, "sections_created": 0,
                    "token_count": 0, "evidence_text_level": None,
                    "error_code": "unexpected_exception",
                })
            else:
                results.append(r)
        for aid in overflow:
            results.append({
                "article_id": aid, "status": "skipped_cap_exceeded",
                "source": None, "license": None, "license_category": None,
                "can_ingest_for_summary": False, "sections_created": 0,
                "token_count": 0, "evidence_text_level": None,
                "error_code": "cap_exceeded",
            })
        summary = {
            "fetched": sum(1 for r in results if r["status"] == "fetched"),
            "cache_hits": sum(1 for r in results if r["status"] == "cache_hit"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "skipped_ineligible": sum(1 for r in results if r["status"] == "skipped_ineligible"),
            "skipped_license": sum(1 for r in results if r["status"] == "skipped_license"),
            "skipped_cap_exceeded": sum(1 for r in results if r["status"] == "skipped_cap_exceeded"),
            "disabled": sum(1 for r in results if r["status"] == "disabled"),
        }
        return {"results": results, "summary": summary}

    # ----------------------------------------------------------------- helpers
    def _disabled_result(self, article_id: str) -> dict[str, Any]:
        return {
            "article_id": article_id, "status": "disabled",
            "source": None, "license": None, "license_category": None,
            "can_ingest_for_summary": False, "sections_created": 0,
            "token_count": 0, "evidence_text_level": None,
            "error_code": "disabled",
        }

    async def _upsert_access_location(
        self, *, article_id: str, session_id: str,
        source: Optional[str], license_: Optional[str],
        license_category: str, can_ingest: bool,
        status: str, error_code: Optional[str],
        sections_created: int, token_count: int,
    ) -> None:
        if not source:
            return
        db = get_db()
        now = _now()
        url = (
            "https://www.ebi.ac.uk/europepmc/article/PMC/" if source == "europe_pmc_oa"
            else "https://pmc.ncbi.nlm.nih.gov/articles/"
        )
        existing = await db[Collections.ARTICLE_ACCESS_LOCATIONS].find_one({
            "article_id": article_id, "search_session_id": session_id,
            "access_category": source,
        }, {"_id": 0, "id": 1})
        doc = {
            "article_id": article_id,
            "search_session_id": session_id,
            "source": source,
            "access_category": source,
            "url": url,
            "is_direct_pdf": False,
            "is_oa": True,
            "requires_login": False,
            "can_view_in_app": True,
            "can_ingest_for_summary": can_ingest,
            "license": license_,
            "license_category": license_category,
            "version": "fetched",
            "evidence_text_level": "oa_full_text" if can_ingest else None,
            "oa_fetch_status": status,
            "oa_fetch_error_code": error_code,
            "oa_sections_created": int(sections_created),
            "oa_token_count": int(token_count),
            "oa_last_fetched_at": now,
            "last_checked_at": now,
            "updated_at": now,
        }
        if existing:
            await db[Collections.ARTICLE_ACCESS_LOCATIONS].update_one(
                {"id": existing["id"]}, {"$set": doc}
            )
        else:
            doc["id"] = uuid.uuid4().hex
            doc["created_at"] = now
            await db[Collections.ARTICLE_ACCESS_LOCATIONS].insert_one(doc)

    async def _update_source_configs_status(
        self, *, source: Optional[str], ok: bool, error: Optional[str] = None,
    ) -> None:
        if not source:
            return
        name = "europe_pmc_oa_fetcher" if source == "europe_pmc_oa" else "pmc_oa_fetcher"
        db = get_db()
        now = _now()
        update: dict[str, Any] = {"updated_at": now}
        if ok:
            update["last_success_at"] = now
            update["status_message"] = (
                "Europe PMC fullTextXML retrieval. License-gated."
                if name == "europe_pmc_oa_fetcher"
                else "PMC OA Web Service + BioC retrieval. License-gated."
            )
        else:
            update["last_error_at"] = now
            if error:
                update["last_error_message"] = str(error)[:120]
        await db[Collections.SOURCE_CONFIGS].update_one(
            {"source_name": name}, {"$set": update}
        )


oa_fetcher = OAFetcherService()
