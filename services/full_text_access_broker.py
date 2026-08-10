"""FullTextAccessBrokerService — Phase 2C.

OA-first cascade producing rows in ``article_access_locations`` per article.
NEVER scrapes publishers. NEVER bypasses paywalls. NEVER downloads paywalled
PDFs. NEVER stores institutional credentials. NEVER returns fake links.

Sources (in order, all best-effort + tolerant to failure):
  A. Europe PMC / PMC — uses already-merged article metadata (pmcid +
     fullTextUrlList captured during retrieval).
  B. Unpaywall — only if UNPAYWALL_EMAIL set. License-aware.
  C. Semantic Scholar OA — rescue lookup; OA rescue only, NEVER primary.
  D. Crossref — license + text-mining link discovery; only if CROSSREF_MAILTO.
     Always adds a `publisher_doi_link` row for any DOI.
  E. Institutional link generators (per-user) — OpenAthens, OpenURL,
     Semantic Scholar institutional fallback. DYNAMICALLY built per request;
     NEVER persisted into article_access_locations (avoids leaking one user's
     institution to another).
  F. Fallback rows — abstract_only / unknown.

`can_ingest_for_summary` rule (Phase 2C):
  TRUE iff (is_oa AND license in CC_OR_PUBLIC_DOMAIN AND
            access_category in {pmc_oa, europe_pmc_oa, oa_pdf, oa_html}).
  Unknown license → False.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote, urlencode

import httpx

from core.db import Collections, get_db
from services.audit import audit_service

logger = logging.getLogger("clinevidence.access_broker")

DEFAULT_TIMEOUT_S = 10.0
MAX_RETRIES = 2

# Licenses that permit ingestion of full text for downstream summarization.
INGESTIBLE_LICENSES = {
    "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "cc-by-nd",
    "cc-by-nc-nd", "cc0", "public-domain",
}
INGESTIBLE_ACCESS_CATEGORIES = {"pmc_oa", "europe_pmc_oa", "oa_pdf", "oa_html"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_license(raw: Any) -> Optional[str]:
    """Normalize license strings/URLs into canonical short codes."""
    if not raw:
        return None
    s = str(raw).lower().strip()
    # Strip URL prefixes like "https://creativecommons.org/licenses/by/4.0/"
    if "creativecommons.org" in s or s.startswith("http"):
        m = re.search(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)", s)
        if m:
            ident = m.group(2)
            if "zero" in ident or "cc0" in ident:
                return "cc0"
            return f"cc-{ident}" if not ident.startswith("cc-") else ident
        if "publicdomain" in s:
            return "public-domain"
    # Plain string forms
    s = s.replace(" ", "-")
    if s in INGESTIBLE_LICENSES:
        return s
    # Common variants
    if s in ("cc-by-4.0", "cc-by 4.0", "by"):
        return "cc-by"
    if s in ("cc-by-sa-4.0", "cc-by-sa 4.0"):
        return "cc-by-sa"
    if s.startswith("cc-by-nc-nd"):
        return "cc-by-nc-nd"
    if s.startswith("cc-by-nc-sa"):
        return "cc-by-nc-sa"
    if s.startswith("cc-by-nc"):
        return "cc-by-nc"
    if s.startswith("cc-by-sa"):
        return "cc-by-sa"
    if s.startswith("cc-by-nd"):
        return "cc-by-nd"
    if s.startswith("cc-by"):
        return "cc-by"
    if s in ("public", "publicdomain"):
        return "public-domain"
    # Unknown / unrecognized — return as-is so transparency shows it,
    # but ingestion rules will treat it as non-ingestible.
    return s


def _ingestible(*, is_oa: bool, license_: Optional[str], access_category: str) -> bool:
    if not is_oa:
        return False
    if access_category not in INGESTIBLE_ACCESS_CATEGORIES:
        return False
    if not license_:
        return False
    return license_ in INGESTIBLE_LICENSES


def _strip_secrets(msg: str) -> str:
    """Strip api keys and email parameters from a URL-embedded error message."""
    if not msg:
        return msg
    msg = re.sub(r"(api_key=)[^&\s]+", r"\1<redacted>", msg)
    msg = re.sub(r"(email=)[^&\s]+", r"\1<redacted>", msg)
    msg = re.sub(r"(mailto=)[^&\s]+", r"\1<redacted>", msg)
    msg = re.sub(r"(x-api-key:\s*)\S+", r"\1<redacted>", msg, flags=re.IGNORECASE)
    return msg


def _make_row(
    *,
    article_id: str,
    source: str,
    access_category: str,
    url: Optional[str],
    is_direct_pdf: bool = False,
    is_oa: bool = False,
    requires_login: bool = False,
    can_view_in_app: bool = False,
    license_: Optional[str] = None,
    version: Optional[str] = None,
    evidence_text_level: str = "unknown",
) -> dict[str, Any]:
    can_ingest = _ingestible(is_oa=is_oa, license_=license_, access_category=access_category)
    now = _now()
    return {
        "id": "",  # filled by caller via uuid4().hex
        "article_id": article_id,
        "source": source,
        "access_category": access_category,
        "url": url,
        "is_direct_pdf": bool(is_direct_pdf),
        "is_oa": bool(is_oa),
        "requires_login": bool(requires_login),
        "can_view_in_app": bool(can_view_in_app),
        "can_ingest_for_summary": bool(can_ingest),
        "license": license_,
        "version": version,
        "evidence_text_level": evidence_text_level,
        "last_checked_at": now,
        "created_at": now,
        "updated_at": now,
    }


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()


# ============================================================================
#  Per-source health stats tracker
# ============================================================================
class _SourceStats:
    """Track per-source call counts within a broker run for transparency."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {
            "unpaywall": {"status": "not_configured", "calls_made": 0, "successes": 0, "last_error": None},
            "semantic_scholar_oa": {"status": "ok", "calls_made": 0, "successes": 0, "last_error": None},
            "crossref": {"status": "not_configured", "calls_made": 0, "successes": 0, "last_error": None},
        }

    def call(self, source: str) -> None:
        if source in self.data:
            self.data[source]["calls_made"] += 1

    def ok(self, source: str) -> None:
        if source in self.data:
            self.data[source]["status"] = "ok"
            self.data[source]["successes"] += 1

    def error(self, source: str, err: str) -> None:
        if source in self.data:
            self.data[source]["status"] = "error"
            self.data[source]["last_error"] = _strip_secrets(str(err))[:300]

    def skip(self, source: str, reason: str) -> None:
        if source in self.data:
            self.data[source]["status"] = reason

    def to_dict(self) -> dict[str, Any]:
        return self.data


# ============================================================================
#  External connector adapters (each is one method; partial-failure tolerant)
# ============================================================================
async def _get_with_retry(
    client: httpx.AsyncClient, url: str, params: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response | None:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = await client.get(url, params=params or {}, headers=headers or {})
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


# ============================================================================
#  FullTextAccessBrokerService
# ============================================================================
class FullTextAccessBrokerService:
    """Construct article_access_locations rows from OA-first cascade."""

    def __init__(self) -> None:
        self.timeout_s = float(os.environ.get("ACCESS_BROKER_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))
        self.unpaywall_email = os.environ.get("UNPAYWALL_EMAIL", "").strip()
        self.unpaywall_enabled = (
            os.environ.get("UNPAYWALL_ENABLED", "").lower() == "true"
            or (os.environ.get("UNPAYWALL_ENABLED") in (None, "") and bool(self.unpaywall_email))
        )
        self.crossref_mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
        self.crossref_enabled = (
            os.environ.get("CROSSREF_ENABLED", "").lower() == "true"
            or (os.environ.get("CROSSREF_ENABLED") in (None, "") and bool(self.crossref_mailto))
        )
        self.ss_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self.ss_enabled = (
            os.environ.get("SEMANTIC_SCHOLAR_ENABLED", "true").lower() != "false"
        )
        self.ss_max_lookups = int(os.environ.get("SS_MAX_LOOKUPS_PER_SEARCH", "50"))

    # ------------------------------------------------------------------
    #  Public entry: broker_for_session
    # ------------------------------------------------------------------
    async def broker_for_session(
        self,
        session_id: str,
        *,
        client: Optional[httpx.AsyncClient] = None,
        replace_existing: bool = True,
    ) -> dict[str, Any]:
        """Run the cascade for all articles in a session.

        Returns: stats dict suitable for the transparency accordion +
        evaluator's `access_broker` block.
        """
        db = get_db()
        # Pull session articles in this session (via search_session_articles).
        sa_rows = await (
            db[Collections.SEARCH_SESSION_ARTICLES]
            .find({"search_session_id": session_id}, {"_id": 0, "article_id": 1})
            .to_list(length=500)
        )
        article_ids = sorted({r["article_id"] for r in sa_rows if r.get("article_id")})
        return await self.broker_for_articles(
            article_ids,
            session_id=session_id,
            client=client,
            replace_existing=replace_existing,
        )

    async def broker_for_articles(
        self,
        article_ids: list[str],
        *,
        session_id: str,
        client: Optional[httpx.AsyncClient] = None,
        replace_existing: bool = True,
    ) -> dict[str, Any]:
        """Run the cascade for an EXPLICIT set of articles.

        The per-article entry point required by the SOW1 contract: the
        full-text-import job must broker exactly the article being imported,
        not every article in the caller-supplied session (up to 500), which is
        the difference between ~26s and ~11,800s of work per message.

        ``broker_for_session`` is now a thin wrapper that resolves a session's
        article ids and delegates here, so both paths share one implementation.
        """
        db = get_db()
        article_ids = sorted({a for a in (article_ids or []) if a})
        articles: list[dict[str, Any]] = []
        if article_ids:
            async for art in db[Collections.ARTICLES].find(
                {"id": {"$in": article_ids}}, {"_id": 0}
            ):
                articles.append(art)

        # Optionally remove existing rows for these articles so we re-run cleanly.
        removed = 0
        if replace_existing and article_ids:
            res = await db[Collections.ARTICLE_ACCESS_LOCATIONS].delete_many(
                {"article_id": {"$in": article_ids}}
            )
            removed = res.deleted_count or 0

        stats = _SourceStats()
        if not self.unpaywall_enabled:
            stats.skip("unpaywall", "not_configured" if not self.unpaywall_email else "disabled")
        if not self.crossref_enabled:
            stats.skip("crossref", "not_configured" if not self.crossref_mailto else "disabled")
        if not self.ss_enabled:
            stats.skip("semantic_scholar_oa", "disabled")

        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout_s)
        # Reset per-run circuit breakers.
        self._ss_circuit_open = False
        articles_processed = 0
        new_rows_total = 0
        # perf — broker every article CONCURRENTLY (bounded). Each article's rows
        # are independent (per-article access_locations), so this is safe and
        # cuts the ``brokering_access`` phase from N serial round-trips to ~N/conc.
        # Semantic-Scholar budget is assigned by INDEX (first ss_max_lookups
        # articles get 1 lookup each) so the total SS-call cap holds without a
        # shared running counter.
        import uuid
        max_conc = max(1, int(os.environ.get("ACCESS_BROKER_MAX_CONCURRENCY", "8")))
        sem = asyncio.Semaphore(max_conc)

        async def _broker(idx: int, art: dict[str, Any]) -> list[dict[str, Any]]:
            ss_budget = 1 if idx < self.ss_max_lookups else 0
            async with sem:
                return await self._broker_one_article(
                    client, art, stats, ss_budget_left=ss_budget, session_id=session_id,
                )

        try:
            per_article = await asyncio.gather(
                *(_broker(i, a) for i, a in enumerate(articles))
            )
            to_insert: list[dict[str, Any]] = []
            for rows in per_article:
                articles_processed += 1
                for r in (rows or []):
                    if not r.get("id"):
                        r["id"] = uuid.uuid4().hex
                    to_insert.append(r)
            if to_insert:
                await db[Collections.ARTICLE_ACCESS_LOCATIONS].insert_many(to_insert)
                new_rows_total = len(to_insert)
        finally:
            if owns_client:
                await client.aclose()

        # Update source_configs status messages (best effort).
        await self._update_source_configs(stats)

        # Compute summary counters for transparency.
        summary = await self._summarize(article_ids)
        summary["sources_status"] = stats.to_dict()
        summary["articles_processed"] = articles_processed
        summary["new_access_locations"] = new_rows_total
        summary["removed_access_locations"] = removed
        summary["note"] = (
            "Subscription full text is link-out only. ClinEvidence never stores "
            "institutional credentials or ingests paywalled PDFs."
        )

        await audit_service.log(
            "access_broker_run",
            search_session_id=session_id,
            payload={
                "articles_processed": articles_processed,
                "new_rows": new_rows_total,
                "removed_rows": removed,
                "sources_status": {k: v.get("status") for k, v in stats.to_dict().items()},
            },
        )
        return summary

    # ------------------------------------------------------------------
    #  Single-article cascade
    # ------------------------------------------------------------------
    async def _broker_one_article(
        self,
        client: httpx.AsyncClient,
        art: dict[str, Any],
        stats: _SourceStats,
        ss_budget_left: int,
        session_id: Optional[str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        article_id = art["id"]
        pmid = (art.get("pmid") or "").strip() or None
        doi = (art.get("doi") or "").strip() or None
        pmcid = (art.get("pmcid") or "").strip() or None

        # ----- A. PMC / Europe PMC -----
        if pmcid:
            rows.append(_make_row(
                article_id=article_id,
                source="pmc",
                access_category="pmc_oa",
                url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                is_oa=True,
                is_direct_pdf=False,
                can_view_in_app=True,
                license_=_normalize_license(art.get("license")),
                evidence_text_level="oa_full_text",
            ))

        # Europe PMC raw indicators may be carried on the article via Phase-2B
        # source_ids_json / source_urls_json. We DO have inEPMC flag only if
        # the connector preserved it; otherwise we derive from URLs.
        epmc_urls = [u for u in (art.get("source_urls_json") or [])
                     if "europepmc.org" in u or "/europepmc/" in u]
        in_epmc = bool((art.get("source_ids_json") or {}).get("inEPMC")) or bool(epmc_urls)
        if in_epmc:
            url = epmc_urls[0] if epmc_urls else (
                f"https://europepmc.org/article/MED/{pmid}" if pmid else None
            )
            if url:
                rows.append(_make_row(
                    article_id=article_id,
                    source="europe_pmc",
                    access_category="europe_pmc_oa",
                    url=url,
                    is_oa=True,
                    can_view_in_app=True,
                    license_=_normalize_license(art.get("license")),
                    evidence_text_level="oa_full_text",
                ))

        # Phase-2B Europe PMC fullTextUrlList[*] availabilityCode in {OA, F}
        # is carried via art["full_text_urls_json"] if normalized stored it.
        for url_entry in (art.get("full_text_urls_json") or []):
            avail = (url_entry.get("availabilityCode") or "").upper()
            doc_style = (url_entry.get("documentStyle") or "").lower()
            ft_url = url_entry.get("url") or ""
            if avail in ("OA", "F") and ft_url and "pdf" in doc_style:
                rows.append(_make_row(
                    article_id=article_id,
                    source="europe_pmc",
                    access_category="oa_pdf",
                    url=ft_url,
                    is_oa=True,
                    is_direct_pdf=True,
                    can_view_in_app=True,
                    license_=_normalize_license(art.get("license")),
                    evidence_text_level="oa_full_text",
                ))

        # ----- B. Unpaywall -----
        if self.unpaywall_enabled and doi:
            try:
                stats.call("unpaywall")
                url = f"https://api.unpaywall.org/v2/{doi}"
                params = {"email": self.unpaywall_email}
                resp = await _get_with_retry(client, url, params=params)
                if resp is not None:
                    body = resp.json()
                    stats.ok("unpaywall")
                    best = body.get("best_oa_location") or {}
                    if best:
                        rows.extend(self._unpaywall_rows(article_id, best))
            except Exception as exc:  # noqa: BLE001
                logger.warning("unpaywall lookup failed: %s", exc)
                stats.error("unpaywall", str(exc))
                await audit_service.log(
                    "source_query_error",
                    search_session_id=session_id,
                    payload={"source": "unpaywall", "truncated_error": _strip_secrets(str(exc))[:300]},
                )

        # ----- C. Semantic Scholar OA rescue -----
        if self.ss_enabled and ss_budget_left > 0 and (doi or pmid):
            try:
                stats.call("semantic_scholar_oa")
                rec = await self._semantic_scholar_lookup(client, doi=doi, pmid=pmid)
                if rec:
                    stats.ok("semantic_scholar_oa")
                    rows.extend(self._semantic_scholar_rows(article_id, rec))
            except Exception as exc:  # noqa: BLE001
                logger.warning("semantic scholar lookup failed: %s", exc)
                stats.error("semantic_scholar_oa", str(exc))
                await audit_service.log(
                    "source_query_error",
                    search_session_id=session_id,
                    payload={"source": "semantic_scholar_oa", "truncated_error": _strip_secrets(str(exc))[:300]},
                )

        # ----- D. Crossref (license + text-mining links) -----
        if self.crossref_enabled and doi:
            try:
                stats.call("crossref")
                rec = await self._crossref_lookup(client, doi=doi)
                if rec:
                    stats.ok("crossref")
                    rows.extend(self._crossref_rows(article_id, rec))
            except Exception as exc:  # noqa: BLE001
                logger.warning("crossref lookup failed: %s", exc)
                stats.error("crossref", str(exc))
                await audit_service.log(
                    "source_query_error",
                    search_session_id=session_id,
                    payload={"source": "crossref", "truncated_error": _strip_secrets(str(exc))[:300]},
                )

        # ----- D'. publisher_doi_link fallback when DOI exists -----
        if doi:
            # Only add if we don't already have a publisher_doi_link via crossref.
            has_doi_link = any(
                r["source"] == "crossref" and r["access_category"] == "publisher_doi_link"
                for r in rows
            )
            if not has_doi_link:
                rows.append(_make_row(
                    article_id=article_id,
                    source="article",
                    access_category="publisher_doi_link",
                    url=f"https://doi.org/{doi}",
                    is_oa=False,
                    requires_login=False,
                    can_view_in_app=False,
                    license_=None,
                    evidence_text_level="metadata_only",
                ))

        # ----- F. Fallback rows -----
        if not rows:
            if (art.get("abstract") or "").strip():
                rows.append(_make_row(
                    article_id=article_id,
                    source="article",
                    access_category="abstract_only",
                    url=None,
                    evidence_text_level="abstract_only",
                ))
            else:
                rows.append(_make_row(
                    article_id=article_id,
                    source="article",
                    access_category="unknown",
                    url=None,
                    evidence_text_level="unknown",
                ))

        return rows

    # ------------------------------------------------------------------
    #  Unpaywall row builder
    # ------------------------------------------------------------------
    @staticmethod
    def _unpaywall_rows(article_id: str, best: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        url_for_pdf = best.get("url_for_pdf") or ""
        url_land = best.get("url") or ""
        host_type = (best.get("host_type") or "").lower()
        license_ = _normalize_license(best.get("license"))
        version = (best.get("version") or "").lower() or None
        if url_for_pdf:
            out.append(_make_row(
                article_id=article_id,
                source="unpaywall",
                access_category="oa_pdf",
                url=url_for_pdf,
                is_oa=True,
                is_direct_pdf=True,
                can_view_in_app=True,
                license_=license_,
                version=version,
                evidence_text_level="oa_full_text",
            ))
        elif url_land:
            category = "oa_html" if host_type == "publisher" else "free_full_text_link"
            out.append(_make_row(
                article_id=article_id,
                source="unpaywall",
                access_category=category,
                url=url_land,
                is_oa=True,
                is_direct_pdf=False,
                can_view_in_app=(category == "oa_html"),
                license_=license_,
                version=version,
                evidence_text_level=(
                    "oa_full_text" if category == "oa_html" else "free_full_text_link"
                ),
            ))
        return out

    # ------------------------------------------------------------------
    #  Semantic Scholar OA rescue
    # ------------------------------------------------------------------
    async def _semantic_scholar_lookup(
        self, client: httpx.AsyncClient, *, doi: Optional[str], pmid: Optional[str]
    ) -> Optional[dict[str, Any]]:
        if getattr(self, "_ss_circuit_open", False):
            # Circuit breaker tripped earlier in this run — skip.
            return None
        headers = {"x-api-key": self.ss_api_key} if self.ss_api_key else {}
        fields = "openAccessPdf,externalIds,isOpenAccess,corpusId,url"
        # Priority: DOI, then PMID
        for key, val in [("DOI", doi), ("PMID", pmid)]:
            if not val:
                continue
            url = f"https://api.semanticscholar.org/graph/v1/paper/{key}:{val}"
            params = {"fields": fields}
            try:
                # Single attempt per identifier; let _get_with_retry handle
                # internal retries, but 429s bypass retries via the override
                # parameter below.
                resp = await self._ss_get_no_retry_on_429(client, url, params=params, headers=headers)
            except httpx.HTTPError:
                continue
            if resp is not None:
                return resp.json()
        return None

    async def _ss_get_no_retry_on_429(
        self, client: httpx.AsyncClient, url: str, *, params: dict, headers: dict
    ) -> httpx.Response | None:
        """Semantic Scholar-specific GET: 429 trips the per-run circuit breaker."""
        try:
            r = await client.get(url, params=params, headers=headers)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                self._ss_circuit_open = True
                return None
            if r.status_code >= 400:
                r.raise_for_status()
            return r
        except httpx.HTTPError:
            raise

    @staticmethod
    def _semantic_scholar_rows(article_id: str, rec: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pdf = rec.get("openAccessPdf") or {}
        pdf_url = pdf.get("url")
        is_oa = bool(rec.get("isOpenAccess"))
        if pdf_url:
            out.append(_make_row(
                article_id=article_id,
                source="semantic_scholar",
                access_category="oa_pdf",
                url=pdf_url,
                is_oa=is_oa,
                is_direct_pdf=True,
                can_view_in_app=True,
                license_=None,  # S2 rarely returns license; conservative → no ingest
                evidence_text_level="oa_full_text" if is_oa else "free_full_text_link",
            ))
        return out

    # ------------------------------------------------------------------
    #  Crossref
    # ------------------------------------------------------------------
    async def _crossref_lookup(
        self, client: httpx.AsyncClient, *, doi: str
    ) -> Optional[dict[str, Any]]:
        url = f"https://api.crossref.org/works/{doi}"
        user_agent = f"ClinEvidenceSearch/2C (mailto:{self.crossref_mailto})"
        headers = {"User-Agent": user_agent}
        resp = await _get_with_retry(client, url, headers=headers)
        if resp is not None:
            return resp.json().get("message") or None
        return None

    @staticmethod
    def _crossref_rows(article_id: str, msg: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        # Pick license — most permissive among entries that started by publication date.
        licenses = msg.get("license") or []
        chosen_license: Optional[str] = None
        if licenses:
            ranked = []
            for lic in licenses:
                code = _normalize_license(lic.get("URL"))
                ranked.append(code)
            for cand in ("cc-by", "cc0", "public-domain", "cc-by-sa", "cc-by-nc"):
                if cand in ranked:
                    chosen_license = cand
                    break
            if not chosen_license and ranked:
                chosen_license = ranked[0]
        # text-mining link
        links = msg.get("link") or []
        for link in links:
            intended = (link.get("intended-application") or "").lower()
            url = link.get("URL") or ""
            if intended == "text-mining" and url:
                out.append(_make_row(
                    article_id=article_id,
                    source="crossref",
                    access_category="publisher_doi_link",
                    url=url,
                    is_oa=False,  # Crossref text-mining URL is TDM-licensed, not free OA.
                    is_direct_pdf=False,
                    requires_login=True,
                    can_view_in_app=False,
                    license_=chosen_license,
                    evidence_text_level="metadata_only",
                ))
        # Always also include the canonical doi.org publisher link.
        doi_url = f"https://doi.org/{msg.get('DOI', '')}" if msg.get("DOI") else None
        if doi_url:
            out.append(_make_row(
                article_id=article_id,
                source="crossref",
                access_category="publisher_doi_link",
                url=doi_url,
                is_oa=False,
                requires_login=False,
                can_view_in_app=False,
                license_=chosen_license,
                evidence_text_level="metadata_only",
            ))
        return out

    # ------------------------------------------------------------------
    #  Source config status sync
    # ------------------------------------------------------------------
    async def _update_source_configs(self, stats: _SourceStats) -> None:
        db = get_db()
        now = _now()
        mapping = {
            "unpaywall": (self.unpaywall_email, "Unpaywall: license-aware OA lookup."),
            "semantic_scholar_oa": (True, "OA rescue + free full-text lookup. Optional API key for higher rate limits."),
            "crossref": (self.crossref_mailto, "Crossref: license + DOI link discovery."),
        }
        for name, (cfg_value, on_msg) in mapping.items():
            row = stats.data.get(name, {})
            status = row.get("status") or "skipped"
            update: dict[str, Any] = {"updated_at": now}
            if not cfg_value and name in ("unpaywall", "crossref"):
                update["status_message"] = (
                    "Unpaywall not configured (set UNPAYWALL_EMAIL)." if name == "unpaywall"
                    else "Crossref not configured (set CROSSREF_MAILTO)."
                )
            elif status == "ok":
                update["last_success_at"] = now
                update["status_message"] = on_msg
            elif status == "error":
                update["last_error_at"] = now
                update["last_error_message"] = (row.get("last_error") or "error")[:300]
                update["status_message"] = f"{on_msg} Last error logged."
            else:
                continue
            try:
                await db[Collections.SOURCE_CONFIGS].update_one(
                    {"source_name": name}, {"$set": update}
                )
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    #  Summary counters for transparency
    # ------------------------------------------------------------------
    @staticmethod
    async def _summarize(article_ids: list[str]) -> dict[str, Any]:
        db = get_db()
        if not article_ids:
            return {
                "articles_with_oa_full_text": 0,
                "articles_with_oa_pdf": 0,
                "articles_with_free_full_text_link": 0,
                "articles_with_institutional_link": 0,
                "articles_abstract_only": 0,
                "articles_unknown_access": 0,
            }
        # Group by article_id, classify whether it has each access kind.
        cursor = db[Collections.ARTICLE_ACCESS_LOCATIONS].find(
            {"article_id": {"$in": article_ids}},
            {"_id": 0, "article_id": 1, "access_category": 1, "is_direct_pdf": 1, "is_oa": 1},
        )
        rows = await cursor.to_list(length=5000)
        by_aid: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_aid.setdefault(r["article_id"], []).append(r)
        oa_ft = oa_pdf = free_ft = abs_only = unknown = 0
        for aid in article_ids:
            arts_rows = by_aid.get(aid, [])
            cats = {r["access_category"] for r in arts_rows}
            if cats & {"pmc_oa", "europe_pmc_oa", "oa_html"} or any(
                r["is_oa"] and not r["is_direct_pdf"] for r in arts_rows
            ):
                oa_ft += 1
            if any(r["is_direct_pdf"] and r["is_oa"] for r in arts_rows):
                oa_pdf += 1
            if "free_full_text_link" in cats:
                free_ft += 1
            if cats == {"abstract_only"} or (not cats and not arts_rows):
                pass
            if cats == {"abstract_only"}:
                abs_only += 1
            if not cats or cats == {"unknown"}:
                unknown += 1
        return {
            "articles_with_oa_full_text": oa_ft,
            "articles_with_oa_pdf": oa_pdf,
            "articles_with_free_full_text_link": free_ft,
            "articles_with_institutional_link": 0,  # institutional rows are dynamic per-user
            "articles_abstract_only": abs_only,
            "articles_unknown_access": unknown,
        }


# ============================================================================
#  Dynamic institutional link generators (per-user, NEVER persisted)
# ============================================================================
def _best_target_url_for(article: dict[str, Any]) -> Optional[str]:
    doi = (article.get("doi") or "").strip()
    pmid = (article.get("pmid") or "").strip()
    urls = article.get("source_urls_json") or []
    if doi:
        return f"https://doi.org/{doi}"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if urls:
        return urls[0]
    return None


def build_openathens_row(article: dict[str, Any], customer_domain: str) -> Optional[dict[str, Any]]:
    target = _best_target_url_for(article)
    if not target or not customer_domain:
        return None
    encoded = quote(target, safe="")
    url = f"https://go.openathens.net/redirector/{customer_domain}?url={encoded}"
    return _make_row(
        article_id=article["id"],
        source="openathens",
        access_category="institutional_openathens",
        url=url,
        is_oa=False,
        requires_login=True,
        can_view_in_app=False,
        license_=None,
        evidence_text_level="institutional_linkout",
    )


def build_openurl_row(article: dict[str, Any], resolver_base: str) -> Optional[dict[str, Any]]:
    if not resolver_base:
        return None
    doi = (article.get("doi") or "").strip()
    pmid = (article.get("pmid") or "").strip()
    title = (article.get("title") or "").strip()
    journal = (article.get("journal") or "").strip()
    year = article.get("publication_year")
    authors = article.get("authors_json") or []
    first_author_last = (authors[0].get("lastName") if authors else "") or ""
    pairs: list[tuple[str, str]] = [
        ("url_ver", "Z39.88-2004"),
        ("rft_val_fmt", "info:ofi/fmt:kev:mtx:journal"),
    ]
    if doi:
        pairs.append(("rft_id", f"info:doi/{doi}"))
    if pmid:
        pairs.append(("rft_id", f"info:pmid/{pmid}"))
    if title:
        pairs.append(("rft.atitle", title[:200]))
    if journal:
        pairs.append(("rft.jtitle", journal[:120]))
    if year:
        pairs.append(("rft.date", str(year)))
    if first_author_last:
        pairs.append(("rft.aulast", first_author_last[:60]))
    # No identifiers beyond these are stored on articles right now.
    qs = urlencode(pairs, quote_via=quote)
    sep = "&" if "?" in resolver_base else "?"
    url = f"{resolver_base}{sep}{qs}"
    return _make_row(
        article_id=article["id"],
        source="openurl",
        access_category="institutional_openurl",
        url=url,
        is_oa=False,
        requires_login=True,
        can_view_in_app=False,
        license_=None,
        evidence_text_level="institutional_linkout",
    )


def build_semantic_scholar_institutional_row(article: dict[str, Any]) -> Optional[dict[str, Any]]:
    doi = (article.get("doi") or "").strip()
    sids = article.get("source_ids_json") or {}
    paper_id = sids.get("semantic_scholar")
    if paper_id:
        url = f"https://www.semanticscholar.org/paper/{paper_id}"
    elif doi:
        url = f"https://www.semanticscholar.org/paper/DOI:{doi}"
    else:
        title = (article.get("title") or "").strip()
        if not title:
            return None
        url = f"https://www.semanticscholar.org/search?q={quote(title[:120])}"
    return _make_row(
        article_id=article["id"],
        source="semantic_scholar",
        access_category="semantic_scholar_institutional_linkout",
        url=url,
        is_oa=False,
        requires_login=True,
        can_view_in_app=False,
        license_=None,
        evidence_text_level="institutional_linkout",
    )


def institutional_rows_for_user(
    article: dict[str, Any],
    settings: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dynamically build institutional rows for a single article + user settings."""
    if not settings:
        return []
    out: list[dict[str, Any]] = []
    if settings.get("openathens_customer_domain"):
        row = build_openathens_row(article, settings["openathens_customer_domain"])
        if row:
            out.append(row)
    if settings.get("openurl_resolver_base_url"):
        row = build_openurl_row(article, settings["openurl_resolver_base_url"])
        if row:
            out.append(row)
    if settings.get("semantic_scholar_fallback_enabled"):
        row = build_semantic_scholar_institutional_row(article)
        if row:
            out.append(row)
    return out


# ============================================================================
#  Best-access primary button
# ============================================================================
def compute_best_access(
    article: dict[str, Any],
    access_locations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Decide the primary button label + target URL.

    Priority (first match wins):
    1. Open PDF — direct OA PDF in-app (PMC > Europe PMC > Unpaywall > S2)
    2. Open Full Text — pmc_oa / europe_pmc_oa / oa_html
    3. Free Full Text — free_full_text_link with is_oa=true
    4. Open via Institution — institutional_openathens / institutional_openurl
    5. Get Full Text — DOI fallback when no institution row
    6. Open in Semantic Scholar — semantic_scholar_institutional_linkout
    7. Open DOI — publisher_doi_link
    8. Abstract Only — disabled
    9. Access Unknown — disabled
    """
    institutional_helper = (
        "Opens through your institution. ClinEvidence does not store "
        "institutional credentials or ingest subscription full text."
    )

    def _first(predicate) -> dict | None:
        for row in access_locations:
            try:
                if predicate(row):
                    return row
            except Exception:
                continue
        return None

    # 1. Direct OA PDF preference order
    pdf_priority = ["pmc", "europe_pmc", "unpaywall", "semantic_scholar"]
    for source in pdf_priority:
        row = _first(lambda r: (
            r.get("source") == source
            and r.get("is_direct_pdf")
            and r.get("can_view_in_app")
            and r.get("is_oa")
            and r.get("url")
        ))
        if row:
            return {
                "button_label": "Open PDF",
                "target_url": row["url"],
                "disabled": False,
                "helper_text": "Open access full-text PDF.",
                "access_category": row.get("access_category"),
            }

    # 2. Open Full Text (OA HTML)
    row = _first(lambda r: (
        r.get("access_category") in {"pmc_oa", "europe_pmc_oa", "oa_html"}
        and r.get("can_view_in_app")
        and r.get("url")
    ))
    if row:
        return {
            "button_label": "Open Full Text",
            "target_url": row["url"],
            "disabled": False,
            "helper_text": "Open-access HTML full text.",
            "access_category": row.get("access_category"),
        }

    # 3. Free Full Text
    row = _first(lambda r: (
        r.get("access_category") == "free_full_text_link"
        and r.get("is_oa")
        and r.get("url")
    ))
    if row:
        return {
            "button_label": "Free Full Text",
            "target_url": row["url"],
            "disabled": False,
            "helper_text": "Free full-text link.",
            "access_category": row.get("access_category"),
        }

    # 4. Open via Institution
    row = _first(lambda r: (
        r.get("access_category") in {"institutional_openathens", "institutional_openurl"}
        and r.get("url")
    ))
    if row:
        return {
            "button_label": "Open via Institution",
            "target_url": row["url"],
            "disabled": False,
            "helper_text": institutional_helper,
            "access_category": row.get("access_category"),
        }

    # 5. Get Full Text via DOI when no institution method is configured
    doi = (article.get("doi") or "").strip()
    has_institutional = any(
        r.get("access_category") in {"institutional_openathens", "institutional_openurl"}
        for r in access_locations
    )
    if doi and not has_institutional:
        return {
            "button_label": "Get Full Text",
            "target_url": f"https://doi.org/{doi}",
            "disabled": False,
            "helper_text": "Opens the publisher page via DOI. Subscription may be required.",
            "access_category": "publisher_doi_link",
        }

    # 6. Open in Semantic Scholar
    row = _first(lambda r: (
        r.get("access_category") == "semantic_scholar_institutional_linkout"
        and r.get("url")
    ))
    if row:
        return {
            "button_label": "Open in Semantic Scholar",
            "target_url": row["url"],
            "disabled": False,
            "helper_text": institutional_helper,
            "access_category": row.get("access_category"),
        }

    # 7. Open DOI
    row = _first(lambda r: (
        r.get("access_category") == "publisher_doi_link" and r.get("url")
    ))
    if row:
        return {
            "button_label": "Open DOI",
            "target_url": row["url"],
            "disabled": False,
            "helper_text": "Opens the publisher page via DOI. Subscription may be required.",
            "access_category": "publisher_doi_link",
        }

    # 8. Abstract Only (disabled)
    row = _first(lambda r: r.get("access_category") == "abstract_only")
    if row:
        return {
            "button_label": "Abstract Only",
            "target_url": None,
            "disabled": True,
            "helper_text": (
                "Only the abstract is available. Configure institutional access in "
                "Settings for additional routes."
            ),
            "access_category": "abstract_only",
        }

    # 9. Access Unknown (disabled)
    return {
        "button_label": "Access Unknown",
        "target_url": None,
        "disabled": True,
        "helper_text": "No access route could be determined.",
        "access_category": "unknown",
    }


# Module-level singleton (consistent with other services in this codebase).
full_text_access_broker = FullTextAccessBrokerService()
