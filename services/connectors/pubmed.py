"""PubMed / NCBI E-utilities connector (Phase 2B).

Endpoints:
- esearch.fcgi  → list of PMIDs
- esummary.fcgi → JSON summaries (journal, pubdate, authors, ids)
- efetch.fcgi   → XML with abstracts, MeSH, publication types

NCBI rate limits:
- 3 req/s without API key
- 10 req/s with API key

NEVER scrapes publisher sites. NEVER bypasses paywalls. The pipeline guarantees
PHI is screened BEFORE this connector is invoked.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from core.db import Collections, get_db
from models.common import new_id, utcnow
from services.audit import audit_service
from services.canonical_identity import make_canonical_key, normalize_doi

logger = logging.getLogger("clinevidence.pubmed")

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DEFAULT_TIMEOUT_S = 10.0
MAX_RETRIES = 2


_GUIDELINE_TOKENS = ("guideline", "practice guideline", "consensus development conference")
_RETRACTED_TOKENS = ("retracted publication",)
_RETRACTION_NOTICE_TOKENS = ("retraction of publication", "retraction notice")
_EOC_TOKENS = ("expression of concern",)


def _strip_key_from_message(msg: str) -> str:
    """Strip api_key= values from any URL-embedded error message."""
    if not msg:
        return msg
    return re.sub(r"(api_key=)[^&\s]+", r"\1<redacted>", msg)


def _classify(pub_types_lower: list[str]) -> str:
    pts = " | ".join(pub_types_lower)
    if any(g in pts for g in _GUIDELINE_TOKENS):
        return "guideline"
    if "systematic review" in pts:
        return "systematic_review"
    if "meta-analysis" in pts or "meta analysis" in pts:
        return "meta_analysis"
    if "randomized controlled trial" in pts:
        return "rct"
    if "clinical trial" in pts:
        return "clinical_trial"
    if "comparative study" in pts and ("controlled" in pts or "trial" in pts):
        return "clinical_trial"
    if "cohort studies" in pts or "cohort study" in pts:
        return "cohort_study"
    if "case-control studies" in pts or "case-control" in pts:
        return "case_control"
    if "cross-sectional studies" in pts or "cross-sectional" in pts:
        return "cross_sectional"
    if "review" in pts:
        return "narrative_review"
    return "other"


def _full_text_status(pmcid: str | None, has_abstract: bool) -> str:
    if pmcid:
        return "full_text_available"
    if has_abstract:
        return "abstract_only"
    return "unknown"


def _canonical_key(pmid: str, doi: str, pmcid: str,
                   title: str, year: int | None,
                   journal: str, first_author: str) -> str:
    if pmid:
        return f"pmid:{pmid}"
    if doi:
        return f"doi:{doi.strip().lower()}"
    if pmcid:
        return f"pmcid:{pmcid}"
    def safe(s):
        return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:64]
    return (
        f"title_year_journal_firstauthor:"
        f"{safe(title)}_{year or 0}_{safe(journal)[:32]}_{safe(first_author)[:32]}"
    )


class _RateLimiter:
    """Simple per-second token-bucket. Asyncio-safe."""

    def __init__(self, per_second: float):
        self.per_second = max(0.1, per_second)
        self._interval = 1.0 / self.per_second
        self._next_at = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                await asyncio.sleep(self._next_at - now)
                now = time.monotonic()
            self._next_at = now + self._interval


class PubMedConnector:
    """Thin async PubMed wrapper. Read-only. No scraping."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("NCBI_API_KEY", "").strip()
        self.email = os.environ.get("NCBI_EMAIL", "").strip()
        self.tool = os.environ.get("NCBI_TOOL", "ClinEvidenceSearch").strip()
        self.timeout_s = float(os.environ.get("NCBI_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))
        self.max_results = int(os.environ.get("PUBMED_MAX_RESULTS_PER_QUERY", "50"))
        rps_no_key = float(os.environ.get("PUBMED_REQUESTS_PER_SECOND_NO_KEY", "3"))
        rps_with_key = float(os.environ.get("PUBMED_REQUESTS_PER_SECOND_WITH_KEY", "10"))
        self.rps = rps_with_key if self.api_key else rps_no_key
        self.limiter = _RateLimiter(self.rps)

    def _common_params(self) -> dict[str, str]:
        p = {"tool": self.tool}
        if self.email:
            p["email"] = self.email
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    async def search(
        self,
        queries: list[dict[str, Any]],
        *,
        client: Optional[httpx.AsyncClient] = None,
        search_session_id: Optional[str] = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return list of (query_type, normalized_article_dict) tuples.

        B5 — the per-query esearch/efetch chains run concurrently (bounded),
        but results are reassembled in the ORIGINAL query order (and original
        within-query record order), so the returned list is byte-identical to
        the former sequential loop. The shared per-second ``_RateLimiter``
        still spaces every NCBI request, so concurrency only overlaps network
        round-trips up to the allowed rate — it never exceeds NCBI limits.
        """
        owns = client is None
        if owns:
            client = httpx.AsyncClient(timeout=self.timeout_s)
        # Bound concurrency so a large query set can't open an unbounded fan-out.
        max_conc = max(1, int(os.environ.get("PUBMED_MAX_CONCURRENT_QUERIES", "5")))
        sem = asyncio.Semaphore(max_conc)

        async def _run_one(q: dict[str, Any]) -> tuple[str, list[dict[str, Any]], Optional[bool]]:
            """Returns (qtype, normalized_records, ok). ok: True=succeeded,
            False=errored, None=skipped (empty query → no status change)."""
            qtype = q.get("query_type", "unknown")
            qstring = q.get("query_string", "")
            if not qstring:
                return qtype, [], None
            async with sem:
                try:
                    pmids = await self._esearch(client, qstring)
                    if not pmids:
                        return qtype, [], True
                    records = await self._fetch_full(client, pmids)
                    norms = [n for raw in records if (n := self._normalize(raw))]
                    return qtype, norms, True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("pubmed query failed (%s): %s", qtype, exc)
                    await audit_service.log(
                        "source_query_error",
                        search_session_id=search_session_id,
                        payload={
                            "source": "pubmed",
                            "query_type": qtype,
                            "status_code": getattr(getattr(exc, "response", None), "status_code", None),
                            "truncated_error": _strip_key_from_message(str(exc))[:500],
                        },
                    )
                    return qtype, [], False

        out: list[tuple[str, dict[str, Any]]] = []
        any_ok = False
        try:
            # gather preserves input order in its result list → byte-identical out.
            per_query = await asyncio.gather(*(_run_one(q) for q in queries))
            for qtype, norms, ok in per_query:
                if ok:
                    any_ok = True
                for norm in norms:
                    out.append((qtype, norm))
        finally:
            if owns:
                await client.aclose()
        await self._update_status(success=any_ok)
        return out

    async def _update_status(self, success: bool, message: str | None = None) -> None:
        now = datetime.now(timezone.utc)
        update = {"updated_at": now}
        if success:
            update["last_success_at"] = now
            if not message:
                key_note = "Using NCBI API key (10 req/s)." if self.api_key else "No NCBI API key configured; using public rate limits (3 req/s)."
                message = key_note
            update["status_message"] = message
        else:
            update["last_error_at"] = now
            update["status_message"] = message or "all pubmed queries failed"
            update["last_error_message"] = _strip_key_from_message(message or "")[:300]
        try:
            await get_db()[Collections.SOURCE_CONFIGS].update_one(
                {"source_name": "pubmed"}, {"$set": update}
            )
        except Exception:  # noqa: BLE001
            pass

    async def _get_with_retry(
        self, client: httpx.AsyncClient, url: str, params: dict
    ) -> httpx.Response:
        await self.limiter.wait()
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = await client.get(url, params=params)
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
        raise RuntimeError("unreachable")

    async def _esearch(self, client: httpx.AsyncClient, query: str) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(self.max_results),
            **self._common_params(),
        }
        r = await self._get_with_retry(client, ESEARCH, params)
        body = r.json()
        return list((body.get("esearchresult") or {}).get("idlist") or [])

    async def _fetch_full(
        self, client: httpx.AsyncClient, pmids: list[str]
    ) -> list[dict[str, Any]]:
        """Combine esummary (JSON) + efetch (XML for abstracts) into raw records."""
        if not pmids:
            return []
        id_csv = ",".join(pmids)
        # esummary
        sum_params = {"db": "pubmed", "id": id_csv, "retmode": "json", **self._common_params()}
        r = await self._get_with_retry(client, ESUMMARY, sum_params)
        sum_body = r.json().get("result") or {}
        # efetch XML for abstract + pub types
        fetch_params = {"db": "pubmed", "id": id_csv, "rettype": "abstract", "retmode": "xml",
                        **self._common_params()}
        r2 = await self._get_with_retry(client, EFETCH, fetch_params)
        xml_records = self._parse_efetch_xml(r2.text)

        out: list[dict[str, Any]] = []
        for pmid in pmids:
            summary = sum_body.get(pmid) or {}
            if not summary:
                continue
            xml = xml_records.get(pmid) or {}
            merged = {"pmid": pmid, **summary, **xml}
            out.append(merged)
        return out

    @staticmethod
    def _parse_efetch_xml(xml_text: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not xml_text:
            return out
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return out
        for article in root.iter("PubmedArticle"):
            pmid_el = article.find("MedlineCitation/PMID")
            if pmid_el is None or not pmid_el.text:
                continue
            pmid = pmid_el.text.strip()
            # Abstract (the article's own English abstract — not OtherAbstract)
            abstract_parts: list[str] = []
            for ab in article.findall("MedlineCitation/Article/Abstract/AbstractText"):
                label = ab.attrib.get("Label")
                text = (ab.text or "").strip()
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = "\n".join([p for p in abstract_parts if p]).strip() or None
            # Publication types
            pub_types: list[str] = []
            for pt in article.findall("MedlineCitation/Article/PublicationTypeList/PublicationType"):
                if pt.text:
                    pub_types.append(pt.text.strip())
            # DOI / PMCID — ONLY the article's own ids. PubmedData/ReferenceList
            # holds the ids of the papers it cites; walking the whole record
            # (``iter("ArticleId")``) stamped the last cited paper's DOI/PMCID
            # on the article.
            doi = None
            pmcid = None
            for el in article.findall("PubmedData/ArticleIdList/ArticleId"):
                idt = (el.attrib.get("IdType") or "").lower()
                if idt == "doi" and el.text and not doi:
                    doi = el.text.strip()
                elif idt == "pmc" and el.text and not pmcid:
                    pmcid = el.text.strip()
                    if not pmcid.upper().startswith("PMC"):
                        pmcid = "PMC" + pmcid
            if not doi:
                for el in article.findall("MedlineCitation/Article/ELocationID"):
                    if (el.attrib.get("EIdType") or "").lower() == "doi" and el.text:
                        doi = el.text.strip()
                        break
            # Language
            lang_el = article.find("MedlineCitation/Article/Language")
            language = lang_el.text.strip() if lang_el is not None and lang_el.text else None
            out[pmid] = {
                "abstractText_xml": abstract,
                "pubTypeList_xml": pub_types,
                "doi_xml": doi,
                "pmcid_xml": pmcid,
                "language": language,
            }
        return out

    def _normalize(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        pmid = (raw.get("pmid") or "").strip()
        if not pmid:
            return None
        title = (raw.get("title") or raw.get("Title") or "").strip()
        # esummary 'pubdate' or 'epubdate', usually "YYYY Mon DD" or "YYYY"
        pubdate = (raw.get("pubdate") or raw.get("epubdate") or "").strip()
        pub_year: int | None = None
        m = re.match(r"(\d{4})", pubdate)
        if m:
            try:
                pub_year = int(m.group(1))
            except ValueError:
                pub_year = None
        # Authors from esummary
        authors_raw = raw.get("authors") or []
        authors_json: list[dict[str, Any]] = []
        for a in authors_raw:
            name = (a.get("name") or "").strip()
            if not name:
                continue
            # PubMed authors: "Lastname Initials" → split.
            parts = name.split(" ")
            last = parts[0] if parts else name
            initials = parts[1] if len(parts) > 1 else ""
            authors_json.append({"fullName": name, "lastName": last, "initials": initials})
        first_author = (authors_json[0]["lastName"] if authors_json else "")
        journal = (raw.get("fulljournalname") or raw.get("source") or "").strip() or None

        # IDs
        doi = raw.get("doi_xml") or ""
        pmcid = raw.get("pmcid_xml") or ""
        # Cross-check articleids array from esummary too.
        for entry in (raw.get("articleids") or []):
            idt = (entry.get("idtype") or "").lower()
            val = (entry.get("value") or "").strip()
            if not val:
                continue
            if idt == "doi" and not doi:
                doi = val
            elif idt == "pmc" and not pmcid:
                pmcid = val if val.upper().startswith("PMC") else "PMC" + val
        doi = doi or None
        pmcid = pmcid or None

        abstract = raw.get("abstractText_xml")
        pub_types_xml = raw.get("pubTypeList_xml") or []
        pub_types_esum = raw.get("pubtype") or []
        if isinstance(pub_types_esum, str):
            pub_types_esum = [pub_types_esum]
        merged_pt = []
        for p in (pub_types_xml + pub_types_esum):
            if p and p not in merged_pt:
                merged_pt.append(p)
        pub_types_lower = [p.lower() for p in merged_pt]

        is_guideline = any(t in pub_types_lower for t in _GUIDELINE_TOKENS)
        is_preprint = any("preprint" in p for p in pub_types_lower)
        is_retracted = any(t in p for p in pub_types_lower for t in _RETRACTED_TOKENS)
        is_retraction_notice = any(any(tok in p for tok in _RETRACTION_NOTICE_TOKENS) for p in pub_types_lower)
        has_eoc = any(any(tok in p for tok in _EOC_TOKENS) for p in pub_types_lower)

        article_type = _classify(pub_types_lower)
        if is_guideline:
            article_type = "guideline"

        source_urls: list[str] = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"]
        if pmcid:
            source_urls.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")
        if doi:
            source_urls.append(f"https://doi.org/{doi}")

        source_ids: dict[str, Any] = {"pubmed": pmid}
        if doi:
            source_ids["doi"] = doi
        if pmcid:
            source_ids["pmc"] = pmcid

        full_text_status = _full_text_status(pmcid, bool(abstract))

        ck = make_canonical_key(
            pmid=pmid, doi=doi or "", pmcid=pmcid or "",
            title=title, year=pub_year,
            journal=journal or "", first_author=first_author,
        )
        doi_norm = normalize_doi(doi or "") or None

        return {
            "id": new_id(),
            "canonical_key": ck,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "publication_year": pub_year,
            "publication_date": pubdate or None,
            "authors_json": authors_json,
            "doi": doi,
            "doi_normalized": doi_norm,
            "pmid": pmid,
            "pmcid": pmcid,
            "source_ids_json": source_ids,
            "publication_types_json": pub_types_lower,
            "article_type_classification": article_type,
            "is_guideline": is_guideline,
            "is_preprint": is_preprint,
            "is_retracted": is_retracted,
            "is_retraction_notice": is_retraction_notice,
            "has_expression_of_concern": has_eoc,
            "full_text_status": full_text_status,
            "source_urls_json": source_urls,
            "language": raw.get("language"),
            "created_at": utcnow(),
            "updated_at": utcnow(),
            "_retrieved_by": "pubmed",
        }


pubmed_connector = PubMedConnector()
