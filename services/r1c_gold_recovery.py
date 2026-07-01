"""Phase R1C — deterministic gold-comparator recovery strategies.

Closed-set strategy enum. Approved sources ONLY (PubMed E-utilities,
Europe PMC REST, Crossref ``/works/{DOI}`` when ``CROSSREF_MAILTO``
is set). NO scraping. NO LLM. NO fabrication of identifiers.

Strategies (executed in declared order; the FIRST that returns the
gold short-circuits the rest; ALL attempted strategies are recorded):

  1. ``exact_title``         — verbatim title search on PubMed + EPMC
  2. ``doi_anchor_term``     — DOI used as an anchor term in the
                                existing search query
  3. ``crossref_metadata_lookup`` — Crossref ``/works/{DOI}`` returns
                                title + container-title; we then use
                                that title against PubMed/EPMC to
                                resolve the PMID. Falls through when
                                ``CROSSREF_MAILTO`` is unset.
  4. ``deeper_pagination``   — page through PubMed esearch +
                                EPMC search up to 5 pages × 100/page
                                = 500 candidates per connector per
                                query family.
  5. ``synonym_expansion``   — re-run with R1B's curated expansion
                                vocabulary as additional OR'd terms.

Closed-set ``connector_recall_miss`` reasons:
  * ``not_indexed_in_pubmed_or_epmc``
  * ``connector_pagination_ceiling_exceeded_500_results``
  * ``connector_returned_empty_for_doi``
  * ``unknown_classify_manually``
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote

import httpx

from services.r1b_query_expansion import (
    CURATED_EXPANSION_MAPS,
)

logger = logging.getLogger("clinevidence.r1c_recovery")

# ---------------------------------------------------------------------
# Closed-set constants.
# ---------------------------------------------------------------------
RECOVERY_STRATEGIES: tuple[str, ...] = (
    "exact_title",
    "doi_anchor_term",
    "crossref_metadata_lookup",
    "deeper_pagination",
    "synonym_expansion",
)

RECALL_MISS_REASONS: frozenset[str] = frozenset({
    "not_indexed_in_pubmed_or_epmc",
    "connector_pagination_ceiling_exceeded_500_results",
    "connector_returned_empty_for_doi",
    "unknown_classify_manually",
})

# Closed-set: pagination knobs (per operator confirmation: 2a).
PAGE_SIZE: int = 100
MAX_PAGES: int = 5  # 5 × 100 = 500 candidates per connector per family

# Closed-set: API endpoints (approved sources only).
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CROSSREF_WORKS = "https://api.crossref.org/works"

# Closed-set: per-request HTTP timeout.
HTTP_TIMEOUT_S: float = 15.0


# ---------------------------------------------------------------------
# Per-strategy fetch helpers — closed-set inputs, closed-set outputs.
# ---------------------------------------------------------------------
async def _pubmed_search(
    client: httpx.AsyncClient,
    query: str,
    *,
    retstart: int = 0,
    retmax: int = PAGE_SIZE,
) -> list[str]:
    """One-page PubMed esearch returning a list of PMID strings."""
    params: dict[str, str] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retstart": str(int(retstart)),
        "retmax": str(int(retmax)),
    }
    key = os.environ.get("PUBMED_API_KEY", "").strip()
    if key:
        params["api_key"] = key
    try:
        r = await client.get(PUBMED_ESEARCH, params=params, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        logger.exception("r1c pubmed esearch failed (non-fatal)")
        return []
    body = r.json() or {}
    return list((body.get("esearchresult") or {}).get("idlist") or [])


async def _pubmed_summary(
    client: httpx.AsyncClient,
    pmids: list[str],
) -> dict[str, dict]:
    if not pmids:
        return {}
    params: dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(pmids[:200]),
        "retmode": "json",
    }
    key = os.environ.get("PUBMED_API_KEY", "").strip()
    if key:
        params["api_key"] = key
    try:
        r = await client.get(PUBMED_ESUMMARY, params=params, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        logger.exception("r1c pubmed esummary failed (non-fatal)")
        return {}
    # R1C.1 — PubMed eSummary's ``result`` object contains a
    # ``uids`` key whose value is a list (NOT a per-PMID dict).
    # Filter to only dict-valued entries so downstream callers can
    # safely iterate without an ``AttributeError`` from the
    # ``uids`` list.
    raw_result = (r.json() or {}).get("result") or {}
    return {
        k: v for k, v in raw_result.items()
        if isinstance(v, dict)
    }


async def _epmc_search(
    client: httpx.AsyncClient,
    query: str,
    *,
    page: int = 1,
    page_size: int = PAGE_SIZE,
) -> list[dict]:
    """One-page Europe PMC REST search returning normalized rows."""
    params: dict[str, str] = {
        "query": query,
        "format": "json",
        "pageSize": str(int(page_size)),
        "page": str(int(page)),
    }
    try:
        r = await client.get(EPMC_SEARCH, params=params, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        logger.exception("r1c epmc search failed (non-fatal)")
        return []
    body = r.json() or {}
    results = (body.get("resultList") or {}).get("result") or []
    out: list[dict] = []
    for rec in results:
        out.append({
            "pmid": rec.get("pmid"),
            "doi": (rec.get("doi") or "").lower(),
            "title": rec.get("title"),
            "journal": rec.get("journalTitle"),
            "publication_year": rec.get("pubYear"),
            "source": "europe_pmc",
            "abstract": rec.get("abstractText"),
            # Closed-set hint — used downstream for ``publication_types_json``.
            "publication_types_json": [rec.get("pubType")] if rec.get("pubType") else [],
        })
    return out


def _matches_gold(
    record: dict,
    *,
    gold_dois: list[str],
    gold_title: str = "",
) -> bool:
    """Closed-set matching by DOI (case-insensitive) OR title prefix."""
    rec_doi = (record.get("doi") or "").strip().lower()
    if rec_doi and any(rec_doi == (d or "").strip().lower() for d in gold_dois):
        return True
    if gold_title:
        rec_title = (record.get("title") or "").strip().lower()
        if rec_title and (gold_title.lower()[:60] in rec_title):
            return True
    return False


def _quote_query(s: str) -> str:
    return s.strip()


async def _crossref_doi_metadata(
    client: httpx.AsyncClient, doi: str,
) -> Optional[dict]:
    """Closed-set Crossref ``/works/{DOI}`` lookup. Returns the
    minimal closed-set metadata projection (title, container-title)
    or None when ``CROSSREF_MAILTO`` is unset / DOI not found.
    """
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    if not mailto:
        return None
    headers = {
        "User-Agent": f"ClinEvidenceSearch/R1C (mailto:{mailto})",
        "Accept": "application/json",
    }
    url = f"{CROSSREF_WORKS}/{quote(doi)}"
    try:
        r = await client.get(url, headers=headers, timeout=HTTP_TIMEOUT_S)
        if r.status_code != 200:
            return None
    except Exception:  # noqa: BLE001
        logger.exception("r1c crossref lookup failed (non-fatal)")
        return None
    body = r.json() or {}
    msg = body.get("message") or {}
    title_list = msg.get("title") or []
    container_list = msg.get("container-title") or []
    return {
        "doi": doi.lower(),
        "title": (title_list[0] if title_list else None),
        "container_title": (container_list[0] if container_list else None),
        "type": msg.get("type"),
    }


# ---------------------------------------------------------------------
# Strategy implementations — each returns a list of recovered records
# (closed-set field projection) + a "found" boolean.
# ---------------------------------------------------------------------
async def _strategy_exact_title(
    client: httpx.AsyncClient,
    *,
    gold_title: str,
    gold_dois: list[str],
) -> list[dict]:
    if not gold_title:
        return []
    q_pubmed = f'"{gold_title}"[Title]'
    q_epmc = f'TITLE:"{gold_title}"'
    pmids = await _pubmed_search(client, q_pubmed, retmax=20)
    pubmed_recs: list[dict] = []
    if pmids:
        summary = await _pubmed_summary(client, pmids)
        for pmid, rec in summary.items():
            pubmed_recs.append({
                "pmid": pmid,
                "doi": (
                    next(
                        (x.get("value") for x in (rec.get("articleids") or [])
                         if (x.get("idtype") or "").lower() == "doi"),
                        "",
                    ) or ""
                ).lower(),
                "title": rec.get("title"),
                "journal": rec.get("source"),
                "publication_year": (rec.get("pubdate") or "")[:4],
                "source": "pubmed",
                "publication_types_json": list(rec.get("pubtype") or []),
            })
    epmc_recs = await _epmc_search(client, q_epmc, page=1, page_size=20)
    candidates = pubmed_recs + epmc_recs
    return [
        c for c in candidates
        if _matches_gold(c, gold_dois=gold_dois, gold_title=gold_title)
    ]


async def _strategy_doi_anchor_term(
    client: httpx.AsyncClient,
    *,
    gold_dois: list[str],
    gold_title: str = "",
) -> list[dict]:
    """DOI used directly as an anchor in the existing search
    interface. Both PubMed and EPMC accept DOI tokens in their
    search query — the connectors return the article when indexed.
    """
    out: list[dict] = []
    for doi in (gold_dois or []):
        if not doi:
            continue
        # PubMed: DOI[AID] field. EPMC: DOI:"…" syntax.
        q_pubmed = f"{doi}[AID]"
        q_epmc = f'DOI:"{doi}"'
        pmids = await _pubmed_search(client, q_pubmed, retmax=10)
        if pmids:
            summary = await _pubmed_summary(client, pmids)
            for pmid, rec in summary.items():
                out.append({
                    "pmid": pmid,
                    "doi": doi.lower(),
                    "title": rec.get("title"),
                    "journal": rec.get("source"),
                    "publication_year": (rec.get("pubdate") or "")[:4],
                    "source": "pubmed",
                    "publication_types_json": list(rec.get("pubtype") or []),
                })
        epmc_recs = await _epmc_search(client, q_epmc, page=1, page_size=10)
        for r in epmc_recs:
            out.append(r)
    return [
        c for c in out
        if _matches_gold(c, gold_dois=gold_dois, gold_title=gold_title)
    ]


async def _strategy_crossref_metadata(
    client: httpx.AsyncClient,
    *,
    gold_dois: list[str],
    gold_title: str = "",
) -> tuple[bool, list[dict]]:
    """Closed-set 1c implementation. Returns ``(crossref_used, found_records)``.

    ``crossref_used`` is True when Crossref returned non-empty
    metadata for at least one DOI (i.e., the wired Crossref path
    actually exercised). Even when True, we still need to resolve
    PMID via PubMed/EPMC because Crossref doesn't return PMID.
    """
    crossref_used = False
    out: list[dict] = []
    for doi in (gold_dois or []):
        meta = await _crossref_doi_metadata(client, doi)
        if not meta:
            continue
        crossref_used = True
        title = (meta.get("title") or "").strip()
        if not title:
            continue
        # Resolve PMID via title.
        found = await _strategy_exact_title(
            client, gold_title=title, gold_dois=gold_dois,
        )
        out.extend(found)
    return crossref_used, [
        c for c in out
        if _matches_gold(c, gold_dois=gold_dois, gold_title=gold_title)
    ]


async def _strategy_deeper_pagination(
    client: httpx.AsyncClient,
    *,
    broad_query: str,
    gold_dois: list[str],
    gold_title: str = "",
) -> tuple[list[dict], int, int, bool]:
    """Page through PubMed esearch + EPMC search up to 5 pages × 100/page.

    Returns ``(matched_records, pubmed_pages_used, epmc_pages_used,
    ceiling_exceeded_without_match)``.
    """
    if not broad_query:
        return [], 0, 0, False
    pubmed_pages = 0
    epmc_pages = 0
    matched: list[dict] = []
    # PubMed pagination.
    for p in range(MAX_PAGES):
        pmids = await _pubmed_search(
            client, broad_query,
            retstart=p * PAGE_SIZE, retmax=PAGE_SIZE,
        )
        pubmed_pages += 1
        if not pmids:
            break
        summary = await _pubmed_summary(client, pmids)
        for pmid, rec in summary.items():
            doi_val = (
                next(
                    (x.get("value") for x in (rec.get("articleids") or [])
                     if (x.get("idtype") or "").lower() == "doi"),
                    "",
                ) or ""
            ).lower()
            cand = {
                "pmid": pmid, "doi": doi_val,
                "title": rec.get("title"),
                "journal": rec.get("source"),
                "publication_year": (rec.get("pubdate") or "")[:4],
                "source": "pubmed",
                "publication_types_json": list(rec.get("pubtype") or []),
            }
            if _matches_gold(cand, gold_dois=gold_dois, gold_title=gold_title):
                matched.append(cand)
        if matched:
            break
        # Gentle: brief pause to respect NCBI rate limits.
        await asyncio.sleep(0.1)
    # EPMC pagination (independent of PubMed result).
    if not matched:
        for p in range(1, MAX_PAGES + 1):
            recs = await _epmc_search(
                client, broad_query, page=p, page_size=PAGE_SIZE,
            )
            epmc_pages += 1
            if not recs:
                break
            for cand in recs:
                if _matches_gold(cand, gold_dois=gold_dois, gold_title=gold_title):
                    matched.append(cand)
            if matched:
                break
            await asyncio.sleep(0.1)
    ceiling_exceeded = (
        not matched
        and (pubmed_pages >= MAX_PAGES or epmc_pages >= MAX_PAGES)
    )
    return matched, pubmed_pages, epmc_pages, ceiling_exceeded


async def _strategy_synonym_expansion(
    client: httpx.AsyncClient,
    *,
    expansion_map_name: Optional[str],
    gold_dois: list[str],
    gold_title: str = "",
) -> list[dict]:
    """Re-run the R1B curated expansion vocabulary as a fresh
    OR-anchored query. Independent of the active session's query
    so it can recover articles that the original retrieval missed.
    """
    if not expansion_map_name:
        return []
    terms = CURATED_EXPANSION_MAPS.get(expansion_map_name) or []
    if not terms:
        return []
    # Closed-set: produce a quoted-OR'd query.
    parts: list[str] = []
    for t in terms:
        s = (t or "").strip()
        if not s:
            continue
        if " " in s or "-" in s or "/" in s:
            parts.append(f'"{s}"')
        else:
            parts.append(s)
    q = " OR ".join(parts)
    if not q:
        return []
    matched: list[dict] = []
    pmids = await _pubmed_search(client, q, retmax=PAGE_SIZE)
    if pmids:
        summary = await _pubmed_summary(client, pmids)
        for pmid, rec in summary.items():
            doi_val = (
                next(
                    (x.get("value") for x in (rec.get("articleids") or [])
                     if (x.get("idtype") or "").lower() == "doi"),
                    "",
                ) or ""
            ).lower()
            cand = {
                "pmid": pmid, "doi": doi_val,
                "title": rec.get("title"),
                "journal": rec.get("source"),
                "publication_year": (rec.get("pubdate") or "")[:4],
                "source": "pubmed",
                "publication_types_json": list(rec.get("pubtype") or []),
            }
            if _matches_gold(cand, gold_dois=gold_dois, gold_title=gold_title):
                matched.append(cand)
    if not matched:
        recs = await _epmc_search(client, q, page=1, page_size=PAGE_SIZE)
        for cand in recs:
            if _matches_gold(cand, gold_dois=gold_dois, gold_title=gold_title):
                matched.append(cand)
    return matched


# ---------------------------------------------------------------------
# Orchestrator — closed-set recovery flow.
# ---------------------------------------------------------------------
async def attempt_recovery(
    *,
    gold_dois: list[str],
    gold_title: str,
    broad_query_string: str,
    expansion_map_name: Optional[str],
    http_client: Optional[httpx.AsyncClient] = None,
    fetch_overrides: Optional[dict[str, Callable[..., Awaitable[Any]]]] = None,
) -> dict[str, Any]:
    """Run the closed-set recovery strategies in declared order.

    Returns:
      {
        "recovery_strategies_attempted": list[str],
        "recovery_strategy_that_surfaced_gold": Optional[str],
        "matched_records": list[dict],
        "crossref_used": bool,
        "pagination_pages_used": {"pubmed": int, "europe_pmc": int},
        "connector_recall_miss_reason": Optional[str],
      }

    The ``fetch_overrides`` parameter exists for tests so the unit
    tests can replace strategy functions with deterministic stubs
    rather than hitting real APIs.
    """
    own_client = False
    if http_client is None:
        http_client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_S,
            headers={"User-Agent": "ClinEvidenceSearch/R1C"},
        )
        own_client = True
    overrides = fetch_overrides or {}
    try:
        attempted: list[str] = []
        chosen_strategy: Optional[str] = None
        matched: list[dict] = []
        crossref_used = False
        pages_used = {"pubmed": 0, "europe_pmc": 0}
        pagination_ceiling_exceeded = False

        # --- 1. exact_title ---
        attempted.append("exact_title")
        fn1 = overrides.get("exact_title", _strategy_exact_title)
        try:
            recs = await fn1(
                http_client, gold_title=gold_title, gold_dois=gold_dois,
            )
        except Exception:  # noqa: BLE001
            recs = []
            logger.exception("r1c exact_title strategy failed (non-fatal)")
        if recs:
            chosen_strategy = "exact_title"
            matched = recs
            return _final(
                attempted=attempted, chosen=chosen_strategy,
                matched=matched, crossref_used=crossref_used,
                pages_used=pages_used,
                ceiling_exceeded=False,
            )

        # --- 2. doi_anchor_term ---
        attempted.append("doi_anchor_term")
        fn2 = overrides.get("doi_anchor_term", _strategy_doi_anchor_term)
        try:
            recs = await fn2(
                http_client, gold_dois=gold_dois, gold_title=gold_title,
            )
        except Exception:  # noqa: BLE001
            recs = []
            logger.exception("r1c doi_anchor_term strategy failed (non-fatal)")
        if recs:
            chosen_strategy = "doi_anchor_term"
            matched = recs
            return _final(
                attempted=attempted, chosen=chosen_strategy,
                matched=matched, crossref_used=crossref_used,
                pages_used=pages_used,
                ceiling_exceeded=False,
            )

        # --- 3. crossref_metadata_lookup ---
        attempted.append("crossref_metadata_lookup")
        fn3 = overrides.get("crossref_metadata_lookup", _strategy_crossref_metadata)
        try:
            crossref_used, recs = await fn3(
                http_client, gold_dois=gold_dois, gold_title=gold_title,
            )
        except Exception:  # noqa: BLE001
            recs = []
            logger.exception("r1c crossref strategy failed (non-fatal)")
        if recs:
            chosen_strategy = "crossref_metadata_lookup"
            matched = recs
            return _final(
                attempted=attempted, chosen=chosen_strategy,
                matched=matched, crossref_used=crossref_used,
                pages_used=pages_used, ceiling_exceeded=False,
            )

        # --- 4. deeper_pagination ---
        attempted.append("deeper_pagination")
        fn4 = overrides.get("deeper_pagination", _strategy_deeper_pagination)
        try:
            recs, pub_pages, epmc_pages, ceiling_hit = await fn4(
                http_client, broad_query=broad_query_string,
                gold_dois=gold_dois, gold_title=gold_title,
            )
            pages_used = {"pubmed": pub_pages, "europe_pmc": epmc_pages}
            pagination_ceiling_exceeded = bool(ceiling_hit)
        except Exception:  # noqa: BLE001
            recs = []
            logger.exception("r1c deeper_pagination strategy failed (non-fatal)")
        if recs:
            chosen_strategy = "deeper_pagination"
            matched = recs
            return _final(
                attempted=attempted, chosen=chosen_strategy,
                matched=matched, crossref_used=crossref_used,
                pages_used=pages_used, ceiling_exceeded=False,
            )

        # --- 5. synonym_expansion ---
        attempted.append("synonym_expansion")
        fn5 = overrides.get("synonym_expansion", _strategy_synonym_expansion)
        try:
            recs = await fn5(
                http_client, expansion_map_name=expansion_map_name,
                gold_dois=gold_dois, gold_title=gold_title,
            )
        except Exception:  # noqa: BLE001
            recs = []
            logger.exception("r1c synonym_expansion strategy failed (non-fatal)")
        if recs:
            chosen_strategy = "synonym_expansion"
            matched = recs
            return _final(
                attempted=attempted, chosen=chosen_strategy,
                matched=matched, crossref_used=crossref_used,
                pages_used=pages_used, ceiling_exceeded=False,
            )

        # All strategies exhausted — classify the miss reason.
        miss_reason: str
        if pagination_ceiling_exceeded:
            miss_reason = "connector_pagination_ceiling_exceeded_500_results"
        elif gold_dois and not crossref_used:
            miss_reason = "connector_returned_empty_for_doi"
        else:
            miss_reason = "not_indexed_in_pubmed_or_epmc"
        return _final(
            attempted=attempted, chosen=None, matched=[],
            crossref_used=crossref_used, pages_used=pages_used,
            ceiling_exceeded=pagination_ceiling_exceeded,
            miss_reason=miss_reason,
        )
    finally:
        if own_client:
            await http_client.aclose()


def _final(
    *,
    attempted: list[str],
    chosen: Optional[str],
    matched: list[dict],
    crossref_used: bool,
    pages_used: dict[str, int],
    ceiling_exceeded: bool,
    miss_reason: Optional[str] = None,
) -> dict[str, Any]:
    """Closed-set finalizer — assembles the recovery result dict."""
    return {
        "recovery_strategies_attempted": list(attempted),
        "recovery_strategy_that_surfaced_gold": chosen,
        "matched_records": list(matched),
        "crossref_used": bool(crossref_used),
        "pagination_pages_used": dict(pages_used),
        "connector_recall_miss_reason": miss_reason,
        "pagination_ceiling_exceeded": bool(ceiling_exceeded),
    }


__all__ = [
    "RECOVERY_STRATEGIES",
    "RECALL_MISS_REASONS",
    "PAGE_SIZE",
    "MAX_PAGES",
    "attempt_recovery",
]
