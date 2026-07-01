"""Phase 5B/5C — ScholarLikeCandidateExpansionService.

Scholar-LIKE behavior (citation-graph + related works + trial-acronym
expansion + recent title/abstract expansion) using ONLY whitelisted
structured APIs:
- PubMed E-utilities (elink) — service surface kept for future wire-up
- Europe PMC (citations/references/search endpoints) — Phase 5C wired
- OpenAlex (related_works, referenced_works, cited_by) — Phase 5C wired
- Existing in-database article references (no external scraping)

Hard rules:
- PHI-blocked sessions → refuses to run, returns empty payload.
- Per-mode caps overridable via env (defaults from Phase 5B Task 7).
- Every expanded candidate gets a closed-set ``candidate_source`` token.
- Every external call is audited via ``audit_service.log`` with
  closed-set reason codes only. No raw question, no raw URLs, no PHI.
- Mode gating: ``SCHOLAR_LIKE_EXPANSION_MODE`` env var.
    - "off"    → noop, returns empty
    - "shadow" → returns candidates BUT caller persists them as
                  ``shadow_only=True``; UI ranking unchanged
    - "active" → returns candidates AND caller merges into ranking

Closed-set HTTP error reason codes for audit:
    http_timeout | http_5xx | http_4xx | parse_error | rate_limited |
    unknown
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import httpx

from services.audit import audit_service

logger = logging.getLogger("clinevidence.scholar_expansion")


# ---------------------------------------------------------------------------
# Closed-set candidate-source codes.
# ---------------------------------------------------------------------------
CANDIDATE_SOURCES = {
    "openalex_related",
    "openalex_referenced",
    "openalex_cited_by",
    "europe_pmc_citations",
    "europe_pmc_references",
    "europe_pmc_similar",
    "pubmed_elink_related",
    "pubmed_elink_referenced",
    "pubmed_elink_cited_by",
    "trial_acronym",
    "title_abstract_recent",
}

# Closed-set HTTP error reason codes.
HTTP_ERROR_CODES = {
    "http_timeout", "http_5xx", "http_4xx",
    "parse_error", "rate_limited", "unknown",
}


# ---------------------------------------------------------------------------
# Env-overridable caps + URLs.
# ---------------------------------------------------------------------------
def _cap(env: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(env, str(default))))
    except (TypeError, ValueError):
        return default


def _fcap(env: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(env, str(default))))
    except (TypeError, ValueError):
        return default


DEFAULT_CAPS = {
    # Per-session candidate quotas from Phase 5B Task 7.
    "scholar_like_citation_expansion": _cap("Q_QUOTA_CITATION_EXPANSION", 100),
    "scholar_like_related_works": _cap("Q_QUOTA_RELATED_WORKS", 100),
    "trial_acronym_expansion": _cap("Q_QUOTA_TRIAL_ACRONYM", 50),
    "title_abstract_recent": _cap("Q_QUOTA_TIAB_RECENT", 50),
    # Network call budget — protects against runaway.
    "max_api_calls_per_session": _cap("EXPANSION_MAX_API_CALLS", 20),
    "seed_articles_max": _cap("EXPANSION_SEED_ARTICLES_MAX", 5),
    # Phase 5C — per-source + total candidate caps.
    "max_candidates_per_source_per_session": _cap(
        "EXPANSION_MAX_CANDIDATES_PER_SOURCE_PER_SESSION", 100,
    ),
    "max_total_candidates_per_session": _cap(
        "EXPANSION_MAX_TOTAL_CANDIDATES_PER_SESSION", 300,
    ),
    "seed_count": _cap("EXPANSION_SEED_COUNT", 5),
    "openalex_related_per_seed": _cap("EXPANSION_OPENALEX_RELATED_PER_SEED", 10),
    "openalex_referenced_per_seed": _cap("EXPANSION_OPENALEX_REFERENCED_PER_SEED", 10),
    "openalex_cited_by_per_seed": _cap("EXPANSION_OPENALEX_CITED_BY_PER_SEED", 10),
    "europe_pmc_citations_per_seed": _cap("EXPANSION_EPMC_CITATIONS_PER_SEED", 10),
    "europe_pmc_references_per_seed": _cap("EXPANSION_EPMC_REFERENCES_PER_SEED", 10),
    "europe_pmc_similar_per_seed": _cap("EXPANSION_EPMC_SIMILAR_PER_SEED", 5),
}

HTTP_TIMEOUT_SECONDS = _fcap("EXPANSION_HTTP_TIMEOUT_SECONDS", 10.0)
HTTP_MAX_RETRIES = _cap("EXPANSION_HTTP_MAX_RETRIES", 2)
HTTP_BACKOFF_BASE_SECONDS = _fcap("EXPANSION_HTTP_BACKOFF_BASE_SECONDS", 0.5)

OPENALEX_BASE = "https://api.openalex.org"
EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def expansion_mode() -> str:
    """Return one of {"off", "shadow", "active"}. Default = shadow in dev.

    Phase 5C.1 — even if ``SCHOLAR_LIKE_EXPANSION_MODE=active`` is set,
    we downgrade to ``shadow`` whenever ``SCHOLAR_LIKE_ACTIVE_ALLOWED``
    is NOT explicitly truthy. This guarantees the spec's invariant that
    active mode never reaches the live clinician pipeline without an
    explicit, audit-trail-creating opt-in by an operator.
    """
    m = (os.environ.get("SCHOLAR_LIKE_EXPANSION_MODE", "shadow") or "shadow").strip().lower()
    if m not in {"off", "shadow", "active"}:
        return "shadow"
    if m == "active" and not _active_allowed():
        return "shadow"
    return m


def _active_allowed() -> bool:
    return (os.environ.get("SCHOLAR_LIKE_ACTIVE_ALLOWED", "false")
             or "false").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExpandedCandidate:
    """A single candidate added by Scholar-like expansion."""
    candidate_source: str
    seed_pmid: str | None = None
    seed_id: str | None = None
    pmid: str | None = None
    doi: str | None = None
    pmcid: str | None = None
    title: str | None = None
    abstract: str | None = None
    publication_year: int | None = None
    openalex_id: str | None = None
    rationale: str = ""
    candidate_tier: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_source": self.candidate_source,
            "candidate_tier": self.candidate_tier,
            "seed_pmid": self.seed_pmid,
            "seed_id": self.seed_id,
            "pmid": self.pmid,
            "doi": self.doi,
            "pmcid": self.pmcid,
            "title": self.title,
            "abstract": (self.abstract or "")[:2000] if self.abstract else None,
            "publication_year": self.publication_year,
            "openalex_id": self.openalex_id,
            "rationale": self.rationale,
        }


@dataclass
class ExpansionResult:
    mode: str
    candidates_by_tier: dict[str, list[ExpandedCandidate]] = field(default_factory=dict)
    seed_articles_used: list[dict[str, Any]] = field(default_factory=list)
    counts_by_source: dict[str, int] = field(default_factory=dict)
    # Per-source HTTP audit: source → {called, succeeded, failed, error_codes: {...}}
    audit_by_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    api_calls: int = 0
    expansion_latency_ms: int = 0
    candidates_merged_into_ranking_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def _audit(self, source: str) -> dict[str, Any]:
        return self.audit_by_source.setdefault(source, {
            "called": 0, "succeeded": 0, "failed": 0,
            "dedup_filtered": 0,
            "error_codes": {},
        })

    def to_dict(self) -> dict[str, Any]:
        tiers: dict[str, list[dict[str, Any]]] = {}
        for tier, cs in self.candidates_by_tier.items():
            tiers[tier] = [c.to_dict() for c in cs]
        per_tier_counts = {tier: len(cs) for tier, cs in self.candidates_by_tier.items()}
        return {
            "mode": self.mode,
            "candidates_by_tier": tiers,
            "per_tier_counts": per_tier_counts,
            "seed_articles_used_for_expansion": self.seed_articles_used,
            "related_cited_reference_candidates_added": dict(self.counts_by_source),
            "audit_by_source": self.audit_by_source,
            "expansion_audit_json": self.audit_by_source,  # alias for diagnostics surface
            "api_calls": self.api_calls,
            "expansion_latency_ms": self.expansion_latency_ms,
            "candidates_merged_into_ranking_count": self.candidates_merged_into_ranking_count,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _classify_http_error(exc: BaseException) -> str:
    """Return a closed-set HTTP error reason code."""
    if isinstance(exc, asyncio.TimeoutError) or isinstance(exc, httpx.TimeoutException):
        return "http_timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code if exc.response is not None else 0
        if code == 429:
            return "rate_limited"
        if 400 <= code < 500:
            return "http_4xx"
        if 500 <= code < 600:
            return "http_5xx"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        # JSON parsing / schema issues.
        return "parse_error"
    return "unknown"


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """GET with retries on 429/5xx. Raises on final failure."""
    last_exc: Exception | None = None
    for attempt in range(HTTP_MAX_RETRIES + 1):
        try:
            r = await client.get(url, params=params or {})
            if r.status_code in (429, 500, 502, 503, 504) and attempt < HTTP_MAX_RETRIES:
                await asyncio.sleep(HTTP_BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            r.raise_for_status()
            return r
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            last_exc = exc
            if attempt >= HTTP_MAX_RETRIES:
                raise
            await asyncio.sleep(HTTP_BACKOFF_BASE_SECONDS * (2 ** attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("_get_with_retry: unreachable")


def _resolve_openalex_abstract(inv_idx: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as inverted index. Resolve to flat text."""
    if not inv_idx:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idx_list in inv_idx.items():
        for i in idx_list:
            positions.append((i, word))
    positions.sort(key=lambda p: p[0])
    return " ".join(w for _, w in positions)


def _strip_openalex_id(url: str) -> str:
    """`https://openalex.org/W12345` → `W12345`. Pass-through otherwise."""
    if not url:
        return ""
    return url.replace("https://openalex.org/", "").replace("http://openalex.org/", "").strip()


def _strip_ids(ids: dict[str, Any] | None) -> dict[str, str]:
    """Normalize an OpenAlex Work's ids block to plain strings."""
    out: dict[str, str] = {}
    if not ids:
        return out
    doi = ids.get("doi")
    if doi:
        out["doi"] = str(doi).replace("https://doi.org/", "").lower()
    pmid = ids.get("pmid")
    if pmid:
        out["pmid"] = (
            str(pmid)
            .replace("https://pubmed.ncbi.nlm.nih.gov/", "")
            .rstrip("/")
        )
    pmcid = ids.get("pmcid")
    if pmcid:
        out["pmcid"] = (
            str(pmcid)
            .replace("https://www.ncbi.nlm.nih.gov/pmc/articles/", "")
            .rstrip("/")
        )
    return out


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ScholarLikeCandidateExpansionService:
    """Phase 5C — live OpenAlex + Europe PMC expansion."""

    async def expand(
        self,
        *,
        seed_articles: Iterable[dict[str, Any]],
        picott: dict[str, Any] | None,
        is_phi_blocked: bool = False,
        mode: str | None = None,
        client: Optional[httpx.AsyncClient] = None,
        search_session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate scholar-like candidate expansions.

        Each call is best-effort: it never throws, and on any error it
        records a warning string and returns whatever partial result it
        managed to build. Per-source try/except — an OpenAlex failure
        does NOT prevent Europe PMC retrieval from completing, and vice
        versa.
        """
        t0 = time.time()
        eff_mode = (mode or expansion_mode()).lower()
        result = ExpansionResult(mode=eff_mode)

        if is_phi_blocked:
            result.warnings.append("phi_blocked_no_expansion")
            return result.to_dict()
        if eff_mode == "off":
            result.warnings.append("expansion_mode_off")
            return result.to_dict()

        seeds_list = list(seed_articles or [])[: DEFAULT_CAPS["seed_count"]]
        result.seed_articles_used = [
            self._seed_meta(s) for s in seeds_list
        ]

        # Deterministic (always-on) expansion tiers from Phase 5B.
        trial_candidates = self._expand_trial_acronyms(picott)
        if trial_candidates:
            self._absorb(result, "trial_acronym_expansion",
                         trial_candidates[: DEFAULT_CAPS["trial_acronym_expansion"]])
        ta_candidates = self._expand_title_abstract_recent(picott)
        if ta_candidates:
            self._absorb(result, "title_abstract_recent",
                         ta_candidates[: DEFAULT_CAPS["title_abstract_recent"]])

        # Make sure these tier buckets always exist for diagnostics.
        result.candidates_by_tier.setdefault("scholar_like_citation_expansion", [])
        result.candidates_by_tier.setdefault("scholar_like_related_works", [])

        # ---- Live HTTP expansion (per-source try/except) ----
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        try:
            # OpenAlex
            try:
                await self._expand_openalex(
                    client, seeds_list, result,
                    search_session_id=search_session_id, user_id=user_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception("openalex expansion top-level failed (non-fatal)")
                result.warnings.append("openalex_expansion_failed")
            # Europe PMC
            try:
                await self._expand_europe_pmc(
                    client, seeds_list, picott, result,
                    search_session_id=search_session_id, user_id=user_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception("europe_pmc expansion top-level failed (non-fatal)")
                result.warnings.append("europe_pmc_expansion_failed")
        finally:
            if owns_client:
                await client.aclose()

        # Total-cap enforcement: trim tier buckets in stable order.
        self._enforce_total_cap(result)

        result.expansion_latency_ms = int((time.time() - t0) * 1000)
        return result.to_dict()

    # ------------------------------------------------------------------
    # Seed meta extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _seed_meta(s: dict[str, Any]) -> dict[str, Any]:
        art = s.get("_article") or s
        title = art.get("title") or s.get("title") or ""
        return {
            "id": art.get("id") or s.get("id"),
            "pmid": art.get("pmid") or s.get("pmid"),
            "doi": art.get("doi") or s.get("doi"),
            "pmcid": art.get("pmcid") or s.get("pmcid"),
            "title": (title[:120] if title else None),
        }

    # ------------------------------------------------------------------
    # OpenAlex
    # ------------------------------------------------------------------
    async def _expand_openalex(
        self,
        client: httpx.AsyncClient,
        seeds: list[dict[str, Any]],
        result: ExpansionResult,
        *,
        search_session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        mailto = (os.environ.get("OPENALEX_MAILTO") or "").strip()
        params = {"mailto": mailto} if mailto else {}
        api_key = (os.environ.get("OPENALEX_API_KEY") or "").strip()
        if api_key:
            params["api_key"] = api_key

        for seed in seeds:
            if result.api_calls >= DEFAULT_CAPS["max_api_calls_per_session"]:
                result.warnings.append("openalex_api_call_budget_exhausted")
                return
            seed_meta = self._seed_meta(seed)
            work = await self._openalex_resolve_seed(
                client, seed_meta, params, result,
                search_session_id=search_session_id, user_id=user_id,
            )
            if not work:
                continue
            await self._openalex_emit_related_referenced(
                client, work, seed_meta, params, result,
                search_session_id=search_session_id,
            )
            await self._openalex_emit_cited_by(
                client, work, seed_meta, params, result,
                search_session_id=search_session_id,
            )

    async def _openalex_resolve_seed(
        self,
        client: httpx.AsyncClient,
        seed_meta: dict[str, Any],
        params: dict[str, str],
        result: ExpansionResult,
        *,
        search_session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any] | None:
        """Resolve seed metadata to a full OpenAlex Work record."""
        # Try DOI → PMID → PMCID
        urls: list[str] = []
        if seed_meta.get("doi"):
            urls.append(f"{OPENALEX_BASE}/works/doi:{seed_meta['doi']}")
        if seed_meta.get("pmid"):
            urls.append(f"{OPENALEX_BASE}/works/pmid:{seed_meta['pmid']}")
        if seed_meta.get("pmcid"):
            urls.append(f"{OPENALEX_BASE}/works/pmcid:{seed_meta['pmcid']}")
        for url in urls:
            audit = result._audit("openalex_seed")
            audit["called"] += 1
            result.api_calls += 1
            try:
                r = await _get_with_retry(client, url, params)
                audit["succeeded"] += 1
                return r.json()
            except Exception as exc:  # noqa: BLE001
                code = _classify_http_error(exc)
                audit["failed"] += 1
                audit["error_codes"][code] = audit["error_codes"].get(code, 0) + 1
                await audit_service.log(
                    "expansion_http_error",
                    user_id=user_id, search_session_id=search_session_id,
                    payload={
                        "source": "openalex", "stage": "seed_lookup",
                        "reason_code": code,
                    },
                )
                continue
        return None

    async def _openalex_emit_related_referenced(
        self,
        client: httpx.AsyncClient,
        work: dict[str, Any],
        seed_meta: dict[str, Any],
        params: dict[str, str],
        result: ExpansionResult,
        *,
        search_session_id: Optional[str] = None,
    ) -> None:
        related_ids = [_strip_openalex_id(x) for x in (work.get("related_works") or [])]
        related_ids = [x for x in related_ids if x][: DEFAULT_CAPS["openalex_related_per_seed"]]
        referenced_ids = [_strip_openalex_id(x) for x in (work.get("referenced_works") or [])]
        referenced_ids = [x for x in referenced_ids if x][: DEFAULT_CAPS["openalex_referenced_per_seed"]]

        async def _fetch_and_emit(ids: list[str], candidate_source: str, tier: str):
            audit = result._audit(candidate_source)
            for oa_id in ids:
                if result.api_calls >= DEFAULT_CAPS["max_api_calls_per_session"]:
                    result.warnings.append(
                        f"{candidate_source}_api_call_budget_exhausted"
                    )
                    return
                audit["called"] += 1
                result.api_calls += 1
                try:
                    r = await _get_with_retry(
                        client, f"{OPENALEX_BASE}/works/{oa_id}", params,
                    )
                    audit["succeeded"] += 1
                    cand = self._openalex_work_to_candidate(
                        r.json(), candidate_source=candidate_source,
                        candidate_tier=tier, seed_meta=seed_meta,
                    )
                    if cand:
                        self._absorb_one(result, tier, candidate_source, cand)
                except Exception as exc:  # noqa: BLE001
                    code = _classify_http_error(exc)
                    audit["failed"] += 1
                    audit["error_codes"][code] = audit["error_codes"].get(code, 0) + 1
                    await audit_service.log(
                        "expansion_http_error",
                        search_session_id=search_session_id,
                        payload={
                            "source": "openalex", "stage": candidate_source,
                            "reason_code": code,
                        },
                    )

        await _fetch_and_emit(
            related_ids, "openalex_related", "scholar_like_related_works",
        )
        await _fetch_and_emit(
            referenced_ids, "openalex_referenced", "scholar_like_citation_expansion",
        )

    async def _openalex_emit_cited_by(
        self,
        client: httpx.AsyncClient,
        work: dict[str, Any],
        seed_meta: dict[str, Any],
        params: dict[str, str],
        result: ExpansionResult,
        *,
        search_session_id: Optional[str] = None,
    ) -> None:
        oa_id = _strip_openalex_id(work.get("id") or "")
        if not oa_id:
            return
        audit = result._audit("openalex_cited_by")
        if result.api_calls >= DEFAULT_CAPS["max_api_calls_per_session"]:
            return
        audit["called"] += 1
        result.api_calls += 1
        try:
            r = await _get_with_retry(
                client, f"{OPENALEX_BASE}/works",
                {**params, "filter": f"cites:{oa_id}",
                 "per_page": str(DEFAULT_CAPS["openalex_cited_by_per_seed"])},
            )
            audit["succeeded"] += 1
            for w in (r.json().get("results") or []):
                cand = self._openalex_work_to_candidate(
                    w, candidate_source="openalex_cited_by",
                    candidate_tier="scholar_like_citation_expansion",
                    seed_meta=seed_meta,
                )
                if cand:
                    self._absorb_one(
                        result, "scholar_like_citation_expansion",
                        "openalex_cited_by", cand,
                    )
        except Exception as exc:  # noqa: BLE001
            code = _classify_http_error(exc)
            audit["failed"] += 1
            audit["error_codes"][code] = audit["error_codes"].get(code, 0) + 1
            await audit_service.log(
                "expansion_http_error",
                search_session_id=search_session_id,
                payload={
                    "source": "openalex", "stage": "cited_by",
                    "reason_code": code,
                },
            )

    @staticmethod
    def _openalex_work_to_candidate(
        work: dict[str, Any], *, candidate_source: str, candidate_tier: str,
        seed_meta: dict[str, Any],
    ) -> ExpandedCandidate | None:
        if not work:
            return None
        ids = _strip_ids(work.get("ids") or {})
        title = work.get("title") or work.get("display_name") or ""
        pub_year = work.get("publication_year")
        abstract = _resolve_openalex_abstract(work.get("abstract_inverted_index"))
        oa_id = _strip_openalex_id(work.get("id") or "")
        return ExpandedCandidate(
            candidate_source=candidate_source,
            candidate_tier=candidate_tier,
            seed_pmid=seed_meta.get("pmid"),
            seed_id=seed_meta.get("id"),
            pmid=ids.get("pmid"),
            doi=ids.get("doi"),
            pmcid=ids.get("pmcid"),
            title=title[:240] if title else None,
            abstract=abstract[:2000] if abstract else None,
            publication_year=int(pub_year) if isinstance(pub_year, int) else None,
            openalex_id=oa_id,
            rationale=f"openalex {candidate_source} of seed pmid={seed_meta.get('pmid') or '—'}",
        )

    # ------------------------------------------------------------------
    # Europe PMC
    # ------------------------------------------------------------------
    async def _expand_europe_pmc(
        self,
        client: httpx.AsyncClient,
        seeds: list[dict[str, Any]],
        picott: dict[str, Any] | None,
        result: ExpansionResult,
        *,
        search_session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        for seed in seeds:
            if result.api_calls >= DEFAULT_CAPS["max_api_calls_per_session"]:
                result.warnings.append("europe_pmc_api_call_budget_exhausted")
                return
            seed_meta = self._seed_meta(seed)
            pmid = seed_meta.get("pmid")
            if pmid:
                await self._europe_pmc_emit(
                    client, "citations", pmid, seed_meta, result,
                    search_session_id=search_session_id,
                )
                await self._europe_pmc_emit(
                    client, "references", pmid, seed_meta, result,
                    search_session_id=search_session_id,
                )

        # Similar / canonical-concept-driven search (one global call,
        # not per-seed).
        await self._europe_pmc_similar(
            client, picott, result,
            search_session_id=search_session_id,
        )

    async def _europe_pmc_emit(
        self,
        client: httpx.AsyncClient,
        kind: str,  # "citations" | "references"
        pmid: str,
        seed_meta: dict[str, Any],
        result: ExpansionResult,
        *,
        search_session_id: Optional[str] = None,
    ) -> None:
        candidate_source = f"europe_pmc_{kind}"
        if kind == "citations":
            tier = "scholar_like_citation_expansion"
            page_size = DEFAULT_CAPS["europe_pmc_citations_per_seed"]
        else:
            tier = "scholar_like_citation_expansion"
            page_size = DEFAULT_CAPS["europe_pmc_references_per_seed"]
        audit = result._audit(candidate_source)
        if result.api_calls >= DEFAULT_CAPS["max_api_calls_per_session"]:
            return
        audit["called"] += 1
        result.api_calls += 1
        try:
            r = await _get_with_retry(
                client,
                f"{EUROPE_PMC_BASE}/article/MED/{pmid}/{kind}",
                {"format": "json", "pageSize": str(page_size)},
            )
            audit["succeeded"] += 1
            body = r.json()
            key = "citationList" if kind == "citations" else "referenceList"
            items = ((body.get(key) or {}).get(kind[:-1])
                      or (body.get(key) or {}).get(kind)
                      or [])
            for it in items[:page_size]:
                cand = self._europe_pmc_record_to_candidate(
                    it, candidate_source=candidate_source, candidate_tier=tier,
                    seed_meta=seed_meta,
                )
                if cand:
                    self._absorb_one(result, tier, candidate_source, cand)
        except Exception as exc:  # noqa: BLE001
            code = _classify_http_error(exc)
            audit["failed"] += 1
            audit["error_codes"][code] = audit["error_codes"].get(code, 0) + 1
            await audit_service.log(
                "expansion_http_error",
                search_session_id=search_session_id,
                payload={"source": "europe_pmc", "stage": kind,
                          "reason_code": code},
            )

    async def _europe_pmc_similar(
        self,
        client: httpx.AsyncClient,
        picott: dict[str, Any] | None,
        result: ExpansionResult,
        *,
        search_session_id: Optional[str] = None,
    ) -> None:
        if not picott:
            return
        must_haves = picott.get("must_have_concepts") or []
        anchors = [
            f'"{(m.get("concept") or "").strip()}"'
            for m in must_haves[:3]
            if (m.get("concept") or "").strip()
        ]
        if not anchors:
            return
        query = " AND ".join(anchors) + " AND HAS_ABSTRACT:y NOT SRC:PPR"
        audit = result._audit("europe_pmc_similar")
        if result.api_calls >= DEFAULT_CAPS["max_api_calls_per_session"]:
            return
        audit["called"] += 1
        result.api_calls += 1
        try:
            r = await _get_with_retry(
                client, f"{EUROPE_PMC_BASE}/search",
                {"query": query, "format": "json", "resultType": "core",
                  "pageSize": str(DEFAULT_CAPS["europe_pmc_similar_per_seed"])},
            )
            audit["succeeded"] += 1
            items = ((r.json().get("resultList") or {}).get("result") or [])
            for it in items[: DEFAULT_CAPS["europe_pmc_similar_per_seed"]]:
                cand = self._europe_pmc_record_to_candidate(
                    it,
                    candidate_source="europe_pmc_similar",
                    candidate_tier="scholar_like_related_works",
                    seed_meta={"pmid": None, "id": None, "doi": None,
                                "pmcid": None, "title": None},
                )
                if cand:
                    self._absorb_one(
                        result, "scholar_like_related_works",
                        "europe_pmc_similar", cand,
                    )
        except Exception as exc:  # noqa: BLE001
            code = _classify_http_error(exc)
            audit["failed"] += 1
            audit["error_codes"][code] = audit["error_codes"].get(code, 0) + 1
            await audit_service.log(
                "expansion_http_error",
                search_session_id=search_session_id,
                payload={"source": "europe_pmc", "stage": "similar",
                          "reason_code": code},
            )

    @staticmethod
    def _europe_pmc_record_to_candidate(
        rec: dict[str, Any], *, candidate_source: str, candidate_tier: str,
        seed_meta: dict[str, Any],
    ) -> ExpandedCandidate | None:
        if not rec:
            return None
        pmid = (rec.get("pmid") or rec.get("PMID") or rec.get("id") or "").strip()
        doi = (rec.get("doi") or rec.get("DOI") or "").strip().lower()
        pmcid = (rec.get("pmcid") or rec.get("PMCID") or "").strip()
        title = rec.get("title") or ""
        abstract = rec.get("abstractText") or ""
        pub_year = rec.get("pubYear") or rec.get("publicationYear")
        try:
            pub_year_int = int(pub_year) if pub_year else None
        except (TypeError, ValueError):
            pub_year_int = None
        if not (pmid or doi or pmcid or title):
            return None
        return ExpandedCandidate(
            candidate_source=candidate_source,
            candidate_tier=candidate_tier,
            seed_pmid=seed_meta.get("pmid"),
            seed_id=seed_meta.get("id"),
            pmid=pmid or None,
            doi=doi or None,
            pmcid=pmcid or None,
            title=title[:240] if title else None,
            abstract=abstract[:2000] if abstract else None,
            publication_year=pub_year_int,
            rationale=(
                f"europe_pmc {candidate_source.split('_', 2)[-1]} of seed "
                f"pmid={seed_meta.get('pmid') or '—'}"
            ),
        )

    # ------------------------------------------------------------------
    # Deterministic-only tiers
    # ------------------------------------------------------------------
    def _expand_trial_acronyms(self, picott: dict[str, Any] | None) -> list[ExpandedCandidate]:
        if not picott:
            return []
        out: list[ExpandedCandidate] = []
        for rc in (picott.get("ranking_concepts") or []):
            concept = (rc.get("concept") or "").strip()
            if not concept:
                continue
            if concept.endswith(" trial") or (
                concept.isupper() and 2 <= len(concept) <= 8
            ):
                out.append(ExpandedCandidate(
                    candidate_source="trial_acronym",
                    candidate_tier="trial_acronym_expansion",
                    title=concept,
                    rationale=f"trial-acronym expansion seed: {concept}",
                ))
        return out

    def _expand_title_abstract_recent(
        self, picott: dict[str, Any] | None,
    ) -> list[ExpandedCandidate]:
        if not picott:
            return []
        canonicals: list[str] = []
        for mh in (picott.get("must_have_concepts") or []):
            c = (mh.get("concept") or "").strip()
            if c:
                canonicals.append(c)
        return [
            ExpandedCandidate(
                candidate_source="title_abstract_recent",
                candidate_tier="title_abstract_recent",
                title=c,
                rationale=f"title/abstract recent-5y expansion seed: {c}",
            )
            for c in canonicals[:6]
        ]

    # ------------------------------------------------------------------
    # Absorb / cap helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _absorb(
        result: ExpansionResult, tier: str, candidates: list[ExpandedCandidate],
    ) -> None:
        bucket = result.candidates_by_tier.setdefault(tier, [])
        for c in candidates:
            bucket.append(c)
            result.counts_by_source[c.candidate_source] = (
                result.counts_by_source.get(c.candidate_source, 0) + 1
            )

    @staticmethod
    def _absorb_one(
        result: ExpansionResult, tier: str, candidate_source: str,
        candidate: ExpandedCandidate,
    ) -> None:
        per_src_cap = DEFAULT_CAPS["max_candidates_per_source_per_session"]
        current = result.counts_by_source.get(candidate_source, 0)
        if current >= per_src_cap:
            audit = result._audit(candidate_source)
            audit["dedup_filtered"] = audit.get("dedup_filtered", 0) + 1
            return
        bucket = result.candidates_by_tier.setdefault(tier, [])
        bucket.append(candidate)
        result.counts_by_source[candidate_source] = current + 1

    def _enforce_total_cap(self, result: ExpansionResult) -> None:
        total = sum(len(v) for v in result.candidates_by_tier.values())
        cap = DEFAULT_CAPS["max_total_candidates_per_session"]
        if total <= cap:
            return
        # Trim deterministically: largest tier first, until total == cap.
        order = sorted(
            result.candidates_by_tier.items(),
            key=lambda kv: -len(kv[1]),
        )
        excess = total - cap
        for tier, lst in order:
            if excess <= 0:
                break
            if not lst:
                continue
            n = min(excess, len(lst))
            del lst[-n:]
            excess -= n
        result.warnings.append(
            f"total_candidate_cap_enforced:{cap}"
        )


scholar_like_expansion = ScholarLikeCandidateExpansionService()
