"""Execute a query portfolio against PubMed and Europe PMC (W2.2).

* Sources run in parallel; each has a record budget and a time budget.
* Every query is written to the ledger with what actually happened: status,
  the source's hit count and (PubMed) its translation of the query, records
  read, pages, and whether the result was cut at the budget.
* A failed or timed-out query is ``failed`` / ``skipped`` — never "no results" —
  and the source's status becomes ``partial`` or ``failed``, so an outage is
  visible as a partial search. A planned query left out because the source's
  reading limit was reached is ``skipped`` with that reason; it is planned
  behaviour, not an outage, so it does not make the source partial.
* Every planned query gets a ledger row, including the ones never sent.
* Each record is stored once per source with every query hit (family, lane and
  rank) that found it.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import httpx

from clinical_engine import store
from clinical_engine.config import settings
from clinical_engine.retrieval import citations, europepmc, http, pubmed
from clinical_engine.retrieval.http import SourceError
from models.common import new_id

SOURCES = ("pubmed", "europepmc")
SOURCE_LABELS = {"pubmed": "PubMed", "europepmc": "Europe PMC"}


@dataclass
class QuerySpec:
    id: str
    lane: str        # aggregated | foundational_primary | recent_primary | alternative | update
    family: str      # main | original_wording | narrow | broad | guidelines | systematic_reviews | ...
    source: str      # pubmed | europepmc
    query: str
    purpose: str
    max_records: int
    condition: Optional[str] = None  # None (always) | "if_sparse"
    notes: list[str] = field(default_factory=list)
    sort: Optional[str] = None       # None (relevance) | "cited" (most-cited first: the landmark lane)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryOutcome:
    spec_id: str
    source: str
    status: str      # ok | zero | partial | failed | skipped
    count: Optional[int] = None
    records_read: int = 0
    error: Optional[str] = None


ProgressFn = Callable[[str, dict[str, Any]], Any]
# Every source query goes through the run's tool gateway (approval check + call budget).
TOOL_FOR_SOURCE = {"pubmed": "pubmed.search", "europepmc": "europepmc.search"}


def record_key_pubmed(rec: dict[str, Any]) -> str:
    return f"pmid:{rec['pmid']}"


def record_key_epmc(raw: dict[str, Any]) -> str:
    return f"{(raw.get('source') or 'EPMC').upper()}:{raw.get('id') or raw.get('pmid') or raw.get('doi')}"


async def _store_records(run_id: str, spec: QuerySpec, records: list[tuple[str, dict[str, Any]]],
                         start_rank: int) -> int:
    items = [(key, doc, {"spec_id": spec.id, "lane": spec.lane, "family": spec.family, "rank": start_rank + i})
             for i, (key, doc) in enumerate(records)]
    await store.add_candidate_hits(run_id, spec.source, items)
    return len(items)


# The most-cited lane ranks this many of the query's best matches by citations (E-utilities' limit).
CITED_POOL = 10_000


async def _run_pubmed_cited(client: httpx.AsyncClient, run_id: str, spec: QuerySpec, ledger: dict[str, Any]) -> None:
    page = await pubmed.esearch(client, spec.query, retmax=CITED_POOL)
    ledger.update({"count": page.count, "translation": page.translation,
                   "warnings": page.warnings[:10], "errors": page.errors[:10], "pool": len(page.ids)})
    if not page.ids:
        return
    try:
        counts = await citations.citation_counts(client, page.ids)
    except SourceError as exc:  # the citation service, not PubMed, failed
        raise SourceError("citation_unavailable", exc.code) from exc
    top = sorted(page.ids, key=lambda p: -counts.get(p, 0))[: spec.max_records]
    recs = {r["pmid"]: r for r in await pubmed.efetch(client, top)}
    docs = []
    for pmid in top:
        rec = recs.get(pmid)
        if rec is None:
            continue
        try:
            article = pubmed.to_article(rec)
        except SourceError:
            continue
        docs.append((record_key_pubmed(rec), {"record_id": new_id(), "article": article, "citation_count": counts.get(pmid),
                                              "pubmed": {k: rec.get(k) for k in (
                                                  "mesh", "nct", "entrez_date", "pub_date", "article_date", "links",
                                                  "keywords", "language", "publication_types")}}))
    await _store_records(run_id, spec, docs, start_rank=1)
    ledger.update({"records_read": len(docs), "pages": 1, "truncated": page.count > len(docs)})


async def _run_pubmed(client: httpx.AsyncClient, run_id: str, spec: QuerySpec, ledger: dict[str, Any]) -> None:
    if spec.sort == "cited":
        return await _run_pubmed_cited(client, run_id, spec, ledger)
    read = 0
    page_no = 0
    count: Optional[int] = None
    while read < spec.max_records:
        page = await pubmed.esearch(client, spec.query, retstart=read,
                                    retmax=min(pubmed.PAGE_SIZE, spec.max_records - read))
        page_no += 1
        if count is None:
            count = page.count
            ledger.update({"count": page.count, "translation": page.translation,
                           "warnings": page.warnings[:10], "errors": page.errors[:10]})
        if not page.ids:
            break
        recs = await pubmed.efetch(client, page.ids)
        docs = []
        for rec in recs:
            try:
                article = pubmed.to_article(rec)
            except SourceError:
                continue
            docs.append((record_key_pubmed(rec), {"record_id": new_id(), "article": article, "pubmed": {
                k: rec.get(k) for k in ("mesh", "nct", "entrez_date", "pub_date", "article_date", "links",
                                        "keywords", "language", "publication_types")}}))
        await _store_records(run_id, spec, docs, start_rank=read + 1)
        read += len(page.ids)
        ledger["records_read"] = read
        ledger["pages"] = page_no
        if read >= (count or 0):
            break
    ledger["truncated"] = bool(count and read < count)


async def _run_europepmc(client: httpx.AsyncClient, run_id: str, spec: QuerySpec, ledger: dict[str, Any]) -> None:
    read = 0
    page_no = 0
    cursor: Optional[str] = "*"
    count: Optional[int] = None
    while cursor and read < spec.max_records:
        page = await europepmc.search_page(client, spec.query, cursor=cursor,
                                           page_size=min(europepmc.PAGE_SIZE, spec.max_records - read),
                                           sort="CITED desc" if spec.sort == "cited" else None)
        page_no += 1
        if count is None:
            count = page.count
            ledger["count"] = page.count
        docs = []
        for raw in page.records[: spec.max_records - read]:
            docs.append((record_key_epmc(raw), {"record_id": new_id(), "article": europepmc.to_article(raw),
                                                "epmc": europepmc.record_extras(raw)}))
        await _store_records(run_id, spec, docs, start_rank=read + 1)
        read += len(docs)
        ledger["records_read"] = read
        ledger["pages"] = page_no
        cursor = page.next_cursor if read < (count or 0) else None
    ledger["truncated"] = bool(count and read < count)


_RUNNERS = {"pubmed": _run_pubmed, "europepmc": _run_europepmc}


async def run_one(client: httpx.AsyncClient, run_id: str, spec: QuerySpec) -> QueryOutcome:
    qid = new_id()
    started = time.perf_counter()
    ledger: dict[str, Any] = {"count": None, "translation": None, "warnings": [], "errors": [],
                              "records_read": 0, "pages": 0, "truncated": False}
    await store.insert_query({
        "id": qid, "run_id": run_id, "spec_id": spec.id, "lane": spec.lane, "family": spec.family,
        "source": spec.source, "query": spec.query, "purpose": spec.purpose, "max_records": spec.max_records,
        "condition": spec.condition, "status": "running", **ledger, "error": None,
        "created_at": store.now(),
    })
    status, error = "ok", None
    try:
        await _RUNNERS[spec.source](client, run_id, spec, ledger)
        status = "ok" if ledger["records_read"] else "zero"
    except SourceError as exc:
        error = exc.code
        status = "partial" if ledger["records_read"] else "failed"
        if exc.code == "citation_unavailable":
            # The most-cited lane is an extra; when its citation service is down the
            # source still answered every relevance search — not an outage.
            status, error = "skipped", "optional_unavailable:citations"
    except asyncio.CancelledError:
        await store.update_query(qid, {**ledger, "status": "skipped", "error": "source_time_budget",
                                       "duration_ms": int((time.perf_counter() - started) * 1000),
                                       "finished_at": store.now()})
        raise
    await store.update_query(qid, {**ledger, "status": status, "error": error,
                                   "duration_ms": int((time.perf_counter() - started) * 1000),
                                   "finished_at": store.now()})
    return QueryOutcome(spec.id, spec.source, status, ledger["count"], ledger["records_read"], error)


async def _ledger_not_sent(run_id: str, spec: QuerySpec, code: str) -> None:
    """A ledger row for a planned query that was never sent (reading or time limit)."""
    await store.insert_query({
        "id": new_id(), "run_id": run_id, "spec_id": spec.id, "lane": spec.lane, "family": spec.family,
        "source": spec.source, "query": spec.query, "purpose": spec.purpose, "max_records": spec.max_records,
        "condition": spec.condition, "status": "skipped", "error": code, "count": None, "translation": None,
        "warnings": [], "errors": [], "records_read": 0, "pages": 0, "truncated": False,
        "created_at": store.now(), "finished_at": store.now(),
    })


def _source_status(outcomes: list[QueryOutcome], total: int, timed_out: bool) -> str:
    done = [o for o in outcomes if o.status != "skipped"]
    bad = [o for o in done if o.status in ("failed", "partial")]
    if done and len(bad) == len(done) and not any(o.records_read for o in done):
        return "failed"
    at_limit = sum(1 for o in outcomes if o.status == "skipped"
                   and (o.error == "source_record_budget" or (o.error or "").startswith("optional_unavailable")))
    if timed_out or bad or len(done) + at_limit < total:
        return "partial"
    return "complete"


async def run_portfolio(run_id: str, specs: list[QuerySpec], *,
                        progress: Optional[ProgressFn] = None,
                        transport: Optional[httpx.AsyncBaseTransport] = None,
                        call_tool: Optional[Callable[..., Any]] = None) -> dict[str, Any]:
    """Run every spec (sources in parallel, each within its budgets).
    Returns the per-source summary written to the run."""
    caps = {"pubmed": settings.RETRIEVAL_MAX_RECORDS_PUBMED, "europepmc": settings.RETRIEVAL_MAX_RECORDS_EUROPEPMC}

    async def run_source(source: str, source_specs: list[QuerySpec]) -> dict[str, Any]:
        outcomes: list[QueryOutcome] = []
        state = {"status": "searching", "queries_total": len(source_specs), "queries_done": 0,
                 "records_read": 0, "failed": 0, "errors": []}
        if progress:
            await progress(source, dict(state))
        budget_left = caps[source]
        in_flight = 0
        budget = asyncio.Condition()
        sem = asyncio.Semaphore(max(1, settings.RETRIEVAL_CONCURRENCY_PER_SOURCE))
        timed_out = False

        async def one(spec: QuerySpec) -> None:
            nonlocal budget_left, in_flight
            async with sem:
                async with budget:
                    # Queries still running hand back what they did not read; wait for
                    # them before deciding the source's reading limit is used up.
                    await budget.wait_for(lambda: budget_left > 0 or in_flight == 0)
                    at_limit = budget_left <= 0
                    if not at_limit:
                        spec.max_records = min(spec.max_records, budget_left)
                        budget_left -= spec.max_records
                        in_flight += 1
                if at_limit:
                    outcomes.append(QueryOutcome(spec.id, source, "skipped", error="source_record_budget"))
                    return
                outcome: Optional[QueryOutcome] = None
                try:
                    if call_tool is not None:
                        outcome = await call_tool(TOOL_FOR_SOURCE[source], run_one, client, run_id, spec)
                    else:
                        outcome = await run_one(client, run_id, spec)
                finally:
                    async with budget:
                        in_flight -= 1
                        if outcome is not None:
                            budget_left += spec.max_records - outcome.records_read  # return what was not used
                        budget.notify_all()
                outcomes.append(outcome)
                state["queries_done"] += 1
                state["records_read"] += outcome.records_read
                if outcome.status in ("failed", "partial"):
                    state["failed"] += 1
                    state["errors"] = (state["errors"] + [outcome.error or "error"])[-5:]
                if progress:
                    await progress(source, dict(state))

        async with httpx.AsyncClient(timeout=settings.RETRIEVAL_HTTP_TIMEOUT_S,
                                     transport=http.transport_or_default(transport)) as client:
            try:
                await asyncio.wait_for(asyncio.gather(*(one(s) for s in source_specs)),
                                       timeout=settings.RETRIEVAL_SOURCE_SECONDS)
            except asyncio.TimeoutError:
                timed_out = True
                ran = {o.spec_id for o in outcomes}
                outcomes.extend(QueryOutcome(s.id, source, "skipped", error="source_time_budget")
                                for s in source_specs if s.id not in ran)
        # Planned queries that were never sent still appear in the ledger, with the reason.
        logged = {q["spec_id"] for q in await store.queries_for_run(run_id) if q["source"] == source}
        reasons = {o.spec_id: o.error for o in outcomes if o.status == "skipped"}
        for s in source_specs:
            if s.id not in logged:
                await _ledger_not_sent(run_id, s, reasons.get(s.id) or "source_time_budget")
        state["status"] = _source_status(outcomes, len(source_specs), timed_out)
        state["timed_out"] = timed_out
        state["outcomes"] = [asdict(o) for o in outcomes]
        if progress:
            await progress(source, {k: v for k, v in state.items() if k != "outcomes"})
        return state

    by_source: dict[str, list[QuerySpec]] = {}
    for spec in specs:
        by_source.setdefault(spec.source, []).append(spec)
    results = await asyncio.gather(*(run_source(s, by_source[s]) for s in SOURCES if s in by_source))
    return dict(zip([s for s in SOURCES if s in by_source], results))
