"""Stage trace for benchmark references (W2.2): where each reference record got
to in an engine run — or the stage at which it was lost, and why — so a
missing benchmark record is never a silent miss.

  retrieved  the query hits that found it (family, source, rank)
  not found  each executed PubMed query is asked whether it matches the record
             at all (``(<query>) AND <pmid>[uid]``): a match means the record
             ranked beyond the records read; when no query matches, the
             approved concept that excludes it is named (or the filters)
  merged     the publication it joined and its identity status
  judged     admission state and reason (or not assessed: beyond the budget)
  listed     section and position

Count-only source calls (nothing is fetched). For offline evaluation, not the
request path.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

from clinical_engine import store
from clinical_engine.retrieval import pubmed as pubmed_src
from clinical_engine.retrieval.http import SourceError, transport_or_default
from clinical_engine.retrieval.portfolio import FAMILY_LABELS
from services import engine_concept_plan as ecp

LISTED = ("main", "other", "more")


def _same(article: dict[str, Any], pmid: str, doi: Optional[str]) -> bool:
    if pmid and str(article.get("pmid") or "") == pmid:
        return True
    return bool(doi) and (article.get("doi_normalized") or (article.get("doi") or "").lower()) == doi


async def _matches(client: httpx.AsyncClient, query: str, pmid: str) -> Optional[bool]:
    try:
        return (await pubmed_src.esearch(client, f"({query}) AND {pmid}[uid]", retmax=0)).count > 0
    except SourceError:
        return None


async def _lost_before_retrieval(client: httpx.AsyncClient, pmid: str, ledger: list[dict[str, Any]],
                                 plan: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not pmid:
        return {"lost_at": "query", "detail": "No PMID to test the PubMed queries against."}
    exists = await _matches(client, "all[sb]", pmid)
    if exists is False:
        return {"lost_at": "not_in_pubmed", "detail": f"PMID {pmid} is not in PubMed."}
    matched = []
    for q in ledger:
        if q["source"] != "pubmed" or q.get("status") not in ("ok", "zero", "partial"):
            continue
        if await _matches(client, q["query"], pmid):
            matched.append(q)
    if matched:
        best = matched[0]
        return {"lost_at": "reading_limit",
                "detail": f"Matched by {FAMILY_LABELS.get(best['family'], best['family'])} (PubMed) but ranked beyond "
                          f"the {best.get('records_read')} records read of {best.get('count')}.",
                "matching_queries": [m["family"] for m in matched]}
    failing = []
    for c in (plan or {}).get("concepts") or []:
        if not c.get("required"):
            continue
        if await _matches(client, ecp.concept_block(c, plan, "pubmed"), pmid) is False:
            failing.append(c.get("term"))
    if failing:
        return {"lost_at": "query",
                "detail": "No query matches it: the record has neither the MeSH heading nor any approved wording for "
                          + ", ".join(f"“{t}”" for t in failing) + " in its title/abstract.",
                "concepts_missing": failing}
    return {"lost_at": "filters",
            "detail": "Every approved concept matches, but the search limits (years, animal-only, preprint) exclude it."}


async def trace(run_id: str, refs: list[dict[str, Any]], *, plan: Optional[dict[str, Any]] = None,
                check_sources: bool = True,
                transport: Optional[httpx.AsyncBaseTransport] = None) -> list[dict[str, Any]]:
    """One entry per reference: ``stage`` reached (listed / judged / merged / retrieved)
    or ``lost_at`` (not_in_pubmed / query / filters / reading_limit / identity /
    admission / budget), with the evidence for it."""
    ledger = await store.queries_for_run(run_id)
    cands = await store.candidates_for_run(run_id)
    pubs = await store.publications_for_run(run_id)
    spec_family = {q["spec_id"]: q["family"] for q in ledger}
    out: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=20.0, transport=transport_or_default(transport)) as client:
        for ref in refs:
            pmid = str(ref.get("pmid") or "").strip()
            doi = (ref.get("doi") or "").strip().lower() or None
            t: dict[str, Any] = {"pmid": pmid or None, "title": ref.get("title"), "role": ref.get("role")}
            hits = [(c["source"], h) for c in cands if _same(c.get("article") or {}, pmid, doi) for h in c["hits"]]
            t["retrieved"] = [{"source": s, "family": h.get("family") or spec_family.get(h.get("spec_id")),
                               "lane": h.get("lane"), "rank": h.get("rank")} for s, h in hits]
            pub = next((p for p in pubs if _same(p.get("article") or {}, pmid, doi)), None)
            if not hits and pub is None:
                t.update(await _lost_before_retrieval(client, pmid, ledger, plan) if check_sources else
                         {"lost_at": "query", "detail": "Not returned by any query (source check skipped)."})
                out.append(t)
                continue
            if pub is None:
                t.update(lost_at="identity", detail="Retrieved but not in the union.")
                out.append(t)
                continue
            adm = pub.get("admission") or {}
            t.update(publication_id=pub["id"], identity=pub.get("identity_status"), state=adm.get("state"),
                     reason=adm.get("reason"), role=adm.get("role") or t["role"], section=pub.get("section"),
                     position=pub.get("position"))
            if pub.get("identity_status") == "quarantined":
                t.update(lost_at="identity", detail="Held back: its identifiers disagree between sources.")
            elif adm.get("state") == "not_assessed":
                t.update(lost_at="budget", detail="Not reviewed by the model: beyond the review budget "
                                                  "(lower search rank than the papers reviewed).")
            elif pub.get("section") in LISTED:
                t["stage"] = "listed"
            elif (pub.get("lineage") or {}).get("edition_id") and not (pub.get("lineage") or {}).get("lead"):
                # A co-publication or summary of a guideline edition is listed under its lead record.
                lead = next((x for x in pubs if (x.get("lineage") or {}).get("edition_id")
                             == pub["lineage"]["edition_id"] and (x.get("lineage") or {}).get("lead")), None)
                t.update(stage="listed", section=(lead or {}).get("section"), position=(lead or {}).get("position"),
                         detail="Listed under its edition's lead record (same work, another journal).")
            else:
                t.update(lost_at="admission", detail=f"Judged {adm.get('state')}: {adm.get('reason') or ''}".strip())
            out.append(t)
    return out


def summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for t in traces:
        k = "listed:" + str(t.get("section")) if t.get("stage") == "listed" else "lost:" + str(t.get("lost_at"))
        counts[k] = counts.get(k, 0) + 1
    return {"references": len(traces), "in_main": sum(1 for t in traces if t.get("section") == "main"),
            "by_stage": counts}
