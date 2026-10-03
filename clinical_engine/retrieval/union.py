"""Identity-safe union of the retrieved records (W2.2).

Every PubMed and Europe PMC record goes through the W1.3 resolver: records are
merged only when their identifiers agree (PMID/DOI/PMCID, compatible titles and
years); disagreeing records are quarantined — held back and listed, never
merged. Each publication keeps every query hit that found it (source, lane,
family, rank), and separate reports of one trial are linked by NCT number.
"""
from __future__ import annotations

from typing import Any

from clinical_engine import store
from clinical_engine.identity.resolver import report_role, resolve

ENGINE_SOURCE = {"pubmed": "pubmed", "europepmc": "europepmc"}


def _resolver_input(c: dict[str, Any]) -> dict[str, Any]:
    a = c.get("article") or {}
    nct = ((c.get("pubmed") or {}).get("nct") or [None])[0]
    return {
        "record_id": c["record_id"], "source": ENGINE_SOURCE.get(c["source"], c["source"]),
        "pmid": a.get("pmid"), "doi": a.get("doi"), "pmcid": a.get("pmcid"), "nct": nct,
        "title": a.get("title"), "journal": a.get("journal"), "year": a.get("publication_year"),
        "publication_date": a.get("publication_date"), "abstract": a.get("abstract"),
        "publication_types": a.get("publication_types_json") or [],
    }


def _best_rank(hits: list[dict[str, Any]], family: str = "main") -> dict[str, int]:
    out: dict[str, int] = {}
    for h in hits:
        if h.get("family") == family:
            src = h["source"]
            out[src] = min(out.get(src, 10**6), int(h.get("rank") or 10**6))
    return out


def publication_docs(run_id: str, candidates: list[dict[str, Any]], *, origin: str) -> tuple[list[dict], dict]:
    """(publication documents, union summary) — pure, no I/O."""
    by_record = {c["record_id"]: c for c in candidates}
    report = resolve([_resolver_input(c) for c in candidates])
    study_of: dict[str, list[str]] = {}
    for st in report.studies:
        for r in st.reports:
            study_of.setdefault(r.publication_id, []).extend(st.registration_ids)
    docs = []
    for pub in report.publications:
        members = [by_record[r] for r in pub.merged_from if r in by_record]
        pm = next((m for m in members if m["source"] == "pubmed"), None)
        ep = next((m for m in members if m["source"] == "europepmc"), None)
        lead = pm or ep or members[0]
        hits = [{**h, "source": m["source"]} for m in members for h in (m.get("hits") or [])]
        docs.append({
            "id": pub.id, "run_id": run_id, "origin": origin,
            "publication": pub.model_dump(mode="json"),
            "identity_status": pub.identity_status,
            "conflicts": [c.model_dump(mode="json") for c in pub.identity_conflicts],
            "record_ids": list(pub.merged_from),
            "sources": sorted({m["source"] for m in members}),
            "article": lead.get("article") or {},
            "pubmed": (pm or {}).get("pubmed") or {},
            "epmc": (ep or {}).get("epmc") or {},
            "hits": hits,
            "lanes": sorted({h["lane"] for h in hits}),
            "families": sorted({h["family"] for h in hits}),
            "best_rank": _best_rank(hits),
            "query_hits": len(hits),
            "report_role": report_role(pub),
            "trial_ids": sorted(set(study_of.get(pub.id, []))),
            "created_at": store.now(),
        })
    summary = {
        "records": len(candidates),
        "records_by_source": {s: sum(1 for c in candidates if c["source"] == s) for s in ("pubmed", "europepmc")},
        "publications": len(report.publications),
        "merged_groups": report.merged_groups,
        "quarantined_records": len(report.quarantined_record_ids),
        "conflicts": report.conflicts[:20],
        "studies_linked": len(report.studies),
    }
    return docs, summary


async def build(run_id: str, *, origin: str = "initial", only_records: set[str] | None = None) -> dict[str, Any]:
    candidates = await store.candidates_for_run(run_id)
    if only_records is not None:
        candidates = [c for c in candidates if c["record_id"] in only_records]
    docs, summary = publication_docs(run_id, candidates, origin=origin)
    await store.replace_publications(run_id, docs, origin=origin)
    return summary
