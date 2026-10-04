"""The evidence list for an engine search (read-only view for the results page).

Everything here comes from what the run stored: the query ledger, the
publications with their identity, admission, lineage and access, and the
run's per-source progress. Nothing is recomputed or re-ranked on read.
"""
from __future__ import annotations

from typing import Any, Optional

from clinical_engine import store
from clinical_engine.contracts.run import TERMINAL_RUN_STATUSES
from clinical_engine.retrieval import registry
from clinical_engine.retrieval.admission import EXCLUSIONS, ROLE_LABELS, STATE_LABELS
from clinical_engine.retrieval.portfolio import FAMILY_LABELS, LANE_LABELS
from clinical_engine.retrieval.run import SOURCE_LABELS

STAGES = [
    ("searching", "Searching PubMed and Europe PMC"),
    ("merging", "Checking identities and removing duplicates"),
    ("judging", "Judging relevance"),
    ("guidelines", "Checking guidelines and newer evidence"),
    ("listing", "Building the list"),
    ("access", "Checking full-text access"),
    ("done", "Done"),
]


def _authors(article: dict[str, Any]) -> str:
    names = [str(a.get("fullName") or a.get("lastName") or "") for a in article.get("authors_json") or []]
    names = [n for n in names if n]
    return ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")


def _access_view(acc: Optional[dict[str, Any]], *, pending: bool = False) -> dict[str, Any]:
    if not acc:
        # Access is checked for the listed papers while the run works; any other paper
        # (further down "More", or excluded) can be checked on request.
        return {"status": "checking" if pending else "not_checked"}
    acq = acc.get("acquired") or {}
    perm = acc.get("permitted") or {}
    readable = acc.get("readable") or {}
    if acq.get("status") == "not_checked":
        badge = "not_checked"
    elif perm.get("text_scope") == "none":
        badge = "not_permitted"
    elif acq.get("status") == "full_text":
        badge = "full_text"
    elif any(a.get("status") == "failed" for a in acc.get("attempts") or []) and acq.get("status") != "full_text":
        badge = "fetch_failed"
    elif readable.get("status") == "free_full_text":
        badge = "free_to_read"
    elif readable.get("status") == "publisher_link":
        badge = "subscription"
    elif acq.get("status") == "abstract_only":
        badge = "abstract_only"
    else:
        badge = "unknown"
    return {
        "status": badge,
        "readable": {"status": readable.get("status"), "links": readable.get("links") or [],
                     "unpaywall": readable.get("unpaywall"), "blocked_links": readable.get("blocked_links") or []},
        "acquired": {k: acq.get(k) for k in ("status", "route", "version", "completeness", "identity", "reason")},
        "permitted": {k: perm.get(k) for k in ("text_scope", "licence", "basis")},
        "used": acc.get("used") or {"status": "not_used"},
        "attempts": acc.get("attempts") or [],
    }


def card(p: dict[str, Any], *, access_pending: bool = False) -> dict[str, Any]:
    a = p.get("article") or {}
    adm = p.get("admission") or {}
    lin = p.get("lineage") or None
    state = adm.get("state")
    return {
        "id": p["id"], "row_id": p.get("legacy_row_id"), "article_id": p.get("legacy_article_id"),
        "section": p.get("section"), "position": p.get("position"),
        "title": a.get("title"), "journal": a.get("journal"), "year": a.get("publication_year"),
        "authors": _authors(a), "pmid": a.get("pmid"), "doi": a.get("doi"), "pmcid": a.get("pmcid"),
        "publication_types": (a.get("publication_types_json") or [])[:6],
        "abstract": (a.get("abstract") or "")[:1200] or None,
        "state": state, "state_label": STATE_LABELS.get(state, (state or "").replace("_", " ").capitalize()),
        "reason": adm.get("reason"), "exclusion": adm.get("exclusion"),
        "exclusion_label": EXCLUSIONS.get(adm.get("exclusion") or "", None),
        "role": adm.get("role"), "role_label": ROLE_LABELS.get(adm.get("role") or "", "Study"),
        "method": adm.get("method"), "model_state": adm.get("model_state"), "dimensions": adm.get("dimensions"),
        "reviewed_on_request": bool(adm.get("reviewed_on_request")), "previous": adm.get("previous"),
        "concepts": adm.get("concepts") or [],
        "lanes": [{"lane": lane, "label": LANE_LABELS.get(lane, lane)} for lane in p.get("lanes") or []],
        "families": [{"family": f, "label": FAMILY_LABELS.get(f, f)} for f in p.get("families") or []],
        "best_rank": p.get("best_rank") or {}, "sources": p.get("sources") or [],
        "identity": {"status": p.get("identity_status"), "conflicts": p.get("conflicts") or [],
                     "trial_ids": p.get("trial_ids") or [], "notes": p.get("projection_notes") or []},
        "lineage": lin, "newer_evidence": p.get("newer_evidence"),
        "access": _access_view(p.get("access"), pending=access_pending),
    }


def _ledger_view(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "spec_id": q["spec_id"], "lane": q["lane"], "lane_label": LANE_LABELS.get(q["lane"], q["lane"]),
        "family": q["family"], "family_label": FAMILY_LABELS.get(q["family"], q["family"]),
        "source": q["source"], "source_label": SOURCE_LABELS.get(q["source"], q["source"]),
        "query": q["query"], "purpose": q.get("purpose"), "status": q.get("status"), "error": q.get("error"),
        "count": q.get("count"), "records_read": q.get("records_read"), "max_records": q.get("max_records"),
        "truncated": q.get("truncated"), "translation": q.get("translation"), "warnings": q.get("warnings") or [],
        "pages": q.get("pages"), "duration_ms": q.get("duration_ms"), "condition": q.get("condition"),
    } for q in queries]


async def build_view(user: dict, session_id: str) -> Optional[dict[str, Any]]:
    run = await store.get_run_for_session(session_id, user["id"])
    if run is None or run.get("retrieval_adapter") != "engine":
        return None
    tasks = {t["key"]: t for t in await store.get_tasks(run["id"])}
    out = {k: (t.get("output") or {}) for k, t in tasks.items() if t.get("status") == "succeeded"}
    pubs = await store.publications_for_run(run["id"])
    queries = await store.queries_for_run(run["id"])
    progress = run.get("progress") or {}
    listed = "project" in out
    pending = run["status"] not in TERMINAL_RUN_STATUSES and progress.get("stage") != "done"
    sections: dict[str, list[dict[str, Any]]] = {k: [] for k in
                                                 ("main", "other", "more", "excluded", "held_back", "not_assessed")}
    if listed:
        for p in sorted(pubs, key=lambda x: (x.get("position") or 10**6)):
            if p.get("section") in sections and p.get("section") != "not_assessed":
                sections[p["section"]].append(card(p, access_pending=pending))
        sections["not_assessed"] = []  # counted, not listed
    guideline_cards = [card(p, access_pending=pending) for p in pubs
                       if (p.get("admission") or {}).get("role") == "guideline" and (p.get("lineage") or {}).get("lead")]
    guideline_cards.sort(key=lambda c: ({"current": 0, "older_than_window": 1, "superseded": 2}.get(
        (c["lineage"] or {}).get("currency"), 1), -((c["lineage"] or {}).get("edition_year") or 0)))
    by_state: dict[str, int] = {}
    for p in pubs:
        s = (p.get("admission") or {}).get("state")
        if s:
            by_state[s] = by_state.get(s, 0) + 1
    retrieve = out.get("retrieve") or {}
    lineage_out = out.get("lineage") or {}
    records_by_source = (out.get("union") or {}).get("union", {}).get("records_by_source") or {}
    return {
        "session_id": session_id,
        "run": {"id": run["id"], "status": run["status"], "mode": run["mode"], "error": run.get("error"),
                "created_at": run.get("created_at"), "finished_at": run.get("finished_at")},
        "stage": progress.get("stage") or "searching",
        "stages": [{"key": k, "label": label} for k, label in STAGES],
        "sources": {s: {**(progress.get("sources") or {}).get(s, {}), "label": SOURCE_LABELS.get(s, s)}
                    for s in ("pubmed", "europepmc")},
        "partial": bool(retrieve.get("partial") or progress.get("partial")),
        "listed": listed,
        "counts": {
            "records_read": {s: sum(int(q.get("records_read") or 0) for q in queries if q["source"] == s)
                             for s in ("pubmed", "europepmc")},
            "unique_records": records_by_source,
            "publications": len(pubs),
            "held_back": sum(1 for p in pubs if p.get("identity_status") == "quarantined"),
            "judged": sum(v for k, v in by_state.items() if k not in ("held_back", "not_assessed")),
            "not_assessed": by_state.get("not_assessed", 0),
            "by_state": by_state,
            "sections": {k: len(v) for k, v in sections.items()} if listed else None,
        },
        "sections": sections,
        "guidelines": guideline_cards,
        "families": lineage_out.get("lineage", {}).get("families") or [],
        "newest_comparable_guideline_year": lineage_out.get("lineage", {}).get("newest_comparable_guideline_year"),
        "update": {**(lineage_out.get("update") or {}), "anchor": lineage_out.get("anchor"),
                   "newer_evidence": lineage_out.get("newer_evidence")},
        "registry": registry.public() | {"note": "Proposed list — the owner confirms which issuers are vetted."},
        "portfolio": {"queries": _ledger_view(queries), "skipped": retrieve.get("skipped") or [],
                      "alternative": retrieve.get("alternative"), "press": retrieve.get("press")},
        "admission": (out.get("admit") or {}).get("admission"),
        "access": (out.get("access") or {}).get("access"),
        "usage": run.get("usage") or {},
    }


def _compact(p: dict[str, Any]) -> dict[str, Any]:
    from clinical_engine.retrieval.admission import design_lane, priority

    a = p.get("article") or {}
    return {
        "id": p["id"], "title": a.get("title"), "year": a.get("publication_year"), "journal": a.get("journal"),
        "authors": _authors(a), "pmid": a.get("pmid"), "doi": a.get("doi"),
        "publication_types": (a.get("publication_types_json") or [])[:4],
        "families": [{"family": f, "label": FAMILY_LABELS.get(f, f)} for f in p.get("families") or []],
        "sources": p.get("sources") or [], "search_strength": round(priority(p), 2),
        "strong_design": design_lane(p), "conflicts": p.get("conflicts") or [],
    }


async def unlisted_view(user: dict, session_id: str) -> Optional[dict[str, Any]]:
    """Papers found but not in the list: never reviewed by the model (beyond the review
    budget — unscreened, not rejected) and held back (identifiers disagree). Excluded
    papers are in the evidence view with their reasons. Strong designs come first."""
    run = await store.get_run_for_session(session_id, user["id"])
    if run is None or run.get("retrieval_adapter") != "engine":
        return None
    pubs = await store.publications_for_run(run["id"])
    from clinical_engine.retrieval.admission import design_lane, priority

    not_reviewed = sorted((p for p in pubs if p.get("section") == "not_assessed"
                           and (p.get("admission") or {}).get("state") == "not_assessed"),
                          key=lambda p: (not design_lane(p), -priority(p)))
    held_back = [p for p in pubs if p.get("identity_status") == "quarantined"]
    return {
        "session_id": session_id,
        "not_reviewed": [_compact(p) for p in not_reviewed],
        "held_back": [_compact(p) for p in held_back],
        "counts": {"not_reviewed": len(not_reviewed), "held_back": len(held_back),
                   "excluded": sum(1 for p in pubs if p.get("section") == "excluded"),
                   "reviewed_on_request": sum(1 for p in pubs if (p.get("admission") or {}).get("reviewed_on_request"))},
    }
