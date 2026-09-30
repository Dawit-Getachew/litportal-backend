"""Search report: how a search from approved terms was built (read-only).

Everything comes from what was stored, never recomputed: the approved question
version, the approved search terms, the plan handed to the search session
(checked against the approval's terms hash), the query strings the pipeline
actually executed, and per-paper provenance (which searches and sources found
each paper). Optional live counts re-run the executed main queries as counts.
"""
from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import quote

from clinical_engine import store
from clinical_engine.contracts.question import ClinicalQuestionMap
from clinical_engine.frameworks import FRAMEWORK_LABELS, INTENT_FOR_TYPE, TYPE_LABELS, field_view
from clinical_engine.interpretation.present import chips
from clinical_engine.search_terms.contracts import SearchTerms
from clinical_engine.search_terms.plan import hit_counts
from services.engine_concept_plan import _words as plan_words

SOURCE_LABELS = {"pubmed": "PubMed", "europe_pmc": "Europe PMC"}
# Plain-language names for the query families the approved-terms plan runs
# (services/engine_concept_plan.py), plus the pipeline's recovery family.
FAMILY_INFO: dict[str, tuple[str, str]] = {
    "mesh_keyword": ("Main search", "Every “must appear” concept, AND-ed. Each concept = its MeSH heading OR its "
                                    "words in the title/abstract."),
    "high_precision": ("Main search + outcomes", "The main search narrowed to papers that also mention an outcome."),
    "study_type_filtered": ("Study designs", "The main search narrowed to the study designs that answer this kind of "
                                             "question."),
    "guideline_review": ("Guidelines & reviews", "The main search narrowed to guidelines, systematic reviews and "
                                                 "meta-analyses."),
    "recent_title_abstract": ("Last 5 years", "The main search limited to the last 5 years."),
    "cochrane_review": ("Cochrane reviews", "The main search within the Cochrane Database of Systematic Reviews "
                                            "(PubMed-indexed)."),
    "rescue_query": ("Recovery search", "Run automatically when the main searches found too little."),
}
_FAMILY_ORDER = list(FAMILY_INFO)


def _per_query_cap(source: str) -> int:
    """How many records the connector reads per query (same env as the connectors)."""
    if source == "pubmed":
        return int(os.environ.get("PUBMED_MAX_RESULTS_PER_QUERY", "50") or 50)
    return int(os.environ.get("EUROPEPMC_PAGE_SIZE", "25") or 25)


def _open_url(source: str, query: str) -> Optional[str]:
    if source == "pubmed":
        return f"https://pubmed.ncbi.nlm.nih.gov/?term={quote(query, safe='')}"
    if source == "europe_pmc":
        return f"https://europepmc.org/search?query={quote(query, safe='')}"
    return None


def _mesh_url(ui: Optional[str]) -> Optional[str]:
    return f"https://meshb.nlm.nih.gov/record/ui?ui={quote(ui)}" if ui else None


def _question_section(cqm: ClinicalQuestionMap, version: dict) -> dict[str, Any]:
    by_id = {c.id: c for c in chips(cqm)}
    fields = []
    for f in field_view(cqm, age_placeholder=False):
        values = [by_id[i].display for i in f["chip_ids"]
                  if i in by_id and by_id[i].state in ("active", "excluded")]
        if values:
            fields.append({"label": f["label"], "values": values})
    intent = INTENT_FOR_TYPE.get(cqm.question_type)
    ia = version.get("interpretation_approval") or {}
    return {
        "version": version["version"],
        "question_text": cqm.question_text,
        "summary": cqm.summary,
        "question_type": cqm.question_type,
        "question_type_label": TYPE_LABELS.get(cqm.question_type, cqm.question_type),
        "intent_label": intent.label if intent else None,
        "framework": cqm.framework,
        "framework_label": FRAMEWORK_LABELS.get(cqm.framework, cqm.framework),
        "fields": fields,
        "suggestions_not_applied": [c.display for c in by_id.values() if c.state == "suggestion"],
        "interpretation_method": (version.get("interpretation") or {}).get("method"),
        "approved_at": ia.get("approved_at") or (version.get("approval") or {}).get("approved_at"),
    }


def _terms_section(terms: SearchTerms) -> dict[str, Any]:
    concepts = []
    for c in terms.concepts:
        mesh = None
        if c.mesh is not None:
            mesh = {"descriptor": c.mesh.descriptor, "ui": c.mesh.ui, "match": c.mesh.match,
                    "enabled": c.mesh.enabled, "scope_note": c.mesh.scope_note, "url": _mesh_url(c.mesh.ui)}
        concepts.append({
            "label": c.label, "role": c.role, "term": c.term, "required": c.required, "mesh": mesh,
            "included": [{"text": t.text, "kind": t.kind, "source": t.source} for t in c.terms if t.enabled],
            "not_included": [{"text": t.text, "kind": t.kind, "source": t.source} for t in c.terms if not t.enabled],
        })
    return {
        "version": terms.version,
        "use_synonyms": terms.use_synonyms,
        "filters": terms.filters.model_dump(),
        "concepts": concepts,
        "mapping": {"mesh": terms.mapping.mesh, "synonyms": terms.mapping.synonyms, "model": terms.mapping.model},
    }


def _shown_counts(rows: list[dict]) -> tuple[int, int]:
    """(kept, broader matches added) as the results list shows them — mirrors
    routers/search_sessions.get_session: top 500 rows by rank, and when few are
    kept, the best excluded rows are added as broader matches."""
    top = sorted(rows, key=lambda r: -float(r.get("overall_rank") or 0))[:500]
    kept = sum(1 for r in top if not r.get("exclusion_reason"))
    excluded = len(top) - kept
    min_included = int(os.environ.get("BROAD_RESULTS_FALLBACK_MIN_INCLUDED", "20") or "20")
    limit = int(os.environ.get("BROAD_RESULTS_FALLBACK_LIMIT", "50"))
    broader = min(excluded, max(1, limit)) if excluded and kept <= min_included else 0
    return kept, broader


def _boolean_section(plan: dict[str, Any], *, outcome_search_ran: bool) -> dict[str, Any]:
    """The search logic exactly as handed to the pipeline (the stored plan), with
    the words the query builder itself produces for each concept."""
    use_syn = bool(plan.get("use_synonyms", True))
    blocks = [{"role": c.get("role"), "term": c["term"], "mesh": c.get("mesh"), "words": plan_words(c, plan)}
              for c in plan.get("concepts") or [] if c.get("required")]
    # Outcomes that need not appear still narrow the "Main search + outcomes" search.
    ranking = [{"role": c.get("role"), "term": c["term"], "mesh": c.get("mesh"), "words": plan_words(c, plan),
                "in_outcome_search": outcome_search_ran and c.get("role") == "outcome"}
               for c in plan.get("concepts") or [] if not c.get("required")]
    f = plan.get("filters") or {}
    return {
        "blocks": blocks,
        "ranking_only": ranking,
        "use_synonyms": use_syn,
        "filters": {
            "publication_years": f.get("publication_years"),
            "exclude_animal_only": f.get("exclude_animal_only", True),
            "exclude_preprints": True,
        },
    }


async def build_report(user: dict, session_id: str, *, live_counts: bool = False) -> Optional[dict[str, Any]]:
    """None when the session was not started from the question review (legacy search)."""
    from core.db import Collections, get_db

    user_id = user["id"]
    run = await store.get_run_for_session(session_id, user_id)
    if run is None:
        return None
    version = await store.get_version_by_id(run["question_version_id"])
    if version is None:
        return None
    db = get_db()
    session = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "user_id": user_id}, {"_id": 0, "engine_concept_plan": 1, "status": 1, "created_at": 1},
    )
    if session is None:
        return None
    cqm = ClinicalQuestionMap.model_validate(version["map"])
    terms = SearchTerms.model_validate(version["search_terms"]) if version.get("search_terms") else None
    approval = version.get("approval") or {}
    plan = session.get("engine_concept_plan") if isinstance(session.get("engine_concept_plan"), dict) else None

    query_rows = [q async for q in db[Collections.SEARCH_QUERIES].find(
        {"search_session_id": session_id},
        {"_id": 0, "source": 1, "query_family": 1, "query_string": 1, "planner_source": 1,
         "is_fallback_strategy": 1, "created_at": 1},
    ).sort("created_at", 1).limit(100)]
    articles = [a async for a in db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": session_id},
        {"_id": 0, "query_types_matched": 1, "sources_present": 1, "exclusion_reason": 1, "overall_rank": 1},
    ).limit(5000)]

    found_by_family: dict[str, int] = {}
    for a in articles:
        for qt in set(a.get("query_types_matched") or []):
            fam = qt[3:] if qt.startswith("cb_") else qt
            found_by_family[fam] = found_by_family.get(fam, 0) + 1

    families: dict[str, dict[str, Any]] = {}
    for q in query_rows:
        fam = q.get("query_family") or "other"
        label, description = FAMILY_INFO.get(fam, (fam.replace("_", " ").capitalize(), ""))
        entry = families.setdefault(fam, {
            "family": fam, "label": label, "description": description,
            "papers_found": found_by_family.get(fam, 0), "queries": [],
        })
        src = q.get("source") or "europe_pmc"
        entry["queries"].append({
            "source": src, "source_label": SOURCE_LABELS.get(src, src), "query": q.get("query_string") or "",
            "open_url": _open_url(src, q.get("query_string") or ""),
            "planner_source": q.get("planner_source"), "fallback": bool(q.get("is_fallback_strategy")),
        })
    ordered = sorted(families.values(), key=lambda f: (_FAMILY_ORDER.index(f["family"])
                                                       if f["family"] in _FAMILY_ORDER else 99))

    sources = sorted({q.get("source") or "europe_pmc" for q in query_rows})
    per_source = {s: {
        "label": SOURCE_LABELS.get(s, s),
        "queries_run": sum(1 for q in query_rows if (q.get("source") or "europe_pmc") == s),
        "records_read_per_query": _per_query_cap(s),
        "papers_found": sum(1 for a in articles if s in (a.get("sources_present") or [])),
    } for s in sources}
    excluded = sum(1 for a in articles if a.get("exclusion_reason"))
    shown_kept, broader = _shown_counts(articles)

    counts = None
    if live_counts:
        main = {q["source"]: q["query"] for f in ordered if f["family"] == "mesh_keyword" for q in f["queries"]}
        counts = await hit_counts({"pubmed": main.get("pubmed"), "europe_pmc": main.get("europe_pmc")})

    return {
        "session_id": session_id,
        "question_id": run["question_id"],
        "run": {"id": run["id"], "mode": run["mode"], "status": run["status"],
                "created_at": run.get("created_at"), "finished_at": run.get("finished_at")},
        "session_status": session.get("status"),
        "question": _question_section(cqm, version),
        "search_terms": _terms_section(terms) if terms is not None else None,
        "approval": {
            "mode": approval.get("mode"), "approved_at": approval.get("approved_at"),
            "search_terms_version": approval.get("search_terms_version"),
        },
        # True when the search ran exactly the approved terms (hash of the plan on
        # the session == hash recorded at approval).
        "plan_matches_approval": bool(plan and approval.get("search_terms_hash")
                                      and plan.get("terms_hash") == approval.get("search_terms_hash")),
        "boolean": (_boolean_section(plan, outcome_search_ran="high_precision" in families)
                    if plan else None),
        "searches": ordered,
        "results": {
            "sources": per_source,
            "unique_papers": len(articles),
            "excluded": excluded,
            "kept": len(articles) - excluded,
            # What the results list shows: kept papers (+ broader matches when few were kept).
            "shown": shown_kept + broader,
            "broader_shown": broader,
        },
        "live_counts": counts,
        "json": {
            "question_map": cqm.model_dump(mode="json"),
            "search_terms": terms.model_dump(mode="json") if terms is not None else None,
            "search_plan": plan,
            "queries": [{"source": q.get("source"), "family": q.get("query_family"),
                         "query": q.get("query_string")} for q in query_rows],
        },
    }
