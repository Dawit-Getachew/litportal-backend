"""Week-2 retrieval tasks (retrieval adapter "engine").

  retrieve  → query portfolio (W2.1) on PubMed + Europe PMC, per-source budgets,
              ledger, partial-search state; the bounded alternative only if the
              main search is sparse (W2.2)
  union     → identity-safe union (W2.2)
  admit     → admission states with reasons (W2.4)
  lineage   → guideline lineage, review cutoffs, currency; the bounded update
              search for newer evidence re-enters union + admission (W2.3)
  project   → the evidence list, written into the session the results page opens
  access    → verified full-text routes, one gateway call per paper (W2.5)

Each handler is restart-safe: the session shell is created once (checkpoint),
retrieval restarts from a clean ledger, later steps rewrite what they own.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from clinical_engine import store
from clinical_engine.config import settings
from clinical_engine.orchestrator.tasks import FatalTaskError, TaskContext, approved_concept_plan
from clinical_engine.retrieval import access, admission, lineage, portfolio, press, projection, union
from clinical_engine.retrieval.run import run_portfolio
from services import engine_concept_plan as ecp


def _plan(ctx: TaskContext) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = approved_concept_plan(ctx.version, ctx.cqm)
    plan = ecp.validate_plan(raw) if raw else None
    if not plan:
        raise FatalTaskError("no_approved_search_terms",
                             "Engine retrieval needs the search terms approved in step 2.")
    return raw, plan


async def _progress(ctx: TaskContext, **fields: Any) -> None:
    await store.update_run(ctx.run["id"], {f"progress.{k}": v for k, v in fields.items()})


async def _model_call(ctx: TaskContext, fn: Any, *args: Any) -> Any:
    return await ctx.tools.call("model.json", fn, *args)


async def engine_retrieve(ctx: TaskContext) -> dict[str, Any]:
    raw, plan = _plan(ctx)
    run_id = ctx.run["id"]
    sid = ctx.checkpoint.get("search_session_id")
    if not sid:
        sid = await projection.create_session_shell(ctx.run, ctx.version, raw)
        await ctx.save_checkpoint({"search_session_id": sid})
    # Link at once: the results page opens and shows live progress.
    await store.update_run(run_id, {"links.search_session_id": sid})
    await _progress(ctx, stage="searching", sources={}, partial=False)
    await store.clear_retrieval(run_id)

    pf = portfolio.build(plan, ctx.cqm.question_text)

    async def progress(source: str, state: dict[str, Any]) -> None:
        await _progress(ctx, **{f"sources.{source}": state})

    states = await run_portfolio(run_id, [s for s in pf["specs"] if s.condition is None],
                                 progress=progress, call_tool=ctx.tools.call)
    queries = await store.queries_for_run(run_id)
    main_hits = [int(q.get("count") or 0) for q in queries if q["family"] == "main" and q.get("status") in ("ok", "zero")]
    sparse = bool(main_hits) and sum(main_hits) < settings.RETRIEVAL_SPARSE_THRESHOLD
    alternative = [s for s in pf["specs"] if s.condition == "if_sparse"]
    alt_ran = False
    if sparse and alternative:
        alt_states = await run_portfolio(run_id, alternative, progress=progress, call_tool=ctx.tools.call)
        alt_ran = True
        for src, st in alt_states.items():
            base = states.get(src) or {}
            states[src] = {**base, "records_read": base.get("records_read", 0) + st.get("records_read", 0),
                           "status": base.get("status") if st.get("status") == "complete" else st.get("status")}
        queries = await store.queries_for_run(run_id)
    pm = next((q for q in queries if q["family"] == "main" and q["source"] == "pubmed"), None)
    report = press.check(plan, pf, pubmed_main=({k: pm.get(k) for k in ("count", "translation", "warnings", "errors")}
                                               if pm else None))
    sources = {src: {k: st.get(k) for k in ("status", "records_read", "queries_total", "queries_done", "failed",
                                            "timed_out", "errors")} for src, st in states.items()}
    partial = any(st.get("status") != "complete" for st in states.values())
    await _progress(ctx, partial=partial, stage="merging")
    return {
        "search_session_id": sid, "sources": sources, "partial": partial,
        "portfolio": [s.public() for s in pf["specs"]], "skipped": pf["skipped"],
        "alternative": {**(pf["alternative"] or {}), "available": bool(alternative), "sparse": sparse, "ran": alt_ran},
        "press": report, "links": {"search_session_id": sid},
    }


async def engine_union(ctx: TaskContext) -> dict[str, Any]:
    summary = await union.build(ctx.run["id"], origin="initial")
    await _progress(ctx, stage="judging", counts={"records": summary["records"],
                                                  "publications": summary["publications"],
                                                  "held_back": summary["quarantined_records"]})
    return {"union": summary}


async def engine_admit(ctx: TaskContext) -> dict[str, Any]:
    _, plan = _plan(ctx)
    pubs = [p for p in await store.publications_for_run(ctx.run["id"]) if p.get("origin") == "initial"]
    usage = await admission.admit(pubs, ctx.cqm, plan, call=lambda fn, *a: _model_call(ctx, fn, *a),
                                  on_tokens=ctx.tools.consume_tokens)
    await store.bulk_update_publications([(p["id"], {"admission": p["admission"]}) for p in pubs])
    await _progress(ctx, stage="guidelines")
    return {"admission": {k: v for k, v in usage.items() if k != "context"},
            "states": dict(Counter(p["admission"]["state"] for p in pubs))}


def _identifier_index(pubs: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    idx: dict[tuple[str, str], str] = {}
    for p in pubs:
        a = p.get("article") or {}
        for kind in ("pmid", "doi_normalized", "pmcid"):
            if a.get(kind):
                idx[(kind, str(a[kind]).lower())] = p["id"]
    return idx


async def engine_lineage(ctx: TaskContext) -> dict[str, Any]:
    _, plan = _plan(ctx)
    run_id = ctx.run["id"]
    await store.clear_retrieval(run_id, origin="update")
    pubs = await store.publications_for_run(run_id)
    lin = lineage.build(pubs)
    anchor = lineage.update_anchor(pubs)
    update: dict[str, Any] = {"ran": False}
    if anchor:
        specs = portfolio.update_specs(plan, anchor["date"], anchor_label=(anchor.get("title") or "the newest review")[:80])
        states = await run_portfolio(run_id, specs, call_tool=ctx.tools.call)
        known = {rid for p in pubs for rid in p.get("record_ids") or []}
        new_cands = [c for c in await store.candidates_for_run(run_id) if c["record_id"] not in known]
        # A record the update search found again (same identifiers) is tagged on
        # the existing publication, never added twice.
        idx = _identifier_index(pubs)
        refound: dict[str, list[dict[str, Any]]] = {}
        fresh = []
        for c in new_cands:
            a = c.get("article") or {}
            hit_pub = next((idx[(k, str(a[k]).lower())] for k in ("pmid", "doi_normalized", "pmcid")
                            if a.get(k) and (k, str(a[k]).lower()) in idx), None)
            if hit_pub:
                refound.setdefault(hit_pub, []).extend({**h, "source": c["source"]} for h in c.get("hits") or [])
            else:
                fresh.append(c)
        by_id = {p["id"]: p for p in pubs}
        await store.bulk_update_publications([
            (pid, {"hits": by_id[pid]["hits"] + hits,
                   "families": sorted(set(by_id[pid]["families"]) | {h["family"] for h in hits}),
                   "lanes": sorted(set(by_id[pid]["lanes"]) | {h["lane"] for h in hits})})
            for pid, hits in refound.items()])
        docs, summary = union.publication_docs(run_id, fresh, origin="update")
        await store.replace_publications(run_id, docs, origin="update")
        usage = await admission.admit(docs, ctx.cqm, plan, call=lambda fn, *a: _model_call(ctx, fn, *a),
                                      on_tokens=ctx.tools.consume_tokens)
        await store.bulk_update_publications([(p["id"], {"admission": p["admission"]}) for p in docs])
        update = {"ran": True, "anchor": anchor, "sources": {s: st.get("status") for s, st in states.items()},
                  "new_publications": len(docs), "refound": len(refound),
                  "admitted": sum(1 for d in docs if d["admission"]["state"] in admission.QUALIFYING),
                  "tokens": usage.get("tokens")}
        pubs = await store.publications_for_run(run_id)
        lin = lineage.build(pubs)
    newer: list[tuple[str, dict]] = []
    if anchor:
        for p in pubs:
            adm = p.get("admission") or {}
            if adm.get("state") not in admission.QUALIFYING + ("supportive",):
                continue
            if adm.get("role") in ("guideline", "systematic_review", "meta_analysis"):
                continue
            added = ((p.get("pubmed") or {}).get("entrez_date") or {}).get("raw") or \
                (p.get("epmc") or {}).get("first_index_date") or ""
            if added and added[:10] > anchor["date"]:
                newer.append((p["id"], {"newer_evidence": {"added": added[:10], "cutoff": anchor["date"],
                                                           "anchor_title": anchor.get("title"),
                                                           "anchor_pmid": anchor.get("pmid")}}))
    await store.bulk_update_publications([(p["id"], {"lineage": p["lineage"]}) for p in pubs if p.get("lineage")]
                                         + newer)
    await _progress(ctx, stage="listing")
    return {"lineage": lin, "anchor": anchor, "update": update, "newer_evidence": len(newer)}


async def engine_project(ctx: TaskContext) -> dict[str, Any]:
    sid = ctx.outputs["retrieve"]["search_session_id"]
    pubs = await store.publications_for_run(ctx.run["id"])
    sections = admission.assign_sections(pubs)
    result = await projection.project(sid, pubs, sections)
    await projection.set_session_status(sid, "awaiting_selection")
    counts = {k: len(v) for k, v in sections.items()}
    await _progress(ctx, stage="access", sections=counts)
    return {"sections": counts, "projection": result, "links": {"search_session_id": sid}}


async def engine_access(ctx: TaskContext) -> dict[str, Any]:
    pubs = await store.publications_for_run(ctx.run["id"])
    targets = [p for p in pubs if p.get("section") in ("main", "other")
               or ((p.get("lineage") or {}).get("lead") and p.get("section") == "more")]

    async def persist(pub: dict[str, Any], result: dict[str, Any]) -> None:
        await store.update_publication(pub["id"], {"access": result})

    result = await access.resolve_many(targets, on_result=persist,
                                       call=lambda fn, *a, **kw: ctx.tools.call("access.resolve", fn, *a, **kw))
    await _progress(ctx, stage="done")
    return {"access": result}


async def engine_delegated_selection(ctx: TaskContext, delegation: dict[str, Any]) -> dict[str, Any]:
    """"Answer for me" on engine results: the main list in order, each record's AI
    use checked (fail-closed); records AI may not use are listed, not dropped."""
    from clinical_engine.contracts.publication import AuthorizedEvidenceSet, EvidenceSetItem
    from clinical_engine.orchestrator.tasks import DELEGATION_POLICY, _manifest_hash

    pubs = sorted((p for p in await store.publications_for_run(ctx.run["id"]) if p.get("section") == "main"),
                  key=lambda p: p.get("position") or 10**6)
    items: list[EvidenceSetItem] = []
    ineligible: list[EvidenceSetItem] = []
    for p in pubs:
        if len(items) >= settings.DELEGATED_MAX_ITEMS:
            break
        a = p.get("article") or {}
        perm = ((p.get("access") or {}).get("permitted")) or access._permission(p, None, full_text=False)  # noqa: SLF001
        acquired = (p.get("access") or {}).get("acquired") or {}
        item = EvidenceSetItem(
            publication_ref=p.get("legacy_row_id") or p["id"], article_id=p.get("legacy_article_id"),
            pmid=a.get("pmid"), doi=a.get("doi"), title=a.get("title"), selected_by="policy",
            selection_reason=f"{DELEGATION_POLICY}: engine list position {p.get('position')} "
                             f"({(p.get('admission') or {}).get('state')})",
            rank=p.get("position"),
            input_scope=("full_text" if perm.get("text_scope") == "full_text" and acquired.get("status") == "full_text"
                         else "abstract" if a.get("abstract") else "metadata_only"),
            ai_processing_allowed=perm.get("text_scope") != "none", permission_basis=str(perm.get("basis")),
        )
        (items if item.ai_processing_allowed else ineligible).append(item)
    evidence_set = AuthorizedEvidenceSet(
        user_id=ctx.run["user_id"], question_id=ctx.run["question_id"],
        question_version_id=ctx.run["question_version_id"], run_id=ctx.run["id"],
        selection_mode="delegated", delegation_id=delegation["id"], policy=DELEGATION_POLICY,
        items=items, ineligible=ineligible,
        omitted_notice=(f"{len(ineligible)} selected record(s) cannot be used by AI under their source policy "
                        "and are listed separately.") if ineligible else None,
    )
    evidence_set.manifest_hash = _manifest_hash(evidence_set)
    await store.insert_evidence_set(evidence_set.model_dump())
    await ctx.save_checkpoint({"evidence_set_id": evidence_set.id})
    return {"evidence_set_id": evidence_set.id, "selected": len(items), "ineligible": len(ineligible),
            "policy": DELEGATION_POLICY, "links": {"evidence_set_id": evidence_set.id}}


HANDLERS = {
    "engine_retrieve": engine_retrieve,
    "engine_union": engine_union,
    "engine_admit": engine_admit,
    "engine_lineage": engine_lineage,
    "engine_project": engine_project,
    "engine_access": engine_access,
}
