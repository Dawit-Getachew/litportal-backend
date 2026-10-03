"""Task handlers. Each is idempotent with respect to its checkpoint: a task
resumed after a crash continues from what it last recorded instead of
repeating writes (e.g. starting a second search session)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from clinical_engine import store
from clinical_engine.config import settings
from clinical_engine.contracts.publication import AuthorizedEvidenceSet, EvidenceSetItem
from clinical_engine.contracts.question import ClinicalQuestionMap
from clinical_engine.identity.resolver import resolve
from clinical_engine.orchestrator.plan import legacy_search_inputs, search_plan
from clinical_engine.orchestrator.tools import ToolGateway


class TaskYield(Exception):
    """Cooperative wait: re-run this task after *delay_s* (no attempt consumed)."""

    def __init__(self, delay_s: float, checkpoint: dict[str, Any]) -> None:
        super().__init__(f"yield {delay_s}s")
        self.delay_s = delay_s
        self.checkpoint = checkpoint


class RetryableTaskError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FatalTaskError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.message = message


@dataclass
class TaskContext:
    run: dict[str, Any]
    task: dict[str, Any]
    version: dict[str, Any]
    tools: ToolGateway
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)  # outputs of finished tasks by key

    @property
    def cqm(self) -> ClinicalQuestionMap:
        return ClinicalQuestionMap.model_validate(self.version["map"])

    @property
    def checkpoint(self) -> dict[str, Any]:
        return dict(self.task.get("checkpoint") or {})

    async def save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.task["checkpoint"] = checkpoint
        await store.save_checkpoint(self.task["id"], checkpoint)


Handler = Callable[[TaskContext], Awaitable[dict[str, Any]]]


# ── plan_search ─────────────────────────────────────────────────────────────


def approved_concept_plan(version: dict[str, Any], cqm: ClinicalQuestionMap) -> Optional[dict[str, Any]]:
    """The search plan from the terms the clinician approved in step 2 — only
    when the approval is bound to those exact terms."""
    from clinical_engine.search_terms.contracts import SearchTerms, terms_hash
    from clinical_engine.search_terms.plan import to_concept_plan

    approval = version.get("approval") or {}
    raw = version.get("search_terms")
    if not raw or approval.get("search_terms_version") is None:
        return None
    terms = SearchTerms.model_validate(raw)
    if terms.version != approval["search_terms_version"] or terms_hash(terms) != approval.get("search_terms_hash"):
        raise FatalTaskError("search_terms_mismatch", "The approved search terms were changed after approval.")
    return to_concept_plan(terms, cqm)


async def plan_search(ctx: TaskContext) -> dict[str, Any]:
    concept_plan = approved_concept_plan(ctx.version, ctx.cqm)
    return {"search_plan": search_plan(ctx.cqm, ctx.version["version"], concept_plan=concept_plan)}


# ── retrieve ────────────────────────────────────────────────────────────────


async def _start_legacy_session(user_id: str, inputs: dict[str, str],
                                concept_plan: Optional[dict[str, Any]] = None) -> str:
    from models.sessions import SearchSessionCreate
    from routers.search_sessions import start_session

    resp = await start_session(
        SearchSessionCreate(question=inputs["question"], population_focus=inputs["population_focus"]),
        user_id, extras={"engine_concept_plan": concept_plan} if concept_plan else None,
    )
    if hasattr(resp, "status_code"):  # JSONResponse → the legacy PHI guard blocked it
        raise FatalTaskError("phi_blocked", "The search text was blocked by the PHI guard.")
    session_id = resp.get("id") if isinstance(resp, dict) else getattr(resp, "id", None)
    if not session_id:
        raise RetryableTaskError("legacy_session_not_created")
    return str(session_id)


async def _find_adoptable_session(user_id: str, question: str, creating_at: Optional[datetime]) -> Optional[str]:
    """Crash between 'session created' and 'checkpoint saved': adopt the session
    this task created instead of starting a duplicate search. Only a session
    created right after this task recorded its intent, and not owned by another
    engine run, qualifies."""
    from datetime import timedelta

    from core.db import Collections, get_db

    if creating_at is None:
        return None
    flt: dict[str, Any] = {
        "user_id": user_id, "original_question": question, "deleted_at": None,
        "created_at": {"$gte": creating_at - timedelta(seconds=1), "$lte": creating_at + timedelta(seconds=120)},
    }
    candidates = [d["id"] async for d in get_db()[Collections.SEARCH_SESSIONS].find(
        flt, {"_id": 0, "id": 1}).sort("created_at", 1).limit(20)]
    owned = await store.sessions_owned_by_runs(candidates)
    for sid in candidates:
        if sid not in owned:
            return sid
    return None


def fixture_candidates(cqm: ClinicalQuestionMap) -> list[dict[str, Any]]:
    """Labelled fixture candidates (NOT evidence): exercise identity checks in
    demos/tests — includes a grafted-identifier conflict and two reports of one trial."""
    topic = (cqm.active("intervention") or cqm.active("exposure") or cqm.active("index_test")
             or cqm.active("condition"))
    t = topic[0].value if topic else "the question topic"
    c = (cqm.active("condition") or [None])[0]
    cond = c.value if c else "the condition"
    return [
        {"record_id": "fx-1", "source": "fixture", "pmid": "90000001", "doi": "10.9999/fixture.1",
         "nct": "NCT09000001", "title": f"[FIXTURE] {t} in {cond}: a randomized controlled trial",
         "year": 2018, "publication_types": ["Randomized Controlled Trial"]},
        {"record_id": "fx-2", "source": "fixture", "pmid": "90000002", "doi": "10.9999/fixture.2",
         "nct": "NCT09000001", "title": f"[FIXTURE] {t} in {cond}: prespecified subgroup analysis by severity",
         "year": 2020, "publication_types": ["Randomized Controlled Trial"]},
        {"record_id": "fx-3", "source": "fixture", "pmid": "90000003", "doi": "10.9999/fixture.3",
         "title": f"[FIXTURE] {t} for {cond}: systematic review and meta-analysis", "year": 2023,
         "publication_types": ["Systematic Review", "Meta-Analysis"]},
        {"record_id": "fx-4", "source": "fixture", "pmid": "90000003", "doi": "10.9999/fixture.4",
         "title": "[FIXTURE] An unrelated paper that was given a borrowed PMID", "year": 2011,
         "publication_types": ["Journal Article"]},
        {"record_id": "fx-5", "source": "fixture", "pmid": "90000005", "doi": "10.9999/fixture.5",
         "title": f"[FIXTURE] Guideline recommendations on {cond}", "year": 2021,
         "publication_types": ["Practice Guideline"]},
    ]


async def _mock_search(cqm: ClinicalQuestionMap) -> list[dict[str, Any]]:
    return fixture_candidates(cqm)


async def retrieve(ctx: TaskContext) -> dict[str, Any]:
    adapter = ctx.run.get("retrieval_adapter", "legacy_pipeline")
    if adapter == "mock":
        candidates = await ctx.tools.call("mock.evidence_search", _mock_search, ctx.cqm)
        return {"adapter": "mock", "evidence_label": "fixture", "candidates": candidates}

    cp = ctx.checkpoint
    inputs = legacy_search_inputs(ctx.cqm, ctx.version.get("change_kind", "initial"))
    concept_plan = approved_concept_plan(ctx.version, ctx.cqm)
    session_id = cp.get("search_session_id")
    if not session_id and cp.get("phase") == "creating":
        session_id = await _find_adoptable_session(ctx.run["user_id"], inputs["question"], cp.get("creating_at"))
    if not session_id:
        await ctx.save_checkpoint({"phase": "creating", "creating_at": store.now()})
        session_id = await ctx.tools.call(
            "legacy.search_session", _start_legacy_session, ctx.run["user_id"], inputs, concept_plan,
        )
        await ctx.save_checkpoint({"phase": "created", "search_session_id": session_id})
    return {
        "adapter": "legacy_pipeline", "evidence_label": "live_legacy_pipeline",
        "search_session_id": session_id, "question_source": inputs["source"],
        "population_focus": inputs["population_focus"],
        # Which approved search terms drove the search (None = the pipeline's own reading).
        "search_terms": ({"terms_version": concept_plan["terms_version"], "terms_hash": concept_plan["terms_hash"]}
                         if concept_plan else None),
        "links": {"search_session_id": session_id},
    }


# ── await_retrieval ─────────────────────────────────────────────────────────

_LEGACY_DONE = {"awaiting_selection", "awaiting_clinical_subject", "summarizing", "complete"}
_LEGACY_FAILED = {"error", "blocked_phi"}


async def await_retrieval(ctx: TaskContext) -> dict[str, Any]:
    from core.db import Collections, get_db

    session_id = ctx.outputs["retrieve"]["search_session_id"]
    doc = await get_db()[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id}, {"_id": 0, "status": 1, "error_message": 1},
    )
    if doc is None:
        raise FatalTaskError("legacy_session_missing")
    status = doc.get("status")
    polls = int(ctx.checkpoint.get("polls", 0)) + 1
    if status in _LEGACY_FAILED:
        raise FatalTaskError("legacy_pipeline_failed", f"legacy status {status}")
    if status not in _LEGACY_DONE:
        raise TaskYield(settings.POLL_INTERVAL_S, {"polls": polls, "last_status": status})
    return {"search_session_id": session_id, "legacy_status": status, "polls": polls}


# ── integrity_check ─────────────────────────────────────────────────────────


async def _legacy_candidates(session_id: str, limit: int = 300) -> tuple[list[dict], dict[str, dict]]:
    from core.db import Collections, get_db

    db = get_db()
    rows = [r async for r in db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": session_id}, {"_id": 0}).limit(limit)]
    ids = [r["article_id"] for r in rows if r.get("article_id")]
    articles = {a["id"]: a async for a in db[Collections.ARTICLES].find({"id": {"$in": ids}}, {"_id": 0})}
    candidates = []
    for r in rows:
        a = articles.get(r.get("article_id"), {})
        candidates.append({
            "record_id": r["id"], "source": "legacy_session", "pmid": a.get("pmid"), "doi": a.get("doi"),
            "pmcid": a.get("pmcid"), "title": a.get("title"), "journal": a.get("journal"),
            "year": a.get("publication_year"), "publication_date": a.get("publication_date"),
            "publication_types": a.get("publication_types_json") or [],
        })
    return candidates, {r["id"]: {**r, "article": articles.get(r.get("article_id"), {})} for r in rows}


async def integrity_check(ctx: TaskContext) -> dict[str, Any]:
    retrieved = ctx.outputs["retrieve"]
    if retrieved.get("adapter") == "mock":
        candidates, label = retrieved.get("candidates") or [], "fixture"
    else:
        candidates, _ = await _legacy_candidates(retrieved["search_session_id"])
        label = "live_legacy_pipeline"
    report = resolve(candidates)
    return {
        "evidence_label": label,
        "input_records": report.input_records,
        "publications": len(report.publications),
        "merged_groups": report.merged_groups,
        "quarantined_record_ids": report.quarantined_record_ids,
        "conflicts": report.conflicts[:20],
        "studies": [{"registration_ids": s.registration_ids,
                     "reports": [r.model_dump() for r in s.reports]} for s in report.studies][:20],
        "scope_note": ("Cross-record identifier checks on the retrieved set. Per-record verification "
                       "against PubMed/Europe PMC arrives with the W2.2 retrieval lanes."),
    }


# ── delegated_selection ("answer for me") ───────────────────────────────────

DELEGATION_POLICY = "delegated_top_ranked_v1"
POLICY_TEXT = (
    "Up to {n} highest-ranked included records from the approved search, excluding records with "
    "identifier conflicts; each record's AI use is checked against its source policy (fail-closed)."
)


def _manifest_hash(set_doc: AuthorizedEvidenceSet) -> str:
    raw = json.dumps({
        "version": set_doc.question_version_id, "policy": set_doc.policy,
        "items": sorted(i.publication_ref for i in set_doc.items),
        "ineligible": sorted(i.publication_ref for i in set_doc.ineligible),
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


async def delegated_selection(ctx: TaskContext) -> dict[str, Any]:
    from services.downstream_eligibility import can_use_article_for_tool

    delegation = ctx.version.get("delegation")
    if not delegation:
        raise FatalTaskError("no_delegation_recorded", "Answer-for-me requires a recorded delegation.")
    if ctx.checkpoint.get("evidence_set_id"):
        return {"evidence_set_id": ctx.checkpoint["evidence_set_id"],
                "links": {"evidence_set_id": ctx.checkpoint["evidence_set_id"]}}
    quarantined = set(ctx.outputs["integrity_check"].get("quarantined_record_ids") or [])
    limit = settings.DELEGATED_MAX_ITEMS
    items: list[EvidenceSetItem] = []
    ineligible: list[EvidenceSetItem] = []
    retrieved = ctx.outputs["retrieve"]

    if retrieved.get("adapter") == "mock":
        for rank, c in enumerate(retrieved.get("candidates") or [], start=1):
            if c["record_id"] in quarantined or len(items) >= limit:
                continue
            items.append(EvidenceSetItem(
                publication_ref=c["record_id"], pmid=c.get("pmid"), doi=c.get("doi"), title=c.get("title"),
                selected_by="policy", selection_reason=f"{DELEGATION_POLICY}: fixture rank {rank}",
                rank=rank, input_scope="metadata_only", ai_processing_allowed=False,
                permission_basis="fixture record (not evidence)",
            ))
    else:
        _, rows = await _legacy_candidates(retrieved["search_session_id"])
        eligible_rows = [
            r for r in rows.values()
            if not r.get("exclusion_reason") and (r.get("included_by_default") or r.get("user_included"))
            and r["id"] not in quarantined
        ]
        eligible_rows.sort(key=lambda r: (r.get("overall_rank") or 10**9, -(r.get("relevance_score") or 0.0)))
        for r in eligible_rows:
            if len(items) >= limit:
                break
            a = r.get("article") or {}
            # The whole record (journal, DOI, PMID) so Cochrane reviews are caught
            # whichever index they came through.
            verdict = can_use_article_for_tool({**a, "sources_present": r.get("sources_present") or []}, "summary")
            item = EvidenceSetItem(
                publication_ref=r["id"], article_id=r.get("article_id"), pmid=a.get("pmid"), doi=a.get("doi"),
                title=a.get("title"), selected_by="policy",
                selection_reason=f"{DELEGATION_POLICY}: legacy rank {r.get('overall_rank') or 'unranked'}",
                rank=r.get("overall_rank") or None,
                input_scope="abstract" if a.get("abstract") else "metadata_only",
                ai_processing_allowed=bool(verdict.get("allowed")),
                permission_basis=f"{verdict.get('source')}: {verdict.get('reason')}",
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


HANDLERS: dict[str, Handler] = {
    "plan_search": plan_search,
    "retrieve": retrieve,
    "await_retrieval": await_retrieval,
    "integrity_check": integrity_check,
    "delegated_selection": delegated_selection,
}
