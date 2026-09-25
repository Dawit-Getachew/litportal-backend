"""Question lifecycle: interpret → (revise)* → confirm | decline  (W1.4).

Human-in-the-loop guarantees implemented here:
  * no evidence search before approval — runs are created only by confirm();
  * approval is bound to one exact version (and its material hash);
  * "answer for me" additionally records a separate delegation;
  * a material edit creates a new version and invalidates earlier runs;
  * blocking conflicts must be resolved before confirmation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from clinical_engine import store
from clinical_engine.config import settings
from clinical_engine.contracts.question import ClinicalQuestionMap, material_hash
from clinical_engine.contracts.run import (
    TERMINAL_RUN_STATUSES,
    ApprovalRecord,
    DelegationRecord,
    InterpretationMeta,
    QuestionDoc,
    QuestionVersionDoc,
    TraceDoc,
)
from clinical_engine.interpretation.consistency import check
from clinical_engine.interpretation.present import (
    TYPE_LABELS,
    ChipEdit,
    EngineInputError,
    apply_edits,
    chips,
)
from clinical_engine.interpretation.service import interpret, phi_preflight
from clinical_engine.orchestrator import runner
from clinical_engine.orchestrator.tasks import DELEGATION_POLICY, POLICY_TEXT

logger = logging.getLogger("litportal.clinical_engine.questions")

APPROVAL_STATEMENT = "I confirm this interpretation of my question (version {v})."
DELEGATION_STATEMENT = (
    "I delegate evidence selection for version {v} to LitPortal's documented policy "
    f"'{DELEGATION_POLICY}': " + POLICY_TEXT
)


class EngineConflict(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class EngineNotFound(Exception):
    pass


def _phi_message() -> str:
    from services.phi_guard import USER_PHI_MESSAGE

    return USER_PHI_MESSAGE


async def _trace(question_id: str, event: str, actor: str, **details: Any) -> None:
    await store.add_trace(TraceDoc(question_id=question_id, event=event, actor=actor, details=details).model_dump())


def _meta_details(meta: InterpretationMeta) -> dict[str, Any]:
    return {"method": meta.method, "model": meta.model, "fallback_reason": meta.fallback_reason,
            "tokens_in": meta.tokens_in, "tokens_out": meta.tokens_out, "latency_ms": meta.latency_ms,
            "rejected_model_fields": meta.rejected_model_fields}


async def _load(user_id: str, question_id: str) -> dict:
    q = await store.get_question(question_id, user_id)
    if q is None:
        raise EngineNotFound(question_id)
    return q


# ── Views ────────────────────────────────────────────────────────────────────


def run_view(run: dict, tasks: list[dict]) -> dict[str, Any]:
    by_key = {t["key"]: t for t in tasks}
    plan = ((by_key.get("plan_search") or {}).get("output") or {}).get("search_plan")
    integrity = (by_key.get("integrity_check") or {}).get("output")
    return {
        "id": run["id"], "question_id": run["question_id"], "version": run["version"],
        "mode": run["mode"], "status": run["status"], "result_kind": run.get("result_kind"),
        "retrieval_adapter": run.get("retrieval_adapter"),
        "evidence_label": "fixture" if run.get("retrieval_adapter") == "mock" else "live_legacy_pipeline",
        "links": run.get("links") or {}, "usage": run.get("usage") or {}, "budget": run.get("budget") or {},
        "invalidated_by_version": run.get("invalidated_by_version"), "error": run.get("error"),
        "created_at": run.get("created_at"), "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "tasks": [{
            "key": t["key"], "type": t["type"], "seq": t["seq"], "status": t["status"],
            "attempts": t.get("attempts", 0), "max_attempts": t.get("max_attempts"),
            "depends_on": t.get("depends_on") or [], "tools": t.get("tools") or [],
            "error": t.get("error"), "started_at": t.get("started_at"), "finished_at": t.get("finished_at"),
        } for t in tasks],
        "search_plan": plan,
        "integrity": integrity,
    }


async def question_view(q: dict, *, include_run: bool = True) -> dict[str, Any]:
    view: dict[str, Any] = {
        "question_id": q["id"], "status": q["status"], "version": q.get("current_version") or 0,
        "created_at": q.get("created_at"), "updated_at": q.get("updated_at"),
    }
    if q["status"] == "blocked_phi":
        view.update({"user_message": _phi_message(), "reason_codes": q.get("reason_codes") or [],
                     "allowed_actions": ["new_question"]})
        return view
    versions = await store.list_versions(q["id"])
    current = next((v for v in versions if v["version"] == q.get("current_version")), None)
    if current is None:
        raise EngineNotFound(q["id"])
    cqm = ClinicalQuestionMap.model_validate(current["map"])
    blocking = [c for c in current.get("conflicts") or [] if c.get("severity") == "blocking"]
    awaiting = q["status"] in ("awaiting_confirmation", "needs_clarification")
    actions = ["revise", "decline"] if awaiting else ["revise"] if q["status"] == "confirmed" else []
    if q["status"] == "awaiting_confirmation" and not blocking:
        actions = ["confirm_review_papers", "confirm_answer_for_me", *actions]
    meta = current.get("interpretation") or {}
    view.update({
        "question_text": cqm.question_text,
        "interpretation": {
            "summary": cqm.summary,
            "question_type": cqm.question_type,
            "question_type_label": TYPE_LABELS.get(cqm.question_type, cqm.question_type),
            "framework": cqm.framework,
            "age_group": cqm.age_group(),
            "chips": [c.model_dump() for c in chips(cqm)],
            "conflicts": current.get("conflicts") or [],
            "warnings": current.get("warnings") or [],
            "ambiguities": cqm.ambiguities,
            "method": meta.get("method"),
            "model": meta.get("model"),
            "fallback_reason": meta.get("fallback_reason"),
        },
        "can_confirm": q["status"] == "awaiting_confirmation" and not blocking,
        "allowed_actions": actions,
        "approval": current.get("approval"),
        "delegation": current.get("delegation"),
        "versions": [{"version": v["version"], "change_kind": v["change_kind"], "status": v["status"],
                      "created_at": v["created_at"], "approved": bool(v.get("approval"))} for v in versions],
        "run": None,
    })
    if include_run and q.get("active_run_id"):
        run = await store.get_run(q["active_run_id"])
        if run is not None:
            view["run"] = run_view(run, await store.get_tasks(run["id"]))
    return view


# ── Lifecycle ────────────────────────────────────────────────────────────────


async def _new_version(q: dict, *, number: int, parent: Optional[int], change_kind: str,
                       cqm: ClinicalQuestionMap, meta: InterpretationMeta, user_id: str) -> dict:
    conflicts, warnings = check(cqm)
    version = QuestionVersionDoc(
        question_id=q["id"], user_id=user_id, version=number, parent_version=parent,
        change_kind=change_kind,  # type: ignore[arg-type]
        map=cqm, material_hash=material_hash(cqm), conflicts=conflicts, warnings=warnings,
        interpretation=meta,
    ).model_dump()
    await store.insert_version(version)
    await _trace(q["id"], "version_created", user_id, version=number, change_kind=change_kind,
                 question_type=cqm.question_type, conflicts=[c.code for c in conflicts],
                 **_meta_details(meta))
    return version


async def create_question(user: dict, text: str) -> dict[str, Any]:
    user_id = user["id"]
    phi = phi_preflight(text or "")
    if phi.blocked:
        q = QuestionDoc(user_id=user_id, status="blocked_phi").model_dump()
        q["reason_codes"] = phi.reason_codes
        await store.insert_question(q)
        await _trace(q["id"], "phi_blocked", user_id, reason_codes=phi.reason_codes)
        return await question_view(q)

    allow_model = (await store.count_interpretation(user_id)) <= settings.MAX_INTERPRETATIONS_PER_HOUR
    result = await interpret(text, allow_model=allow_model)
    q = QuestionDoc(user_id=user_id, status="awaiting_confirmation").model_dump()
    await store.insert_question(q)
    await _trace(q["id"], "question_created", user_id)
    version = await _new_version(q, number=1, parent=None, change_kind="initial", cqm=result.map,
                                 meta=result.meta, user_id=user_id)
    status = "needs_clarification" if any(c.severity == "blocking" for c in result.conflicts) else "awaiting_confirmation"
    await store.update_question(q["id"], {"status": status, "current_version": 1, "current_version_id": version["id"]})
    return await question_view(await _load(user_id, q["id"]))


async def get_question(user: dict, question_id: str) -> dict[str, Any]:
    return await question_view(await _load(user["id"], question_id))


async def list_questions(user: dict, limit: int = 20) -> list[dict[str, Any]]:
    out = []
    for q in await store.list_questions(user["id"], limit=min(max(limit, 1), 50)):
        item = {"question_id": q["id"], "status": q["status"], "version": q.get("current_version"),
                "updated_at": q.get("updated_at"), "summary": None}
        if q.get("current_version_id"):
            v = await store.get_version_by_id(q["current_version_id"])
            if v:
                item["summary"] = (v.get("map") or {}).get("summary")
        out.append(item)
    return out


async def revise_question(user: dict, question_id: str, *, base_version: int,
                          text: Optional[str] = None, edits: Optional[list[ChipEdit]] = None) -> dict[str, Any]:
    user_id = user["id"]
    q = await _load(user_id, question_id)
    if q["status"] in ("declined", "blocked_phi"):
        raise EngineConflict("question_closed", "This question was closed — start a new one.")
    if base_version != q.get("current_version"):
        raise EngineConflict("stale_version", "The interpretation changed — reload and try again.")
    if (q.get("current_version") or 0) >= settings.MAX_VERSIONS_PER_QUESTION:
        raise EngineConflict("version_limit", "Too many revisions — start a new question.")
    if bool(text) == bool(edits):
        raise EngineInputError("revise_needs_text_or_edits", "Send either new question text or chip edits.")
    current = await store.get_version(question_id, q["current_version"])
    if current is None:
        raise EngineNotFound(question_id)

    if text:
        phi = phi_preflight(text)
        if phi.blocked:
            await _trace(question_id, "phi_blocked", user_id, reason_codes=phi.reason_codes, stage="revise_text")
            raise EngineInputError("phi_detected", _phi_message())
        allow_model = (await store.count_interpretation(user_id)) <= settings.MAX_INTERPRETATIONS_PER_HOUR
        result = await interpret(text, allow_model=allow_model)
        new_map, meta, change_kind = result.map, result.meta, "text_revision"
    else:
        new_map = apply_edits(ClinicalQuestionMap.model_validate(current["map"]), edits or [])
        edited_values = " ".join(e.value for e in new_map.elements if e.source == "user_edited")
        if edited_values:
            phi = phi_preflight(edited_values)
            if phi.blocked:
                await _trace(question_id, "phi_blocked", user_id, reason_codes=phi.reason_codes, stage="chip_edit")
                raise EngineInputError("phi_detected", _phi_message())
        meta, change_kind = InterpretationMeta(method="user_edit"), "chip_edit"

    if material_hash(new_map) == current["material_hash"] and new_map.question_text == current["map"]["question_text"]:
        view = await question_view(q)
        view["unchanged"] = True
        return view

    number = q["current_version"] + 1
    version = await _new_version(q, number=number, parent=q["current_version"], change_kind=change_kind,
                                 cqm=new_map, meta=meta, user_id=user_id)
    ok = await store.update_question(
        question_id,
        {"status": "needs_clarification" if any(c["severity"] == "blocking" for c in version["conflicts"])
         else "awaiting_confirmation",
         "current_version": number, "current_version_id": version["id"],
         "confirmed_version": None, "active_run_id": None},
        expect={"current_version": number - 1},
    )
    if not ok:  # a concurrent revision won; ours is a dangling version
        await store.supersede_versions(question_id, keep_version=(await _load(user_id, question_id))["current_version"])
        raise EngineConflict("stale_version", "The interpretation changed — reload and try again.")
    await store.supersede_versions(question_id, keep_version=number)
    invalidated = await runner.supersede_question_runs(question_id, number, user_id)
    if invalidated:
        await _trace(question_id, "prior_results_invalidated", user_id, version=number, runs=invalidated)
    return await question_view(await _load(user_id, question_id))


async def confirm_question(user: dict, question_id: str, *, version: int, mode: str) -> dict[str, Any]:
    user_id = user["id"]
    if mode not in ("review_papers", "answer_for_me"):
        raise EngineInputError("invalid_mode", "Choose 'review_papers' or 'answer_for_me'.")
    q = await _load(user_id, question_id)
    current = await store.get_version(question_id, version)
    if current is None or version != q.get("current_version"):
        raise EngineConflict("stale_version", "The interpretation changed — review the latest version.")
    approval = current.get("approval")
    if approval:  # idempotent re-confirm (double click / retry)
        if approval.get("mode") != mode:
            raise EngineConflict("already_confirmed", "This version was already confirmed with another option.")
        return await question_view(q)
    if q["status"] != "awaiting_confirmation":
        code = "blocking_conflicts" if q["status"] == "needs_clarification" else "not_confirmable"
        raise EngineConflict(code, "Resolve the highlighted problems before confirming." if code == "blocking_conflicts"
                             else "This question can no longer be confirmed.")
    if any(c.get("severity") == "blocking" for c in current.get("conflicts") or []):
        raise EngineConflict("blocking_conflicts", "Resolve the highlighted problems before confirming.")

    approval_rec = ApprovalRecord(user_id=user_id, version=version, material_hash=current["material_hash"],
                                  mode=mode, statement=APPROVAL_STATEMENT.format(v=version))  # type: ignore[arg-type]
    delegation_rec = (DelegationRecord(user_id=user_id, version=version, policy=DELEGATION_POLICY,
                                       statement=DELEGATION_STATEMENT.format(v=version, n=settings.DELEGATED_MAX_ITEMS))
                      if mode == "answer_for_me" else None)
    if not await store.record_approval(current["id"], approval_rec.model_dump(),
                                       delegation_rec.model_dump() if delegation_rec else None):
        return await question_view(await _load(user_id, question_id))
    await _trace(question_id, "approval_recorded", user_id, version=version, mode=mode,
                 material_hash=current["material_hash"], approval_id=approval_rec.id)
    if delegation_rec is not None:
        await _trace(question_id, "delegation_recorded", user_id, version=version, policy=DELEGATION_POLICY,
                     delegation_id=delegation_rec.id)

    version_doc = await store.get_version_by_id(current["id"])
    run = await runner.create_run(version_doc, mode=mode, user_id=user_id)
    await store.update_question(question_id, {"status": "confirmed", "confirmed_version": version,
                                              "active_run_id": run["id"]})
    runner.spawn(run["id"])
    return await question_view(await _load(user_id, question_id))


DECLINE_REASONS = ("wrong_interpretation", "new_question", "other")


async def decline_question(user: dict, question_id: str, reason: str = "new_question") -> dict[str, Any]:
    user_id = user["id"]
    if reason not in DECLINE_REASONS:
        raise EngineInputError("invalid_reason", f"Reason must be one of {', '.join(DECLINE_REASONS)}.")
    q = await _load(user_id, question_id)
    if q["status"] not in ("awaiting_confirmation", "needs_clarification"):
        raise EngineConflict("not_declinable", "Only an unconfirmed interpretation can be declined.")
    await store.update_question(question_id, {"status": "declined", "decline_reason": reason})
    await _trace(question_id, "question_declined", user_id, reason=reason, version=q.get("current_version"))
    return await question_view(await _load(user_id, question_id))


# ── Runs ─────────────────────────────────────────────────────────────────────


async def _load_run(user_id: str, run_id: str) -> dict:
    run = await store.get_run(run_id, user_id)
    if run is None:
        raise EngineNotFound(run_id)
    return run


async def get_run(user: dict, run_id: str) -> dict[str, Any]:
    run = await _load_run(user["id"], run_id)
    return run_view(run, await store.get_tasks(run_id))


async def control_run(user: dict, run_id: str, action: str) -> dict[str, Any]:
    run = await _load_run(user["id"], run_id)
    if action == "pause":
        ok = await runner.pause(run, user["id"])
    elif action == "resume":
        ok = await runner.resume(run, user["id"])
    elif action == "cancel":
        ok = await runner.cancel(run, user["id"])
    else:
        raise EngineInputError("invalid_action", "Unknown run action.")
    if not ok:
        raise EngineConflict(f"cannot_{action}", f"This run cannot be {action}d in its current state "
                                                 f"({run['status']}).")
    return await get_run(user, run_id)


async def run_trace(user: dict, run_id: str) -> dict[str, Any]:
    await _load_run(user["id"], run_id)
    return {"run_id": run_id, "events": await store.traces(run_id=run_id)}


async def question_trace(user: dict, question_id: str) -> dict[str, Any]:
    await _load(user["id"], question_id)
    return {"question_id": question_id, "events": await store.traces(question_id=question_id)}


async def get_evidence_set(user: dict, set_id: str) -> dict[str, Any]:
    doc = await store.get_evidence_set(set_id, user["id"])
    if doc is None:
        raise EngineNotFound(set_id)
    return doc


def is_terminal(status: str) -> bool:
    return status in TERMINAL_RUN_STATUSES
