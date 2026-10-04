"""Review one paper on request, after the list was written (W2.4 transparency).

A paper the model never reviewed (beyond the review budget) gets a first look; an
excluded paper gets a second look that leans towards keeping it. The judgment is
made exactly as during the run (signals → model → fixed rules), marked "reviewed on
request", and the paper is placed at the end of "More" (or stays in "Excluded"), so
it can be selected, summarised or saved like any listed paper.
"""
from __future__ import annotations

from typing import Any

from clinical_engine import store
from clinical_engine.retrieval import admission, projection

REVIEWABLE = ("not_assessed", "excluded")
MAX_PER_RUN = 50


class ReviewError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code)
        self.code = code
        self.message = message


async def review_one(run: dict[str, Any], pub: dict[str, Any], session_id: str) -> dict[str, Any]:
    from clinical_engine.contracts.question import ClinicalQuestionMap
    from clinical_engine.orchestrator.tasks import approved_concept_plan
    from services import engine_concept_plan as ecp

    if pub.get("section") not in REVIEWABLE or pub.get("identity_status") == "quarantined":
        raise ReviewError("not_reviewable", "Only papers that were not reviewed or were excluded can be reviewed.")
    pubs = await store.publications_for_run(run["id"])
    if sum(1 for p in pubs if (p.get("admission") or {}).get("reviewed_on_request")) >= MAX_PER_RUN:
        raise ReviewError("review_limit", f"Up to {MAX_PER_RUN} papers can be reviewed on request per search.")
    version = await store.get_version_by_id(run["question_version_id"])
    cqm = ClinicalQuestionMap.model_validate(version["map"])
    raw = approved_concept_plan(version, cqm)
    plan = ecp.validate_plan(raw) if raw else None
    if not plan:
        raise ReviewError("no_plan", "This search has no approved search terms.")
    primary = {t for p in pubs if p.get("report_role") == "primary_result" for t in p.get("trial_ids") or []}
    sig = admission.signals(pub, plan, primary_trials=primary)
    previous = dict(pub.get("admission") or {})
    if sig.get("certain_exclusion"):
        adm = admission.finalize(sig, None, method="rules")
    else:
        gateway = admission.admission_gateway()
        if not gateway.is_configured():
            raise ReviewError("model_unavailable", "The relevance model is not configured.")
        ctx = admission.question_context(cqm, plan)
        note = admission.SECOND_LOOK if pub.get("section") == "excluded" else None
        got, res = await admission.judge_batch(gateway, ctx, [("c1", pub, sig)], note)
        if "c1" not in got:
            raise ReviewError("model_failed", "The relevance model did not answer — try again.")
        family = admission.TYPE_FAMILY.get(ctx["question_type"])
        adm = admission.finalize(sig, got["c1"], method="model_on_request", narrative_ok=family == "background",
                                 has_abstract=bool(((pub.get("article") or {}).get("abstract") or "").strip()))
        adm["tokens"] = res.tokens
    adm["signals"] = {k: v for k, v in sig.items() if k != "concepts"}
    adm["concepts"] = sig["concepts"]
    adm["reviewed_on_request"] = True
    adm["previous"] = {"state": previous.get("state"), "reason": previous.get("reason")}
    section = "excluded" if adm["state"] == "excluded" else "more"
    position = max([p.get("position") or 0 for p in pubs] + [0]) + 1
    placed = await projection.place(session_id, {**pub, "admission": adm}, section=section, position=position)
    await store.update_publication(pub["id"], {"admission": adm, **placed})
    from clinical_engine.orchestrator.runner import trace

    await trace(run, "paper_reviewed_on_request", publication_id=pub["id"], previous=previous.get("state"),
                state=adm["state"], tokens=adm.get("tokens", 0))
    return {**pub, "admission": adm, **placed}
