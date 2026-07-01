"""Summary router — Phase 3.

Endpoints (all under /api):

  POST /search-sessions/{sid}/generate-summary
  GET  /search-sessions/{sid}/summary
  GET  /search-sessions/{sid}/summary-status
  POST /search-sessions/{sid}/regenerate-summary
  GET  /llm/summary-readiness                (clinician-facing, auth required)
  GET  /admin/llm-status                     (admin only)

Hard scope: all summarization is grounded ONLY in EvidenceSpan objects built
from `article_text_extracts` rows belonging to the SELECTED set of articles.
The summarizer LLM and the verifier LLM never see unselected/excluded
articles, institutional link-out text, or PDF bytes. PHI guard runs at
session creation; this router refuses to summarize blocked_phi sessions.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.db import Collections, get_db
from core.security import get_current_user, require_admin
from models.sessions import SummaryDoc
from services.audit import audit_service
from services.evidence_extraction import evidence_extraction
from services.jobs import job_service
from services.llm_provider import llm_provider
from services.summarization.evidence_spans import (
    EvidenceSpan,
    build_evidence_spans,
    citation_label_for,
)
from services.summarization.prompts import PROMPT_VERSION
from services.summarization.schemas import DraftSummary, VerifierResult
from services.summarization.summarizer import summarizer
from services.summarization.verifier import build_revision_hint, citation_verifier
from services.diagnostics import (
    STAGE_NAMES as DIAG_STAGE_NAMES,
    SessionDiagnostics,
    bind as diag_bind,
    reset as diag_reset,
)

logger = logging.getLogger("clinevidence.summary")

router = APIRouter(tags=["summary"])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


MAX_SELECTED_ARTICLES = _env_int("MAX_SELECTED_ARTICLES_FOR_SUMMARY", 10)
MAX_REVISIONS = _env_int("MAX_SUMMARY_REVISIONS", 1)


SAFE_FAILURE_USER_MESSAGE = (
    "We could not verify that all claims in the draft summary were supported by "
    "the selected evidence. Please adjust selected articles or try again."
)


async def _get_owned_session(session_id: str, user_id: str) -> dict:
    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "user_id": user_id, "deleted_at": None},
        {"_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Search session not found.")
    return doc


async def _selected_session_articles(session_id: str, db) -> list[dict]:
    sa_rows = await db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": session_id},
        {
            "_id": 0,
            "id": 1,
            "article_id": 1,
            "user_included": 1,
            "included_by_default": 1,
            "exclusion_reason": 1,
            "rank_explanation": 1,
        },
    ).to_list(length=500)
    return sa_rows


def _is_excluded_article_record(art: dict) -> bool:
    return bool(
        art.get("is_preprint")
        or art.get("is_retracted")
        or art.get("is_retraction_notice")
    )


async def _build_summary_inputs(
    *,
    session: dict,
    user_id: str,
    db,
    selected_article_ids_override: Optional[list[str]] = None,
) -> tuple[list[dict], dict[str, dict], dict[str, list[dict]], dict[str, Any]]:
    """Build (selected_studies_metadata, articles_by_id, extracts_by_article, packet_summary)."""
    session_id = session["id"]
    sa_rows = await _selected_session_articles(session_id, db)
    sa_by_aid = {r["article_id"]: r for r in sa_rows}

    if selected_article_ids_override is not None:
        # Update user_included flags to match the override (audit each change).
        wanted = set(selected_article_ids_override)
        now = datetime.now(timezone.utc)
        for r in sa_rows:
            desired = r["article_id"] in wanted
            if bool(r.get("user_included")) != desired and not r.get("exclusion_reason"):
                await db[Collections.SEARCH_SESSION_ARTICLES].update_one(
                    {"id": r["id"]},
                    {"$set": {"user_included": desired, "updated_at": now}},
                )
                await audit_service.log(
                    "session_article_updated",
                    user_id=user_id,
                    search_session_id=session_id,
                    payload={
                        "article_id": r["article_id"],
                        "user_included": desired,
                        "source": "generate_summary",
                    },
                )
                r["user_included"] = desired

    # Determine selected set after override. Strict `user_included` semantics:
    # `included_by_default` is the initial value, but the moment a clinician
    # toggles, `user_included` is the single source of truth so the summarizer
    # never sees an article the clinician explicitly deselected.
    #
    # Phase 6F — evidence-packet validation. Articles whose
    # ``exclusion_reason`` is a Phase 6F wrong-anchor cap code are
    # filtered here as an EXTRA defense-in-depth check. The existing
    # ``not r.get("exclusion_reason")`` clause already filters these,
    # but we make the closed-set intent explicit so the test suite and
    # any future code change can rely on this invariant. Per the
    # Phase 6F directive: wrong_primary_anchor / wrong_condition /
    # wrong_drug must NEVER reach the summarizer by default — only via
    # an explicit user override that's already encoded as
    # ``user_included=True`` AND ``exclusion_reason=None`` (which the
    # Default+Excluded self-heal in the search-sessions GET route
    # enforces).
    _WRONG_ANCHOR_EXCLUSION_REASONS: frozenset[str] = frozenset({
        "cap_reason_wrong_drug",
        "cap_reason_wrong_condition",
        "cap_reason_wrong_primary_anchor",
        "cap_reason_missing_primary_anchor",
        "cap_reason_missing_intervention_anchor",
        "cap_reason_missing_condition_anchor",
        "cap_reason_drug_safety_no_adverse_effect",
        "cap_reason_treatment_sequencing_wrong_disease",
        "cap_reason_disease_only_match",
    })
    selected_ids = [
        r["article_id"]
        for r in sa_rows
        if r.get("user_included") is True
        and not r.get("exclusion_reason")
        and (r.get("exclusion_reason") or "") not in _WRONG_ANCHOR_EXCLUSION_REASONS
    ]

    # Hydrate article records.
    articles_by_id: dict[str, dict] = {}
    if selected_ids:
        async for art in db[Collections.ARTICLES].find(
            {"id": {"$in": selected_ids}}, {"_id": 0}
        ):
            articles_by_id[art["id"]] = art

    # Defensive: drop articles excluded by record (preprint/retracted) AND
    # articles whose SOURCE policy blocks AI summary ingestion (B1/B16 — e.g.
    # Cochrane / MyHealthfinder-HHS). Ineligible articles NEVER reach the
    # summarizer; the reason is surfaced on the packet so the caller can
    # explain the exclusion. Conservative: only gate when the source is known
    # (so pipeline articles from pubmed/europepmc/etc. are never over-excluded).
    from services.downstream_eligibility import article_sources, can_use_article_for_tool
    safe_ids: list[str] = []
    ineligible_articles: list[dict] = []
    for aid in selected_ids:
        art = articles_by_id.get(aid)
        if not art:
            continue
        if _is_excluded_article_record(art):
            continue
        if article_sources(art):
            elig = can_use_article_for_tool(art, "summary")
            if not elig["allowed"]:
                ineligible_articles.append({
                    "article_id": aid,
                    "source": elig.get("source"),
                    "reason": elig.get("reason"),
                    "source_policy_reason": elig.get("source_policy_reason"),
                })
                continue
        safe_ids.append(aid)

    # Gather extracts per article (user-uploaded preferred over abstract).
    extracts_by_article: dict[str, list[dict]] = {}
    for aid in safe_ids:
        rows = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
            {
                "search_session_id": session_id,
                "article_id": aid,
                "include_in_summary": True,
                "$or": [
                    {"text_source": "user_uploaded_full_text", "user_id": user_id},
                    {"text_source": "oa_full_text"},
                    {"text_source": "abstract"},
                ],
            },
            {"_id": 0},
        ).to_list(length=200)
        if not rows:
            continue
        # Prefer user_uploaded > oa > abstract.
        has_user = any(r.get("text_source") == "user_uploaded_full_text" for r in rows)
        has_oa = any(r.get("text_source") == "oa_full_text" for r in rows)
        if has_user:
            rows = [r for r in rows if r["text_source"] == "user_uploaded_full_text"]
        elif has_oa:
            rows = [r for r in rows if r["text_source"] == "oa_full_text"]
        extracts_by_article[aid] = rows

    # Build selected_studies metadata list — used by both summarizer and
    # verifier mechanical floor.
    # Phase 6B.2 — label each study's ``directness``:
    #   - ``direct``       : no cap_reasons on the session_session_articles row
    #   - ``override``     : cap_reasons present AND user toggled user_included
    #                        (i.e., user override of a capped article).
    #   - ``indirect``     : cap_reasons present but the article passed the
    #                        relevance gate and wasn't capped to <60.
    selected_studies: list[dict] = []
    override_caveats: list[dict] = []
    direct_count = 0
    indirect_override_count = 0
    for aid in safe_ids:
        art = articles_by_id.get(aid)
        if not art:
            continue
        # Determine evidence_text_level from the chosen extract set.
        rows = extracts_by_article.get(aid) or []
        if rows:
            src = (rows[0].get("text_source") or "").lower()
            if src == "user_uploaded_full_text":
                level = "user_uploaded_full_text"
            elif src == "oa_full_text":
                level = "oa_full_text"
            else:
                level = "abstract"
        else:
            level = "abstract"
        # Phase 6B.2 — directness derivation.
        sa_row = sa_by_aid.get(aid) or {}
        cap_reasons_for_row = list(sa_row.get("cap_reasons") or [])
        included_by_default = bool(sa_row.get("included_by_default"))
        user_included = bool(sa_row.get("user_included"))
        if cap_reasons_for_row:
            if user_included and not included_by_default:
                directness = "override"
            else:
                directness = "indirect"
        else:
            directness = "direct"
        if directness == "direct":
            direct_count += 1
        else:
            indirect_override_count += 1
            override_caveats.append({
                "article_id": aid,
                "citation_label": citation_label_for(art),
                "title": art.get("title") or "",
                "directness": directness,
                "cap_reasons": cap_reasons_for_row,
                "missing_must_have": list(sa_row.get("missing_must_have") or []),
            })
        selected_studies.append({
            "article_id": aid,
            "citation_label": citation_label_for(art),
            "pmid": art.get("pmid"),
            "doi": art.get("doi"),
            "pmcid": art.get("pmcid"),
            "title": art.get("title") or "",
            "journal": art.get("journal"),
            "publication_year": art.get("publication_year"),
            "study_type": art.get("article_type_classification") or "unspecified",
            "evidence_text_level": level,
            # Phase 6B.2 — directness for the summarizer + verifier.
            "directness": directness,
            "cap_reasons": cap_reasons_for_row,
            "_is_excluded": False,
        })

    packet = await evidence_extraction.prepare_evidence_packet(
        session=session, user_id=user_id
    )
    # Phase 6B.2 — augment packet with caveat counts so the router can
    # persist ``evidence_packet_summary`` on the session document.
    packet["override_caveats"] = override_caveats
    packet["directly_relevant_count"] = direct_count
    packet["indirect_override_count"] = indirect_override_count
    # B7 — source-ineligible selections excluded from AI summary (with reason).
    packet["ineligible_articles"] = ineligible_articles
    packet["ineligible_count"] = len(ineligible_articles)
    return selected_studies, articles_by_id, extracts_by_article, packet


def _audit_safe_payload(
    *,
    summary_id: str,
    packet: dict,
    selected_count: int,
    attempts: int,
    failed_reason_code: Optional[str] = None,
) -> dict:
    """Counts and IDs only — never raw evidence text or prompts."""
    return {
        "summary_id": summary_id,
        "selected_article_count": selected_count,
        "abstract_only_count": int(packet.get("abstract_only_count") or 0),
        "oa_count": int(packet.get("oa_full_text_count") or 0),
        "user_upload_count": int(packet.get("user_uploaded_full_text_count") or 0),
        "verification_attempts": attempts,
        "failed_reason_code": failed_reason_code,
    }


# ----------------------------------------------------------------------------
# Readiness endpoints (Task 1)
# ----------------------------------------------------------------------------


@router.get("/llm/summary-readiness")
async def summary_readiness(current_user: dict = Depends(get_current_user)):
    status = llm_provider.status()
    if not status["configured"]:
        return {
            "summarization_enabled": False,
            "reason": "llm_not_configured",
            "message": "Configure an LLM provider to enable evidence summarization.",
            "provider": status["provider"],
        }
    return {
        "summarization_enabled": True,
        "reason": "ready",
        "message": "Summarization is configured. Evidence is grounded only in selected articles and verified before display.",
        "provider": status["provider"],
        "max_selected_articles": MAX_SELECTED_ARTICLES,
    }


@router.get("/admin/llm-status", dependencies=[Depends(require_admin)])
async def admin_llm_status():
    # status() never returns key values.
    return llm_provider.status()


# ----------------------------------------------------------------------------
# Generate-summary pipeline
# ----------------------------------------------------------------------------


class GenerateSummaryBody(BaseModel):
    selected_article_ids: Optional[list[str]] = None
    include_abstract_only: bool = True
    acknowledge_abstract_only_limitations: bool = False


async def _set_session_status(session_id: str, status: str, *, db) -> None:
    await db[Collections.SEARCH_SESSIONS].update_one(
        {"id": session_id},
        {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}},
    )


async def _persist_failed_summary(
    *,
    db,
    session: dict,
    selected_studies: list[dict],
    packet: dict,
    attempts: int,
    reason: str,
    notes: str,
) -> dict:
    doc = SummaryDoc(
        search_session_id=session["id"],
        selected_article_ids_json=[s["article_id"] for s in selected_studies],
        answer_json={},
        verification_status="failed",
        verifier_notes=notes[:2000],
        evidence_packet_summary_json=packet,
        claim_map_json=[],
        model_provider=llm_provider.provider,
        summary_model=llm_provider.summary_model,
        verifier_model=llm_provider.verifier_model,
        prompt_version=PROMPT_VERSION,
        verification_attempts=attempts,
        failed_reason_code=reason,  # type: ignore[arg-type]
        displayed_to_user=False,
        expires_at=session.get("expires_at"),
    )
    out = doc.model_dump()
    out["created_at"] = out["created_at"].astimezone(timezone.utc)
    out["updated_at"] = out["updated_at"].astimezone(timezone.utc)
    if out.get("expires_at"):
        out["expires_at"] = out["expires_at"].astimezone(timezone.utc)
    await db[Collections.SUMMARIES].insert_one(out)
    return out


async def _persist_passed_summary(
    *,
    db,
    session: dict,
    selected_studies: list[dict],
    packet: dict,
    attempts: int,
    draft: DraftSummary,
    verifier_notes: str,
) -> dict:
    answer = draft.model_dump()
    doc = SummaryDoc(
        search_session_id=session["id"],
        selected_article_ids_json=[s["article_id"] for s in selected_studies],
        answer_json=answer,
        verification_status="passed",
        verifier_notes=verifier_notes[:2000],
        evidence_packet_summary_json=packet,
        claim_map_json=answer.get("claim_map") or [],
        model_provider=llm_provider.provider,
        summary_model=llm_provider.summary_model,
        verifier_model=llm_provider.verifier_model,
        prompt_version=PROMPT_VERSION,
        verification_attempts=attempts,
        failed_reason_code=None,
        displayed_to_user=True,
        expires_at=session.get("expires_at"),
    )
    out = doc.model_dump()
    out["created_at"] = out["created_at"].astimezone(timezone.utc)
    out["updated_at"] = out["updated_at"].astimezone(timezone.utc)
    if out.get("expires_at"):
        out["expires_at"] = out["expires_at"].astimezone(timezone.utc)
    await db[Collections.SUMMARIES].insert_one(out)
    return out


async def _run_generate(session_id: str, current_user: dict, body: GenerateSummaryBody):
    db = get_db()
    session = await _get_owned_session(session_id, current_user["id"])

    if session.get("status") == "blocked_phi":
        raise HTTPException(
            status_code=400,
            detail={"detail": "Session is blocked for PHI.", "reason": "phi_blocked_session"},
        )
    if session.get("deleted_at"):
        raise HTTPException(status_code=410, detail={"detail": "Session expired.", "reason": "session_expired"})

    if not llm_provider.is_configured():
        # Persist nothing; just return 503 with a fixed safe message.
        await audit_service.log(
            "summary_generation_blocked",
            user_id=current_user["id"],
            search_session_id=session_id,
            payload={"reason": "llm_not_configured"},
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Summarization is not configured.",
                "reason": "llm_not_configured",
                "message": "Configure an LLM provider to enable evidence summarization.",
            },
        )

    # Phase 6F.2 — bind a SessionDiagnostics for the summary path,
    # seeded from previously-persisted values so the LLM-call /
    # token / latency totals are CUMULATIVE across the search pipeline
    # AND the summarizer + citation_verifier stages. Best-effort:
    # diagnostics must NEVER break the summary path. Flushed via
    # try/finally below.
    diag = SessionDiagnostics()
    persisted = dict(session.get("session_diagnostics_json") or {})
    _pc = persisted.get("llm_calls_count")
    if isinstance(_pc, int) and _pc >= 0:
        diag.llm_calls_count = _pc
    _pbs = persisted.get("llm_calls_by_stage") or {}
    if isinstance(_pbs, dict):
        for k, v in _pbs.items():
            if k in DIAG_STAGE_NAMES and isinstance(v, int):
                diag.llm_calls_by_stage[k] = int(v)
    _te = persisted.get("token_estimate") or {}
    if isinstance(_te, dict):
        diag.prompt_tokens = int(_te.get("prompt_tokens") or 0)
        diag.completion_tokens = int(_te.get("completion_tokens") or 0)
        _em = _te.get("estimate_method")
        if _em == "provider_reported":
            diag._method_provider = 1
        elif _em == "char_div_4_estimate":
            diag._method_estimate = 1
        elif _em == "mixed":
            diag._method_provider = 1
            diag._method_estimate = 1
    _lat = persisted.get("latency_by_stage_json") or {}
    if isinstance(_lat, dict):
        for k, v in _lat.items():
            if k in DIAG_STAGE_NAMES and isinstance(v, int) and v >= 0:
                diag.latency_by_stage[k] = int(v)
    diag_token = diag_bind(diag)
    try:
        with diag.stage("evidence_packet_preparation"):
            selected_studies, articles_by_id, extracts_by_article, packet = await _build_summary_inputs(
                session=session,
                user_id=current_user["id"],
                db=db,
                selected_article_ids_override=body.selected_article_ids,
            )
        return await _run_generate_inner(
            session_id=session_id,
            current_user=current_user,
            body=body,
            db=db,
            session=session,
            selected_studies=selected_studies,
            articles_by_id=articles_by_id,
            extracts_by_article=extracts_by_article,
            packet=packet,
            diag=diag,
        )
    finally:
        # Phase 6F.2 — flush diagnostics back to the session. Closed-set
        # numerics only — see services/diagnostics.py::SessionDiagnostics.
        # NEVER raise: must not mask a real error.
        try:
            payload = diag.to_dict()
            await db[Collections.SEARCH_SESSIONS].update_one(
                {"id": session_id},
                {"$set": {
                    "session_diagnostics_json.llm_calls_count":
                        payload["llm_calls_count"],
                    "session_diagnostics_json.llm_calls_by_stage":
                        payload["llm_calls_by_stage"],
                    "session_diagnostics_json.token_estimate":
                        payload["token_estimate"],
                    "session_diagnostics_json.latency_by_stage_json":
                        payload["latency_by_stage_json"],
                    "updated_at": datetime.now(timezone.utc),
                }},
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phase6f2 diagnostics flush failed (non-fatal) for session %s",
                session_id,
            )
        diag_reset(diag_token)


async def _run_generate_inner(
    *,
    session_id: str,
    current_user: dict,
    body: GenerateSummaryBody,
    db,
    session: dict,
    selected_studies: list[dict],
    articles_by_id: dict[str, dict],
    extracts_by_article: dict[str, list[dict]],
    packet: dict[str, Any],
    diag: SessionDiagnostics,
):

    if not selected_studies:
        await audit_service.log(
            "summary_generation_blocked",
            user_id=current_user["id"],
            search_session_id=session_id,
            payload={"reason": "no_selected_articles"},
        )
        raise HTTPException(status_code=400, detail={
            "detail": "Select at least one article before generating a summary.",
            "reason": "no_selected_articles",
            "packet_summary": packet,
        })

    if len(selected_studies) > MAX_SELECTED_ARTICLES:
        await audit_service.log(
            "summary_generation_blocked",
            user_id=current_user["id"],
            search_session_id=session_id,
            payload={"reason": "max_articles_exceeded", "selected": len(selected_studies)},
        )
        raise HTTPException(status_code=400, detail={
            "detail": f"Please reduce selection to {MAX_SELECTED_ARTICLES} articles or fewer.",
            "reason": "max_articles_exceeded",
            "max_selected_articles": MAX_SELECTED_ARTICLES,
        })

    abstract_only = sum(1 for s in selected_studies if s["evidence_text_level"] == "abstract")
    if abstract_only > 0 and not body.acknowledge_abstract_only_limitations:
        await audit_service.log(
            "summary_generation_blocked",
            user_id=current_user["id"],
            search_session_id=session_id,
            payload={"reason": "abstract_only_acknowledgement_required", "count": abstract_only},
        )
        raise HTTPException(status_code=400, detail={
            "detail": "abstract_only_acknowledgement_required",
            "abstract_only_count": abstract_only,
        })

    if not packet.get("ready_for_summary"):
        await audit_service.log(
            "summary_generation_blocked",
            user_id=current_user["id"],
            search_session_id=session_id,
            payload={"reason": packet.get("block_reason") or "no_usable_evidence"},
        )
        raise HTTPException(status_code=400, detail={
            "detail": "Evidence packet is not ready for summary.",
            "reason": packet.get("block_reason") or "no_usable_evidence",
            "packet_summary": packet,
        })

    # Build evidence spans (only from extracts where include_in_summary=true).
    spans: list[EvidenceSpan] = build_evidence_spans(
        session_id=session_id,
        selected_article_ids=[s["article_id"] for s in selected_studies],
        articles_by_id=articles_by_id,
        extracts_by_article=extracts_by_article,
    )
    if not spans:
        raise HTTPException(status_code=400, detail={
            "detail": "No usable evidence spans in the selected articles.",
            "reason": "no_usable_evidence",
            "packet_summary": packet,
        })

    selected_count = len(selected_studies)
    audit_base = {
        "selected_article_count": selected_count,
        "abstract_only_count": int(packet.get("abstract_only_count") or 0),
        "oa_count": int(packet.get("oa_full_text_count") or 0),
        "user_upload_count": int(packet.get("user_uploaded_full_text_count") or 0),
        "spans_count": len(spans),
    }
    await audit_service.log(
        "summary_requested",
        user_id=current_user["id"],
        search_session_id=session_id,
        payload=audit_base,
    )

    await _set_session_status(session_id, "summarizing", db=db)

    pico = session.get("pico_json") or {}
    sanitized_question = session.get("sanitized_question") or session.get("original_question") or ""

    revision_hint: Optional[str] = None
    last_verifier: Optional[VerifierResult] = None
    attempts = 0
    failed_reason: Optional[str] = None
    for attempt in range(MAX_REVISIONS + 1):  # 0..MAX_REVISIONS inclusive (revisions count)
        attempts += 1
        with diag.stage("summarizer"):
            draft, reason = await summarizer.draft(
                sanitized_question=sanitized_question,
                pico=pico,
                selected_studies=selected_studies,
                spans=spans,
                revision_hint=revision_hint,
                session_id=session_id,
                override_caveats=packet.get("override_caveats") or [],
            )
        if draft is None:
            failed_reason = reason or "other"
            break
        await audit_service.log(
            "summary_drafted",
            user_id=current_user["id"],
            search_session_id=session_id,
            payload={**audit_base, "attempt": attempts},
        )
        await audit_service.log(
            "summary_verification_started",
            user_id=current_user["id"],
            search_session_id=session_id,
            payload={"attempt": attempts},
        )
        with diag.stage("citation_verifier"):
            verifier_result = await citation_verifier.verify(
                draft=draft,
                spans=spans,
                selected_studies=selected_studies,
                session_id=session_id,
            )
        last_verifier = verifier_result
        if verifier_result.verification_status == "passed":
            stored = await _persist_passed_summary(
                db=db,
                session=session,
                selected_studies=selected_studies,
                packet=packet,
                attempts=attempts,
                draft=draft,
                verifier_notes=verifier_result.verifier_notes,
            )
            await audit_service.log(
                "summary_verified",
                user_id=current_user["id"],
                search_session_id=session_id,
                payload=_audit_safe_payload(
                    summary_id=stored["id"], packet=packet,
                    selected_count=selected_count, attempts=attempts,
                ),
            )
            await _set_session_status(session_id, "complete", db=db)
            await audit_service.log(
                "summary_displayed",
                user_id=current_user["id"],
                search_session_id=session_id,
                payload={"summary_id": stored["id"]},
            )
            return {
                "status": "passed",
                "summary_id": stored["id"],
                "summary": stored["answer_json"],
                "packet_summary": packet,
                "model_provider": stored["model_provider"],
                "summary_model": stored["summary_model"],
                "verifier_model": stored["verifier_model"],
                "prompt_version": stored["prompt_version"],
                "verification_attempts": attempts,
            }
        # Build a revision hint for the next attempt.
        revision_hint = build_revision_hint(verifier_result)
    # Fell through — either summarizer failed to produce JSON or revisions exhausted.
    reason = failed_reason or (
        "revision_limit_exceeded" if last_verifier is not None else "other"
    )
    notes = (
        last_verifier.verifier_notes if last_verifier else "Summarizer did not produce a verifiable draft."
    )
    stored = await _persist_failed_summary(
        db=db,
        session=session,
        selected_studies=selected_studies,
        packet=packet,
        attempts=attempts,
        reason=reason,
        notes=notes,
    )
    await audit_service.log(
        "summary_failed",
        user_id=current_user["id"],
        search_session_id=session_id,
        payload=_audit_safe_payload(
            summary_id=stored["id"], packet=packet, selected_count=selected_count,
            attempts=attempts, failed_reason_code=reason,
        ),
    )
    await _set_session_status(session_id, "awaiting_selection", db=db)
    return {
        "status": "failed",
        "user_message": SAFE_FAILURE_USER_MESSAGE,
        "failed_reason_code": reason,
        "packet_summary": packet,
        "verification_attempts": attempts,
    }


async def _dispatch_summary_job(
    session_id: str,
    current_user: dict,
    body: GenerateSummaryBody,
):
    """Phase 4B — create a durable job and dispatch the LLM/verifier work
    in the background. Returns ``{job_id, status, session_id}`` immediately.
    Cheap pre-checks (PHI/session-expired/LLM-not-configured) still run sync
    so the user gets a 4xx/5xx instead of a doomed queued job.
    """
    session = await _get_owned_session(session_id, current_user["id"])
    if session.get("status") == "blocked_phi":
        raise HTTPException(
            status_code=400,
            detail={"detail": "Session is blocked for PHI.", "reason": "phi_blocked_session"},
        )
    if session.get("deleted_at"):
        raise HTTPException(status_code=410, detail={"detail": "Session expired.", "reason": "session_expired"})
    if not llm_provider.is_configured():
        await audit_service.log(
            "summary_generation_blocked",
            user_id=current_user["id"],
            search_session_id=session_id,
            payload={"reason": "llm_not_configured"},
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Summarization is not configured.",
                "reason": "llm_not_configured",
                "message": "Configure an LLM provider to enable evidence summarization.",
            },
        )

    # Deduplicate: if another summary job for this user/session is queued or
    # running, return that job instead of starting a second one.
    existing = await job_service.find_running(
        user_id=current_user["id"], job_type="summary_generation",
        search_session_id=session_id,
    )
    if existing:
        return {
            "job_id": existing["id"],
            "status": existing.get("status"),
            "session_id": session_id,
            "duplicate_suppressed": True,
        }

    job = await job_service.create(
        user_id=current_user["id"],
        job_type="summary_generation",
        search_session_id=session_id,
        total_steps=4,
        expires_at=session.get("expires_at"),
    )
    await audit_service.log(
        "summary_job_queued",
        user_id=current_user["id"],
        search_session_id=session_id,
        payload={"job_id": job["id"]},
    )

    async def _runner() -> None:
        await job_service.mark_running(
            job_id=job["id"], stage="preparing_evidence_packet",
            current_step=1, percent=10,
        )
        try:
            await job_service.update_progress(
                job_id=job["id"], stage="drafting_summary",
                current_step=2, percent=40,
            )
            result = await _run_generate(session_id, current_user, body)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"detail": str(exc.detail)}
            reason = str(detail.get("reason") or "internal_error")[:64]
            msg = str(detail.get("detail") or "Summary generation failed.")[:240]
            await job_service.fail(
                job_id=job["id"], error_code=reason, error_message_safe=msg,
                result_ref={"session_id": session_id, "failed_reason_code": reason,
                             "user_message_code": reason},
            )
            return
        except Exception:  # noqa: BLE001
            logger.exception("summary job %s crashed", job["id"])
            await job_service.fail(
                job_id=job["id"], error_code="internal_error",
                error_message_safe="Summary generation failed with an internal error.",
                result_ref={"session_id": session_id,
                             "user_message_code": "internal_error"},
            )
            return

        if isinstance(result, JSONResponse):
            await job_service.fail(
                job_id=job["id"], error_code="llm_not_configured",
                error_message_safe="Summarization is not configured.",
                result_ref={"session_id": session_id,
                             "user_message_code": "llm_not_configured"},
            )
            return

        await job_service.update_progress(
            job_id=job["id"], stage="verifying_citations",
            current_step=3, percent=80,
        )

        if result.get("status") == "passed":
            await job_service.succeed(
                job_id=job["id"],
                result_ref={
                    "summary_id": result.get("summary_id"),
                    "session_id": session_id,
                    "verification_attempts": int(result.get("verification_attempts") or 0),
                    "model_provider": result.get("model_provider"),
                    "summary_model": result.get("summary_model"),
                    "verifier_model": result.get("verifier_model"),
                    "prompt_version": result.get("prompt_version"),
                },
                stage="complete", percent=100,
            )
        else:
            # Verification failed — DO NOT leak the draft. The job result only
            # carries a closed-set reason code + a safe user message code.
            await job_service.fail(
                job_id=job["id"],
                error_code="verification_failed",
                error_message_safe=SAFE_FAILURE_USER_MESSAGE,
                result_ref={
                    "session_id": session_id,
                    "failed_reason_code": result.get("failed_reason_code"),
                    "verification_attempts": int(result.get("verification_attempts") or 0),
                    "user_message_code": "verification_failed",
                },
                stage="verification_failed", percent=100,
            )

    job_service.dispatch(job_id=job["id"], runner=_runner, job_type="summary_generation")
    return {
        "job_id": job["id"],
        "status": "queued",
        "session_id": session_id,
    }


@router.post("/search-sessions/{session_id}/generate-summary")
async def generate_summary(
    session_id: str,
    body: GenerateSummaryBody = Body(default_factory=GenerateSummaryBody),
    sync: bool = False,
    current_user: dict = Depends(get_current_user),
):
    # Default behavior (Phase 4B): create a durable job + dispatch async.
    # `?sync=true` runs the legacy synchronous pipeline (used by unit tests).
    if sync:
        return await _run_generate(session_id, current_user, body)
    return await _dispatch_summary_job(session_id, current_user, body)


@router.post("/search-sessions/{session_id}/regenerate-summary")
async def regenerate_summary(
    session_id: str,
    body: GenerateSummaryBody = Body(default_factory=GenerateSummaryBody),
    sync: bool = False,
    current_user: dict = Depends(get_current_user),
):
    if sync:
        return await _run_generate(session_id, current_user, body)
    return await _dispatch_summary_job(session_id, current_user, body)


@router.get("/search-sessions/{session_id}/summary")
async def get_summary(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    await _get_owned_session(session_id, current_user["id"])
    doc = await db[Collections.SUMMARIES].find_one(
        {"search_session_id": session_id, "verification_status": "passed"},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    if not doc:
        raise HTTPException(status_code=404, detail="No verified summary for this session.")
    if not doc.get("displayed_to_user"):
        await db[Collections.SUMMARIES].update_one(
            {"id": doc["id"]},
            {"$set": {"displayed_to_user": True, "updated_at": datetime.now(timezone.utc)}},
        )
        await audit_service.log(
            "summary_displayed",
            user_id=current_user["id"],
            search_session_id=session_id,
            payload={"summary_id": doc["id"]},
        )
        doc["displayed_to_user"] = True
    return doc


@router.get("/search-sessions/{session_id}/summary-status")
async def summary_status(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    await _get_owned_session(session_id, current_user["id"])
    doc = await db[Collections.SUMMARIES].find_one(
        {"search_session_id": session_id},
        {"_id": 0, "verification_status": 1, "verification_attempts": 1,
         "failed_reason_code": 1, "id": 1, "displayed_to_user": 1, "created_at": 1},
        sort=[("created_at", -1)],
    )
    if not doc:
        return {"status": "none", "attempts": 0, "failed_reason_code": None}
    return {
        "status": doc.get("verification_status") or "none",
        "attempts": int(doc.get("verification_attempts") or 0),
        "failed_reason_code": doc.get("failed_reason_code"),
        "summary_id": doc.get("id"),
        "displayed_to_user": bool(doc.get("displayed_to_user")),
    }


# ---------------------------------------------------------------------------
# Phase 4B — failed-verification guidance.
# ---------------------------------------------------------------------------

_VERIFICATION_GUIDANCE = {
    "unsupported_claims": [
        "Try selecting fewer articles so each claim is well-supported.",
        "Prefer articles where you have full text (OA or your own upload).",
        "If only abstracts are available, expect the certainty rating to be lower.",
    ],
    "incorrect_citations": [
        "Double-check the article selection: include only articles you have read.",
        "Re-run after deselecting any items you are unsure about.",
    ],
    "fabricated_identifiers": [
        "Ensure your selected articles all have DOIs or PMIDs in their metadata.",
        "Try uploading full text for high-priority articles so identifiers are extracted.",
    ],
    "overstated_certainty": [
        "Acknowledge the abstract-only limitation, or add more rigorous studies.",
        "Confirm the question matches the design of the selected studies.",
    ],
    "revision_limit_exceeded": [
        "Reduce the number of selected articles and try again.",
        "Replace abstract-only items with full-text articles where possible.",
    ],
    "parser_validation_failed": [
        "Retry the request; transient model errors are usually one-off.",
        "If retries keep failing, reduce the number of selected articles.",
    ],
    "no_usable_evidence": [
        "Ensure at least one selected article has an abstract or full text in the session.",
        "Upload a PDF for at least one article and re-attest your rights to use it.",
    ],
    "llm_not_configured": [
        "Ask your administrator to configure the LLM provider before generating summaries.",
    ],
    "verification_failed": [
        "Reduce the number of selected articles.",
        "Replace abstract-only items with full-text articles where possible.",
        "Try again — verification is performed independently each time.",
    ],
    "timeout": [
        "Try again with fewer selected articles; the LLM took too long.",
    ],
    "internal_error": [
        "Try again in a moment. If this persists, contact your administrator.",
    ],
}


@router.get("/search-sessions/{session_id}/verification-guidance")
async def verification_guidance(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Phase 4B — return a safe guidance payload for the LAST summary attempt.

    NEVER returns raw draft text. NEVER returns verifier raw output. Always
    returns: ``{has_failure, failed_reason_code, user_message, suggestions[]}``.
    """
    db = get_db()
    await _get_owned_session(session_id, current_user["id"])
    doc = await db[Collections.SUMMARIES].find_one(
        {"search_session_id": session_id},
        {"_id": 0, "verification_status": 1, "failed_reason_code": 1,
         "verification_attempts": 1, "created_at": 1, "id": 1},
        sort=[("created_at", -1)],
    )
    if not doc or doc.get("verification_status") != "failed":
        return {
            "has_failure": False,
            "failed_reason_code": None,
            "user_message": None,
            "suggestions": [],
        }
    reason = doc.get("failed_reason_code") or "verification_failed"
    suggestions = _VERIFICATION_GUIDANCE.get(
        reason, _VERIFICATION_GUIDANCE["verification_failed"]
    )
    return {
        "has_failure": True,
        "failed_reason_code": reason,
        "verification_attempts": int(doc.get("verification_attempts") or 0),
        "user_message": SAFE_FAILURE_USER_MESSAGE,
        "suggestions": suggestions,
    }
