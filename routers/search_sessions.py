"""Search sessions router (Phase 2A) — PHI gate, pipeline kickoff, polling,
transparency, and per-article include/exclude toggle.

PHI policy: when PHIGuardService blocks a question, the search_session row is
created with original_question=null and sanitized_question=null. The
pipeline is NEVER scheduled for blocked sessions; no LLM call, no Europe PMC
call ever sees the raw text.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import settings
from core.db import Collections, get_db
from core.security import get_current_user
from models.sessions import (
    EvidenceLensUpdate,
    SearchSessionCreate,
    SearchSessionDoc,
    SearchSessionResponse,
)
from services import evidence_lens
from services.article_lane_classifier import (
    _FORBIDDEN_DIAG_SUBSTRINGS as _K3A_FORBIDDEN_DIAG_SUBSTRINGS,
)
from services.audit import audit_service
from services.full_text_access_broker import full_text_access_broker
from services.interpreted_question_display import (
    DISPLAY_SCHEMA_VERSION,
    build_interpreted_question_display,
    build_m4_interpreted_display,
)
from services.phi_guard import USER_PHI_MESSAGE, phi_guard
from services.picott_term_mapping_service import (
    build_m4_picott_term_mapping,
    build_picott_term_mapping,
    derive_display_from_mapping,
    needs_display_self_heal,
)
from services.pipeline import schedule as schedule_pipeline

logger = logging.getLogger("clinevidence.search_sessions")
router = APIRouter(prefix="/search-sessions", tags=["search-sessions"])


def _clinician_m3_2_block(persisted: Any) -> Optional[dict]:
    """Phase M3.2 code-review patch — restrict the CLINICIAN surface
    to ``{applied, shape, chip}`` only. Operator-facing internals
    (``decision``, ``kill_switch_state``, ``timeout_status``,
    ``fallback_status``, ``route_owned_by_k4a_specialty``,
    ``sidecar_doc_id``, etc.) MUST NOT be exposed via the
    clinician-facing search-session GET. They remain on the admin
    diagnostics surface for adjudication.

    Returns ``None`` for:
      * legacy sessions (no M3.2 block persisted)
      * sessions where ``applied=false`` (the chip would not render)
      * malformed inputs

    Returns the closed-set projection ``{applied, shape, chip}``
    otherwise.
    """
    if not isinstance(persisted, dict):
        return None
    if persisted.get("applied") is not True:
        return None
    chip = persisted.get("chip")
    return {
        "applied": True,
        "shape": persisted.get("shape"),
        "chip": chip if isinstance(chip, dict) else None,
    }


def _clinician_m4_active_block(persisted: Any) -> Optional[dict]:
    """Phase M4.1 — restrict the CLINICIAN surface to ``{applied, chip}``
    only. Operator-facing internals (``decision``, ``kill_switch_state``,
    ``sidecar_doc_id``) are NOT exposed via the clinician GET; they
    remain on the admin diagnostics surface.

    Returns ``None`` for legacy sessions, ``applied=false`` sessions
    (the indicator would not render), and malformed inputs.
    """
    if not isinstance(persisted, dict):
        return None
    if persisted.get("applied") is not True:
        return None
    chip = persisted.get("chip")
    return {
        "applied": True,
        "chip": chip if isinstance(chip, dict) else None,
    }


def _m4_applied_candidate(doc: dict) -> Optional[dict]:
    """Phase M4.1.6 — return the ACCEPTED + APPLIED M4 candidate dict for
    a session whose bounded-LLM active repair was promoted, else ``None``.

    ``m4_active.applied`` lives on the top-level ``session_diagnostics_json``
    (BC fallback: the ``pico_json`` copy). The validated candidate itself
    is persisted under
    ``pico_json.session_diagnostics_json.m4_full_question_interpreter.candidate``.
    Returns ``None`` for legacy / non-applied / production (hard-off)
    sessions — so the M4 display projection never runs there.
    """
    if not isinstance(doc, dict):
        return None
    sd_top = doc.get("session_diagnostics_json") or {}
    m4a = sd_top.get("m4_active")
    pj_sd = (doc.get("pico_json") or {}).get("session_diagnostics_json") or {}
    if not (isinstance(m4a, dict) and m4a.get("applied") is True):
        m4a = pj_sd.get("m4_active")
        if not (isinstance(m4a, dict) and m4a.get("applied") is True):
            return None
    m4fqi = pj_sd.get("m4_full_question_interpreter") or {}
    cand = m4fqi.get("candidate")
    return cand if (isinstance(cand, dict) and cand) else None



def _to_response(doc: dict) -> SearchSessionResponse:
    next_step = None
    status = doc["status"]
    if status == "draft":
        next_step = "Evidence search pipeline will run here in the next build step."
    elif status == "searching":
        next_step = "Searching Europe PMC for evidence…"
    elif status == "ranking":
        next_step = "Normalizing, deduplicating, and ranking evidence…"
    elif status == "awaiting_selection":
        next_step = "Review the ranked evidence and adjust your selections."
    elif status == "error":
        next_step = "An error occurred while running the search pipeline."
    return SearchSessionResponse(
        id=doc["id"],
        user_id=doc["user_id"],
        original_question=doc.get("original_question"),
        sanitized_question=doc.get("sanitized_question"),
        topic_label=doc.get("topic_label"),
        clinical_question_type=doc.get("clinical_question_type"),
        pico_json=doc.get("pico_json", {}) or {},
        # Phase 5A
        picott_json=(doc.get("picott_json")
                      or (doc.get("pico_json") or {}).get("picott_json")
                      or {}),
        spelling_corrections=((doc.get("pico_json") or {}).get("spelling_corrections")
                               or []),
        status=status,
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        expires_at=doc["expires_at"],
        deleted_at=doc.get("deleted_at"),
        next_step=next_step,
        evaluator_json=doc.get("evaluator_json"),
        error_message=doc.get("error_message"),
        # Phase 5D.1 — surface the new fields on every GET response.
        interpreted_question_display_json=doc.get(
            "interpreted_question_display_json"
        ),
        default_selection_summary=doc.get("default_selection_summary"),
        # Phase 6F closeout (IMPL-2) — clarification banner field.
        # Populated only when status==``awaiting_clinical_subject``.
        clarification_message=doc.get("clarification_message_text"),
        # Phase 6F closeout (PROOF E) — closed-set efficiency
        # instrumentation. Computed at GET time from persisted state
        # and attached to the doc via the ``_session_diagnostics_json``
        # synthetic key. ``None`` for the list-view + legacy callsites
        # that don't compute diagnostics.
        session_diagnostics_json=doc.get("_session_diagnostics_json"),
        # Phase L1 — explicit evidence-lens / population-focus prefs.
        evidence_lens=evidence_lens.normalize_lens(doc.get("evidence_lens")),
        population_focus=evidence_lens.normalize_population(
            doc.get("population_focus")
        ),
        preference_mode=(
            doc.get("preference_mode") or evidence_lens.DEFAULT_PREFERENCE_MODE
        ),
        l1_preference_diagnostics=doc.get("l1_preference_diagnostics_json"),
        l1_controls_visible=evidence_lens.l1_ui_visible(),
    )


_VALID_SESSION_STATUSES = frozenset({
    "draft", "blocked_phi", "searching", "ranking", "brokering_access",
    "fetching_oa_full_text", "awaiting_selection", "awaiting_clinical_subject",
    "summarizing", "complete", "error",
})


def _minimal_response(doc: dict) -> SearchSessionResponse:
    """Fallback list entry for a session whose full serialization fails (e.g. a
    legacy / partially-written ``*_json`` field of the wrong type). Keeps the
    session visible in history with safe defaults instead of 500-ing the entire
    list because of one bad doc. Uses only ``.get()`` + coercions so it cannot
    itself raise for a well-formed-enough doc."""
    now = datetime.now(timezone.utc)
    status = doc.get("status")
    if status not in _VALID_SESSION_STATUSES:
        status = "error"
    created = doc.get("created_at") or now
    cqt = doc.get("clinical_question_type")
    em = doc.get("error_message")
    return SearchSessionResponse(
        id=str(doc.get("id") or ""),
        user_id=str(doc.get("user_id") or ""),
        original_question=doc.get("original_question"),
        sanitized_question=doc.get("sanitized_question"),
        topic_label=doc.get("topic_label"),
        clinical_question_type=cqt if isinstance(cqt, str) else None,
        status=status,
        created_at=created,
        updated_at=doc.get("updated_at") or created,
        expires_at=doc.get("expires_at") or (created + timedelta(days=30)),
        error_message=em if isinstance(em, str) else None,
    )


def _json_safe(obj: Any) -> tuple[Any, int]:
    """Recursively replace NaN / +-Infinity floats (which Starlette's
    JSONResponse rejects with ``allow_nan=False`` → 500) with ``None``. A stray
    NaN buried in a persisted ``*_json`` field (e.g. a score) otherwise 500s the
    ENTIRE list after the per-doc mapping. Returns (cleaned, replaced_count)."""
    bad = 0

    def walk(o: Any) -> Any:
        nonlocal bad
        if isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                bad += 1
                return None
            return o
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        return o

    return walk(obj), bad


@router.post("/")
async def create_session(
    body: SearchSessionCreate,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    now = datetime.now(timezone.utc)
    # Phase L1 — normalize explicit preferences (if any) at creation.
    _l1_lens = evidence_lens.normalize_lens(body.evidence_lens)
    _l1_pop = evidence_lens.normalize_population(body.population_focus)
    # Phase 6F.2 — measure PHI guard latency. This is the ONLY stage
    # that runs for a PHI-blocked session, so we record it inline
    # (no SessionDiagnostics binding needed — the route handler isn't
    # the pipeline). For non-blocked sessions the pipeline will own
    # its own stage timing, including a separate ``phi_guard`` reading
    # if needed (we intentionally do not stamp it twice).
    import time as _phi_time
    _phi_start = _phi_time.perf_counter()
    screen = phi_guard.screen(body.question)
    _phi_ms = int(round((_phi_time.perf_counter() - _phi_start) * 1000))
    expires_at = now + timedelta(days=settings.SESSION_TTL_DAYS)

    if screen.is_blocked:
        # Do NOT persist the raw question OR sanitized version.
        # Do NOT schedule the pipeline.
        session = SearchSessionDoc(
            user_id=current_user["id"],
            original_question=None,
            sanitized_question=None,
            status="blocked_phi",
            expires_at=expires_at,
        )
        doc = session.model_dump()
        for k in ("created_at", "updated_at", "expires_at"):
            doc[k] = doc[k].astimezone(timezone.utc)
        # Phase L1 — record the requested preference for audit even
        # though the pipeline never runs for a PHI-blocked session.
        doc["evidence_lens"] = _l1_lens
        doc["population_focus"] = _l1_pop
        # Phase 6C — closed-set expansion metadata. PHI-blocked sessions
        # never invoke scholar_like_expansion; the metadata records
        # WHY for the admin audit trail.
        from services.scholar_like_expansion import expansion_mode as _exp_mode
        doc["expansion_mode_used"] = _exp_mode()
        doc["active_expansion_applied"] = False
        doc["active_expansion_candidate_count"] = 0
        doc["active_expansion_latency_ms"] = 0
        doc["active_expansion_reason"] = "phi_blocked"
        # Phase 6F.2 — closed-set diagnostics block. For PHI-blocked
        # sessions all three fields read as zeros / empty per the
        # directive: 0 LLM calls, 0 tokens, and only the ``phi_guard``
        # stage in ``latency_by_stage_json``.
        doc["session_diagnostics_json"] = {
            "llm_calls_count": 0,
            "llm_calls_by_stage": {},
            "token_estimate": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimate_method": "provider_reported",
            },
            "latency_by_stage_json": {"phi_guard": _phi_ms},
        }
        await db[Collections.SEARCH_SESSIONS].insert_one(doc)
        await audit_service.log(
            "phi_blocked",
            user_id=current_user["id"],
            search_session_id=session.id,
            payload={"reason_codes": screen.reason_codes},
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "blocked_phi",
                "session_id": session.id,
                "reason_codes": screen.reason_codes,
                "user_message": USER_PHI_MESSAGE,
            },
        )

    # Clean path — create session in searching state, kick pipeline.
    session = SearchSessionDoc(
        user_id=current_user["id"],
        original_question=body.question,
        sanitized_question=body.question,
        status="searching",
        expires_at=expires_at,
    )
    doc = session.model_dump()
    for k in ("created_at", "updated_at", "expires_at"):
        doc[k] = doc[k].astimezone(timezone.utc)
    # Phase L1 — persist explicit preferences so the pipeline's
    # evidence-lens stage reads them when ranking. Defaults
    # (clinically_direct / any) are a no-op.
    doc["evidence_lens"] = _l1_lens
    doc["population_focus"] = _l1_pop
    # Phase 6F.2 — pre-stamp the phi_guard latency. The pipeline's
    # finally-flush will merge in the remaining stage timings via the
    # ``_seed_diag_from_doc`` helper in services/pipeline.py — so this
    # value survives.
    doc["session_diagnostics_json"] = {
        "latency_by_stage_json": {"phi_guard": _phi_ms},
    }
    await db[Collections.SEARCH_SESSIONS].insert_one(doc)
    await audit_service.log(
        "search_session_created",
        user_id=current_user["id"],
        search_session_id=session.id,
    )
    await audit_service.log(
        "question_submitted",
        user_id=current_user["id"],
        search_session_id=session.id,
        payload={"question_length": len(body.question)},
    )

    # Schedule the pipeline (fire and forget).
    schedule_pipeline(session.id, current_user["id"], body.question)

    return _to_response(doc)


@router.post("/{session_id}/evidence-lens")
async def update_evidence_lens(
    session_id: str,
    body: EvidenceLensUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Phase L1 — re-apply the clinician's explicit evidence-lens /
    population-focus preference over already-retrieved evidence.

    This performs NO new retrieval and NO summary/citation work. It is a
    soft-boost/shadow re-prioritization of the persisted articles. Every
    boost is re-clamped to each article's persisted relevance cap
    ceiling, so safety caps always dominate; nothing is hidden. Default
    selection flags are left untouched (non-destructive).
    """
    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "user_id": current_user["id"], "deleted_at": None},
        {"_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    if doc.get("status") == "blocked_phi":
        raise HTTPException(
            status_code=409,
            detail="Preferences are unavailable for a PHI-blocked session.",
        )

    # Phase L1.0.1 — production visibility gate. When the L1 UI is hidden
    # (production default), the preference action is refused WITHOUT any
    # mutation, so L1 can never alter production ranking/display even via
    # a direct call. Returns a closed-set disabled response.
    if not evidence_lens.l1_ui_visible():
        return JSONResponse(
            status_code=403,
            content={
                "enabled": False,
                "reason": "l1_preference_ui_disabled_in_production",
                "preference_mode": "shadow",
            },
        )

    new_lens = evidence_lens.normalize_lens(
        body.evidence_lens if body.evidence_lens is not None
        else doc.get("evidence_lens")
    )
    new_pop = evidence_lens.normalize_population(
        body.population_focus if body.population_focus is not None
        else doc.get("population_focus")
    )
    mode = evidence_lens.resolve_preference_mode()

    # Load non-excluded articles + their docs.
    sa_rows: list[dict] = []
    async for r in db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": session_id}, {"_id": 0},
    ):
        if r.get("exclusion_reason"):
            continue
        sa_rows.append(r)
    article_ids = [r["article_id"] for r in sa_rows]
    articles_by_id: dict[str, dict] = {}
    if article_ids:
        async for art in db[Collections.ARTICLES].find(
            {"id": {"$in": article_ids}}, {"_id": 0},
        ):
            articles_by_id[art["id"]] = art

    rows: list[dict] = []
    for sa in sa_rows:
        art = articles_by_id.get(sa["article_id"])
        if not art:
            continue
        base = sa.get("base_rank_score")
        if base is None:
            base = sa.get("overall_rank") or 0
        rows.append({
            "_sa_id": sa["id"],
            "_article": art,
            "overall_rank_score": float(base),
            "relevance_cap_applied": sa.get("relevance_cap_applied"),
        })
    # Deterministic baseline order before applying the preference.
    rows.sort(key=lambda r: -float(r.get("overall_rank_score") or 0.0))

    diag = evidence_lens.apply_evidence_lens(
        rows, lens=new_lens, population=new_pop, mode=mode,
    )

    # Persist per-article updates.
    for r in rows:
        await db[Collections.SEARCH_SESSION_ARTICLES].update_one(
            {"id": r["_sa_id"]},
            {"$set": {
                "overall_rank": int(round(float(r.get("overall_rank_score") or 0.0))),
                "l1_preference_boost": float(r.get("l1_preference_boost") or 0.0),
                "l1_preference_matched": bool(r.get("l1_preference_matched")),
                "l1_preference_signals": list(r.get("l1_preference_signals") or []),
                "l1_preference_rank_delta": int(r.get("l1_preference_rank_delta") or 0),
                "updated_at": datetime.now(timezone.utc),
            }},
        )

    await db[Collections.SEARCH_SESSIONS].update_one(
        {"id": session_id},
        {"$set": {
            "evidence_lens": new_lens,
            "population_focus": new_pop,
            "preference_mode": mode,
            "preference_provenance": evidence_lens.PREFERENCE_PROVENANCE,
            "l1_preference_diagnostics_json": diag,
            "updated_at": datetime.now(timezone.utc),
        }},
    )
    await audit_service.log(
        "evidence_lens_updated",
        user_id=current_user["id"],
        search_session_id=session_id,
        payload={
            "evidence_lens": new_lens,
            "population_focus": new_pop,
            "preference_mode": mode,
            "articles_affected": int(diag.get("articles_boosted") or 0),
        },
    )

    updated = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id}, {"_id": 0},
    )
    return _to_response(updated)


def _bucket(created_at: datetime, now: datetime) -> str:
    if created_at.tzinfo is not None:
        created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    delta = now - created_at
    if delta < timedelta(days=1):
        return "today"
    if delta < timedelta(days=7):
        return "previous_7_days"
    return "previous_30_days"


@router.get("/")
async def list_sessions(
    group: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=settings.SESSION_TTL_DAYS)
    cursor = (
        db[Collections.SEARCH_SESSIONS]
        .find(
            {
                "user_id": current_user["id"],
                "deleted_at": None,
                "created_at": {"$gte": cutoff},
            },
            {"_id": 0},
        )
        .sort("created_at", -1)
        # Opt into external (disk) sort: session docs carry large embedded
        # diagnostics JSON, and a user with many sessions otherwise blows past
        # MongoDB's 32MB in-memory sort cap ("Sort exceeded memory limit ...
        # did not opt in to external sorting", code 292) → 500 on the whole list.
        .allow_disk_use(True)
    )
    # The flat list honors the caller's pagination (the FE sends limit/skip) so
    # the response payload stays small; ``group=recent`` needs the full window.
    fetch_len = 500
    if group != "recent":
        if skip:
            cursor = cursor.skip(skip)
        cursor = cursor.limit(limit)
        fetch_len = limit
    try:
        docs = await cursor.to_list(length=fetch_len)
    except Exception:  # noqa: BLE001
        logger.exception(
            "list_sessions: session fetch failed for user=%s", current_user.get("id"),
        )
        docs = []
    # Resilient mapping: a single session doc that fails full serialization
    # (e.g. a legacy / bad-typed ``*_json`` field) must NOT 500 the whole list.
    # Fall back to a minimal entry and record the failure kind for diagnostics.
    items: list[SearchSessionResponse] = []
    degraded: list[str] = []
    for d in docs:
        try:
            items.append(_to_response(d))
        except Exception as exc:  # noqa: BLE001
            degraded.append(type(exc).__name__)
            logger.exception(
                "list_sessions: _to_response failed for session id=%s status=%r; "
                "rendering minimal entry",
                d.get("id"), d.get("status"),
            )
            try:
                items.append(_minimal_response(d))
            except Exception:  # noqa: BLE001
                logger.exception(
                    "list_sessions: minimal render also failed for id=%s; skipping",
                    d.get("id"),
                )
    if group == "recent":
        buckets: dict[str, list[SearchSessionResponse]] = {
            "today": [], "previous_7_days": [], "previous_30_days": [],
        }
        for it in items:
            buckets[_bucket(it.created_at, now)].append(it)
        payload: Any = buckets
    else:
        payload = items
    # Final JSON-safety pass. A NaN / Infinity float buried in a persisted
    # ``*_json`` field (e.g. a score) makes Starlette's JSONResponse
    # (allow_nan=False) 500 AFTER the per-doc mapping above — so scrub it here.
    try:
        safe, nan_count = _json_safe(jsonable_encoder(payload))
    except Exception as exc:  # noqa: BLE001
        # Exotic non-serializable value somewhere — fall back to minimal-only.
        degraded.append("encode:" + type(exc).__name__)
        logger.exception("list_sessions: payload encoding failed; minimal-only fallback")
        minimal: list[SearchSessionResponse] = []
        for d in docs:
            try:
                minimal.append(_minimal_response(d))
            except Exception:  # noqa: BLE001
                logger.exception("list_sessions: minimal render failed for id=%s", d.get("id"))
        if group == "recent":
            mb: dict[str, list[SearchSessionResponse]] = {
                "today": [], "previous_7_days": [], "previous_30_days": [],
            }
            for it in minimal:
                mb[_bucket(it.created_at, now)].append(it)
            safe, nan_count = _json_safe(jsonable_encoder(mb))
        else:
            safe, nan_count = _json_safe(jsonable_encoder(minimal))
    if nan_count:
        degraded.append(f"nan:{nan_count}")
    resp = JSONResponse(content=safe)
    if degraded:
        # Non-fatal diagnostics header (counts / exception kinds only, no PHI).
        resp.headers["X-List-Degraded"] = ",".join(sorted(set(degraded)))[:200]
    return resp


async def _owned_session(session_id: str, user_id: str) -> dict:
    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "user_id": user_id, "deleted_at": None},
        {"_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Search session not found.")
    return doc


async def _readable_session(
    session_id: str, current_user: dict,
) -> dict:
    """Phase 5F.2 — owner-or-admin readable session loader. Returns 404
    on missing OR non-owner-non-admin access (no info leak — cross-user
    callers cannot distinguish "session doesn't exist" from "session
    exists but you can't see it")."""
    db = get_db()
    is_admin = (current_user.get("role") == "admin")
    query: dict = {"id": session_id, "deleted_at": None}
    if not is_admin:
        query["user_id"] = current_user["id"]
    doc = await db[Collections.SEARCH_SESSIONS].find_one(query, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Search session not found.")
    return doc


@router.get("/{session_id}")
async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    doc = await _readable_session(session_id, current_user)

    # Load related rows.
    queries = await (
        db[Collections.SEARCH_QUERIES]
        .find({"search_session_id": session_id}, {"_id": 0})
        .sort("created_at", 1)
        .to_list(length=50)
    )
    sa_rows = await (
        db[Collections.SEARCH_SESSION_ARTICLES]
        .find({"search_session_id": session_id}, {"_id": 0})
        .sort("overall_rank", -1)
        .to_list(length=500)
    )

    # Phase 6F.1 — stable secondary sort.
    # MongoDB's primary sort on ``overall_rank`` (desc) is correct in
    # isolation, but two articles with the same overall_rank (e.g., an
    # on-target article with raw overall_rank=20 vs a wrong-anchor
    # article capped at 20) tie on the primary key and MongoDB's
    # cursor order across ties is implementation-defined. The visible
    # symptom from the Phase 6F tester verdict was wrong-disease
    # articles surfacing above on-target articles at the same numeric
    # rank.
    #
    # Phase 6F closeout (IMPL-1) — within each bucket use strict
    # relevance_score DESC, then evidence_design_score DESC, then
    # canonical_key ASC. ``overall_rank`` is intentionally dropped from
    # the WITHIN-bucket key so the bucket order reflects the
    # user-facing relevance number directly (after the Phase 6F.1
    # anchor_match boost has been applied).
    #
    # Sort key:
    #   1. bucket: included_by_default=true rows first (0), then false (1)
    #   2. relevance_score DESC  (the user-facing concept-match score)
    #   3. evidence_design_score DESC  (design strength tiebreaker)
    #   4. canonical_key ASC  (deterministic final tiebreaker)
    sa_rows.sort(
        key=lambda r: (
            0 if bool(r.get("included_by_default")) else 1,
            -float(r.get("relevance_score") or 0.0),
            -float(r.get("evidence_design_score") or 0.0),
            str(
                (r.get("_article") or {}).get("canonical_key")
                or r.get("article_id")
                or ""
            ),
        )
    )

    # Phase 6F closeout (PROOF E) — efficiency instrumentation panel.
    # Build the closed-set ``session_diagnostics_json`` block from
    # persisted state. NO PHI, NO question text, NO free-text labels.
    anchors_payload = (doc.get("primary_clinical_anchors_json") or {})
    primary_anchors = anchors_payload.get("primary_anchors") or []
    generic_modifiers = anchors_payload.get("generic_modifiers") or []
    anchor_failure_reason = anchors_payload.get("anchor_failure_reason")
    raw_query_rows = await (
        db[Collections.SEARCH_QUERIES]
        .find({"search_session_id": session_id}, {"_id": 0})
        .to_list(length=200)
    )
    # Per-source call breakdown — closed-set integer counts only.
    src_breakdown: dict[str, int] = {}
    for q in raw_query_rows:
        s = (q.get("source") or "unknown").lower()
        src_breakdown[s] = int(src_breakdown.get(s) or 0) + 1
    raw_candidate_count = len(sa_rows)
    filtered_candidate_count = sum(
        1 for r in sa_rows if not r.get("exclusion_reason")
    )
    # Anchor coverage in the visible top-10 (closed-set: count of rows
    # whose ranking_boost_reasons contains an ``anchor_match_*`` entry).
    top10 = sa_rows[:10]
    top10_anchor_coverage = sum(
        1 for r in top10
        if any(
            isinstance(b, str) and b.startswith("anchor_match_")
            for b in (r.get("ranking_boost_reasons") or [])
        )
    )
    fallback_triggered = any(
        (q.get("planner_source") == "legacy_planner")
        and bool(q.get("is_fallback_strategy"))
        for q in raw_query_rows
    )
    if doc.get("status") == "blocked_phi":
        anchor_validation_status = "phi_blocked"
    elif anchor_failure_reason == "no_primary_clinical_entity":
        anchor_validation_status = "no_primary_clinical_entity"
    elif primary_anchors:
        anchor_validation_status = "passed"
    else:
        anchor_validation_status = "no_anchor_recorded"
    # Phase 6F.2 — merge persisted diagnostics (written by the pipeline's
    # final flush and the PHI-guard pre-stamp) for the three closed-set
    # fields. These were ``null`` in 6F closeout and are now populated.
    persisted_diag = dict(doc.get("session_diagnostics_json") or {})
    persisted_llm_count = persisted_diag.get("llm_calls_count")
    persisted_llm_by_stage = persisted_diag.get("llm_calls_by_stage") or {}
    persisted_token_estimate = persisted_diag.get("token_estimate")
    persisted_latency = persisted_diag.get("latency_by_stage_json") or {}
    # Phase H1 — closed-set Clinical Context Resolver diagnostics
    # surfaces. These are populated by the pipeline's
    # ``resolver_diag`` dict in the final flush. Default values when
    # the pipeline never touched the resolver path (e.g.
    # PHI-blocked, early generic-only short-circuit before the
    # post-PICOTT anchor gate) are deterministic — bool False, None
    # for nullable fields, 0 for ints, {} for nested dicts, and
    # ``deterministic`` for ``interpretation_source``.
    persisted_resolver_called = persisted_diag.get(
        "clinical_context_resolver_called",
    )
    persisted_resolver_mode = persisted_diag.get(
        "clinical_context_resolver_mode",
    )
    persisted_resolver_accepted = persisted_diag.get(
        "clinical_context_resolver_accepted",
    )
    persisted_resolver_skipped = persisted_diag.get(
        "clinical_context_resolver_skipped_reason",
    )
    persisted_resolver_confidence = persisted_diag.get(
        "resolver_confidence",
    )
    persisted_resolver_rejection = persisted_diag.get(
        "resolver_rejection_reason",
    )
    persisted_resolver_latency = persisted_diag.get(
        "resolver_latency_ms",
    )
    persisted_resolver_token_estimate = persisted_diag.get(
        "resolver_token_estimate",
    )
    persisted_interpretation_source = persisted_diag.get(
        "interpretation_source",
    )
    session_diagnostics = {
        "phase": "phase6f_closeout",
        "anchor_detected": bool(primary_anchors)
            and anchor_failure_reason != "no_primary_clinical_entity",
        "primary_anchor_count": len(primary_anchors),
        "generic_modifier_count": len(generic_modifiers),
        "anchor_repair_applied": bool(anchors_payload.get("anchor_repair_applied")),
        "anchor_validation_status": anchor_validation_status,
        "queries_emitted_count": len(raw_query_rows),
        "source_api_call_count": sum(src_breakdown.values()),
        "source_api_call_breakdown": src_breakdown,
        "raw_candidate_count": raw_candidate_count,
        "filtered_candidate_count": filtered_candidate_count,
        "top10_anchor_coverage": int(top10_anchor_coverage),
        "fallback_triggered": bool(fallback_triggered),
        # Phase 6F.2 — closed-set efficiency instrumentation. NO PHI,
        # NO prompts, NO completions ever land in these fields. The
        # pipeline's ``services/diagnostics.py::SessionDiagnostics``
        # writes integer counters only.
        "llm_calls_count": (
            int(persisted_llm_count)
            if isinstance(persisted_llm_count, int) else None
        ),
        "llm_calls_by_stage": dict(persisted_llm_by_stage)
            if isinstance(persisted_llm_by_stage, dict) else {},
        "token_estimate": (
            dict(persisted_token_estimate)
            if isinstance(persisted_token_estimate, dict) else None
        ),
        "latency_by_stage_json": (
            dict(persisted_latency)
            if isinstance(persisted_latency, dict) else None
        ),
        # Phase H1 — closed-set Clinical Context Resolver diagnostics.
        # NO PHI / NO prompt / NO completion text ever lands here —
        # the resolver service validates this with the
        # ``_scrub_raw_output`` regex pass + Pydantic schema lock.
        "clinical_context_resolver_called": bool(persisted_resolver_called) if persisted_resolver_called is not None else False,
        "clinical_context_resolver_mode": (
            persisted_resolver_mode
            if isinstance(persisted_resolver_mode, str) else "fallback_active"
        ),
        "clinical_context_resolver_accepted": bool(persisted_resolver_accepted) if persisted_resolver_accepted is not None else False,
        "clinical_context_resolver_skipped_reason": (
            persisted_resolver_skipped
            if (persisted_resolver_skipped is None
                or isinstance(persisted_resolver_skipped, str))
            else None
        ),
        "resolver_confidence": (
            float(persisted_resolver_confidence)
            if isinstance(persisted_resolver_confidence, (int, float))
            else None
        ),
        "resolver_rejection_reason": (
            persisted_resolver_rejection
            if (persisted_resolver_rejection is None
                or isinstance(persisted_resolver_rejection, str))
            else None
        ),
        "resolver_latency_ms": (
            int(persisted_resolver_latency)
            if isinstance(persisted_resolver_latency, int)
            else 0
        ),
        "resolver_token_estimate": (
            dict(persisted_resolver_token_estimate)
            if isinstance(persisted_resolver_token_estimate, dict)
            else {}
        ),
        "interpretation_source": (
            persisted_interpretation_source
            if isinstance(persisted_interpretation_source, str)
            else "deterministic"
        ),
        # Phase M3.2 — preview-only Active Semantic Repair chip
        # surface. Closed-set fields persisted by the pipeline.
        # ``None`` for sessions that pre-date Phase M3.2 OR where
        # the active path never ran (PHI-blocked / clarification).
        # The clinician UI gates chip rendering on ``applied=True``;
        # all other states (kill switch engaged, fallback, K4A
        # excluded) keep the chip hidden but the diagnostic block
        # is still queryable for adjudication. NO prompts / NO
        # completions / NO PHI ever land here.
        #
        # Phase M3.2 code-review patch — the CLINICIAN surface is
        # restricted to ``{applied, shape, chip}`` only. Operator-
        # facing internals (decision string, kill_switch_state,
        # timeout/fallback status, route_owned, sidecar_doc_id) are
        # NOT exposed to clinicians; they remain on the admin
        # diagnostics surface for adjudication. ``None`` if not
        # applied — the chip simply will not render.
        "m3_2_active_repair": _clinician_m3_2_block(
            persisted_diag.get("m3_2_active_repair"),
        ),
        # Phase M4.1 — preview-only Active M4 transparency indicator.
        # Clinician surface restricted to ``{applied, chip}``. ``None``
        # for legacy / non-applied sessions (indicator will not render).
        "m4_active": _clinician_m4_active_block(
            persisted_diag.get("m4_active"),
        ),
    }

    # Phase K3A — closed-set article lane diagnostics projection.
    # Reads ``session_diagnostics_json.article_lane_diagnostics`` (set
    # by ``services/pipeline.py``) and projects ONLY closed-set fields.
    # NEVER exposes raw prompts, completions, PHI, evidence text,
    # uploaded content, emails, credentials, or secrets. Every
    # diagnostic carries ``cap_mode="shadow"``, ``would_cap=False``,
    # ``would_exclude=False``.
    raw_lane_diag = (
        doc.get("session_diagnostics_json", {}) or {}
    ).get("article_lane_diagnostics")
    if isinstance(raw_lane_diag, dict):
        try:
            import json as _json_k3a
            _blob = _json_k3a.dumps(raw_lane_diag)
            _forbidden_hit = any(
                s in _blob for s in _K3A_FORBIDDEN_DIAG_SUBSTRINGS
            )
        except Exception:  # noqa: BLE001
            _forbidden_hit = False
        if not _forbidden_hit:
            # Phase K3A.1 — clinician per-article projection is trimmed
            # to ONLY the closed-set fields a clinician needs. We
            # explicitly DROP the raw ``cap_candidates`` array,
            # per-rule trace, ``branch_alignment``, ``matched_branch_ids``,
            # ``evidence_role_candidate``, ``article_intent_fit``,
            # ``lane_confidence``, ``cap_severity``, ``cap_confidence``.
            # These remain on the ADMIN diagnostics endpoint for QA
            # work. ``explanation`` is downgraded from a list of rule
            # IDs to a single ``,`` -joined string capped at 400 chars.
            _slim_lane: list[dict[str, Any]] = []
            for d in (raw_lane_diag.get("diagnostics") or []):
                if not isinstance(d, dict):
                    continue
                _expl_items = [
                    str(x) for x in (d.get("explanation") or [])
                    if isinstance(x, str)
                ]
                _expl_joined = ",".join(_expl_items)
                if len(_expl_joined) > 400:
                    _expl_joined = _expl_joined[:400] + "...[truncated]"
                _qa = [
                    str(w) for w in (d.get("qa_warnings") or [])
                    if isinstance(w, str)
                ][:10]
                _cap_codes = list(d.get("cap_reason_codes") or [])
                _slim_lane.append({
                    "article_id": str(d.get("article_id") or ""),
                    "search_session_article_id": d.get(
                        "search_session_article_id"
                    ),
                    "lane_label": str(d.get("lane_label") or ""),
                    "directness_label": str(
                        d.get("directness_label") or "indirect"
                    ),
                    "has_cap_candidates": bool(
                        d.get("cap_candidates") or []
                    ),
                    "cap_reason_codes": _cap_codes,
                    "explanation": _expl_joined,
                    "qa_warnings": _qa,
                    # Defense-in-depth shadow invariants kept visible.
                    "cap_mode": "shadow",
                    "would_cap": False,
                    "would_exclude": False,
                    # Phase K3A.3 — closed-set join status on the
                    # ``trimmed_v1.1`` contract.
                    "article_join_status": str(
                        d.get("article_join_status") or "stable"
                    ),
                    # Phase K3B Wave 1 Step 1 — closed-set shadow
                    # enforcement fields on the ``trimmed_v1.3``
                    # contract. Phase K3B Wave 1 Step 1.5 — Item 2:
                    # both ``enforcement_explanation`` and
                    # ``would_have_been_excluded_reason_codes`` are
                    # explicitly normalized to ``List[str]`` (every
                    # element coerced to ``str``; non-strings filtered).
                    # ``enforcement_explanation`` is the same closed-
                    # set rule-id list as on admin (capped at 10 on
                    # clinician for compactness). Downstream consumers
                    # must adapt to the list shape.
                    "would_have_been_excluded_in_active": bool(
                        d.get("would_have_been_excluded_in_active")
                    ),
                    "would_have_been_excluded_reason_codes": [
                        str(c) for c in (
                            d.get("would_have_been_excluded_reason_codes")
                            or []
                        )
                        if isinstance(c, str)
                    ],
                    "enforcement_mode": str(
                        d.get("enforcement_mode") or "disabled"
                    ),
                    "enforcement_explanation": [
                        str(x) for x in (
                            d.get("enforcement_explanation_full") or []
                        ) if isinstance(x, str)
                    ][:10],
                })
            session_diagnostics["article_lane_diagnostics"] = {
                "schema_version": str(raw_lane_diag.get(
                    "schema_version"
                ) or "K3A.1.0"),
                "lane_label_mode": str(raw_lane_diag.get(
                    "lane_label_mode"
                ) or "shadow"),
                "cap_detection_mode": str(raw_lane_diag.get(
                    "cap_detection_mode"
                ) or "shadow"),
                # Phase K3B Wave 1 Step 1.5 — projection contract
                # bumped from ``trimmed_v1.2`` to ``trimmed_v1.3``
                # (``enforcement_explanation`` is now ``List[str]``
                # — same closed-set rule IDs, type normalization
                # only — downstream consumers must adapt).
                "projection_contract": "trimmed_v1.3",
                "diagnostics": _slim_lane,
            }

    # Phase K3B Wave 1 Step 2 — closed-set clinician-facing banner
    # for actively-excluded retracted articles. Renders ONLY when
    # ``N > 0`` (no empty state). Closed-set fields only — NO rule
    # IDs, NO raw prompts/completions, NO evidence text, NO PHI.
    raw_k3b_active = (
        doc.get("session_diagnostics_json", {}) or {}
    ).get("k3b_active_exclusions")
    if (
        isinstance(raw_k3b_active, dict)
        and isinstance(raw_k3b_active.get("exclusions"), list)
        and len(raw_k3b_active.get("exclusions") or []) > 0
    ):
        # Defense-in-depth leak guard on EVERY string field in the
        # banner payload. The pipeline's persistence path already
        # scrubs at write-time, but we re-scrub on read so legacy
        # rows / direct-DB writes / future regressions can NEVER
        # leak a forbidden sentinel through the clinician banner.
        def _scrub_banner_str(v: Any) -> Any:
            if not isinstance(v, str):
                return v
            return v if not any(
                s in v for s in _K3A_FORBIDDEN_DIAG_SUBSTRINGS
            ) else "[REDACTED]"

        _excl_rows: list[dict[str, Any]] = []
        for r in raw_k3b_active.get("exclusions") or []:
            if not isinstance(r, dict):
                continue
            # Map closed-set source_signal → short human label for
            # the clinician banner. No rule traces / rule IDs leak
            # into the clinician surface.
            sig = str(r.get("source_signal") or "")
            label = {
                "pubmed_retracted_publication_type":
                    "PubMed retraction notice",
                "pubmed_retraction_in_link":
                    "PubMed retraction notice",
            }.get(sig, "PubMed retraction notice")
            _excl_rows.append({
                "article_id": str(r.get("article_id") or ""),
                "pmid": (
                    _scrub_banner_str(str(r.get("pmid")))
                    if r.get("pmid") else None
                ),
                "doi": (
                    _scrub_banner_str(str(r.get("doi")))
                    if r.get("doi") else None
                ),
                "title": _scrub_banner_str(
                    str(r.get("title") or "")[:200]
                ),
                "journal": _scrub_banner_str(
                    str(r.get("journal") or "")[:160]
                ),
                "publication_year": r.get("publication_year"),
                "source_signal_label": label,
            })
        n = len(_excl_rows)
        # Exact banner text per spec.
        banner_text = (
            f"{n} retracted article(s) were excluded from these "
            "results because they have been retracted by their "
            "journals."
        )
        session_diagnostics["k3b_active_exclusions_banner"] = {
            "schema_version": "k3b.wave1.step2.banner.v1",
            "total": n,
            "banner_text": banner_text,
            "exclusions": _excl_rows,
            "enforcement_mode": "active",
        }

    # Phase K1.J — closed-set ClinicalQuestionMap diagnostics surface.
    # Reads ``clinical_question_map_json`` from the session document
    # (persisted by ``services/pipeline.py``) and projects ONLY the
    # closed-set fields the testing surface + transparency panel need.
    # NEVER exposes raw prompts, completions, PHI, evidence text,
    # uploaded content, emails, credentials, or secrets.
    cqm_persisted = doc.get("clinical_question_map_json")
    if isinstance(cqm_persisted, dict):
        branches_raw = cqm_persisted.get("branches") or []
        if not isinstance(branches_raw, list):
            branches_raw = []
        # slot_search_policy distribution — closed-set counters.
        policy_dist: dict[str, int] = {
            "required_for_retrieval": 0,
            "optional_for_retrieval": 0,
            "ranking_only": 0,
            "display_only": 0,
            "clarification_needed": 0,
        }

        def _tally(slot_list: Any) -> None:
            if not isinstance(slot_list, list):
                return
            for s in slot_list:
                if not isinstance(s, dict):
                    continue
                p = s.get("slot_search_policy")
                if isinstance(p, str) and p in policy_dist:
                    policy_dist[p] += 1

        _tally(cqm_persisted.get("framework_neutral_slots"))
        for br in branches_raw:
            if not isinstance(br, dict):
                continue
            _tally(br.get("required_concepts"))
            _tally(br.get("optional_concepts"))
            _tally(br.get("ranking_only_concepts"))
            _tally(br.get("display_only_concepts"))

        # Inferred concepts — closed-set list of concept strings only.
        # NO ``rationale`` field is exposed via the diagnostics surface.
        inferred_list_raw = cqm_persisted.get("inferred_concepts") or []
        inferred_concepts_brief: list[dict[str, Any]] = []
        for ic in inferred_list_raw if isinstance(inferred_list_raw, list) else []:
            if isinstance(ic, dict):
                inferred_concepts_brief.append({
                    "concept": ic.get("concept") or "",
                    "confidence": (
                        float(ic.get("confidence") or 0.0)
                        if isinstance(ic.get("confidence"), (int, float))
                        else 0.0
                    ),
                })

        # Branches — slim closed-set projection (no slot payloads).
        branches_brief: list[dict[str, Any]] = []
        for br in branches_raw:
            if not isinstance(br, dict):
                continue
            branches_brief.append({
                "branch_id": br.get("branch_id") or "",
                "intent": br.get("intent") or "",
                "framework": br.get("framework") or "",
                "branch_priority": (
                    int(br.get("branch_priority"))
                    if isinstance(br.get("branch_priority"), int) else 0
                ),
                "branch_confidence": (
                    float(br.get("branch_confidence") or 0.0)
                    if isinstance(br.get("branch_confidence"), (int, float))
                    else 0.0
                ),
                "guideline_first": bool(br.get("guideline_first")),
                "rare_or_complication_rescue": bool(
                    br.get("rare_or_complication_rescue")
                ),
                "retrieval_strategy_hint": (
                    br.get("retrieval_strategy_hint") or "standard"
                ),
            })

        # Phase K1.1 follow-up — surface ``framework_neutral_slots`` on
        # the public clinician GET response. The slots carry ONLY
        # closed-set canonical concepts + slot_role labels (same
        # redaction class as ``slot_search_policy_distribution`` above);
        # never raw prompts, completions, PHI, evidence text, uploaded
        # content, emails, credentials, or secrets. The frontend
        # ``ClinicalQuestionMapPanel`` needs this list to render the
        # per-slot policy chips per the K1.K spec.
        fns_raw = cqm_persisted.get("framework_neutral_slots") or []
        framework_neutral_slots_brief: list[dict[str, Any]] = []
        # Closed-set Literal value sets for defensive validation.
        _SLOT_POLICY_VALUES = {
            "required_for_retrieval", "optional_for_retrieval",
            "ranking_only", "display_only", "clarification_needed",
        }
        _SLOT_SOURCE_VALUES = {
            "deterministic_parser", "deterministic_concept_pack",
            "bounded_resolver_fallback", "manual", "unknown",
        }
        for s in fns_raw if isinstance(fns_raw, list) else []:
            if not isinstance(s, dict):
                continue
            policy = s.get("slot_search_policy")
            policy_str = (
                policy if isinstance(policy, str)
                and policy in _SLOT_POLICY_VALUES
                else "optional_for_retrieval"
            )
            src = s.get("source")
            src_str = (
                src if isinstance(src, str) and src in _SLOT_SOURCE_VALUES
                else "unknown"
            )
            framework_neutral_slots_brief.append({
                "original_phrase": str(s.get("original_phrase") or ""),
                "canonical_concept": str(s.get("canonical_concept") or ""),
                "entity_type": str(s.get("entity_type") or "unknown"),
                "slot_role": str(s.get("slot_role") or "clinical_problem"),
                "slot_search_policy": policy_str,
                "required_for_search": bool(s.get("required_for_search")),
                "inferred": bool(s.get("inferred")),
                "confidence": (
                    float(s.get("confidence") or 0.0)
                    if isinstance(s.get("confidence"), (int, float))
                    else 0.0
                ),
                "synonyms": [
                    str(x) for x in (s.get("synonyms") or [])
                    if isinstance(x, str)
                ],
                "search_terms": [
                    str(x) for x in (s.get("search_terms") or [])
                    if isinstance(x, str)
                ],
                "source": src_str,
            })

        # Phase K2 — closed-set ``branch_query_plan`` projection.
        # Reads ``branch_query_plan`` from the persisted CQM payload
        # and projects ONLY closed-set fields onto the diagnostics
        # surface. NEVER exposes raw prompts, completions, PHI,
        # evidence text, uploaded content, emails, credentials, or
        # secrets. Every per-branch entry carries ``active_for_retrieval=false``
        # and ``mode="shadow"`` by schema construction in the planner.
        branch_query_plan_raw = cqm_persisted.get("branch_query_plan") or {}
        bqp_brief: dict[str, Any] = {}
        if isinstance(branch_query_plan_raw, dict):
            try:
                # The persisted plan is already a closed-set
                # ``BranchQueryPlan.model_dump(mode="json")`` dict. We
                # pass it through with defensive truncation on the
                # query_string lengths (cap at 4 KB per query string)
                # and a sanitised forbidden-substring grep.
                from services.branch_aware_boolean_query_planner import (
                    _FORBIDDEN_QUERY_SUBSTRINGS as _BQP_FORBIDDEN,
                )

                def _safe_query_string(q: str) -> str:
                    if not isinstance(q, str):
                        return ""
                    if any(s in q for s in _BQP_FORBIDDEN):
                        return ""
                    if len(q) > 4096:
                        return q[:4096] + "...[truncated]"
                    return q

                br_list = []
                for br in (branch_query_plan_raw.get("branches") or []):
                    if not isinstance(br, dict):
                        continue
                    qfams = []
                    for fam in (br.get("query_families") or []):
                        if not isinstance(fam, dict):
                            continue
                        qfams.append({
                            "family_name": fam.get("family_name") or "",
                            "source": fam.get("source") or "",
                            "qa_checks_passed": bool(fam.get("qa_checks_passed")),
                            "query_string": _safe_query_string(
                                fam.get("query_string") or ""
                            ),
                            "notes": list(fam.get("notes") or [])[:8],
                        })
                    src_specific = {}
                    raw_src = br.get("source_specific_query_strings") or {}
                    if isinstance(raw_src, dict):
                        for k, v in raw_src.items():
                            if isinstance(k, str) and isinstance(v, str):
                                src_specific[k] = _safe_query_string(v)
                    br_list.append({
                        "branch_id": br.get("branch_id") or "",
                        "intent": br.get("intent") or "",
                        "framework": br.get("framework") or "",
                        "branch_priority": int(br.get("branch_priority") or 0),
                        "branch_confidence": (
                            float(br.get("branch_confidence") or 0.0)
                            if isinstance(br.get("branch_confidence"), (int, float))
                            else 0.0
                        ),
                        "required_concepts_used": list(
                            br.get("required_concepts_used") or []
                        ),
                        "optional_concepts_used": list(
                            br.get("optional_concepts_used") or []
                        ),
                        "ranking_only_concepts_not_searched": list(
                            br.get("ranking_only_concepts_not_searched") or []
                        ),
                        "display_only_concepts_not_searched": list(
                            br.get("display_only_concepts_not_searched") or []
                        ),
                        "inferred_concepts_used": list(
                            br.get("inferred_concepts_used") or []
                        ),
                        "inferred_concepts_not_required": list(
                            br.get("inferred_concepts_not_required") or []
                        ),
                        "validated_MeSH_terms": [
                            str(t) for t in (br.get("validated_MeSH_terms") or [])
                            if isinstance(t, str)
                        ],
                        "unvalidated_MeSH_terms_rejected_or_pending": list(
                            br.get(
                                "unvalidated_MeSH_terms_rejected_or_pending"
                            ) or []
                        ),
                        "title_abstract_terms": [
                            str(t) for t in (br.get("title_abstract_terms") or [])
                            if isinstance(t, str)
                        ][:50],
                        "MeSH_subheadings_used": [
                            str(t) for t in (br.get("MeSH_subheadings_used") or [])
                            if isinstance(t, str)
                        ],
                        "query_families": qfams,
                        "source_specific_query_strings": src_specific,
                        "query_warnings": [
                            str(w) for w in (br.get("query_warnings") or [])
                            if isinstance(w, str)
                        ][:50],
                        "query_quality_checks": list(
                            br.get("query_quality_checks") or []
                        )[:50],
                        "active_for_retrieval": False,
                        "mode": "shadow",
                    })
                bqp_brief = {
                    "schema_version": branch_query_plan_raw.get(
                        "schema_version"
                    ) or "K2.1.0",
                    "mode": branch_query_plan_raw.get("mode") or "shadow",
                    "active_for_retrieval": False,
                    "generated_at": branch_query_plan_raw.get(
                        "generated_at"
                    ) or "",
                    "branches": br_list,
                }
            except Exception:  # noqa: BLE001
                bqp_brief = {
                    "schema_version": "K2.1.0",
                    "mode": "disabled",
                    "active_for_retrieval": False,
                    "generated_at": "",
                    "branches": [],
                }

        cqm_diag = {
            "interpretation_provenance": (
                cqm_persisted.get("interpretation_provenance") or "deterministic_parser"
            ),
            "primary_intent": cqm_persisted.get("primary_intent") or "",
            "secondary_intents": (
                list(cqm_persisted.get("secondary_intents") or [])
            ),
            "frameworks": list(cqm_persisted.get("frameworks") or []),
            "branches": branches_brief,
            "slot_search_policy_distribution": policy_dist,
            "framework_neutral_slots": framework_neutral_slots_brief,
            "inferred_concepts": inferred_concepts_brief,
            "branch_query_plan": bqp_brief,
            "resolver_called": bool(persisted_resolver_called) if persisted_resolver_called is not None else False,
            "resolver_accepted": bool(persisted_resolver_accepted) if persisted_resolver_accepted is not None else False,
            "rejection_reason": (
                persisted_resolver_rejection
                if (persisted_resolver_rejection is None
                    or isinstance(persisted_resolver_rejection, str))
                else None
            ),
            "schema_version": (
                cqm_persisted.get("schema_version") or "1.0"
            ),
            "prompt_version": cqm_persisted.get("prompt_version"),
            "clarification_required": bool(
                cqm_persisted.get("clarification_required")
            ),
            "confidence": (
                float(cqm_persisted.get("confidence") or 0.0)
                if isinstance(cqm_persisted.get("confidence"), (int, float))
                else 0.0
            ),
            "deterministic_signal_summary": dict(
                cqm_persisted.get("deterministic_signal_summary") or {}
            ),
        }
        session_diagnostics["clinical_question_map"] = cqm_diag

    # Phase 6F + Phase 6G.3 — Default+Excluded contradiction self-heal.
    # An article must NEVER be both
    # ``user_included=True``/``included_by_default=True`` AND carry a
    # non-empty ``exclusion_reason``. This contradiction can appear when:
    #   • A stale UI cache toggled user_included=True after the article
    #     was already excluded.
    #   • A re-rank emitted a new ``exclusion_reason`` that wasn't
    #     reflected in the user_included / default flags.
    #   • Default-selection ran with an older ranker version that
    #     pre-dated the exclusion. (Phase 6G.3)
    # We fix this on read: clear both inclusion flags on any
    # contradicting row, then emit a closed-set
    # ``default_selection_self_healed`` audit event with COUNTS ONLY
    # (no article IDs / titles / PMIDs).
    contradicting_ids: list[str] = []
    for r in sa_rows:
        has_excl = bool(r.get("exclusion_reason"))
        if not has_excl:
            continue
        if r.get("user_included") is True or r.get("included_by_default") is True:
            contradicting_ids.append(r["id"])
    if contradicting_ids:
        now = datetime.now(timezone.utc)
        await db[Collections.SEARCH_SESSION_ARTICLES].update_many(
            {"id": {"$in": contradicting_ids}},
            {"$set": {
                "user_included": False,
                "included_by_default": False,
                "updated_at": now,
            }},
        )
        # Reflect the in-memory rows so the response is consistent.
        contra_set = set(contradicting_ids)
        for r in sa_rows:
            if r["id"] in contra_set:
                r["user_included"] = False
                r["included_by_default"] = False
        # Closed-set audit — counts only. No article identifiers leak.
        try:
            await audit_service.log(
                "default_selection_self_healed",
                user_id=current_user.get("id"),
                search_session_id=session_id,
                payload={
                    "contradicting_count": len(contradicting_ids),
                    "phase": "phase6f",
                    "trigger": "get_session_read",
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phase6f default_selection_self_healed audit failed (non-fatal)"
            )
    # Hydrate articles
    art_ids = [r["article_id"] for r in sa_rows]
    articles_by_id: dict[str, dict] = {}
    if art_ids:
        # Phase H2-tail Task A — defensive author-email scrub on read so
        # legacy records (ingested before the centralized persistence
        # redactor was wired) never leak emails through this endpoint.
        from services.article_email_redactor import redact_article_emails
        async for art in db[Collections.ARTICLES].find({"id": {"$in": art_ids}}, {"_id": 0}):
            articles_by_id[art["id"]] = redact_article_emails(art)

    included: list[dict] = []
    excluded: list[dict] = []
    for r in sa_rows:
        art = articles_by_id.get(r["article_id"])
        if not art:
            continue
        row = {**r, "article": art}
        if r.get("exclusion_reason"):
            excluded.append(row)
        else:
            included.append(row)

    # ---- Graceful "always surface the best available" fallback ----
    # When strict anchor ranking excluded EVERY retrieved article — typically a
    # parsing/anchor imperfection such as an over-specific or fused-phrase
    # anchor ("treatment options for stage 4 prostate cancer") — returning "No
    # results found" despite having retrieved 200+ real, topically-related
    # articles is a strictly worse experience than showing the closest matches
    # (which is what the legacy ClinEvidence build did). Promote the
    # top-ranked excluded rows into the shown list, flagged so the UI /
    # transparency panel can label them as broader, lower-confidence matches.
    # Non-destructive: this only re-partitions THIS GET response; the persisted
    # exclusion_reason / default-selection state is untouched.
    broad_fallback_applied = False
    # Fire when the shown set is too thin. Default threshold 0 preserves the
    # original "only when nothing passed strict ranking" behavior; raising
    # BROAD_RESULTS_FALLBACK_MIN_INCLUDED widens it to the "only 1-2 papers"
    # case as a deterministic, no-LLM floor.
    _min_included = int(os.environ.get("BROAD_RESULTS_FALLBACK_MIN_INCLUDED", "20") or "20")
    if excluded and len(included) <= _min_included:
        _broad_limit = int(os.environ.get("BROAD_RESULTS_FALLBACK_LIMIT", "50"))
        _ranked_ex = sorted(
            excluded,
            key=lambda r: -float(
                r.get("overall_rank_score") or r.get("overall_rank") or 0.0
            ),
        )
        _promote = _ranked_ex[:max(1, _broad_limit)]
        _promoted_marker = set(map(id, _promote))
        for _row in _promote:
            _row["broad_fallback"] = True
            _row["broad_fallback_reason"] = _row.get("exclusion_reason")
        # APPEND (not replace) so any real included rows are preserved when the
        # threshold is >0. When included was empty this is identical to the
        # original ``included = _promote``. Promoted broader/lower-confidence
        # rows go after the strict matches.
        included = included + _promote
        excluded = [r for r in excluded if id(r) not in _promoted_marker]
        broad_fallback_applied = True
        logger.info(
            "broad_results_fallback: session=%s promoted %d excluded articles "
            "(shown set had %d, threshold %d)",
            session_id, len(_promote), len(included) - len(_promote), _min_included,
        )

    body = _to_response({**doc, "_session_diagnostics_json": session_diagnostics}).model_dump()
    body["queries"] = queries
    body["articles"] = included
    body["excluded_articles"] = excluded
    # Signals the UI to show a "no exact matches — showing broader results"
    # banner instead of an empty state, and to render these cards with a
    # lower-confidence treatment.
    body["broad_fallback_applied"] = broad_fallback_applied
    # Phase 5F — surface the context-engine outputs on the GET response
    # so the transparency panel and the testing agent can see them.
    # These keys are not part of the legacy SearchSessionResponse model;
    # they're appended on top of model_dump().
    for key in (
        "shorthand_substitutions_json",
        "ontology_mappings_json",
        "ontology_lifts_json",
        "role_repairs_json",
        "role_weights_json",
        "multi_intent_json",
        # Phase 6B.2 — surface the new diagnostics blocks for the
        # transparency panel + admin views.
        "default_selection_summary",
        "evidence_packet_summary",
        # Phase 6C — closed-set per-session expansion metadata. PHI-
        # blocked sessions surface ``active_expansion_reason="phi_blocked"``
        # so the cap-dominance test + admin diagnostics can verify the
        # expansion path was never invoked.
        "expansion_mode_used",
        "active_expansion_applied",
        "active_expansion_candidate_count",
        "active_expansion_latency_ms",
        "active_expansion_reason",
        "scholar_like_expansion_json",
        # Phase 6F — primary clinical entity anchors + generic
        # modifiers + anchor_failure_reason (closed-set). Surfaced for
        # the transparency panel so the UI never shows a blank
        # Background/primary_care interpretation when a drug or
        # disease anchor is present.
        "primary_clinical_anchors_json",
    ):
        body[key] = doc.get(key)

    # Phase 6B.2.1 — Rebuild ``default_selection_summary`` if it's stale.
    # A session indexed BEFORE the 6B.2.1 fix could carry a summary that
    # references article_ids no longer present in ``articles[]`` (the
    # legacy ``evidence_ranker`` set ``exclusion_reason="low_relevance"``
    # on those rows before the 6B.2 recalibrated default-selection
    # cleared the flag). Drop entries whose article_ids are not in the
    # currently-included articles[] list and adjust ``selected_count``
    # to match. Do NOT re-run selection — just reconcile against the
    # source of truth for what the GET response actually exposes.
    ds = body.get("default_selection_summary") or {}
    if isinstance(ds, dict) and ds.get("reason_each_selected"):
        included_ids = {r.get("article_id") for r in included}
        kept = [
            entry for entry in (ds.get("reason_each_selected") or [])
            if entry.get("article_id") in included_ids
        ]
        if len(kept) != len(ds.get("reason_each_selected") or []):
            rebuilt = dict(ds)
            rebuilt["reason_each_selected"] = kept
            rebuilt["selected_count"] = len(kept)
            lane = dict(rebuilt.get("selected_by_lane") or {})
            lane["primary"] = len(kept)
            rebuilt["selected_by_lane"] = lane
            rebuilt["selected_article_ids"] = [
                e.get("article_id") for e in kept if e.get("article_id")
            ]
            rebuilt["stale_summary_reconciled"] = True
            body["default_selection_summary"] = rebuilt

    # Phase 5D.1 — build the canonical interpreted-question display JSON
    # on read (BC for pre-Phase-5D.1 sessions). Cache the result on the
    # session doc after first compute so subsequent GETs skip the build.
    # Phase 6B.3.1 — also self-heal sessions cached with the broken
    # ``"Other (<subtype>)"`` legacy format. A truthy cached_display
    # whose ``type_of_question`` matches that pattern is invalidated
    # and rebuilt with the new canonical-cascade serializer.
    # Phase M4.1.6 — when the bounded-LLM M4 active repair was APPLIED,
    # the clinician PICOTT card must reflect the M4 structured
    # interpretation, not the degraded baseline. ``None`` for legacy /
    # non-applied / production (hard-off) sessions.
    _m4_candidate = _m4_applied_candidate(doc)

    # Phase 5D.1 — build the canonical interpreted-question display JSON
    cached_display = doc.get("interpreted_question_display_json")
    _stale_other_paren = False
    _stale_schema_version = False
    if cached_display:
        _cached_toq = (cached_display.get("type_of_question") or "")
        if _cached_toq.startswith("Other (") and _cached_toq.endswith(")"):
            _stale_other_paren = True
        # Rebuild whenever the display SHAPE has moved on. Without this
        # the only invalidation is the narrow ``Other (...)`` heal
        # above, so every session created before a schema change keeps
        # serving the old card indefinitely.
        if int(cached_display.get("display_schema_version") or 1) != int(
            DISPLAY_SCHEMA_VERSION
        ):
            _stale_schema_version = True
    if cached_display and not _stale_other_paren and not _stale_schema_version:
        body["interpreted_question_display_json"] = cached_display
    else:
        try:
            display = build_interpreted_question_display(doc)
            body["interpreted_question_display_json"] = display
            await db[Collections.SEARCH_SESSIONS].update_one(
                {"id": session_id},
                {"$set": {
                    "interpreted_question_display_json": display,
                    "updated_at": datetime.now(timezone.utc),
                }},
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "interpreted_question_display build failed (non-fatal); "
                "falling back to legacy pico_json on response",
            )
            body["interpreted_question_display_json"] = None

    # Phase 6G — surface the canonical PICOTT term mapping payload.
    # Prefer the finalize-persisted version; if absent (legacy session
    # pre-Phase-6G), build it fresh from the persisted artifacts.
    persisted_mapping = doc.get("picott_term_mapping_json")
    if isinstance(persisted_mapping, dict) and persisted_mapping:
        body["picott_term_mapping_json"] = persisted_mapping
        mapping_for_heal = None
        try:
            from models.picott_term_mapping import PicottTermMapping
            mapping_for_heal = PicottTermMapping(**persisted_mapping)
        except Exception:  # noqa: BLE001
            mapping_for_heal = None
    else:
        try:
            mapping_obj = build_picott_term_mapping(
                doc, queries=queries,
            )
            mapping_payload = mapping_obj.model_dump(mode="json")
            body["picott_term_mapping_json"] = mapping_payload
            await db[Collections.SEARCH_SESSIONS].update_one(
                {"id": session_id},
                {"$set": {
                    "picott_term_mapping_json": mapping_payload,
                    "updated_at": datetime.now(timezone.utc),
                }},
            )
            mapping_for_heal = mapping_obj
        except Exception:  # noqa: BLE001
            logger.exception(
                "phase6g picott_term_mapping build failed (non-fatal) "
                "session=%s", session_id,
            )
            body["picott_term_mapping_json"] = None
            mapping_for_heal = None

    # Phase M4.1.6 — M4 read-side projection for the PICOTT term-mapping
    # card. When the bounded-LLM M4 active repair was APPLIED, rebuild
    # the mapping from the validated M4 candidate so the clinician
    # "PICOTT TERM MAPPING" card shows the M4 structured interpretation
    # instead of the degraded baseline. Specialty / scope / query_blocks
    # are preserved from the baseline mapping. Display-only; production
    # keeps M4 hard-off so this never runs there.
    if _m4_candidate is not None:
        try:
            base_mapping = body.get("picott_term_mapping_json") or {}
            m4_mapping_obj = build_m4_picott_term_mapping(
                _m4_candidate, base_mapping, queries=queries,
            )
            m4_mapping_payload = m4_mapping_obj.model_dump(mode="json")
            body["picott_term_mapping_json"] = m4_mapping_payload
            mapping_for_heal = m4_mapping_obj
            if m4_mapping_payload != doc.get("picott_term_mapping_json"):
                await db[Collections.SEARCH_SESSIONS].update_one(
                    {"id": session_id},
                    {"$set": {
                        "picott_term_mapping_json": m4_mapping_payload,
                        "updated_at": datetime.now(timezone.utc),
                    }},
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseM4_1_6 M4 picott_term_mapping projection failed "
                "(non-fatal) session=%s", session_id,
            )

    # Phase 6G Task 5 — read-time display self-heal. When the persisted
    # ``interpreted_question_display_json`` still carries the legacy
    # "primary_care" / "Background" labels (or has blank PICO rows
    # while the new mapping has filled buckets), regenerate the display
    # from the mapping. Emits a closed-set audit event with COUNTS
    # ONLY — never the question text, term text, or rationale text.
    if mapping_for_heal is not None and _m4_candidate is None:
        current_display = body.get("interpreted_question_display_json")
        if needs_display_self_heal(current_display, mapping_for_heal):
            try:
                healed = derive_display_from_mapping(
                    current_display, mapping_for_heal,
                )
                body["interpreted_question_display_json"] = healed
                await db[Collections.SEARCH_SESSIONS].update_one(
                    {"id": session_id},
                    {"$set": {
                        "interpreted_question_display_json": healed,
                        "updated_at": datetime.now(timezone.utc),
                    }},
                )
                fields_filled = sum(
                    1 for k in (
                        "population",
                        "intervention_or_exposure_or_index_test",
                        "outcomes_or_target_diagnosis",
                    ) if isinstance(healed.get(k), list)
                    and healed.get(k)
                    and healed.get(k) != ["Not specified"]
                )
                generic_only = bool(
                    mapping_for_heal.completeness.clarification_needed
                )
                try:
                    await audit_service.log(
                        "picott_display_self_healed",
                        user_id=current_user.get("id"),
                        search_session_id=session_id,
                        payload={
                            "session_id": session_id,
                            "fields_filled_count": int(fields_filled),
                            "generic_only_branch": generic_only,
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "phase6g picott_display_self_healed audit failed "
                        "(non-fatal) session=%s", session_id,
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "phase6g display self-heal failed (non-fatal) "
                    "session=%s", session_id,
                )

    # Phase M4.1.6 — FINAL read-side override. When the bounded-LLM M4
    # active repair was APPLIED, project the validated M4 candidate onto
    # the clinician PICOTT card so it reflects the M4 structured
    # interpretation instead of the degraded baseline. This runs LAST so
    # neither the legacy build nor the Phase-6G self-heal can clobber it.
    # Display-only: retrieval / ranking / default-selection are untouched.
    # Production keeps M4 hard-off, so ``_m4_candidate`` is always None
    # there and this block never executes.
    if _m4_candidate is not None:
        try:
            base_display = body.get("interpreted_question_display_json") or {}
            m4_display = build_m4_interpreted_display(
                _m4_candidate, base_display,
            )
            body["interpreted_question_display_json"] = m4_display
            # Cache only when the M4 projection differs from what is
            # already persisted (avoids write churn on repeat GETs).
            if m4_display != doc.get("interpreted_question_display_json"):
                await db[Collections.SEARCH_SESSIONS].update_one(
                    {"id": session_id},
                    {"$set": {
                        "interpreted_question_display_json": m4_display,
                        "updated_at": datetime.now(timezone.utc),
                    }},
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseM4_1_6 M4 display projection failed (non-fatal) "
                "session=%s", session_id,
            )

    return body


@router.get("/{session_id}/official-guideline-sources")
async def official_guideline_sources(
    session_id: str, current_user: dict = Depends(get_current_user),
):
    """Phase K5.2A — on-demand official-source metadata/link-out for the
    session's topic, via the no-key HHS/ODPHP MyHealthfinder connector.

    Metadata + link-out ONLY. NO full text. NO AI ingestion. Distinct from
    the K5.1 PubMed-indexed guideline lane. Owner-scoped. PHI-blocked
    sessions never reach the connector.
    """
    from services.connectors.myhealthfinder import (
        MYHF_ORGANIZATION,
        MYHF_PROVENANCE,
        build_keyword,
        myhealthfinder_connector,
    )

    db = get_db()
    doc = await _owned_session(session_id, current_user["id"])

    base = {
        "session_id": session_id,
        "lane_kind": "official_source_metadata_link_out",
        "lane_label": "Official source metadata / link-out (HHS/ODPHP MyHealthfinder)",
        "lane_tooltip": (
            "Official HHS/ODPHP prevention/screening recommendation topics — "
            "metadata + link-out only. Distinct from the PubMed-indexed "
            "guideline lane. Not full text; not AI-ingestible."
        ),
        "provenance": MYHF_PROVENANCE,
        "organization": MYHF_ORGANIZATION,
        "ai_ingestion_allowed": False,
        "results": [],
        "status": "skipped",
    }

    # PHI-blocked sessions never reach the connector.
    if doc.get("status") == "blocked_phi":
        base["status"] = "skipped_phi_blocked"
        return base

    # Defensive registry gate — only call when the source is enabled +
    # retrieval-allowed in the guideline registry.
    reg = await db[Collections.GUIDELINE_SOURCES].find_one(
        {"source_name": "myhealthfinder_hhs"},
        {"_id": 0, "enabled": 1, "retrieval_allowed": 1},
    )
    if not reg or not (reg.get("enabled") and reg.get("retrieval_allowed")):
        base["status"] = "source_not_enabled"
        return base

    anchors = (doc.get("primary_clinical_anchors_json") or {}).get("primary_anchors") or []
    primary_anchor = None
    for a in anchors:
        if isinstance(a, dict) and a.get("canonical_concept"):
            primary_anchor = str(a.get("canonical_concept"))
            break
    keyword = build_keyword(
        primary_anchor=primary_anchor,
        question=doc.get("sanitized_question"),
    )
    conn = await myhealthfinder_connector.search(keyword)
    base["status"] = conn.get("status")
    base["keyword"] = conn.get("keyword")
    base["results"] = conn.get("results") or []
    return base


@router.get("/{session_id}/status")
async def session_status(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    doc = await _owned_session(session_id, current_user["id"])
    status = doc["status"]
    counts = {"retrieved": 0, "deduplicated": 0, "excluded": 0}
    articles_ready = status in {"awaiting_selection", "complete"}
    if articles_ready:
        retrieved = await db[Collections.SEARCH_SESSION_ARTICLES].count_documents(
            {"search_session_id": session_id}
        )
        excluded = await db[Collections.SEARCH_SESSION_ARTICLES].count_documents(
            {"search_session_id": session_id, "exclusion_reason": {"$ne": None}}
        )
        counts = {
            "retrieved": retrieved,
            "deduplicated": retrieved,
            "excluded": excluded,
        }
    stage_map = {
        "draft": "queued",
        "blocked_phi": "blocked",
        "searching": "searching",
        "ranking": "ranking",
        "brokering_access": "brokering_access",
        "fetching_oa_full_text": "fetching_full_text",
        "awaiting_clinical_subject": "needs_clarification",
        "summarizing": "summarizing",
        "awaiting_selection": "ready",
        "complete": "ready",
        "error": "error",
    }
    # Monotonic progress across the ACTUAL pipeline phase order
    #   searching -> ranking -> brokering_access -> fetching_oa_full_text
    #   -> awaiting_selection.
    # Every SessionStatus is covered so the percentage never regresses (the old
    # map returned 0 for brokering_access/fetching_oa_full_text, which made the
    # bar jump 60% -> 0% mid-run and the client poll-loop never finish). All
    # terminal states report 100 so the client's "poll until 100%" loop ends.
    progress_map = {
        "draft": 5,
        "searching": 25,
        "ranking": 45,
        "brokering_access": 65,
        "fetching_oa_full_text": 85,
        "summarizing": 92,
        "awaiting_selection": 100,
        "complete": 100,
        "awaiting_clinical_subject": 100,  # terminal clarification — stop polling
        "blocked_phi": 100,                # terminal — stop polling
        "error": 100,                      # terminal — stop polling
    }
    terminal = status in {
        "awaiting_selection", "complete", "error",
        "blocked_phi", "awaiting_clinical_subject",
    }
    return {
        "status": status,
        "stage": stage_map.get(status, status),
        "progress": progress_map.get(status, 0),
        "done": terminal,
        "parsed": doc.get("pico_json") or None,
        "evaluator": doc.get("evaluator_json"),
        "counts": counts,
        "articles_ready": articles_ready,
        "error_message": doc.get("error_message"),
    }


@router.get("/{session_id}/transparency")
async def transparency(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    doc = await _owned_session(session_id, current_user["id"])
    queries = await (
        db[Collections.SEARCH_QUERIES]
        .find({"search_session_id": session_id}, {"_id": 0})
        .sort("created_at", 1)
        .to_list(length=50)
    )
    counts = {
        "retrieved": await db[Collections.SEARCH_SESSION_ARTICLES].count_documents({"search_session_id": session_id}),
        "excluded": await db[Collections.SEARCH_SESSION_ARTICLES].count_documents({"search_session_id": session_id, "exclusion_reason": {"$ne": None}}),
    }
    # Sources actually searched in this session — derived from search_queries.
    sources_searched_set = sorted({(q.get("source") or "europe_pmc") for q in queries})
    evaluator = doc.get("evaluator_json")
    source_coverage = (evaluator or {}).get("source_coverage") or {
        "sources_searched": sources_searched_set,
        "per_source_raw_counts": {},
        "sources_with_evidence": [],
        "cross_source_confirmed_articles": 0,
    }
    return {
        "parsed": doc.get("pico_json") or {},
        "queries": queries,
        "sources_searched": sources_searched_set,
        "source_coverage": source_coverage,
        "retrieval_evaluator": evaluator,
        "openalex_enrichment": (evaluator or {}).get("openalex_enrichment"),
        "access_broker": (evaluator or {}).get("access_broker"),
        "counts": counts,
        "ranking_criteria_description": (
            "Composite rank = 0.55·relevance + 0.35·evidence-design + "
            "0.05·recency + 0.05·full-text, with a +3/+5 boost for "
            "cross-source confirmation. Buckets: high≥80, moderate 60–79, "
            "low excluded. Defaults: best guideline, best SR/MA, best primary "
            "study appropriate to the question type are pre-selected."
        ),
    }


@router.post("/{session_id}/refresh-access")
async def refresh_access(
    session_id: str, current_user: dict = Depends(get_current_user)
):
    """Re-run the access broker ONLY on the existing articles in this session.

    Does NOT rerun parser/planner/retrieval/dedup/safety/ranker. The broker
    rebuilds article_access_locations rows so any newly-configured
    institutional settings or refreshed OA status become visible. Returns
    {updated_articles, new_access_locations, removed_access_locations}.
    """
    db = get_db()
    doc = await _owned_session(session_id, current_user["id"])
    if doc.get("status") in ("blocked_phi", "searching", "ranking"):
        raise HTTPException(
            status_code=409,
            detail="Refresh access is only available once the session has results.",
        )

    summary = await full_text_access_broker.broker_for_session(
        session_id, replace_existing=True
    )
    # Update the session's evaluator_json.access_broker block so transparency
    # reflects the refreshed state.
    evaluator = doc.get("evaluator_json") or {}
    evaluator["access_broker"] = summary
    await db[Collections.SEARCH_SESSIONS].update_one(
        {"id": session_id},
        {"$set": {"evaluator_json": evaluator, "updated_at": datetime.now(timezone.utc)}},
    )
    await audit_service.log(
        "access_broker_refresh",
        user_id=current_user["id"],
        search_session_id=session_id,
    )
    return {
        "status": "ok",
        "updated_articles": summary.get("articles_processed", 0),
        "new_access_locations": summary.get("new_access_locations", 0),
        "removed_access_locations": summary.get("removed_access_locations", 0),
    }


class ArticleSelectionPatch(BaseModel):
    user_included: Optional[bool] = None
    exclusion_reason: Optional[str] = None


@router.patch("/{session_id}/articles/{article_identifier}")
async def patch_session_article(
    session_id: str,
    article_identifier: str,
    body: ArticleSelectionPatch,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()

    # Ownership check — admin bypass allowed but uniformly 404 on missing.
    session = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "deleted_at": None}, {"_id": 0}
    )
    if not session:
        raise HTTPException(status_code=404, detail="Article not found in this search session.")
    is_admin = current_user.get("role") == "admin"
    if session["user_id"] != current_user["id"] and not is_admin:
        # Uniform 404 to avoid leaking existence to non-owners.
        raise HTTPException(status_code=404, detail="Article not found in this search session.")

    # Resolve identifier: first as search_session_article.id, then as article_id.
    row = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"search_session_id": session_id, "id": article_identifier}, {"_id": 0}
    )
    if row is None:
        row = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
            {"search_session_id": session_id, "article_id": article_identifier}, {"_id": 0}
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Article not found in this search session.")

    update: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
    fields_set = body.model_fields_set
    if "user_included" in fields_set:
        update["user_included"] = bool(body.user_included) if body.user_included is not None else False
    if "exclusion_reason" in fields_set:
        # Explicit-null clears the prior value; omitted key leaves it untouched.
        update["exclusion_reason"] = body.exclusion_reason if body.exclusion_reason else None
    if len(update) == 1:  # only updated_at
        raise HTTPException(status_code=400, detail="No fields to update.")

    await db[Collections.SEARCH_SESSION_ARTICLES].update_one(
        {"id": row["id"]},
        {"$set": update},
    )
    await audit_service.log(
        "session_article_updated",
        user_id=current_user["id"],
        search_session_id=session_id,
        payload={
            "article_id": row["article_id"],
            "user_included": update.get("user_included"),
            "exclusion_reason": update.get("exclusion_reason"),
            "admin_override": is_admin and session["user_id"] != current_user["id"],
        },
    )
    fresh = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"id": row["id"]}, {"_id": 0}
    )
    # Phase 5D.1 — guardrail: if the user is including an article that
    # the default-selection module rejected, surface the
    # ``requires_override_confirmation`` flag + the canonical warning
    # string. The client should call the ``include-override`` endpoint
    # for an explicit, audited acknowledgement.
    rejection = fresh.get("default_selection_rationale")
    is_capped_rejection = bool(
        rejection
        and isinstance(rejection, str)
        and (
            rejection.startswith("cap_reason_")
            or rejection in {
                "negative_concept_fired",
                "relevance_below_60",
                "excluded_retracted_or_preprint_or_unsafe",
            }
        )
    )
    response = {
        "search_session_article_id": fresh["id"],
        "article_id": fresh["article_id"],
        "user_included": fresh.get("user_included"),
        "exclusion_reason": fresh.get("exclusion_reason"),
        "status": "updated",
    }
    if fresh.get("user_included") and is_capped_rejection \
            and not fresh.get("user_override_included"):
        response["requires_override_confirmation"] = True
        response["override_warning"] = _OVERRIDE_WARNING
        response["default_selection_rationale"] = rejection
    return response


# Phase 5D.1 — include-override flow.
#
# When the user attempts to include an article that has a cap_reason in
# the closed-set exclusion list (the default-selection function rejected
# it), the regular PATCH endpoint above flags it via a guardrail and
# returns ``requires_override_confirmation=true``. The user can then
# POST to this endpoint with explicit ``{"override_acknowledged": true}``
# to persist ``user_override_included=true`` and the warning string.
class IncludeOverrideRequest(BaseModel):
    override_acknowledged: bool = False


_OVERRIDE_WARNING = (
    "This article is missing required concepts for the interpreted "
    "question. It may not directly answer the question."
)


@router.post("/{session_id}/articles/{article_identifier}/include-override")
async def include_article_with_override(
    session_id: str,
    article_identifier: str,
    body: IncludeOverrideRequest,
    current_user: dict = Depends(get_current_user),
):
    """Explicit user override to include a capped/excluded article in the
    evidence packet. Records ``user_override_included=true`` plus the
    closed-set warning string verbatim. The summarizer + diagnostics
    surface this caveat clearly."""
    if not body.override_acknowledged:
        raise HTTPException(
            status_code=400,
            detail=(
                "override_acknowledged=true is required to include a "
                "capped article. The default-selection module rejected "
                "this article — see ``default_selection_rationale`` on "
                "the article row."
            ),
        )
    db = get_db()
    session = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "deleted_at": None}, {"_id": 0},
    )
    if not session:
        raise HTTPException(status_code=404, detail="Article not found in this search session.")
    is_admin = current_user.get("role") == "admin"
    if session["user_id"] != current_user["id"] and not is_admin:
        raise HTTPException(status_code=404, detail="Article not found in this search session.")

    row = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"search_session_id": session_id, "id": article_identifier}, {"_id": 0},
    )
    if row is None:
        row = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
            {"search_session_id": session_id, "article_id": article_identifier},
            {"_id": 0},
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Article not found in this search session.")

    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSION_ARTICLES].update_one(
        {"id": row["id"]},
        {"$set": {
            "user_override_included": True,
            "user_included": True,
            "user_override_acknowledged_at": now,
            "user_override_warning": _OVERRIDE_WARNING,
            "exclusion_reason": None,
            "updated_at": now,
        }},
    )
    await audit_service.log(
        "session_article_include_override",
        user_id=current_user["id"],
        search_session_id=session_id,
        payload={
            "article_id": row["article_id"],
            "previous_cap_reasons": row.get("cap_reasons") or [],
            "previous_default_selection_rationale":
                row.get("default_selection_rationale"),
            "warning": _OVERRIDE_WARNING,
        },
    )
    return {
        "search_session_article_id": row["id"],
        "article_id": row["article_id"],
        "user_override_included": True,
        "user_override_warning": _OVERRIDE_WARNING,
        "status": "override_applied",
    }


@router.delete("/{session_id}", status_code=200)
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    now = datetime.now(timezone.utc)
    res = await db[Collections.SEARCH_SESSIONS].update_one(
        {"id": session_id, "user_id": current_user["id"], "deleted_at": None},
        {"$set": {"deleted_at": now, "updated_at": now}},
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Search session not found.")
    await audit_service.log(
        "search_session_deleted",
        user_id=current_user["id"],
        search_session_id=session_id,
    )
    return {"status": "ok"}
