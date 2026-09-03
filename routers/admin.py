"""Admin router — dashboard, cleanup trigger, source-config + audit listings."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.config import settings
from core.db import Collections, get_db
from core.security import require_admin
from models.source_configs import SourceConfigPatch
from services.audit import audit_service
from services.article_lane_classifier import (
    _FORBIDDEN_DIAG_SUBSTRINGS as _K3A_FORBIDDEN_DIAG_SUBSTRINGS,
)
from services.retention_cleanup import retention_cleanup
from services.search_benchmark import (
    BENCHMARK_CASES,
    search_benchmark_service,
)


async def _build_oa_fetcher_metrics(db) -> dict:
    """Phase 4A — small admin block. Inline rather than importing the router."""
    counts = {}
    for evt in (
        "oa_fetch_attempt", "oa_fetch_success", "oa_fetch_failed",
        "oa_fetch_cache_hit", "oa_fetch_skipped_ineligible",
        "oa_fetch_skipped_license",
    ):
        counts[evt] = await db[Collections.AUDIT_EVENTS].count_documents({"event_type": evt})
    skipped_nc = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "oa_fetch_skipped_license",
         "event_payload_json.error_code": "license_noncommercial"}
    )
    skipped_unk = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "oa_fetch_skipped_license",
         "event_payload_json.error_code": "license_unknown"}
    )
    skipped_nd = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "oa_fetch_skipped_license",
         "event_payload_json.error_code": "license_restricted"}
    )
    avg_latency: Optional[float] = None
    async for row in db[Collections.AUDIT_EVENTS].aggregate([
        {"$match": {"event_type": "oa_fetch_success"}},
        {"$group": {"_id": None, "avg": {"$avg": "$event_payload_json.latency_ms"}}},
    ]):
        if row.get("avg") is not None:
            avg_latency = round(float(row["avg"]), 1)
    epmc = await db[Collections.SOURCE_CONFIGS].find_one(
        {"source_name": "europe_pmc_oa_fetcher"}, {"_id": 0}
    )
    pmc = await db[Collections.SOURCE_CONFIGS].find_one(
        {"source_name": "pmc_oa_fetcher"}, {"_id": 0}
    )

    def _stat(row):
        if not row:
            return "skipped"
        if row.get("last_error_at") and (
            not row.get("last_success_at") or row["last_error_at"] > row["last_success_at"]
        ):
            return "error"
        return "ok" if row.get("last_success_at") else "skipped"

    upgraded = len(await db[Collections.ARTICLE_TEXT_EXTRACTS].distinct(
        "article_id", {"text_source": "oa_full_text"}
    ))
    return {
        "fetch_attempts": counts["oa_fetch_attempt"],
        "fetch_successes": counts["oa_fetch_success"],
        "fetch_failures": counts["oa_fetch_failed"],
        "fetch_cache_hits": counts["oa_fetch_cache_hit"],
        "skipped_ineligible": counts["oa_fetch_skipped_ineligible"],
        "skipped_noncommercial": skipped_nc,
        "skipped_unknown_license": skipped_unk,
        "skipped_no_derivatives": skipped_nd,
        "articles_upgraded_abstract_to_oa": upgraded,
        "average_fetch_latency_ms": avg_latency,
        "europe_pmc_oa_status": _stat(epmc),
        "pmc_oa_status": _stat(pmc),
    }

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


async def _build_job_metrics(db) -> dict:
    """Phase 4B — durable job metrics. Counts only, never raw payloads."""
    counts: dict[str, int] = {}
    for status_ in ("queued", "running", "succeeded", "failed", "canceled", "expired"):
        counts[status_] = await db[Collections.JOBS].count_documents({"status": status_})
    summary_timeouts = await db[Collections.JOBS].count_documents(
        {"job_type": "summary_generation", "error_code": "timeout"}
    )
    summary_verif_failed = await db[Collections.JOBS].count_documents(
        {"job_type": "summary_generation", "error_code": "verification_failed"}
    )
    summary_llm_unconfigured = await db[Collections.JOBS].count_documents(
        {"job_type": "summary_generation", "error_code": "llm_not_configured"}
    )
    stale = await db[Collections.JOBS].count_documents({"error_code": "stale_job"})
    runner_mode = (os.environ.get("JOB_RUNNER_MODE") or "in_process").strip().lower()
    return {
        "runner_mode": runner_mode,
        "in_process_warning": (
            "Job runner is in_process: jobs do not survive multiple backend "
            "replicas. Switch JOB_RUNNER_MODE before scaling out."
        ) if runner_mode == "in_process" else None,
        "counts": counts,
        "summary_generation_timeouts": summary_timeouts,
        "summary_generation_verification_failures": summary_verif_failed,
        "summary_generation_llm_unconfigured": summary_llm_unconfigured,
        "stale_jobs": stale,
    }


async def _build_malware_scan_metrics(db) -> dict:
    """Phase 4B — malware-scan honest status (mode/configured/uploads_allowed)
    plus aggregate scan outcome counts from audit events."""
    from services.malware_scan import malware_scan_service
    status = malware_scan_service.status()
    allowed, code, msg = malware_scan_service.uploads_allowed()

    counts: dict[str, int] = {}
    for result in ("clean", "infected", "suspicious", "scanner_unavailable",
                   "scan_error", "disabled"):
        counts[result] = await db[Collections.AUDIT_EVENTS].count_documents(
            {"event_type": "upload_malware_scan",
             "event_payload_json.result": result}
        )
    uploads_blocked_malware = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].count_documents(
        {"processing_error_code": "malware_detected"}
    )
    return {
        **status,
        "uploads_allowed": allowed,
        "uploads_allowed_reason": code,
        "uploads_allowed_message": msg,
        "scan_result_counts": counts,
        "uploads_blocked_malware": uploads_blocked_malware,
    }



@router.get("/dashboard")
async def dashboard():
    db = get_db()
    source_configs_total = await db[Collections.SOURCE_CONFIGS].count_documents({})
    source_configs_enabled = await db[Collections.SOURCE_CONFIGS].count_documents({"enabled": True})
    phi_blocked_submissions = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "phi_blocked"}
    )
    source_query_errors_24h = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "source_query_error"}
    )
    upload_total = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].count_documents({})
    upload_pending = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].count_documents(
        {"upload_status": "pending_review"}
    )
    upload_processed = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].count_documents(
        {"upload_status": "processed"}
    )
    upload_blocked_phi = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].count_documents(
        {"upload_status": "blocked_phi"}
    )
    upload_mismatch = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].count_documents(
        {"upload_status": "mismatch"}
    )
    upload_deleted = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].count_documents(
        {"$or": [{"upload_status": "deleted"}, {"deleted_at": {"$ne": None}}]}
    )

    # Phase 3 summary metrics.
    from services.llm_provider import llm_provider as _llm
    summaries_requested = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "summary_requested"}
    )
    summaries_verified = await db[Collections.SUMMARIES].count_documents(
        {"verification_status": "passed"}
    )
    summaries_failed = await db[Collections.SUMMARIES].count_documents(
        {"verification_status": "failed"}
    )
    # average_verification_attempts + average_selected_articles_per_summary
    pipeline_attempts = [
        {"$group": {
            "_id": None,
            "avg_attempts": {"$avg": "$verification_attempts"},
            "avg_selected": {"$avg": {"$size": {"$ifNull": ["$selected_article_ids_json", []]}}},
            "abstract_only": {"$sum": {"$ifNull": ["$evidence_packet_summary_json.abstract_only_count", 0]}},
            "oa_full_text": {"$sum": {"$ifNull": ["$evidence_packet_summary_json.oa_full_text_count", 0]}},
            "user_upload": {"$sum": {"$ifNull": ["$evidence_packet_summary_json.user_uploaded_full_text_count", 0]}},
        }},
    ]
    avg_attempts = None
    avg_selected = None
    abstract_only_summary_count = 0
    oa_summary_count = 0
    user_uploaded_summary_count = 0
    async for row in db[Collections.SUMMARIES].aggregate(pipeline_attempts):
        avg_attempts = round(row.get("avg_attempts") or 0, 2)
        avg_selected = round(row.get("avg_selected") or 0, 2)
        abstract_only_summary_count = int(row.get("abstract_only") or 0)
        oa_summary_count = int(row.get("oa_full_text") or 0)
        user_uploaded_summary_count = int(row.get("user_upload") or 0)

    summarization_errors_24h = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "summary_failed"}
    )
    citation_verification_failures_24h = await db[Collections.SUMMARIES].count_documents(
        {"verification_status": "failed"}
    )

    # Storage estimate (excluding deleted bytes).
    pipeline_storage = [
        {"$match": {"storage_deleted_at": None, "deleted_at": None}},
        {"$group": {"_id": None, "total": {"$sum": "$file_size"}}},
    ]
    storage_bytes = 0
    async for row in db[Collections.UPLOADED_FULL_TEXT_ASSETS].aggregate(pipeline_storage):
        storage_bytes = int(row.get("total") or 0)
    total_non_error = upload_total - await db[Collections.UPLOADED_FULL_TEXT_ASSETS].count_documents(
        {"upload_status": "error"}
    )
    avg_success_pct = (
        round(100 * upload_processed / total_non_error, 1) if total_non_error else None
    )
    recent_audit_events = (
        await db[Collections.AUDIT_EVENTS]
        .find({}, {"_id": 0})
        .sort("created_at", -1)
        .limit(20)
        .to_list(length=20)
    )
    # Per-source health snapshot (no API keys ever leak).
    source_health = await (
        db[Collections.SOURCE_CONFIGS]
        .find(
            {},
            {
                "_id": 0,
                "source_name": 1,
                "enabled": 1,
                "requires_api_key": 1,
                "status_message": 1,
                "last_success_at": 1,
                "last_error_at": 1,
                "last_error_message": 1,
            },
        )
        .sort("source_name", 1)
        .to_list(length=200)
    )
    return {
        "source_configs": {
            "total": source_configs_total,
            "enabled": source_configs_enabled,
            "disabled": source_configs_total - source_configs_enabled,
        },
        "source_health": source_health,
        "api_health": {"status": "ok"},
        "usage_metrics": {
            "search_sessions_total": await db[Collections.SEARCH_SESSIONS].count_documents({}),
            "users_total": await db[Collections.USERS].count_documents({}),
        },
        "phi_blocked_submissions": phi_blocked_submissions,
        "source_query_errors": source_query_errors_24h,
        "search_latency_p50_ms": {"placeholder": True, "value": None},
        "summarization_latency_p50_ms": {"placeholder": True, "value": None},
        "citation_verification_failures": citation_verification_failures_24h,
        "upload_metrics": {
            "total": upload_total,
            "pending_review": upload_pending,
            "processed": upload_processed,
            "blocked_phi": upload_blocked_phi,
            "mismatch": upload_mismatch,
            "deleted": upload_deleted,
            "avg_extraction_success_pct": avg_success_pct,
            "storage_bytes_estimate": storage_bytes,
        },
        "summary_metrics": {
            "summaries_requested": summaries_requested,
            "summaries_verified": summaries_verified,
            "summaries_failed": summaries_failed,
            "average_verification_attempts": avg_attempts,
            "average_selected_articles_per_summary": avg_selected,
            "abstract_only_summary_count": abstract_only_summary_count,
            "user_uploaded_full_text_summary_count": user_uploaded_summary_count,
            "oa_full_text_summary_count": oa_summary_count,
            "llm_provider_configured": _llm.is_configured(),
            "llm_provider": _llm.provider,
            "summary_model": _llm.summary_model,
            "verifier_model": _llm.verifier_model,
            "summarization_errors_24h": summarization_errors_24h,
            "citation_verification_failures_24h": citation_verification_failures_24h,
        },
        "oa_fetcher_metrics": await _build_oa_fetcher_metrics(db),
        "job_metrics": await _build_job_metrics(db),
        "malware_scan_metrics": await _build_malware_scan_metrics(db),
        "upload_feature_enabled": (os.environ.get("ENABLE_USER_PDF_UPLOADS", "true").lower() != "false"),
        "max_upload_mb": int(os.environ.get("MAX_UPLOAD_MB", "25") or "25"),
        "pdf_malware_scan_enabled": (os.environ.get("PDF_MALWARE_SCAN_ENABLED", "false").lower() == "true"),
        "pdf_malware_scan_status": (
            "configured" if os.environ.get("PDF_MALWARE_SCAN_ENABLED", "false").lower() == "true"
            else "not_configured"
        ),
        "recent_audit_events": recent_audit_events,
    }


@router.post("/cleanup")
async def trigger_cleanup(current_admin: dict = Depends(require_admin)):
    result = await retention_cleanup.cleanup_expired_sessions()
    await audit_service.log(
        "admin_cleanup_triggered",
        user_id=current_admin["id"],
        payload=result,
    )
    return result


# ---- Phase K3A.1 — Cap-Observability summary ----
@router.get("/cap-observability")
async def cap_observability_summary(
    window_days: int = Query(7, ge=1, le=30),
    current_admin: dict = Depends(require_admin),  # noqa: ARG001
):
    """Closed-set, admin-only aggregation of shadow lane + cap-
    candidate diagnostics across recent (≤30 day) non-PHI sessions.
    NEVER emits PHI, raw question text, user emails/ids, raw prompts,
    completions, or evidence text — only closed-set histogram keys +
    session_id-only top-3 lists.

    NOTE on route naming: the original spec proposed
    ``/api/admin/cap-observability/summary``; that suffix collides
    with an existing test teardown that strips any path ending in
    ``/summary``. Using the bare resource path is also cleaner — the
    aggregate IS the resource. See K3A.1 report for the choice.
    """
    from services.cap_observability import build_summary

    db = get_db()
    summary = await build_summary(db, window_days=window_days)
    return summary


# ---- Phase K3B Wave 1 Step 2.1 — Audit-Trail Export Endpoint ----
@router.get("/k3b-wave1-active-exclusions/export")
async def k3b_wave1_active_exclusions_export(
    window_days: int = Query(7, ge=1, le=30),
    format: str = Query("json"),
    current_admin: dict = Depends(require_admin),  # noqa: ARG001
):
    """Closed-set audit-trail export of every actively-excluded
    article in the past ``window_days`` (cap 30). Admin-only.

    Every record carries ONLY the closed-set fields defined in the
    Phase K3B Wave 1 Step 2.1 spec; never question text, never user
    identifiers / emails / raw prompts / completions / evidence text.
    ``blocked_phi`` sessions are excluded at query time AND
    defensively re-checked in the loop.

    Format:
      * ``format=json`` → ``[{...}, ...]`` array (default).
      * ``format=csv``  → ``text/csv`` with a header row,
        RFC 4180-escaped values.
    """
    from datetime import datetime, timedelta, timezone

    if format not in ("json", "csv"):
        raise HTTPException(
            status_code=422,
            detail="format must be 'json' or 'csv'",
        )
    db = get_db()
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=int(window_days))

    # Source: ``audit_events`` of type ``k3b_wave1_active_exclusion``
    # filtered to the window. Joined to ``articles`` for closed-set
    # title / journal / publication_year. Joined to
    # ``search_sessions`` for ``status`` (blocked_phi filter) +
    # ``provenance`` + ``clinical_question_map_json.primary_intent``.
    audit_cursor = db["audit_events"].find(
        {
            "event_type": "k3b_wave1_active_exclusion",
            "created_at": {"$gte": window_start},
        },
        {
            "_id": 0,
            "search_session_id": 1,
            "event_payload_json": 1,
            "created_at": 1,
        },
    ).sort("created_at", -1)

    # Pre-collect session ids and article ids so we can batch the
    # ``articles`` / ``search_sessions`` reads — cheap on the
    # small Wave-1 corpus.
    raw_rows: list[dict[str, Any]] = []
    session_ids: set[str] = set()
    article_ids: set[str] = set()
    async for ae in audit_cursor:
        sid = str(ae.get("search_session_id") or "")
        payload = ae.get("event_payload_json") or {}
        aid = str(payload.get("article_id") or "")
        if not sid or not aid:
            continue
        session_ids.add(sid)
        article_ids.add(aid)
        raw_rows.append({
            "session_id": sid,
            "article_id": aid,
            "created_at": ae.get("created_at"),
            "payload": payload,
        })

    # Batched ``search_sessions`` lookup with the closed-set
    # projection. ``status`` lets us re-check for blocked_phi.
    sessions_map: dict[str, dict[str, Any]] = {}
    if session_ids:
        async for sdoc in db["search_sessions"].find(
            {"id": {"$in": list(session_ids)}, "deleted_at": None},
            {
                "_id": 0,
                "id": 1,
                "status": 1,
                "provenance": 1,
                "clinical_question_map_json.primary_intent": 1,
            },
        ):
            sessions_map[str(sdoc.get("id"))] = sdoc

    # Batched ``articles`` lookup with the closed-set projection.
    articles_map: dict[str, dict[str, Any]] = {}
    if article_ids:
        async for adoc in db["articles"].find(
            {"id": {"$in": list(article_ids)}},
            {
                "_id": 0,
                "id": 1,
                "pmid": 1,
                "doi": 1,
                "title": 1,
                "journal": 1,
                "publication_year": 1,
            },
        ):
            articles_map[str(adoc.get("id"))] = adoc

    # Batched ``search_session_articles`` lookup so we can surface
    # the SSA id alongside (nullable per spec).
    ssa_map: dict[tuple[str, str], str] = {}
    if session_ids and article_ids:
        async for ssa in db["search_session_articles"].find(
            {
                "search_session_id": {"$in": list(session_ids)},
                "article_id": {"$in": list(article_ids)},
            },
            {
                "_id": 0,
                "id": 1,
                "search_session_id": 1,
                "article_id": 1,
            },
        ):
            ssa_map[(
                str(ssa.get("search_session_id")),
                str(ssa.get("article_id")),
            )] = str(ssa.get("id"))

    # Build the closed-set records.
    import hashlib
    records: list[dict[str, Any]] = []
    for r in raw_rows:
        sid = r["session_id"]
        aid = r["article_id"]
        payload = r["payload"]
        session = sessions_map.get(sid) or {}
        # Defense-in-depth: skip if the session is missing or blocked_phi.
        if not session:
            continue
        if str(session.get("status")) == "blocked_phi":
            continue
        article = articles_map.get(aid) or {}
        cqm = session.get("clinical_question_map_json") or {}
        primary_intent = (
            cqm.get("primary_intent") if isinstance(cqm, dict) else None
        )
        provenance = str(session.get("provenance") or "organic").lower()
        if provenance not in ("organic", "seeded"):
            provenance = "organic"
        ssa_id = ssa_map.get((sid, aid))
        # Deterministic exclusion_id for downstream dedup.
        exclusion_id = hashlib.sha256(
            f"{sid}|{aid}|k3b_wave1_active_exclusion".encode("utf-8"),
        ).hexdigest()
        rec = {
            "exclusion_id": exclusion_id,
            "session_id": sid,
            "article_id": aid,
            "search_session_article_id": ssa_id,
            "pmid": (
                str(payload.get("pmid"))
                if payload.get("pmid") else (
                    str(article.get("pmid"))
                    if article.get("pmid") else None
                )
            ),
            "doi": (
                str(payload.get("doi"))
                if payload.get("doi") else (
                    str(article.get("doi"))
                    if article.get("doi") else None
                )
            ),
            "title": str(article.get("title") or "")[:200],
            "journal": str(article.get("journal") or ""),
            "publication_year": article.get("publication_year"),
            "cap_reason_codes": [
                str(c) for c in (
                    payload.get("cap_reason_codes") or []
                ) if isinstance(c, str)
            ],
            "source_signal": str(
                payload.get("source_signal")
                or "pubmed_retracted_publication_type"
            ),
            "enforcement_explanation": [
                str(x) for x in (
                    payload.get("enforcement_explanation") or []
                ) if isinstance(x, str)
            ][:64],
            "article_join_status": str(
                payload.get("article_join_status") or "stable"
            ),
            "enforcement_mode": str(
                payload.get("enforcement_mode") or "active"
            ),
            "exclusion_timestamp": (
                r["created_at"].isoformat()
                if hasattr(r["created_at"], "isoformat")
                else str(r["created_at"])
            ),
            "primary_intent": (
                str(primary_intent) if primary_intent else None
            ),
            "branch_intent": (
                str(primary_intent) if primary_intent else None
            ),
            "provenance": provenance,
        }
        records.append(rec)

    # Leak-guard sweep on every string in every record.
    from services.article_lane_classifier import (
        _FORBIDDEN_DIAG_SUBSTRINGS as _LG,
    )

    def _safe(value: Any) -> Any:
        if isinstance(value, str):
            return value if not any(
                s in value for s in _LG
            ) else "[REDACTED]"
        if isinstance(value, list):
            return [_safe(v) for v in value]
        if isinstance(value, dict):
            return {k: _safe(v) for k, v in value.items()}
        return value

    records = [_safe(r) for r in records]

    if format == "json":
        return records

    # CSV path — RFC 4180-escaped.
    import csv as _csv
    import io as _io
    header = [
        "exclusion_id", "session_id", "article_id",
        "search_session_article_id", "pmid", "doi",
        "title", "journal", "publication_year",
        "cap_reason_codes", "source_signal",
        "enforcement_explanation", "article_join_status",
        "enforcement_mode", "exclusion_timestamp",
        "primary_intent", "branch_intent", "provenance",
    ]
    sio = _io.StringIO()
    w = _csv.writer(sio, quoting=_csv.QUOTE_MINIMAL)
    w.writerow(header)
    for r in records:
        w.writerow([
            r.get("exclusion_id", ""),
            r.get("session_id", ""),
            r.get("article_id", ""),
            r.get("search_session_article_id") or "",
            r.get("pmid") or "",
            r.get("doi") or "",
            r.get("title") or "",
            r.get("journal") or "",
            (
                "" if r.get("publication_year") is None
                else r.get("publication_year")
            ),
            "|".join(r.get("cap_reason_codes") or []),
            r.get("source_signal") or "",
            "|".join(r.get("enforcement_explanation") or []),
            r.get("article_join_status") or "",
            r.get("enforcement_mode") or "",
            r.get("exclusion_timestamp") or "",
            r.get("primary_intent") or "",
            r.get("branch_intent") or "",
            r.get("provenance") or "organic",
        ])
    from fastapi.responses import Response as _Response
    return _Response(
        content=sio.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                'attachment; '
                'filename="k3b_wave1_active_exclusions.csv"'
            ),
        },
    )


# ---- Phase K3B Wave 1 Step 2.1 — Deployment-fingerprint preflight ----
@router.get("/deployment-fingerprint")
async def deployment_fingerprint(
    current_admin: dict = Depends(require_admin),  # noqa: ARG001
):
    """Admin-only operator preflight tool. Closed-set response.
    NEVER emits raw secret values — exposes only the resolved state
    plus the closed-set boolean ``fingerprint_flag_set``. Safe to
    log, safe to share.
    """
    from core.config import settings
    from services.cap_enforcement_guard import (
        _resolved_cap_enforcement_mode,
    )

    is_production = bool(settings.IS_PRODUCTION)
    resolved_mode = _resolved_cap_enforcement_mode()
    cap_detection_mode = (
        os.environ.get("CLINICAL_ARTICLE_CAP_DETECTION_MODE", "shadow")
        or ""
    ).strip().lower()
    fp_primary = (
        os.environ.get("DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE", "")
        or ""
    ).strip().lower() == "true"
    fp_alias = (
        os.environ.get("ALLOW_PRODUCTION_CAP_ACTIVE", "")
        or ""
    ).strip().lower() == "true"
    fingerprint_flag_set = bool(fp_primary or fp_alias)
    if fp_primary:
        flag_source = "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE"
    elif fp_alias:
        flag_source = "ALLOW_PRODUCTION_CAP_ACTIVE"
    else:
        flag_source = "neither_set"
    allowlist_raw = (
        os.environ.get("ACTIVE_CAP_REASON_ALLOWLIST", "") or ""
    )
    active_cap_reason_allowlist = [
        s.strip().lower()
        for s in allowlist_raw.split(",")
        if s.strip()
    ]
    # Compute whether the startup guardrail WOULD permit ``active``
    # right now if the operator flipped the mode on this env.
    if is_production:
        if resolved_mode == "active":
            production_active_allowed = (
                fingerprint_flag_set
                and "retracted" in active_cap_reason_allowlist
            )
        else:
            # Mode is not currently active — compute the hypothetical
            # "would it be allowed if flipped?".
            production_active_allowed = (
                fingerprint_flag_set
                and "retracted" in active_cap_reason_allowlist
            )
    else:
        # Non-production — guardrail never fires.
        production_active_allowed = True

    if not is_production or resolved_mode != "active":
        guard_status = (
            "no_check_required_non_production_or_non_active"
        )
    elif production_active_allowed:
        guard_status = "permitted_to_run_active"
    else:
        guard_status = "would_fail_closed_on_startup"

    return {
        "deployment_environment":
            "production" if is_production else "development",
        "is_production": is_production,
        "cap_detection_mode": cap_detection_mode,
        "cap_enforcement_mode": resolved_mode,
        "fingerprint_flag_set": fingerprint_flag_set,
        "fingerprint_flag_source": flag_source,
        "active_cap_reason_allowlist": active_cap_reason_allowlist,
        "production_active_allowed": bool(production_active_allowed),
        "guard_status": guard_status,
        "schema_version": "K3B.deployment_fingerprint.v1",
    }


# ---- source configs (admin) ----
@router.get("/source-configs")
async def list_source_configs():
    db = get_db()
    return await db[Collections.SOURCE_CONFIGS].find({}, {"_id": 0}).sort("source_name", 1).to_list(length=200)


@router.patch("/source-configs/{config_id}")
async def patch_source_config(
    config_id: str,
    body: SourceConfigPatch,
    current_admin: dict = Depends(require_admin),
):
    db = get_db()
    update = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not update:
        raise HTTPException(status_code=400, detail="No fields to update.")
    # Phase 5D Fix 3 — block disabling PubMed in dev/preview when the
    # DEV_FORCE_ENABLE_PUBMED override is on. Otherwise admin toggles
    # (or a buggy UI auto-toggle) defeat the startup re-enable hook
    # and the clinical_queries family silently goes dark.
    if (
        settings.DEV_FORCE_ENABLE_PUBMED
        and not settings.IS_PRODUCTION
        and update.get("enabled") is False
    ):
        existing = await db[Collections.SOURCE_CONFIGS].find_one(
            {"id": config_id}, {"_id": 0, "source_name": 1},
        )
        if existing and existing.get("source_name") == "pubmed":
            raise HTTPException(
                status_code=409,
                detail=(
                    "PubMed cannot be disabled in this environment: "
                    "DEV_FORCE_ENABLE_PUBMED=true. Unset that env var "
                    "to allow admin toggles."
                ),
            )
    update["updated_at"] = datetime.now(timezone.utc)
    res = await db[Collections.SOURCE_CONFIGS].update_one({"id": config_id}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Source config not found.")
    await audit_service.log(
        "source_config_updated",
        user_id=current_admin["id"],
        payload={"source_config_id": config_id, "changes": update},
    )
    return await db[Collections.SOURCE_CONFIGS].find_one({"id": config_id}, {"_id": 0})


@router.get("/audit-events")
async def list_audit_events(
    limit: int = Query(default=100, ge=1, le=500),
    event_type: Optional[str] = Query(default=None),
):
    db = get_db()
    query: dict = {}
    if event_type:
        query["event_type"] = event_type
    return (
        await db[Collections.AUDIT_EVENTS]
        .find(query, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
        .to_list(length=limit)
    )


# ---------------------------------------------------------------------------
# Phase 6B.1 — planner-source counts per session for admin diagnostics.
# Returns the count of search_queries rows grouped by planner_source +
# source + is_primary_strategy/is_fallback_strategy/fallback_reason.
# Used by the admin UI + the Phase 6B.1 acceptance smoke checks.
# ---------------------------------------------------------------------------
@router.get("/search-sessions/{session_id}/planner-counts")
async def planner_counts_for_session(session_id: str):
    db = get_db()
    sess = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "deleted_at": None}, {"_id": 0, "id": 1,
                                                  "evaluator_json": 1}
    )
    if not sess:
        raise HTTPException(status_code=404, detail="Search session not found.")
    rows = await db[Collections.SEARCH_QUERIES].find(
        {"search_session_id": session_id},
        {"_id": 0, "planner_source": 1, "source": 1, "query_family": 1,
         "is_primary_strategy": 1, "is_fallback_strategy": 1,
         "fallback_reason": 1},
    ).to_list(length=1000)
    counts_by_planner: dict[str, int] = {}
    counts_by_planner_and_source: dict[str, dict[str, int]] = {}
    fallback_reasons: dict[str, int] = {}
    families_by_planner: dict[str, dict[str, int]] = {}
    primary_count = 0
    fallback_count = 0
    for r in rows:
        ps = r.get("planner_source") or "unspecified"
        src = r.get("source") or "unknown"
        fam = r.get("query_family") or r.get("query_type") or "unknown"
        counts_by_planner[ps] = counts_by_planner.get(ps, 0) + 1
        counts_by_planner_and_source.setdefault(ps, {})
        counts_by_planner_and_source[ps][src] = (
            counts_by_planner_and_source[ps].get(src, 0) + 1
        )
        families_by_planner.setdefault(ps, {})
        families_by_planner[ps][fam] = families_by_planner[ps].get(fam, 0) + 1
        if r.get("is_primary_strategy"):
            primary_count += 1
        if r.get("is_fallback_strategy"):
            fallback_count += 1
        fb = r.get("fallback_reason")
        if fb:
            fallback_reasons[fb] = fallback_reasons.get(fb, 0) + 1
    return {
        "session_id": session_id,
        "total_queries": len(rows),
        "primary_query_count": primary_count,
        "fallback_query_count": fallback_count,
        "counts_by_planner_source": counts_by_planner,
        "counts_by_planner_and_source": counts_by_planner_and_source,
        "families_by_planner_source": families_by_planner,
        "fallback_reasons": fallback_reasons,
        "planner_strategy": (sess.get("evaluator_json") or {}).get(
            "planner_strategy"
        ),
    }



# ---------------------------------------------------------------------------
# Phase 5A — Search benchmark harness + per-session diagnostics.
# ---------------------------------------------------------------------------


@router.get("/search-benchmarks/cases")
async def list_search_benchmark_cases():
    """Closed-set benchmark case list — used by the admin UI + tests.
    Never contains PHI."""
    return {
        "cases": [
            {
                "id": c.id,
                "question": c.question,
                "description": c.description,
                "expected_type_of_question": sorted(c.expected_type_of_question),
                "expected_specialty": c.expected_specialty,
                "gold_pmid_count": len(c.gold_pmids),
                "must_have_anchors": list(c.must_have_anchors),
                "negative_concepts": list(c.negative_concepts),
                "negative_waiver_concepts": list(c.negative_waiver_concepts),
            }
            for c in BENCHMARK_CASES
        ],
    }


@router.post("/search-benchmarks/run")
async def run_search_benchmarks(
    mode: str = Query(default="parse_only", regex="^(parse_only|retrieval)$"),
    current_admin: dict = Depends(require_admin),
):
    """Phase 5A — runs the benchmark harness.

    Modes:
    - ``parse_only`` (default) — parser + concept-block query generation
      only. No retrieval. Safe for any environment.
    - ``retrieval`` — parser + ranker post-process applied to the
      per-case deterministic fixture corpus. Emits the spec's retrieval-
      side fields (``top_10_pmids``, ``gold_retrieved_count``, etc.).

    Both modes are deterministic and never touch the network. Live-source
    retrieval is intentionally out of scope this phase.
    """
    if mode == "retrieval":
        out = await search_benchmark_service.run_all_with_fixtures()
    else:
        out = await search_benchmark_service.run_all_parse_only()
    await audit_service.log(
        "search_benchmark_run",
        user_id=current_admin["id"],
        payload={"mode": mode, "case_count": len(out.get("cases") or [])},
    )
    return out


# ---------------------------------------------------------------------------
# Phase 5C — benchmark_runs history persistence + admin endpoints.
# ---------------------------------------------------------------------------

@router.post("/benchmark-runs/run")
async def run_benchmark_sweep(
    execution_mode: str = Query(default="mocked", regex="^(mocked|live)$"),
    case_ids: str | None = Query(default=None),
    sleep_between_cases_s: float = Query(default=4.0, ge=0.0, le=30.0),
    current_admin: dict = Depends(require_admin),
):
    """Phase 5C / 6C — run the benchmark sweep.

    ``execution_mode``:
    - ``mocked`` — uses the deterministic fixture corpus (CI-safe).
    - ``live``   — Phase 6C paced live sweep. Calls the real
                    PubMed/Europe PMC pipeline for each case in
                    ``case_ids`` (CSV) or, by default, the 2 real-gold
                    cases (``milrinone_sah`` + ``ami_hemoglobin_transfusion``).
                    The other cases are flagged
                    ``deferred_pending_gold_curation``.
                    ``sleep_between_cases_s`` defaults to 4s per
                    Phase 6C brief.
    """
    from services.benchmark_runs import benchmark_runs_service
    if execution_mode == "live":
        cids = (
            [s.strip() for s in case_ids.split(",") if s.strip()]
            if case_ids else None
        )
        out = await benchmark_runs_service.run_live_sweep(
            triggered_by_user_id=current_admin["id"],
            case_ids=cids,
            sleep_between_cases_s=float(sleep_between_cases_s),
        )
    else:
        out = await benchmark_runs_service.run_and_persist(
            triggered_by_user_id=current_admin["id"],
            execution_mode=execution_mode,
        )
    return {
        "run_id": out["id"],
        "execution_mode": execution_mode,
        "questions_total": out["questions_total"],
        "questions_passed": out["questions_passed"],
        "questions_failed": out["questions_failed"],
        "aggregate": out["aggregate"],
        "deferred_cases": out.get("deferred_cases") or [],
    }


@router.get("/benchmark-runs")
async def list_benchmark_runs(
    limit: int = Query(default=5, ge=1, le=50),
    current_admin: dict = Depends(require_admin),
):
    """Phase 5C — return the last N benchmark runs (aggregates only)."""
    db = get_db()
    cur = (
        db[Collections.BENCHMARK_RUNS]
        .find({}, {"_id": 0, "per_case_results_json": 0})
        .sort("started_at", -1)
        .limit(limit)
    )
    return {"runs": await cur.to_list(length=limit)}


# ---------------------------------------------------------------------------
# Phase 5C.1 — comparison + recommendation routes. MUST be registered
# BEFORE ``/benchmark-runs/{run_id}`` so the literal-path matches first.
# ---------------------------------------------------------------------------

@router.get("/benchmark-runs/comparison")
async def benchmark_runs_comparison(
    run_ids: str | None = Query(default=None),
    execution_mode: str | None = Query(
        default=None, regex="^(mocked|live)$",
    ),
    limit: int = Query(default=5, ge=1, le=20),
    current_admin: dict = Depends(require_admin),
):
    """Phase 5C.1 — aggregate comparison report across selected runs.

    - ``run_ids`` (csv) selects exact runs. If omitted, the most recent
      ``limit`` runs (optionally filtered by ``execution_mode``) are
      pooled.
    - Response strips internal Mongo ``_id`` and is redaction-safe (no
      PHI / raw prompts / raw evidence text — the underlying
      ``comparison_report`` block is already designed for that).
    """
    db = get_db()
    q: dict[str, Any] = {}
    if run_ids:
        ids = [s.strip() for s in run_ids.split(",") if s.strip()]
        q["id"] = {"$in": ids}
    if execution_mode:
        q["execution_mode"] = execution_mode
    cur = (
        db[Collections.BENCHMARK_RUNS]
        .find(q, {"_id": 0})
        .sort("started_at", -1)
        .limit(limit)
    )
    runs = await cur.to_list(length=limit)
    pooled_cases: list[dict[str, Any]] = []
    for r in runs:
        cr = (r.get("comparison_report") or {}).get("per_case_three_mode") or []
        pooled_cases.extend(cr)
    return {
        "run_ids_included": [r.get("id") for r in runs],
        "runs_count": len(runs),
        "per_case_three_mode": pooled_cases,
        "modes_evaluated": ["off", "shadow", "active_simulation"],
        "execution_mode_filter": execution_mode,
    }


@router.get("/benchmark-runs/recommendation")
async def benchmark_runs_recommendation(
    run_ids: str | None = Query(default=None),
    execution_mode: str | None = Query(
        default=None, regex="^(mocked|live)$",
    ),
    limit: int = Query(default=3, ge=1, le=10),
    current_admin: dict = Depends(require_admin),
):
    """Phase 5C.1 — GO / CONTINUE_SHADOW / NO_GO recommendation."""
    from services.benchmark_recommendation import compute_recommendation
    db = get_db()
    q: dict[str, Any] = {}
    if run_ids:
        ids = [s.strip() for s in run_ids.split(",") if s.strip()]
        q["id"] = {"$in": ids}
    if execution_mode:
        q["execution_mode"] = execution_mode
    cur = (
        db[Collections.BENCHMARK_RUNS]
        .find(q, {"_id": 0})
        .sort("started_at", -1)
        .limit(limit)
    )
    runs = await cur.to_list(length=limit)
    rec = compute_recommendation(runs)
    return rec.to_dict()


@router.get("/benchmark-runs/{run_id}")
@router.get("/benchmark-runs/{run_id}/", include_in_schema=False)
async def get_benchmark_run(
    run_id: str,
    redact: bool = Query(default=False),
    current_admin: dict = Depends(require_admin),
):
    """Phase 5C — full per-case results for a given run.

    Phase 6C — when ``redact=true`` the response is filtered through a
    closed-set allowlist that strips any field that could contain
    PHI, raw evidence text, raw LLM prompts, secrets, or uploaded
    full text. Used by the admin dashboard "Download REDACTED JSON"
    button.
    """
    db = get_db()
    doc = await db[Collections.BENCHMARK_RUNS].find_one(
        {"id": run_id}, {"_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="benchmark run not found")
    if redact:
        from services.benchmark_redaction import redact_benchmark_run  # noqa: WPS433
        return redact_benchmark_run(doc)
    return doc


# ---------------------------------------------------------------------------
# Phase 5C.1 — benchmark case curation workflow.
# ---------------------------------------------------------------------------

_BENCHMARK_CASE_STATUS = {
    "parse_only", "needs_gold_set", "active_benchmark", "retired",
}


@router.get("/benchmark-cases")
async def list_benchmark_cases(
    current_admin: dict = Depends(require_admin),
):
    """Phase 5C.1 — list every benchmark case with its curation status."""
    from services.benchmark_runs import all_phase5c_cases
    cases = all_phase5c_cases()
    return {
        "cases": [
            {
                "id": c.id,
                "question": c.question,
                "expected_specialty": c.expected_specialty,
                "specialty_tag": c.specialty_tag,
                "gold_pmids": list(c.gold_pmids or []),
                "negative_title_patterns": list(c.negative_title_patterns or []),
                "provisional": c.provisional,
                "status": getattr(c, "status", None) or (
                    "needs_gold_set" if c.provisional else "active_benchmark"
                ),
                "updated_by_user_id": getattr(c, "updated_by_user_id", None),
                "updated_at": getattr(c, "updated_at", None),
            }
            for c in cases
        ],
    }


class BenchmarkCasePatch(BaseModel):
    status: str | None = None
    gold_pmids: list[str] | None = None
    negative_title_patterns: list[str] | None = None


@router.patch("/benchmark-cases/{case_id}")
async def patch_benchmark_case(
    case_id: str,
    patch: BenchmarkCasePatch,
    current_admin: dict = Depends(require_admin),
):
    """Phase 5C.1 — update a benchmark case (status / gold / negatives).

    Closed-set status: parse_only | needs_gold_set | active_benchmark | retired.
    Updates are in-memory for the current process; future phases can
    persist to Mongo.
    """
    from services.benchmark_runs import all_phase5c_cases
    cases = all_phase5c_cases()
    case = next((c for c in cases if c.id == case_id), None)
    if case is None:
        raise HTTPException(status_code=404, detail="benchmark case not found")
    if patch.status is not None:
        if patch.status not in _BENCHMARK_CASE_STATUS:
            raise HTTPException(
                status_code=400,
                detail=f"status must be one of {sorted(_BENCHMARK_CASE_STATUS)}",
            )
        case.status = patch.status
        case.provisional = patch.status in {"needs_gold_set", "parse_only"}
    if patch.gold_pmids is not None:
        case.gold_pmids = list(patch.gold_pmids)
    if patch.negative_title_patterns is not None:
        case.negative_title_patterns = list(patch.negative_title_patterns)
    case.updated_by_user_id = current_admin["id"]
    case.updated_at = datetime.now(timezone.utc).isoformat()
    await audit_service.log(
        "benchmark_case_updated",
        user_id=current_admin["id"],
        payload={
            "case_id": case_id,
            "status": case.status,
            "gold_pmid_count": len(case.gold_pmids or []),
            "negative_pattern_count": len(case.negative_title_patterns or []),
        },
    )
    return {
        "id": case.id,
        "status": case.status,
        "gold_pmids": list(case.gold_pmids or []),
        "negative_title_patterns": list(case.negative_title_patterns or []),
        "updated_by_user_id": case.updated_by_user_id,
        "updated_at": case.updated_at,
    }


async def _load_m3_2_sidecar_for_session(session_id: str) -> Optional[dict]:
    """Phase M3.2 — best-effort sidecar lookup for the admin
    diagnostics surface. Returns ``None`` when no sidecar exists.
    Never raises.
    """
    try:
        from services.semantic_repair_active_pipeline import (
            get_session_decision_latest,
        )
        return await get_session_decision_latest(session_id)
    except Exception:  # noqa: BLE001
        return None


@router.get("/search-sessions/{session_id}/diagnostics")
async def search_session_diagnostics(
    session_id: str,
    current_admin: dict = Depends(require_admin),
):
    """Phase 5A — Search Quality Diagnostics panel.

    Returns the full PICOTT + queries + top-50 articles for the session.
    Refuses to expose data for PHI-blocked sessions. Used by the admin
    dev panel and the benchmark UI.

    Phase M3.2 — the response now also surfaces the active-pipeline
    sidecar under ``m3_2_session_decision_sidecar`` (read-only,
    ``None`` for legacy sessions) and the closed-set diagnostic
    block under ``m3_2_active_repair``.
    """
    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id}, {"_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found.")
    if doc.get("status") == "blocked_phi":
        raise HTTPException(
            status_code=403,
            detail="Diagnostics unavailable for PHI-blocked sessions",
        )
    queries = await (
        db[Collections.SEARCH_QUERIES]
        .find({"search_session_id": session_id}, {"_id": 0})
        .sort("created_at", 1)
        .to_list(length=200)
    )
    sa_docs = await (
        db[Collections.SEARCH_SESSION_ARTICLES]
        .find({"search_session_id": session_id}, {"_id": 0})
        .sort("overall_rank", -1)
        .limit(50)
        .to_list(length=50)
    )
    art_ids = [s.get("article_id") for s in sa_docs if s.get("article_id")]
    art_lookup: dict[str, dict[str, Any]] = {}
    if art_ids:
        cur = db[Collections.ARTICLES].find(
            {"id": {"$in": art_ids}},
            {"_id": 0, "id": 1, "title": 1, "pmid": 1, "doi": 1,
             "article_type_classification": 1, "publication_year": 1},
        )
        async for row in cur:
            art_lookup[row.get("id") or ""] = row
    enriched_articles = []
    for s in sa_docs:
        art = art_lookup.get(s.get("article_id") or "") or {}
        enriched_articles.append({
            "rank": s.get("overall_rank"),
            "relevance_score": s.get("relevance_score"),
            "evidence_design_score": s.get("evidence_design_score"),
            "title": art.get("title"),
            "pmid": art.get("pmid"),
            "doi": art.get("doi"),
            "article_type_classification": art.get("article_type_classification"),
            "publication_year": art.get("publication_year"),
            "query_family_matches": s.get("query_family_matches") or s.get("query_types_matched") or [],
            "matched_must_have": s.get("matched_must_have") or [],
            "missing_must_have": s.get("missing_must_have") or [],
            "matched_ranking_concepts": s.get("matched_ranking_concepts") or [],
            "penalty_concepts": s.get("penalty_concepts") or [],
            "relevance_cap_applied": s.get("relevance_cap_applied"),
            "cap_reasons": s.get("cap_reasons") or [],
            # Phase 5D Fix 5 — structured boost provenance.
            "ranking_boost_reasons": s.get("ranking_boost_reasons") or [],
            "ranking_boost_total": int(s.get("ranking_boost_total") or 0),
            "rank_explanation": s.get("rank_explanation"),
        })

    # ---- Phase 5A Task 14 retest — normalized response fields. ----
    picott = doc.get("picott_json") or {}
    sources_at_session = list(doc.get("sources_searched") or [])

    # Build a flattened ``retrieval_evaluator`` block so the admin UI +
    # tests can rely on explicit, predictable fields. The raw evaluator
    # dict (with ``picott_diagnostics`` etc.) is preserved alongside it
    # for backward compatibility.
    ev_raw = doc.get("evaluator_json") or {}
    pd = ev_raw.get("picott_diagnostics") or {}
    top10 = pd.get("anchor_coverage_top10") or {}
    top20 = pd.get("anchor_coverage_top20") or {}

    def _coverage_pct(block: dict[str, Any], default: float) -> float:
        val = block.get("coverage_pct") if isinstance(block, dict) else None
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    has_must_haves = bool(picott.get("must_have_concepts"))
    # When there are no must-have anchors we cannot compute coverage —
    # return ``None`` so the UI can render "No data" honestly.
    anchor_cov_10 = _coverage_pct(top10, 100.0) if has_must_haves else None
    anchor_cov_20 = _coverage_pct(top20, 100.0) if has_must_haves else None

    # Preferred study-type coverage in top-10: % of top-10 whose
    # article_type_classification appears in picott.ideal_study_types.
    preferred_types = set((picott.get("ideal_study_types") or [])[:5])
    if preferred_types:
        top10_for_type = enriched_articles[:10]
        if top10_for_type:
            target_hits = sum(
                1 for a in top10_for_type
                if (a.get("article_type_classification") or "") in preferred_types
            )
            preferred_cov_10: Any = 100.0 * target_hits / len(top10_for_type)
        else:
            preferred_cov_10 = 0.0
    else:
        preferred_cov_10 = None  # No declared preference — render as "No data".

    retrieval_evaluator = {
        # Spec-mandated explicit flat fields.
        "anchor_coverage_top_10": anchor_cov_10,
        "anchor_coverage_top_20": anchor_cov_20,
        "recommended_next_action": ev_raw.get("recommended_next_action") or "none",
        "broad_recall_only_domination": bool(
            pd.get("broad_recall_only_dominates_topk")
        ),
        "preferred_study_type_coverage_top_10": preferred_cov_10,
        # Auxiliary context — kept so the frontend can render reason +
        # missing-anchor lists without parsing the raw evaluator.
        "missing_anchors_top_10": list(top10.get("missing_anchors_in_topk") or []),
        "missing_anchors_top_20": list(top20.get("missing_anchors_in_topk") or []),
        "rescue_query_needed": bool(pd.get("rescue_query_needed")),
        "rescue_must_haves": list(pd.get("rescue_must_haves") or []),
        "reason": ev_raw.get("reason") or "",
        "problems": list(ev_raw.get("problems") or []),
        # Pass-through of the raw nested object for any UI that still
        # reads it directly. Always a dict — never bare ``None`` — so
        # the frontend's optional-chaining always succeeds.
        "picott_diagnostics": pd,
    }

    # Phase K2 — admin-level branch_query_plan projection. The plan is
    # persisted by the pipeline under
    # ``clinical_question_map_json.branch_query_plan`` (sibling of
    # ``branches``). The clinician GET already surfaces this under
    # ``session_diagnostics_json.clinical_question_map.branch_query_plan``;
    # the admin diagnostics endpoint surfaces the SAME closed-set
    # projection here so the admin tester can compare K2 + K3A side by
    # side without two round-trips. Same forbidden-substring leak guard.
    admin_branch_query_plan: Optional[dict[str, Any]] = None
    raw_cqm_admin = doc.get("clinical_question_map_json") or {}
    if isinstance(raw_cqm_admin, dict):
        raw_bqp_admin = raw_cqm_admin.get("branch_query_plan")
        if isinstance(raw_bqp_admin, dict):
            try:
                import json as _json_k2_admin
                _bqp_blob = _json_k2_admin.dumps(raw_bqp_admin)
                _bqp_forbidden_hit = any(
                    s in _bqp_blob for s in _K3A_FORBIDDEN_DIAG_SUBSTRINGS
                )
            except Exception:  # noqa: BLE001
                _bqp_forbidden_hit = False
            if not _bqp_forbidden_hit:
                # Pass-through of the persisted closed-set plan, with
                # the shadow-invariant flags force-stamped at the
                # projection layer as defense-in-depth.
                admin_branch_query_plan = dict(raw_bqp_admin)
                admin_branch_query_plan["active_for_retrieval"] = False
                admin_branch_query_plan["mode"] = "shadow"

    # Phase K3A — admin-level article lane diagnostics projection.
    # Same closed-set leak guard as the clinician projection but with
    # the full ``explanation`` rule-id list (up to 2 KB), the full
    # ``qa_warnings`` list, and the full per-article ``cap_candidates``
    # array. ``cap_mode`` is force-stamped to ``"shadow"`` and the
    # ``would_cap`` / ``would_exclude`` flags to ``False`` regardless of
    # what was persisted — K3A is shadow only.
    raw_lane_diag_admin = (
        doc.get("session_diagnostics_json", {}) or {}
    ).get("article_lane_diagnostics")
    diag_persisted = doc.get("session_diagnostics_json") or {}
    admin_lane_block: Optional[dict[str, Any]] = None
    if isinstance(raw_lane_diag_admin, dict):
        try:
            import json as _json_k3a_admin
            _blob_admin = _json_k3a_admin.dumps(raw_lane_diag_admin)
            _forbidden_hit_admin = any(
                s in _blob_admin for s in _K3A_FORBIDDEN_DIAG_SUBSTRINGS
            )
        except Exception:  # noqa: BLE001
            _forbidden_hit_admin = False
        if not _forbidden_hit_admin:
            _full_lane: list[dict[str, Any]] = []
            for d in (raw_lane_diag_admin.get("diagnostics") or []):
                if not isinstance(d, dict):
                    continue
                _explanation_full = [
                    str(x) for x in (d.get("explanation") or [])
                    if isinstance(x, str)
                ]
                # 2 KB cap on the joined explanation rule-id list.
                if sum(len(x) for x in _explanation_full) > 2048:
                    _running = 0
                    _capped: list[str] = []
                    for x in _explanation_full:
                        if _running + len(x) > 2048:
                            break
                        _capped.append(x)
                        _running += len(x)
                    _explanation_full = _capped
                _full_lane.append({
                    "article_id": str(d.get("article_id") or ""),
                    "search_session_article_id": d.get(
                        "search_session_article_id"
                    ),
                    "lane_label": str(d.get("lane_label") or ""),
                    "lane_confidence": (
                        float(d.get("lane_confidence") or 0.0)
                        if isinstance(d.get("lane_confidence"), (int, float))
                        else 0.0
                    ),
                    "branch_alignment": str(
                        d.get("branch_alignment") or "unknown"
                    ),
                    "matched_branch_ids": list(
                        d.get("matched_branch_ids") or []
                    ),
                    "article_intent_fit": str(
                        d.get("article_intent_fit") or "none"
                    ),
                    "directness_label": str(
                        d.get("directness_label") or "indirect"
                    ),
                    "cap_candidates": list(d.get("cap_candidates") or []),
                    "cap_reason_codes": list(
                        d.get("cap_reason_codes") or []
                    ),
                    "cap_severity": str(d.get("cap_severity") or "low"),
                    "cap_confidence": (
                        float(d.get("cap_confidence") or 0.0)
                        if isinstance(d.get("cap_confidence"), (int, float))
                        else 0.0
                    ),
                    "cap_mode": "shadow",
                    "would_cap": False,
                    "would_exclude": False,
                    "explanation": _explanation_full,
                    "evidence_role_candidate": str(
                        d.get("evidence_role_candidate") or "unknown"
                    ),
                    "qa_warnings": list(d.get("qa_warnings") or []),
                    # Phase K3A.3 — closed-set join status on the
                    # ``full_v1.1`` contract.
                    "article_join_status": str(
                        d.get("article_join_status") or "stable"
                    ),
                    # Phase K3B Wave 1 Step 1.5 — full shadow-
                    # enforcement payload on the ``full_v1.3``
                    # contract. ``enforcement_explanation`` is now
                    # explicitly ``List[str]`` (every element coerced
                    # to ``str`` and non-strings filtered) — Item 2
                    # type normalization. Cap stays at ≤ 64 entries.
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
                    "enforcement_candidate_type": str(
                        d.get("enforcement_candidate_type")
                        or "retracted_only"
                    ),
                    "enforcement_mode": str(
                        d.get("enforcement_mode") or "disabled"
                    ),
                    "enforcement_explanation": [
                        str(x) for x in (
                            d.get("enforcement_explanation_full") or []
                        )
                        if isinstance(x, str)
                    ][:64],
                })
            admin_lane_block = {
                "schema_version": str(raw_lane_diag_admin.get(
                    "schema_version"
                ) or "K3A.1.0"),
                "lane_label_mode": str(raw_lane_diag_admin.get(
                    "lane_label_mode"
                ) or "shadow"),
                "cap_detection_mode": str(raw_lane_diag_admin.get(
                    "cap_detection_mode"
                ) or "shadow"),
                # Phase K3B Wave 1 Step 1.5 — admin payload bumped
                # from ``full_v1.2`` to ``full_v1.3``. Item 2 type
                # normalization: ``enforcement_explanation`` is now
                # explicitly ``List[str]`` (closed-set rule IDs).
                # Downstream consumers must read the field as a list
                # of strings.
                "projection_contract": "full_v1.3",
                "diagnostics": _full_lane,
            }

    # Phase K3B Wave 1 Step 2 — admin-level active-exclusion block.
    # Surfaces the same closed-set per-article diagnostic that drives
    # the clinician banner, PLUS full closed-set enforcement
    # explanation rule IDs for the operator. Same closed-set leak
    # guard. ``None`` if no actively-excluded articles exist.
    admin_active_exclusions_block: Optional[dict[str, Any]] = None
    raw_k3b_active_admin = (
        diag_persisted.get("k3b_active_exclusions")
        if isinstance(diag_persisted, dict) else None
    )
    if isinstance(raw_k3b_active_admin, dict) and isinstance(
        raw_k3b_active_admin.get("exclusions"), list,
    ):
        excl_rows_admin = raw_k3b_active_admin.get("exclusions") or []
        try:
            import json as _json_k3b
            _blob_k3b = _json_k3b.dumps(raw_k3b_active_admin)
            _k3b_admin_forbidden = any(
                s in _blob_k3b for s in _K3A_FORBIDDEN_DIAG_SUBSTRINGS
            )
        except Exception:  # noqa: BLE001
            _k3b_admin_forbidden = False
        if not _k3b_admin_forbidden:
            _admin_excl: list[dict[str, Any]] = []
            for r in excl_rows_admin:
                if not isinstance(r, dict):
                    continue
                _admin_excl.append({
                    "article_id": str(r.get("article_id") or ""),
                    "pmid": (
                        str(r.get("pmid"))
                        if r.get("pmid") else None
                    ),
                    "doi": (
                        str(r.get("doi"))
                        if r.get("doi") else None
                    ),
                    "title": str(r.get("title") or "")[:200],
                    "journal": str(r.get("journal") or "")[:160],
                    "publication_year": r.get("publication_year"),
                    "source_signal": str(
                        r.get("source_signal")
                        or "pubmed_retracted_publication_type"
                    ),
                    "cap_reason_codes": [
                        str(c) for c in (
                            r.get("cap_reason_codes") or []
                        ) if isinstance(c, str)
                    ],
                    "enforcement_explanation": [
                        str(x) for x in (
                            r.get("enforcement_explanation") or []
                        ) if isinstance(x, str)
                    ][:64],
                    "article_join_status": "stable",
                    "enforcement_mode": "active",
                })
            admin_active_exclusions_block = {
                "schema_version": "k3b.wave1.step2.v1",
                "projection_contract": "k3b_active_exclusions_v1",
                "enforcement_mode": "active",
                "enforcement_candidate_type": "retracted_only",
                "total": len(_admin_excl),
                "exclusions": _admin_excl,
            }

    return {
        "session_id": session_id,
        "original_question": doc.get("original_question"),
        "sanitized_question": doc.get("sanitized_question"),
        # Phase L1 — explicit evidence-lens / population-focus preference
        # diagnostics. Closed-set tokens + counts only (no PHI / no free
        # text). Surfaces selected prefs, affected article count, the
        # would-be / applied rank delta, and the safety-caps-dominant
        # invariant for the operator.
        "l1_preference_diagnostics": (
            doc.get("l1_preference_diagnostics_json")
            if isinstance(doc.get("l1_preference_diagnostics_json"), dict)
            else None
        ),
        # Phase 6F.2 — closed-set efficiency instrumentation. NO PHI,
        # NO prompts, NO completions are ever stored in these fields.
        # Read directly from session_diagnostics_json populated by the
        # pipeline's finally-flush + the summarizer router's flush.
        # See services/diagnostics.py::SessionDiagnostics.to_dict for
        # the closed-set shape.
        "session_diagnostics": {
            "llm_calls_count": (
                (doc.get("session_diagnostics_json") or {}).get("llm_calls_count")
            ),
            "llm_calls_by_stage": (
                (doc.get("session_diagnostics_json") or {}).get("llm_calls_by_stage")
                or {}
            ),
            "token_estimate": (
                (doc.get("session_diagnostics_json") or {}).get("token_estimate")
            ),
            "latency_by_stage_json": (
                (doc.get("session_diagnostics_json") or {}).get("latency_by_stage_json")
                or {}
            ),
        },
        # Phase 5A retest B1 — the question after PICOTT normalization.
        # When no normalization happened, this falls back to the
        # sanitized question so the field is never null.
        "normalized_question": picott.get("normalized_question")
            or doc.get("sanitized_question") or doc.get("original_question"),
        # Phase 5B Task 12 — translated question + concept translations.
        "translated_question": picott.get("translated_question") or "",
        "concept_translations": picott.get("concept_translations") or [],
        "latent_concepts": picott.get("latent_concepts") or [],
        "topic_label": doc.get("topic_label"),
        "picott": picott,
        "legacy_parsed": doc.get("pico_json") or {},
        # Phase 5A retest B2 — admins need to know which sources were
        # enabled when this specific session ran so a missing source
        # under ``queries_by_family`` is interpretable.
        "sources_enabled_at_session_time": sources_at_session,
        # Phase 5B Task 12 — Scholar-like expansion mode + counts.
        "scholar_like_expansion_mode": (
            os.environ.get("SCHOLAR_LIKE_EXPANSION_MODE", "shadow") or "shadow"
        ).strip().lower(),
        "scholar_like_expansion": doc.get("scholar_like_expansion_json") or {},
        "candidate_expansion_tiers": (
            (doc.get("scholar_like_expansion_json") or {}).get("per_tier_counts")
            or {}
        ),
        "seed_articles_used_for_expansion": (
            (doc.get("scholar_like_expansion_json") or {})
            .get("seed_articles_used_for_expansion") or []
        ),
        "related_cited_reference_candidates_added": (
            (doc.get("scholar_like_expansion_json") or {})
            .get("related_cited_reference_candidates_added") or {}
        ),
        # Phase 5B — semantic rerank score sample (top-10 most-relevant rows).
        "semantic_reranker_scores": [
            {
                "rank": a["rank"],
                "pmid": a.get("pmid"),
                "title": (a.get("title") or "")[:120],
                "semantic_relevance_score": a.get("semantic_relevance_score"),
                "direct_answer_likelihood": a.get("direct_answer_likelihood"),
                "matched_canonical_concepts": (a.get("matched_canonical_concepts") or [])[:6],
                "missing_canonical_concepts": (a.get("missing_canonical_concepts") or [])[:6],
            }
            for a in enriched_articles[:10]
        ],
        # Phase 5B — top-k before vs after semantic rerank.
        "top_k_before_semantic_rerank": (
            doc.get("top_k_before_semantic_rerank_json") or []
        ),
        "top_k_after_semantic_rerank": (
            doc.get("top_k_after_semantic_rerank_json") or []
        ),
        "queries_by_family": [
            {
                "source": q.get("source"),
                "query_family": q.get("query_family") or q.get("query_type"),
                "query_type": q.get("query_type"),
                "candidate_tier": q.get("candidate_tier") or q.get("query_family") or q.get("query_type"),
                "query_string": q.get("query_string"),
                "mandatory_anchors": q.get("mandatory_anchors_json") or [],
                "optional_terms": q.get("optional_terms_json") or [],
                "rationale": q.get("rationale"),
                "max_results": q.get("max_results"),
                "triggered_by": q.get("triggered_by"),
                # Phase 5D — field tags + MeSH provenance + Clinical Queries scope.
                "field_tags_used": q.get("field_tags_used_json") or [],
                "mesh_terms_used": q.get("mesh_terms_used_json") or [],
                "cq_category": q.get("cq_category"),
                "cq_scope": q.get("cq_scope"),
            }
            for q in queries
        ],
        "top_articles": enriched_articles,
        # Phase M0/M1/M2-shadow — bounded semantic anchor repair
        # diagnostics. Closed-set fields only; the candidate JSON is
        # already validated + scrubbed (no PHI, no article refs, no
        # unapproved sources, no answer text). NO raw prompt /
        # completion is persisted by the pipeline so none can leak
        # here. ``None`` for legacy sessions persisted before
        # Phase M.
        "semantic_anchor_repair": (
            (doc.get("session_diagnostics_json") or {}).get("semantic_anchor_repair")
            or None
        ),
        # Phase M3.2 — preview-only Active Semantic Repair decision
        # block. Closed-set fields persisted by the pipeline
        # (``services/pipeline.py``) plus the audit sidecar from
        # ``m3_2_session_decisions`` (looked up by session_id).
        # ``None`` when this session pre-dates Phase M3.2 or no
        # sidecar was written (e.g., PHI-blocked path).
        "m3_2_active_repair": (
            (doc.get("session_diagnostics_json") or {}).get("m3_2_active_repair")
            or None
        ),
        "m3_2_session_decision_sidecar": await _load_m3_2_sidecar_for_session(
            session_id,
        ),
        # CQM -> legacy question-type reconciliation. The whole point of
        # ``CQM_TYPE_RECONCILIATION_MODE=shadow`` is to review the
        # would-be change before flipping to ``active`` — which is
        # impossible if the block is persisted but never exposed. This
        # is the read surface for that soak.
        # Closed-set enum tokens + booleans ONLY (mode, applied,
        # skipped_reason, primary_intent, baseline/proposed
        # type_of_question / question_subtype / question_scope,
        # would_change). NO raw question text, NO prompts, NO PHI.
        # ``None`` for legacy sessions and whenever the mode is
        # ``disabled`` (the default), because the pipeline persists
        # nothing in that case.
        "cqm_type_reconciliation": (
            (doc.get("session_diagnostics_json") or {}).get(
                "cqm_type_reconciliation"
            )
            or None
        ),
        # Phase 5A retest B3 — predictable flat evaluator response.
        "retrieval_evaluator": retrieval_evaluator,
        # Phase K3A — admin-level shadow lane diagnostics. Full
        # explanation rule-id list (≤ 2 KB), full qa_warnings, full
        # per-article cap_candidates. ``cap_mode`` is always
        # ``"shadow"``, ``would_cap`` / ``would_exclude`` always
        # ``False``. ``None`` when the pipeline never persisted lane
        # diagnostics (legacy session, K3A flag disabled, or no
        # branches in the ClinicalQuestionMap).
        "article_lane_diagnostics": admin_lane_block,
        # Phase K3B Wave 1 Step 2 — admin-level active-exclusion
        # diagnostics. ``None`` when no articles were actively
        # excluded. Closed-set, leak-guarded.
        "k3b_active_exclusions": admin_active_exclusions_block,
        # Phase K2 — admin-level shadow branch_query_plan. Same closed
        # -set leak guard; ``active_for_retrieval`` / ``mode`` force-
        # stamped to shadow at projection layer. ``None`` when no plan
        # was persisted (legacy session, K2 flag disabled, or no
        # branches in the ClinicalQuestionMap).
        "branch_query_plan": admin_branch_query_plan,
        # Backward-compat: keep the raw evaluator dict for any older UI.
        "evaluator": ev_raw,
    }



# ============================================================
# Phase R2A — Rejected-Article Rescue Classifier (SHADOW, admin-only)
# ============================================================
@router.post("/search-sessions/{session_id}/rescue-classifier-shadow")
async def rescue_classifier_shadow(
    session_id: str,
    current_admin: dict = Depends(require_admin),
):
    """Phase R2A — bounded-AI SHADOW triage of a session's rejected /
    capped-as-low-relevance article list.

    Admin-only. Does NOT change clinician-visible ranking, default
    selection, the evidence packet, the summary, or citation
    verification. The LLM returns closed-set labels/reasons only; every
    article identifier is supplied deterministically. Policy-excluded
    (retracted / preprint) records stay excluded. Preview/dev only by
    default (``r2a_shadow_enabled``); refuses for PHI-blocked sessions.
    """
    from services.rejected_article_rescue import (
        r2a_shadow_enabled,
        run_rescue_shadow,
    )

    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id}, {"_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found.")
    # PHI guard remains first — a PHI-blocked session never reached
    # retrieval, so there is no rejected list and the classifier must not
    # run on it.
    if doc.get("status") == "blocked_phi":
        raise HTTPException(
            status_code=403,
            detail="Rescue classifier unavailable for PHI-blocked sessions.",
        )
    if not r2a_shadow_enabled():
        return {
            "enabled": False,
            "reason": "r2a_rescue_shadow_disabled_in_production",
            "mode": "shadow",
        }

    # Rejected / low-relevance candidate rows (highest-scoring first —
    # the most likely borderline rescues).
    rejected_rows = await (
        db[Collections.SEARCH_SESSION_ARTICLES]
        .find(
            {
                "search_session_id": session_id,
                "$or": [
                    {"exclusion_reason": {"$ne": None}},
                    {"overall_rank": {"$lt": 60}},
                ],
            },
            {"_id": 0},
        )
        .sort("overall_rank", -1)
        .limit(80)
        .to_list(length=80)
    )
    art_ids = [r.get("article_id") for r in rejected_rows if r.get("article_id")]
    articles_by_id: dict[str, Any] = {}
    if art_ids:
        async for art in db[Collections.ARTICLES].find(
            {"id": {"$in": art_ids}}, {"_id": 0},
        ):
            articles_by_id[art.get("id") or ""] = art

    cqm = doc.get("clinical_question_map_json") or doc.get("picott_json") or {}
    question = doc.get("sanitized_question") or doc.get("original_question") or ""

    shadow = await run_rescue_shadow(
        question=question,
        cqm=cqm,
        rejected_rows=rejected_rows,
        articles_by_id=articles_by_id,
    )

    # Persist the shadow result for admin review only. This field is NOT
    # surfaced by the clinician session response (_to_response), so it can
    # never reach a clinician.
    #
    # SHADOW INVARIANT (R2A): persist ONLY the admin-only sidecar. We
    # deliberately do NOT bump ``updated_at`` here — that field IS surfaced
    # by the clinician session response (``_to_response``) and is used for
    # "recently updated" ordering, so touching it would make the
    # clinician-visible payload change. A shadow run must leave the
    # clinician response byte-for-byte identical.
    await db[Collections.SEARCH_SESSIONS].update_one(
        {"id": session_id},
        {"$set": {"r2a_rescue_shadow_json": shadow}},
    )
    await audit_service.log(
        "r2a_rescue_shadow_run",
        user_id=current_admin["id"],
        search_session_id=session_id,
        payload={
            "total_rejected_reviewed": shadow.get("total_rejected_reviewed"),
            "direct_rescue_candidates": shadow.get("direct_rescue_candidates"),
            "borderline_candidates": shadow.get("borderline_candidates"),
            "indirect_candidates": shadow.get("indirect_candidates"),
        },
    )
    return shadow



# ============================================================
# Phase ImpactRank-1 — iCite/OpenAlex citation-impact enrichment (SHADOW)
# ============================================================
@router.post("/search-sessions/{session_id}/impact-enrichment-shadow")
async def impact_enrichment_shadow(
    session_id: str,
    current_admin: dict = Depends(require_admin),
):
    """Phase ImpactRank-1 — deterministic, admin-only SHADOW citation-impact
    enrichment (NIH iCite RCR/APT/cited_by_clin + OpenAlex cited_by_count)
    over a session's article pool, plus a bounded SHADOW impact-boost and a
    shadow rank simulation.

    Does NOT change clinician-visible ranking, default selection, the
    evidence packet, the summary, or citation verification. Clinical
    directness + safety caps + lane relevance DOMINATE citation impact.
    Preview/dev only by default; refuses PHI-blocked sessions.
    """
    from services.impact_enrichment import (
        impactrank_shadow_enabled,
        run_impact_enrichment_shadow,
    )

    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id}, {"_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found.")
    if doc.get("status") == "blocked_phi":
        raise HTTPException(
            status_code=403,
            detail="Impact enrichment unavailable for PHI-blocked sessions.",
        )
    if not impactrank_shadow_enabled():
        return {
            "enabled": False,
            "reason": "impactrank_shadow_disabled_in_production",
            "mode": "shadow",
        }

    # Full article pool for the session (included + rejected), highest rank
    # first, so the rank simulation has the clinician-visible ordering.
    sa_rows = await (
        db[Collections.SEARCH_SESSION_ARTICLES]
        .find({"search_session_id": session_id}, {"_id": 0})
        .sort("overall_rank", -1)
        .limit(200)
        .to_list(length=200)
    )
    art_ids = [r.get("article_id") for r in sa_rows if r.get("article_id")]
    articles_by_id: dict[str, Any] = {}
    if art_ids:
        async for art in db[Collections.ARTICLES].find(
            {"id": {"$in": art_ids}}, {"_id": 0},
        ):
            articles_by_id[art.get("id") or ""] = art

    # R2A rescue labels (reuse the persisted R2A sidecar when present —
    # run the rescue-classifier-shadow endpoint first to populate it)
    # provide directness for rejected articles. Impact enrichment itself
    # is LLM-free and never triggers the R2A LLM pass.
    r2a = doc.get("r2a_rescue_shadow_json")
    r2a_labels = {
        row["article_id"]: {
            "rescue_label": row.get("rescue_label"),
            "relevance_reason": row.get("rescue_reason"),
        }
        for row in ((r2a or {}).get("rejection_reason_vs_rescue") or [])
        if row.get("article_id")
    }

    shadow = await run_impact_enrichment_shadow(
        sa_rows=sa_rows,
        articles_by_id=articles_by_id,
        r2a_candidates_by_article_id=r2a_labels,
        db=db,
    )

    # Admin-only sidecar; do NOT bump updated_at (clinician-visible).
    await db[Collections.SEARCH_SESSIONS].update_one(
        {"id": session_id},
        {"$set": {"impact_enrichment_shadow_json": shadow}},
    )
    await audit_service.log(
        "impactrank_shadow_run",
        user_id=current_admin["id"],
        search_session_id=session_id,
        payload={
            "articles_total": shadow.get("articles_total"),
            "articles_enriched": shadow.get("articles_enriched"),
            "boost_eligible_count": shadow.get("boost_eligible_count"),
        },
    )
    return shadow


# ============================================================
# Phase 6D — Library aggregate metrics (admin-only)
# ============================================================
@router.get("/library-metrics")
async def library_metrics(current_admin: dict = Depends(require_admin)):
    """Counts ONLY. Admin must NEVER see folder names / notes / titles /
    saved-packet titles / library item notes.

    Surfaces system-wide aggregate counts so the operator can monitor
    usage without exposing any user content.
    """
    db = get_db()
    library_items_total = await db[Collections.USER_LIBRARY_ITEMS].count_documents({})
    library_items_active = await db[Collections.USER_LIBRARY_ITEMS].count_documents(
        {"status": "active"},
    )
    library_items_archived = await db[Collections.USER_LIBRARY_ITEMS].count_documents(
        {"status": "archived"},
    )
    library_items_favorited = await db[Collections.USER_LIBRARY_ITEMS].count_documents(
        {"is_favorite": True},
    )
    folders_total = await db[Collections.FOLDERS].count_documents({})
    folder_items_total = await db[Collections.FOLDER_ARTICLES].count_documents({})
    saved_packets_total = await db[Collections.SAVED_EVIDENCE_PACKETS].count_documents(
        {"deleted_at": None},
    )

    distinct_users_with_library = len(
        await db[Collections.USER_LIBRARY_ITEMS].distinct("user_id"),
    )
    distinct_users_with_folders = len(
        await db[Collections.FOLDERS].distinct("user_id"),
    )

    library_phi_blocked = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "library_note_phi_blocked"},
    )
    folder_phi_blocked = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "library_folder_phi_blocked"},
    )
    library_imports_total = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "library_article_imported"},
    )
    library_batch_saves_total = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "library_batch_saved"},
    )

    return {
        "library_items_total": library_items_total,
        "library_items_active": library_items_active,
        "library_items_archived": library_items_archived,
        "library_items_favorited": library_items_favorited,
        "folders_total": folders_total,
        "folder_items_total": folder_items_total,
        "saved_evidence_packets_total": saved_packets_total,
        "distinct_users_with_library": distinct_users_with_library,
        "distinct_users_with_folders": distinct_users_with_folders,
        "phi_blocked_library_text_total": library_phi_blocked + folder_phi_blocked,
        "library_imports_total": library_imports_total,
        "library_batch_saves_total": library_batch_saves_total,
    }



# ---------------------------------------------------------------------
# Phase H2 — Clinical Context Resolver observability tile.
#
# Closed-set rolling-7-day aggregate over the resolver fields persisted
# on ``search_sessions.session_diagnostics_json`` (and only those
# fields). NO new audit events. NO new prompt / completion storage.
# NO session ids, no user emails, no question text in the response.
# ---------------------------------------------------------------------


_RESOLVER_MIN_SAMPLE_SIZE: int = 10


@router.get("/clinical-context-resolver/metrics")
async def get_clinical_context_resolver_metrics(
    db=Depends(get_db),
    current_admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Phase H2 — admin-only rolling-7-day aggregate metrics for the
    bounded LLM Clinical Context Resolver.

    Response is a closed-set numeric payload. **No** raw prompts,
    completions, question text, user emails, secrets, session ids, or
    article references are ever included. The data source is the
    existing ``session_diagnostics_json`` block (Phase H1) plus the
    ``audit_events`` collection. No new audit events are emitted.
    """
    from datetime import datetime, timedelta, timezone
    # Import resolver enums for the closed-set distribution shape.
    from services.clinical_context_resolver import REJECTION_REASONS

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=7)

    sessions_coll = db[Collections.SEARCH_SESSIONS]

    # Sessions in window. Use created_at as the time anchor.
    window_filter = {
        "created_at": {"$gte": window_start, "$lte": now},
    }
    total_session_count = await sessions_coll.count_documents(window_filter)

    # Resolver call counters via $facet for a single round-trip.
    pipeline = [
        {"$match": window_filter},
        {"$facet": {
            "called": [
                {"$match": {
                    "session_diagnostics_json.clinical_context_resolver_called": True,
                }},
                {"$count": "n"},
            ],
            "accepted": [
                {"$match": {
                    "session_diagnostics_json.clinical_context_resolver_accepted": True,
                }},
                {"$count": "n"},
            ],
            "rejected": [
                {"$match": {
                    "$and": [
                        {"session_diagnostics_json.clinical_context_resolver_called": True},
                        {"session_diagnostics_json.clinical_context_resolver_accepted": False},
                    ],
                }},
                {"$count": "n"},
            ],
            "fallback_active_sessions": [
                {"$match": {
                    "session_diagnostics_json.interpretation_source": "llm_resolver_fallback",
                }},
                {"$count": "n"},
            ],
            "shadow_sessions": [
                {"$match": {
                    "session_diagnostics_json.interpretation_source": "llm_resolver_shadow_recorded",
                }},
                {"$count": "n"},
            ],
            "generic_only": [
                {"$match": {
                    "session_diagnostics_json.clinical_context_resolver_skipped_reason": "generic_only_fast_path_fired",
                }},
                {"$count": "n"},
            ],
            "phi_blocked_resolver_skipped": [
                {"$match": {
                    "$and": [
                        {"status": "blocked_phi"},
                        {"$or": [
                            {"session_diagnostics_json.clinical_context_resolver_called": False},
                            {"session_diagnostics_json.clinical_context_resolver_called": {"$exists": False}},
                        ]},
                    ],
                }},
                {"$count": "n"},
            ],
            "rejection_reason_dist": [
                {"$match": {
                    "session_diagnostics_json.resolver_rejection_reason": {"$ne": None},
                }},
                {"$group": {
                    "_id": "$session_diagnostics_json.resolver_rejection_reason",
                    "count": {"$sum": 1},
                }},
            ],
            "latency_samples": [
                {"$match": {
                    "session_diagnostics_json.clinical_context_resolver_accepted": True,
                    "session_diagnostics_json.resolver_latency_ms": {"$gt": 0},
                }},
                {"$project": {
                    "_id": 0,
                    "latency_ms": "$session_diagnostics_json.resolver_latency_ms",
                    "total_tokens": "$session_diagnostics_json.resolver_token_estimate.total_tokens",
                }},
            ],
        }},
    ]

    facets = []
    async for row in sessions_coll.aggregate(pipeline):
        facets.append(row)
    facet = facets[0] if facets else {}

    def _n(key: str) -> int:
        bucket = facet.get(key) or []
        if not bucket:
            return 0
        return int(bucket[0].get("n") or 0)

    called_count = _n("called")
    accepted_count = _n("accepted")
    rejected_count = _n("rejected")
    fallback_active = _n("fallback_active_sessions")
    shadow_count = _n("shadow_sessions")
    generic_only = _n("generic_only")
    phi_blocked_skipped = _n("phi_blocked_resolver_skipped")

    # Closed-set rejection-reason distribution. Initialise every known
    # rejection reason to 0 so consumers see the full enum surface even
    # when a reason has zero occurrences.
    rejection_dist: dict[str, int] = {r: 0 for r in REJECTION_REASONS}
    for row in (facet.get("rejection_reason_dist") or []):
        reason = (row.get("_id") or "")
        if reason in rejection_dist:
            rejection_dist[reason] = int(row.get("count") or 0)

    # Latency percentile estimates from the persisted samples.
    samples = facet.get("latency_samples") or []
    latencies = sorted(int(s.get("latency_ms") or 0) for s in samples if s.get("latency_ms"))
    token_totals = [int(s.get("total_tokens") or 0) for s in samples]

    def _pct(values: list[int], p: float) -> int:
        if not values:
            return 0
        # Closest-rank (no interpolation; closed-set integer output).
        idx = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
        return int(values[idx])

    latency_p50 = _pct(latencies, 50)
    latency_p95 = _pct(latencies, 95)
    token_total_sum = int(sum(token_totals))

    accept_rate: float | None = None
    if called_count > 0:
        accept_rate = round(float(accepted_count) / float(called_count), 4)

    insufficient_sample = called_count < _RESOLVER_MIN_SAMPLE_SIZE

    payload = {
        "window_start": window_start.isoformat(),
        "window_end": now.isoformat(),
        "generated_at": now.isoformat(),
        "min_sample_size": _RESOLVER_MIN_SAMPLE_SIZE,
        "insufficient_sample": bool(insufficient_sample),
        "total_session_count": int(total_session_count),
        "context_resolver_called": called_count,
        "context_resolver_accepted": accepted_count,
        "context_resolver_rejected": rejected_count,
        "resolver_accept_rate": accept_rate if not insufficient_sample else None,
        "rejection_reason_distribution": rejection_dist,
        "resolver_latency_p50_ms": latency_p50,
        "resolver_latency_p95_ms": latency_p95,
        "resolver_token_estimate_total": token_total_sum,
        "fallback_active_session_count": fallback_active,
        "shadow_session_count": shadow_count,
        "generic_only_clarification_count": generic_only,
        "phi_blocked_resolver_skipped_count": phi_blocked_skipped,
    }
    return payload


# ---------------------------------------------------------------------
# Phase M3.1 — preview/dev-only admin A/B retrieval benchmark.
# Endpoints below are gated by the M3_1_BENCHMARK_ENABLED settings
# flag (default OFF). Production fail-closed guard lives in
# ``services/semantic_repair_active_guard.verify_m3_1_benchmark_startup``.
# ---------------------------------------------------------------------

class _M3_1BenchmarkBody(BaseModel):
    session_id: Optional[str] = None
    raw_question: Optional[str] = None


@router.post("/m3_1/benchmark")
async def m3_1_benchmark(
    body: _M3_1BenchmarkBody,
    _admin: dict = Depends(require_admin),
):
    """POST /api/admin/m3_1/benchmark — run an admin-only A/B
    retrieval benchmark for a search session.

    Body MUST contain exactly one of ``session_id`` (preferred for
    M3.1) or ``raw_question`` (deferred — returns 501 on PHI-clean,
    same PHI-block shape as session creation on PHI-detected)."""
    if not settings.M3_1_BENCHMARK_ENABLED:
        raise HTTPException(
            status_code=503,
            detail={"status": "m3_1_disabled"},
        )

    has_sid = bool(body.session_id and body.session_id.strip())
    has_raw = bool(body.raw_question and body.raw_question.strip())
    if has_sid == has_raw:
        raise HTTPException(
            status_code=400,
            detail={"status": "exactly_one_of_session_id_or_raw_question_required"},
        )

    if has_raw:
        # PHI guard FIRST — never persist raw_question text.
        from services.phi_guard import PHIGuardService
        phi = PHIGuardService()
        screen = phi.screen(body.raw_question or "")
        if screen.is_blocked:
            # Match the existing PHI-blocked shape from
            # ``POST /api/search-sessions/``.
            return {
                "status": "blocked_phi",
                "reason_codes": list(screen.reason_codes or []),
                "user_message": (
                    "Your input contains personally identifiable / "
                    "protected health information. Please remove "
                    "these and resubmit. This tool must not receive PHI."
                ),
            }
        # PHI-clean → defer per scope decision. No persistence, no
        # LLM call, no retrieval.
        raise HTTPException(
            status_code=501,
            detail={"status": "m3_1_raw_question_deferred"},
        )

    # session_id path.
    from services.m3_1_benchmark import run_benchmark_for_session
    result = await run_benchmark_for_session(
        session_id=(body.session_id or "").strip(),
        admin_user_id=str(_admin.get("id") or _admin.get("_id") or ""),
    )
    return result


@router.get("/m3_1/runs")
async def m3_1_runs_list(
    limit: int = Query(default=25, ge=1, le=100),
    _admin: dict = Depends(require_admin),
):
    if not settings.M3_1_BENCHMARK_ENABLED:
        raise HTTPException(
            status_code=503,
            detail={"status": "m3_1_disabled"},
        )
    from services.m3_1_benchmark import get_recent_benchmark_runs
    runs = await get_recent_benchmark_runs(limit=limit)
    return {"runs": runs, "count": len(runs)}


@router.get("/m3_1/runs/{run_id}")
async def m3_1_run_get(
    run_id: str,
    _admin: dict = Depends(require_admin),
):
    if not settings.M3_1_BENCHMARK_ENABLED:
        raise HTTPException(
            status_code=503,
            detail={"status": "m3_1_disabled"},
        )
    from services.m3_1_benchmark import get_benchmark_run_by_id
    d = await get_benchmark_run_by_id(run_id)
    if not d:
        raise HTTPException(status_code=404, detail={"status": "run_not_found"})
    return d


# ============================================================
# Phase M3.2 — Preview-only Active Semantic Repair admin surface.
# ============================================================

class _M3_2KillSwitchBody(BaseModel):
    engaged: bool


@router.get("/m3_2/kill-switch")
async def m3_2_kill_switch_get(_admin: dict = Depends(require_admin)):
    """Return the combined kill-switch state for M3.2 active semantic
    repair. Reads BOTH the env var and the Mongo feature flag and
    returns the OR-combined ``engaged`` boolean (either source
    engaged → kill).

    Closed-set response shape:
        {
            "env_kill_switch_engaged": bool,
            "mongo_kill_switch_engaged": bool,
            "combined_engaged": bool,
            "fail_closed_default": True,
            "flag_key": "m3_2_active_repair_kill_switch"
        }
    """
    from services.m3_2_feature_flags import read_kill_switch_state
    state = await read_kill_switch_state()
    return state


@router.get("/m3_2/config")
async def m3_2_config_get(_admin: dict = Depends(require_admin)):
    """Phase M3.2 — live-config introspection (admin-only, read-only).

    Returns the **live** ``SemanticRepairActiveConfig`` snapshot from
    the running backend process, so the operator can verify env-var
    changes (e.g., ``SEMANTIC_REPAIR_ACTIVE_MAX_LATENCY_MS``) without
    waiting for a new sidecar row to be written. Same closed-set
    field shape as ``m3_2_session_decisions.config_snapshot`` plus
    the kill-switch state for convenience.

    NO secrets. NO PHI. NO LLM prompts/completions ever surface.
    """
    from services.m3_2_feature_flags import read_kill_switch_state
    from services.semantic_repair_active_config import load_active_config
    cfg = load_active_config()
    kill_state = await read_kill_switch_state(cfg)
    return {
        "enabled": bool(cfg.enabled),
        "current_env": cfg.current_env,
        "env_allowlist": list(cfg.env_allowlist),
        "shapes_allowlist": list(cfg.shapes_allowlist),
        "max_latency_ms": int(cfg.max_latency_ms),
        "prod_allow_active": bool(cfg.prod_allow_active),
        "is_production_env": bool(cfg.is_production_env),
        "effective_enabled": bool(cfg.effective_enabled),
        "kill_switch_state": kill_state,
        "phase": "M3.2",
    }


@router.post("/m3_2/kill-switch")
async def m3_2_kill_switch_set(
    body: _M3_2KillSwitchBody,
    _admin: dict = Depends(require_admin),
):
    """Toggle the Mongo-backed runtime kill-switch. The env var
    side is NOT modified — it is the fail-closed default. Writes a
    closed-set audit event ``m3_2_kill_switch_toggle``.

    Either source (env OR mongo) being engaged keeps M3.2 OFF —
    this endpoint controls the Mongo side only.
    """
    from services.m3_2_feature_flags import (
        read_kill_switch_state, set_mongo_kill_switch_state,
    )
    actor = str(_admin.get("id") or _admin.get("_id") or "")
    result = await set_mongo_kill_switch_state(
        engaged=bool(body.engaged), actor_user_id=actor,
    )
    state = await read_kill_switch_state()
    return {"set": result, "state": state}


@router.get("/m3_2/session-decisions/{session_id}")
async def m3_2_session_decision_get(
    session_id: str,
    _admin: dict = Depends(require_admin),
):
    """Return the most-recent M3.2 active-pipeline sidecar row for
    ``session_id``. Returns 404 if no sidecar has been written for
    this session yet (e.g., session pre-dates M3.2).
    """
    from services.semantic_repair_active_pipeline import (
        get_session_decision_latest,
    )
    d = await get_session_decision_latest(session_id)
    if not d:
        raise HTTPException(
            status_code=404,
            detail={"status": "m3_2_no_sidecar_for_session"},
        )
    return d


@router.get("/m3_2/session-decisions")
async def m3_2_session_decisions_list(
    limit: int = Query(default=50, ge=1, le=200),
    _admin: dict = Depends(require_admin),
):
    """Return the most-recent N M3.2 active-pipeline sidecar rows
    across all sessions. Used by the admin preview-adjudication
    panel to scan recent decisions.
    """
    from services.semantic_repair_active_pipeline import (
        list_recent_session_decisions,
    )
    rows = await list_recent_session_decisions(limit=limit)
    return {"rows": rows, "count": len(rows)}


# ----- Phase M4 — Full-Question interpreter shadow diagnostics -----


@router.get("/m4/shadow-decisions/{session_id}")
async def m4_shadow_decision_get(
    session_id: str,
    _admin: dict = Depends(require_admin),
):
    """Return the most-recent M4 full-question interpreter shadow
    sidecar row for ``session_id``. 404 if none (e.g., session did not
    trigger M4, or M4 disabled in this env).
    """
    from services.m4_shadow_sidecar import get_m4_decision_latest
    d = await get_m4_decision_latest(session_id)
    if not d:
        raise HTTPException(
            status_code=404,
            detail={"status": "m4_no_sidecar_for_session"},
        )
    return d


@router.get("/m4/shadow-decisions")
async def m4_shadow_decisions_list(
    limit: int = Query(default=50, ge=1, le=200),
    _admin: dict = Depends(require_admin),
):
    """Return the most-recent N M4 full-question interpreter shadow
    sidecar rows across all sessions. Admin adjudication surface."""
    from services.m4_shadow_sidecar import list_recent_m4_decisions
    rows = await list_recent_m4_decisions(limit=limit)
    return {"rows": rows, "count": len(rows)}


# ----- Phase M4.1 — Active M4 kill switch + config (admin-only) -----


class _M4ActiveKillSwitchBody(BaseModel):
    engaged: bool


@router.get("/m4/active-kill-switch")
async def m4_active_kill_switch_get(_admin: dict = Depends(require_admin)):
    """Return the combined (env OR Mongo) kill-switch state for M4.1
    active promotion. Either source engaged → M4.1 dormant."""
    from services.m4_active_feature_flags import read_m4_kill_switch_state
    return await read_m4_kill_switch_state()


@router.post("/m4/active-kill-switch")
async def m4_active_kill_switch_set(
    body: _M4ActiveKillSwitchBody,
    _admin: dict = Depends(require_admin),
):
    """Toggle the Mongo-backed runtime kill switch for M4.1 active
    promotion. The env var side is NOT modified (fail-closed default).
    Writes a closed-set audit event ``m4_active_kill_switch_toggle``."""
    from services.m4_active_feature_flags import (
        read_m4_kill_switch_state, set_m4_mongo_kill_switch_state,
    )
    actor = str(_admin.get("id") or _admin.get("_id") or "")
    result = await set_m4_mongo_kill_switch_state(
        engaged=bool(body.engaged), actor_user_id=actor,
    )
    state = await read_m4_kill_switch_state()
    return {"set": result, "state": state}


@router.get("/m4/active-config")
async def m4_active_config_get(_admin: dict = Depends(require_admin)):
    """Phase M4.1 — live-config introspection (admin-only, read-only).
    NO secrets. NO PHI. NO LLM prompts/completions."""
    from services.m4_active_config import load_m4_active_config
    from services.m4_active_feature_flags import read_m4_kill_switch_state
    cfg = load_m4_active_config()
    kill_state = await read_m4_kill_switch_state(cfg)
    return {
        "enabled": bool(cfg.enabled),
        "current_env": cfg.current_env,
        "env_allowlist": list(cfg.env_allowlist),
        "max_latency_ms": int(cfg.max_latency_ms),
        "prod_allow_active": bool(cfg.prod_allow_active),
        "is_production_env": bool(cfg.is_production_env),
        "effective_enabled": bool(cfg.effective_enabled),
        "kill_switch_state": kill_state,
        "phase": "M4.1",
    }


# ----- Adjudication export + PATCH -----


class _M3_2AdjudicationPatchBody(BaseModel):
    """Closed-set body. Each field optional. ``verdict`` MUST be in
    ``{better, same, worse, not_adjudicable}`` or null;
    ``acceptable_yes_no`` MUST be in ``{yes, no}`` or null;
    ``reason`` is a free-text field truncated to 1200 chars.

    NO other fields are mutable on the sidecar — this body is the
    closed-set physician adjudication surface.
    """
    verdict: Optional[str] = None
    reason: Optional[str] = None
    acceptable_yes_no: Optional[str] = None


@router.get("/m3_2/adjudication-export")
async def m3_2_adjudication_export(
    limit: int = Query(default=50, ge=1, le=200),
    shape: Optional[str] = Query(default=None),
    applied: Optional[str] = Query(default=None, description="true|false|all"),
    _admin: dict = Depends(require_admin),
):
    """Phase M3.2 — Adjudication Export.

    Returns paginated sidecar rows joined to the session + top-10
    articles + per-source error summary, with the physician
    adjudication block ready to be PATCHed.

    Closed-set parameters:
      * ``limit``: 1..200 (default 50)
      * ``shape``: optional shape filter (one of the three
        allowlisted shapes)
      * ``applied``: ``"true"`` / ``"false"`` / ``"all"`` (default ``"all"``)

    Hard safety contracts:
      * Admin auth required (403 otherwise).
      * When BOTH kill-switch sources are engaged (combined kill=true)
        AND no sidecar rows exist, returns ``{"rows": [], "count": 0,
        "kill_switch_engaged": true, ...}`` — no fake rows.
      * Every row passes ``leak_scan_passes`` BEFORE being added. A
        failing row emits closed-set audit event
        ``m3_2_adjudication_export_leak_blocked`` and is replaced by
        a ``{search_session_id, row_redacted: true, ...}`` marker.
      * No PHI / prompts / completions / credentials / abstracts /
        full text / evidence text in the response.
    """
    from services.m3_2_feature_flags import read_kill_switch_state
    from services.semantic_repair_active_pipeline import (
        build_adjudication_export_rows,
    )
    from services.m3_2_banner_state import banner_envelope
    kill_state = await read_kill_switch_state()
    applied_filter: Optional[bool]
    raw_applied = (applied or "").strip().lower()
    if raw_applied == "true":
        applied_filter = True
    elif raw_applied == "false":
        applied_filter = False
    elif raw_applied in ("", "all"):
        applied_filter = None
    else:
        raise HTTPException(
            status_code=400,
            detail={"status": "applied_must_be_true_false_or_all"},
        )
    rows, count, leaked = await build_adjudication_export_rows(
        limit=limit,
        shape_filter=shape,
        applied_filter=applied_filter,
    )
    # Phase R1A — Part 2A — closed-set leak-scan banner envelope.
    # Replaces the ambiguous "0 redacted (leak scan failed)" UI
    # wording with one of four closed-set states.
    leak_envelope = banner_envelope(
        total_rows=int(count or 0),
        redacted_count=int(leaked or 0),
        unscanned_count=0,
    )
    return {
        "rows": rows,
        "count": count,
        "leaked_redacted_count": leaked,
        "leak_scan_banner": leak_envelope,
        "kill_switch_state": kill_state,
        "filters": {
            "limit": int(limit),
            "shape": shape,
            "applied": applied or "all",
        },
        "phase": "M3.2",
    }


@router.patch("/m3_2/adjudication-export/{search_session_id}")
async def m3_2_adjudication_patch(
    search_session_id: str,
    body: _M3_2AdjudicationPatchBody,
    _admin: dict = Depends(require_admin),
):
    """Phase M3.2 — PATCH the physician adjudication block on the
    most-recent sidecar row for ``search_session_id``.

    Closed-set whitelist of mutable fields:
      * ``verdict`` ∈ {better, same, worse, not_adjudicable} or null
      * ``reason``: free text, truncated to 1200 chars or null
      * ``acceptable_yes_no`` ∈ {yes, no} or null

    NO other sidecar field is mutable (A-arm CQM / M-repaired CQM
    / boolean queries / latency / kill_switch_state /
    config_snapshot / fallback_status are immutable post-write).

    Emits closed-set audit event ``m3_2_physician_adjudication_patched``.
    """
    from services.semantic_repair_active_pipeline import (
        set_physician_adjudication,
    )
    actor = str(_admin.get("id") or _admin.get("_id") or "")
    try:
        result = await set_physician_adjudication(
            search_session_id=search_session_id,
            actor_user_id=actor,
            verdict=body.verdict,
            reason=body.reason,
            acceptable_yes_no=body.acceptable_yes_no,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"status": "invalid_adjudication_body", "error": str(exc)},
        ) from exc
    if not result:
        raise HTTPException(
            status_code=404,
            detail={"status": "m3_2_no_sidecar_for_session"},
        )
    return result


# ======================================================================
# Phase R1A — admin-only shadow reranking observability endpoints.
#
# Closed-set. NO clinician exposure. NO mutation of clinician-facing
# ranking. The simulator writes its sidecar rows under
# ``r1a_shadow_decisions`` (TTL 30 days) and the gold-comparator
# benchmark cases live under ``r1a_gold_comparator_cases``.
# ======================================================================
@router.get("/r1a/shadow-decisions")
async def r1a_shadow_decisions_list(
    limit: int = Query(default=50, ge=1, le=200),
    shape: Optional[str] = Query(default=None),
    benchmark_case_id: Optional[str] = Query(default=None),
    _admin: dict = Depends(require_admin),
):
    """Phase R1A — list recent shadow-decision sidecar rows.

    Closed-set filters: ``shape`` (one of the eligible R1A shapes),
    ``benchmark_case_id`` (exact match), and ``limit`` (1..200).
    Returns most-recent-first ordering.

    NO PHI / NO abstracts / NO prompts / NO completions in the
    response — the row was already leak-scanned at write time.
    """
    from services.r1a_shadow_simulator import list_shadow_decisions
    rows = await list_shadow_decisions(
        limit=int(limit),
        shape=shape,
        benchmark_case_id=benchmark_case_id,
    )
    return {
        "rows": rows,
        "count": len(rows),
        "filters": {
            "limit": int(limit),
            "shape": shape,
            "benchmark_case_id": benchmark_case_id,
        },
        "phase": "R1A",
    }


@router.get("/r1a/benchmark-cases")
async def r1a_benchmark_cases_list(
    _admin: dict = Depends(require_admin),
):
    """Phase R1A — list the closed-set gold-comparator benchmark
    cases. Read-only; cases are seeded at startup.
    """
    from services.r1a_gold_comparator_cases import (
        list_gold_comparator_cases,
    )
    cases = await list_gold_comparator_cases()
    return {"cases": cases, "count": len(cases), "phase": "R1A"}


@router.post("/r1a/replay/{case_id}")
async def r1a_replay_case(
    case_id: str,
    _admin: dict = Depends(require_admin),
):
    """Phase R1A — synchronously replay a gold-comparator benchmark
    case against the most-recent matching ``search_session`` (by
    sanitized_question). Writes a new sidecar row and returns its
    id. Admin-blocking on purpose (no live clinician path).
    """
    from services.r1a_shadow_simulator import replay_benchmark_case
    actor = str(_admin.get("id") or _admin.get("_id") or "")
    try:
        result = await replay_benchmark_case(
            case_id=case_id,
            actor_user_id=actor or None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"status": "unknown_case_id", "error": str(exc)},
        ) from exc
    return result



class R1AAdjudicationBody(BaseModel):
    """Closed-set body for ``PATCH /api/admin/r1a/shadow-decisions/{id}``.

    ALL fields are optional; the PATCH must include at least one
    mutation, otherwise the simulator raises
    ``R1AAdjudicationError("nothing_to_update")``.
    """

    verdict: Optional[str] = None  # closed-set R1A_CASE_VERDICTS
    reason: Optional[str] = None  # free text, leak-scanned
    acceptable_for_future_active: Optional[str] = None  # closed-set
    mixed_signal_verdict_overrides: Optional[Dict[str, str]] = None
    # ``mixed_signal_verdict_overrides`` keys are stringified shadow
    # row indices (per JSON-object key constraint); values are the
    # closed-set MIXED_SIGNAL_VERDICTS.


@router.patch("/r1a/shadow-decisions/{sidecar_id}")
async def r1a_patch_shadow_decision(
    sidecar_id: str,
    body: R1AAdjudicationBody,
    _admin: dict = Depends(require_admin),
):
    """Phase R1A — admin-only adjudication PATCH.

    Closed-set mutations only. Everything else is rejected with a
    structured ``detail.error`` code. PHI screen + audit are
    enforced inside the simulator.

    On success, returns the freshly-updated sidecar document.
    """
    from services.r1a_shadow_simulator import (
        R1AAdjudicationError,
        patch_shadow_decision_adjudication,
    )
    actor = str(_admin.get("id") or _admin.get("_id") or "")
    try:
        result = await patch_shadow_decision_adjudication(
            sidecar_id=sidecar_id,
            actor_user_id=actor or None,
            verdict=body.verdict,
            reason=body.reason,
            acceptable_for_future_active=body.acceptable_for_future_active,
            mixed_signal_verdict_overrides=body.mixed_signal_verdict_overrides,
        )
    except R1AAdjudicationError as exc:
        if exc.code == "sidecar_not_found":
            raise HTTPException(
                status_code=404,
                detail={"status": exc.code, "error": exc.code},
            ) from exc
        raise HTTPException(
            status_code=400,
            detail={"status": "r1a_adjudication_validation_failed",
                    "error": exc.code},
        ) from exc
    return result



# ======================================================================
# Phase R1B — admin-only retrieval-recall + directness shadow.
#
# Closed-set. Synchronous. Admin-only. Deterministic query
# expansion via ``services.r1b_query_expansion`` + a single-pass
# fetch via the existing PubMed + Europe PMC connectors. NO LLM.
# NO new connectors. NO clinician-visible side effects.
# ======================================================================
class R1BReplayBody(BaseModel):
    """Optional body for ``POST /api/admin/r1b/replay/{case_id}``.

    The operator may pin a specific ``search_session_id`` to use
    as the active arm. When omitted, the simulator looks up the
    most-recent session matching the case's sanitized question.
    """

    search_session_id: Optional[str] = None


@router.post("/r1b/replay/{case_id}")
async def r1b_replay_case(
    case_id: str,
    body: Optional[R1BReplayBody] = None,
    _admin: dict = Depends(require_admin),
):
    """Phase R1B — synchronously replay a gold benchmark case
    through the deterministic query-expansion + connector re-fetch
    pipeline. Returns the freshly written sidecar id and a compact
    summary (gold rank delta, over-rank counter deltas, provenance
    counts). Closed-set error codes for unknown case / ineligible
    shape.
    """
    from services.r1b_shadow_simulator import (
        R1BReplayError,
        replay_benchmark_case_r1b,
    )
    actor = str(_admin.get("id") or _admin.get("_id") or "")
    override = (body.search_session_id if body else None)
    try:
        result = await replay_benchmark_case_r1b(
            case_id=case_id,
            actor_user_id=actor or None,
            search_session_id_override=override,
        )
    except R1BReplayError as exc:
        if exc.code == "unknown_case_id":
            raise HTTPException(
                status_code=404,
                detail={"status": exc.code, "error": exc.code},
            ) from exc
        raise HTTPException(
            status_code=400,
            detail={"status": "r1b_replay_validation_failed",
                    "error": exc.code},
        ) from exc
    return result


@router.get("/r1b/shadow-decisions")
async def r1b_shadow_decisions_list(
    limit: int = Query(default=50, ge=1, le=200),
    shape: Optional[str] = Query(default=None),
    benchmark_case_id: Optional[str] = Query(default=None),
    _admin: dict = Depends(require_admin),
):
    """Phase R1B — list recent R1B sidecar rows.

    Closed-set filters (limit / shape / benchmark_case_id). Returns
    most-recent-first ordering. Each row was leak-scanned at write
    time; admin endpoint does no further mutation.
    """
    from services.r1b_shadow_simulator import list_r1b_shadow_decisions
    rows = await list_r1b_shadow_decisions(
        limit=int(limit),
        shape=shape,
        case_id=benchmark_case_id,
    )
    return {
        "rows": rows,
        "count": len(rows),
        "filters": {
            "limit": int(limit),
            "shape": shape,
            "benchmark_case_id": benchmark_case_id,
        },
        "phase": "R1B",
    }


@router.get("/r1b/shadow-decisions/{sidecar_id}")
async def r1b_shadow_decision_by_id(
    sidecar_id: str,
    _admin: dict = Depends(require_admin),
):
    """Phase R1B — fetch a single sidecar by id."""
    from services.r1b_shadow_simulator import get_r1b_shadow_decision_by_id
    row = await get_r1b_shadow_decision_by_id(sidecar_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"status": "r1b_sidecar_not_found",
                    "error": "r1b_sidecar_not_found"},
        )
    return row



# ======================================================================
# Phase R1C — admin-only gold-recall + anchor-specific shadow.
#
# Closed-set. Synchronous. Admin-only. Builds on R1B's expansion
# + connector re-fetch and adds R1C anchor-specific labels + gold
# recovery via approved sources (PubMed + EPMC + Crossref-when-
# wired). NO clinician-visible side effect.
# ======================================================================
class R1CReplayBody(BaseModel):
    """Optional body for ``POST /api/admin/r1c/replay/{case_id}``."""

    search_session_id: Optional[str] = None


@router.post("/r1c/replay/{case_id}")
async def r1c_replay_case(
    case_id: str,
    body: Optional[R1CReplayBody] = None,
    _admin: dict = Depends(require_admin),
):
    """Phase R1C — synchronously replay a gold benchmark case
    through R1B expansion + R1C anchor labels + gold-comparator
    recovery. Returns the sidecar id + compact summary.
    """
    from services.r1c_shadow_simulator import (
        R1CReplayError,
        replay_benchmark_case_r1c,
    )
    actor = str(_admin.get("id") or _admin.get("_id") or "")
    override = (body.search_session_id if body else None)
    try:
        result = await replay_benchmark_case_r1c(
            case_id=case_id,
            actor_user_id=actor or None,
            search_session_id_override=override,
        )
    except R1CReplayError as exc:
        if exc.code == "unknown_case_id":
            raise HTTPException(
                status_code=404,
                detail={"status": exc.code, "error": exc.code},
            ) from exc
        raise HTTPException(
            status_code=400,
            detail={"status": "r1c_replay_validation_failed",
                    "error": exc.code},
        ) from exc
    return result


@router.get("/r1c/shadow-decisions")
async def r1c_shadow_decisions_list(
    limit: int = Query(default=50, ge=1, le=200),
    shape: Optional[str] = Query(default=None),
    benchmark_case_id: Optional[str] = Query(default=None),
    _admin: dict = Depends(require_admin),
):
    """Phase R1C — list recent sidecar rows. Closed-set filters."""
    from services.r1c_shadow_simulator import list_r1c_shadow_decisions
    rows = await list_r1c_shadow_decisions(
        limit=int(limit), shape=shape, case_id=benchmark_case_id,
    )
    return {
        "rows": rows, "count": len(rows),
        "filters": {
            "limit": int(limit), "shape": shape,
            "benchmark_case_id": benchmark_case_id,
        },
        "phase": "R1C",
    }


@router.get("/r1c/shadow-decisions/{sidecar_id}")
async def r1c_shadow_decision_by_id(
    sidecar_id: str,
    _admin: dict = Depends(require_admin),
):
    from services.r1c_shadow_simulator import get_r1c_shadow_decision_by_id
    row = await get_r1c_shadow_decision_by_id(sidecar_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"status": "r1c_sidecar_not_found",
                    "error": "r1c_sidecar_not_found"},
        )
    return row
