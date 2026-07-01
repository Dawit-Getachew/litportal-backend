"""OA fetcher router — Phase 4A.

Endpoints (all `/api`):
  POST /search-sessions/{sid}/articles/{aid}/fetch-oa-full-text
  POST /search-sessions/{sid}/fetch-oa-full-text
  GET  /search-sessions/{sid}/oa-fetch-status
  GET  /admin/oa-fetcher-metrics
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from core.db import Collections, get_db
from core.security import get_current_user, require_admin
from services.oa_fetcher import oa_fetcher

router = APIRouter(tags=["oa-fetcher"])


def _env_flag(key: str, default: bool) -> bool:
    raw = (os.environ.get(key) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


async def _owned_session(session_id: str, user_id: str) -> dict:
    db = get_db()
    s = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "user_id": user_id, "deleted_at": None},
        {"_id": 0},
    )
    if not s:
        raise HTTPException(status_code=404, detail="Search session not found.")
    return s


async def _ensure_article_in_session(session_id: str, article_id: str) -> None:
    db = get_db()
    sa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"search_session_id": session_id, "article_id": article_id},
        {"_id": 0, "id": 1},
    )
    if not sa:
        raise HTTPException(status_code=404, detail="Article not in this search session.")


class FetchOneBody(BaseModel):
    force_refresh: bool = False


class FetchManyBody(BaseModel):
    article_ids: Optional[list[str]] = None
    force_refresh: bool = False


class RetryOABody(BaseModel):
    force_refresh: bool = True


def _disabled_payload() -> dict:
    return {
        "status": "disabled",
        "source": None, "license": None, "license_category": None,
        "can_ingest_for_summary": False, "sections_created": 0,
        "token_count": 0, "evidence_text_level": None,
        "error_code": "disabled",
    }


@router.post("/search-sessions/{session_id}/articles/{article_id}/fetch-oa-full-text")
async def fetch_one(
    session_id: str,
    article_id: str,
    body: FetchOneBody = Body(default_factory=FetchOneBody),
    current_user: dict = Depends(get_current_user),
):
    if not _env_flag("OA_FETCHER_ENABLED", True):
        return _disabled_payload()
    await _owned_session(session_id, current_user["id"])
    await _ensure_article_in_session(session_id, article_id)
    return await oa_fetcher.fetch_for_article(
        article_id=article_id, session_id=session_id, user_id=current_user["id"],
        force_refresh=body.force_refresh,
    )


@router.post("/search-sessions/{session_id}/articles/{article_id}/retry-oa-fetch")
async def retry_oa_fetch(
    session_id: str,
    article_id: str,
    body: RetryOABody = Body(default_factory=RetryOABody),
    current_user: dict = Depends(get_current_user),
):
    """Phase 4B — retry a failed/transient OA fetch with rate limiting.

    Enforces:
      - per-(user, session, article) attempt counter (OA_RETRY_MAX_ATTEMPTS)
      - per-(user, session, article) minimum interval (OA_RETRY_MIN_INTERVAL_SECONDS)
      - Refuses retries when the previous result was a *terminal* skip
        (article_retracted, article_is_preprint, license_*) — these will not
        change on retry.
    Returns the same per-article result shape as fetch_one, or a structured
    rate-limit/terminal response with `error_code`.
    """
    if not _env_flag("OA_FETCHER_ENABLED", True):
        return _disabled_payload()
    await _owned_session(session_id, current_user["id"])
    await _ensure_article_in_session(session_id, article_id)

    db = get_db()
    now = datetime.now(timezone.utc)
    max_attempts = max(1, int(os.environ.get("OA_RETRY_MAX_ATTEMPTS", "3") or "3"))
    min_interval_s = max(0, int(os.environ.get("OA_RETRY_MIN_INTERVAL_SECONDS", "10") or "10"))

    # Look up the existing access-location row to count prior attempts +
    # determine whether the previous outcome was terminal.
    loc = await db[Collections.ARTICLE_ACCESS_LOCATIONS].find_one(
        {"article_id": article_id, "search_session_id": session_id,
         "access_category": {"$in": ["pmc_oa", "europe_pmc_oa"]}},
        {"_id": 0},
    )
    TERMINAL_CODES = {
        "article_retracted", "article_retraction_notice", "article_is_preprint",
        "license_noncommercial", "license_restricted", "license_unknown",
        "license_skipped",
    }
    prior_code = (loc or {}).get("oa_fetch_error_code")
    if prior_code in TERMINAL_CODES:
        return {
            "article_id": article_id,
            "status": "terminal",
            "error_code": prior_code,
            "message_safe": (
                "This article cannot be retrieved automatically. Upload the "
                "full text manually if you have rights to use it."
            ),
        }

    attempt_count = int((loc or {}).get("oa_retry_attempt_count") or 0)
    last_retry_at = (loc or {}).get("oa_last_retry_at")
    if attempt_count >= max_attempts:
        return {
            "article_id": article_id,
            "status": "rate_limited",
            "error_code": "max_retries_exceeded",
            "attempts": attempt_count,
            "max_attempts": max_attempts,
            "message_safe": (
                "Maximum automatic retries reached. Upload the full text "
                "manually if you have rights to use it."
            ),
        }
    if last_retry_at:
        # Mongo can return naive datetimes; normalize to UTC.
        if last_retry_at.tzinfo is None:
            last_retry_at = last_retry_at.replace(tzinfo=timezone.utc)
        elapsed = (now - last_retry_at).total_seconds()
        if elapsed < min_interval_s:
            return {
                "article_id": article_id,
                "status": "rate_limited",
                "error_code": "retry_too_soon",
                "retry_after_seconds": int(min_interval_s - elapsed),
                "message_safe": "Please wait before retrying this fetch.",
            }

    # Bump attempt counter BEFORE the fetch so concurrent retries are tracked.
    if loc:
        await db[Collections.ARTICLE_ACCESS_LOCATIONS].update_one(
            {"id": loc["id"]},
            {"$set": {"oa_last_retry_at": now,
                       "oa_retry_attempt_count": attempt_count + 1,
                       "updated_at": now}},
        )

    from services.audit import audit_service
    await audit_service.log(
        "oa_fetch_retry",
        user_id=current_user["id"],
        search_session_id=session_id,
        payload={"article_id": article_id, "attempt": attempt_count + 1},
    )
    result = await oa_fetcher.fetch_for_article(
        article_id=article_id, session_id=session_id, user_id=current_user["id"],
        force_refresh=body.force_refresh,
    )
    result["attempts"] = attempt_count + 1
    result["max_attempts"] = max_attempts
    return result


@router.post("/search-sessions/{session_id}/fetch-oa-full-text")
async def fetch_many(
    session_id: str,
    body: FetchManyBody = Body(default_factory=FetchManyBody),
    current_user: dict = Depends(get_current_user),
):
    if not _env_flag("OA_FETCHER_ENABLED", True):
        return {"results": [], "summary": {
            "fetched": 0, "cache_hits": 0, "failed": 0,
            "skipped_ineligible": 0, "skipped_license": 0,
            "skipped_cap_exceeded": 0, "disabled": 1,
        }, "status": "disabled"}
    await _owned_session(session_id, current_user["id"])

    db = get_db()
    if body.article_ids is None:
        rows = await db[Collections.SEARCH_SESSION_ARTICLES].find(
            {"search_session_id": session_id, "user_included": True, "exclusion_reason": None},
            {"_id": 0, "article_id": 1},
        ).to_list(length=200)
        article_ids = [r["article_id"] for r in rows]
    else:
        article_ids = list(body.article_ids)

    return await oa_fetcher.fetch_for_articles(
        article_ids=article_ids, session_id=session_id, user_id=current_user["id"],
        force_refresh=body.force_refresh,
    )


@router.get("/search-sessions/{session_id}/oa-fetch-status")
async def status(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    await _owned_session(session_id, current_user["id"])

    sa_rows = await db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": session_id},
        {"_id": 0, "article_id": 1},
    ).to_list(length=200)
    aids = [r["article_id"] for r in sa_rows]
    if not aids:
        return {"articles": []}
    arts = {a["id"]: a async for a in db[Collections.ARTICLES].find(
        {"id": {"$in": aids}}, {"_id": 0}
    )}
    extracts = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
        {
            "search_session_id": session_id,
            "text_source": "oa_full_text",
        },
        {"_id": 0},
    ).to_list(length=400)
    sec_count_by_aid: dict[str, int] = {}
    last_fetched_by_aid: dict[str, Optional[datetime]] = {}
    for e in extracts:
        sec_count_by_aid[e["article_id"]] = sec_count_by_aid.get(e["article_id"], 0) + 1
        prev = last_fetched_by_aid.get(e["article_id"])
        cur = e.get("oa_last_fetched_at")
        if cur and (prev is None or cur > prev):
            last_fetched_by_aid[e["article_id"]] = cur
    locs = await db[Collections.ARTICLE_ACCESS_LOCATIONS].find(
        {"search_session_id": session_id, "access_category": {"$in": ["pmc_oa", "europe_pmc_oa"]}},
        {"_id": 0},
    ).to_list(length=400)
    loc_by_aid: dict[str, dict] = {}
    for loc in locs:
        loc_by_aid.setdefault(loc["article_id"], loc)
    out = []
    for aid in aids:
        art = arts.get(aid) or {}
        loc = loc_by_aid.get(aid)
        from services.oa_fetcher.license_gate import is_article_eligible
        eligible, reason = is_article_eligible(art)
        sec_count = sec_count_by_aid.get(aid, 0)
        if sec_count:
            fetch_status = "fetched"
            evidence_level = "oa_full_text"
        elif loc:
            fetch_status = loc.get("oa_fetch_status") or "not_attempted"
            evidence_level = "oa_full_text" if loc.get("can_ingest_for_summary") else "abstract"
        else:
            fetch_status = "not_attempted" if eligible else "skipped_ineligible"
            evidence_level = "abstract" if art.get("abstract") else "metadata_only"
        out.append({
            "article_id": aid,
            "title": art.get("title") or "",
            "eligible": eligible,
            "fetch_status": fetch_status,
            "license": (loc or {}).get("license"),
            "license_category": (loc or {}).get("license_category"),
            "evidence_text_level": evidence_level,
            "sections_created": sec_count,
            "last_fetched_at": last_fetched_by_aid.get(aid).isoformat() if last_fetched_by_aid.get(aid) else None,
            "warning_code": (loc or {}).get("oa_fetch_error_code") if not (loc or {}).get("can_ingest_for_summary") else None,
        })
    return {"articles": out}


# ---------------- admin metrics endpoint -------------------------------------


@router.get("/admin/oa-fetcher-metrics", dependencies=[Depends(require_admin)])
async def admin_oa_metrics():
    db = get_db()
    counts = {}
    for evt in (
        "oa_fetch_attempt", "oa_fetch_success", "oa_fetch_failed",
        "oa_fetch_cache_hit", "oa_fetch_skipped_ineligible",
        "oa_fetch_skipped_license",
    ):
        counts[evt] = await db[Collections.AUDIT_EVENTS].count_documents({"event_type": evt})
    # license-skip subcategories.
    skipped_noncomm = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "oa_fetch_skipped_license", "event_payload_json.error_code": "license_noncommercial"}
    )
    skipped_unknown = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "oa_fetch_skipped_license", "event_payload_json.error_code": "license_unknown"}
    )
    skipped_no_deriv = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "oa_fetch_skipped_license", "event_payload_json.error_code": "license_restricted"}
    )
    # avg fetch latency from oa_fetch_success payloads.
    avg_latency = None
    pipeline = [
        {"$match": {"event_type": "oa_fetch_success"}},
        {"$group": {"_id": None, "avg": {"$avg": "$event_payload_json.latency_ms"}}},
    ]
    async for row in db[Collections.AUDIT_EVENTS].aggregate(pipeline):
        if row.get("avg") is not None:
            avg_latency = round(float(row["avg"]), 1)
    # Source status from source_configs.
    epmc_cfg = await db[Collections.SOURCE_CONFIGS].find_one(
        {"source_name": "europe_pmc_oa_fetcher"}, {"_id": 0}
    )
    pmc_cfg = await db[Collections.SOURCE_CONFIGS].find_one(
        {"source_name": "pmc_oa_fetcher"}, {"_id": 0}
    )

    def _status(row: Optional[dict]) -> str:
        if not row:
            return "skipped"
        if row.get("last_error_at") and (
            not row.get("last_success_at") or row["last_error_at"] > row["last_success_at"]
        ):
            return "error"
        return "ok" if row.get("last_success_at") else "skipped"

    # articles_upgraded_abstract_to_oa: distinct articles where there is now
    # at least one OA cache row OR an OA session row.
    upgraded = len(await db[Collections.ARTICLE_TEXT_EXTRACTS].distinct(
        "article_id", {"text_source": "oa_full_text"}
    ))
    return {
        "fetch_attempts": counts["oa_fetch_attempt"],
        "fetch_successes": counts["oa_fetch_success"],
        "fetch_failures": counts["oa_fetch_failed"],
        "fetch_cache_hits": counts["oa_fetch_cache_hit"],
        "skipped_ineligible": counts["oa_fetch_skipped_ineligible"],
        "skipped_noncommercial": skipped_noncomm,
        "skipped_unknown_license": skipped_unknown,
        "skipped_no_derivatives": skipped_no_deriv,
        "articles_upgraded_abstract_to_oa": upgraded,
        "average_fetch_latency_ms": avg_latency,
        "europe_pmc_oa_status": _status(epmc_cfg),
        "pmc_oa_status": _status(pmc_cfg),
        "fetcher_enabled": _env_flag("OA_FETCHER_ENABLED", True),
    }
