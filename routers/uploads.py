"""Uploads router — Phase 2D.

POST /api/search-sessions/{session_id}/articles/{article_id}/upload-full-text
POST /api/uploads/{upload_id}/confirm-match
GET  /api/search-sessions/{session_id}/uploads
DELETE /api/uploads/{upload_id}
GET  /api/search-sessions/{session_id}/evidence-extracts
POST /api/search-sessions/{session_id}/prepare-evidence-packet
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile

from core.db import Collections, get_db
from core.security import get_current_user
from services.audit import audit_service
from services.evidence_extraction import evidence_extraction
from services.uploads import RIGHTS_ATTESTATION_VERSION, USER_MESSAGES
from services.user_uploaded_full_text import user_uploaded_full_text

logger = logging.getLogger("clinevidence.uploads.router")

router = APIRouter(tags=["uploads"])

DISALLOWED_ARTICLE_FLAGS = ("is_preprint", "is_retracted", "is_retraction_notice")


def _feature_enabled() -> bool:
    return os.environ.get("ENABLE_USER_PDF_UPLOADS", "true").lower() != "false"


def _max_upload_mb() -> int:
    try:
        return int(os.environ.get("MAX_UPLOAD_MB", "25"))
    except Exception:  # noqa: BLE001
        return 25


async def _get_owned_session(session_id: str, user_id: str) -> dict:
    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "user_id": user_id, "deleted_at": None},
        {"_id": 0},
    )
    if not doc or doc.get("status") == "blocked_phi":
        raise HTTPException(status_code=404, detail="Search session not found.")
    return doc


async def _get_article_in_session(session_id: str, article_id: str) -> dict:
    db = get_db()
    sa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"search_session_id": session_id, "article_id": article_id},
        {"_id": 0, "exclusion_reason": 1, "included_by_default": 1, "user_included": 1},
    )
    if not sa:
        raise HTTPException(status_code=404, detail="Article not found in session.")
    art = await db[Collections.ARTICLES].find_one({"id": article_id}, {"_id": 0})
    if not art:
        raise HTTPException(status_code=404, detail="Article not found in session.")
    return {**art, "_sa": sa}


def _is_excluded_article(article: dict) -> bool:
    if any(article.get(flag) for flag in DISALLOWED_ARTICLE_FLAGS):
        return True
    sa = article.get("_sa") or {}
    reason = sa.get("exclusion_reason")
    if reason in {"preprint", "retracted", "retraction_notice"}:
        return True
    return False


# ============================================================================
#  POST /api/search-sessions/{session_id}/articles/{article_id}/upload-full-text
# ============================================================================
@router.post("/search-sessions/{session_id}/articles/{article_id}/upload-full-text")
async def upload_full_text(
    session_id: str,
    article_id: str,
    file: UploadFile = File(...),
    retention_mode: str = Form("use_once"),
    rights_attested: str = Form("false"),
    rights_attestation_version: str = Form(RIGHTS_ATTESTATION_VERSION),
    current_user: dict = Depends(get_current_user),
):
    if not _feature_enabled():
        raise HTTPException(status_code=403, detail="PDF uploads are disabled by administrator.")

    # Phase 4B — malware-scanner gate. If scanner is enabled but unconfigured
    # AND uploads-without-AV are disallowed, refuse the upload upfront.
    from services.malware_scan import malware_scan_service
    allowed, _code, gate_msg = malware_scan_service.uploads_allowed()
    if not allowed:
        raise HTTPException(status_code=503, detail=gate_msg or "Uploads temporarily disabled.")

    # Retention validation (saved_private explicitly rejected in Phase 2D).
    if retention_mode == "saved_private":
        raise HTTPException(status_code=400, detail="saved_private retention is not yet enabled.")
    if retention_mode not in {"use_once", "session_30_days"}:
        raise HTTPException(status_code=400, detail=USER_MESSAGES["retention_invalid"])

    # Rights attestation (422 to signal missing-required-field).
    rights_truthy = str(rights_attested).strip().lower() in ("true", "1", "yes", "on")
    if not rights_truthy:
        raise HTTPException(status_code=422, detail=USER_MESSAGES["attestation_required"])

    # Content-type + extension preflight.
    if (file.content_type or "").lower() != "application/pdf":
        raise HTTPException(status_code=400, detail=USER_MESSAGES["not_a_pdf"])
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=USER_MESSAGES["not_a_pdf"])

    # Ownership + article presence.
    session = await _get_owned_session(session_id, current_user["id"])
    article = await _get_article_in_session(session_id, article_id)
    if _is_excluded_article(article):
        raise HTTPException(
            status_code=400,
            detail="Cannot upload full text for excluded/retracted articles.",
        )

    # Read body.
    max_bytes = _max_upload_mb() * 1024 * 1024
    body = await file.read()
    if len(body) > max_bytes:
        raise HTTPException(status_code=413, detail="PDF exceeds the maximum allowed size.")
    if len(body) == 0:
        raise HTTPException(status_code=400, detail=USER_MESSAGES["not_a_pdf"])

    result = await user_uploaded_full_text.process(
        file_bytes=body,
        filename=filename,
        user_id=current_user["id"],
        search_session_id=session_id,
        article=article,
        session=session,
        retention_mode=retention_mode,
        rights_attested=True,
        rights_attestation_version=rights_attestation_version or RIGHTS_ATTESTATION_VERSION,
    )
    return result


# ============================================================================
#  POST /api/uploads/{upload_id}/confirm-match
# ============================================================================
@router.post("/uploads/{upload_id}/confirm-match")
async def confirm_match(
    upload_id: str,
    payload: dict = Body(...),
    current_user: dict = Depends(get_current_user),
):
    confirmed = bool(payload.get("confirmed"))
    result = await user_uploaded_full_text.confirm_match(upload_id, current_user["id"], confirmed)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Upload not found.")
    if result["status"] == "invalid_state":
        raise HTTPException(
            status_code=409,
            detail="Confirm/reject is only valid for moderate or low match confidence.",
        )
    return result


# ============================================================================
#  GET /api/search-sessions/{session_id}/uploads
# ============================================================================
@router.get("/search-sessions/{session_id}/uploads")
async def list_session_uploads(
    session_id: str, current_user: dict = Depends(get_current_user)
):
    await _get_owned_session(session_id, current_user["id"])
    db = get_db()
    rows = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find(
        {
            "search_session_id": session_id,
            "user_id": current_user["id"],
            "deleted_at": None,
        },
        {
            "_id": 0,
            "id": 1, "article_id": 1, "original_filename": 1, "file_size": 1,
            "page_count": 1, "upload_status": 1, "match_confidence": 1,
            "matched_by": 1, "extracted_text_status": 1, "retention_mode": 1,
            "user_confirmation_required": 1, "user_confirmed_match_at": 1,
            "storage_deleted_at": 1, "evidence_text_level": 1,
            "processing_error_code": 1, "processing_error_message_safe": 1,
            "created_at": 1, "updated_at": 1,
        },
    ).sort("created_at", -1).to_list(length=200)
    return rows


# ============================================================================
#  DELETE /api/uploads/{upload_id}
# ============================================================================
@router.delete("/uploads/{upload_id}")
async def delete_upload(
    upload_id: str, current_user: dict = Depends(get_current_user)
):
    ok = await user_uploaded_full_text.delete_upload(upload_id, current_user["id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Upload not found.")
    return {"status": "deleted"}


# ============================================================================
#  GET /api/search-sessions/{session_id}/evidence-extracts
# ============================================================================
@router.get("/search-sessions/{session_id}/evidence-extracts")
async def evidence_extracts(
    session_id: str, current_user: dict = Depends(get_current_user)
):
    await _get_owned_session(session_id, current_user["id"])
    db = get_db()
    sa_rows = await db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": session_id},
        {
            "_id": 0,
            "article_id": 1,
            "included_by_default": 1,
            "user_included": 1,
            "exclusion_reason": 1,
        },
    ).to_list(length=500)
    selected_ids = [
        r["article_id"] for r in sa_rows
        if (r.get("included_by_default") or r.get("user_included")) and not r.get("exclusion_reason")
    ]
    articles_by_id: dict[str, dict] = {}
    if selected_ids:
        async for art in db[Collections.ARTICLES].find(
            {"id": {"$in": selected_ids}},
            {"_id": 0, "id": 1, "title": 1, "abstract": 1},
        ):
            articles_by_id[art["id"]] = art

    uploads = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find(
        {
            "search_session_id": session_id,
            "user_id": current_user["id"],
            "deleted_at": None,
        },
        {"_id": 0, "id": 1, "article_id": 1, "upload_status": 1, "user_confirmation_required": 1},
    ).to_list(length=500)
    upload_status_by_aid: dict[str, str] = {}
    for u in uploads:
        key = u["article_id"]
        existing = upload_status_by_aid.get(key)
        status = u["upload_status"]
        # Prefer most "ready" status: processed > matched > pending > blocked_phi > deleted
        order = {"processed": 5, "matched": 4, "pending_review": 3, "blocked_phi": 2, "mismatch": 1, "deleted": 0, "error": 0}
        if existing is None or order.get(status, 0) > order.get(existing, 0):
            upload_status_by_aid[key] = status

    out = []
    for aid in selected_ids:
        art = articles_by_id.get(aid)
        if not art:
            continue
        # Gather extracts (user_uploaded preferred over abstract).
        extracts = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
            {
                "search_session_id": session_id,
                "article_id": aid,
                "$or": [
                    {"text_source": "user_uploaded_full_text", "user_id": current_user["id"]},
                    {"text_source": "abstract"},
                ],
            },
            {"_id": 0},
        ).to_list(length=200)
        if any(e["text_source"] == "user_uploaded_full_text" for e in extracts):
            extracts = [e for e in extracts if e["text_source"] == "user_uploaded_full_text"]
            evidence_level = "user_uploaded_full_text"
        elif extracts:
            evidence_level = "abstract"
        else:
            evidence_level = "none"
        sections = [
            {
                "section_name": e["section_name"],
                "token_count": e.get("token_count"),
                "include_in_summary": bool(e.get("include_in_summary")),
                "preview": (e.get("section_text") or "")[:200],
            }
            for e in extracts
        ]
        total_tokens = sum(s.get("token_count") or 0 for s in sections)
        ups = upload_status_by_aid.get(aid)
        out.append({
            "article_id": aid,
            "title": art.get("title"),
            "selected": True,
            "evidence_text_level": evidence_level,
            "sections": sections,
            "total_tokens": total_tokens,
            "user_upload_status": (
                "needs_confirmation" if ups == "matched"
                else "blocked" if ups == "blocked_phi"
                else ups or "none"
            ),
        })
    return {"articles": out}


# ============================================================================
#  POST /api/search-sessions/{session_id}/prepare-evidence-packet
# ============================================================================
@router.post("/search-sessions/{session_id}/prepare-evidence-packet")
async def prepare_evidence_packet(
    session_id: str, current_user: dict = Depends(get_current_user)
):
    session = await _get_owned_session(session_id, current_user["id"])
    summary = await evidence_extraction.prepare_evidence_packet(
        session=session, user_id=current_user["id"]
    )
    # Phase 6B.2 — persist a sanitized ``evidence_packet_summary`` on
    # the session document so the admin + clinician UIs (and the new
    # ``test_evidence_packet_summary_persisted_with_all_fields`` test)
    # can read it without re-running prepare. We also compute the
    # directness counts + override caveats from the session's
    # search_session_articles + selected_studies cohort here so the
    # /prepare-evidence-packet endpoint stays the single source of
    # truth even when the summary router calls a richer build.
    try:
        from core.db import Collections as _C, get_db as _gd
        db = _gd()
        sa_rows = await db[_C.SEARCH_SESSION_ARTICLES].find(
            {"search_session_id": session_id, "user_included": True,
             "exclusion_reason": None},
            {"_id": 0, "article_id": 1, "cap_reasons": 1,
             "included_by_default": 1, "user_included": 1},
        ).to_list(length=500)
        direct_count = 0
        indirect_override_count = 0
        selected_articles_by_subquestion: dict[str, list[str]] = {"primary": []}
        for row in sa_rows:
            aid = row.get("article_id") or ""
            if aid:
                selected_articles_by_subquestion["primary"].append(aid)
            cap_reasons = list(row.get("cap_reasons") or [])
            if cap_reasons:
                indirect_override_count += 1
            else:
                direct_count += 1
        epk_summary = {
            "directly_relevant_count": direct_count,
            "indirect_override_count": indirect_override_count,
            "subquestion_coverage": {"primary": len(sa_rows)},
            "selected_articles_by_subquestion": selected_articles_by_subquestion,
            "abstract_only_count": int(summary.get("abstract_only_count") or 0),
            "oa_full_text_count": int(summary.get("oa_full_text_count") or 0),
            "user_uploaded_full_text_count":
                int(summary.get("user_uploaded_full_text_count") or 0),
        }
        from datetime import datetime, timezone
        await db[_C.SEARCH_SESSIONS].update_one(
            {"id": session_id},
            {"$set": {
                "evidence_packet_summary": epk_summary,
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        summary["evidence_packet_summary"] = epk_summary
    except Exception:  # noqa: BLE001
        logger.exception(
            "phase6b2 evidence_packet_summary persistence failed (non-fatal)",
        )
    await audit_service.log(
        "evidence_packet_prepared",
        user_id=current_user["id"],
        search_session_id=session_id,
        payload={"counts": {k: v for k, v in summary.items() if isinstance(v, int)}},
    )
    return summary
