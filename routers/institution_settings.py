"""Institution settings router (per-user upsert).

PHASE 2C HARDENING:
- The Settings page must never ingest credentials. The pydantic upsert model
  enforces structural validation. Additionally, this router runs a defensive
  scan over the raw request body to reject anything that LOOKS like a
  username/password/token/api-key field. Both the field NAME and the field
  VALUE are inspected.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from core.db import Collections, get_db
from core.security import get_current_user
from models.users import InstitutionSettingsDoc, InstitutionSettingsUpsert
from services.audit import audit_service

router = APIRouter(prefix="/user/institution-settings", tags=["institution-settings"])


_CREDENTIAL_FIELD_NAMES_RE = re.compile(
    r"(password|secret|token|api[_-]?key|username|user[_-]?id|auth)",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE_PATTERNS_RE = re.compile(
    r"(password\s*[:=]|api[_-]?key\s*[:=]|secret\s*[:=]|token\s*[:=]|"
    r"username\s*[:=]|bearer\s+\S+|basic\s+[A-Za-z0-9+/=]+)",
    re.IGNORECASE,
)

CREDENTIAL_REJECTION_MESSAGE = (
    "Institutional credentials must never be entered here. "
    "ClinEvidence only stores link-out configuration."
)


def _scan_for_credentials(body: dict) -> bool:
    """Return True if the raw JSON body contains credential-like content."""
    if not isinstance(body, dict):
        return False
    for k, v in body.items():
        if isinstance(k, str) and _CREDENTIAL_FIELD_NAMES_RE.search(k):
            # The field name itself is suspect — reject.
            return True
        if isinstance(v, str) and _CREDENTIAL_VALUE_PATTERNS_RE.search(v):
            return True
    return False


@router.get("")
async def get_my_settings(current_user: dict = Depends(get_current_user)):
    db = get_db()
    doc = await db[Collections.USER_INSTITUTION_SETTINGS].find_one(
        {"user_id": current_user["id"]}, {"_id": 0}
    )
    if doc is None:
        default = InstitutionSettingsDoc(user_id=current_user["id"])
        return default.model_dump()
    return doc


@router.put("")
async def upsert_my_settings(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    # 1) Defensive credential scan against the raw JSON body.
    try:
        raw_body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    if _scan_for_credentials(raw_body or {}):
        await audit_service.log(
            "institution_settings_credential_rejected",
            user_id=current_user["id"],
            payload={"reason": "credential_like_field_or_value"},
        )
        raise HTTPException(status_code=400, detail=CREDENTIAL_REJECTION_MESSAGE)

    # 2) Pydantic validation (URL format, OpenAthens domain shape).
    try:
        body = InstitutionSettingsUpsert(**(raw_body or {}))
    except ValidationError as exc:
        # Bubble up Pydantic's structured validation errors (str-only payload).
        details = [
            {"field": ".".join(str(p) for p in (e.get("loc") or [])), "message": e.get("msg")}
            for e in exc.errors()
        ]
        raise HTTPException(status_code=422, detail=details)

    db = get_db()
    now = datetime.now(timezone.utc)
    payload = body.model_dump()
    payload["updated_at"] = now
    existing = await db[Collections.USER_INSTITUTION_SETTINGS].find_one(
        {"user_id": current_user["id"]}, {"_id": 0}
    )
    if existing:
        await db[Collections.USER_INSTITUTION_SETTINGS].update_one(
            {"user_id": current_user["id"]}, {"$set": payload}
        )
    else:
        doc = InstitutionSettingsDoc(user_id=current_user["id"], **payload)
        new_doc = doc.model_dump()
        for k in ("created_at", "updated_at"):
            new_doc[k] = new_doc[k].astimezone(timezone.utc)
        await db[Collections.USER_INSTITUTION_SETTINGS].insert_one(new_doc)
    await audit_service.log(
        "institution_settings_updated",
        user_id=current_user["id"],
        payload={
            "openathens_set": bool(payload.get("openathens_customer_domain")),
            "openurl_set": bool(payload.get("openurl_resolver_base_url")),
            "ss_fallback_enabled": bool(payload.get("semantic_scholar_fallback_enabled")),
        },
    )
    fresh = await db[Collections.USER_INSTITUTION_SETTINGS].find_one(
        {"user_id": current_user["id"]}, {"_id": 0}
    )
    return fresh


@router.delete("")
async def clear_my_settings(current_user: dict = Depends(get_current_user)):
    db = get_db()
    await db[Collections.USER_INSTITUTION_SETTINGS].delete_many(
        {"user_id": current_user["id"]}
    )
    await audit_service.log(
        "institution_settings_cleared",
        user_id=current_user["id"],
    )
    default = InstitutionSettingsDoc(user_id=current_user["id"])
    return default.model_dump()
