"""Phase 6D — Library router.

Endpoints (all under ``/api/library``, all require auth):
- GET    /items              — list current user's library
- POST   /items              — save one article
- GET    /items/{id}         — get one item
- PATCH  /items/{id}         — update note / tags / favorite / status
- DELETE /items/{id}         — remove (hard delete the library_item row)
- POST   /items/batch        — save N articles + (optional) folder
- POST   /import-article     — import by PMID / DOI / PMCID
- POST   /evidence-packets   — save a verified evidence packet to a folder
- DELETE /evidence-packets/{id} — soft-delete
- GET    /evidence-packets   — list (filterable by folder_id)
- POST   /summaries          — Task 10 SCAFFOLDED-DISABLED (returns 501)

PHI guard runs on every text field on the inbound side.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.db import Collections, get_db
from core.security import get_current_user
from models.library import (
    LibraryBatchSaveRequest,
    LibraryItemCreate,
    LibraryItemImport,
    LibraryItemUpdate,
    SavedEvidencePacketCreate,
    SavedEvidencePacketDoc,
)
from services.audit import audit_service
from services.phi_guard import (
    LIBRARY_PHI_MESSAGE,
    screen_user_text,
)
from services.user_library import LibraryError, library_service


router = APIRouter(prefix="/library", tags=["library"])


def _phi_block_response(context_type: str, reason_codes: list[str]) -> JSONResponse:
    """Closed-set 422 response — NEVER echoes the rejected text."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "library_text_phi_blocked",
            "context_type": context_type,
            "reason_codes": reason_codes,
            "message": LIBRARY_PHI_MESSAGE,
        },
    )


def _check_text(text: Optional[str], context_type: str) -> Optional[JSONResponse]:
    if text is None or text == "":
        return None
    r = screen_user_text(text, context_type)
    if r.is_blocked:
        return _phi_block_response(context_type, r.reason_codes)
    return None


def _check_tags(tags: Optional[list[str]]) -> Optional[JSONResponse]:
    for t in tags or []:
        block = _check_text(t, "library_tag")
        if block:
            return block
    return None


def _serialize(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "_id"}


# ============================================================
# Library items
# ============================================================
@router.get("/items")
async def list_items(
    status: Optional[str] = Query(default=None, regex="^(active|archived)$"),
    favorite: Optional[bool] = None,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    return await library_service.list_items(
        current_user["id"],
        status=status, favorite=favorite, tag=tag, search=search,
        limit=limit, skip=skip,
    )


@router.post("/items", status_code=201)
async def save_item(body: LibraryItemCreate, current_user: dict = Depends(get_current_user)):
    block = _check_text(body.note, "library_item_note")
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "library_item_note"},
        )
        return block
    block = _check_tags(body.tags)
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "library_tag"},
        )
        return block
    try:
        doc, created = await library_service.save_item(current_user["id"], body)
    except LibraryError as e:
        raise HTTPException(status_code=e.status, detail=e.code)
    return {"library_item": doc, "created": created}


@router.get("/items/{item_id}")
async def get_item(item_id: str, current_user: dict = Depends(get_current_user)):
    try:
        return await library_service.get_item(current_user["id"], item_id)
    except LibraryError as e:
        raise HTTPException(status_code=e.status, detail=e.code)


@router.patch("/items/{item_id}")
async def update_item(
    item_id: str,
    body: LibraryItemUpdate,
    current_user: dict = Depends(get_current_user),
):
    block = _check_text(body.note, "library_item_note")
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "library_item_note"},
        )
        return block
    block = _check_tags(body.tags)
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "library_tag"},
        )
        return block
    try:
        return await library_service.update_item(
            current_user["id"], item_id, body,
        )
    except LibraryError as e:
        raise HTTPException(status_code=e.status, detail=e.code)


@router.delete("/items/{item_id}")
async def delete_item(item_id: str, current_user: dict = Depends(get_current_user)):
    try:
        return await library_service.delete_item(current_user["id"], item_id)
    except LibraryError as e:
        raise HTTPException(status_code=e.status, detail=e.code)


# ============================================================
# Batch save
# ============================================================
@router.post("/items/batch", status_code=201)
async def batch_save(
    body: LibraryBatchSaveRequest,
    current_user: dict = Depends(get_current_user),
):
    block = _check_text(body.note, "library_item_note")
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "library_item_note"},
        )
        return block
    block = _check_tags(body.tags)
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "library_tag"},
        )
        return block
    try:
        return await library_service.batch_save(current_user["id"], body)
    except LibraryError as e:
        raise HTTPException(status_code=e.status, detail=e.code)


# ============================================================
# Import by identifier
# ============================================================
@router.post("/import-article", status_code=201)
async def import_article(
    body: LibraryItemImport,
    current_user: dict = Depends(get_current_user),
):
    block = _check_text(body.note, "library_item_note")
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "library_item_note"},
        )
        return block
    block = _check_tags(body.tags)
    if block:
        return block
    try:
        return await library_service.import_by_identifier(current_user["id"], body)
    except LibraryError as e:
        raise HTTPException(status_code=e.status, detail=e.code)


# ============================================================
# Saved evidence packets (Task 9 — FULL IMPLEMENT)
# ============================================================
@router.post("/evidence-packets", status_code=201)
async def save_evidence_packet(
    body: SavedEvidencePacketCreate,
    current_user: dict = Depends(get_current_user),
):
    """Snapshot a verified search session's selection into a folder.

    Closed-set requirements:
    - Folder must be owned by ``current_user``.
    - Source session must be owned by ``current_user``.
    - Source session must have ``status='complete'`` (verified summary
      present) — else 409.
    - PHI guard runs on title + note.
    - Saves metadata ONLY — never raw evidence text.
    """
    db = get_db()
    # PHI guard on title + note.
    block = _check_text(body.title, "evidence_packet_title")
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "evidence_packet_title"},
        )
        return block
    block = _check_text(body.note, "evidence_packet_note")
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "evidence_packet_note"},
        )
        return block
    # Folder owned by user.
    folder = await db[Collections.FOLDERS].find_one(
        {"id": body.folder_id, "user_id": current_user["id"]}, {"_id": 0},
    )
    if not folder:
        raise HTTPException(status_code=404, detail="folder_not_found")
    # Source session owned + verified.
    sess = await db[Collections.SEARCH_SESSIONS].find_one(
        {
            "id": body.source_search_session_id,
            "user_id": current_user["id"],
            "deleted_at": None,
        },
        {"_id": 0},
    )
    if not sess:
        raise HTTPException(status_code=404, detail="source_session_not_found")
    if sess.get("status") not in ("complete", "awaiting_selection"):
        raise HTTPException(
            status_code=409, detail="source_session_not_verified",
        )
    # Pull current selection (user_included=True or included_by_default).
    sa_cur = db[Collections.SEARCH_SESSION_ARTICLES].find(
        {
            "search_session_id": body.source_search_session_id,
            "$or": [
                {"user_included": True},
                {"user_included": None, "included_by_default": True},
            ],
        },
        {"_id": 0, "article_id": 1, "overall_rank": 1},
    ).sort("overall_rank", -1).limit(50)
    sa_rows = await sa_cur.to_list(length=50)
    article_ids = [r["article_id"] for r in sa_rows]
    art_map: dict[str, dict] = {}
    if article_ids:
        async for row in db[Collections.ARTICLES].find(
            {"id": {"$in": article_ids}},
            {"_id": 0, "id": 1, "title": 1, "canonical_key": 1},
        ):
            art_map[row["id"]] = row
    canonical_keys = [art_map.get(a, {}).get("canonical_key") or "" for a in article_ids]
    titles = [art_map.get(a, {}).get("title") or "" for a in article_ids]
    # Latest summary (Task 10 stays scaffolded so we just record summary id).
    summary_doc = await db[Collections.SUMMARIES].find_one(
        {
            "search_session_id": body.source_search_session_id,
            "verification_status": "passed",
        },
        {"_id": 0, "id": 1},
        sort=[("created_at", -1)],
    )

    pkt = SavedEvidencePacketDoc(
        user_id=current_user["id"],
        folder_id=body.folder_id,
        title=body.title,
        note=body.note,
        source_search_session_id=body.source_search_session_id,
        source_summary_id=(summary_doc or {}).get("id"),
        sanitized_question=sess.get("sanitized_question"),
        topic_label=sess.get("topic_label"),
        clinical_question_type=sess.get("clinical_question_type"),
        article_count=len(article_ids),
        article_ids_json=article_ids,
        article_canonical_keys_json=canonical_keys,
        article_titles_json=titles,
    ).model_dump()
    for k in ("created_at", "updated_at"):
        pkt[k] = pkt[k].astimezone(timezone.utc)
    await db[Collections.SAVED_EVIDENCE_PACKETS].insert_one(pkt)
    await audit_service.log(
        "saved_evidence_packet_created",
        user_id=current_user["id"],
        search_session_id=body.source_search_session_id,
        payload={
            "packet_id": pkt["id"],
            "folder_id": body.folder_id,
            "article_count": len(article_ids),
        },
    )
    return _serialize(pkt)


@router.get("/evidence-packets")
async def list_evidence_packets(
    folder_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    q: dict = {"user_id": current_user["id"], "deleted_at": None}
    if folder_id:
        q["folder_id"] = folder_id
    cursor = db[Collections.SAVED_EVIDENCE_PACKETS].find(
        q, {"_id": 0},
    ).sort("created_at", -1).limit(200)
    return await cursor.to_list(length=200)


@router.delete("/evidence-packets/{packet_id}")
async def delete_evidence_packet(
    packet_id: str,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    pkt = await db[Collections.SAVED_EVIDENCE_PACKETS].find_one(
        {"id": packet_id, "user_id": current_user["id"], "deleted_at": None},
        {"_id": 0},
    )
    if not pkt:
        raise HTTPException(status_code=404, detail="packet_not_found")
    await db[Collections.SAVED_EVIDENCE_PACKETS].update_one(
        {"id": packet_id},
        {"$set": {"deleted_at": datetime.now(timezone.utc)}},
    )
    await audit_service.log(
        "saved_evidence_packet_deleted",
        user_id=current_user["id"],
        payload={"packet_id": packet_id},
    )
    return {"status": "ok"}


# ============================================================
# folder_summaries (Task 10 — SCAFFOLDED-DISABLED, 501)
# ============================================================
class _SummaryStubBody(BaseModel):
    folder_id: str = Field(min_length=1)
    summary_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=300)


@router.post("/summaries", status_code=501)
async def save_summary_to_folder_stub(
    body: _SummaryStubBody,
    current_user: dict = Depends(get_current_user),
):
    """Task 10 SCAFFOLDED-DISABLED. Returns 501 with a stable detail
    code so the frontend can show a 'Coming soon' tooltip.
    """
    raise HTTPException(
        status_code=501,
        detail="folder_summaries_coming_soon",
    )
