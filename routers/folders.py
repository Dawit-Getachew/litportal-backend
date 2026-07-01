"""Folders + folder_articles router."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from core.db import Collections, get_db
from core.security import get_current_user
from models.folders import (
    FolderArticleBatchCreate,
    FolderArticleCreate,
    FolderArticleDoc,
    FolderArticleUpdate,
    FolderCreate,
    FolderDoc,
    FolderUpdate,
)
from services.audit import audit_service
from services.phi_guard import LIBRARY_PHI_MESSAGE, screen_user_text

router = APIRouter(prefix="/folders", tags=["folders"])


def _phi_block_response(context_type: str, codes: list[str]) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": "library_text_phi_blocked",
            "context_type": context_type,
            "reason_codes": codes,
            "message": LIBRARY_PHI_MESSAGE,
        },
    )


def _check(text: str | None, context_type: str) -> JSONResponse | None:
    if text is None or text == "":
        return None
    r = screen_user_text(text, context_type)
    if r.is_blocked:
        return _phi_block_response(context_type, r.reason_codes)
    return None


def _check_tags(tags: list[str] | None) -> JSONResponse | None:
    for t in (tags or []):
        block = _check(t, "library_tag")
        if block:
            return block
    return None


def _serialize_folder(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "_id"}


@router.post("/", status_code=201)
async def create_folder(body: FolderCreate, current_user: dict = Depends(get_current_user)):
    block = _check(body.name, "folder_name")
    if block:
        await audit_service.log(
            "library_folder_phi_blocked",
            user_id=current_user["id"],
            payload={"context_type": "folder_name"},
        )
        return block
    block = _check(body.description, "folder_description")
    if block:
        await audit_service.log(
            "library_folder_phi_blocked",
            user_id=current_user["id"],
            payload={"context_type": "folder_description"},
        )
        return block
    db = get_db()
    folder = FolderDoc(user_id=current_user["id"], name=body.name.strip(), description=body.description)
    doc = folder.model_dump()
    for k in ("created_at", "updated_at"):
        doc[k] = doc[k].astimezone(timezone.utc)
    await db[Collections.FOLDERS].insert_one(doc)
    await audit_service.log("folder_created", user_id=current_user["id"], payload={"folder_id": folder.id})
    return _serialize_folder(doc)


@router.get("/")
async def list_folders(current_user: dict = Depends(get_current_user)):
    db = get_db()
    cursor = db[Collections.FOLDERS].find({"user_id": current_user["id"]}, {"_id": 0}).sort("created_at", -1)
    folders = await cursor.to_list(length=500)
    # Phase 6D — hydrate each folder with article_count for the UI grid.
    for f in folders:
        f["article_count"] = await db[Collections.FOLDER_ARTICLES].count_documents(
            {"folder_id": f["id"]},
        )
    return folders


async def _get_owned_folder(folder_id: str, user_id: str) -> dict:
    db = get_db()
    folder = await db[Collections.FOLDERS].find_one({"id": folder_id, "user_id": user_id}, {"_id": 0})
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found.")
    return folder


@router.get("/{folder_id}")
async def get_folder(folder_id: str, current_user: dict = Depends(get_current_user)):
    return await _get_owned_folder(folder_id, current_user["id"])


@router.patch("/{folder_id}")
async def update_folder(
    folder_id: str,
    body: FolderUpdate,
    current_user: dict = Depends(get_current_user),
):
    block = _check(body.name, "folder_name")
    if block:
        await audit_service.log(
            "library_folder_phi_blocked",
            user_id=current_user["id"],
            payload={"context_type": "folder_name"},
        )
        return block
    block = _check(body.description, "folder_description")
    if block:
        await audit_service.log(
            "library_folder_phi_blocked",
            user_id=current_user["id"],
            payload={"context_type": "folder_description"},
        )
        return block
    await _get_owned_folder(folder_id, current_user["id"])
    db = get_db()
    update = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if update:
        update["updated_at"] = datetime.now(timezone.utc)
        await db[Collections.FOLDERS].update_one({"id": folder_id}, {"$set": update})
        await audit_service.log("folder_updated", user_id=current_user["id"], payload={"folder_id": folder_id})
    fresh = await db[Collections.FOLDERS].find_one({"id": folder_id}, {"_id": 0})
    return fresh


@router.delete("/{folder_id}", status_code=200)
async def delete_folder(folder_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_folder(folder_id, current_user["id"])
    db = get_db()
    await db[Collections.FOLDER_ARTICLES].delete_many({"folder_id": folder_id})
    await db[Collections.FOLDERS].delete_one({"id": folder_id})
    await audit_service.log("folder_deleted", user_id=current_user["id"], payload={"folder_id": folder_id})
    return {"status": "ok"}


# ---- folder_articles ----
@router.post("/{folder_id}/articles", status_code=201)
async def add_article(
    folder_id: str,
    body: FolderArticleCreate,
    current_user: dict = Depends(get_current_user),
):
    block = _check(body.user_notes, "folder_item_note")
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "folder_item_note"},
        )
        return block
    block = _check_tags(body.tags)
    if block:
        return block
    await _get_owned_folder(folder_id, current_user["id"])
    db = get_db()
    fa = FolderArticleDoc(
        folder_id=folder_id,
        article_id=body.article_id,
        user_notes=body.user_notes,
        saved_from_search_session_id=body.saved_from_search_session_id,
        from_library_item_id=body.from_library_item_id,
        tags=list(body.tags or []),
    )
    doc = fa.model_dump()
    doc["created_at"] = doc["created_at"].astimezone(timezone.utc)
    await db[Collections.FOLDER_ARTICLES].insert_one(doc)
    await audit_service.log(
        "folder_item_added",
        user_id=current_user["id"],
        payload={"folder_id": folder_id, "article_id": body.article_id},
    )
    return {k: v for k, v in doc.items() if k != "_id"}


@router.post("/{folder_id}/articles/batch", status_code=201)
async def add_articles_batch(
    folder_id: str,
    body: FolderArticleBatchCreate,
    current_user: dict = Depends(get_current_user),
):
    """Phase 6D — batch-add N articles to a folder."""
    block = _check(body.user_notes, "folder_item_note")
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "folder_item_note"},
        )
        return block
    block = _check_tags(body.tags)
    if block:
        return block
    await _get_owned_folder(folder_id, current_user["id"])
    db = get_db()
    created: list[dict] = []
    skipped: list[str] = []
    for art_id in body.article_ids:
        # Skip if already in folder.
        existing = await db[Collections.FOLDER_ARTICLES].find_one(
            {"folder_id": folder_id, "article_id": art_id},
            {"_id": 0, "id": 1},
        )
        if existing:
            skipped.append(art_id)
            continue
        fa = FolderArticleDoc(
            folder_id=folder_id,
            article_id=art_id,
            user_notes=body.user_notes,
            saved_from_search_session_id=body.saved_from_search_session_id,
            tags=list(body.tags or []),
        )
        doc = fa.model_dump()
        doc["created_at"] = doc["created_at"].astimezone(timezone.utc)
        await db[Collections.FOLDER_ARTICLES].insert_one(doc)
        created.append({"id": doc["id"], "article_id": art_id})
    await audit_service.log(
        "folder_item_added",
        user_id=current_user["id"],
        payload={
            "folder_id": folder_id,
            "n_added": len(created),
            "n_skipped_existing": len(skipped),
            "is_batch": True,
        },
    )
    return {
        "folder_id": folder_id,
        "added_count": len(created),
        "skipped_count": len(skipped),
        "created": created,
        "skipped_article_ids": skipped,
    }


@router.get("/{folder_id}/articles")
async def list_folder_articles(folder_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_folder(folder_id, current_user["id"])
    db = get_db()
    rows = await db[Collections.FOLDER_ARTICLES].find(
        {"folder_id": folder_id}, {"_id": 0},
    ).sort("created_at", -1).to_list(length=500)
    # Phase 6D — hydrate with article metadata.
    article_ids = [r["article_id"] for r in rows]
    art_map: dict[str, dict] = {}
    if article_ids:
        async for row in db[Collections.ARTICLES].find(
            {"id": {"$in": article_ids}},
            {
                "_id": 0, "id": 1, "title": 1, "authors_json": 1,
                "journal": 1, "publication_year": 1,
                "doi": 1, "pmid": 1, "pmcid": 1, "canonical_key": 1,
                "article_type_classification": 1, "is_guideline": 1,
                "source_urls_json": 1, "full_text_status": 1,
            },
        ):
            art_map[row["id"]] = row
    # Phase H2-tail Task A — defensive author-email scrub on hydrated
    # article cards so legacy records never surface a corresponding-
    # author email through this read endpoint.
    if art_map:
        from services.article_email_redactor import redact_article_emails
        for _aid in list(art_map.keys()):
            art_map[_aid] = redact_article_emails(art_map[_aid])
    for r in rows:
        r["article"] = art_map.get(r["article_id"])
    return rows


@router.patch("/{folder_id}/articles/{folder_article_id}")
async def update_folder_article(
    folder_id: str,
    folder_article_id: str,
    body: FolderArticleUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Phase 6D — update folder-item notes / tags / pinned status."""
    block = _check(body.user_notes, "folder_item_note")
    if block:
        await audit_service.log(
            "library_note_phi_blocked", user_id=current_user["id"],
            payload={"context_type": "folder_item_note"},
        )
        return block
    block = _check_tags(body.tags)
    if block:
        return block
    await _get_owned_folder(folder_id, current_user["id"])
    db = get_db()
    update = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not update:
        fresh = await db[Collections.FOLDER_ARTICLES].find_one(
            {"id": folder_article_id, "folder_id": folder_id}, {"_id": 0},
        )
        if not fresh:
            raise HTTPException(status_code=404, detail="folder_article_not_found")
        return fresh
    res = await db[Collections.FOLDER_ARTICLES].update_one(
        {"id": folder_article_id, "folder_id": folder_id},
        {"$set": update},
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="folder_article_not_found")
    await audit_service.log(
        "folder_item_note_updated",
        user_id=current_user["id"],
        payload={
            "folder_id": folder_id,
            "folder_article_id": folder_article_id,
            "fields": list(update.keys()),
        },
    )
    return await db[Collections.FOLDER_ARTICLES].find_one(
        {"id": folder_article_id, "folder_id": folder_id}, {"_id": 0},
    )


@router.delete("/{folder_id}/articles/{folder_article_id}", status_code=200)
async def remove_folder_article(
    folder_id: str,
    folder_article_id: str,
    current_user: dict = Depends(get_current_user),
):
    await _get_owned_folder(folder_id, current_user["id"])
    db = get_db()
    res = await db[Collections.FOLDER_ARTICLES].delete_one(
        {"id": folder_article_id, "folder_id": folder_id}
    )
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Folder article not found.")
    await audit_service.log(
        "folder_item_removed",
        user_id=current_user["id"],
        payload={"folder_id": folder_id, "folder_article_id": folder_article_id},
    )
    return {"status": "ok"}
