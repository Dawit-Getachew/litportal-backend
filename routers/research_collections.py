"""Research Collections router — user-owned collections of saved papers.

Mirrors routers/folders.py conventions: PHI screening of free-text fields via
``screen_user_text`` (422 on block), closed-set audit events, and per-user
ownership enforced in the service. Persistence + LitHub mirroring live in
services/research_collection_service.py.
"""
from __future__ import annotations

from typing import Any, Awaitable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from core.security import get_current_user
from models.research_collections import (
    AddRecordsRequest,
    CollectionCreate,
    CollectionUpdate,
    MoveRecordRequest,
)
from services.audit import audit_service
from services.phi_guard import LIBRARY_PHI_MESSAGE, screen_user_text
from services.research_collection_service import (
    CollectionError,
    research_collection_service,
)

router = APIRouter(prefix="/research-collections", tags=["research-collections"])


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
    if not text:
        return None
    r = screen_user_text(text, context_type)
    if r.is_blocked:
        return _phi_block_response(context_type, r.reason_codes)
    return None


async def _run(awaitable: Awaitable[Any]) -> Any:
    try:
        return await awaitable
    except CollectionError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.code) from exc


# ── Collections ─────────────────────────────────────────────────────
@router.post("", status_code=201)
async def create_collection(body: CollectionCreate, current_user: dict = Depends(get_current_user)):
    for text, ctx in ((body.name, "collection_name"), (body.description, "collection_description")):
        block = _check(text, ctx)
        if block:
            await audit_service.log(
                "research_collection_phi_blocked", user_id=current_user["id"],
                payload={"context_type": ctx},
            )
            return block
    return await _run(research_collection_service.create(current_user["id"], body))


@router.get("")
async def list_collections(current_user: dict = Depends(get_current_user)):
    return await research_collection_service.list(current_user["id"])


@router.get("/default")
async def get_default_collection(current_user: dict = Depends(get_current_user)):
    """B13 — the user's default 'LitPortal Imports' collection (auto-created)."""
    return await research_collection_service.get_or_create_default(current_user["id"])


@router.get("/{collection_id}")
async def get_collection(collection_id: str, current_user: dict = Depends(get_current_user)):
    return await _run(research_collection_service.get(current_user["id"], collection_id))


@router.patch("/{collection_id}")
async def update_collection(
    collection_id: str,
    body: CollectionUpdate,
    current_user: dict = Depends(get_current_user),
):
    for text, ctx in ((body.name, "collection_name"), (body.description, "collection_description")):
        block = _check(text, ctx)
        if block:
            await audit_service.log(
                "research_collection_phi_blocked", user_id=current_user["id"],
                payload={"context_type": ctx},
            )
            return block
    return await _run(research_collection_service.update(current_user["id"], collection_id, body))


@router.delete("/{collection_id}", status_code=200)
async def delete_collection(collection_id: str, current_user: dict = Depends(get_current_user)):
    return await _run(research_collection_service.delete(current_user["id"], collection_id))


# ── Items / records ─────────────────────────────────────────────────
@router.post("/{collection_id}/records", status_code=201)
async def add_records(
    collection_id: str,
    body: AddRecordsRequest,
    current_user: dict = Depends(get_current_user),
):
    for rec in body.records:
        for text, ctx in ((rec.title, "collection_item_title"), (rec.notes, "collection_item_note")):
            block = _check(text, ctx)
            if block:
                await audit_service.log(
                    "research_collection_phi_blocked", user_id=current_user["id"],
                    payload={"context_type": ctx},
                )
                return block
    return await _run(research_collection_service.add_records(current_user["id"], collection_id, body))


@router.get("/{collection_id}/records")
async def list_records(collection_id: str, current_user: dict = Depends(get_current_user)):
    return await _run(research_collection_service.list_items(current_user["id"], collection_id))


@router.delete("/{collection_id}/records/{item_id}", status_code=200)
async def remove_record(
    collection_id: str,
    item_id: str,
    current_user: dict = Depends(get_current_user),
):
    return await _run(
        research_collection_service.remove_record(current_user["id"], collection_id, item_id),
    )


@router.post("/{collection_id}/records/{item_id}/move", status_code=200)
async def move_record(
    collection_id: str,
    item_id: str,
    body: MoveRecordRequest,
    current_user: dict = Depends(get_current_user),
):
    return await _run(
        research_collection_service.move_record(
            current_user["id"], collection_id, item_id, body.target_collection_id,
        ),
    )
