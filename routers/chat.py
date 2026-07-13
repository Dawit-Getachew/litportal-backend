"""Portal Engine grounded-chat endpoint — frontend compatibility.

The integrated LitPortal/LitScholar frontend generates grounded answers and
"clinical evidence summaries" by POSTing to the Portal Engine
``/api/v1/chat/stream`` (and the un-versioned ``/api/chat/stream``) with a
``search_id``, a ``message``, and the ``selected_record_ids`` the clinician
picked. That endpoint existed on the old LitBridge Portal Engine; the new
litportal backend never ported it, so the UI's "generate summary" / LitScholar
chat call returned 404. This reimplements a compatible **Server-Sent Events**
endpoint on litportal, grounding the answer ONLY in the selected articles.

Contract (must match the frontend SSE consumer in
``frontend/src/lib/portalEngineClient.ts`` and ``LitBridge/lib/api.ts``):
  event: chat_started    data: {"conversation_id": str, "resolved_records": [...]}
  event: thinking        data: {"chunk": str}          # deltas; client concatenates
  event: chat_completed  data: {"conversation_id": str, "message_id": str|null}
  event: error           data: {"error": str, "code"?: str}

Grounding: the model is told to cite ONLY the numbered selected papers using
``[1]..[n]`` markers whose order matches ``selected_record_ids`` (the frontend
maps ``[n]`` back to the n-th selected record). No invented citations.

Intentionally UNAUTHENTICATED — the frontend calls this without an
Authorization header (matches the old LitBridge endpoint and the sibling
``/api/enrichment`` route). Record resolution is scoped by ``search_id``; the
payload is grounded public bibliographic content, no PHI, no user-scoped data.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.db import Collections, get_db
from services.llm_provider import llm_provider

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Bound the grounding prompt so a huge selection can't blow the context window.
_MAX_RECORDS = 12
_ABSTRACT_CHARS = 900


class ChatStreamRequest(BaseModel):
    search_id: str
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    selected_record_ids: Optional[list[str]] = None
    # Accepted (LitBridge sends it) but not yet used — kept for contract parity.
    deep_thinking: bool = False


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event exactly like the old Portal Engine
    (``event: <name>\\ndata: <json>\\n\\n``) so the frontend parser matches."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _resolve_article_id(db, search_id: str, record_id: str) -> Optional[str]:
    """A selected_record_id is normally the ``search_session_articles`` ROW id
    (scoped to the session); fall back to the unscoped row, then to an article
    id directly (mirrors routers/enrichment.py)."""
    ssa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"id": record_id, "search_session_id": search_id},
        {"_id": 0, "article_id": 1},
    )
    if ssa and ssa.get("article_id"):
        return ssa["article_id"]
    ssa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"id": record_id}, {"_id": 0, "article_id": 1},
    )
    if ssa and ssa.get("article_id"):
        return ssa["article_id"]
    art = await db[Collections.ARTICLES].find_one({"id": record_id}, {"_id": 0, "id": 1})
    return art["id"] if art else None


async def _default_selected_article_ids(db, search_id: str) -> list[str]:
    """When the caller sends no selection, ground on the session's
    included-by-default (or user-included) articles."""
    ids: list[str] = []
    async for r in db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": search_id}, {"_id": 0, "article_id": 1,
                                           "included_by_default": 1, "user_included": 1,
                                           "exclusion_reason": 1},
    ):
        if r.get("exclusion_reason"):
            continue
        if r.get("user_included") is True or r.get("included_by_default") is True:
            aid = r.get("article_id")
            if aid:
                ids.append(aid)
    return ids


async def _resolve_records(db, req: ChatStreamRequest) -> list[dict[str, Any]]:
    """Resolve the selection to ordered, hydrated article dicts (order preserved
    so ``[1]..[n]`` line up with the frontend's selected list)."""
    article_ids: list[str] = []
    seen: set[str] = set()
    if req.selected_record_ids:
        for rid in req.selected_record_ids:
            aid = await _resolve_article_id(db, req.search_id, str(rid))
            if aid and aid not in seen:
                seen.add(aid)
                article_ids.append(aid)
    if not article_ids:
        for aid in await _default_selected_article_ids(db, req.search_id):
            if aid not in seen:
                seen.add(aid)
                article_ids.append(aid)
    article_ids = article_ids[:_MAX_RECORDS]

    by_id: dict[str, dict] = {}
    if article_ids:
        async for art in db[Collections.ARTICLES].find(
            {"id": {"$in": article_ids}}, {"_id": 0},
        ):
            by_id[art["id"]] = art
    # Preserve request order.
    return [by_id[a] for a in article_ids if a in by_id]


def _format_context(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, art in enumerate(records, start=1):
        authors = art.get("authors_json") or []
        author_str = ""
        if isinstance(authors, list) and authors:
            names = [
                (a.get("name") or a.get("full_name") or a.get("last_name") or "")
                for a in authors if isinstance(a, dict)
            ]
            names = [n for n in names if n]
            if names:
                author_str = ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")
        ident = art.get("pmid") or art.get("doi") or art.get("pmcid") or ""
        abstract = (art.get("abstract") or "").strip().replace("\n", " ")
        if len(abstract) > _ABSTRACT_CHARS:
            abstract = abstract[:_ABSTRACT_CHARS] + "…"
        lines.append(
            f"[{i}] {art.get('title') or 'Untitled'}\n"
            f"    Authors: {author_str or 'n/a'}\n"
            f"    Year: {art.get('publication_year') or 'n/a'} | "
            f"Journal: {art.get('journal') or 'n/a'} | Id: {ident or 'n/a'}\n"
            f"    Abstract: {abstract or 'No abstract available.'}"
        )
    return "\n\n".join(lines)


_SYSTEM = (
    "You are LitScholar, a clinical evidence assistant. You answer ONLY from the "
    "numbered papers provided by the user. You DO NOT use outside knowledge and "
    "you DO NOT invent facts, statistics, or citations.\n"
    "\n"
    "Rules:\n"
    "- Cite every claim with bracketed markers [1], [2], … that refer to the "
    "numbered papers exactly as given. Never cite a number that was not provided.\n"
    "- If the papers do not contain enough information to answer, say so plainly "
    "and summarize only what they do support.\n"
    "- Be concise and clinically useful. Note important limitations (e.g. small "
    "studies, abstract-only evidence, indirect populations) when relevant.\n"
    "- Never include patient identifiers.\n"
    "\n"
    "Return JSON with a single key \"answer\" whose value is the full answer text "
    "(including the [n] citation markers)."
)


def _chunk_text(text: str) -> list[str]:
    """Split into sentence-ish deltas for a streamed typing effect. The client
    concatenates chunks, so any split is contract-valid; this just feels live."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    for i, p in enumerate(parts):
        if not p:
            continue
        chunks.append(p if i == len(parts) - 1 else p + " ")
    return chunks or [text]


async def _events(req: ChatStreamRequest) -> AsyncGenerator[str, None]:
    conversation_id = req.conversation_id or str(uuid.uuid4())
    try:
        db = get_db()
        records = await _resolve_records(db, req)
    except Exception:  # noqa: BLE001
        logger.exception("chat/stream: record resolution failed (search_id=%s)", req.search_id)
        yield _sse("error", {"error": "Failed to load the selected records.", "code": "resolution_failed"})
        return

    if not records:
        yield _sse("error", {
            "error": (
                "No selected papers were found for this search. Select at least "
                "one result, then try again."
            ),
            "code": "no_records",
        })
        return

    if not llm_provider.is_configured():
        yield _sse("error", {"error": "Summarization is not configured.", "code": "llm_not_configured"})
        yield _sse("chat_completed", {"conversation_id": conversation_id, "message_id": None})
        return

    yield _sse("chat_started", {
        "conversation_id": conversation_id,
        "resolved_records": [
            {"index": i + 1, "id": r.get("id"), "title": r.get("title")}
            for i, r in enumerate(records)
        ],
    })

    user_prompt = (
        f"Question / task:\n{req.message}\n\n"
        f"Numbered papers (cite ONLY these, by [number]):\n\n"
        f"{_format_context(records)}"
    )
    try:
        result = await llm_provider.complete_json(
            _SYSTEM, user_prompt, schema_hint={"answer": "..."}, max_tokens=1200,
        )
    except Exception:  # noqa: BLE001
        logger.exception("chat/stream: LLM call failed (search_id=%s)", req.search_id)
        result = None

    answer = None
    if isinstance(result, dict):
        answer = result.get("answer") or result.get("text") or result.get("summary")
        if isinstance(answer, list):
            answer = " ".join(str(x) for x in answer)
    if not answer or not str(answer).strip():
        yield _sse("error", {
            "error": "The language model returned no usable answer. Please try again.",
            "code": "llm_empty_response",
        })
        yield _sse("chat_completed", {"conversation_id": conversation_id, "message_id": None})
        return

    for chunk in _chunk_text(str(answer)):
        yield _sse("thinking", {"chunk": chunk})
    yield _sse("chat_completed", {"conversation_id": conversation_id, "message_id": None})


@router.post("/chat/stream")
async def stream_chat(req: ChatStreamRequest) -> StreamingResponse:
    # Unauthenticated by design (see module docstring). Scoped by search_id.
    return StreamingResponse(
        _events(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
