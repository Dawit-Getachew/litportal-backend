"""Article-level OA cache helpers.

Cache rows live in the existing `article_text_extracts` collection. They are
distinguished by:
  - `oa_cache = True`
  - `search_session_id IS None`
  - `text_source = "oa_full_text"`
  - `evidence_text_level = "oa_full_text"`

Only commercially-reuse-allowed licenses (CC0 / CC BY / CC BY-SA /
public-domain) ever land in the cache; everything else is fetched per-session
when the appropriate `OA_ALLOW_*` flag is enabled and the result is written
ONLY as a session-scoped row.

User-uploaded full text NEVER enters the OA cache: those rows have
`text_source="user_uploaded_full_text"` and `oa_source=None`.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from core.db import Collections, get_db


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


CACHE_DAYS = _env_int("OA_CACHE_DAYS", 30)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _checksum(sections: Iterable[dict]) -> str:
    blob = "\n--\n".join((s.get("section_text") or "") for s in sections)
    return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()


async def lookup_article_cache(article_id: str) -> list[dict]:
    """Return cached extract rows for this article that have not expired.

    Returns [] on cache miss. Caller is responsible for cloning these into
    session-scoped rows (see `clone_cache_to_session`).
    """
    db = get_db()
    now = _utcnow()
    rows = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
        {
            "article_id": article_id,
            "oa_cache": True,
            "search_session_id": None,
            "text_source": "oa_full_text",
            "oa_expires_at": {"$gt": now},
        },
        {"_id": 0},
    ).to_list(length=200)
    return rows


async def write_cache_rows(
    *,
    article_id: str,
    oa_source: str,
    oa_license: Optional[str],
    oa_license_category: str,
    sections: list[dict],
) -> int:
    """Write article-level cache rows. Caller MUST have verified that
    `oa_license_category == "commercial_reuse_allowed"` before invoking.
    """
    if oa_license_category != "commercial_reuse_allowed":
        return 0
    db = get_db()
    now = _utcnow()
    expires = now + timedelta(days=CACHE_DAYS)
    checksum = _checksum(sections)
    # Purge stale cache rows for the same article + source first (idempotent).
    await db[Collections.ARTICLE_TEXT_EXTRACTS].delete_many(
        {
            "article_id": article_id,
            "oa_cache": True,
            "search_session_id": None,
            "oa_source": oa_source,
        }
    )
    docs = []
    for s in sections:
        docs.append({
            "id": uuid.uuid4().hex,
            "article_id": article_id,
            "uploaded_full_text_asset_id": None,
            "search_session_id": None,
            "text_source": "oa_full_text",
            "section_name": s["section_name"],
            "section_text": s["section_text"],
            "token_count": int(s.get("token_count") or 0),
            "include_in_summary": bool(s.get("include_in_summary")),
            "evidence_text_level": "oa_full_text",
            "created_at": now,
            "expires_at": None,
            "oa_cache": True,
            "oa_source": oa_source,
            "oa_license": oa_license,
            "oa_license_category": oa_license_category,
            "oa_fetch_checksum": checksum,
            "oa_last_fetched_at": now,
            "oa_expires_at": expires,
        })
    if not docs:
        return 0
    await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_many(docs)
    return len(docs)


async def clone_cache_to_session(
    *,
    article_id: str,
    search_session_id: str,
    session_expires_at: Optional[datetime],
) -> int:
    """Copy cache rows into session-scoped rows. Returns the number created."""
    cached = await lookup_article_cache(article_id)
    if not cached:
        return 0
    db = get_db()
    now = _utcnow()
    # First, drop any prior session-scoped OA rows so we never double-count.
    await db[Collections.ARTICLE_TEXT_EXTRACTS].delete_many(
        {
            "article_id": article_id,
            "search_session_id": search_session_id,
            "text_source": "oa_full_text",
            "oa_cache": {"$ne": True},
        }
    )
    docs = []
    for r in cached:
        docs.append({
            "id": uuid.uuid4().hex,
            "article_id": article_id,
            "uploaded_full_text_asset_id": None,
            "search_session_id": search_session_id,
            "text_source": "oa_full_text",
            "section_name": r["section_name"],
            "section_text": r["section_text"],
            "token_count": int(r.get("token_count") or 0),
            "include_in_summary": bool(r.get("include_in_summary")),
            "evidence_text_level": "oa_full_text",
            "created_at": now,
            "expires_at": session_expires_at,
            "oa_cache": False,
            "oa_source": r.get("oa_source"),
            "oa_license": r.get("oa_license"),
            "oa_license_category": r.get("oa_license_category"),
            "oa_fetch_checksum": r.get("oa_fetch_checksum"),
            "oa_last_fetched_at": r.get("oa_last_fetched_at"),
            "oa_expires_at": None,
        })
    if docs:
        await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_many(docs)
    return len(docs)


async def write_session_rows(
    *,
    article_id: str,
    search_session_id: str,
    session_expires_at: Optional[datetime],
    oa_source: str,
    oa_license: Optional[str],
    oa_license_category: str,
    sections: list[dict],
) -> int:
    """Write session-scoped OA rows (used when license not eligible for cache,
    or when force_refresh skipped the cache write).
    """
    db = get_db()
    now = _utcnow()
    await db[Collections.ARTICLE_TEXT_EXTRACTS].delete_many(
        {
            "article_id": article_id,
            "search_session_id": search_session_id,
            "text_source": "oa_full_text",
            "oa_cache": {"$ne": True},
        }
    )
    checksum = _checksum(sections)
    docs = []
    for s in sections:
        docs.append({
            "id": uuid.uuid4().hex,
            "article_id": article_id,
            "uploaded_full_text_asset_id": None,
            "search_session_id": search_session_id,
            "text_source": "oa_full_text",
            "section_name": s["section_name"],
            "section_text": s["section_text"],
            "token_count": int(s.get("token_count") or 0),
            "include_in_summary": bool(s.get("include_in_summary")),
            "evidence_text_level": "oa_full_text",
            "created_at": now,
            "expires_at": session_expires_at,
            "oa_cache": False,
            "oa_source": oa_source,
            "oa_license": oa_license,
            "oa_license_category": oa_license_category,
            "oa_fetch_checksum": checksum,
            "oa_last_fetched_at": now,
            "oa_expires_at": None,
        })
    if docs:
        await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_many(docs)
    return len(docs)


async def purge_expired_cache_rows() -> int:
    """Sweep expired cache rows. Called by `retention_cleanup` periodically."""
    db = get_db()
    now = _utcnow()
    res = await db[Collections.ARTICLE_TEXT_EXTRACTS].delete_many(
        {
            "oa_cache": True,
            "search_session_id": None,
            "oa_expires_at": {"$lte": now},
        }
    )
    return int(res.deleted_count or 0)
