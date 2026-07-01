"""Retention cleanup — Phase 2D extended.

Phase 1 baseline: soft-deletes ``search_sessions`` whose ``expires_at`` is
in the past.

Phase 2D: when a session is being soft-deleted, ALSO:
  - delete the user-uploaded full-text assets attached to that session (and
    their GridFS bytes if still present)
  - delete the per-session ``article_text_extracts``

Folder metadata + ``folder_articles`` rows remain untouched — folders never
expire (Phase 1 behaviour) and never carry uploaded full text.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from core.db import Collections, get_db

logger = logging.getLogger("clinevidence.retention")


class RetentionCleanupService:
    """Marks expired search sessions as deleted (soft delete) +
    cleans up Phase 2D session-scoped assets (uploads, extracts)."""

    async def cleanup_expired_sessions(self) -> dict:
        now = datetime.now(timezone.utc)
        db = get_db()

        # Identify expiring sessions BEFORE updating so we can cascade.
        expiring = await db[Collections.SEARCH_SESSIONS].find(
            {"expires_at": {"$lt": now}, "deleted_at": None},
            {"_id": 0, "id": 1},
        ).to_list(length=1000)
        session_ids = [d["id"] for d in expiring]

        deleted_uploads = 0
        deleted_extracts = 0
        deleted_access_locs = 0

        if session_ids:
            # Per-session cascade — uploads (with GridFS bytes), text extracts,
            # and per-user upload access_locations rows.
            from services.user_uploaded_full_text import user_uploaded_full_text
            assets = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find(
                {"search_session_id": {"$in": session_ids}, "deleted_at": None},
                {"_id": 0, "id": 1, "user_id": 1},
            ).to_list(length=5000)
            for a in assets:
                try:
                    await user_uploaded_full_text.delete_upload(a["id"], None)
                    deleted_uploads += 1
                except Exception:  # noqa: BLE001
                    logger.exception("retention: failed to delete upload %s", a.get("id"))

            ext_res = await db[Collections.ARTICLE_TEXT_EXTRACTS].delete_many(
                {"search_session_id": {"$in": session_ids}}
            )
            deleted_extracts = int(ext_res.deleted_count or 0)
            al_res = await db[Collections.ARTICLE_ACCESS_LOCATIONS].delete_many({
                "search_session_id": {"$in": session_ids},
                "access_category": "user_uploaded_full_text",
            })
            deleted_access_locs = int(al_res.deleted_count or 0)
            # Phase 3: also delete summaries scoped to these sessions.
            sm_res = await db[Collections.SUMMARIES].delete_many(
                {"search_session_id": {"$in": session_ids}}
            )
            deleted_summaries = int(sm_res.deleted_count or 0)
        else:
            deleted_summaries = 0

        # Soft-delete the sessions themselves.
        result = await db[Collections.SEARCH_SESSIONS].update_many(
            {"expires_at": {"$lt": now}, "deleted_at": None},
            {"$set": {"deleted_at": now, "updated_at": now}},
        )
        return {
            "deleted_count": int(result.modified_count),
            "deleted_uploads": deleted_uploads,
            "deleted_extracts": deleted_extracts,
            "deleted_upload_access_locations": deleted_access_locs,
            "deleted_summaries": deleted_summaries,
        }


retention_cleanup = RetentionCleanupService()
