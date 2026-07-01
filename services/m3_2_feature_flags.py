# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M3.2 active-repair Mongo kill
# switch. M3.2 is dormant + superseded by M4 (services.m4_active_feature_flags).
# The flag remains the fail-closed default (engaged) and is RETAINED for the
# admin kill-switch read endpoint + service-level unit tests. M3.2 is not invoked
# in the forward path. Do not re-wire M3.2 into active retrieval.

"""Phase M3.2 — Mongo-backed runtime feature flags.

DETERMINISTIC. NO LLM. NO RETRIEVAL.

Owns the **second** kill-switch source for M3.2 active semantic
repair. The first source is the env var
``SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH`` (default ``true`` = engaged)
loaded by ``services.semantic_repair_active_config``.

**Fail-closed contract**: M3.2 active repair runs **only when BOTH
sources report disengaged**. Either source engaged = repair is
NOT attempted, A-arm CQM is used unchanged.

This module exposes:
    * ``is_mongo_kill_engaged()`` — async Mongo read; defaults to
      ``True`` when the flag row is missing OR the read fails.
    * ``set_mongo_kill_switch_state(engaged, actor_user_id)`` —
      admin-only upsert that writes a closed-set audit event
      ``m3_2_kill_switch_toggle``. Never raises on audit failure.
    * ``read_kill_switch_state(config)`` — closed-set summary of
      env state + mongo state + combined ``engaged`` flag for the
      admin diagnostics surface.

Closed-set keys persisted in ``feature_flags``:
    {
        "key": "m3_2_active_repair_kill_switch",
        "engaged": bool,
        "updated_by_user_id": str | None,
        "updated_at": datetime (UTC),
        "version": int,
    }
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from core.db import Collections, get_db
from services.audit import audit_service
from services.semantic_repair_active_config import (
    SemanticRepairActiveConfig,
)

logger = logging.getLogger("clinevidence.m3_2_feature_flags")


# Closed-set flag key. Adding new flags is intentionally a 1-line
# change here so the surface stays auditable.
M3_2_KILL_SWITCH_KEY: str = "m3_2_active_repair_kill_switch"

# Default Mongo state when the row is missing. Fail-closed = engaged.
_MONGO_DEFAULT_ENGAGED: bool = True


async def is_mongo_kill_engaged() -> bool:
    """Async Mongo read. Returns ``True`` (engaged) on any failure
    or when the row is missing — fail-closed by construction.

    Never raises. Caller can rely on the returned bool as the
    authoritative Mongo-side answer.
    """
    try:
        doc = await get_db()[Collections.FEATURE_FLAGS].find_one(
            {"key": M3_2_KILL_SWITCH_KEY},
            {"_id": 0, "engaged": 1},
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "m3_2_kill_switch_mongo_read_failed — defaulting to engaged",
        )
        return True
    if not doc or "engaged" not in doc:
        return _MONGO_DEFAULT_ENGAGED
    return bool(doc.get("engaged"))


async def set_mongo_kill_switch_state(
    *,
    engaged: bool,
    actor_user_id: Optional[str],
) -> dict[str, Any]:
    """Admin-only upsert. Writes a closed-set audit event
    ``m3_2_kill_switch_toggle`` with ``prev_state`` / ``new_state``
    / ``actor_user_id`` / ``timestamp``.

    Returns a closed-set dict ``{key, engaged, prev_state, new_state,
    updated_by_user_id, updated_at, version}``. Never raises on audit
    failure (audit_service.log already swallows internal errors).
    """
    db = get_db()
    now = datetime.now(timezone.utc)
    existing = await db[Collections.FEATURE_FLAGS].find_one(
        {"key": M3_2_KILL_SWITCH_KEY},
        {"_id": 0, "engaged": 1, "version": 1},
    )
    prev_state = bool((existing or {}).get("engaged", _MONGO_DEFAULT_ENGAGED))
    version = int((existing or {}).get("version", 0)) + 1
    new_doc = {
        "key": M3_2_KILL_SWITCH_KEY,
        "engaged": bool(engaged),
        "updated_by_user_id": actor_user_id,
        "updated_at": now,
        "version": version,
    }
    await db[Collections.FEATURE_FLAGS].update_one(
        {"key": M3_2_KILL_SWITCH_KEY},
        {"$set": new_doc},
        upsert=True,
    )
    try:
        await audit_service.log(
            "m3_2_kill_switch_toggle",
            user_id=actor_user_id,
            search_session_id=None,
            payload={
                "flag_key": M3_2_KILL_SWITCH_KEY,
                "prev_state": prev_state,
                "new_state": bool(engaged),
                "version": version,
                "phase": "M3.2",
            },
        )
    except Exception:  # noqa: BLE001 — audit must never break caller
        logger.exception("m3_2_kill_switch_toggle audit failed (non-fatal)")
    return {
        "key": M3_2_KILL_SWITCH_KEY,
        "engaged": bool(engaged),
        "prev_state": prev_state,
        "new_state": bool(engaged),
        "updated_by_user_id": actor_user_id,
        "updated_at": now.isoformat(),
        "version": version,
    }


async def read_kill_switch_state(
    config: Optional[SemanticRepairActiveConfig] = None,
) -> dict[str, Any]:
    """Closed-set summary for the admin surface. Reads BOTH the env
    var (via the supplied or freshly-loaded config) and the Mongo
    flag, and returns the OR-combined ``engaged`` boolean.

    Returns:
        {
            "env_kill_switch_engaged": bool,
            "mongo_kill_switch_engaged": bool,
            "combined_engaged": bool,
            "fail_closed_default": True,
            "flag_key": "m3_2_active_repair_kill_switch",
        }
    """
    from services.semantic_repair_active_config import load_active_config
    cfg = config if config is not None else load_active_config()
    env_engaged = bool(cfg.kill_switch_engaged)
    mongo_engaged = await is_mongo_kill_engaged()
    return {
        "env_kill_switch_engaged": env_engaged,
        "mongo_kill_switch_engaged": mongo_engaged,
        "combined_engaged": bool(env_engaged or mongo_engaged),
        "fail_closed_default": True,
        "flag_key": M3_2_KILL_SWITCH_KEY,
    }


__all__ = [
    "M3_2_KILL_SWITCH_KEY",
    "is_mongo_kill_engaged",
    "set_mongo_kill_switch_state",
    "read_kill_switch_state",
]
