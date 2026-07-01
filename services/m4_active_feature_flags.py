"""Phase M4.1 — Mongo-backed runtime kill switch for active M4.

DETERMINISTIC. NO LLM. NO RETRIEVAL.

Owns the **second** kill-switch source for M4.1 active promotion. The
first source is the env var ``M4_ACTIVE_KILL_SWITCH`` (default ``true``
= engaged) loaded by ``services.m4_active_config``.

**Fail-closed contract**: M4.1 active promotion runs **only when BOTH
sources report disengaged**. Either source engaged = promotion is NOT
attempted, baseline deterministic retrieval is used unchanged.

Mirrors ``services.m3_2_feature_flags`` exactly but with an
independent closed-set flag key so M4.1 and M3.2 toggle separately.

Closed-set keys persisted in ``feature_flags``:
    {
        "key": "m4_active_kill_switch",
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

logger = logging.getLogger("clinevidence.m4_active_feature_flags")


# Closed-set flag key for the M4.1 active kill switch.
M4_ACTIVE_KILL_SWITCH_KEY: str = "m4_active_kill_switch"

# Default Mongo state when the row is missing. Fail-closed = engaged.
_MONGO_DEFAULT_ENGAGED: bool = True


async def is_m4_mongo_kill_engaged() -> bool:
    """Async Mongo read. Returns ``True`` (engaged) on any failure or
    when the row is missing — fail-closed by construction. Never raises.
    """
    try:
        doc = await get_db()[Collections.FEATURE_FLAGS].find_one(
            {"key": M4_ACTIVE_KILL_SWITCH_KEY},
            {"_id": 0, "engaged": 1},
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "m4_active_kill_switch_mongo_read_failed — defaulting to engaged",
        )
        return True
    if not doc or "engaged" not in doc:
        return _MONGO_DEFAULT_ENGAGED
    return bool(doc.get("engaged"))


async def set_m4_mongo_kill_switch_state(
    *,
    engaged: bool,
    actor_user_id: Optional[str],
) -> dict[str, Any]:
    """Admin-only upsert. Writes a closed-set audit event
    ``m4_active_kill_switch_toggle``. Never raises on audit failure."""
    db = get_db()
    now = datetime.now(timezone.utc)
    existing = await db[Collections.FEATURE_FLAGS].find_one(
        {"key": M4_ACTIVE_KILL_SWITCH_KEY},
        {"_id": 0, "engaged": 1, "version": 1},
    )
    prev_state = bool((existing or {}).get("engaged", _MONGO_DEFAULT_ENGAGED))
    version = int((existing or {}).get("version", 0)) + 1
    new_doc = {
        "key": M4_ACTIVE_KILL_SWITCH_KEY,
        "engaged": bool(engaged),
        "updated_by_user_id": actor_user_id,
        "updated_at": now,
        "version": version,
    }
    await db[Collections.FEATURE_FLAGS].update_one(
        {"key": M4_ACTIVE_KILL_SWITCH_KEY},
        {"$set": new_doc},
        upsert=True,
    )
    try:
        await audit_service.log(
            "m4_active_kill_switch_toggle",
            user_id=actor_user_id,
            search_session_id=None,
            payload={
                "flag_key": M4_ACTIVE_KILL_SWITCH_KEY,
                "prev_state": prev_state,
                "new_state": bool(engaged),
                "version": version,
                "phase": "M4.1",
            },
        )
    except Exception:  # noqa: BLE001 — audit must never break caller
        logger.exception(
            "m4_active_kill_switch_toggle audit failed (non-fatal)",
        )
    return {
        "key": M4_ACTIVE_KILL_SWITCH_KEY,
        "engaged": bool(engaged),
        "prev_state": prev_state,
        "new_state": bool(engaged),
        "updated_by_user_id": actor_user_id,
        "updated_at": now.isoformat(),
        "version": version,
    }


async def read_m4_kill_switch_state(
    config: Optional[Any] = None,
) -> dict[str, Any]:
    """Closed-set summary for the admin surface. OR-combines the env
    var (via the supplied or freshly-loaded config) with the Mongo flag.

    Returns:
        {
            "env_kill_switch_engaged": bool,
            "mongo_kill_switch_engaged": bool,
            "combined_engaged": bool,
            "fail_closed_default": True,
            "flag_key": "m4_active_kill_switch",
        }
    """
    from services.m4_active_config import load_m4_active_config
    cfg = config if config is not None else load_m4_active_config()
    env_engaged = bool(cfg.kill_switch_engaged)
    mongo_engaged = await is_m4_mongo_kill_engaged()
    return {
        "env_kill_switch_engaged": env_engaged,
        "mongo_kill_switch_engaged": mongo_engaged,
        "combined_engaged": bool(env_engaged or mongo_engaged),
        "fail_closed_default": True,
        "flag_key": M4_ACTIVE_KILL_SWITCH_KEY,
    }


__all__ = [
    "M4_ACTIVE_KILL_SWITCH_KEY",
    "is_m4_mongo_kill_engaged",
    "set_m4_mongo_kill_switch_state",
    "read_m4_kill_switch_state",
]
