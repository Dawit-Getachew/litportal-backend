"""Shared helpers for the LitPortal AI agents.

``run_json_agent`` wraps services.llm_provider.complete_json with strict,
fail-closed Pydantic validation (mirrors the summarizer/verifier pattern):
returns ``(model, None)`` on success or ``(None, reason_code)`` on any failure
(not configured, call failed, schema invalid). It NEVER raises.

``store_agent_decision`` appends an advisory decision to AI_AGENT_DECISIONS
(best-effort; never raises). No raw prompts/completions/PHI are stored.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

from core.db import Collections, get_db
from models.common import new_id, utcnow

logger = logging.getLogger("litportal.agents")


async def run_json_agent(
    *,
    system: str,
    user: str,
    schema_hint: Optional[dict[str, Any]],
    validator: Callable[[dict[str, Any]], Any],
    max_tokens: int = 1024,
    session_id: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Run one closed-JSON LLM agent call. Returns (model, None) or (None, reason)."""
    # Imported lazily so importing an agent never drags the LLM stack in.
    from services.llm_provider import llm_provider

    if not llm_provider.is_configured():
        return None, "llm_not_configured"
    try:
        raw = await llm_provider.complete_json(
            system,
            user,
            schema_hint=schema_hint,
            temperature=0.0,
            max_tokens=max_tokens,
            session_id=session_id,
        )
    except Exception:  # noqa: BLE001 — provider already swallows, belt-and-suspenders
        logger.exception("agent llm call raised")
        return None, "llm_call_failed"
    if not isinstance(raw, dict):
        return None, "llm_call_failed"
    try:
        model = validator(raw)
    except Exception:  # noqa: BLE001 — ValidationError or anything else → fail closed
        return None, "parser_validation_failed"
    return model, None


async def store_agent_decision(
    *,
    agent: str,
    session_id: str,
    user_id: str,
    decision: Optional[dict[str, Any]],
    applied: bool = False,
    failure_reason: Optional[str] = None,
) -> None:
    """Append an advisory decision row (best-effort; never raises)."""
    try:
        await get_db()[Collections.AI_AGENT_DECISIONS].insert_one({
            "id": new_id(),
            "agent": agent,
            "search_session_id": session_id,
            "user_id": user_id,
            "decision": decision,
            "applied": applied,
            "failure_reason": failure_reason,
            "created_at": utcnow(),
        })
    except Exception:  # noqa: BLE001
        logger.exception("store_agent_decision failed (non-fatal) agent=%s", agent)
