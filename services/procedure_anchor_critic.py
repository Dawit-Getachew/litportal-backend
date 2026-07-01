"""Phase 6G.4P — ProcedureAnchorCritic (SHADOW-ONLY scaffold).

Feature-flagged stub for a future LLM/agentic procedure-anchor critic.
Defaults to DISABLED — the production request path does NOT call this
module unless BOTH environment flags are set:

  PROCEDURE_ANCHOR_CRITIC_ENABLED=true
  PROCEDURE_ANCHOR_CRITIC_MODE=shadow   (any non-"shadow" value is ignored)

Even when enabled, the critic ONLY writes a closed-set audit event
(``procedure_anchor_critic_suggested``). It NEVER mutates search
results, NEVER overrides deterministic caps, NEVER calls a network
source, and NEVER stores raw user text (the original phrase is
explicitly NOT persisted — payload carries ``original_phrase_host_only``
which is always ``None``).

Import-safe regardless of flag state.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("clinevidence.procedure_anchor_critic")


def is_enabled() -> bool:
    """Return True only when both flags are set to the required values."""
    if (os.getenv("PROCEDURE_ANCHOR_CRITIC_ENABLED") or "").strip().lower() != "true":
        return False
    if (os.getenv("PROCEDURE_ANCHOR_CRITIC_MODE", "shadow")
            or "shadow").strip().lower() != "shadow":
        return False
    return True


class ProcedureCriticSuggestion(BaseModel):
    """Closed-set critic output. ``extra='forbid'`` enforces schema."""

    model_config = ConfigDict(extra="forbid")

    suggested_canonical: Optional[str] = None
    suggested_procedure_family: Optional[str] = None
    suggested_specialty: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""  # MUST be ≤ 200 chars in production


class ProcedureAnchorCritic:
    """SHADOW-only stub. Returns ``None`` when disabled. When enabled it
    returns a closed-set suggestion record but DOES NOT mutate the
    request — callers are expected to write an audit event and discard
    the suggestion in the request path."""

    def suggest(
        self, original_phrase: str, current_attempts: list[dict] | None = None,
    ) -> Optional[ProcedureCriticSuggestion]:
        if not is_enabled():
            return None
        # Intentional placeholder — no LLM call wired. Future work
        # (Phase H1/H2) will plug an emergent-key Claude / GPT call
        # behind the same shadow flag. The audit payload contract
        # below is the only stable contract for now.
        suggestion = ProcedureCriticSuggestion(
            suggested_canonical=None,
            suggested_procedure_family=None,
            suggested_specialty=[],
            confidence=0.0,
            rationale="shadow_stub_not_wired",
        )
        logger.debug(
            "procedure_anchor_critic_shadow_called",
            extra={
                "had_phrase": bool(original_phrase),
                "attempts_count": len(current_attempts or []),
            },
        )
        return suggestion


procedure_anchor_critic = ProcedureAnchorCritic()


__all__ = [
    "ProcedureAnchorCritic",
    "ProcedureCriticSuggestion",
    "procedure_anchor_critic",
    "is_enabled",
]
