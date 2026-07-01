"""Phase M4.1 — Preview-only Active M4 configuration.

DETERMINISTIC. NO LLM. NO RETRIEVAL by itself.

Owns the closed-set env-driven configuration that gates whether an
ACCEPTED M4 full-question-interpreter candidate may be PROMOTED into
the clinician-visible active retrieval pipeline. Fail-closed by
construction: the master enabled flag is OFF, the env kill switch is
ENGAGED, the production env is excluded from the allowlist, and a
separate ``M4_ACTIVE_PROD_ALLOW_ACTIVE`` flag is required to ever run
in production (which we never set).

This mirrors ``services.semantic_repair_active_config`` (M3.2) so the
two active-repair surfaces share the same fail-closed semantics, but
the env-var namespace and the consumers are independent. M3.2 stays
dormant; M4.1 is its own engine.

Env vars (all optional, all closed-set defaults):

    M4_ACTIVE_ENABLED            ::= "true" | "false"   [default false]
    M4_ACTIVE_ENV_ALLOWLIST      ::= csv of env labels  [default dev,preview,...]
    M4_ACTIVE_MAX_LATENCY_MS     ::= int                [default 4000]
    M4_ACTIVE_KILL_SWITCH        ::= "true" | "false"   [default true]
    M4_ACTIVE_PROD_ALLOW_ACTIVE  ::= "true" | "false"   [default false]
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

# Default env allowlist — production is intentionally NOT included.
_DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = (
    "development", "dev", "preview", "staging",
)


def _bool_env(name: str, default: bool, env: Optional[dict[str, str]] = None) -> bool:
    src = env if env is not None else os.environ
    raw = (src.get(name, "") or "").strip().lower()
    if raw == "":
        return default
    return raw == "true"


def _csv_env(
    name: str,
    default: Iterable[str],
    env: Optional[dict[str, str]] = None,
) -> tuple[str, ...]:
    src = env if env is not None else os.environ
    raw = (src.get(name, "") or "").strip()
    if not raw:
        return tuple(default)
    out: list[str] = []
    for tok in raw.split(","):
        s = tok.strip().lower()
        if s and s not in out:
            out.append(s)
    return tuple(out) if out else tuple(default)


def _int_env(
    name: str,
    default: int,
    env: Optional[dict[str, str]] = None,
) -> int:
    src = env if env is not None else os.environ
    raw = (src.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


class M4ActiveConfig(BaseModel):
    """Frozen closed-set struct describing the M4.1 active config.
    Surfaced (without secrets) in admin diagnostics so an operator can
    confirm the OFF/kill-engaged stance without env-shell access."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    env_allowlist: tuple[str, ...] = Field(
        default_factory=lambda: _DEFAULT_ENV_ALLOWLIST,
    )
    max_latency_ms: int = 4000
    kill_switch: bool = True
    prod_allow_active: bool = False
    current_env: str = "unknown"

    @property
    def kill_switch_engaged(self) -> bool:
        return bool(self.kill_switch)

    @property
    def env_in_allowlist(self) -> bool:
        return (self.current_env or "").strip().lower() in {
            s.lower() for s in self.env_allowlist
        }

    @property
    def is_production_env(self) -> bool:
        return (self.current_env or "").strip().lower() == "production"

    @property
    def effective_enabled(self) -> bool:
        """Single source of truth for the ENV side. Returns ``True``
        only when EVERY env gate is open. The Mongo kill switch is
        OR-combined separately by the orchestrator (either engaged =
        dormant)."""
        if not self.enabled:
            return False
        if self.kill_switch_engaged:
            return False
        if not self.env_in_allowlist:
            return False
        if self.is_production_env and not self.prod_allow_active:
            return False
        return True


def _resolve_current_env(env: Optional[dict[str, str]] = None) -> str:
    src = env if env is not None else os.environ
    candidate = (src.get("APP_ENV") or "").strip().lower()
    if candidate:
        return candidate
    try:
        from core.config import settings  # type: ignore
        return "production" if bool(settings.IS_PRODUCTION) else "dev"
    except Exception:  # noqa: BLE001
        return "unknown"


def load_m4_active_config(
    env: Optional[dict[str, str]] = None,
) -> M4ActiveConfig:
    """Build a fresh ``M4ActiveConfig`` from the supplied env mapping
    (or the live process env). All fields are closed-set + validated;
    the returned struct is frozen and safe to surface in admin
    diagnostics."""
    return M4ActiveConfig(
        enabled=_bool_env("M4_ACTIVE_ENABLED", False, env=env),
        env_allowlist=_csv_env(
            "M4_ACTIVE_ENV_ALLOWLIST", _DEFAULT_ENV_ALLOWLIST, env=env,
        ),
        max_latency_ms=_int_env("M4_ACTIVE_MAX_LATENCY_MS", 4000, env=env),
        kill_switch=_bool_env("M4_ACTIVE_KILL_SWITCH", True, env=env),
        prod_allow_active=_bool_env(
            "M4_ACTIVE_PROD_ALLOW_ACTIVE", False, env=env,
        ),
        current_env=_resolve_current_env(env=env),
    )


__all__ = [
    "M4ActiveConfig",
    "load_m4_active_config",
]
