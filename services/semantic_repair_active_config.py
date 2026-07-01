# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M3.2 active semantic-repair config,
# superseded by services.m4_active_config. NOT invoked in the active pipeline
# (gated off via pipeline._LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED=False). Retained
# for the fail-closed startup guard + service-level unit tests. M3.2 stays dormant
# + production-hard-off. Do not re-wire into the forward semantic path.

"""Phase M3-preflight — Active Semantic Repair Configuration.

DETERMINISTIC. NO LLM. NO RETRIEVAL. NO ACTIVE BEHAVIOR.

This module owns the closed-set env-driven configuration for the
future M3 active-repair phase. In **M3-preflight** (this phase) the
config is *read* into a frozen Pydantic struct, but every consumer
hard-fails-closed: the kill switch is engaged by default, the
master enabled flag is OFF, and the production env is excluded
from the env allowlist. The struct is surfaced in admin diagnostics
so an operator can SEE what the would-be M3 active configuration
looks like without ever actually flipping it.

Env vars (all optional, all closed-set defaults):

    SEMANTIC_REPAIR_ACTIVE_ENABLED        ::= "true" | "false"      [default false]
    SEMANTIC_REPAIR_ACTIVE_ENV_ALLOWLIST  ::= csv of env labels     [default "dev,preview"]
    SEMANTIC_REPAIR_ACTIVE_SHAPES_ALLOWLIST::= csv of shape names   [default canonical 3]
    SEMANTIC_REPAIR_ACTIVE_MAX_LATENCY_MS ::= int                   [default 3500]
    SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH    ::= "true" | "false"      [default true]
    SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE     ::= "true" | "false"      [default false]

The closed-set shape allowlist values mirror the M-phase trigger
reasons grouped into clinical shapes:

    etiology_risk            ↔ missing_risk_etiology_framework
    pathophysiology_mechanism ↔ missing_mechanism_pathophysiology_framework
    procedure_complication   ↔ missing_complication_after_procedure_split

Hard rules:
    * No mutation API. ``load_active_config()`` builds a fresh
      ``SemanticRepairActiveConfig`` from env on each call.
    * The struct is ``frozen=True``. Attempts to mutate raise.
    * ``effective_enabled`` is the **single source of truth** for
      consumers. It returns ``True`` ONLY when: master enabled is
      true AND kill switch is disengaged AND env is in allowlist
      AND (if env is production) ``SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE``
      is explicitly true.
    * Production fail-closed wiring lives in
      ``services.semantic_repair_active_guard``.
"""
from __future__ import annotations

import os
from typing import FrozenSet, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field


# Closed-set canonical shape labels. Maps M-phase trigger reasons
# into the shape namespace consumed by the active allowlist.
SEMANTIC_REPAIR_ACTIVE_SHAPE_LABELS: FrozenSet[str] = frozenset({
    "etiology_risk",
    "pathophysiology_mechanism",
    "procedure_complication",
    "prognosis",
    "epidemiology",
    "severity_marker",
    "diagnostic_ruleout",
})

# Default closed-set shape allowlist — the three shapes that hit the
# 100% conditional schema-valid rate in the M2.1 5x5 smoke. Operators
# can override via env but should NEVER expand without a separate
# benchmark.
_DEFAULT_SHAPE_ALLOWLIST: tuple[str, ...] = (
    "etiology_risk",
    "pathophysiology_mechanism",
    "procedure_complication",
)

_DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = ("development", "dev", "preview", "staging")


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


class SemanticRepairActiveConfig(BaseModel):
    """Frozen closed-set struct describing the M3 active-repair
    configuration. Surfaced in admin diagnostics so the operator
    can confirm the OFF/kill-engaged stance without env-shell access."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    env_allowlist: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_ENV_ALLOWLIST)
    shapes_allowlist: tuple[str, ...] = Field(
        default_factory=lambda: _DEFAULT_SHAPE_ALLOWLIST,
    )
    max_latency_ms: int = 3500
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

    def shape_in_allowlist(self, shape: Optional[str]) -> bool:
        if not shape:
            return False
        return shape.strip().lower() in {
            s.lower() for s in self.shapes_allowlist
        }

    @property
    def effective_enabled(self) -> bool:
        """Single source of truth for any consumer. Returns ``True``
        only when EVERY gate is open. Returns ``False`` aggressively.
        """
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
    """Resolve the current env label using the same primitives the
    rest of the backend uses. Falls back to ``APP_ENV`` directly."""
    src = env if env is not None else os.environ
    candidate = (src.get("APP_ENV") or "").strip().lower()
    if not candidate:
        # Fall back to settings.IS_PRODUCTION → production / dev.
        try:
            from core.config import settings  # type: ignore
            return "production" if bool(settings.IS_PRODUCTION) else "dev"
        except Exception:  # noqa: BLE001
            return "unknown"
    return candidate or "unknown"


def load_active_config(
    env: Optional[dict[str, str]] = None,
) -> SemanticRepairActiveConfig:
    """Build a fresh ``SemanticRepairActiveConfig`` from the supplied
    env mapping (or the live process env). All fields are
    closed-set and validated by Pydantic; the returned struct is
    frozen and safe to surface in admin diagnostics."""
    return SemanticRepairActiveConfig(
        enabled=_bool_env("SEMANTIC_REPAIR_ACTIVE_ENABLED", False, env=env),
        env_allowlist=_csv_env(
            "SEMANTIC_REPAIR_ACTIVE_ENV_ALLOWLIST", _DEFAULT_ENV_ALLOWLIST, env=env,
        ),
        shapes_allowlist=_csv_env(
            "SEMANTIC_REPAIR_ACTIVE_SHAPES_ALLOWLIST", _DEFAULT_SHAPE_ALLOWLIST, env=env,
        ),
        max_latency_ms=_int_env(
            "SEMANTIC_REPAIR_ACTIVE_MAX_LATENCY_MS", 3500, env=env,
        ),
        kill_switch=_bool_env("SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH", True, env=env),
        prod_allow_active=_bool_env(
            "SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE", False, env=env,
        ),
        current_env=_resolve_current_env(env=env),
    )


__all__ = [
    "SEMANTIC_REPAIR_ACTIVE_SHAPE_LABELS",
    "SemanticRepairActiveConfig",
    "load_active_config",
]
