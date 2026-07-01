# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M3.2 / M3.1 fail-closed startup
# guards. M3.2 active is superseded by M4 (services.m4_active_guard). These guards
# are RETAINED + still registered at startup because they are fail-closed safety
# checks (they only ever REFUSE an unsafe production boot; they never enable
# anything). M3.2 stays dormant + production-hard-off. Not part of the forward path.

"""Phase M3-preflight — Active Semantic Repair startup guard.

Mirrors the K3B ``cap_enforcement_guard.verify_startup_fingerprint``
pattern: a deterministic fail-closed guardrail invoked at backend
startup. The guard refuses to boot if any unsafe combination of M3
active-repair env vars is detected on a production env.

Rules (closed-set):

  * Non-production env → guard is a no-op (logs ``passed``).
  * Production env + master enabled OFF + kill switch ON → no-op
    (the default safe stance).
  * Production env + master enabled ON without explicit
    ``SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE=true`` → REFUSE to boot.
  * Production env + kill switch DISENGAGED without explicit
    ``SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE=true`` → REFUSE to boot.
  * Production env + ``SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE=true`` but
    env allowlist does NOT contain ``production`` → REFUSE to boot
    (defense in depth: the operator must have ALSO added ``production``
    to ``SEMANTIC_REPAIR_ACTIVE_ENV_ALLOWLIST``).

The guard is NOT bypassable. The RuntimeError it raises propagates
out of ``lifespan`` and crashes startup. Operator runbook for a
future approved production active rollout: see the M3 closeout
report.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from services.semantic_repair_active_config import (
    SemanticRepairActiveConfig,
    load_active_config,
)


def verify_semantic_repair_active_startup(
    *,
    is_production: Optional[bool] = None,
    config: Optional[SemanticRepairActiveConfig] = None,
    env: Optional[dict[str, str]] = None,
    logger_obj: Optional[Any] = None,
) -> None:
    """Backend-startup fail-closed guardrail for the M3 active-repair
    config. See module docstring for the full behavior spec.

    Raises:
        RuntimeError: with a closed-set message on production +
        unsafe config combinations.
    """
    log = logger_obj if logger_obj is not None else logging.getLogger(
        "clinevidence",
    )

    if is_production is None:
        try:
            from core.config import settings  # type: ignore
            is_production = bool(settings.IS_PRODUCTION)
        except Exception:  # noqa: BLE001
            is_production = False

    cfg = config if config is not None else load_active_config(env=env)

    env_label = "production" if is_production else "non_production"

    # Non-production OR default-safe stance: no-op.
    if not is_production:
        log.info(
            "semantic_repair_active_startup_check passed env=%s "
            "enabled=%s kill_switch=%s",
            env_label, cfg.enabled, cfg.kill_switch,
        )
        return

    # Production env. Default-safe stance: enabled=false AND kill on.
    if (not cfg.enabled) and cfg.kill_switch_engaged:
        log.info(
            "semantic_repair_active_startup_check passed env=production "
            "enabled=false kill_switch=true (default safe stance)"
        )
        return

    # Production + unsafe stance: enforce the explicit prod-allow flag.
    if not cfg.prod_allow_active:
        log.error(
            "semantic_repair_active_startup_check failed env=production "
            "enabled=%s kill_switch=%s reason=prod_allow_active_missing",
            cfg.enabled, cfg.kill_switch,
        )
        raise RuntimeError(
            "Production active semantic repair requires explicit "
            "SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE=true"
        )

    # Production + prod_allow=true: also require ``production`` in the
    # env allowlist (defense in depth — two flags must agree).
    if "production" not in {s.lower() for s in cfg.env_allowlist}:
        log.error(
            "semantic_repair_active_startup_check failed env=production "
            "prod_allow_active=true env_allowlist=%r "
            "reason=env_allowlist_missing_production",
            cfg.env_allowlist,
        )
        raise RuntimeError(
            "Production active semantic repair requires "
            "SEMANTIC_REPAIR_ACTIVE_ENV_ALLOWLIST to contain 'production'"
        )

    # All explicit operator flags are set. Log loudly so the
    # rollout is visible in ops logs.
    log.warning(
        "semantic_repair_active_startup_check passed env=production "
        "enabled=%s kill_switch=%s prod_allow_active=true "
        "env_allowlist=%r — ACTIVE SEMANTIC REPAIR ENABLED IN PRODUCTION",
        cfg.enabled, cfg.kill_switch, cfg.env_allowlist,
    )


__all__ = [
    "verify_semantic_repair_active_startup",
    "verify_m3_1_benchmark_startup",
]


def verify_m3_1_benchmark_startup(
    *,
    is_production: Optional[bool] = None,
    enabled: Optional[bool] = None,
    env: Optional[dict[str, str]] = None,
    logger_obj: Optional[Any] = None,
) -> None:
    """Phase M3.1 — production fail-closed guard.

    Refuses to boot if ``APP_ENV=production`` AND
    ``M3_1_BENCHMARK_ENABLED=true``. M3.1 is preview/dev-only by
    design and the operator must not deploy it to production.

    Args:
        is_production: override the production detection (test hook)
        enabled: override the env-var read (test hook)
        env: explicit env mapping (test hook)

    Raises:
        RuntimeError: with a closed-set message on production + enabled.
    """
    log = logger_obj if logger_obj is not None else logging.getLogger(
        "clinevidence",
    )
    if is_production is None:
        try:
            from core.config import settings  # type: ignore
            is_production = bool(settings.IS_PRODUCTION)
        except Exception:  # noqa: BLE001
            is_production = False
    if enabled is None:
        src = env if env is not None else os.environ
        enabled = (src.get("M3_1_BENCHMARK_ENABLED", "") or "").strip().lower() == "true"
    if not is_production:
        log.info(
            "m3_1_benchmark_startup_check passed env=non_production enabled=%s",
            enabled,
        )
        return
    if not enabled:
        log.info(
            "m3_1_benchmark_startup_check passed env=production enabled=false "
            "(default safe stance)"
        )
        return
    log.error(
        "m3_1_benchmark_startup_check failed env=production enabled=true "
        "reason=m3_1_is_preview_dev_only"
    )
    raise RuntimeError(
        "M3.1 A/B retrieval benchmark is preview/dev-only and MUST NOT be "
        "enabled in production (M3_1_BENCHMARK_ENABLED=true on APP_ENV=production)"
    )
