"""Phase M4.1 — Active M4 startup guard.

Mirrors ``services.semantic_repair_active_guard`` — a deterministic
fail-closed guardrail invoked at backend startup. The guard refuses to
boot if any unsafe combination of M4.1 active env vars is detected on a
production env.

Rules (closed-set):

  * Non-production env → guard is a no-op (logs ``passed``).
  * Production env + master enabled OFF + kill switch ON → no-op
    (the default safe stance).
  * Production env + master enabled ON without explicit
    ``M4_ACTIVE_PROD_ALLOW_ACTIVE=true`` → REFUSE to boot.
  * Production env + kill switch DISENGAGED without explicit
    ``M4_ACTIVE_PROD_ALLOW_ACTIVE=true`` → REFUSE to boot.
  * Production env + ``M4_ACTIVE_PROD_ALLOW_ACTIVE=true`` but env
    allowlist does NOT contain ``production`` → REFUSE to boot
    (defense in depth: two flags must agree).

The guard is NOT bypassable. The RuntimeError it raises propagates out
of ``lifespan`` and crashes startup.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from services.m4_active_config import M4ActiveConfig, load_m4_active_config


def verify_m4_active_startup(
    *,
    is_production: Optional[bool] = None,
    config: Optional[M4ActiveConfig] = None,
    env: Optional[dict[str, str]] = None,
    logger_obj: Optional[Any] = None,
) -> None:
    """Backend-startup fail-closed guardrail for the M4.1 active config.

    Raises:
        RuntimeError: with a closed-set message on production + unsafe
        config combinations.
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

    cfg = config if config is not None else load_m4_active_config(env=env)
    env_label = "production" if is_production else "non_production"

    if not is_production:
        log.info(
            "m4_active_startup_check passed env=%s enabled=%s kill_switch=%s",
            env_label, cfg.enabled, cfg.kill_switch,
        )
        return

    # Production env. Default-safe stance: enabled=false AND kill on.
    if (not cfg.enabled) and cfg.kill_switch_engaged:
        log.info(
            "m4_active_startup_check passed env=production enabled=false "
            "kill_switch=true (default safe stance)"
        )
        return

    # Production + unsafe stance: enforce the explicit prod-allow flag.
    if not cfg.prod_allow_active:
        log.error(
            "m4_active_startup_check failed env=production enabled=%s "
            "kill_switch=%s reason=prod_allow_active_missing",
            cfg.enabled, cfg.kill_switch,
        )
        raise RuntimeError(
            "Production active M4 requires explicit "
            "M4_ACTIVE_PROD_ALLOW_ACTIVE=true"
        )

    if "production" not in {s.lower() for s in cfg.env_allowlist}:
        log.error(
            "m4_active_startup_check failed env=production "
            "prod_allow_active=true env_allowlist=%r "
            "reason=env_allowlist_missing_production",
            cfg.env_allowlist,
        )
        raise RuntimeError(
            "Production active M4 requires M4_ACTIVE_ENV_ALLOWLIST to "
            "contain 'production'"
        )

    log.warning(
        "m4_active_startup_check passed env=production enabled=%s "
        "kill_switch=%s prod_allow_active=true env_allowlist=%r "
        "— ACTIVE M4 ENABLED IN PRODUCTION",
        cfg.enabled, cfg.kill_switch, cfg.env_allowlist,
    )


__all__ = ["verify_m4_active_startup"]
