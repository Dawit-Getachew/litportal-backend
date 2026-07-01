"""Phase K3B Wave 1 Step 2.0.1 — Startup-fingerprint guardrail tests.

Verifies the closed-set behaviour of
``services.cap_enforcement_guard.verify_startup_fingerprint``:

  1. Production + active + no fingerprint flag → ``RuntimeError``
     (the production-leak incident class).
  2. Production + active + ``DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE=true``
     + ``ACTIVE_CAP_REASON_ALLOWLIST=retracted`` → starts cleanly.
  3. Production + active + alias ``ALLOW_PRODUCTION_CAP_ACTIVE=true``
     + ``ACTIVE_CAP_REASON_ALLOWLIST=retracted`` → starts cleanly.
  4. Production + active + fingerprint flag set BUT
     ``ACTIVE_CAP_REASON_ALLOWLIST=""`` → fail closed.
  5. Production + active + fingerprint flag set BUT
     ``ACTIVE_CAP_REASON_ALLOWLIST="expression_of_concern"`` (no
     ``retracted``) → fail closed.
  6. Dev / staging + active + no fingerprint flag → starts normally.
  7. Production + shadow → starts normally.
  7b. Production + disabled → starts normally.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.cap_enforcement_guard import (  # noqa: E402
    verify_startup_fingerprint,
)


def _logger():
    """Return an in-memory logger so tests can also assert log
    output if needed. The default behaviour delegates to the named
    logger, which pytest's capturing handles already.
    """
    return logging.getLogger(
        "test.k3b_startup_fingerprint_guardrail",
    )


# ---------------------------------------------------------------------
# 1. THE incident we're guarding against.
# ---------------------------------------------------------------------


def test_t1_prod_active_no_fingerprint_raises():
    with pytest.raises(RuntimeError) as exc:
        verify_startup_fingerprint(
            is_production=True, resolved_mode="active",
            env={}, logger_obj=_logger(),
        )
    assert (
        "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE=true"
        in str(exc.value)
    )
    assert (
        "ALLOW_PRODUCTION_CAP_ACTIVE=true"
        in str(exc.value)
    )


# ---------------------------------------------------------------------
# 2. Happy path with primary fingerprint flag.
# ---------------------------------------------------------------------


def test_t2_prod_active_primary_fingerprint_and_allowlist_ok():
    verify_startup_fingerprint(
        is_production=True, resolved_mode="active",
        env={
            "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE": "true",
            "ACTIVE_CAP_REASON_ALLOWLIST": "retracted",
        },
        logger_obj=_logger(),
    )


# ---------------------------------------------------------------------
# 3. Happy path with alias fingerprint flag.
# ---------------------------------------------------------------------


def test_t3_prod_active_alias_fingerprint_and_allowlist_ok():
    verify_startup_fingerprint(
        is_production=True, resolved_mode="active",
        env={
            "ALLOW_PRODUCTION_CAP_ACTIVE": "true",
            "ACTIVE_CAP_REASON_ALLOWLIST": "retracted",
        },
        logger_obj=_logger(),
    )


# ---------------------------------------------------------------------
# 4. Fingerprint flag present but empty allowlist → fail closed.
# ---------------------------------------------------------------------


def test_t4_prod_active_fingerprint_ok_but_empty_allowlist_raises():
    with pytest.raises(RuntimeError) as exc:
        verify_startup_fingerprint(
            is_production=True, resolved_mode="active",
            env={
                "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE": "true",
                "ACTIVE_CAP_REASON_ALLOWLIST": "",
            },
            logger_obj=_logger(),
        )
    assert "ACTIVE_CAP_REASON_ALLOWLIST" in str(exc.value)
    assert "'retracted'" in str(exc.value)


# ---------------------------------------------------------------------
# 5. Fingerprint flag present but allowlist missing ``retracted`` →
#    fail closed. (Per spec this MUST fail even if the allowlist
#    contains other Wave-2 candidate codes like
#    ``expression_of_concern``.)
# ---------------------------------------------------------------------


def test_t5_prod_active_fingerprint_ok_but_allowlist_missing_retracted_raises():
    with pytest.raises(RuntimeError) as exc:
        verify_startup_fingerprint(
            is_production=True, resolved_mode="active",
            env={
                "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE": "true",
                "ACTIVE_CAP_REASON_ALLOWLIST": "expression_of_concern",
            },
            logger_obj=_logger(),
        )
    assert "ACTIVE_CAP_REASON_ALLOWLIST" in str(exc.value)


# ---------------------------------------------------------------------
# 6. Dev / staging + active → starts normally (guard does not fire).
# ---------------------------------------------------------------------


def test_t6_dev_active_no_fingerprint_does_not_raise():
    # No env at all — guard must pass cleanly because the env is not
    # production.
    verify_startup_fingerprint(
        is_production=False, resolved_mode="active",
        env={},
        logger_obj=_logger(),
    )


# ---------------------------------------------------------------------
# 7. Production + shadow / disabled → starts normally.
# ---------------------------------------------------------------------


def test_t7a_prod_shadow_does_not_raise():
    verify_startup_fingerprint(
        is_production=True, resolved_mode="shadow",
        env={},  # no fingerprint flag needed when mode != active
        logger_obj=_logger(),
    )


def test_t7b_prod_disabled_does_not_raise():
    verify_startup_fingerprint(
        is_production=True, resolved_mode="disabled",
        env={},
        logger_obj=_logger(),
    )


# ---------------------------------------------------------------------
# Defense-in-depth: parsing invariants of the allowlist.
# ---------------------------------------------------------------------


def test_allowlist_is_case_insensitive():
    """The allowlist parser lowers + trims every entry so an env
    value like ``"  Retracted  "`` still satisfies the check.
    """
    verify_startup_fingerprint(
        is_production=True, resolved_mode="active",
        env={
            "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE": "true",
            "ACTIVE_CAP_REASON_ALLOWLIST": "  Retracted  ,  foo",
        },
        logger_obj=_logger(),
    )


def test_fingerprint_flag_value_is_case_insensitive():
    """``"True"`` / ``"TRUE"`` / ``"true"`` all satisfy the flag."""
    for v in ("true", "True", "TRUE", " true "):
        verify_startup_fingerprint(
            is_production=True, resolved_mode="active",
            env={
                "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE": v,
                "ACTIVE_CAP_REASON_ALLOWLIST": "retracted",
            },
            logger_obj=_logger(),
        )


def test_fingerprint_flag_must_be_explicitly_true():
    """``"1"`` / ``"yes"`` / ``"on"`` are NOT accepted — the gate
    requires the literal string ``"true"`` (case-insensitive).
    """
    for v in ("1", "yes", "on", "active", "y"):
        with pytest.raises(RuntimeError):
            verify_startup_fingerprint(
                is_production=True, resolved_mode="active",
                env={
                    "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE": v,
                    "ACTIVE_CAP_REASON_ALLOWLIST": "retracted",
                },
                logger_obj=_logger(),
            )


def test_resolved_mode_defaults_to_env_var_when_omitted(monkeypatch):
    """When ``resolved_mode`` is not passed, the helper reads the
    same env var the rest of the code reads.
    """
    # Make the env safe: production + non-active.
    monkeypatch.setenv(
        "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "shadow",
    )
    verify_startup_fingerprint(
        is_production=True,
        # resolved_mode omitted on purpose; reads from env.
        logger_obj=_logger(),
    )


def test_is_production_defaults_to_settings(monkeypatch):
    """When ``is_production`` is not passed, the helper reads
    ``settings.IS_PRODUCTION`` (same primitive as every other
    production-aware path in the codebase).
    """
    from core.config import settings
    # Settings already reflects APP_ENV from the running env.
    # On dev the default is ``False`` → guard is a no-op.
    assert settings.IS_PRODUCTION is False
    verify_startup_fingerprint(
        # is_production omitted on purpose; reads from settings.
        resolved_mode="active",
        env={},
        logger_obj=_logger(),
    )
