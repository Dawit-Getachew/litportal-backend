"""Phase M3-preflight — config + startup-guard fail-closed tests.

DETERMINISTIC. NO LLM. NO RETRIEVAL. NO PRODUCTION ENV TOUCH.

These tests inject env mappings via the dedicated ``env=`` argument
on ``load_active_config`` and ``verify_semantic_repair_active_startup``
so the live process env is never modified. The closed-set contract:

  * Default config = enabled OFF, kill switch ON, env allowlist
    {dev, preview}, shapes allowlist {etiology_risk,
    pathophysiology_mechanism, procedure_complication}, max_latency
    3500ms (Phase M3.2 adjudication target), prod_allow_active OFF.
  * ``effective_enabled`` returns True ONLY when EVERY gate is open.
  * Production startup raises RuntimeError on any unsafe combination
    without explicit ``SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE=true`` AND
    ``production`` in ``SEMANTIC_REPAIR_ACTIVE_ENV_ALLOWLIST``.
"""
from __future__ import annotations

import pytest

from services.semantic_repair_active_config import (
    SEMANTIC_REPAIR_ACTIVE_SHAPE_LABELS,
    SemanticRepairActiveConfig,
    load_active_config,
)
from services.semantic_repair_active_guard import (
    verify_semantic_repair_active_startup,
)


# =====================================================================
# load_active_config — closed-set defaults + env-driven overrides.
# =====================================================================
def test_M3_preflight_config_defaults_are_safe() -> None:
    cfg = load_active_config(env={})
    assert cfg.enabled is False
    assert cfg.kill_switch is True
    assert cfg.kill_switch_engaged is True
    assert cfg.prod_allow_active is False
    assert cfg.max_latency_ms == 3500
    assert tuple(cfg.shapes_allowlist) == (
        "etiology_risk",
        "pathophysiology_mechanism",
        "procedure_complication",
    )
    assert tuple(cfg.env_allowlist) == ("development", "dev", "preview", "staging")


def test_M3_preflight_env_allowlist_excludes_production_by_default() -> None:
    cfg = load_active_config(env={})
    assert "production" not in {s.lower() for s in cfg.env_allowlist}


def test_M3_preflight_shape_allowlist_subset_of_closed_set() -> None:
    cfg = load_active_config(env={})
    assert set(cfg.shapes_allowlist).issubset(SEMANTIC_REPAIR_ACTIVE_SHAPE_LABELS)


def test_M3_preflight_effective_enabled_false_by_default() -> None:
    cfg = load_active_config(env={"APP_ENV": "dev"})
    assert cfg.effective_enabled is False


def test_M3_preflight_kill_switch_overrides_enabled() -> None:
    """Even if the master enabled flag is True, the kill switch
    being engaged ALWAYS wins. This is the dead-man's-switch
    invariant."""
    cfg = load_active_config(env={
        "APP_ENV": "preview",
        "SEMANTIC_REPAIR_ACTIVE_ENABLED": "true",
        "SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH": "true",
    })
    assert cfg.enabled is True
    assert cfg.kill_switch_engaged is True
    assert cfg.effective_enabled is False


def test_M3_preflight_env_not_in_allowlist_blocks_effective_enabled() -> None:
    cfg = load_active_config(env={
        "APP_ENV": "production",
        "SEMANTIC_REPAIR_ACTIVE_ENABLED": "true",
        "SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH": "false",
        "SEMANTIC_REPAIR_ACTIVE_ENV_ALLOWLIST": "dev,preview",
    })
    assert cfg.is_production_env is True
    assert cfg.env_in_allowlist is False
    assert cfg.effective_enabled is False


def test_M3_preflight_production_without_prod_allow_blocks_even_with_allowlist() -> None:
    """Even if an operator adds ``production`` to the env allowlist,
    they ALSO must set ``SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE=true``.
    Defense in depth: two flags must agree."""
    cfg = load_active_config(env={
        "APP_ENV": "production",
        "SEMANTIC_REPAIR_ACTIVE_ENABLED": "true",
        "SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH": "false",
        "SEMANTIC_REPAIR_ACTIVE_ENV_ALLOWLIST": "dev,preview,production",
        # SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE NOT SET (=false).
    })
    assert cfg.effective_enabled is False


def test_M3_preflight_production_with_all_flags_yields_effective_enabled() -> None:
    """ALL operator-explicit flags must be set together — this is the
    only path to effective_enabled=True on production."""
    cfg = load_active_config(env={
        "APP_ENV": "production",
        "SEMANTIC_REPAIR_ACTIVE_ENABLED": "true",
        "SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH": "false",
        "SEMANTIC_REPAIR_ACTIVE_ENV_ALLOWLIST": "dev,preview,production",
        "SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE": "true",
    })
    assert cfg.effective_enabled is True


def test_M3_preflight_config_is_frozen() -> None:
    cfg = load_active_config(env={})
    with pytest.raises(Exception):  # pydantic v2 frozen → ValidationError
        cfg.enabled = True  # type: ignore[misc]


def test_M3_preflight_shape_allowlist_overridable() -> None:
    cfg = load_active_config(env={
        "SEMANTIC_REPAIR_ACTIVE_SHAPES_ALLOWLIST": "etiology_risk",
    })
    assert cfg.shapes_allowlist == ("etiology_risk",)
    assert cfg.shape_in_allowlist("etiology_risk") is True
    assert cfg.shape_in_allowlist("pathophysiology_mechanism") is False


def test_M3_preflight_max_latency_overridable_positive_int_only() -> None:
    cfg = load_active_config(env={
        "SEMANTIC_REPAIR_ACTIVE_MAX_LATENCY_MS": "5000",
    })
    assert cfg.max_latency_ms == 5000
    cfg_bad = load_active_config(env={
        "SEMANTIC_REPAIR_ACTIVE_MAX_LATENCY_MS": "not-a-number",
    })
    assert cfg_bad.max_latency_ms == 3500  # falls back to Phase-M3.2 default
    cfg_neg = load_active_config(env={
        "SEMANTIC_REPAIR_ACTIVE_MAX_LATENCY_MS": "-100",
    })
    assert cfg_neg.max_latency_ms == 3500  # negative → Phase-M3.2 default


# =====================================================================
# verify_semantic_repair_active_startup — production fail-closed.
# =====================================================================
def test_M3_preflight_startup_noop_on_dev() -> None:
    """Dev env is always a no-op. Even unsafe combinations don't
    block dev boot."""
    cfg = SemanticRepairActiveConfig(
        enabled=True, kill_switch=False,
        env_allowlist=("dev",), shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=False, current_env="dev",
    )
    verify_semantic_repair_active_startup(
        is_production=False, config=cfg, env={},
    )  # no raise


def test_M3_preflight_startup_noop_on_production_default_safe() -> None:
    """Production env + default-safe config (enabled=false +
    kill_switch=true) is a no-op."""
    cfg = SemanticRepairActiveConfig(
        enabled=False, kill_switch=True,
        env_allowlist=("dev", "preview"),
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=False,
        current_env="production",
    )
    verify_semantic_repair_active_startup(
        is_production=True, config=cfg, env={},
    )  # no raise


def test_M3_preflight_startup_refuses_on_production_enabled_without_prod_allow() -> None:
    cfg = SemanticRepairActiveConfig(
        enabled=True, kill_switch=True,
        env_allowlist=("dev", "preview"),
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=False,
        current_env="production",
    )
    with pytest.raises(RuntimeError, match="SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE"):
        verify_semantic_repair_active_startup(
            is_production=True, config=cfg, env={},
        )


def test_M3_preflight_startup_refuses_on_production_kill_off_without_prod_allow() -> None:
    cfg = SemanticRepairActiveConfig(
        enabled=False, kill_switch=False,
        env_allowlist=("dev", "preview"),
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=False,
        current_env="production",
    )
    with pytest.raises(RuntimeError, match="SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE"):
        verify_semantic_repair_active_startup(
            is_production=True, config=cfg, env={},
        )


def test_M3_preflight_startup_refuses_on_production_prod_allow_but_no_env_allowlist() -> None:
    """Defense in depth — even with prod_allow_active=true, the env
    allowlist must contain ``production``."""
    cfg = SemanticRepairActiveConfig(
        enabled=True, kill_switch=False,
        env_allowlist=("dev", "preview"),
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=True,
        current_env="production",
    )
    with pytest.raises(
        RuntimeError,
        match="SEMANTIC_REPAIR_ACTIVE_ENV_ALLOWLIST",
    ):
        verify_semantic_repair_active_startup(
            is_production=True, config=cfg, env={},
        )


def test_M3_preflight_startup_passes_on_production_with_all_flags() -> None:
    cfg = SemanticRepairActiveConfig(
        enabled=True, kill_switch=False,
        env_allowlist=("dev", "preview", "production"),
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=True,
        current_env="production",
    )
    verify_semantic_repair_active_startup(
        is_production=True, config=cfg, env={},
    )  # no raise — but logs WARNING level for visibility


def test_M3_preflight_startup_unsafe_combo_not_silently_activated() -> None:
    """If an operator sets ONLY the prod_allow_active flag (e.g. via
    a copy-paste error) but does NOT add ``production`` to the env
    allowlist, the guard refuses to boot. This proves the unsafe
    state is detected, not silently activated."""
    cfg = SemanticRepairActiveConfig(
        enabled=True, kill_switch=False,
        env_allowlist=("dev", "preview"),  # production NOT here
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=5000, prod_allow_active=True,
        current_env="production",
    )
    with pytest.raises(RuntimeError):
        verify_semantic_repair_active_startup(
            is_production=True, config=cfg, env={},
        )
