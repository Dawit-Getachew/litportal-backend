"""Phase Release (M4.2 consolidation) — production hard-off pre-deploy proofs.

Simulates the PRODUCTION deployment env and proves:
  * M4 / M3.2 are HARD-OFF in production (effective_enabled False) under
    BOTH the safe stance AND the worst-case "preview active flags carried".
  * The fail-closed startup guard PASSES under the safe production stance
    and CRASHES (fail-closed) if the preview active flags are carried to
    production without the explicit prod-allow flag — documenting the
    deploy requirement that production env must use the safe stance.
  * cap enforcement resolves to ``disabled`` and the legacy semantic
    runtime is gated off.

No live process env is touched. These are the pre-deploy invariants for
shipping the accepted M4.2 build to https://clinician-answers.emergent.host.
"""
from __future__ import annotations

import pytest

import services.pipeline as pipeline
from services.cap_enforcement_guard import (
    _resolved_cap_enforcement_mode,
    verify_startup_fingerprint,
)
from services.m4_active_config import load_m4_active_config
from services.m4_active_guard import verify_m4_active_startup
from services.semantic_repair_active_config import load_active_config
from services.semantic_repair_active_guard import (
    verify_m3_1_benchmark_startup,
    verify_semantic_repair_active_startup,
)

# The 5 production env vars the operator confirmed remain.
_PROD_BASE = {
    "APP_ENV": "production",
    "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE": "disabled",
    "SEED_TEST_USERS": "false",
    "DEV_FORCE_ENABLE_PUBMED": "false",
    "CORS_ORIGINS": "https://clinician-answers.emergent.host",
    "M3_1_BENCHMARK_ENABLED": "false",
}

# SAFE production stance (what production MUST run): active engines OFF +
# kill engaged (or simply omitted → these are the code defaults).
PROD_ENV_SAFE = dict(
    _PROD_BASE,
    M4_ACTIVE_ENABLED="false",
    M4_ACTIVE_KILL_SWITCH="true",
    SEMANTIC_REPAIR_ACTIVE_ENABLED="false",
    SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH="true",
)

# WORST CASE: the preview .env active flags get carried to production
# (enabled=true, kill disengaged, no PROD_ALLOW_ACTIVE).
PROD_ENV_PREVIEW_FLAGS = dict(
    _PROD_BASE,
    M4_ACTIVE_ENABLED="true",
    M4_ACTIVE_KILL_SWITCH="false",
    M4_ACTIVE_ENV_ALLOWLIST="development,dev,preview,staging",
    SEMANTIC_REPAIR_ACTIVE_ENABLED="true",
    SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH="false",
)


# ---- HARD-OFF holds regardless of how the flags arrive ----

@pytest.mark.parametrize("env", [PROD_ENV_SAFE, PROD_ENV_PREVIEW_FLAGS])
def test_m4_effective_hard_off_in_production(env):
    cfg = load_m4_active_config(env=env)
    assert cfg.is_production_env is True
    assert cfg.prod_allow_active is False
    assert cfg.effective_enabled is False          # HARD-OFF


@pytest.mark.parametrize("env", [PROD_ENV_SAFE, PROD_ENV_PREVIEW_FLAGS])
def test_m3_2_effective_hard_off_in_production(env):
    cfg = load_active_config(env=env)
    assert cfg.is_production_env is True
    assert cfg.effective_enabled is False


def test_m4_hard_off_even_if_production_misadded_to_allowlist():
    env = dict(PROD_ENV_PREVIEW_FLAGS,
               M4_ACTIVE_ENV_ALLOWLIST="development,preview,production")
    cfg = load_m4_active_config(env=env)
    assert cfg.env_in_allowlist is True
    # is_production_env + prod_allow_active=False still forces hard-off.
    assert cfg.effective_enabled is False


# ---- Startup guard: PASSES under the safe stance ----

def test_m4_startup_guard_passes_with_safe_production_stance():
    verify_m4_active_startup(is_production=True, env=PROD_ENV_SAFE)


def test_m3_2_startup_guard_passes_with_safe_production_stance():
    verify_semantic_repair_active_startup(is_production=True, env=PROD_ENV_SAFE)


# ---- Startup guard: FAILS CLOSED (crashes boot) if preview active flags
# leak to production. This is the SAFE design — it prevents accidental
# production activation. It also documents the DEPLOY REQUIREMENT that
# production env must NOT carry enabled=true + kill=false. ----

def test_m4_startup_guard_fails_closed_if_preview_flags_carried():
    with pytest.raises(RuntimeError):
        verify_m4_active_startup(is_production=True, env=PROD_ENV_PREVIEW_FLAGS)


def test_m3_2_startup_guard_fails_closed_if_preview_flags_carried():
    with pytest.raises(RuntimeError):
        verify_semantic_repair_active_startup(
            is_production=True, env=PROD_ENV_PREVIEW_FLAGS)


# ---- Other production posture invariants ----

def test_m3_1_benchmark_guard_safe_in_production():
    verify_m3_1_benchmark_startup(is_production=True, env=PROD_ENV_SAFE)


def test_cap_enforcement_disabled_in_production():
    assert _resolved_cap_enforcement_mode(env=PROD_ENV_SAFE) == "disabled"


def test_cap_startup_fingerprint_safe_in_production():
    verify_startup_fingerprint(
        is_production=True, resolved_mode="disabled", env=PROD_ENV_SAFE)


def test_legacy_semantic_runtime_gated_off():
    assert pipeline._LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED is False
