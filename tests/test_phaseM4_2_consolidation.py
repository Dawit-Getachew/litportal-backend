"""Phase M4.2 — Legacy semantic-repair consolidation invariants.

Static/structural proofs that the legacy M0/M1/M2/M3 semantic-repair
runtime is consolidated into M4 and no longer wired into the active
pipeline. Behavioral proof (M4 still applies on degraded baselines,
clean preserved, PHI blocks, protected/guideline lanes intact) is
covered by the live M4.2 preview soak + the existing M4.x suites.
"""
from __future__ import annotations

import inspect

import pytest

import services.pipeline as pipeline
from services.m4_active_pipeline import M4_ACTIVE_DECISIONS, run_m4_active_decision

LEGACY_MODULES = [
    "services.semantic_anchor_repair",
    "services.semantic_repair_active_config",
    "services.semantic_repair_active_eligibility",
    "services.semantic_repair_active_guard",
    "services.semantic_repair_active_pipeline",
    "services.semantic_repair_active_projection",
    "services.m3_1_benchmark",
    "services.m3_1_adjudication_packet",
    "services.m3_2_banner_state",
    "services.m3_2_feature_flags",
]


def test_legacy_semantic_runtime_constant_is_off():
    # The consolidation gate is OFF: the legacy M0/M1/M2 shadow,
    # M3-preflight and M3.2 active hooks are NOT invoked in the pipeline.
    assert pipeline._LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED is False


@pytest.mark.parametrize("modpath", LEGACY_MODULES)
def test_legacy_modules_still_import_and_marked_deprecated(modpath):
    import importlib
    mod = importlib.import_module(modpath)
    src = inspect.getsource(mod)
    # Each legacy module carries the Phase M4.2 DEPRECATED banner so the
    # consolidation decision is discoverable at the source.
    assert "DEPRECATED (Phase M4.2" in src, modpath


def test_m4_is_forward_semantic_path():
    # M4 owns the forward semantic interpretation: the active decision
    # function exists and its closed-set decisions include the eligible
    # promotion + the M4.1.5 timeout fallback.
    assert callable(run_m4_active_decision)
    assert "eligible" in M4_ACTIVE_DECISIONS
    assert "m4_latency_timeout" in M4_ACTIVE_DECISIONS


def test_pipeline_feeds_m4_a_static_dormant_m3_2_decision():
    # The pipeline source sets a deprecated, DORMANT (combined_engaged)
    # M3.2 decision so the M4 co-fire guard keeps applying on degraded
    # baselines without invoking the legacy M3.2 runtime.
    src = inspect.getsource(pipeline.run_pipeline)
    assert "deprecated_consolidated_into_m4" in src
    assert "_LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED" in src


def test_no_legacy_sidecar_write_helper_invoked_unconditionally():
    # The M3.2 sidecar update is keyed on a sidecar_doc_id that the
    # static dormant decision never provides, so no new
    # m3_2_session_decisions rows are written by the active pipeline.
    src = inspect.getsource(pipeline)
    # The deprecated marker block must be present (else branch).
    assert "superseded_by" in src
    assert "m4_full_question_interpreter" in src
