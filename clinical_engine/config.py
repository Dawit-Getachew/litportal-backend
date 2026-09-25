"""Engine settings. Class attributes read from env at import (the repo's
pattern — see core/config.py); tests override with monkeypatch.setattr."""
from __future__ import annotations

import os


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


class EngineSettings:
    # Master switch. Default OFF: routes answer 404 and nothing runs at startup.
    ENABLED: bool = _bool("CLINICAL_ENGINE_ENABLED", False)

    # Retrieval adapter used after approval (Week 1). "legacy_pipeline" starts the
    # existing search pipeline with the approved question; "mock" returns labelled
    # fixture candidates (demo/tests only — never presented as live evidence).
    RETRIEVAL_ADAPTER: str = os.environ.get("CLINICAL_ENGINE_RETRIEVAL_ADAPTER", "legacy_pipeline")

    # Interpretation (pre-approval) limits.
    INTERPRETER_MODEL: str = os.environ.get("CLINICAL_ENGINE_INTERPRETER_MODEL", "")
    INTERPRETER_MAX_TOKENS: int = _int("CLINICAL_ENGINE_INTERPRETER_MAX_TOKENS", 900)
    INTERPRETER_TIMEOUT_S: float = _float("CLINICAL_ENGINE_INTERPRETER_TIMEOUT_S", 20.0)
    MAX_QUESTION_CHARS: int = _int("CLINICAL_ENGINE_MAX_QUESTION_CHARS", 1000)
    MAX_VERSIONS_PER_QUESTION: int = _int("CLINICAL_ENGINE_MAX_VERSIONS_PER_QUESTION", 20)
    MAX_INTERPRETATIONS_PER_HOUR: int = _int("CLINICAL_ENGINE_MAX_INTERPRETATIONS_PER_HOUR", 60)

    # Per-run budgets (enforced by the tool gateway).
    RUN_MAX_TOOL_CALLS: int = _int("CLINICAL_ENGINE_RUN_MAX_TOOL_CALLS", 20)
    RUN_MAX_LLM_TOKENS: int = _int("CLINICAL_ENGINE_RUN_MAX_LLM_TOKENS", 20000)
    RUN_MAX_WALL_SECONDS: int = _int("CLINICAL_ENGINE_RUN_MAX_WALL_SECONDS", 1800)
    TASK_MAX_ATTEMPTS: int = _int("CLINICAL_ENGINE_TASK_MAX_ATTEMPTS", 3)
    RETRY_BACKOFF_BASE_S: float = _float("CLINICAL_ENGINE_RETRY_BACKOFF_BASE_S", 2.0)

    # Orchestrator mechanics.
    LEASE_SECONDS: int = _int("CLINICAL_ENGINE_LEASE_SECONDS", 60)
    POLL_INTERVAL_S: float = _float("CLINICAL_ENGINE_POLL_INTERVAL_S", 3.0)
    RECOVERY_INTERVAL_S: float = _float("CLINICAL_ENGINE_RECOVERY_INTERVAL_S", 30.0)

    # "Answer for me" delegated selection policy (Week-1 bridge to the legacy results).
    DELEGATED_MAX_ITEMS: int = _int("CLINICAL_ENGINE_DELEGATED_MAX_ITEMS", 8)


settings = EngineSettings
