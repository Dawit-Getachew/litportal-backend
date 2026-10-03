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
    # Master switch. Default ON; CLINICAL_ENGINE_ENABLED=false makes every route
    # except /capabilities answer 404 and starts nothing at boot.
    ENABLED: bool = _bool("CLINICAL_ENGINE_ENABLED", True)

    # Retrieval adapter used after approval (Week 1). "legacy_pipeline" starts the
    # existing search pipeline with the approved question; "mock" returns labelled
    # fixture candidates (demo/tests only — never presented as live evidence).
    RETRIEVAL_ADAPTER: str = os.environ.get("CLINICAL_ENGINE_RETRIEVAL_ADAPTER", "engine")
    # Week 2: "engine" runs the engine's own lanes (portfolio → retrieval → union →
    # admission → lineage → list → access) for questions with approved search terms;
    # questions without reviewed terms still use "legacy_pipeline".

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
    # Engine-retrieval runs make many more calls: one per source query, per model
    # batch and per access check.
    ENGINE_RUN_MAX_TOOL_CALLS: int = _int("CLINICAL_ENGINE_ENGINE_RUN_MAX_TOOL_CALLS", 220)
    ENGINE_RUN_MAX_LLM_TOKENS: int = _int("CLINICAL_ENGINE_ENGINE_RUN_MAX_LLM_TOKENS", 220000)
    TASK_MAX_ATTEMPTS: int = _int("CLINICAL_ENGINE_TASK_MAX_ATTEMPTS", 3)
    RETRY_BACKOFF_BASE_S: float = _float("CLINICAL_ENGINE_RETRY_BACKOFF_BASE_S", 2.0)

    # Orchestrator mechanics.
    LEASE_SECONDS: int = _int("CLINICAL_ENGINE_LEASE_SECONDS", 60)
    POLL_INTERVAL_S: float = _float("CLINICAL_ENGINE_POLL_INTERVAL_S", 3.0)
    RECOVERY_INTERVAL_S: float = _float("CLINICAL_ENGINE_RECOVERY_INTERVAL_S", 30.0)

    # "Answer for me" delegated selection policy (Week-1 bridge to the legacy results).
    DELEGATED_MAX_ITEMS: int = _int("CLINICAL_ENGINE_DELEGATED_MAX_ITEMS", 8)

    # Search-terms review (second approval step): MeSH lookup + term classification.
    MESH_TIMEOUT_S: float = _float("CLINICAL_ENGINE_MESH_TIMEOUT_S", 12.0)
    # Model for classifying search terms (blank = the interpreter model). A stronger
    # model gives cleaner equivalent/narrower/related labels for ~1k tokens per question.
    TERMS_MODEL: str = os.environ.get("CLINICAL_ENGINE_TERMS_MODEL", "")
    TERMS_MAX_TOKENS: int = _int("CLINICAL_ENGINE_TERMS_MAX_TOKENS", 1400)
    TERMS_TIMEOUT_S: float = _float("CLINICAL_ENGINE_TERMS_TIMEOUT_S", 25.0)
    MAX_TERM_REVISIONS: int = _int("CLINICAL_ENGINE_MAX_TERM_REVISIONS", 60)

    # Week-2 retrieval (engine adapter): per-source budgets. A source that runs out
    # of time or fails is reported as a partial search, never as "no results".
    RETRIEVAL_HTTP_TIMEOUT_S: float = _float("CLINICAL_ENGINE_RETRIEVAL_HTTP_TIMEOUT_S", 20.0)
    RETRIEVAL_SOURCE_SECONDS: float = _float("CLINICAL_ENGINE_RETRIEVAL_SOURCE_SECONDS", 75.0)
    RETRIEVAL_MAX_RECORDS_PUBMED: int = _int("CLINICAL_ENGINE_RETRIEVAL_MAX_RECORDS_PUBMED", 700)
    RETRIEVAL_MAX_RECORDS_EUROPEPMC: int = _int("CLINICAL_ENGINE_RETRIEVAL_MAX_RECORDS_EUROPEPMC", 400)
    RETRIEVAL_CONCURRENCY_PER_SOURCE: int = _int("CLINICAL_ENGINE_RETRIEVAL_CONCURRENCY_PER_SOURCE", 3)
    # The bounded alternative search runs only when the main search matches fewer
    # records than this in total (PubMed + Europe PMC hit counts).
    RETRIEVAL_SPARSE_THRESHOLD: int = _int("CLINICAL_ENGINE_RETRIEVAL_SPARSE_THRESHOLD", 30)

    # Admission (W2.4): the model judges the records the approved searches point at
    # most strongly, in batches. Blank model = the terms model, then the interpreter's.
    ADMISSION_MODEL: str = os.environ.get("CLINICAL_ENGINE_ADMISSION_MODEL", "")
    ADMISSION_MAX_JUDGED: int = _int("CLINICAL_ENGINE_ADMISSION_MAX_JUDGED", 100)
    ADMISSION_AGGREGATED_EXTRA: int = _int("CLINICAL_ENGINE_ADMISSION_AGGREGATED_EXTRA", 40)
    ADMISSION_BATCH_SIZE: int = _int("CLINICAL_ENGINE_ADMISSION_BATCH_SIZE", 10)
    ADMISSION_CONCURRENCY: int = _int("CLINICAL_ENGINE_ADMISSION_CONCURRENCY", 4)
    ADMISSION_MAX_TOKENS: int = _int("CLINICAL_ENGINE_ADMISSION_MAX_TOKENS", 3000)
    ADMISSION_TIMEOUT_S: float = _float("CLINICAL_ENGINE_ADMISSION_TIMEOUT_S", 60.0)


settings = EngineSettings
