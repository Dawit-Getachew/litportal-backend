"""Phase 6F.2 — session-scoped diagnostics instrumentation.

Closed-set, no PHI. Used by the search-session pipeline to populate the
three previously-null fields in ``session_diagnostics_json``:

  - ``llm_calls_count`` + ``llm_calls_by_stage`` (closed-set stage keys)
  - ``token_estimate`` ``{prompt_tokens, completion_tokens, total_tokens,
    estimate_method}`` where estimate_method ∈ ``{provider_reported,
    char_div_4_estimate, mixed}``
  - ``latency_by_stage_json`` ``{stage_name: ms_int}``

Wiring:
  - ``services/llm_provider.py`` records provider-reported usage (or a
    char/4 fallback) via ``record_llm_call`` whenever a session-scoped
    diagnostics object is bound to the current contextvar.
  - ``services/pipeline.py`` wraps each stage with the
    ``diagnostics.stage(name)`` context manager so latencies sum
    cumulatively per stage (retries land in the same bucket).
  - ``routers/search_sessions.py::create_session`` also wraps the PHI
    guard call with ``diagnostics.stage("phi_guard")`` so PHI-blocked
    sessions still populate the latency block for that single stage.

Safety invariants (must remain true after every change to this file):
  * **NO PHI / NO prompts / NO completions / NO model parameters beyond
    closed-set fields are EVER persisted.** ``to_dict()`` returns only
    integers + closed-set strings.
  * Stage names are CLOSED-SET. Anything not in ``STAGE_NAMES`` raises.
  * The current-stage attribution for an LLM call falls back to
    ``"unknown"`` when no stage is active. ``"unknown"`` is intentionally
    NOT in ``STAGE_NAMES`` and is filtered out of
    ``llm_calls_by_stage`` before persistence.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

# Closed-set stage names per the Phase 6F.2 directive.
# Phase H1 — added ``clinical_context_resolver`` (17th stage) for the
# bounded LLM Clinical Context Resolver that runs ONLY as a fallback
# when the deterministic anchor pipeline cannot produce a primary
# anchor.
STAGE_NAMES: frozenset[str] = frozenset({
    "phi_guard",
    "normalization",
    "primary_clinical_anchor",
    "picott_parsing",
    "clinical_context_resolver",
    "boolean_query_compilation",
    "pubmed_retrieval",
    "europe_pmc_retrieval",
    "openalex_retrieval",
    "deduplication",
    "safety_filter",
    "ranking",
    "default_selection",
    "full_text_access_broker",
    "evidence_packet_preparation",
    "summarizer",
    "citation_verifier",
    # Phase L1 — explicit evidence-lens / population-focus soft-boost stage.
    "l1_evidence_lens",
})

# Closed-set estimate-method labels.
ESTIMATE_METHODS: frozenset[str] = frozenset({
    "provider_reported", "char_div_4_estimate", "mixed",
})


class SessionDiagnostics:
    """Mutable session-scoped accumulator. NOT thread-safe across asyncio
    workers — bind one instance per pipeline invocation.
    """

    def __init__(self) -> None:
        self.llm_calls_count: int = 0
        self.llm_calls_by_stage: dict[str, int] = {}
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self._method_provider: int = 0
        self._method_estimate: int = 0
        self.latency_by_stage: dict[str, int] = {}
        self._current_stage: str = "unknown"

    # ------------------------------------------------------------------
    # LLM call instrumentation (called by llm_provider only)
    # ------------------------------------------------------------------
    def record_llm_call(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        method: str,
        stage: Optional[str] = None,
    ) -> None:
        """Record a single LLM call. ``method`` MUST be one of
        ``ESTIMATE_METHODS``. ``stage`` defaults to the current pipeline
        stage when set, else ``"unknown"`` (which is filtered out before
        persistence).
        """
        if method not in ("provider_reported", "char_div_4_estimate"):
            # Closed-set guard. We intentionally do NOT raise — the
            # diagnostics block is informational and must never break
            # the pipeline. Drop to estimate so the call is still
            # counted.
            method = "char_div_4_estimate"
        s = stage if (stage and stage in STAGE_NAMES) else self._current_stage
        self.llm_calls_count += 1
        if s in STAGE_NAMES:
            self.llm_calls_by_stage[s] = (
                int(self.llm_calls_by_stage.get(s) or 0) + 1
            )
        # Token tallies. Clamp to >= 0 so a bogus provider response
        # cannot make the totals negative.
        self.prompt_tokens += max(0, int(prompt_tokens or 0))
        self.completion_tokens += max(0, int(completion_tokens or 0))
        if method == "provider_reported":
            self._method_provider += 1
        else:
            self._method_estimate += 1

    # ------------------------------------------------------------------
    # Stage timing
    # ------------------------------------------------------------------
    @contextmanager
    def stage(self, name: str):
        """Context manager that times a pipeline stage. Multiple
        invocations of the same stage accumulate (retries / loops land
        in the same bucket). ``name`` MUST be in ``STAGE_NAMES``.
        """
        if name not in STAGE_NAMES:
            raise ValueError(
                f"phase6f2: stage {name!r} not in closed-set STAGE_NAMES"
            )
        prev_stage = self._current_stage
        self._current_stage = name
        start = time.perf_counter()
        try:
            yield self
        finally:
            elapsed_ms = int(round((time.perf_counter() - start) * 1000))
            self.latency_by_stage[name] = (
                int(self.latency_by_stage.get(name) or 0) + elapsed_ms
            )
            self._current_stage = prev_stage

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return the closed-set dict that gets merged into
        ``session_diagnostics_json``. NO PHI, NO prompts, NO completions.
        """
        total = self.prompt_tokens + self.completion_tokens
        if self._method_provider and not self._method_estimate:
            estimate_method = "provider_reported"
        elif self._method_estimate and not self._method_provider:
            estimate_method = "char_div_4_estimate"
        elif self._method_provider and self._method_estimate:
            estimate_method = "mixed"
        else:
            # No LLM calls at all (PHI-blocked, awaiting_clinical_subject).
            # We still label deterministically.
            estimate_method = "provider_reported"
        return {
            "llm_calls_count": int(self.llm_calls_count),
            "llm_calls_by_stage": dict(self.llm_calls_by_stage),
            "token_estimate": {
                "prompt_tokens": int(self.prompt_tokens),
                "completion_tokens": int(self.completion_tokens),
                "total_tokens": int(total),
                "estimate_method": estimate_method,
            },
            "latency_by_stage_json": dict(self.latency_by_stage),
        }


# Session-scoped context. Async pipeline coroutines all share the same
# ``SessionDiagnostics`` instance via this contextvar.
_current: ContextVar[Optional[SessionDiagnostics]] = ContextVar(
    "session_diagnostics", default=None,
)


def bind(diagnostics: SessionDiagnostics) -> object:
    """Bind a diagnostics object to the current async context. Returns
    a token that must be passed to ``reset()`` when the pipeline
    finishes (typical try/finally pattern).
    """
    return _current.set(diagnostics)


def reset(token: object) -> None:
    """Clear the contextvar back to its previous value."""
    try:
        _current.reset(token)  # type: ignore[arg-type]
    except (LookupError, ValueError):
        # Defensive — never let context cleanup raise.
        pass


def get_current() -> Optional[SessionDiagnostics]:
    """Return the currently-bound ``SessionDiagnostics`` or None."""
    return _current.get()


__all__ = [
    "STAGE_NAMES",
    "ESTIMATE_METHODS",
    "SessionDiagnostics",
    "bind",
    "reset",
    "get_current",
]
