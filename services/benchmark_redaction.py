"""Phase 6C — Benchmark-run redaction.

Closed-set allowlist filter that strips any field that could contain
PHI, raw evidence text, raw LLM prompts, secrets, or uploaded full
text. Used by the admin "Download REDACTED JSON" button on the
benchmark dashboard.

Implementation is intentionally minimal and explicit:
- The top-level allowlist names exactly the fields that are safe to
  expose on a redacted payload.
- For nested ``per_case_three_mode`` rows, a per-row allowlist also
  enumerates safe fields and truncates ``title`` strings to 200
  characters.
- Anything not on the allowlist is dropped. No regex sniffing, no
  best-effort redaction — the allowlist IS the contract.
"""
from __future__ import annotations

from typing import Any

_TOP_LEVEL_ALLOWLIST: set[str] = {
    "id",
    "schema_version",
    "created_at",
    "execution_mode",
    "expansion_mode",
    "mode",  # legacy alias of expansion_mode
    "three_mode",
    "comparison_report",
    "aggregate_metrics",
    "summary_metrics",
    "latency_json",
    "source_api_errors_json",
    "recommendation",
    "recommendation_inputs_json",
    "ranked_cases_count",
    "evaluated_at",
}

_PER_CASE_ALLOWLIST: set[str] = {
    "benchmark_case_id",
    "provisional",
    "placeholder_gold",
    "status",
    "parser_pass",
    "retrieval_pass",
    "by_mode",
    "delta_analysis",
    "diagnostic_notes",
    "top_10_baseline_json",
    "top_10_active_simulation_json",
    "top_10_before_expansion",
    "top_10_after_active_simulation",
    "expansion_candidates_json",
    "promoted_candidates_json",
    "demoted_candidates_json",
    "source_api_errors_json",
    "latency_json",
}

_TITLE_MAX_LEN = 200


def _truncate_title(value: Any) -> Any:
    if isinstance(value, str):
        return value[:_TITLE_MAX_LEN]
    return value


def _scrub_candidate(cand: dict[str, Any]) -> dict[str, Any]:
    """Per-candidate allowlist: pmid + truncated title + closed-set
    metadata only. NEVER pass through raw abstracts, full text, or
    free-form fields."""
    safe: dict[str, Any] = {}
    for k in ("pmid", "candidate_source", "candidate_tier"):
        if k in cand:
            safe[k] = cand[k]
    if "title" in cand:
        safe["title"] = _truncate_title(cand["title"])
    return safe


def _scrub_per_case(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for k, v in row.items():
        if k not in _PER_CASE_ALLOWLIST:
            continue
        if k in (
            "expansion_candidates_json",
            "promoted_candidates_json",
            "demoted_candidates_json",
        ) and isinstance(v, list):
            safe[k] = [_scrub_candidate(c) if isinstance(c, dict) else {} for c in v]
        elif k in (
            "top_10_baseline_json",
            "top_10_active_simulation_json",
            "top_10_before_expansion",
            "top_10_after_active_simulation",
        ):
            # Pure list of PMID strings; pass through after stringifying.
            if isinstance(v, list):
                safe[k] = [str(p) for p in v if p is not None]
            else:
                safe[k] = []
        else:
            safe[k] = v
    return safe


def redact_benchmark_run(doc: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted copy of a benchmark-run document."""
    out: dict[str, Any] = {}
    for k, v in (doc or {}).items():
        if k not in _TOP_LEVEL_ALLOWLIST:
            continue
        if k == "comparison_report" and isinstance(v, dict):
            cr_safe: dict[str, Any] = {}
            for ck, cv in v.items():
                if ck == "per_case_three_mode" and isinstance(cv, list):
                    cr_safe[ck] = [
                        _scrub_per_case(row) for row in cv
                        if isinstance(row, dict)
                    ]
                elif ck in ("safety_signals", "schema_version",
                            "modes_evaluated"):
                    cr_safe[ck] = cv
            out[k] = cr_safe
        else:
            out[k] = v
    # Always flag that the payload is redacted.
    out["_redacted"] = True
    out["_redaction_schema"] = "phase6c_v1"
    return out


__all__ = ["redact_benchmark_run"]
