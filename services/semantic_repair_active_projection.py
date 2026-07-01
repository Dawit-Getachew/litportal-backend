# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M3-preflight / M3.2 compiler-input
# projection, superseded by services.m4_compiler_input_projection. NOT invoked in
# the active pipeline (gated off via
# pipeline._LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED=False). Retained for
# service-level unit tests only. Do not re-wire into the forward semantic path.

"""Phase M3-preflight — Structural would-be projection for active
semantic repair.

DETERMINISTIC. NO LLM. NO RETRIEVAL. NO SOURCE DISPATCH.

Given:
  * the deterministic question parser's current ClinicalQuestionMap
    fields (primary_intent / frameworks / required_anchors / branch
    count)
  * the validator-accepted semantic-repair candidate

…produces a diagnostics-only projection of what the deterministic
Boolean query compiler **would receive** as inputs if M3 were active
on this session. The closed-set output includes a structural diff
against the current active inputs.

**Scope note for M3-preflight:** invoking the deterministic Boolean
query compiler in dry-run mode (returning would-be Boolean query
strings without dispatch) is **deferred to M3 active**. The compiler
is tightly coupled with source dispatch in ``services/pipeline.py``
and restructuring it would violate the M3-preflight no-touch list.

This module surfaces the would-be **inputs** to the compiler — which
is the information an operator needs to decide whether to grant M3
sign-off — and emits a closed-set
``compiler_dry_run_status="deferred_to_m3_active"`` field documenting
the scoping decision.
"""
from __future__ import annotations

from typing import Any, Optional


COMPILER_DRY_RUN_STATUSES: frozenset[str] = frozenset({
    "not_attempted",
    "deferred_to_m3_active",
    "would_invoke_in_m3_active",
})


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _list_of_str(xs: Any) -> list[str]:
    if not isinstance(xs, list):
        return []
    return [str(x).strip() for x in xs if isinstance(x, str) and x.strip()]


def project_would_be_compiler_input(
    *,
    candidate_dict: Optional[dict],
    current_primary_intent: Optional[str],
    current_frameworks: list[str],
    current_required_anchors: list[str],
    current_branch_count: int,
) -> dict[str, Any]:
    """Compute the closed-set structural projection of compiler inputs
    that M3 active would synthesize from the validator-accepted
    candidate. Returns a closed-set dict with:

        would_be_primary_intent      str | None
        would_be_frameworks          list[str]
        would_be_required_anchors    list[str]
        would_be_branch_count        int
        would_be_semantic_repair_reason  str | None
        would_be_diff_against_current    {added/removed/changed}
        compiler_dry_run_status      closed-set enum
        active_retrieval_changed     bool  (always False in preflight)
    """
    cand = candidate_dict or {}

    would_be_primary_intent: Optional[str] = (
        cand.get("primary_intent") if isinstance(cand, dict) else None
    )
    would_be_frameworks = _list_of_str(cand.get("frameworks") if isinstance(cand, dict) else None)
    would_be_required_anchors = _list_of_str(
        cand.get("required_anchors") if isinstance(cand, dict) else None,
    )
    would_be_branches = (
        cand.get("branches") if isinstance(cand, dict) else None
    ) or []
    would_be_branch_count = int(len(would_be_branches)) if isinstance(
        would_be_branches, list,
    ) else 0
    would_be_semantic_repair_reason: Optional[str] = (
        cand.get("semantic_repair_reason") if isinstance(cand, dict) else None
    )

    # Diff against current active inputs.
    cur_intent = current_primary_intent or None
    cur_frameworks = _list_of_str(current_frameworks)
    cur_required_anchors = _list_of_str(current_required_anchors)

    diff: dict[str, Any] = {
        "intent": {
            "current": cur_intent,
            "would_be": would_be_primary_intent,
            "changed": _norm(cur_intent) != _norm(would_be_primary_intent),
        },
        "frameworks": {
            "current": cur_frameworks,
            "would_be": would_be_frameworks,
            "added": [
                fw for fw in would_be_frameworks if fw not in cur_frameworks
            ],
            "removed": [
                fw for fw in cur_frameworks if fw not in would_be_frameworks
            ],
            "changed": set(map(_norm, would_be_frameworks))
                       != set(map(_norm, cur_frameworks)),
        },
        "required_anchors": {
            "current": cur_required_anchors,
            "would_be": would_be_required_anchors,
            "added": [
                a for a in would_be_required_anchors if a not in cur_required_anchors
            ],
            "removed": [
                a for a in cur_required_anchors if a not in would_be_required_anchors
            ],
            "changed": set(map(_norm, would_be_required_anchors))
                       != set(map(_norm, cur_required_anchors)),
        },
        "branch_count": {
            "current": int(current_branch_count or 0),
            "would_be": would_be_branch_count,
            "changed": int(current_branch_count or 0) != would_be_branch_count,
        },
    }

    return {
        "would_be_primary_intent": would_be_primary_intent,
        "would_be_frameworks": would_be_frameworks,
        "would_be_required_anchors": would_be_required_anchors,
        "would_be_branch_count": would_be_branch_count,
        "would_be_semantic_repair_reason": would_be_semantic_repair_reason,
        "would_be_diff_against_current": diff,
        # M3-preflight scope note: structural projection only. The
        # Boolean query compiler is NOT invoked in dry-run mode here.
        # That deliverable is gated on M3 active phase (separate brief).
        "compiler_dry_run_status": "deferred_to_m3_active",
        # Asserted invariant: M3-preflight NEVER promotes the
        # candidate. Active retrieval queries are byte-identical to
        # pre-M3-preflight behavior on this session.
        "active_retrieval_changed": False,
    }


__all__ = [
    "COMPILER_DRY_RUN_STATUSES",
    "project_would_be_compiler_input",
]
