# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M3-preflight eligibility,
# superseded by the M4 active eligibility decision (services.m4_active_pipeline).
# NOT invoked in the active pipeline (gated off via
# pipeline._LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED=False). Retained for
# service-level unit tests only. Do not re-wire into the forward semantic path.

"""Phase M3-preflight — Active Semantic Repair eligibility decision.

DETERMINISTIC. NO LLM. NO RETRIEVAL.

Pure function that consumes:
  * the deterministic question parser's current map / anchors
  * the bounded LLM semantic-repair candidate (already
    validator-accepted in M2.1)
  * the closed-set ``SemanticRepairActiveConfig``
  * the per-session M-phase diagnostics block

…and returns:

  ``(eligible, pass_reasons, block_reasons, would_be_eligible_if_flags_enabled)``

The ``pass_reasons`` / ``block_reasons`` are closed-set strings
matching the M3-preflight brief. The ``would_be_eligible_if_flags_enabled``
flag answers: "if the operator flipped ``enabled=true``,
``kill_switch=false``, AND ``env=production`` were allowed, would
this session be eligible based on the CLINICAL gates alone?"

In M3-preflight, ``eligible`` is always ``False`` (because the
flags are off and the kill switch is engaged) but ``would_be_eligible_if_flags_enabled``
exposes the would-be decision for operator review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from services.semantic_repair_active_config import (
    SemanticRepairActiveConfig,
)


# Closed-set pass reasons.
ELIGIBILITY_PASS_REASONS: frozenset[str] = frozenset({
    "phi_guard_passed",
    "suspicious_grammar_triggered",
    "candidate_schema_valid",
    "candidate_confidence_meets_threshold",
    "no_unsafe_or_refused_output",
    "no_unvalidated_mesh_marked_validated",
    "non_generic_required_anchor_present",
    "required_anchors_clinically_central",
    "repair_preserves_central_anchors",
    "active_shape_in_allowlist",
    "route_not_owned_by_k4a_specialty",
    "latency_under_threshold",
    "active_flag_enabled",
    "env_in_allowlist",
    "kill_switch_disengaged",
})

# Closed-set block reasons.
ELIGIBILITY_BLOCK_REASONS: frozenset[str] = frozenset({
    "phi_blocked",
    "detector_did_not_trigger",
    "candidate_schema_invalid",
    "candidate_confidence_below_threshold",
    "candidate_refused_or_unsafe",
    "unvalidated_mesh_treated_as_validated",
    "generic_only_required_anchor",
    "central_anchor_removed",
    "shape_not_in_allowlist",
    "route_owned_by_k4a_specialty",
    "latency_above_threshold",
    "active_flag_disabled",
    "env_not_in_allowlist",
    "kill_switch_engaged",
    "production_env_fail_closed",
})


# K4A.*-owned question subtypes. Mirrors the closed set in
# ``services.semantic_anchor_repair._K4A_FIRED_SUBTYPES``.
_K4A_OWNED_SUBTYPES: frozenset[str] = frozenset({
    "severity_marker_overview",
    "rule_out_test_utility",
    "epidemiology_incidence_complication_risk",
})


_SEMANTIC_REASON_TO_SHAPE: dict[str, str] = {
    "missing_risk_etiology_framework": "etiology_risk",
    "missing_mechanism_pathophysiology_framework": "pathophysiology_mechanism",
    "missing_complication_after_procedure_split": "procedure_complication",
    "missing_prognosis_framework": "prognosis",
    "missing_epidemiology_framework_grammar": "epidemiology",
    "missing_severity_marker_split": "severity_marker",
    "missing_diagnostic_ruleout_split": "diagnostic_ruleout",
    "literal_full_phrase_anchor": "etiology_risk",  # fallback — most common parent
    "disease_population_fused_into_single_anchor": "etiology_risk",
}


@dataclass(frozen=True)
class EligibilityDecision:
    eligible: bool
    pass_reasons: tuple[str, ...]
    block_reasons: tuple[str, ...]
    would_be_eligible_if_flags_enabled: bool
    shape: Optional[str]

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "eligible": bool(self.eligible),
            "pass_reasons": list(self.pass_reasons),
            "block_reasons": list(self.block_reasons),
            "would_be_eligible_if_flags_enabled": bool(
                self.would_be_eligible_if_flags_enabled,
            ),
            "shape": self.shape,
        }


def _resolve_shape(
    semantic_repair_reason: Optional[str],
    triggers: Iterable[str],
) -> Optional[str]:
    """Map the candidate's semantic_repair_reason or one of its
    triggers to the active-shape namespace. Returns ``None`` when no
    closed-set mapping is found."""
    if semantic_repair_reason:
        mapped = _SEMANTIC_REASON_TO_SHAPE.get(semantic_repair_reason)
        if mapped:
            return mapped
    for t in triggers or []:
        mapped = _SEMANTIC_REASON_TO_SHAPE.get(t)
        if mapped:
            return mapped
    return None


def is_semantic_repair_active_eligible(
    *,
    session_blocked_phi: bool,
    detector_triggered: bool,
    trigger_reasons: list[str],
    candidate_validated: bool,
    candidate_rejection_reasons: list[str],
    candidate_dict: Optional[dict],
    deterministic_primary_anchors: list[dict],
    question_subtype: Optional[str],
    latency_ms: int,
    config: SemanticRepairActiveConfig,
) -> EligibilityDecision:
    """Pure deterministic decision function. See module docstring."""

    pass_reasons: list[str] = []
    block_reasons: list[str] = []

    # ---- Clinical gates ----

    # 1) PHI dominance — if PHI was detected upstream, this is
    # never eligible.
    if session_blocked_phi:
        block_reasons.append("phi_blocked")
    else:
        pass_reasons.append("phi_guard_passed")

    # 2) Detector triggered.
    if detector_triggered and (trigger_reasons or []):
        pass_reasons.append("suspicious_grammar_triggered")
    else:
        block_reasons.append("detector_did_not_trigger")

    # 3) Candidate schema valid.
    cand = candidate_dict or {}
    if candidate_validated and isinstance(cand, dict) and cand:
        pass_reasons.append("candidate_schema_valid")
    else:
        if "refused" in (candidate_rejection_reasons or []):
            block_reasons.append("candidate_refused_or_unsafe")
        elif any(
            r in (candidate_rejection_reasons or [])
            for r in (
                "phi_pattern_detected",
                "article_reference_detected",
                "unapproved_source_referenced",
                "answer_text_detected",
                "unsafe_text_detected",
            )
        ):
            block_reasons.append("candidate_refused_or_unsafe")
        elif "unvalidated_mesh_marked_validated" in (
            candidate_rejection_reasons or []
        ):
            block_reasons.append("unvalidated_mesh_treated_as_validated")
        elif any(
            r in (candidate_rejection_reasons or [])
            for r in (
                "empty_required_anchors",
                "generic_only_required_anchor",
                "generic_modifier_only_required_anchor",
                "no_valid_required_anchor",
            )
        ):
            block_reasons.append("generic_only_required_anchor")
        elif "would_remove_central_anchor" in (
            candidate_rejection_reasons or []
        ):
            block_reasons.append("central_anchor_removed")
        elif "confidence_below_threshold" in (
            candidate_rejection_reasons or []
        ):
            block_reasons.append("candidate_confidence_below_threshold")
        else:
            block_reasons.append("candidate_schema_invalid")

    # 4) Confidence threshold (read from the candidate).
    if candidate_validated and isinstance(cand, dict):
        conf = float(cand.get("confidence") or 0.0)
        if conf >= 0.70:
            pass_reasons.append("candidate_confidence_meets_threshold")
        else:
            # Already counted under schema gating if rejected.
            pass

    # 5) No unsafe / refused — the validator already enforces this.
    if candidate_validated:
        pass_reasons.append("no_unsafe_or_refused_output")
        pass_reasons.append("no_unvalidated_mesh_marked_validated")

    # 6) Non-generic required anchor present.
    if candidate_validated and isinstance(cand, dict):
        anchors = cand.get("required_anchors") or []
        if anchors:
            pass_reasons.append("non_generic_required_anchor_present")

    # 7) Required anchors clinically central — i.e., they overlap
    # with at least one deterministic primary anchor.
    if candidate_validated and isinstance(cand, dict):
        det_terms = {
            ((a or {}).get("canonical_concept") or (a or {}).get("text") or "")
            .strip().lower()
            for a in (deterministic_primary_anchors or [])
            if isinstance(a, dict)
        }
        det_terms.discard("")
        cand_terms = {
            (t or "").strip().lower()
            for t in (cand.get("required_anchors") or [])
        }
        cand_terms.discard("")
        if det_terms and cand_terms and any(
            (ct in dt or dt in ct)
            for ct in cand_terms for dt in det_terms
        ):
            pass_reasons.append("required_anchors_clinically_central")
            pass_reasons.append("repair_preserves_central_anchors")
        elif not det_terms:
            # No deterministic anchors to compare against — pass
            # by default (the M2.1 validator already enforced
            # ``would_remove_central_anchor`` against those).
            pass_reasons.append("required_anchors_clinically_central")
            pass_reasons.append("repair_preserves_central_anchors")

    # 8) Active-shape allowlist.
    shape = _resolve_shape(
        (cand or {}).get("semantic_repair_reason") if isinstance(cand, dict) else None,
        trigger_reasons or [],
    )
    if shape is not None and config.shape_in_allowlist(shape):
        pass_reasons.append("active_shape_in_allowlist")
    elif shape is not None:
        block_reasons.append("shape_not_in_allowlist")
    else:
        # No shape resolved — treat as not in allowlist.
        block_reasons.append("shape_not_in_allowlist")

    # 9) Route owned by K4A.* specialty. If the upstream parser
    # already routed this question through K4A.2/K4A.3/K4A.4, then
    # M3 must NEVER promote — those routes own the shape.
    st = (question_subtype or "").strip().lower()
    if st in _K4A_OWNED_SUBTYPES:
        block_reasons.append("route_owned_by_k4a_specialty")
    else:
        pass_reasons.append("route_not_owned_by_k4a_specialty")

    # 10) Latency under threshold.
    try:
        latency_int = int(latency_ms or 0)
    except (TypeError, ValueError):
        latency_int = 0
    if latency_int <= int(config.max_latency_ms or 0):
        pass_reasons.append("latency_under_threshold")
    else:
        block_reasons.append("latency_above_threshold")

    # ---- Operational gates ----
    # 11) Active flag.
    if config.enabled:
        pass_reasons.append("active_flag_enabled")
    else:
        block_reasons.append("active_flag_disabled")

    # 12) Env allowlist.
    if config.env_in_allowlist:
        pass_reasons.append("env_in_allowlist")
    else:
        block_reasons.append("env_not_in_allowlist")

    # 13) Kill switch.
    if not config.kill_switch_engaged:
        pass_reasons.append("kill_switch_disengaged")
    else:
        block_reasons.append("kill_switch_engaged")

    # 14) Production fail-closed defense in depth.
    if config.is_production_env and not config.prod_allow_active:
        block_reasons.append("production_env_fail_closed")

    # Dedupe in order.
    pass_reasons = list(dict.fromkeys(pass_reasons))
    block_reasons = list(dict.fromkeys(block_reasons))

    # ---- Decision ----
    operational_block = {
        "active_flag_disabled",
        "kill_switch_engaged",
        "env_not_in_allowlist",
        "production_env_fail_closed",
        # ``latency_above_threshold`` is an SLO knob — operators can
        # tune ``SEMANTIC_REPAIR_ACTIVE_MAX_LATENCY_MS`` upward without
        # changing clinical safety. The would-be-eligible-if-flags-
        # enabled answer should NOT depend on latency.
        "latency_above_threshold",
    }
    clinical_block_set = set(block_reasons) - operational_block
    clinical_eligible = (
        len(clinical_block_set) == 0
        and bool(detector_triggered)
        and bool(candidate_validated)
    )
    eligible = bool(clinical_eligible) and config.effective_enabled

    would_be_eligible_if_flags_enabled = clinical_eligible

    return EligibilityDecision(
        eligible=eligible,
        pass_reasons=tuple(pass_reasons),
        block_reasons=tuple(block_reasons),
        would_be_eligible_if_flags_enabled=would_be_eligible_if_flags_enabled,
        shape=shape,
    )


__all__ = [
    "ELIGIBILITY_PASS_REASONS",
    "ELIGIBILITY_BLOCK_REASONS",
    "EligibilityDecision",
    "is_semantic_repair_active_eligible",
]
