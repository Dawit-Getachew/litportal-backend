"""Phase 5F Task 3 — RoleWeightingService.

Per-question-type PICOTT role weighting. The weights are applied as a
relevance-score modifier ONLY — Phase 5A mandatory-anchor gates still
apply with lowest-wins after the role-weighted score is computed.

The weights live in ``ROLE_WEIGHTS`` (closed-set; integer percent values).
Each question type's weights sum to ~100. Roles not listed for a given
question type fall back to a small neutral default (5).

Public API:
  - ``role_weight(type_of_question, role) -> int`` — return percentage weight
  - ``weighted_relevance(base_relevance, matched_roles, missing_roles,
       type_of_question) -> int`` — apply a deterministic blend

Hard rule: this service NEVER bypasses Phase 5A anchor caps. The caller is
expected to apply the anchor cap (``min(weighted, cap)``) AFTER calling
``weighted_relevance``. This file documents the contract; the ranker
enforces it.
"""
from __future__ import annotations



# ---------------------------------------------------------------------------
# Closed-set role-weight table.
# Each entry's values are integer percent (0-100) and SHOULD sum to ~100.
# ---------------------------------------------------------------------------
ROLE_WEIGHTS: dict[str, dict[str, int]] = {
    "therapy": {
        "intervention": 35, "population": 25, "outcome": 20,
        "comparator": 15, "study_design": 5,
    },
    "pharmacology": {
        "intervention": 40, "outcome": 25, "population": 20,
        "comparator": 10, "study_design": 5,
    },
    "dosing_protocol": {
        "intervention": 35, "outcome": 20, "population": 20,
        "dose_terms": 15, "study_design": 10,
    },
    "diagnosis": {
        "index_test": 35, "target_condition": 25,
        "reference_standard": 20, "outcome": 15, "study_design": 5,
    },
    "etiology_harm": {
        "exposure": 35, "outcome": 25, "population": 25,
        "study_design": 15,
    },
    "prognosis": {
        "outcome": 35, "population": 30, "time_horizon": 15,
        "study_design": 20,
    },
    "prevention": {
        "intervention": 30, "population": 25, "outcome": 25,
        "comparator": 15, "study_design": 5,
    },
    "vaccine_schedule": {
        "intervention": 30, "schedule_terms": 25, "population": 25,
        "outcome": 15, "study_design": 5,
    },
    "background": {
        "concept_clarity": 50, "review_or_guideline": 30,
        "specialty_match": 20,
    },
    "other": {
        "population": 30, "intervention": 30, "outcome": 25,
        "study_design": 15,
    },
}

# Role-name normalization for callers (legacy/parser-emitted names →
# canonical ROLE_WEIGHTS keys).
_ROLE_ALIASES: dict[str, str] = {
    "intervention_or_exposure": "intervention",
    "intervention_or_exposure_or_index_test": "intervention",
    "drug": "intervention",
    "drug_class": "intervention",
    "vaccine": "intervention",
    "exposure": "exposure",
    "index_test": "index_test",
    "comparator_or_reference_standard": "comparator",
    "reference_standard": "reference_standard",
    "outcomes_or_target_diagnosis": "outcome",
    "target_diagnosis": "outcome",
    "target_condition": "target_condition",
    "condition": "population",
    "patient_group": "population",
    "biomarker": "outcome",
    "lab": "outcome",
    "study_design": "study_design",
    "time_horizon": "time_horizon",
    "dose_terms": "dose_terms",
    "schedule_terms": "schedule_terms",
    "concept_clarity": "concept_clarity",
    "specialty_match": "specialty_match",
    "review_or_guideline": "review_or_guideline",
}

_NEUTRAL_WEIGHT = 5  # fallback for unknown role keys


def _canonical_qtype(toq: str | None) -> str:
    t = (toq or "").strip().lower()
    return t if t in ROLE_WEIGHTS else "other"


def _canonical_role(role: str | None) -> str:
    r = (role or "").strip().lower()
    return _ROLE_ALIASES.get(r, r)


def role_weight(type_of_question: str, role: str) -> int:
    """Return the integer percent weight for ``role`` in ``type_of_question``.

    Falls back to ``_NEUTRAL_WEIGHT`` when the role isn't listed for the
    question type. Unknown question types use the ``"other"`` table.
    """
    table = ROLE_WEIGHTS.get(_canonical_qtype(type_of_question)) \
            or ROLE_WEIGHTS["other"]
    return int(table.get(_canonical_role(role), _NEUTRAL_WEIGHT))


def weighted_relevance(
    *,
    base_relevance: float,
    matched_roles: list[str],
    missing_roles: list[str] | None = None,
    type_of_question: str,
) -> int:
    """Compute a role-weighted relevance score on a 0-100 scale.

    Algorithm (deterministic):
      1. For each matched role, add ``role_weight(type, role)`` to a
         ``coverage`` accumulator (capped at 100).
      2. ``weighted = 0.6 * base_relevance + 0.4 * coverage``. This
         blends the existing relevance signal with the role coverage so
         the ranker doesn't regress on questions that already produced
         a strong base relevance.
      3. Penalize each missing high-weight role (weight >= 25) by 5
         points (capped at 20 total).
      4. Clamp to [0, 100] and return as int.

    Phase 5A anchor caps (``relevance_cap_applied``) MUST be applied by
    the caller AFTER this function returns. This service does not look
    at caps.
    """
    qtype = _canonical_qtype(type_of_question)
    matched_clean = {_canonical_role(r) for r in (matched_roles or []) if r}
    coverage = sum(
        role_weight(qtype, r) for r in matched_clean
    )
    coverage = min(100, coverage)
    blended = 0.6 * float(base_relevance) + 0.4 * float(coverage)

    # Penalty for missing high-weight roles.
    penalty = 0
    if missing_roles:
        for r in missing_roles:
            if role_weight(qtype, r) >= 25:
                penalty += 5
        penalty = min(20, penalty)
    blended -= penalty

    return int(max(0, min(100, round(blended))))


class RoleWeightingService:
    """Stateless wrapper. ``weights_for(qtype)`` returns a *copy* of the
    weight dict for diagnostics."""

    def weights_for(self, type_of_question: str) -> dict[str, int]:
        qtype = _canonical_qtype(type_of_question)
        return dict(ROLE_WEIGHTS.get(qtype) or ROLE_WEIGHTS["other"])

    def role_weight(self, type_of_question: str, role: str) -> int:
        return role_weight(type_of_question, role)

    def weighted_relevance(
        self, *, base_relevance: float, matched_roles: list[str],
        missing_roles: list[str] | None = None,
        type_of_question: str,
    ) -> int:
        return weighted_relevance(
            base_relevance=base_relevance,
            matched_roles=matched_roles,
            missing_roles=missing_roles,
            type_of_question=type_of_question,
        )


role_weighting = RoleWeightingService()


__all__ = [
    "ROLE_WEIGHTS",
    "RoleWeightingService",
    "role_weighting",
    "role_weight",
    "weighted_relevance",
]
