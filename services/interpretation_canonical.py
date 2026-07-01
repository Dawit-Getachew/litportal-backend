"""Phase 6B.2 — Canonical interpretation helpers.

Single source of truth for *what the Clinical Context Engine says this
question is*. Rankers, summarizers, and verifiers MUST call into this
module instead of reading ``picott.type_of_question`` or
``clinical_question_type`` directly.

The cascade mirrors ``services/interpreted_question_display.py``:

  1. ``picott.type_of_question`` when set AND not literally "other"
  2. ``role_weights_json.type_of_question`` when set AND not "other"
     (the Clinical Context Engine routinely re-classifies "other" → a
     concrete type via subtype heuristics + role weighting)
  3. ``multi_intent.sub_questions[0].subtype``
  4. legacy ``clinical_question_type`` when set AND not "other"
  5. literal "other" (only when truly nothing else is known)

The helper returns BOTH the raw token AND a humanized display label so
callers can stop manually title-casing strings everywhere.
"""
from __future__ import annotations

from typing import Any


def _humanize(toq: str, subtype: str | None) -> str:
    label = (toq or "").replace("_", " ").strip().title() or "Other"
    if subtype:
        sub = subtype.replace("_", " ").strip()
        if sub:
            return f"{label} / {sub}"
    return label


# Phase 6B.3 — closed-set subtype → parent ``type_of_question`` mapping.
# Promoted to module scope in Phase 6B.3.1 so the
# ``interpreted_question_display`` serializer + new tests can import
# the table and stay in sync with this helper.
_SUBTYPE_PARENT_TYPE: dict[str, str] = {
    # Pharmacology family.
    "metabolic_biomarker_effect": "pharmacology",
    "drug_drug_interaction": "pharmacology",
    "drug_class_effect": "pharmacology",
    "drug question": "pharmacology",
    # Therapy family.
    "transfusion_threshold_comparison": "therapy",
    "noninferiority_or_superiority": "therapy",
    # Dosing family.
    "timing_dose_protocol": "dosing_protocol",
    "dosing or protocol question": "dosing_protocol",
    # Diagnosis / harm.
    "diagnostic_accuracy": "diagnosis",
    "drug_safety_signal": "etiology_harm",
    "exposure_outcome": "etiology_harm",
    # Vaccine / immunization.
    "immunization_schedule": "vaccine_schedule",
    "vaccine_effectiveness_benefit": "vaccine_schedule",
}


def canonical_question_type(
    *,
    picott: dict | None = None,
    role_weights: dict | None = None,
    multi_intent: dict | None = None,
    legacy_qtype: str | None = None,
) -> dict[str, Any]:
    """Resolve the canonical question type for downstream consumers.

    Returns
    -------
    ``{"raw": str, "humanized": str, "subtype": str|None, "source": str}``

    ``source`` ∈ {``picott``, ``role_weights``, ``multi_intent``,
    ``legacy``, ``other``}. Callers should prefer ``humanized`` for
    UI strings (e.g., the "Question type:" prefix in rank_explanation)
    and ``raw`` for table lookups (e.g., design weights).
    """
    picott = picott or {}
    role_weights = role_weights or {}
    multi_intent = multi_intent or {}

    # 1. PICOTT type_of_question wins when concrete.
    picott_toq = (picott.get("type_of_question") or "").strip().lower()
    picott_subtype = (picott.get("question_subtype") or "").strip() or None
    if picott_toq and picott_toq != "other":
        return {
            "raw": picott_toq,
            "humanized": _humanize(picott_toq, picott_subtype),
            "subtype": picott_subtype,
            "source": "picott",
        }

    # 2. Role-weights override (the Clinical Context Engine repaired
    #    type_of_question from subtype heuristics + ontology).
    rw_toq = (role_weights.get("type_of_question") or "").strip().lower()
    rw_subtype = (role_weights.get("question_subtype") or "").strip() or None
    if rw_toq and rw_toq != "other":
        return {
            "raw": rw_toq,
            "humanized": _humanize(rw_toq, rw_subtype or picott_subtype),
            "subtype": rw_subtype or picott_subtype,
            "source": "role_weights",
        }
    # Phase 6B.3 — when ``type_of_question`` is "other" on BOTH PICOTT
    # and role_weights but a concrete ``question_subtype`` survives,
    # the brief mandates we display the subtype rather than the literal
    # "other". Map the subtype to its closed-set parent type when known;
    # otherwise display the subtype string itself.
    candidate_subtype = rw_subtype or picott_subtype
    if candidate_subtype:
        parent = _SUBTYPE_PARENT_TYPE.get(candidate_subtype.lower())
        if parent:
            return {
                "raw": parent,
                "humanized": _humanize(parent, candidate_subtype),
                "subtype": candidate_subtype,
                "source": "role_weights_subtype",
            }
        # No parent mapping — display the bare subtype rather than "other".
        return {
            "raw": "other",
            "humanized": _humanize("other", candidate_subtype),
            "subtype": candidate_subtype,
            "source": "role_weights_subtype",
        }

    # 3. Multi-intent first sub-question's subtype.
    sub_qs = multi_intent.get("sub_questions") or []
    if sub_qs:
        first = sub_qs[0] or {}
        mi_sub = (first.get("subtype") or first.get("question_subtype")
                  or "").strip()
        if mi_sub and mi_sub.lower() != "other":
            # Use the subtype as the raw type if it's in ROLE_WEIGHTS;
            # otherwise keep ``other`` raw but display the subtype.
            from services.role_weighting import ROLE_WEIGHTS  # noqa: WPS433
            raw_candidate = mi_sub.lower()
            if raw_candidate in ROLE_WEIGHTS:
                return {
                    "raw": raw_candidate,
                    "humanized": _humanize(raw_candidate, None),
                    "subtype": None,
                    "source": "multi_intent",
                }
            return {
                "raw": "other",
                "humanized": _humanize("other", mi_sub),
                "subtype": mi_sub,
                "source": "multi_intent",
            }

    # 4. Legacy clinical_question_type fallback.
    legacy = (legacy_qtype or "").strip().lower()
    if legacy and legacy != "other":
        return {
            "raw": legacy,
            "humanized": _humanize(legacy, None),
            "subtype": None,
            "source": "legacy",
        }

    # 5. Truly nothing known.
    return {
        "raw": "other",
        "humanized": "other",
        "subtype": None,
        "source": "other",
    }


def canonical_label_for_rank_explanation(
    *,
    picott: dict | None = None,
    role_weights: dict | None = None,
    multi_intent: dict | None = None,
    legacy_qtype: str | None = None,
) -> str:
    """Convenience — return ONLY the humanized label, suitable to drop
    into a ``"Question type: <label>"`` rank_explanation header.

    NEVER returns "Other" when the engine has a concrete classification.
    Returns the literal "other" string only when source=="other" so the
    handful of legacy tests that assert that prefix still pass.
    """
    out = canonical_question_type(
        picott=picott, role_weights=role_weights,
        multi_intent=multi_intent, legacy_qtype=legacy_qtype,
    )
    if out["source"] == "other":
        return "other"
    return out["humanized"]


__all__ = [
    "canonical_question_type",
    "canonical_label_for_rank_explanation",
]
