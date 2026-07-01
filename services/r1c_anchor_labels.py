"""Phase R1C — closed-set anchor-specific directness labels.

Two new closed-set label catalogues layered ON TOP of R1A's
existing label set (R1A weights UNCHANGED):

  * ``procedure_complication_*`` outcome anchoring labels for the
    ``procedure_complication_risk`` shape (Case D + future
    generalizable parser-driven outcome targets).
  * ``mechanism_target_*`` mechanism-target preservation labels
    for the ``pathophysiology_mechanism`` shape (Cases A/B/C +
    future generalizable parser-driven mechanism targets).

Detection is DETERMINISTIC — case-insensitive whole-word matching
against article title + abstract via regex (NO LLM). Each
benchmark case carries its own curated ``disease_anchor_terms``,
``outcome_anchor_terms``, and ``mechanism_target_terms``.

Ranking multipliers attached to each label so the R1C simulator
can deprioritize misaligned articles without ANY mutation of R1A
weights or R1B configuration.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# =====================================================================
# Closed-set label catalogues.
# =====================================================================
PROCEDURE_OUTCOME_LABELS: tuple[str, ...] = (
    "procedure_complication_outcome_exact",
    "procedure_complication_outcome_missing",
    "procedure_complication_wrong_outcome",
    "procedure_complication_general_not_outcome_specific",
)

MECHANISM_TARGET_LABELS: tuple[str, ...] = (
    "mechanism_target_exact",
    "mechanism_target_missing",
    "disease_only_indirect_for_mechanism",
    "treatment_mentions_mechanism_but_not_primary",
    "mechanism_review_with_treatment_discussion",
)

R1C_ALL_LABELS: frozenset[str] = frozenset(
    PROCEDURE_OUTCOME_LABELS + MECHANISM_TARGET_LABELS
)

# =====================================================================
# Closed-set per-label ranking multipliers (shadow simulation only —
# DO NOT touch clinician-visible ranking).
#
# The simulator multiplies the article's R1A-based score by EACH
# applicable R1C multiplier (default 1.0 when label absent). All
# multipliers ≤ 1.0 so the rule can only deprioritize.
# =====================================================================
R1C_LABEL_MULTIPLIERS: dict[str, float] = {
    # Procedure-complication outcome anchoring.
    "procedure_complication_outcome_exact": 1.0,  # no penalty — preserve direct
    "procedure_complication_outcome_missing": 0.5,
    "procedure_complication_wrong_outcome": 0.3,
    "procedure_complication_general_not_outcome_specific": 0.5,
    # Mechanism target preservation.
    "mechanism_target_exact": 1.0,
    "mechanism_target_missing": 0.5,
    "disease_only_indirect_for_mechanism": 0.3,
    "treatment_mentions_mechanism_but_not_primary": 0.4,
    "mechanism_review_with_treatment_discussion": 1.0,  # flag-only (mirrors R1A mixed-signal)
}


# =====================================================================
# Closed-set per-benchmark-case anchor configuration.
#
# Verbatim from the R1C brief. NOT clinician-facing; NOT used for
# any clinician path. Used ONLY by the shadow simulator and the
# admin replay endpoint.
# =====================================================================
CASE_ANCHOR_CONFIG: dict[str, dict[str, list[str]]] = {
    "r1a-case-A-cholestatic-pruritus": {
        "disease_anchor_terms": [
            "cholestasis", "cholestatic", "cholestatic liver disease",
            "primary biliary cholangitis", "primary sclerosing cholangitis",
        ],
        "mechanism_target_terms": [
            "pruritus", "itch", "pruritogen", "itch pathway",
        ],
        # No outcome anchor — mechanism shape.
        "outcome_anchor_terms": [],
        # Procedure anchor empty too — not a procedure shape.
        "procedure_anchor_terms": [],
    },
    "r1a-case-B-pcos-insulin-resistance": {
        "disease_anchor_terms": [
            "polycystic ovary syndrome", "PCOS",
        ],
        "mechanism_target_terms": [
            "insulin resistance", "insulin signaling", "insulin sensitivity",
            "hyperinsulinemia",
        ],
        "outcome_anchor_terms": [],
        "procedure_anchor_terms": [],
    },
    "r1a-case-C-amyloid-diastolic-dysfunction": {
        "disease_anchor_terms": [
            "cardiac amyloidosis", "amyloid cardiomyopathy",
            "transthyretin amyloid", "ATTR-CM", "AL amyloidosis",
        ],
        "mechanism_target_terms": [
            "diastolic dysfunction", "impaired relaxation",
            "myocardial stiffness", "restrictive physiology",
        ],
        "outcome_anchor_terms": [],
        "procedure_anchor_terms": [],
    },
    "r1a-case-D-cvc-pneumothorax": {
        "disease_anchor_terms": [],
        "mechanism_target_terms": [],
        "outcome_anchor_terms": [
            "pneumothorax", "iatrogenic pneumothorax",
            "catheter-related pneumothorax",
            "procedure-related pneumothorax",
        ],
        "procedure_anchor_terms": [
            "central venous catheter", "central venous catheterization",
            "CVC", "central venous access", "subclavian catheterization",
            "internal jugular catheterization", "central line",
        ],
    },
}

# Closed-set wrong-outcome candidates for procedure-complication shape.
# These are common procedure complications that are NOT the asked
# outcome (Case D's asked outcome is pneumothorax — so DVT, BSI, etc.
# are "wrong outcomes" relative to a pneumothorax-specific question).
PROCEDURE_WRONG_OUTCOME_TERMS_FOR_CASE: dict[str, list[str]] = {
    "r1a-case-D-cvc-pneumothorax": [
        "deep vein thrombosis", "DVT",
        "venous thromboembolism", "VTE",
        "bloodstream infection", "central line-associated bloodstream",
        "CLABSI", "catheter-related infection",
        "air embolism",
        "arterial puncture", "arterial cannulation",
        "hemothorax", "chylothorax",
        "cardiac tamponade", "perforation",
        "bleeding", "hematoma",
    ],
}

# Closed-set "treatment" terms used to detect when a mechanism
# article ALSO discusses treatment (mixed-signal / "mechanism review
# with treatment discussion" label).
_TREATMENT_TERMS: tuple[str, ...] = (
    "treatment", "therapy", "therapeutic", "trial", "randomized",
    "rct", "pharmacotherapy", "drug", "inhibitor", "agonist",
    "antagonist", "intervention", "management",
)


# =====================================================================
# Detection helpers.
# =====================================================================
def _normalize_text(article: dict) -> str:
    parts: list[str] = []
    for k in ("title", "abstract", "abstractText"):
        v = article.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
    return " ".join(parts).lower()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    """Closed-set whole-word / phrase containment check (case-insensitive).

    For phrases (multi-word terms) we do a plain substring match. For
    single-word terms we do a word-boundary regex match to avoid
    false positives from substrings (e.g., ``itch`` must not match
    ``twitch``).
    """
    for raw in terms:
        t = (raw or "").strip().lower()
        if not t:
            continue
        if " " in t or "-" in t:
            if t in text:
                return True
        else:
            if re.search(rf"\b{re.escape(t)}\b", text):
                return True
    return False


def detect_procedure_outcome_labels(
    *,
    case_id: str,
    article: dict,
) -> list[str]:
    """Closed-set procedure-complication outcome anchoring detection.

    Returns at most one of ``PROCEDURE_OUTCOME_LABELS``. Returns
    an empty list when the case doesn't carry procedure +
    outcome anchor configuration.
    """
    cfg = CASE_ANCHOR_CONFIG.get(case_id)
    if not cfg:
        return []
    proc_terms = cfg.get("procedure_anchor_terms") or []
    out_terms = cfg.get("outcome_anchor_terms") or []
    if not proc_terms or not out_terms:
        return []
    wrong_terms = PROCEDURE_WRONG_OUTCOME_TERMS_FOR_CASE.get(case_id, [])
    text = _normalize_text(article)
    has_proc = _contains_any(text, proc_terms)
    has_correct_outcome = _contains_any(text, out_terms)
    has_wrong_outcome = (
        bool(wrong_terms) and _contains_any(text, wrong_terms)
    )
    if has_proc and has_correct_outcome:
        return ["procedure_complication_outcome_exact"]
    if has_proc and has_wrong_outcome and not has_correct_outcome:
        return ["procedure_complication_wrong_outcome"]
    if has_proc and not has_correct_outcome:
        # Closed-set: if it has the procedure but no correct outcome
        # AND it reads like a general overview (no specific
        # complication outcome at all), mark it as
        # ``general_not_outcome_specific``. Otherwise mark it
        # ``outcome_missing``.
        is_general = (
            "overview" in text
            or "review" in text
            or "general" in text
            or "approach" in text
        )
        if is_general and not has_wrong_outcome:
            return ["procedure_complication_general_not_outcome_specific"]
        return ["procedure_complication_outcome_missing"]
    return []


def detect_mechanism_target_labels(
    *,
    case_id: str,
    article: dict,
) -> list[str]:
    """Closed-set mechanism-target preservation detection.

    Returns at most two labels: the primary target label (exact /
    missing / disease-only / treatment-not-primary) AND optionally
    the ``mechanism_review_with_treatment_discussion`` flag.
    """
    cfg = CASE_ANCHOR_CONFIG.get(case_id)
    if not cfg:
        return []
    disease_terms = cfg.get("disease_anchor_terms") or []
    mech_terms = cfg.get("mechanism_target_terms") or []
    if not disease_terms or not mech_terms:
        return []
    text = _normalize_text(article)
    has_disease = _contains_any(text, disease_terms)
    has_mech_target = _contains_any(text, mech_terms)
    has_treatment = _contains_any(text, _TREATMENT_TERMS)
    is_review = "review" in text

    labels: list[str] = []
    if has_disease and has_mech_target:
        labels.append("mechanism_target_exact")
        # Mixed-signal flag — mechanism review that ALSO discusses
        # treatment. Closed-set flag, NO penalty (mirrors R1A
        # mixed-signal posture).
        if is_review and has_treatment:
            labels.append("mechanism_review_with_treatment_discussion")
        return labels
    if has_disease and not has_mech_target:
        # Two further closed-set sub-cases:
        if has_treatment:
            # Treatment paper that mentions the disease but not the
            # mechanism target → ``treatment_not_primary``.
            return ["treatment_mentions_mechanism_but_not_primary"]
        # Disease-only broad overview without mechanism target.
        return ["disease_only_indirect_for_mechanism"]
    if not has_disease and has_mech_target:
        # Mechanism term present but not the disease — counts as
        # ``mechanism_target_missing`` because the article isn't
        # in the right disease lane.
        return ["mechanism_target_missing"]
    # No disease, no mechanism target → not labelable by R1C.
    return []


def detect_r1c_labels(
    *,
    case_id: str,
    article: dict,
    shape: str,
) -> list[str]:
    """Closed-set dispatch — returns the union of applicable R1C
    labels for this article in this case's context. Returns the
    empty list when the shape doesn't carry R1C-eligible anchors.
    """
    if shape in ("procedure_complication_risk", "procedure_complication"):
        return detect_procedure_outcome_labels(
            case_id=case_id, article=article,
        )
    if shape == "pathophysiology_mechanism":
        return detect_mechanism_target_labels(
            case_id=case_id, article=article,
        )
    return []


def compute_r1c_multiplier(labels: Iterable[str]) -> float:
    """Deterministic — product of all applicable R1C multipliers.

    Pure: depends only on the closed-set ``labels`` argument and
    the closed-set ``R1C_LABEL_MULTIPLIERS`` table.
    """
    m = 1.0
    for lab in labels or []:
        m *= float(R1C_LABEL_MULTIPLIERS.get(lab, 1.0))
    return m


__all__ = [
    "PROCEDURE_OUTCOME_LABELS",
    "MECHANISM_TARGET_LABELS",
    "R1C_ALL_LABELS",
    "R1C_LABEL_MULTIPLIERS",
    "CASE_ANCHOR_CONFIG",
    "PROCEDURE_WRONG_OUTCOME_TERMS_FOR_CASE",
    "detect_procedure_outcome_labels",
    "detect_mechanism_target_labels",
    "detect_r1c_labels",
    "compute_r1c_multiplier",
]
