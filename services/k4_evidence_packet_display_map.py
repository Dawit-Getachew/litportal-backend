"""Phase K4 — closed-set R-series-label → clinician-readable display map.

Translates internal R1A/R1B/R1C/R1C.1 directness labels into plain
English strings rendered on the clinician evidence-card UI as small
pills with hover/focus tooltips. The mapping is EXHAUSTIVE over the
union of R1A + R1C label catalogs — any new R-series label added
without updating this map will raise ``KeyError`` at packet-build
time (fail loud — never ship raw closed-set strings to clinicians).

NO LLM. NO retrieval. NO ranking impact. Display-only metadata —
labels are computed and attached to packets, but they DO NOT mutate
any clinician ranking, default-selection, summary, or citation-
verification behavior.
"""
from __future__ import annotations

from typing import Iterable

from services.r1a_directness_labels import R1A_ALL_LABELS
from services.r1c_anchor_labels import R1C_ALL_LABELS

# ---------------------------------------------------------------------
# Closed-set mapping. KEYS must match the union of
# ``R1A_ALL_LABELS`` and ``R1C_ALL_LABELS`` exactly.
# Each value carries:
#   * ``human``  — short clinician-readable pill text
#   * ``tooltip`` — 1-sentence hover/focus explanation
#   * ``polarity`` — closed-set {"positive", "negative", "flag"} for
#     pill styling (positive = on-target, negative = indirect/off,
#     flag = mixed-signal / informational)
# ---------------------------------------------------------------------
R_SERIES_DISPLAY_MAP: dict[str, dict[str, str]] = {
    # === R1A pathophysiology_mechanism POSITIVE ===
    "pathophysiology_mechanism_direct": {
        "human": "Mechanism-focused review",
        "tooltip": (
            "This article focuses on the disease mechanism or "
            "pathophysiology directly."
        ),
        "polarity": "positive",
    },
    "mechanism_review": {
        "human": "Mechanism review",
        "tooltip": (
            "Review article whose primary focus is the mechanism / "
            "pathophysiology."
        ),
        "polarity": "positive",
    },
    "clinical_mechanism_review": {
        "human": "Clinical mechanism review",
        "tooltip": (
            "Review of mechanisms in human (clinical) populations rather "
            "than only basic-science models."
        ),
        "polarity": "positive",
    },
    "human_translational_mechanism": {
        "human": "Human / translational mechanism",
        "tooltip": (
            "Mechanism evidence from human samples or translational data, "
            "not solely from animal or in-vitro models."
        ),
        "polarity": "positive",
    },
    "molecular_pathogenesis": {
        "human": "Molecular pathogenesis",
        "tooltip": (
            "Article discusses molecular-level pathogenesis "
            "(receptors, signaling, cytokines, etc.)."
        ),
        "polarity": "positive",
    },
    "disease_pathogenesis_review": {
        "human": "Disease pathogenesis review",
        "tooltip": (
            "Review specifically of the disease's pathogenesis."
        ),
        "polarity": "positive",
    },
    "mechanism_with_named_process": {
        "human": "Mechanism (named process)",
        "tooltip": (
            "Article names a specific biological process or pathway "
            "(e.g., bile acids, autotaxin, fibrosis, remodeling)."
        ),
        "polarity": "positive",
    },
    # === R1A pathophysiology_mechanism NEGATIVE ===
    "treatment_modality_indirect_for_mechanism": {
        "human": "Treatment-focused — indirect for mechanism",
        "tooltip": (
            "Article primarily discusses treatment/therapy — useful but "
            "less direct for a mechanism question."
        ),
        "polarity": "negative",
    },
    "diagnostic_modality_indirect_for_mechanism": {
        "human": "Diagnostic-focused — indirect for mechanism",
        "tooltip": (
            "Article primarily discusses diagnostic methods rather than "
            "the underlying mechanism."
        ),
        "polarity": "negative",
    },
    "management_guideline_indirect_for_mechanism": {
        "human": "Management guideline — indirect for mechanism",
        "tooltip": (
            "Article is a management guideline or recommendation — "
            "less direct for a mechanism question."
        ),
        "polarity": "negative",
    },
    "broad_disease_overview_indirect_for_mechanism": {
        "human": "Broad disease overview",
        "tooltip": (
            "Broad review/overview of the disease rather than a focused "
            "mechanism account."
        ),
        "polarity": "negative",
    },
    "animal_basic_science_lower_priority": {
        "human": "Animal / basic science",
        "tooltip": (
            "Animal model or basic-science study — lower priority for a "
            "clinical mechanism question."
        ),
        "polarity": "negative",
    },
    "missing_mechanism_focus": {
        "human": "Missing mechanism focus",
        "tooltip": (
            "Article lacks explicit mechanism or pathophysiology focus."
        ),
        "polarity": "negative",
    },
    # === R1A procedure_complication_risk POSITIVE ===
    "procedure_complication_direct": {
        "human": "Procedure complication — direct",
        "tooltip": (
            "Article directly addresses procedure-related complications."
        ),
        "polarity": "positive",
    },
    "procedure_complication_incidence": {
        "human": "Procedure complication incidence",
        "tooltip": (
            "Article reports the incidence rate of the procedural "
            "complication."
        ),
        "polarity": "positive",
    },
    "procedure_complication_risk_factor": {
        "human": "Risk factor",
        "tooltip": (
            "Article identifies risk factors or predictors of the "
            "procedural complication."
        ),
        "polarity": "positive",
    },
    "procedure_complication_cohort": {
        "human": "Cohort study",
        "tooltip": (
            "Cohort study (retrospective or prospective) of the "
            "procedure complication."
        ),
        "polarity": "positive",
    },
    "procedure_complication_registry": {
        "human": "Registry data",
        "tooltip": (
            "Registry or administrative-database analysis of the "
            "procedure complication."
        ),
        "polarity": "positive",
    },
    "procedure_complication_systematic_review": {
        "human": "Systematic review",
        "tooltip": (
            "Systematic review or meta-analysis of the procedure "
            "complication."
        ),
        "polarity": "positive",
    },
    "mechanical_complication_rate": {
        "human": "Mechanical complication rate",
        "tooltip": (
            "Article reports the mechanical-complication rate of the "
            "procedure."
        ),
        "polarity": "positive",
    },
    # === R1A procedure_complication_risk NEGATIVE ===
    "diagnostic_followup_indirect_for_complication_risk": {
        "human": "Diagnostic follow-up — indirect",
        "tooltip": (
            "Article focuses on post-procedure imaging/follow-up "
            "rather than the complication itself."
        ),
        "polarity": "negative",
    },
    "routine_imaging_utility_indirect": {
        "human": "Routine imaging utility",
        "tooltip": (
            "Article discusses the utility of routine imaging — "
            "not the complication rate or risk factors directly."
        ),
        "polarity": "negative",
    },
    "training_or_detection_method_indirect": {
        "human": "Training / detection method",
        "tooltip": (
            "Article focuses on training or detection methodology "
            "rather than the complication itself."
        ),
        "polarity": "negative",
    },
    "procedure_general_overview_indirect": {
        "human": "Procedure general overview",
        "tooltip": (
            "General overview of the procedure without a complication-"
            "specific focus."
        ),
        "polarity": "negative",
    },
    "complication_missing_outcome": {
        "human": "Complication outcome missing",
        "tooltip": (
            "Article mentions the procedure but does not address the "
            "specific complication outcome."
        ),
        "polarity": "negative",
    },
    "procedure_missing_exposure": {
        "human": "Procedure exposure missing",
        "tooltip": (
            "Article mentions the complication but not the specific "
            "procedure exposure."
        ),
        "polarity": "negative",
    },
    # === R1C procedure_complication outcome anchors ===
    "procedure_complication_outcome_exact": {
        "human": "Outcome-specific",
        "tooltip": (
            "Article addresses the exact procedural complication "
            "outcome you asked about."
        ),
        "polarity": "positive",
    },
    "procedure_complication_outcome_missing": {
        "human": "Outcome anchor missing",
        "tooltip": (
            "Article mentions the procedure but does not name the "
            "specific complication outcome you asked about."
        ),
        "polarity": "negative",
    },
    "procedure_complication_wrong_outcome": {
        "human": "Wrong outcome",
        "tooltip": (
            "Article discusses a different procedural complication "
            "than the one you asked about."
        ),
        "polarity": "negative",
    },
    "procedure_complication_general_not_outcome_specific": {
        "human": "General complication overview",
        "tooltip": (
            "General overview of procedural complications without "
            "the specific outcome focus."
        ),
        "polarity": "negative",
    },
    # === R1C mechanism target anchors ===
    "mechanism_target_exact": {
        "human": "Mechanism target — exact",
        "tooltip": (
            "Article addresses both the disease AND the specific "
            "mechanism target you asked about."
        ),
        "polarity": "positive",
    },
    "mechanism_target_missing": {
        "human": "Mechanism target — missing",
        "tooltip": (
            "Article mentions the mechanism vocabulary but not in the "
            "specific disease lane you asked about."
        ),
        "polarity": "negative",
    },
    "disease_only_indirect_for_mechanism": {
        "human": "Disease-only — indirect for mechanism",
        "tooltip": (
            "Article discusses the disease broadly without the specific "
            "mechanism target you asked about."
        ),
        "polarity": "negative",
    },
    "treatment_mentions_mechanism_but_not_primary": {
        "human": "Treatment paper — mechanism not primary",
        "tooltip": (
            "Article is primarily about treatment; it mentions the "
            "mechanism only secondarily."
        ),
        "polarity": "negative",
    },
    "mechanism_review_with_treatment_discussion": {
        "human": "Mechanism review (also treatment)",
        "tooltip": (
            "Mechanism review that also discusses treatment — useful "
            "mixed-signal article."
        ),
        "polarity": "flag",
    },
}


# ---------------------------------------------------------------------
# Exhaustiveness contract — module-load assertion.
# Any new R-series label added without updating this map fails here.
# ---------------------------------------------------------------------
ALL_R_SERIES_LABELS: frozenset[str] = R1A_ALL_LABELS | R1C_ALL_LABELS

_MAP_KEYS = frozenset(R_SERIES_DISPLAY_MAP.keys())
_MISSING = ALL_R_SERIES_LABELS - _MAP_KEYS
_EXTRA = _MAP_KEYS - ALL_R_SERIES_LABELS
assert not _MISSING, (
    f"k4 display-map missing R-series labels: {sorted(_MISSING)}"
)
assert not _EXTRA, (
    f"k4 display-map contains unknown labels: {sorted(_EXTRA)}"
)


# ---------------------------------------------------------------------
# Public mapping helpers.
# ---------------------------------------------------------------------
def map_label_to_display(label: str) -> dict[str, str]:
    """Return the closed-set display entry for ``label``.

    Raises ``KeyError`` if ``label`` is not a known R-series label.
    Fail-loud by design — the clinician UI must NEVER render a raw
    internal label.
    """
    if label not in R_SERIES_DISPLAY_MAP:
        raise KeyError(
            f"r_series label has no clinician display mapping: {label!r}"
        )
    return dict(R_SERIES_DISPLAY_MAP[label])


def map_labels_to_displays(
    labels: Iterable[str],
) -> list[dict[str, str]]:
    """Map a sequence of labels; preserves order; dedupes."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for lab in labels or []:
        if not lab or lab in seen:
            continue
        seen.add(lab)
        out.append({"label": lab, **map_label_to_display(lab)})
    return out


__all__ = [
    "R_SERIES_DISPLAY_MAP",
    "ALL_R_SERIES_LABELS",
    "map_label_to_display",
    "map_labels_to_displays",
]
