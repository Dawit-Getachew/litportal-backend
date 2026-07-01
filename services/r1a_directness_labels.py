"""Phase R1A — Part 3 + 4 — shadow directness label catalog.

DETERMINISTIC. NO LLM. NO RETRIEVAL.

Closed-set label constants + keyword/heuristic detectors for two
clinician-question shapes:
  * ``pathophysiology_mechanism``
  * ``procedure_complication_risk``

The labels are written to the ``r1a_shadow_decisions`` sidecar
ONLY. They are NEVER attached to the active ``articles`` or
``search_session_articles`` collections, and they NEVER appear on
the clinician UI in this phase.

Detection inputs (per article): closed-set fields ``title``,
``abstract`` (when available — abstract text is NOT persisted to
sidecar; only ITS LABEL OUTPUTS are), ``publication_types_json``,
``journal``, ``publication_year``. All other fields are ignored.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# ---------------------------------------------------------------------
# pathophysiology_mechanism — closed-set positive labels.
# ---------------------------------------------------------------------
PATHO_POSITIVE_LABELS: tuple[str, ...] = (
    "pathophysiology_mechanism_direct",
    "mechanism_review",
    "clinical_mechanism_review",
    "human_translational_mechanism",
    "molecular_pathogenesis",
    "disease_pathogenesis_review",
    "mechanism_with_named_process",
)

# ---------------------------------------------------------------------
# pathophysiology_mechanism — closed-set negative labels.
# ---------------------------------------------------------------------
PATHO_NEGATIVE_LABELS: tuple[str, ...] = (
    "treatment_modality_indirect_for_mechanism",
    "diagnostic_modality_indirect_for_mechanism",
    "management_guideline_indirect_for_mechanism",
    "broad_disease_overview_indirect_for_mechanism",
    "animal_basic_science_lower_priority",
    "missing_mechanism_focus",
)

# ---------------------------------------------------------------------
# procedure_complication_risk — closed-set positive labels.
# ---------------------------------------------------------------------
PROC_POSITIVE_LABELS: tuple[str, ...] = (
    "procedure_complication_direct",
    "procedure_complication_incidence",
    "procedure_complication_risk_factor",
    "procedure_complication_cohort",
    "procedure_complication_registry",
    "procedure_complication_systematic_review",
    "mechanical_complication_rate",
)

# ---------------------------------------------------------------------
# procedure_complication_risk — closed-set negative labels.
# ---------------------------------------------------------------------
PROC_NEGATIVE_LABELS: tuple[str, ...] = (
    "diagnostic_followup_indirect_for_complication_risk",
    "routine_imaging_utility_indirect",
    "training_or_detection_method_indirect",
    "procedure_general_overview_indirect",
    "complication_missing_outcome",
    "procedure_missing_exposure",
)

# Closed-set union of all labels — used by leak-scan and test assertions.
R1A_ALL_LABELS: frozenset[str] = frozenset(
    PATHO_POSITIVE_LABELS
    + PATHO_NEGATIVE_LABELS
    + PROC_POSITIVE_LABELS
    + PROC_NEGATIVE_LABELS
)


# ---------------------------------------------------------------------
# Closed-set keyword groups (case-insensitive).
# ---------------------------------------------------------------------
_PATHO_MECH_KW: tuple[str, ...] = (
    "mechanism", "mechanisms", "pathophysiology", "physiopathology",
    "pathogenesis", "molecular pathogenesis", "signaling", "signalling",
    "pathway", "pathways", "inflammatory", "metabolic", "endocrine",
    "fibrosis", "stiffness", "relaxation", "pruritogen", "pruritogens",
    "itch pathway", "itch pathways",
)

_PATHO_MOLECULAR_KW: tuple[str, ...] = (
    "molecular", "receptor", "receptors", "ligand", "ligands",
    "kinase", "phosphorylation", "transcription factor", "cytokine",
    "inflammation", "oxidative stress", "extracellular matrix",
)

_PATHO_REVIEW_KW: tuple[str, ...] = (
    "review", "systematic review", "narrative review",
    "scoping review", "umbrella review",
)

_PATHO_TREATMENT_KW: tuple[str, ...] = (
    "treatment", "therapy", "therapeutic", "drug", "pharmacological",
    "pharmacotherapy", "intervention", "interventions",
    "randomized controlled trial", "clinical trial",
    "double blind", "placebo controlled",
)

_PATHO_DIAGNOSTIC_KW: tuple[str, ...] = (
    "diagnostic", "diagnosis", "imaging", "ultrasound",
    "echocardiography", "magnetic resonance", "ct scan", "ct angiography",
    "biomarker test", "screening",
)

_PATHO_GUIDELINE_KW: tuple[str, ...] = (
    "guideline", "guidelines", "consensus statement", "protocol",
    "recommendation", "recommendations", "management",
)

_PATHO_BROAD_OVERVIEW_KW: tuple[str, ...] = (
    "review of", "overview of", "update on", "current concepts",
    "what we know", "approach to",
)

_PATHO_ANIMAL_KW: tuple[str, ...] = (
    "mouse model", "murine model", "rat model", "rodent model",
    "in vitro", "cell culture", "knockout mouse",
)

# Procedure_complication_risk — closed-set keyword groups.
_PROC_PROCEDURE_KW: tuple[str, ...] = (
    "central venous catheter", "central venous catheterization",
    "central line", "central line placement", "subclavian catheter",
    "internal jugular catheter", "femoral catheter",
    "thoracentesis", "paracentesis", "lumbar puncture",
    "thoracic central venous", "cvc",
    "endotracheal intubation", "intubation",
    "appendectomy", "cholecystectomy", "laparoscopic",
    "endoscopy", "colonoscopy", "biopsy",
)

_PROC_COMPLICATION_KW: tuple[str, ...] = (
    "pneumothorax", "iatrogenic pneumothorax",
    "catheter-related pneumothorax", "procedure-related pneumothorax",
    "complication", "complications", "complication rate",
    "complication rates", "adverse event", "adverse events",
    "incidence", "incidence rate", "hemothorax", "bleeding",
    "infection", "perforation", "rupture", "leak", "leakage",
    "mechanical complication", "mechanical complications",
)

_PROC_RISK_KW: tuple[str, ...] = (
    "risk", "risk factor", "risk factors", "predictors", "predictor",
    "associated with", "predisposing factor", "predisposing factors",
)

_PROC_COHORT_KW: tuple[str, ...] = (
    "cohort", "cohort study", "retrospective cohort", "prospective cohort",
    "case series", "case-control", "cross-sectional",
)

_PROC_REGISTRY_KW: tuple[str, ...] = (
    "registry", "national registry", "database analysis",
    "administrative database", "claims data",
)

_PROC_REVIEW_KW: tuple[str, ...] = (
    "systematic review", "meta-analysis", "meta analysis",
    "umbrella review",
)

_PROC_DIAGNOSTIC_INDIRECT_KW: tuple[str, ...] = (
    "post-procedure imaging utility", "line confirmation",
    "lung ultrasound", "ultrasound training", "pneumothorax detection",
    "imaging follow-up", "routine chest radiograph",
    "post-procedure chest x-ray", "post-line chest film",
    "detection method", "detection of complications",
)

_PROC_TRAINING_KW: tuple[str, ...] = (
    "training", "simulation", "education program",
    "trainee", "fellow training", "resident training",
)


def _has_any(text: str, terms: Iterable[str]) -> bool:
    if not text:
        return False
    tl = text.lower()
    return any(t.lower() in tl for t in terms)


def _publication_types_lower(pubs: Any) -> list[str]:
    if not isinstance(pubs, list):
        return []
    return [str(p).lower() for p in pubs if isinstance(p, str)]


# ---------------------------------------------------------------------
# pathophysiology_mechanism — label detector.
# ---------------------------------------------------------------------
def detect_patho_mechanism_labels(article: dict) -> list[str]:
    """Deterministic closed-set label assignment for one candidate
    article on a ``pathophysiology_mechanism`` shape. Multiple labels
    may apply. Pure function.

    Inputs read (closed set): ``title``, ``abstract``,
    ``publication_types_json``. All other fields ignored.
    """
    if not isinstance(article, dict):
        return []
    title = str(article.get("title") or "")
    abstract = str(article.get("abstract") or "")
    body = f"{title}\n{abstract}"
    pubs = _publication_types_lower(article.get("publication_types_json"))

    labels: list[str] = []

    # ---- Positive labels ----
    if _has_any(body, _PATHO_MECH_KW):
        labels.append("pathophysiology_mechanism_direct")
    if _has_any(body, ("mechanism", "mechanisms")) and any(
        "review" in p for p in pubs
    ):
        labels.append("mechanism_review")
    if _has_any(body, ("clinical mechanism", "human mechanism")) and any(
        "review" in p for p in pubs
    ):
        labels.append("clinical_mechanism_review")
    if _has_any(body, ("translational", "patient-derived", "human samples")):
        labels.append("human_translational_mechanism")
    if _has_any(body, _PATHO_MOLECULAR_KW):
        labels.append("molecular_pathogenesis")
    if _has_any(body, ("pathogenesis",)) and any("review" in p for p in pubs):
        labels.append("disease_pathogenesis_review")
    if _has_any(body, (
        "lysophosphatidic acid", "autotaxin", "bile acids",
        "sulfated progesterone metabolites", "mrgprx4", "tgr5",
        "c-fibres", "myocardial stiffness", "impaired relaxation",
        "extracellular volume", "fibrosis", "remodeling",
    )):
        labels.append("mechanism_with_named_process")

    # ---- Negative labels ----
    if _has_any(body, _PATHO_TREATMENT_KW):
        labels.append("treatment_modality_indirect_for_mechanism")
    if _has_any(body, _PATHO_DIAGNOSTIC_KW):
        labels.append("diagnostic_modality_indirect_for_mechanism")
    if _has_any(body, _PATHO_GUIDELINE_KW) or any(
        "guideline" in p for p in pubs
    ):
        labels.append("management_guideline_indirect_for_mechanism")
    if _has_any(body, _PATHO_BROAD_OVERVIEW_KW) and not _has_any(
        body, _PATHO_MECH_KW
    ):
        labels.append("broad_disease_overview_indirect_for_mechanism")
    if _has_any(body, _PATHO_ANIMAL_KW):
        labels.append("animal_basic_science_lower_priority")
    if not _has_any(body, _PATHO_MECH_KW + _PATHO_MOLECULAR_KW):
        labels.append("missing_mechanism_focus")

    # Deduplicate while preserving order.
    return list(dict.fromkeys(labels))


# ---------------------------------------------------------------------
# procedure_complication_risk — label detector.
# ---------------------------------------------------------------------
def detect_procedure_complication_labels(article: dict) -> list[str]:
    """Deterministic closed-set label assignment for one candidate
    article on a ``procedure_complication_risk`` shape.

    HIGH-PRECISION RULE: any of the ``procedure_complication_*``
    positive labels (except ``procedure_complication_systematic_review``,
    which carries its own filter) require **both** a procedure
    anchor AND a complication anchor in the title or abstract.
    """
    if not isinstance(article, dict):
        return []
    title = str(article.get("title") or "")
    abstract = str(article.get("abstract") or "")
    body = f"{title}\n{abstract}"
    pubs = _publication_types_lower(article.get("publication_types_json"))

    has_proc = _has_any(body, _PROC_PROCEDURE_KW)
    has_comp = _has_any(body, _PROC_COMPLICATION_KW)
    high_precision = has_proc and has_comp

    labels: list[str] = []

    # ---- Positive labels (gated by high_precision). ----
    if high_precision:
        labels.append("procedure_complication_direct")
        if _has_any(body, ("incidence", "incidence rate")):
            labels.append("procedure_complication_incidence")
        if _has_any(body, _PROC_RISK_KW):
            labels.append("procedure_complication_risk_factor")
        if _has_any(body, _PROC_COHORT_KW):
            labels.append("procedure_complication_cohort")
        if _has_any(body, _PROC_REGISTRY_KW):
            labels.append("procedure_complication_registry")
        if _has_any(body, ("mechanical complication", "mechanical complications")):
            labels.append("mechanical_complication_rate")
    # Systematic review can apply even without both anchors but with
    # complication anchor present.
    if has_comp and (
        _has_any(body, _PROC_REVIEW_KW)
        or any("systematic review" in p or "meta" in p for p in pubs)
    ):
        labels.append("procedure_complication_systematic_review")

    # ---- Negative labels ----
    if _has_any(body, _PROC_DIAGNOSTIC_INDIRECT_KW) and not (
        has_proc and has_comp
        and _has_any(body, ("incidence", "risk factor", "risk factors"))
    ):
        labels.append("diagnostic_followup_indirect_for_complication_risk")
    if _has_any(body, ("ultrasound", "chest radiograph", "chest x-ray")) and (
        _has_any(body, ("utility", "performance", "accuracy", "sensitivity"))
    ):
        labels.append("routine_imaging_utility_indirect")
    if _has_any(body, _PROC_TRAINING_KW):
        labels.append("training_or_detection_method_indirect")
    if has_proc and not has_comp and _has_any(body, ("review", "overview")):
        labels.append("procedure_general_overview_indirect")
    if has_proc and not has_comp:
        labels.append("complication_missing_outcome")
    if has_comp and not has_proc:
        labels.append("procedure_missing_exposure")

    return list(dict.fromkeys(labels))


# ---------------------------------------------------------------------
# Closed-set weight table for shadow score.
# ---------------------------------------------------------------------
SHADOW_LABEL_WEIGHTS: dict[str, float] = {
    # pathophysiology_mechanism positive
    "pathophysiology_mechanism_direct": +1.0,
    "mechanism_review": +1.0,
    "clinical_mechanism_review": +1.0,
    "human_translational_mechanism": +0.8,
    "molecular_pathogenesis": +0.7,
    "disease_pathogenesis_review": +0.9,
    "mechanism_with_named_process": +1.2,
    # pathophysiology_mechanism negative
    "treatment_modality_indirect_for_mechanism": -0.6,
    "diagnostic_modality_indirect_for_mechanism": -0.4,
    "management_guideline_indirect_for_mechanism": -0.5,
    "broad_disease_overview_indirect_for_mechanism": -0.5,
    "animal_basic_science_lower_priority": -0.3,
    "missing_mechanism_focus": -0.8,
    # procedure_complication_risk positive
    "procedure_complication_direct": +1.0,
    "procedure_complication_incidence": +1.1,
    "procedure_complication_risk_factor": +1.0,
    "procedure_complication_cohort": +0.9,
    "procedure_complication_registry": +1.0,
    "procedure_complication_systematic_review": +1.2,
    "mechanical_complication_rate": +1.0,
    # procedure_complication_risk negative
    "diagnostic_followup_indirect_for_complication_risk": -0.6,
    "routine_imaging_utility_indirect": -0.5,
    "training_or_detection_method_indirect": -0.6,
    "procedure_general_overview_indirect": -0.4,
    "complication_missing_outcome": -0.7,
    "procedure_missing_exposure": -0.7,
}

# Sanity contract — every label has a weight.
assert set(SHADOW_LABEL_WEIGHTS.keys()) == set(R1A_ALL_LABELS), (
    "SHADOW_LABEL_WEIGHTS keys must exactly cover R1A_ALL_LABELS"
)


def compute_shadow_score(
    *,
    base_score: float,
    labels: Iterable[str],
) -> float:
    """Deterministic, transparent shadow score.

    score = base_score + Σ(positive_weight) − Σ(|negative_weight|)
    """
    total = float(base_score or 0.0)
    for lab in labels:
        w = SHADOW_LABEL_WEIGHTS.get(lab)
        if w is None:
            continue
        total += w
    return total


# ---------------------------------------------------------------------
# Phase R1C.1 — shape-conditional recency cap (SHADOW-only).
#
# Rationale: the legacy ``base_score = max(0, year - 1990)`` recency
# component dominated the R1A label deltas (label weights 0.3–1.2 vs
# year-derived 30+). For ``pathophysiology_mechanism`` shape the
# operator confirmed that clinical mechanism directness must dominate
# recency so older highly-relevant mechanism reviews (e.g., Beuers
# 2023) can outrank newer treatment/diagnostic/broad-overview noise.
#
# All other shapes (``procedure_complication_risk`` etc.) keep the
# linear year-based recency contribution UNCHANGED. R1A
# ``SHADOW_LABEL_WEIGHTS`` is byte-identical.
# ---------------------------------------------------------------------
MECHANISM_RECENCY_HARD_CAP: float = 2.0


def compute_recency_base_score(*, year, shape: str = "") -> float:
    """Shape-conditional recency base score.

    * ``pathophysiology_mechanism`` shape: hard-capped at
      ``MECHANISM_RECENCY_HARD_CAP`` (2.0) so directness labels can
      dominate.
    * Any other shape (or no shape supplied): preserves the legacy
      ``max(0.0, year - 1990.0)`` linear signal.

    Pure function; no I/O.
    """
    try:
        y = float(year) if year is not None else 1990.0
    except (TypeError, ValueError):
        y = 1990.0
    raw = max(0.0, y - 1990.0)
    sl = (shape or "").strip().lower()
    if sl == "pathophysiology_mechanism":
        return min(raw, MECHANISM_RECENCY_HARD_CAP)
    return raw


def detect_labels_for_shape(
    shape: str,
    article: dict,
) -> list[str]:
    """Dispatch to the correct detector by shape. Returns empty list
    for unrecognized shapes (defensive — no exceptions)."""
    sl = (shape or "").strip().lower()
    if sl == "pathophysiology_mechanism":
        return detect_patho_mechanism_labels(article)
    if sl in ("procedure_complication_risk", "procedure_complication"):
        return detect_procedure_complication_labels(article)
    return []


__all__ = [
    "PATHO_POSITIVE_LABELS",
    "PATHO_NEGATIVE_LABELS",
    "PROC_POSITIVE_LABELS",
    "PROC_NEGATIVE_LABELS",
    "R1A_ALL_LABELS",
    "SHADOW_LABEL_WEIGHTS",
    "MECHANISM_RECENCY_HARD_CAP",
    "detect_patho_mechanism_labels",
    "detect_procedure_complication_labels",
    "detect_labels_for_shape",
    "compute_shadow_score",
]
