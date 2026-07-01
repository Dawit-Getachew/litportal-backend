"""Phase R1B — deterministic query-expansion module.

NO LLM. NO external lookups. Pure compilation from closed-set
curated concept maps + generalizable shape-vocab. Audit-traceable.

Two outputs per call:
  * ``broad_family`` — OR'd union of curated map terms +
    generalizable shape vocab.
  * ``high_precision_family`` — REQUIRED for the
    ``procedure_complication_risk`` shape; ``None`` for all other
    shapes. Form: ``(procedure_anchor) AND (outcome_anchor)``.

The R1B simulator OR-combines the broad and high-precision
streams when running against PubMed / Europe PMC connectors so
articles matched by either stream enter the candidate set,
tagged with closed-set provenance.

This module is intentionally pure — every input maps
deterministically to a single output object. The sidecar writer
persists the exact families used so the closeout can show the
deterministic compilation.
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------
# Curated concept maps — closed-set. Verbatim from the R1B brief.
# ---------------------------------------------------------------------
#
# Each entry is keyed by a closed-set map name. Term order is
# stable so the compiled query string is reproducible. NO PHI. NO
# clinician identifiers.
CURATED_EXPANSION_MAPS: dict[str, list[str]] = {
    "cholestatic_liver_disease_mechanism": [
        "cholestasis",
        "cholestatic pruritus",
        "cholestasis-associated itch",
        "itch",
        "pruritogen",
        "autotaxin",
        "lysophosphatidic acid",
        "bile acids",
        "MRGPRX4",
        "TGR5",
    ],
    "amyloid_cardiomyopathy_mechanism": [
        "cardiac amyloidosis",
        "transthyretin amyloid cardiomyopathy",
        "ATTR-CM",
        "AL amyloidosis",
        "restrictive cardiomyopathy",
        "HFpEF",
        "myocardial stiffness",
        "impaired relaxation",
        "extracellular volume",
        "strain",
    ],
    "pcos_insulin_resistance_mechanism": [
        "insulin signaling",
        "hyperandrogenism",
        "inflammation",
        "oxidative stress",
        "adipose dysfunction",
        "WNT/beta-catenin",
        "ovarian steroidogenesis",
        "metabolic dysfunction",
    ],
    # Procedure-complication side has TWO maps that compose into the
    # high-precision (procedure AND outcome) family.
    "central_line_pneumothorax_complication": [
        "central venous catheterization",
        "central venous catheter",
        "CVC",
        "central venous access",
        "subclavian catheterization",
        "internal jugular catheterization",
    ],
    "pneumothorax_outcome": [
        "iatrogenic pneumothorax",
        "catheter-related pneumothorax",
        "procedure-related pneumothorax",
    ],
    # Outcome-boost vocab. Lower-priority — applied as an
    # additional OR family in the broad stream.
    "procedure_complication_boost": [
        "incidence",
        "risk",
        "risk factor",
        "complication rate",
        "cohort",
        "registry",
        "systematic review",
    ],
}

# ---------------------------------------------------------------------
# Generalizable extension hooks — closed-set vocab applied to ALL
# sessions on the matching shape (not just the curated cases).
# ---------------------------------------------------------------------
MECHANISM_CONCEPT_TERMS: tuple[str, ...] = (
    "mechanism",
    "pathophysiology",
    "pathogenesis",
    "signaling",
    "pathway",
    "molecular",
    "inflammatory",
    "metabolic",
    "fibrosis",
    "stiffness",
    "relaxation",
    "pruritogen",
)

PROCEDURE_COMPLICATION_OUTCOME_VOCAB: tuple[str, ...] = (
    "complication",
    "complication rate",
    "incidence",
    "risk",
    "risk factor",
    "adverse event",
    "iatrogenic",
    "procedure-related",
    "catheter-related",
)

# Closed-set procedure-anchor vocab used when no curated map matches
# the session's procedure entity. Used to build the high-precision
# (procedure AND outcome) family even off the curated path.
GENERIC_PROCEDURE_ANCHOR_VOCAB: tuple[str, ...] = (
    "procedure",
    "catheterization",
    "catheter placement",
    "intervention",
    "insertion",
)

# Closed-set: shapes where R1B will produce an expansion. Other
# shapes are explicitly NOT expanded (we don't have curated maps
# for them and the operator brief restricts R1B to these two).
R1B_ELIGIBLE_SHAPES: frozenset[str] = frozenset({
    "pathophysiology_mechanism",
    "procedure_complication_risk",
    "procedure_complication",
})

# Closed-set provenance tags written per article in the R1B
# sidecar. Membership rules:
#   * ``high_precision_family`` — only articles surfaced by the
#     (procedure AND outcome) query family.
#   * ``broad_family``           — surfaced ONLY by the broad
#     OR-family.
#   * ``both``                   — surfaced by BOTH families.
#   * ``active_only``            — present in the bound session's
#     active top-N but not surfaced by either R1B stream (rare
#     fallback; can occur when R1B re-fetch returns fewer rows).
R1B_PROVENANCE_TAGS: frozenset[str] = frozenset({
    "high_precision_family",
    "broad_family",
    "both",
    "active_only",
})

# Map benchmark case id → curated expansion map name. Closed-set.
CASE_ID_TO_EXPANSION_MAP: dict[str, str] = {
    "r1a-case-A-cholestatic-pruritus": "cholestatic_liver_disease_mechanism",
    "r1a-case-B-pcos-insulin-resistance": "pcos_insulin_resistance_mechanism",
    "r1a-case-C-amyloid-diastolic-dysfunction": "amyloid_cardiomyopathy_mechanism",
    "r1a-case-D-cvc-pneumothorax": "central_line_pneumothorax_complication",
}


def select_expansion_map_for_case(case_id: str) -> Optional[str]:
    """Closed-set: return the curated expansion map name for the
    benchmark case. Returns None when the case is not in the
    curated set (the caller should fall back to shape-vocab only).
    """
    return CASE_ID_TO_EXPANSION_MAP.get(case_id)


def _or_join(terms: list[str]) -> str:
    """Produce an EPMC/PubMed-compatible OR'd term string.

    Multi-word terms are wrapped in double quotes so they're
    treated as phrases by both PubMed and Europe PMC. Single-word
    terms are bare. Deterministic ordering preserved.
    """
    parts: list[str] = []
    for t in terms:
        s = (t or "").strip()
        if not s:
            continue
        if " " in s or "/" in s or "-" in s:
            parts.append(f'"{s}"')
        else:
            parts.append(s)
    return " OR ".join(parts)


def compile_expanded_queries(
    *,
    shape: str,
    expansion_map_name: Optional[str],
) -> dict[str, object]:
    """Pure compilation.

    Returns a closed-set dict:
      {
        "shape": str,
        "expansion_map_used": Optional[str],
        "broad_family_terms": list[str],
        "high_precision_family_terms": Optional[dict] (only set
            when shape is procedure_complication_risk),
        "broad_family_query_string": str,
        "high_precision_family_query_string": Optional[str],
        "high_precision_family_required_both_anchors": bool,
      }

    The R1B sidecar writes ``expansion_map_used``,
    ``broad_family_terms``, and (when present) the high-precision
    family terms verbatim for the audit trail.
    """
    map_terms: list[str] = []
    if expansion_map_name and expansion_map_name in CURATED_EXPANSION_MAPS:
        map_terms = list(CURATED_EXPANSION_MAPS[expansion_map_name])

    # Closed-set: shape-specific generalizable vocab.
    if shape in ("pathophysiology_mechanism",):
        broad_extras = list(MECHANISM_CONCEPT_TERMS)
    elif shape in ("procedure_complication_risk", "procedure_complication"):
        broad_extras = list(PROCEDURE_COMPLICATION_OUTCOME_VOCAB)
    else:
        broad_extras = []

    # The broad family is the union of curated map + shape vocab.
    # Dedupe while preserving insertion order.
    seen: set[str] = set()
    broad_terms: list[str] = []
    for t in [*map_terms, *broad_extras]:
        key = t.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        broad_terms.append(t)

    broad_query = _or_join(broad_terms)

    high_precision_query: Optional[str] = None
    high_precision_terms: Optional[dict[str, list[str]]] = None
    high_precision_required = False

    if shape in ("procedure_complication_risk", "procedure_complication"):
        # Compose the procedure anchor and the outcome anchor. The
        # procedure anchor draws from the curated procedure map (e.g.,
        # ``central_line_pneumothorax_complication``) plus generic
        # procedure vocab. The outcome anchor draws from the curated
        # outcome map (e.g., ``pneumothorax_outcome``) plus generic
        # outcome vocab.
        procedure_terms: list[str] = []
        outcome_terms: list[str] = []
        if expansion_map_name == "central_line_pneumothorax_complication":
            procedure_terms = list(
                CURATED_EXPANSION_MAPS["central_line_pneumothorax_complication"]
            )
            outcome_terms = list(
                CURATED_EXPANSION_MAPS["pneumothorax_outcome"]
            )
        else:
            # Generic fallback path.
            procedure_terms = list(GENERIC_PROCEDURE_ANCHOR_VOCAB)
            outcome_terms = list(PROCEDURE_COMPLICATION_OUTCOME_VOCAB)
        # Append generic boost vocab to outcomes (e.g., ``incidence``,
        # ``risk factor``) — these strengthen the AND condition's
        # outcome side without weakening it.
        for boost in CURATED_EXPANSION_MAPS["procedure_complication_boost"]:
            if boost.lower() not in {o.lower() for o in outcome_terms}:
                outcome_terms.append(boost)
        proc_or = _or_join(procedure_terms)
        out_or = _or_join(outcome_terms)
        high_precision_query = f"({proc_or}) AND ({out_or})"
        high_precision_terms = {
            "procedure_anchor_terms": procedure_terms,
            "outcome_anchor_terms": outcome_terms,
        }
        high_precision_required = True

    return {
        "shape": shape,
        "expansion_map_used": expansion_map_name,
        "broad_family_terms": broad_terms,
        "high_precision_family_terms": high_precision_terms,
        "broad_family_query_string": broad_query,
        "high_precision_family_query_string": high_precision_query,
        "high_precision_family_required_both_anchors": (
            high_precision_required
        ),
    }


# ---------------------------------------------------------------------
# Closed-set: PubMed / EPMC URL practical query-length ceiling. We
# enforce this ceiling defensively so the simulator never silently
# truncates terms. ``EXCEEDS_URL_BUDGET`` is surfaced in the sidecar
# safety_flags so the closeout can document any case that hits it.
# ---------------------------------------------------------------------
MAX_QUERY_STRING_LEN: int = 1800  # conservative bound for both EPMC + PubMed esearch


def query_exceeds_length_budget(q: Optional[str]) -> bool:
    return bool(q) and len(q or "") > MAX_QUERY_STRING_LEN


__all__ = [
    "CURATED_EXPANSION_MAPS",
    "MECHANISM_CONCEPT_TERMS",
    "PROCEDURE_COMPLICATION_OUTCOME_VOCAB",
    "GENERIC_PROCEDURE_ANCHOR_VOCAB",
    "R1B_ELIGIBLE_SHAPES",
    "R1B_PROVENANCE_TAGS",
    "CASE_ID_TO_EXPANSION_MAP",
    "MAX_QUERY_STRING_LEN",
    "select_expansion_map_for_case",
    "compile_expanded_queries",
    "query_exceeds_length_budget",
]
