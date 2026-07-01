"""Phase M4.0.1-shadow — Compiler-Input Contract for accepted M4 candidates.

PURE DETERMINISTIC PROJECTION. SHADOW ONLY. PREVIEW/DEV ONLY.

Closes the gap between an ACCEPTED M4 ClinicalQuestionMap candidate and the
would-be compiler input: instead of carrying only ``required_anchors`` (which
risks broadening a stage-specific question to the bare disease), we project a
complete, compiler-ready object that preserves structured context (stage,
biomarker, population, outcomes, mechanism, procedure, drug) under conservative
slot policies, plus dry-run query families.

HARD invariants:
  * NEVER dispatches retrieval. NEVER calls PubMed / Europe PMC / Crossref /
    MyHealthfinder / any source. Produces structured slots + dry-run families only.
  * NEVER promotes unvalidated MeSH/synonyms into query families (they stay in
    ``*_unvalidated`` slots; validation rejects any leak).
  * required_anchors are the ONLY broad-recall AND concepts. stage / biomarker /
    outcome / population are precision/ranking terms — they must NOT shrink recall
    on the broad family, and must NOT be dropped from the precision families.
"""
from __future__ import annotations

from typing import Any, List, Optional

from services.m4_full_question_interpreter import (
    M4Candidate,
    _is_long_literal,
    _generic_only_required_anchor,
    _load_generic_denylist,
)

# Closed-set projection-validation flags.
M4_PROJECTION_FLAGS: frozenset[str] = frozenset({
    "stage_dropped",
    "biomarker_dropped",
    "disease_anchor_generic",
    "literal_phrase_in_required_anchor",
    "ranking_terms_over_anded",
    "stage_specific_question_broadened",
    "unvalidated_mesh_applied",
    "no_required_anchor",
})

# Intent families that unlock the corresponding dry-run query family.
_PROGNOSIS_INTENTS: frozenset[str] = frozenset({
    "prognosis", "prognosis_natural_history", "prognostic_factor",
})
_MECHANISM_INTENTS: frozenset[str] = frozenset({
    "mechanism", "pathophysiology",
})
_PROCEDURE_COMPLICATION_INTENTS: frozenset[str] = frozenset({
    "complication_after_procedure", "complication_risk",
    "procedure_complication_risk",
})
_DRUG_SAFETY_INTENTS: frozenset[str] = frozenset({
    "drug_safety", "therapy_safety", "other",
})


def _dedupe(seq: List[str]) -> List[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq or []:
        k = (s or "").strip()
        if not k:
            continue
        kl = k.lower()
        if kl in seen:
            continue
        seen.add(kl)
        out.append(k)
    return out


def _lower_set(seq: List[str]) -> set[str]:
    return {(s or "").strip().lower() for s in (seq or []) if (s or "").strip()}


def project_compiler_input(candidate: M4Candidate) -> dict[str, Any]:
    """Deterministically map an accepted M4 candidate to a compiler-ready
    object. Pure: no dispatch, no source calls."""
    required = _dedupe(candidate.required_anchors)
    stage = _dedupe(candidate.stage)
    biomarker = _dedupe(candidate.biomarker)
    population = _dedupe(candidate.population)
    outcome = _dedupe(candidate.outcome)
    procedure = _dedupe(candidate.procedure)
    drug = _dedupe(candidate.medication)
    mechanism = _dedupe(candidate.mechanism)
    setting = _dedupe(candidate.setting)
    timeframe = _dedupe(candidate.timeframe)
    comparator = _dedupe(candidate.comparator)
    exposure = _dedupe(candidate.exposure)
    optional = _dedupe(candidate.optional_or_ranking_only_terms)
    synonyms = _dedupe(candidate.candidate_synonyms_unvalidated)
    mesh = _dedupe(candidate.candidate_mesh_terms_unvalidated)

    # Conservative slot policy (see module docstring / spec §2).
    context_terms = _dedupe(population + setting + timeframe + comparator + exposure)
    high_precision_terms = _dedupe(stage + biomarker + mechanism)
    ranking_only_terms = _dedupe(optional + stage + biomarker + outcome + population)

    query_policy = {
        "required_anchors": "broad_recall_AND",
        "context_terms": "precision_and_ranking_only",
        "high_precision_terms": "precision_family_AND_plus_ranking",
        "ranking_only_terms": "boost_not_filter",
        "stage_terms": "precision_and_ranking_never_broad_required",
        "biomarker_terms": "precision_and_ranking_never_broad_required",
        "outcome_terms": "ranking_and_prognosis_precision",
        "population_terms": "ranking_unless_central",
        "procedure_terms": "required_when_central_else_precision",
        "drug_terms": "required_when_central_else_precision",
        "mechanism_terms": "mechanism_precision_and_ranking",
        "synonym_terms_unvalidated": "not_applied_pending_validation",
        "mesh_terms_unvalidated": "not_applied_pending_validation",
    }

    return {
        "required_anchors": required,
        "context_terms": context_terms,
        "high_precision_terms": high_precision_terms,
        "ranking_only_terms": ranking_only_terms,
        "optional_terms": optional,
        "outcome_terms": outcome,
        "stage_terms": stage,
        "biomarker_terms": biomarker,
        "population_terms": population,
        "procedure_terms": procedure,
        "drug_terms": drug,
        "mechanism_terms": mechanism,
        "synonym_terms_unvalidated": synonyms,
        "mesh_terms_unvalidated": mesh,
        "query_policy": query_policy,
    }


def project_query_families(
    candidate: M4Candidate, compiler_input: dict[str, Any],
) -> list[dict[str, Any]]:
    """Dry-run query families ONLY — structured slot bundles, never compiled
    or dispatched. ``must`` = AND (broad recall anchors); ``precision`` = AND
    qualifiers (recall-preserving precision family); ``should`` = boost/OR.
    Unvalidated MeSH/synonyms are NEVER placed in any family.
    """
    required = compiler_input["required_anchors"]
    stage = compiler_input["stage_terms"]
    biomarker = compiler_input["biomarker_terms"]
    outcome = compiler_input["outcome_terms"]
    mechanism = compiler_input["mechanism_terms"]
    procedure = compiler_input["procedure_terms"]
    drug = compiler_input["drug_terms"]
    population = compiler_input["population_terms"]

    intent = candidate.primary_intent
    families: list[dict[str, Any]] = []

    # broad_recall — recall anchor only (no precision filters).
    families.append({
        "family": "broad_recall",
        "must": list(required),
        "precision": [],
        "should": _dedupe(population + outcome),
        "note": "recall anchor(s) only; precision/ranking terms boost, never filter",
    })

    # high_precision — anchor AND stage/biomarker precision (recall preserved
    # because broad_recall family still exists).
    if stage or biomarker:
        families.append({
            "family": "high_precision",
            "must": list(required),
            "precision": _dedupe(stage + biomarker),
            "should": list(outcome),
            "note": "stage/biomarker as AND precision; broad_recall retains recall",
        })

    # prognosis_secondary_evidence — when prognosis intent.
    if intent in _PROGNOSIS_INTENTS:
        families.append({
            "family": "prognosis_secondary_evidence",
            "must": list(required),
            "precision": list(stage),
            "should": _dedupe(outcome + biomarker),
            "note": "prognosis outcomes as ranking/should; stage as precision",
        })

    # mechanism_translational — when mechanism present.
    if intent in _MECHANISM_INTENTS or mechanism:
        families.append({
            "family": "mechanism_translational",
            "must": list(required),
            "precision": list(mechanism),
            "should": [],
            "note": "mechanism target as precision",
        })

    # procedure_complication — when procedure complication.
    if intent in _PROCEDURE_COMPLICATION_INTENTS and (procedure or required):
        families.append({
            "family": "procedure_complication",
            "must": _dedupe(procedure or required),
            "precision": list(outcome),
            "should": [],
            "note": "procedure as anchor; complication outcome as precision",
        })

    # medication_safety — when drug safety.
    if intent in _DRUG_SAFETY_INTENTS and drug:
        families.append({
            "family": "medication_safety",
            "must": list(drug),
            "precision": [],
            "should": _dedupe(["safety", "adverse effects", "dosing"]),
            "note": "medication as anchor; safety terms as ranking/should",
        })

    return families


def validate_projection(
    candidate: M4Candidate,
    compiler_input: dict[str, Any],
    families: list[dict[str, Any]],
) -> list[str]:
    """Closed-set validation. Returns matched ``M4_PROJECTION_FLAGS`` (empty =
    clean)."""
    flags: list[str] = []
    required = compiler_input["required_anchors"]
    denylist = _load_generic_denylist()

    if not required:
        flags.append("no_required_anchor")
    elif _generic_only_required_anchor(required, generic_denylist=denylist):
        flags.append("disease_anchor_generic")

    if any(_is_long_literal(a) for a in required):
        flags.append("literal_phrase_in_required_anchor")

    # Structured context preservation.
    if candidate.stage and not compiler_input["stage_terms"]:
        flags.append("stage_dropped")
    if candidate.biomarker and not compiler_input["biomarker_terms"]:
        flags.append("biomarker_dropped")

    # Ranking/optional terms must NOT be AND-ed (in ``must``) across ALL families.
    ranking = _lower_set(compiler_input["ranking_only_terms"])
    if ranking and families and all(
        ranking & _lower_set(f.get("must", [])) for f in families
    ):
        flags.append("ranking_terms_over_anded")

    # A clearly stage-specific question must keep stage in some family
    # (precision/must/should) — never broadened to the bare disease only.
    if candidate.stage:
        stage_set = _lower_set(candidate.stage)
        in_any = any(
            stage_set & _lower_set(
                list(f.get("precision", [])) + list(f.get("must", [])) + list(f.get("should", []))
            )
            for f in families
        )
        if not in_any:
            flags.append("stage_specific_question_broadened")

    # Unvalidated MeSH must NEVER appear inside a query family — UNLESS the
    # same string is independently a legitimate clinical concept (e.g.,
    # "pneumothorax" can be both an outcome term and a MeSH heading). Only a
    # term that is EXCLUSIVELY an unvalidated MeSH heading constitutes a leak.
    mesh = _lower_set(compiler_input["mesh_terms_unvalidated"])
    legitimate = (
        _lower_set(compiler_input["required_anchors"])
        | _lower_set(compiler_input["stage_terms"])
        | _lower_set(compiler_input["biomarker_terms"])
        | _lower_set(compiler_input["outcome_terms"])
        | _lower_set(compiler_input["procedure_terms"])
        | _lower_set(compiler_input["drug_terms"])
        | _lower_set(compiler_input["mechanism_terms"])
        | _lower_set(compiler_input["population_terms"])
        | _lower_set(compiler_input["context_terms"])
    )
    mesh_only = mesh - legitimate
    if mesh_only:
        for f in families:
            applied = _lower_set(
                list(f.get("must", [])) + list(f.get("should", [])) + list(f.get("precision", []))
            )
            if mesh_only & applied:
                flags.append("unvalidated_mesh_applied")
                break

    # Dedupe, closed-set.
    seen: set[str] = set()
    out: list[str] = []
    for fl in flags:
        if fl in seen or fl not in M4_PROJECTION_FLAGS:
            continue
        seen.add(fl)
        out.append(fl)
    return out


# Candidate slot -> projected compiler-input slot (for preserved/dropped report).
_SLOT_MAP: tuple[tuple[str, str], ...] = (
    ("clinical_problem", "required_anchors"),
    ("stage", "stage_terms"),
    ("biomarker", "biomarker_terms"),
    ("population", "population_terms"),
    ("outcome", "outcome_terms"),
    ("procedure", "procedure_terms"),
    ("medication", "drug_terms"),
    ("mechanism", "mechanism_terms"),
)


def compute_preserved_dropped(
    candidate: M4Candidate, compiler_input: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Report which candidate slots survived into the projected compiler input."""
    cand_dump = candidate.model_dump()
    preserved: list[str] = []
    dropped: list[str] = []
    for cand_slot, proj_slot in _SLOT_MAP:
        cand_val = cand_dump.get(cand_slot) or []
        proj_val = compiler_input.get(proj_slot) or []
        if not cand_val:
            continue
        if proj_val:
            preserved.append(proj_slot)
        else:
            dropped.append(proj_slot)
    return preserved, dropped


def build_m4_0_1_projection(
    candidate: Optional[M4Candidate],
    *,
    baseline_primary_intent: Optional[str] = None,
    baseline_frameworks: Optional[List[str]] = None,
    baseline_required_anchors: Optional[List[str]] = None,
) -> dict[str, Any]:
    """Top-level M4.0.1 projection for an accepted candidate. Returns the full
    diagnostics block (baseline compiler input, projected compiler input,
    preserved/dropped slots, query_policy, dry-run query families, validation).
    Returns ``{}`` when there is no accepted candidate.
    """
    if candidate is None:
        return {}
    compiler_input = project_compiler_input(candidate)
    families = project_query_families(candidate, compiler_input)
    flags = validate_projection(candidate, compiler_input, families)
    preserved, dropped = compute_preserved_dropped(candidate, compiler_input)
    return {
        "baseline_compiler_input": {
            "primary_intent": baseline_primary_intent,
            "frameworks": list(baseline_frameworks or []),
            "required_anchors": list(baseline_required_anchors or []),
        },
        "projected_compiler_input": compiler_input,
        "query_policy": compiler_input["query_policy"],
        "dry_run_query_families": families,
        "preserved_slots": preserved,
        "dropped_slots": dropped,
        "projection_validation_status": "passed" if not flags else "flagged",
        "projection_validation_flags": flags,
        "active_retrieval_changed": False,
        "dispatched_any_source": False,
    }


__all__ = [
    "M4_PROJECTION_FLAGS",
    "project_compiler_input",
    "project_query_families",
    "validate_projection",
    "compute_preserved_dropped",
    "build_m4_0_1_projection",
]
