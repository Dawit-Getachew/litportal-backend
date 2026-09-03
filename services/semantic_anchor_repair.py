# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M0/M1/M2 semantic-repair shadow,
# superseded by the M4 full-question ClinicalQuestionMap interpreter. NOT invoked
# in the active pipeline (gated off via
# pipeline._LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED=False). Retained for
# service-level unit tests only. Do not re-wire into the forward semantic path.

"""Phase M0/M1/M2-shadow — Bounded Clinical Semantics & Context Interpreter v2.

The module runs ONLY in shadow mode for sessions where:

  1. The deterministic question-shape routes (K4A.2 severity-marker /
     K4A.3 rule-out / K4A.4 epidemiology) did NOT fire AND
  2. The deterministic anchor pipeline produced an interpretation that
     looks suspicious by closed-set grammar patterns (e.g. literal
     full-phrase anchor, missing risk/etiology framework on a "causes
     of X in Y" shape, missing mechanism framework, missing
     complication-after-procedure split).

When triggered, the module asks the bounded LLM (the SAME
``llm_provider.complete_json`` path used by the H1 Clinical Context
Resolver) to emit a closed-set, schema-locked candidate interpretation.
The candidate is deterministically validated against the closed-set
rejection-reason enum below, scrubbed for PHI / article-references /
unapproved-sources / answer-text, and stored in shadow diagnostics
ONLY — active retrieval, ranking, default selection, evidence packets,
summaries, and citation verification are NEVER affected by this phase.

Safety invariants enforced by THIS module:

  * NEVER searches sources. NEVER recommends articles. NEVER generates
    clinical answers. NEVER emits PMIDs / DOIs / specific article
    references. NEVER invents validated MeSH (the schema's MeSH-like
    fields carry an ``_unvalidated`` suffix and must never be promoted
    to validated MeSH).
  * NEVER references Google Scholar / ResearchGate / Sci-Hub.
  * CANNOT override deterministic caps, K4A.2.1 generic-anchor
    denylist, K4A.3 modules, K4A.* directness labels, ranking, or
    branch-aware retrieval activation (which stays shadow).
  * NEVER runs when PHI was detected upstream (the pipeline short-
    circuits to ``blocked_phi`` long before this module is reached).
    The module ALSO scans every output string for PHI patterns as
    defense-in-depth.

NO raw question text, prompt, or completion is persisted anywhere
downstream. Only the validated structured candidate, closed-set
trigger/rejection reasons, and counters reach
``session_diagnostics_json``.
"""
from __future__ import annotations

import logging
import os as _os
import re
import time
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from services.diagnostics import SessionDiagnostics
from services.llm_provider import llm_provider

logger = logging.getLogger("clinevidence.semantic_anchor_repair")


# ---------------------------------------------------------------------
# Closed-set TRIGGER reasons emitted by the deterministic detector.
# Each reason is a single, stable identifier so admin diagnostics can
# group/aggregate without parsing free text.
# ---------------------------------------------------------------------
SEMANTIC_REPAIR_TRIGGER_REASONS: frozenset[str] = frozenset({
    "literal_full_phrase_anchor",
    "missing_risk_etiology_framework",
    "missing_mechanism_pathophysiology_framework",
    "missing_complication_after_procedure_split",
    "missing_prognosis_framework",
    "missing_epidemiology_framework_grammar",
    "missing_severity_marker_split",
    "missing_diagnostic_ruleout_split",
    "disease_population_fused_into_single_anchor",
})


# Closed-set REJECTION reasons. Mirrors the H1 resolver pattern.
# Phase M2.1 extended this set with more specific failure-mode names so the
# admin diagnostics surface can distinguish (e.g.) "extra_field" from
# "invalid_enum" from "invalid_slot_policy" without staring at an opaque
# "schema_invalid". The legacy names are RETAINED for backward compatibility
# with the M0/M1/M2 unit tests; the validator emits the NEW specific names
# going forward.
SEMANTIC_REPAIR_REJECTION_REASONS: frozenset[str] = frozenset({
    # Legacy umbrella reasons (M0/M1/M2). Retained for back-compat.
    "schema_invalid",
    "extra_fields_present",
    "no_valid_required_anchor",
    "generic_modifier_only_required_anchor",
    "confidence_below_threshold",
    "enum_violation",
    "phi_pattern_detected",
    "article_reference_detected",
    "unapproved_source_referenced",
    "answer_text_detected",
    "unvalidated_mesh_promoted",
    "branch_count_exceeded",
    "would_remove_central_anchor",
    "provenance_mismatch",
    "empty_output",
    # Phase M2.1 — specific failure modes. The validator emits these
    # going forward. Pydantic v2 ValidationError categories are mapped
    # deterministically to one of the below by ``_pydantic_errors_to_closed_set``.
    "missing_required_field",
    "extra_field",
    "invalid_enum",
    "invalid_branch_shape",
    "invalid_slot_type",
    "invalid_slot_policy",
    "invalid_confidence",
    "empty_required_anchors",
    "generic_only_required_anchor",
    "unvalidated_mesh_marked_validated",
    "malformed_json",
    "unsafe_text_detected",
    "refused",
})


# Closed-set primary-intent enum (drawn from clinical_question_map +
# H1 resolver vocabularies — values that already exist in the codebase
# closed-set surface area, so admin serializers can route uniformly).
ALLOWED_PRIMARY_INTENTS: frozenset[str] = frozenset({
    "background_overview", "disease_overview", "drug_overview",
    "definition",
    "therapy", "therapy_safety", "drug_safety",
    "diagnostic_accuracy", "diagnosis",
    "risk_etiology", "risk_etiology_harm", "etiology",
    "complication_risk", "complication_after_procedure",
    "epidemiology_prevalence_incidence",
    "prognosis", "prognosis_natural_history", "prognostic_factor",
    "mechanism", "pathophysiology",
    "severity_marker_overview",
    "rule_out_test_utility",
    "screening",
    "other",
})


ALLOWED_FRAMEWORKS: frozenset[str] = frozenset({
    "PICO", "PICOT", "PICOTT", "CoCoPop", "PIRD", "PFO",
    "risk_etiology", "mechanism", "prognosis",
    "complication", "diagnostic_accuracy", "narrative_overview",
})


ALLOWED_SLOT_POLICIES: frozenset[str] = frozenset({
    "required_for_retrieval",
    "optional_for_retrieval",
    "display_only",
    "ranking_only",
})


# ---------------------------------------------------------------------
# Phase M2.1 — closed-set enum canonicalization map.
#
# Applied as a pure-function pre-validation pass to map common LLM
# phrasings to the schema's approved enum values. NEW VALUES ARE NEVER
# CREATED — every alias must map to an enum value that already exists
# in ``ALLOWED_PRIMARY_INTENTS``, ``ALLOWED_FRAMEWORKS``,
# ``ALLOWED_SLOT_POLICIES``, or ``SEMANTIC_REPAIR_TRIGGER_REASONS``.
# Unknown aliases pass through unchanged and fail strict validation
# downstream. Sorted one-alias-per-line for ease of review.
# ---------------------------------------------------------------------

_PRIMARY_INTENT_ALIASES: dict[str, str] = {
    # Risk / etiology family.
    "aetiology": "risk_etiology",
    "causes": "risk_etiology",
    "etiology_or_risk": "risk_etiology",
    "etiology/risk_factors": "risk_etiology",
    "risk": "risk_etiology",
    "risk_etiology_or_harm": "risk_etiology_harm",
    "risk_factor": "risk_etiology",
    "risk_factors": "risk_etiology",
    # Mechanism / pathophysiology family — both canonical values
    # already exist; treat lowercase typos as no-ops.
    "mechanism_of_action": "mechanism",
    "mechanism_pathophysiology": "mechanism",
    "pathophysiology_mechanism": "pathophysiology",
    # Diagnosis / rule-out family.
    "diagnostic": "diagnostic_accuracy",
    "diagnostic_test": "diagnostic_accuracy",
    "diagnostic_test_accuracy": "diagnostic_accuracy",
    "diagnostic_utility": "diagnostic_accuracy",
    "diagnostic_value": "diagnostic_accuracy",
    "negative_predictive_value": "rule_out_test_utility",
    "npv": "rule_out_test_utility",
    "rule_out": "rule_out_test_utility",
    "rule_out_diagnosis": "rule_out_test_utility",
    # Complication family.
    "complication": "complication_risk",
    "complications": "complication_risk",
    "post_procedure_complication": "complication_after_procedure",
    "post_procedural_complication": "complication_after_procedure",
    "postprocedural_complication": "complication_after_procedure",
    # Epidemiology family.
    "epidemiology": "epidemiology_prevalence_incidence",
    "frequency": "epidemiology_prevalence_incidence",
    "incidence": "epidemiology_prevalence_incidence",
    "occurrence": "epidemiology_prevalence_incidence",
    "prevalence": "epidemiology_prevalence_incidence",
    # Prognosis family — keep canonical pair.
    "long_term_outcome": "prognosis_natural_history",
    "long_term_outcomes": "prognosis_natural_history",
    "mortality": "prognosis",
    "natural_history": "prognosis_natural_history",
    "survival": "prognosis",
    # Therapy family.
    "drug_therapy": "therapy",
    "pharmacotherapy": "therapy",
    "treatment": "therapy",
    "treatment_safety": "therapy_safety",
    # Overview / definition.
    "background": "background_overview",
    "definition_of": "definition",
    "disease_definition": "disease_overview",
    "drug_definition": "drug_overview",
    "overview": "background_overview",
}


_FRAMEWORK_ALIASES: dict[str, str] = {
    # Casing fixes.
    "cocopop": "CoCoPop",
    "COCOPOP": "CoCoPop",
    "Cocopop": "CoCoPop",
    "pico": "PICO",
    "pico_t": "PICOT",
    "pico_tt": "PICOTT",
    "picot": "PICOT",
    "picotss": "PICOTT",
    "picott": "PICOTT",
    "pird": "PIRD",
    "pfo": "PFO",
    # Family-grouping fixes.
    "complication_framework": "complication",
    "diagnostic": "diagnostic_accuracy",
    "diagnostic_accuracy_framework": "diagnostic_accuracy",
    "mechanism_framework": "mechanism",
    "narrative": "narrative_overview",
    "narrative_review": "narrative_overview",
    "pathophysiology": "mechanism",
    "peco": "CoCoPop",
    "PECO": "CoCoPop",
    "prognosis_framework": "prognosis",
    "risk_etiology_framework": "risk_etiology",
}


_SLOT_POLICY_ALIASES: dict[str, str] = {
    "display": "display_only",
    "mandatory": "required_for_retrieval",
    "optional": "optional_for_retrieval",
    "rank": "ranking_only",
    "ranking": "ranking_only",
    "required": "required_for_retrieval",
}


_SEMANTIC_REPAIR_REASON_ALIASES: dict[str, str] = {
    "complication_after_procedure": "missing_complication_after_procedure_split",
    "diagnostic_ruleout": "missing_diagnostic_ruleout_split",
    "disease_population_fused": "disease_population_fused_into_single_anchor",
    "epidemiology": "missing_epidemiology_framework_grammar",
    "literal_anchor": "literal_full_phrase_anchor",
    "mechanism": "missing_mechanism_pathophysiology_framework",
    "pathophysiology": "missing_mechanism_pathophysiology_framework",
    "prognosis": "missing_prognosis_framework",
    "risk_etiology": "missing_risk_etiology_framework",
    "severity_marker": "missing_severity_marker_split",
}


def _canonicalize_str_enum(
    raw: Optional[str], alias_map: dict[str, str],
) -> Optional[str]:
    """Map ``raw`` through ``alias_map`` after a closed-set
    normalization (lowercase + strip + collapse to ``_``). NEW VALUES
    ARE NEVER PRODUCED — only existing alias-map targets are returned;
    everything else passes through unchanged.
    """
    if raw is None or not isinstance(raw, str):
        return raw
    key = raw.strip()
    if not key:
        return raw
    # Try literal first (preserves casing for things like PICOTT).
    if key in alias_map:
        return alias_map[key]
    # Normalized.
    norm = "_".join(key.lower().replace("/", " ").replace("-", " ").split())
    if norm in alias_map:
        return alias_map[norm]
    return raw


def canonicalize_candidate(raw: Optional[dict]) -> Optional[dict]:
    """Apply the closed-set alias maps to a raw LLM dict BEFORE strict
    Pydantic validation. NEVER mutates the input. Returns a NEW dict
    when changes were applied, or the same reference if not.
    """
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    # primary_intent.
    if "primary_intent" in out:
        out["primary_intent"] = _canonicalize_str_enum(
            out.get("primary_intent"), _PRIMARY_INTENT_ALIASES,
        )
    # secondary_intents — list of strings.
    if isinstance(out.get("secondary_intents"), list):
        out["secondary_intents"] = [
            _canonicalize_str_enum(s, _PRIMARY_INTENT_ALIASES)
            for s in out["secondary_intents"]
        ]
    # frameworks — list of strings.
    if isinstance(out.get("frameworks"), list):
        out["frameworks"] = [
            _canonicalize_str_enum(s, _FRAMEWORK_ALIASES)
            for s in out["frameworks"]
        ]
    # slot_policies — dict[str,str]; canonicalize values only.
    if isinstance(out.get("slot_policies"), dict):
        out["slot_policies"] = {
            k: _canonicalize_str_enum(v, _SLOT_POLICY_ALIASES)
            for k, v in out["slot_policies"].items()
        }
    # semantic_repair_reason.
    if "semantic_repair_reason" in out:
        out["semantic_repair_reason"] = _canonicalize_str_enum(
            out.get("semantic_repair_reason"),
            _SEMANTIC_REPAIR_REASON_ALIASES,
        )
    return out


# Confidence floor — env-tunable per the M-phase brief, but default 0.7.
CONFIDENCE_THRESHOLD: float = float(
    (_os.environ.get("SEMANTIC_REPAIR_CONFIDENCE_THRESHOLD") or "0.7").strip()
    or "0.7"
)
BRANCH_COUNT_CAP: int = int(
    (_os.environ.get("SEMANTIC_REPAIR_BRANCH_CAP") or "6").strip() or "6"
)


# Closed-set subtypes that mean a K4A.* route has already fired and
# the semantic repair MUST be skipped (cross-phase preservation).
_K4A_FIRED_SUBTYPES: frozenset[str] = frozenset({
    "severity_marker_overview",
    "rule_out_test_utility",
    "epidemiology_incidence_complication_risk",
})


# ---------------------------------------------------------------------
# Deterministic suspicious-grammar regex bank.
# Closed-set patterns. No LLM in the detection step.
# ---------------------------------------------------------------------

_CAUSES_OF_X_IN_Y_RE = re.compile(
    r"\b(?:causes?|aetiolog(?:y|ies)|etiolog(?:y|ies)|"
    r"risk\s+factors?|predisposing\s+factors?)\s+(?:of|for)\s+",
    re.IGNORECASE,
)

_MECHANISM_RE = re.compile(
    r"\b(?:mechanism(?:s)?|pathophysiolog(?:y|ies)|pathogenesis|"
    r"underlying\s+cause(?:s)?)\s+(?:of|for|behind)\s+",
    re.IGNORECASE,
)

_COMPLICATION_AFTER_PROC_RE = re.compile(
    r"\b(?:complication(?:s)?|risk(?:s)?\s+of\s+\w+|infection|bleeding|"
    r"thrombosis|reoperation|migration|dislodg(?:e|ement))\s+"
    r"(?:after|following|post[\-\s]?)\s+",
    re.IGNORECASE,
)

_PROGNOSIS_RE = re.compile(
    r"\b(?:prognosis|natural\s+history|long[\-\s]?term\s+outcome(?:s)?|"
    r"prognostic\s+factor(?:s)?|survival|mortality)\s+(?:of|in|after|for)\s+",
    re.IGNORECASE,
)

_INCIDENCE_GRAMMAR_RE = re.compile(
    r"\b(?:incidence|prevalence|frequency|occurrence|rate)\s+of\s+",
    re.IGNORECASE,
)

_SEVERITY_MARKER_GRAMMAR_RE = re.compile(
    r"\b(?:marker(?:s)?|predictor(?:s)?|biomarker(?:s)?|"
    r"scoring\s+systems?|risk\s+scores?)\s+(?:of|for)\s+"
    r"(?:severe|severity)\b",
    re.IGNORECASE,
)

_DIAGNOSTIC_RULEOUT_GRAMMAR_RE = re.compile(
    r"\b(?:rule[\-\s]?out|rules?\s+out|negative\s+predictive\s+value|"
    r"sensitivity|specificity)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Detector — pure function, returns closed-set trigger reasons.
# ---------------------------------------------------------------------
def detect_repair_triggers(
    *,
    normalized_question: str,
    question_subtype: Optional[str],
    type_of_question: Optional[str],
    primary_anchors: List[dict],
    generic_modifiers: List[str],
    anchor_failure_reason: Optional[str],
) -> List[str]:
    """Closed-set deterministic detector. Returns the list of
    ``SEMANTIC_REPAIR_TRIGGER_REASONS`` that matched the inputs.

    The detector NEVER fires for sessions that already routed through
    K4A.2 / K4A.3 / K4A.4. Cross-phase preservation is enforced here
    AND in the pipeline guard for defense in depth.

    It also skips when ``anchor_failure_reason ==
    no_primary_clinical_entity`` because the bounded H1 Clinical
    Context Resolver already covers that path.
    """
    q = (normalized_question or "").strip()
    if not q:
        return []
    st = (question_subtype or "").strip().lower()
    if st in _K4A_FIRED_SUBTYPES:
        return []
    if (anchor_failure_reason or "").strip().lower() == "no_primary_clinical_entity":
        # Resolver path handles this case — out of M-phase scope.
        return []

    triggers: list[str] = []
    toq = (type_of_question or "").strip().lower()

    # (a) Literal full-phrase primary anchor — closed-set "≥5 words AND
    # contains a grammar connective" check. The K4A.2.1 anchor denylist
    # already covers a few specific phrases; this detector catches the
    # broader class.
    GRAMMAR_CONNECTIVES = (
        " in ", " of ", " for ", " after ", " following ", " with ",
        " among ", " during ", " from ",
    )
    for a in (primary_anchors or []):
        if not isinstance(a, dict):
            continue
        text = (
            a.get("canonical_concept")
            or a.get("text")
            or ""
        ).strip().lower()
        if not text:
            continue
        words = text.split()
        if len(words) >= 5 and any(c in f" {text} " for c in GRAMMAR_CONNECTIVES):
            triggers.append("literal_full_phrase_anchor")
            triggers.append("disease_population_fused_into_single_anchor")
            break

    # (b) Risk / etiology grammar not mapped to risk_etiology.
    if _CAUSES_OF_X_IN_Y_RE.search(q):
        if toq not in {
            "risk_etiology", "risk_etiology_harm", "etiology",
            "complication_risk", "complication_after_procedure",
        }:
            triggers.append("missing_risk_etiology_framework")

    # (c) Mechanism / pathophysiology grammar.
    if _MECHANISM_RE.search(q):
        if toq not in {"mechanism", "pathophysiology"}:
            triggers.append("missing_mechanism_pathophysiology_framework")

    # (d) Complication-after-procedure grammar.
    if _COMPLICATION_AFTER_PROC_RE.search(q):
        if toq not in {
            "complication_risk", "complication_after_procedure",
            "risk_etiology_harm",
        }:
            triggers.append("missing_complication_after_procedure_split")

    # (e) Prognosis grammar.
    if _PROGNOSIS_RE.search(q):
        if toq not in {
            "prognosis", "prognosis_natural_history", "prognostic_factor",
        }:
            triggers.append("missing_prognosis_framework")

    # (f) Epidemiology / incidence grammar — but K4A.4 should have
    # fired; flag only when it didn't.
    if _INCIDENCE_GRAMMAR_RE.search(q):
        if toq not in {"epidemiology", "epidemiology_prevalence_incidence"}:
            triggers.append("missing_epidemiology_framework_grammar")

    # (g) Severity-marker grammar — K4A.2 territory; flag only when
    # K4A.2 didn't fire.
    if _SEVERITY_MARKER_GRAMMAR_RE.search(q):
        triggers.append("missing_severity_marker_split")

    # (h) Diagnostic rule-out grammar — K4A.3 territory; flag only
    # when K4A.3 didn't fire.
    if _DIAGNOSTIC_RULEOUT_GRAMMAR_RE.search(q):
        triggers.append("missing_diagnostic_ruleout_split")

    # Closed-set dedupe — preserve insertion order.
    seen: set[str] = set()
    out: list[str] = []
    for t in triggers:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    # Final filter — every emitted reason must be in the closed-set.
    out = [t for t in out if t in SEMANTIC_REPAIR_TRIGGER_REASONS]
    return out


# ---------------------------------------------------------------------
# Pydantic schema (schema-locked, extra="forbid").
# ---------------------------------------------------------------------


class RepairSlots(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population: List[str] = Field(default_factory=list)
    condition: List[str] = Field(default_factory=list)
    intervention_or_exposure: List[str] = Field(default_factory=list)
    comparator: List[str] = Field(default_factory=list)
    outcome: List[str] = Field(default_factory=list)
    index_test: List[str] = Field(default_factory=list)
    target_condition: List[str] = Field(default_factory=list)
    procedure_or_device: List[str] = Field(default_factory=list)
    timeframe: List[str] = Field(default_factory=list)
    setting: List[str] = Field(default_factory=list)
    mechanism: List[str] = Field(default_factory=list)


class RepairBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    branch_label: str
    branch_intent: str
    anchors: List[str] = Field(default_factory=list)


class SemanticRepairCandidate(BaseModel):
    """Schema-locked candidate proposed by the bounded LLM. Strict
    extra-fields rejection; all enum-like fields validated post-load
    against the M-phase closed-set above.
    """

    model_config = ConfigDict(extra="forbid")

    primary_intent: str
    secondary_intents: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list, max_length=4)
    branches: List[RepairBranch] = Field(default_factory=list)
    slots: RepairSlots = Field(default_factory=RepairSlots)
    slot_policies: dict[str, str] = Field(default_factory=dict)
    required_anchors: List[str] = Field(default_factory=list)
    optional_or_ranking_only_concepts: List[str] = Field(default_factory=list)
    inferred_concepts: List[str] = Field(default_factory=list)
    candidate_keywords_unvalidated: List[str] = Field(default_factory=list)
    candidate_mesh_terms_unvalidated: List[str] = Field(default_factory=list)
    semantic_repair_reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: Literal["bounded_semantic_repair_shadow"]


# ---------------------------------------------------------------------
# Scrubbing patterns (mirrors clinical_context_resolver).
# ---------------------------------------------------------------------
_PMID_RE = re.compile(r"\b(?:PMID|pmid)\s*[:#]?\s*\d{5,}\b", re.IGNORECASE)
_BARE_LONG_DIGITS_RE = re.compile(r"\b\d{7,}\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+\b")
_ARTICLE_REF_RE = re.compile(
    r"\b(?:doi:|pmcid|nct\d+|clinicaltrials\.gov)\b", re.IGNORECASE,
)
_UNAPPROVED_SOURCE_RE = re.compile(
    r"\b(?:google\s*scholar|researchgate|sci[\s-]*hub)\b", re.IGNORECASE,
)
_PHI_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHI_PHONE_RE = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=[^0-9A-Za-z]))"
    r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b",
)
_PHI_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHI_MRN_RE = re.compile(r"\bMRN\s*[:#]?\s*\d+\b", re.IGNORECASE)
_PHI_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

_ANSWER_TEXT_INDICATORS: tuple[str, ...] = (
    " recommend", " is recommended", " should be given",
    " administer", " contraindicated",
    " in conclusion", " treatment of choice",
    " preferred regimen",
)


def _iter_strings(obj: Any):
    """Yield every string anywhere inside a nested dict / list."""
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
        return
    if isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def _generic_only_required_anchor(
    required_anchors: List[str],
    *,
    generic_denylist: frozenset[str],
) -> bool:
    """Cross-check the required anchor list against the K4A.2.1 closed
    generic-anchor denylist (passed in, never duplicated). The
    candidate is rejected if all required anchors are generic — same
    semantics the query compiler enforces pre-dispatch.
    """
    cleaned = [(a or "").strip().lower() for a in (required_anchors or [])]
    cleaned = [a for a in cleaned if a]
    if not cleaned:
        return True
    for a in cleaned:
        if a in generic_denylist:
            continue
        # Multi-word and not exactly in denylist → not purely generic.
        return False
    # All entries were in the denylist.
    return True


def _load_generic_denylist() -> frozenset[str]:
    """Reuses the K4A.2.1 generic-anchor denylist from
    ``primary_clinical_anchor.GENERIC_MODIFIERS`` so this validator
    can never disagree with the upstream guard."""
    try:
        from services.primary_clinical_anchor import GENERIC_MODIFIERS
        return frozenset(s.strip().lower() for s in GENERIC_MODIFIERS if s)
    except Exception:  # noqa: BLE001
        return frozenset()


# ---------------------------------------------------------------------
# Phase M2.1 — deterministic Pydantic ValidationError → closed-set
# rejection-reason translator. Pure function, no LLM. Mirrors Pydantic
# v2's ``error['type']`` taxonomy.
# ---------------------------------------------------------------------

_PYDANTIC_TYPE_TO_CLOSED_SET: dict[str, str] = {
    "missing": "missing_required_field",
    "extra_forbidden": "extra_field",
    "literal_error": "invalid_enum",
    "enum": "invalid_enum",
    "string_type": "invalid_slot_type",
    "list_type": "invalid_slot_type",
    "dict_type": "invalid_slot_type",
    "model_type": "invalid_branch_shape",
    "float_parsing": "invalid_confidence",
    "int_parsing": "invalid_confidence",
    "float_type": "invalid_confidence",
    "greater_than_equal": "invalid_confidence",
    "less_than_equal": "invalid_confidence",
    "json_invalid": "malformed_json",
}


def _pydantic_errors_to_closed_set(ve: ValidationError) -> list[str]:
    """Translate Pydantic v2 ValidationError errors into the
    M-phase closed-set rejection-reason vocabulary. Returns the list
    of distinct reasons, insertion-order preserved. The list is NEVER
    empty when a ValidationError is passed in — at minimum
    ``["schema_invalid"]`` is returned so callers can rely on at least
    one entry.
    """
    seen: set[str] = set()
    out: list[str] = []
    try:
        errs = ve.errors()
    except Exception:  # noqa: BLE001
        return ["schema_invalid"]
    for err in errs or []:
        etype = (err.get("type") or "").strip()
        loc = err.get("loc") or ()
        loc_str = ".".join(str(x) for x in loc)
        # Provenance Literal mismatch — special-case because it's not
        # just a generic "invalid_enum"; it's the bounded-shadow tag.
        if etype == "literal_error" and "provenance" in loc_str:
            mapped = "provenance_mismatch"
        else:
            mapped = _PYDANTIC_TYPE_TO_CLOSED_SET.get(etype, "schema_invalid")
        # Branch sub-model failures map to invalid_branch_shape; slot
        # sub-model failures map to invalid_slot_type when they hit
        # the inner RepairSlots model.
        if mapped == "invalid_slot_type" and loc_str.startswith("branches"):
            mapped = "invalid_branch_shape"
        if mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    if not out:
        out.append("schema_invalid")
    return out


# ---------------------------------------------------------------------
# Deterministic validator. Returns ``(candidate, [])`` on success or
# ``(None, [rejection_reason, ...])`` on failure. Each entry is a
# member of ``SEMANTIC_REPAIR_REJECTION_REASONS``.
#
# Phase M2.1 changes:
#   * Closed-set canonicalization (``canonicalize_candidate``) runs
#     BEFORE Pydantic so common LLM phrasings normalize to approved
#     enums.
#   * Pydantic ValidationError translates to specific closed-set
#     reasons (``missing_required_field``, ``extra_field``,
#     ``invalid_enum``, ``invalid_slot_type``, ``invalid_branch_shape``,
#     ``invalid_confidence``, ``provenance_mismatch``).
#   * ``{"refused": true, "refusal_reason": "..."}`` short-circuits
#     with ``["refused"]``.
#   * Multiple rejection reasons can be emitted (Pydantic surfaces
#     several at once); insertion-order preserved.
# ---------------------------------------------------------------------
def validate_repair_candidate(
    raw: Optional[dict],
    *,
    deterministic_primary_anchors: List[dict],
) -> tuple[Optional[SemanticRepairCandidate], list[str]]:
    if not raw:
        return None, ["empty_output"]
    if not isinstance(raw, dict):
        return None, ["malformed_json"]

    # 0. Honor an explicit LLM refusal envelope.
    if raw.get("refused") is True:
        return None, ["refused"]

    # 1. Closed-set canonicalization BEFORE strict validation.
    canon = canonicalize_candidate(raw)
    if not isinstance(canon, dict):
        return None, ["malformed_json"]

    # 2. Schema-locked pydantic load.
    try:
        cand = SemanticRepairCandidate.model_validate(canon)
    except ValidationError as ve:
        translated = _pydantic_errors_to_closed_set(ve)
        # Keep the legacy ``extra_fields_present`` alias too when any
        # error is an extra-field — back-compat for older callers.
        return None, translated
    except Exception:  # noqa: BLE001
        return None, ["schema_invalid"]

    # 3. Provenance MUST be the bounded shadow tag. (Pydantic Literal
    # also catches this; we keep the explicit check for clarity and
    # because canonicalization is intentionally NOT applied to
    # provenance.)
    if cand.provenance != "bounded_semantic_repair_shadow":
        return None, ["provenance_mismatch"]

    # 4. Enum validations against the M-phase closed-sets. (Cheap,
    # specific. We emit ``invalid_enum`` for each, but stop at the
    # first to keep diagnostics small.)
    if cand.primary_intent not in ALLOWED_PRIMARY_INTENTS:
        return None, ["invalid_enum"]
    for fw in cand.frameworks:
        if fw not in ALLOWED_FRAMEWORKS:
            return None, ["invalid_enum"]
    for si in cand.secondary_intents:
        if si not in ALLOWED_PRIMARY_INTENTS:
            return None, ["invalid_enum"]
    for pol in (cand.slot_policies or {}).values():
        if pol not in ALLOWED_SLOT_POLICIES:
            return None, ["invalid_slot_policy"]
    if cand.semantic_repair_reason not in SEMANTIC_REPAIR_TRIGGER_REASONS:
        return None, ["invalid_enum"]

    # 5. Confidence floor.
    if cand.confidence < CONFIDENCE_THRESHOLD:
        return None, ["confidence_below_threshold"]

    # 6. Branch count cap.
    if len(cand.branches) > BRANCH_COUNT_CAP:
        return None, ["branch_count_exceeded"]

    # 7. Required anchors must exist and must NOT be purely generic.
    if not cand.required_anchors:
        return None, ["empty_required_anchors"]
    denylist = _load_generic_denylist()
    if _generic_only_required_anchor(
        cand.required_anchors, generic_denylist=denylist,
    ):
        return None, ["generic_only_required_anchor"]

    # 8. Unvalidated MeSH must NOT be promoted. The schema labels these
    # fields with the ``_unvalidated`` suffix, and the schema does NOT
    # carry a ``mesh_terms`` validated field. If any candidate string
    # claims to be validated MeSH (e.g. ``"MeSH:Pneumonia"``), reject.
    for s in _iter_strings(canon):
        if isinstance(s, str) and s.lower().startswith("mesh:") and not s.lower().endswith("_unvalidated"):
            return None, ["unvalidated_mesh_marked_validated"]

    # 9. Defense-in-depth scrubbing — PHI / article refs / unapproved
    # sources / answer-text. Mirrors the H1 resolver's defenders.
    for s in _iter_strings(cand.model_dump()):
        if not isinstance(s, str):
            continue
        if _PMID_RE.search(s) or _DOI_RE.search(s) or _ARTICLE_REF_RE.search(s):
            return None, ["article_reference_detected"]
        if _BARE_LONG_DIGITS_RE.search(s):
            return None, ["article_reference_detected"]
        if _UNAPPROVED_SOURCE_RE.search(s):
            return None, ["unapproved_source_referenced"]
        if (
            _PHI_EMAIL_RE.search(s)
            or _PHI_PHONE_RE.search(s)
            or _PHI_SSN_RE.search(s)
            or _PHI_MRN_RE.search(s)
            or _PHI_DATE_RE.search(s)
        ):
            return None, ["phi_pattern_detected"]
        s_low = " " + s.lower() + " "
        if any(ind in s_low for ind in _ANSWER_TEXT_INDICATORS):
            return None, ["unsafe_text_detected"]

    # 10. Would NOT remove a central deterministic anchor. If the
    # deterministic pipeline already identified a non-generic primary
    # anchor (e.g. ``aspiration pneumonia``, ``pacemaker``,
    # ``rituximab``), the candidate's required_anchors MUST keep that
    # concept present. This blocks repairs that pull the conversation
    # off-topic.
    central_terms: list[str] = []
    for a in (deterministic_primary_anchors or []):
        if not isinstance(a, dict):
            continue
        concept = (
            a.get("canonical_concept") or a.get("text") or ""
        ).strip().lower()
        if not concept:
            continue
        # Skip clearly generic anchors (denylist).
        if concept in denylist:
            continue
        # Only count single-word or short multi-word disease/drug
        # tokens as "central". Long phrases were the suspicious-anchor
        # signal in the first place.
        if 1 <= len(concept.split()) <= 4:
            central_terms.append(concept)
    if central_terms:
        cand_terms_low = {
            (t or "").strip().lower()
            for t in (
                cand.required_anchors
                + cand.optional_or_ranking_only_concepts
                + cand.inferred_concepts
            )
        }
        cand_terms_low.discard("")
        for ct in central_terms:
            # Substring match in either direction — covers
            # ``"aspiration pneumonia"`` vs ``"pneumonia"`` repairs.
            if not any(
                ct in t or t in ct
                for t in cand_terms_low
            ):
                return None, ["would_remove_central_anchor"]

    return cand, []


# ---------------------------------------------------------------------
# System prompt — module-level constant so tests can inspect / hash it.
# Phase M2.1 rewrite — explicit "NOT" contract, refusal envelope, and
# 5 anchored few-shot examples (A/B/E/F/G canonical reproducers).
# ---------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are the Bounded Semantic Anchor Repair interpreter for a clinician-facing evidence retrieval product.

Hard contract — read carefully before producing output:
- You are NOT answering the clinical question.
- You are NOT searching literature.
- You are NOT selecting articles.
- You are NOT ranking evidence.
- You are NOT generating citations, PMIDs, DOIs, NCT IDs, or article references.
- You are NOT validating MeSH terms — the schema only carries ``candidate_mesh_terms_unvalidated``.
- You are NOT recommending treatment or producing clinical advice.
- You MUST NOT reference Google Scholar, ResearchGate, or Sci-Hub.
- You are ONLY producing a structured semantic-repair candidate JSON object for deterministic validation.

Output discipline:
- Return STRICT JSON only. No prose. No markdown code fences. No commentary outside JSON.
- Use ONLY values from the closed-set enums listed below. Do not invent new enum values.
- All listed fields are required. ``provenance`` MUST equal the literal string ``bounded_semantic_repair_shadow``.

Refusal envelope:
- If you cannot produce a valid candidate (ambiguous question, off-topic, unsafe), return exactly:
  ``{"refused": true, "refusal_reason": "<closed-set-reason>"}``
  and nothing else. Closed-set refusal reasons: ``ambiguous_question``, ``off_topic``, ``unsafe_request``, ``insufficient_information``.

Closed-set enums:
  primary_intent ∈ {background_overview, disease_overview, drug_overview, definition,
                    therapy, therapy_safety, drug_safety,
                    diagnostic_accuracy, diagnosis, rule_out_test_utility, screening,
                    risk_etiology, risk_etiology_harm, etiology,
                    complication_risk, complication_after_procedure,
                    epidemiology_prevalence_incidence,
                    prognosis, prognosis_natural_history, prognostic_factor,
                    mechanism, pathophysiology,
                    severity_marker_overview, other}
  frameworks elements ∈ {PICO, PICOT, PICOTT, CoCoPop, PIRD, PFO,
                         risk_etiology, mechanism, prognosis, complication,
                         diagnostic_accuracy, narrative_overview}
  slot_policies values ∈ {required_for_retrieval, optional_for_retrieval, display_only, ranking_only}
  semantic_repair_reason ∈ {literal_full_phrase_anchor, missing_risk_etiology_framework,
                            missing_mechanism_pathophysiology_framework,
                            missing_complication_after_procedure_split,
                            missing_prognosis_framework,
                            missing_epidemiology_framework_grammar,
                            missing_severity_marker_split,
                            missing_diagnostic_ruleout_split,
                            disease_population_fused_into_single_anchor}

Schema fields (all required; use defaults shown when not applicable):
  primary_intent: closed-set enum.
  secondary_intents: list of closed-set enums (default []).
  frameworks: list of closed-set framework names (max 4, default []).
  branches: list of {branch_label: string, branch_intent: string, anchors: list[string]} blocks. Max 6. Default [].
  slots: object with keys {population, condition, intervention_or_exposure, comparator, outcome, index_test, target_condition, procedure_or_device, timeframe, setting, mechanism}; each value is a list of strings.
  slot_policies: object whose keys are slot names and values are closed-set policy enums (default {}).
  required_anchors: list of canonical disease/drug/test/procedure/device concepts that MUST be retrievable. Multi-word concepts allowed. NEVER use purely generic tokens such as ``patients``, ``adults``, ``illness``, ``severity``, ``marker``, ``causes``, ``risk factor``, ``dose``, ``treatment``, ``infection``, ``thrombosis`` alone — pair these outcomes with a disease/drug/procedure anchor.
  optional_or_ranking_only_concepts: list of additional concepts for boost / ranking only (default []).
  inferred_concepts: list of concepts you derived that are NOT literally in the question (default []).
  candidate_keywords_unvalidated: list of candidate search terms; explicitly NOT validated synonyms.
  candidate_mesh_terms_unvalidated: list of candidate MeSH-like terms; explicitly NOT validated by the curated resolver. Never prefix these with ``MeSH:`` — that prefix is reserved for the deterministic resolver.
  semantic_repair_reason: copy the SINGLE trigger reason supplied in the user payload.
  confidence: float 0.0..1.0. Use ≥0.7 only when you can cleanly separate the clinical concepts.
  provenance: MUST be the literal string ``bounded_semantic_repair_shadow``.

Few-shot examples (these are the canonical Phase M shapes — match their structure):

Example 1 — risk/etiology in a population:
  Input:
    normalized_question: What are the causes of schizophrenia in young adults?
    semantic_repair_reason: missing_risk_etiology_framework
  Output:
    {"primary_intent":"risk_etiology","secondary_intents":[],"frameworks":["risk_etiology"],"branches":[],"slots":{"population":["young adults"],"condition":["schizophrenia"],"intervention_or_exposure":[],"comparator":[],"outcome":[],"index_test":[],"target_condition":[],"procedure_or_device":[],"timeframe":[],"setting":[],"mechanism":[]},"slot_policies":{"condition":"required_for_retrieval"},"required_anchors":["schizophrenia"],"optional_or_ranking_only_concepts":["young adults"],"inferred_concepts":[],"candidate_keywords_unvalidated":["etiology","risk factors","causes"],"candidate_mesh_terms_unvalidated":["Schizophrenia"],"semantic_repair_reason":"missing_risk_etiology_framework","confidence":0.85,"provenance":"bounded_semantic_repair_shadow"}

Example 2 — risk factors with exposure population:
  Input:
    normalized_question: What are risk factors for pancreatic cancer in smokers?
    semantic_repair_reason: missing_risk_etiology_framework
  Output:
    {"primary_intent":"risk_etiology","secondary_intents":[],"frameworks":["risk_etiology"],"branches":[],"slots":{"population":["smokers"],"condition":["pancreatic cancer"],"intervention_or_exposure":["smoking"],"comparator":[],"outcome":["pancreatic cancer"],"index_test":[],"target_condition":[],"procedure_or_device":[],"timeframe":[],"setting":[],"mechanism":[]},"slot_policies":{"condition":"required_for_retrieval"},"required_anchors":["pancreatic cancer"],"optional_or_ranking_only_concepts":["smoking","smokers"],"inferred_concepts":[],"candidate_keywords_unvalidated":["risk factors","tobacco","smoking"],"candidate_mesh_terms_unvalidated":["Pancreatic Neoplasms","Smoking"],"semantic_repair_reason":"missing_risk_etiology_framework","confidence":0.85,"provenance":"bounded_semantic_repair_shadow"}

Example 3 — mechanism / pathophysiology:
  Input:
    normalized_question: What is the mechanism of thrombosis in antiphospholipid syndrome?
    semantic_repair_reason: missing_mechanism_pathophysiology_framework
  Output:
    {"primary_intent":"mechanism","secondary_intents":["pathophysiology"],"frameworks":["mechanism"],"branches":[],"slots":{"population":[],"condition":["antiphospholipid syndrome"],"intervention_or_exposure":[],"comparator":[],"outcome":["thrombosis"],"index_test":[],"target_condition":[],"procedure_or_device":[],"timeframe":[],"setting":[],"mechanism":["thrombosis"]},"slot_policies":{"condition":"required_for_retrieval"},"required_anchors":["antiphospholipid syndrome"],"optional_or_ranking_only_concepts":["thrombosis","hypercoagulability"],"inferred_concepts":[],"candidate_keywords_unvalidated":["pathophysiology","mechanism","hypercoagulability"],"candidate_mesh_terms_unvalidated":["Antiphospholipid Syndrome","Thrombosis"],"semantic_repair_reason":"missing_mechanism_pathophysiology_framework","confidence":0.85,"provenance":"bounded_semantic_repair_shadow"}

Example 4 — complication after procedure:
  Input:
    normalized_question: What is the risk of infection after pacemaker implantation?
    semantic_repair_reason: missing_complication_after_procedure_split
  Output:
    {"primary_intent":"complication_after_procedure","secondary_intents":["complication_risk"],"frameworks":["complication"],"branches":[],"slots":{"population":[],"condition":[],"intervention_or_exposure":[],"comparator":[],"outcome":["infection"],"index_test":[],"target_condition":[],"procedure_or_device":["pacemaker","pacemaker implantation"],"timeframe":["post-implantation"],"setting":[],"mechanism":[]},"slot_policies":{"procedure_or_device":"required_for_retrieval"},"required_anchors":["pacemaker"],"optional_or_ranking_only_concepts":["pacemaker implantation","infection"],"inferred_concepts":[],"candidate_keywords_unvalidated":["device infection","CIED infection","post-implantation infection"],"candidate_mesh_terms_unvalidated":["Pacemaker, Artificial","Prosthesis-Related Infections"],"semantic_repair_reason":"missing_complication_after_procedure_split","confidence":0.85,"provenance":"bounded_semantic_repair_shadow"}

Example 5 — diagnostic rule-out (test utility):
  Input:
    normalized_question: Does a negative D-dimer rule out pulmonary embolism?
    semantic_repair_reason: missing_diagnostic_ruleout_split
  Output:
    {"primary_intent":"rule_out_test_utility","secondary_intents":["diagnostic_accuracy"],"frameworks":["PIRD","diagnostic_accuracy"],"branches":[],"slots":{"population":[],"condition":[],"intervention_or_exposure":[],"comparator":[],"outcome":[],"index_test":["D-dimer"],"target_condition":["pulmonary embolism"],"procedure_or_device":[],"timeframe":[],"setting":[],"mechanism":[]},"slot_policies":{"index_test":"required_for_retrieval","target_condition":"required_for_retrieval"},"required_anchors":["D-dimer","pulmonary embolism"],"optional_or_ranking_only_concepts":["negative predictive value","sensitivity","specificity"],"inferred_concepts":[],"candidate_keywords_unvalidated":["rule out","NPV","negative predictive value"],"candidate_mesh_terms_unvalidated":["Fibrin Fibrinogen Degradation Products","Pulmonary Embolism"],"semantic_repair_reason":"missing_diagnostic_ruleout_split","confidence":0.85,"provenance":"bounded_semantic_repair_shadow"}

Final reminder: output is JSON only. No code fences. No prose. Use one of the closed-set ``semantic_repair_reason`` values supplied in the user payload.
"""


# ---------------------------------------------------------------------
# Public top-level entrypoint — async, shadow-only.
# ---------------------------------------------------------------------
def _build_user_payload(
    *, normalized_question: str, trigger_reasons: List[str],
) -> str:
    fr = (trigger_reasons[0] if trigger_reasons else "literal_full_phrase_anchor").strip()
    # Echo back only the SINGLE primary trigger reason. Multiple
    # triggers can fire deterministically, but the LLM only needs one
    # to pick its repair lane. NO question echo back beyond the
    # ``normalized_question`` — and NO deterministic anchors echoed
    # back; the LLM must work from the question text alone.
    return (
        f"normalized_question: {normalized_question}\n"
        f"semantic_repair_reason: {fr}\n"
        "\n"
        "Emit a single JSON object matching the schema above. Return JSON only."
    )


async def run_semantic_anchor_repair_shadow(
    *,
    normalized_question: str,
    trigger_reasons: List[str],
    deterministic_primary_anchors: List[dict],
    diagnostics: Optional[SessionDiagnostics] = None,
) -> tuple[Optional[SemanticRepairCandidate], list[str], int]:
    """Run the bounded LLM and validate. Returns
    ``(candidate, rejection_reasons, latency_ms)``. ``rejection_reasons``
    is an empty list on success and a closed-set list otherwise.

    Never raises. When the LLM is not configured, returns
    ``(None, ["empty_output"], 0)`` quickly.
    """
    if not trigger_reasons:
        return None, [], 0
    user_payload = _build_user_payload(
        normalized_question=normalized_question or "",
        trigger_reasons=trigger_reasons,
    )
    t0 = time.perf_counter()
    raw: Optional[dict] = None
    # NOTE: ``diag.stage`` is intentionally NOT used here — its
    # ``STAGE_NAMES`` allowlist is closed-set and adding a new stage
    # name would require touching the diagnostics module. The
    # ``latency_ms`` below is the authoritative timing surface for
    # the M-phase shadow path.
    try:
        raw = await llm_provider.complete_json(
            system=SYSTEM_PROMPT,
            user=user_payload,
            temperature=0.0,
            max_tokens=1500,
        )
    except Exception:  # noqa: BLE001
        logger.exception("semantic_anchor_repair llm call failed (non-fatal)")
        raw = None
    latency_ms = int(round((time.perf_counter() - t0) * 1000))
    if raw is None:
        return None, ["empty_output"], latency_ms
    candidate, rejections = validate_repair_candidate(
        raw,
        deterministic_primary_anchors=deterministic_primary_anchors,
    )
    # Closed-set rejection-reason logging (no raw LLM text). Helps
    # admins see the failure distribution for M2.1 reliability work.
    if rejections:
        try:
            logger.warning(
                "semantic_anchor_repair rejected reasons=%s",
                ",".join(sorted(set(rejections))),
            )
        except Exception:  # noqa: BLE001
            pass
    return candidate, rejections, latency_ms


# ---------------------------------------------------------------------
# Closed-set diff between current deterministic state and the candidate.
# ---------------------------------------------------------------------
def diff_repair_candidate(
    *,
    candidate: Optional[SemanticRepairCandidate],
    current_intent: Optional[str],
    current_frameworks: List[str],
    current_required_anchors: List[str],
    current_branch_count: int,
) -> dict:
    """Return a closed-set dict describing which top-level interpretation
    fields differ between the deterministic interpretation and the
    candidate. ALL booleans, never echoes free-text differences.
    """
    if candidate is None:
        return {
            "would_change_intent": False,
            "would_change_framework": False,
            "would_change_required_anchors": False,
            "would_change_branch_plan": False,
        }
    cur_intent_low = (current_intent or "").strip().lower()
    cur_frameworks_low = {(s or "").strip().lower() for s in (current_frameworks or [])}
    cur_required_low = {(s or "").strip().lower() for s in (current_required_anchors or [])}
    cand_required_low = {(s or "").strip().lower() for s in (candidate.required_anchors or [])}
    cand_frameworks_low = {(s or "").strip().lower() for s in (candidate.frameworks or [])}
    return {
        "would_change_intent": cand_intent_changed(
            cur_intent_low, candidate.primary_intent,
        ),
        "would_change_framework": cand_frameworks_low != cur_frameworks_low,
        "would_change_required_anchors": cand_required_low != cur_required_low,
        "would_change_branch_plan": (
            len(candidate.branches) != int(current_branch_count or 0)
        ),
    }


def cand_intent_changed(cur_intent_low: str, cand_intent: str) -> bool:
    cand = (cand_intent or "").strip().lower()
    if not cur_intent_low and not cand:
        return False
    if cur_intent_low == cand:
        return False
    # Treat closely-related synonyms as no-change.
    EQ_GROUPS = (
        {"risk_etiology", "risk_etiology_harm", "etiology"},
        {"prognosis", "prognosis_natural_history", "prognostic_factor"},
        {"mechanism", "pathophysiology"},
        {"epidemiology_prevalence_incidence", "complication_risk"},
        {"diagnostic_accuracy", "diagnosis", "rule_out_test_utility"},
        {"therapy", "therapy_safety", "drug_safety"},
        {"background_overview", "disease_overview", "drug_overview", "definition"},
    )
    for grp in EQ_GROUPS:
        if cur_intent_low in grp and cand in grp:
            return False
    return True


__all__ = [
    "SEMANTIC_REPAIR_TRIGGER_REASONS",
    "SEMANTIC_REPAIR_REJECTION_REASONS",
    "ALLOWED_PRIMARY_INTENTS",
    "ALLOWED_FRAMEWORKS",
    "ALLOWED_SLOT_POLICIES",
    "CONFIDENCE_THRESHOLD",
    "BRANCH_COUNT_CAP",
    "SemanticRepairCandidate",
    "RepairBranch",
    "RepairSlots",
    "SYSTEM_PROMPT",
    "canonicalize_candidate",
    "detect_repair_triggers",
    "validate_repair_candidate",
    "run_semantic_anchor_repair_shadow",
    "diff_repair_candidate",
]
