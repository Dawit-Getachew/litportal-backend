"""Phase K2 — Shadow Branch-Aware Boolean Query Planner.

Consumes a ``ClinicalQuestionMap`` (Phase K1) and emits a closed-set
``BranchQueryPlan`` per branch. SHADOW / DIAGNOSTICS ONLY in K2.

Hard rules:
  * NEVER runs in ``active`` mode. The ``plan_branch_queries(..., mode=...)``
    entry-point assert-raises if ``mode == "active"``.
  * NEVER calls an LLM. NEVER touches the network. Pure / sync / no I/O.
  * NEVER returns raw prompts, completions, PHI, evidence text, emails,
    credentials, or secrets.
  * NEVER trusts resolver-supplied candidate MeSH descriptors until they
    pass the deterministic ``services.mesh.mesh_resolver.resolve``
    validator. Untrusted candidates land in
    ``unvalidated_MeSH_terms_rejected_or_pending`` with a closed-set
    reason — they NEVER enter the compiled query.
  * Source-specific query strings are validated with the existing
    ``services.query_compiler.validate_query_syntax`` (balanced parens,
    valid field tags). Failures are recorded — they NEVER ship.

The planner is purely additive — the existing
``BooleanQueryCompilerService`` remains the FINAL query generator for
active retrieval. K2 is shadow-only and persisted under
``clinical_question_map_json.branch_query_plan`` (additive field;
``ClinicalQuestionMap`` model permits it via the closed-set GET
projection only — the in-Python ``ClinicalQuestionMap`` schema has
``extra="forbid"`` and is NOT mutated; the plan is persisted alongside
the CQM JSON on the session document and surfaced under the diagnostics
projection).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from services.clinical_question_map import (
    DEFERRED_INTENTS,
    is_generic_modifier_token,
    ClinicalQuestionMap,
)
from services.mesh import mesh_resolver
from services.query_compiler import validate_query_syntax

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Closed-set enums.
# ---------------------------------------------------------------------

PlannerMode = Literal["disabled", "shadow", "active"]

QueryFamilyName = Literal[
    "high_recall",
    "high_precision",
    "secondary_evidence",
    "guideline_or_consensus",
    "recent",
    "rescue",
    "case_series_or_pharmacovigilance",
    "pathophysiology_translational",
    "prediction_validation",
]

QuerySource = Literal["pubmed", "europe_pmc", "openalex"]

#: Closed-set MeSH-rejection reasons. NEVER widen without updating tests.
UNVALIDATED_MESH_REASONS: frozenset[str] = frozenset({
    "not_in_mesh_table",       # mesh_resolver returned None
    "keyword_only_concept",     # mesh_resolver marked concept keyword-only
    "below_confidence_floor",   # resolver-supplied but below threshold
    "generic_modifier_blocked", # candidate equals a generic modifier
    "phi_or_secret_pattern",    # defensive — should never happen
    "empty_or_invalid",         # blank / non-string candidate
})

#: Intent-specific query-family routing table (closed-set). Each intent
#: maps to the families the planner emits. Defensive: any intent that
#: falls outside this table emits only ``high_recall`` + ``high_precision``.
_INTENT_FAMILY_TABLE: dict[str, tuple[QueryFamilyName, ...]] = {
    "therapy": (
        "high_recall", "high_precision",
        "secondary_evidence", "guideline_or_consensus",
    ),
    "therapy_safety": (
        "high_recall", "high_precision",
        "secondary_evidence", "case_series_or_pharmacovigilance",
    ),
    "drug_safety": (
        "high_recall", "high_precision",
        "case_series_or_pharmacovigilance",
    ),
    "diagnosis_accuracy": (
        "high_recall", "high_precision", "secondary_evidence",
    ),
    "diagnostic_workup": (
        "high_recall", "guideline_or_consensus", "secondary_evidence",
    ),
    "differential_diagnosis": (
        "high_recall", "secondary_evidence", "guideline_or_consensus",
    ),
    "prognosis_natural_history": (
        "high_recall", "high_precision", "secondary_evidence",
    ),
    "prognostic_factor": (
        "high_recall", "high_precision", "secondary_evidence",
    ),
    "prediction_model": (
        "high_recall", "high_precision",
        "prediction_validation", "secondary_evidence",
    ),
    "risk_etiology_harm": (
        "high_recall", "high_precision", "secondary_evidence",
    ),
    "complication_risk": (
        "high_recall", "high_precision",
        "case_series_or_pharmacovigilance", "secondary_evidence", "rescue",
    ),
    "disease_complication": (
        "high_recall", "high_precision", "secondary_evidence",
    ),
    "drug_complication": (
        "high_recall", "high_precision",
        "case_series_or_pharmacovigilance", "secondary_evidence",
    ),
    "procedure_complication": (
        "high_recall", "high_precision",
        "case_series_or_pharmacovigilance", "secondary_evidence",
    ),
    "device_complication": (
        "high_recall", "high_precision",
        "case_series_or_pharmacovigilance", "secondary_evidence", "rescue",
    ),
    "epidemiology_prevalence_incidence": (
        "high_recall", "high_precision", "secondary_evidence",
    ),
    "pathophysiology_mechanism": (
        "high_recall", "pathophysiology_translational", "secondary_evidence",
    ),
    "monitoring_followup_surveillance": (
        "high_recall", "guideline_or_consensus", "secondary_evidence",
    ),
    "stabilization_emergency_management": (
        "high_recall", "guideline_or_consensus", "secondary_evidence",
    ),
    "guideline_recommendation": (
        "guideline_or_consensus", "high_recall", "secondary_evidence",
    ),
    "background_overview": (
        "high_recall", "secondary_evidence",
    ),
    "prevention_screening": (
        "high_recall", "guideline_or_consensus", "secondary_evidence",
    ),
}


# ---------------------------------------------------------------------
# Intent-specific method/anchor term packs (closed-set, no LLM).
# Each entry is (canonical_method_concept, [synonyms]).
# ---------------------------------------------------------------------

_INTENT_METHOD_PACKS: dict[str, list[tuple[str, list[str]]]] = {
    "epidemiology_prevalence_incidence": [
        ("prevalence", ["prevalence", "incidence", "epidemiology",
                        "population-based", "cross-sectional"]),
    ],
    "prognosis_natural_history": [
        ("prognosis", ["prognosis", "survival", "natural history",
                       "follow-up", "long-term outcome"]),
    ],
    "prognostic_factor": [
        ("prognostic factor", ["prognostic factor", "predictor",
                               "prognostic marker", "outcome predictor"]),
    ],
    "prediction_model": [
        ("prediction model", ["prediction model", "risk score",
                              "clinical score", "validation",
                              "calibration", "discrimination",
                              "c-statistic", "AUC"]),
    ],
    "diagnosis_accuracy": [
        ("diagnostic accuracy", ["sensitivity", "specificity",
                                 "diagnostic accuracy",
                                 "likelihood ratio", "ROC curve",
                                 "AUC"]),
    ],
    "diagnostic_workup": [
        ("workup", ["evaluation", "workup", "diagnostic approach",
                    "algorithm", "differential"]),
    ],
    "differential_diagnosis": [
        ("differential diagnosis", ["differential diagnosis", "causes",
                                    "mimics", "red flags"]),
    ],
    "pathophysiology_mechanism": [
        ("pathophysiology", ["pathophysiology", "mechanism",
                             "pathogenesis", "physiopathology",
                             "molecular basis"]),
    ],
    "risk_etiology_harm": [
        ("risk factor", ["risk factor", "exposure", "etiology",
                         "harm", "association"]),
    ],
    "complication_risk": [
        ("complication", ["complication", "adverse event",
                          "adverse effect"]),
    ],
    "device_complication": [
        ("device complication", ["complication", "adverse event",
                                 "device-related", "device failure"]),
    ],
    "procedure_complication": [
        ("procedure complication", ["complication", "adverse event",
                                    "post-procedure", "postoperative"]),
    ],
    "drug_complication": [
        ("drug adverse event", ["adverse event", "adverse drug reaction",
                                "adverse drug event", "toxicity"]),
    ],
    "monitoring_followup_surveillance": [
        ("monitoring", ["monitoring", "follow-up", "surveillance",
                        "screening interval"]),
    ],
    "stabilization_emergency_management": [
        ("stabilization", ["acute management", "emergency management",
                           "stabilization", "code"]),
    ],
    "guideline_recommendation": [
        ("guideline", ["guideline", "consensus statement",
                       "practice recommendation"]),
    ],
    "prevention_screening": [
        ("prevention", ["prevention", "screening", "prophylaxis"]),
    ],
    "therapy_safety": [
        ("safety", ["safety", "tolerability", "adverse event"]),
    ],
    "drug_safety": [
        ("drug safety", ["safety", "adverse drug reaction",
                         "post-marketing surveillance"]),
    ],
    "background_overview": [],  # no method pack
    "therapy": [],              # method pack not needed — intervention is the anchor
}


# Closed-set intent-prefix strip patterns. Match a leading intent prefix
# anchored at the start of the concept string (case-insensitive). The
# replacement is the bare clinical noun.
_INTENT_PREFIX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*prevalence\s+of\s+", re.IGNORECASE), "prevalence of"),
    (re.compile(r"^\s*incidence\s+of\s+", re.IGNORECASE), "incidence of"),
    (re.compile(r"^\s*epidemiology\s+of\s+", re.IGNORECASE), "epidemiology of"),
    (re.compile(r"^\s*mechanism\s+of\s+", re.IGNORECASE), "mechanism of"),
    (re.compile(r"^\s*pathogenesis\s+of\s+", re.IGNORECASE), "pathogenesis of"),
    (re.compile(r"^\s*pathophysiology\s+of\s+", re.IGNORECASE), "pathophysiology of"),
    (re.compile(r"^\s*prognosis\s+of\s+", re.IGNORECASE), "prognosis of"),
    (re.compile(r"^\s*natural\s+history\s+of\s+", re.IGNORECASE), "natural history of"),
    (re.compile(r"^\s*risk\s+of\s+", re.IGNORECASE), "risk of"),
    (re.compile(r"^\s*risks?\s+and\s+benefits?\s+of\s+", re.IGNORECASE), "risks and benefits of"),
    (re.compile(r"^\s*benefits?\s+and\s+risks?\s+of\s+", re.IGNORECASE), "benefits and risks of"),
    (re.compile(r"^\s*management\s+of\s+", re.IGNORECASE), "management of"),
    (re.compile(r"^\s*treatment\s+of\s+", re.IGNORECASE), "treatment of"),
    (re.compile(r"^\s*diagnosis\s+of\s+", re.IGNORECASE), "diagnosis of"),
    (re.compile(r"^\s*workup\s+of\s+", re.IGNORECASE), "workup of"),
    (re.compile(r"^\s*evaluation\s+of\s+", re.IGNORECASE), "evaluation of"),
    (re.compile(r"^\s*complications?\s+of\s+", re.IGNORECASE), "complications of"),
    (re.compile(r"^\s*survival\s+in\s+", re.IGNORECASE), "survival in"),
]


# Closed-set "specific outcome / complication" allow-list. These are
# legitimate clinical entities — they MUST NOT be filtered by the
# generic-modifier deny-list even when the surrounding role is unusual.
_SPECIFIC_OUTCOME_ALLOWLIST: frozenset[str] = frozenset({
    "mortality", "infection", "bleeding", "stroke", "recurrence",
    "lead dislodgement", "pacemaker migration", "thrombosis",
    "myocardial infarction", "death", "embolism", "ischemia",
    "rejection",
})


# ---------------------------------------------------------------------
# Pydantic schemas.
# ---------------------------------------------------------------------


class ConceptSummary(BaseModel):
    """Closed-set summary of a concept used (or deliberately NOT used)
    in a branch query plan. NEVER contains free-text payloads beyond
    ``canonical_concept`` + ``original_phrase`` + closed-set synonyms.
    """

    model_config = ConfigDict(extra="forbid")

    canonical_concept: str
    original_phrase: str
    slot_role: str
    entity_type: str
    slot_search_policy: str
    inferred: bool
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)


class UnvalidatedMeshEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    term: str
    reason: str

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        if self.reason not in UNVALIDATED_MESH_REASONS:
            raise ValueError(
                f"unvalidated_mesh reason {self.reason!r} not in closed-set"
            )


class QueryFamily(BaseModel):
    """A single query family within a branch plan."""

    model_config = ConfigDict(extra="forbid")

    family_name: QueryFamilyName
    source: QuerySource
    query_string: str
    qa_checks_passed: bool
    notes: List[str] = Field(default_factory=list)


class QueryQualityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    passed: bool
    detail: str = ""


class BranchQueryFamilies(BaseModel):
    """All shadow query families for a single branch."""

    model_config = ConfigDict(extra="forbid")

    branch_id: str
    intent: str
    framework: str
    branch_priority: int
    branch_confidence: float = Field(ge=0.0, le=1.0)

    required_concepts_used: List[ConceptSummary] = Field(default_factory=list)
    optional_concepts_used: List[ConceptSummary] = Field(default_factory=list)
    ranking_only_concepts_not_searched: List[ConceptSummary] = Field(
        default_factory=list,
    )
    display_only_concepts_not_searched: List[ConceptSummary] = Field(
        default_factory=list,
    )

    inferred_concepts_used: List[ConceptSummary] = Field(default_factory=list)
    inferred_concepts_not_required: List[ConceptSummary] = Field(
        default_factory=list,
    )

    validated_MeSH_terms: List[str] = Field(default_factory=list)
    unvalidated_MeSH_terms_rejected_or_pending: List[UnvalidatedMeshEntry] = (
        Field(default_factory=list)
    )

    title_abstract_terms: List[str] = Field(default_factory=list)
    MeSH_subheadings_used: List[str] = Field(default_factory=list)

    query_families: List[QueryFamily] = Field(default_factory=list)
    source_specific_query_strings: dict[str, str] = Field(default_factory=dict)
    query_warnings: List[str] = Field(default_factory=list)
    query_quality_checks: List[QueryQualityCheck] = Field(default_factory=list)

    active_for_retrieval: Literal[False] = False
    mode: Literal["shadow", "disabled"] = "shadow"


class BranchQueryPlan(BaseModel):
    """Top-level shadow query plan persisted under
    ``clinical_question_map_json.branch_query_plan``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["K2.1.0"] = "K2.1.0"
    mode: Literal["disabled", "shadow"] = "shadow"
    active_for_retrieval: Literal[False] = False
    generated_at: str
    branches: List[BranchQueryFamilies] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


_INTENT_PREFIX_LABELS: set[str] = {p[1] for p in _INTENT_PREFIX_PATTERNS}


def _strip_intent_prefix(text: str) -> tuple[str, Optional[str]]:
    """Return ``(stripped_text, stripped_prefix_label or None)``."""
    if not text:
        return text, None
    for patt, label in _INTENT_PREFIX_PATTERNS:
        new = patt.sub("", text, count=1)
        if new != text and new.strip():
            return new.strip(), label
    return text, None


def _is_specific_outcome(canonical: str) -> bool:
    return (canonical or "").strip().lower() in _SPECIFIC_OUTCOME_ALLOWLIST


def _concept_summary_from_slot(
    slot: dict[str, Any],
    *,
    sources: Optional[List[str]] = None,
) -> ConceptSummary:
    return ConceptSummary(
        canonical_concept=str(slot.get("canonical_concept") or ""),
        original_phrase=str(slot.get("original_phrase") or ""),
        slot_role=str(slot.get("slot_role") or "clinical_problem"),
        entity_type=str(slot.get("entity_type") or "unknown"),
        slot_search_policy=str(
            slot.get("slot_search_policy") or "optional_for_retrieval"
        ),
        inferred=bool(slot.get("inferred")),
        confidence=float(slot.get("confidence") or 0.0)
        if isinstance(slot.get("confidence"), (int, float)) else 0.0,
        sources=list(sources or [str(slot.get("source") or "unknown")]),
    )


def _dedup_key(slot: dict[str, Any]) -> tuple[str, str]:
    """Closed-set dedup key — canonical_concept (lower-cased,
    whitespace-normalised) + role-equivalence-class. Disease vs
    clinical-problem collapse into one equivalence class so concept-
    pack + resolver-emitted entries merge cleanly.
    """
    role = str(slot.get("slot_role") or "").strip().lower()
    role_class = {
        "clinical_problem": "disease_or_condition",
        "target_condition": "disease_or_condition",
    }.get(role, role)
    canon = " ".join(str(slot.get("canonical_concept") or "").lower().split())
    return canon, role_class


def _adjudicate_slots(
    slots: list[dict[str, Any]],
    *,
    branch_intent: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Run dedup + intent-prefix-strip + role adjudication +
    generic-modifier deny-list.

    Returns a new list of slot dicts (deep-copy-safe — original slots
    are NOT mutated). Each adjudication action emits a closed-set
    ``query_warnings`` entry.
    """
    adjudicated: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    for raw in slots:
        if not isinstance(raw, dict):
            continue
        slot = dict(raw)  # shallow copy
        canonical = str(slot.get("canonical_concept") or "")
        original = str(slot.get("original_phrase") or canonical)
        role = str(slot.get("slot_role") or "")
        entity_type = str(slot.get("entity_type") or "")
        is_specific = _is_specific_outcome(canonical) or _is_specific_outcome(
            original
        )

        # K2.A.4 — generic-modifier filter. Specific outcomes pass.
        if (
            is_generic_modifier_token(canonical)
            or is_generic_modifier_token(original)
        ) and not is_specific:
            if role == "population":
                # Already handled by K1.2 normalizer, but defensive.
                warnings.append(
                    f"generic_modifier_demoted:{canonical or original}"
                )
                continue
            if slot.get("required_for_search"):
                warnings.append(
                    f"generic_modifier_demoted:{canonical or original}"
                )
                slot["required_for_search"] = False
                slot["slot_search_policy"] = "display_only"
            # Keep the demoted slot — it remains visible in the plan
            # under display_only_concepts_not_searched but never enters
            # required / optional anchor lists for retrieval.
            adjudicated.append(slot)
            continue

        # K2.A.2 — intent-prefix strip for disease / problem slots.
        if role in {
            "disease_or_condition", "clinical_problem", "target_condition",
        }:
            stripped, prefix = _strip_intent_prefix(canonical)
            if prefix is not None:
                slot["canonical_concept"] = stripped
                slot["original_phrase"] = (
                    stripped if original.strip().lower() == canonical.strip().lower()
                    else original
                )
                warnings.append(f"intent_prefix_stripped:{prefix}")
            stripped_o, prefix_o = _strip_intent_prefix(original)
            if (
                prefix_o is not None
                and slot["original_phrase"] != stripped_o
            ):
                slot["original_phrase"] = stripped_o
                warnings.append(f"intent_prefix_stripped:{prefix_o}")

        # K2.A.3 — role/entity adjudication.
        # Example 1: troponin elevation tagged outcome but
        # entity_type=biomarker_or_lab AND branch intent is prognostic_factor.
        if (
            entity_type == "biomarker_or_lab"
            and role == "outcome"
            and branch_intent in {
                "prognostic_factor", "prediction_model",
                "prognosis_natural_history",
            }
        ):
            slot["slot_role"] = "prognostic_factor"
            warnings.append(
                f"role_adjudicated:{canonical}:prognostic_factor"
            )
        # Example 2: "mortality" / "death" tagged clinical_problem →
        # treat as outcome for query planning.
        if (
            canonical.strip().lower() in {"mortality", "death"}
            and role == "clinical_problem"
        ):
            slot["slot_role"] = "outcome"
            slot["entity_type"] = "complication_or_adverse_event"
            warnings.append(
                f"role_adjudicated:{canonical}:outcome"
            )

        # K2.A.1 — dedup. Merge with existing slot when keys match.
        key = _dedup_key(slot)
        if key in seen:
            existing = seen[key]
            # Prefer the slot with the higher required_for_search +
            # higher confidence. Merge sources for transparency.
            keep = existing
            other = slot
            if bool(slot.get("required_for_search")) and not bool(
                existing.get("required_for_search")
            ):
                keep, other = slot, existing
            elif (
                bool(slot.get("required_for_search"))
                == bool(existing.get("required_for_search"))
                and float(slot.get("confidence") or 0.0)
                > float(existing.get("confidence") or 0.0)
            ):
                keep, other = slot, existing
            # Merge "source" tracking under ``_planner_sources``.
            merged_sources = list(keep.get("_planner_sources") or [
                str(keep.get("source") or "unknown")
            ])
            other_src = str(other.get("source") or "unknown")
            if other_src not in merged_sources:
                merged_sources.append(other_src)
            keep["_planner_sources"] = merged_sources
            seen[key] = keep
            warnings.append(f"slot_deduplicated:{key[0]}")
            continue
        seen[key] = slot
        adjudicated.append(slot)

    # Re-order to match insertion order (Python dict preserves
    # insertion); but we rebuilt ``adjudicated`` only for first
    # encounter — for dedup'd cases the kept version may have changed.
    # Reconstruct the final list from ``seen`` to ensure the winning
    # version is used.
    final: list[dict[str, Any]] = []
    placed: set[tuple[str, str]] = set()
    for slot in adjudicated:
        key = _dedup_key(slot)
        if key in placed:
            continue
        placed.add(key)
        final.append(seen.get(key, slot))
    return final


# ---------------------------------------------------------------------
# MeSH validation.
# ---------------------------------------------------------------------


def _validate_mesh_candidates(
    candidates: Iterable[str],
) -> tuple[list[str], list[UnvalidatedMeshEntry]]:
    validated: list[str] = []
    rejected: list[UnvalidatedMeshEntry] = []
    seen_v: set[str] = set()
    seen_r: set[str] = set()
    for raw in candidates:
        term = (raw or "").strip()
        if not term:
            if "" not in seen_r:
                seen_r.add("")
                rejected.append(UnvalidatedMeshEntry(
                    term="", reason="empty_or_invalid",
                ))
            continue
        if is_generic_modifier_token(term):
            if term.lower() not in seen_r:
                seen_r.add(term.lower())
                rejected.append(UnvalidatedMeshEntry(
                    term=term, reason="generic_modifier_blocked",
                ))
            continue
        if mesh_resolver.is_keyword_only(term):
            if term.lower() not in seen_r:
                seen_r.add(term.lower())
                rejected.append(UnvalidatedMeshEntry(
                    term=term, reason="keyword_only_concept",
                ))
            continue
        res = mesh_resolver.resolve(term)
        if res is None:
            if term.lower() not in seen_r:
                seen_r.add(term.lower())
                rejected.append(UnvalidatedMeshEntry(
                    term=term, reason="not_in_mesh_table",
                ))
            continue
        descriptor = res.mesh_descriptor
        if descriptor and descriptor.lower() not in seen_v:
            seen_v.add(descriptor.lower())
            validated.append(descriptor)
    return validated, rejected


# ---------------------------------------------------------------------
# Concept block builders (per source).
# ---------------------------------------------------------------------


def _strip_quotes(s: str) -> str:
    """Sanitise a concept fragment so it never closes an open quote
    accidentally and never injects nested parens."""
    return re.sub(r'[\"\(\)\[\]]', " ", s).strip()


def _pubmed_concept_block(
    *, ta_terms: list[str], mesh_terms: list[str],
    subheadings: list[str] | None = None,
) -> str:
    parts: list[str] = []
    for t in ta_terms:
        c = _strip_quotes(t)
        if c:
            parts.append(f'"{c}"[Title/Abstract]')
    for m in mesh_terms:
        c = _strip_quotes(m)
        if c:
            parts.append(f'"{c}"[MeSH Terms]')
    for sh in subheadings or []:
        c = _strip_quotes(sh)
        if c:
            parts.append(f'"{c}"[MeSH Subheading]')
    if not parts:
        return ""
    return "(" + " OR ".join(parts) + ")"


def _epmc_concept_block(
    *, ta_terms: list[str], mesh_terms: list[str],
) -> str:
    parts: list[str] = []
    for t in ta_terms:
        c = _strip_quotes(t)
        if c:
            parts.append(f'TITLE_ABS:"{c}"')
    for m in mesh_terms:
        c = _strip_quotes(m)
        if c:
            parts.append(f'MESH:"{c}"')
    if not parts:
        return ""
    return "(" + " OR ".join(parts) + ")"


def _openalex_concept_block(*, ta_terms: list[str]) -> str:
    """OpenAlex doesn't support MeSH — title.search OR abstract.search
    on each title-abstract term joined by OR.
    """
    parts: list[str] = []
    for t in ta_terms:
        c = _strip_quotes(t)
        if not c:
            continue
        # Treat multi-word phrases as quoted phrase searches; the
        # OpenAlex filter API accepts ``title_and_abstract.search:`` —
        # we model the query string as a closed-set fragment.
        parts.append(f'title_and_abstract.search:"{c}"')
    if not parts:
        return ""
    return "(" + " OR ".join(parts) + ")"


# ---------------------------------------------------------------------
# Branch planner.
# ---------------------------------------------------------------------


def _build_concept_text_and_mesh(
    slot: dict[str, Any],
) -> tuple[list[str], list[str], list[UnvalidatedMeshEntry]]:
    """Return ``(title_abstract_terms, validated_mesh, rejected_mesh)``.

    Title-abstract terms = canonical_concept + closed-set synonyms.
    MeSH candidates = canonical + synonyms passed through
    ``mesh_resolver.resolve``. Resolver-supplied "candidate" MeSH
    descriptors that fail validation land in
    ``unvalidated_MeSH_terms_rejected_or_pending``.
    """
    canonical = str(slot.get("canonical_concept") or "")
    original = str(slot.get("original_phrase") or canonical)
    syns = [str(s) for s in (slot.get("synonyms") or []) if isinstance(s, str)]
    candidates: list[str] = []
    for c in [canonical, original, *syns]:
        c = c.strip()
        if c and c not in candidates:
            candidates.append(c)
    validated, rejected = _validate_mesh_candidates(candidates)
    ta_terms = [c for c in candidates if c]
    return ta_terms, validated, rejected


def _emit_family_query_strings(
    *,
    family_name: QueryFamilyName,
    intent: str,
    required_blocks_by_source: dict[str, list[str]],
    optional_blocks_by_source: dict[str, list[str]],
    method_blocks_by_source: dict[str, str],
    study_type_filters_by_source: dict[str, str],
) -> dict[str, str]:
    """Combine concept blocks + method block + (optional) study-type
    filter into per-source query strings for a single family.

    Closed-set per-family policy:
      * ``high_recall`` — required + method (where applicable); NO
        study-type filter.
      * ``high_precision`` — required + optional + method +
        study-type filter (PubMed: clinical-queries-style filter).
      * ``secondary_evidence`` — required + method + SR/meta-analysis
        study-type filter.
      * ``guideline_or_consensus`` — required + ``guideline OR
        consensus`` filter.
      * ``recent`` — required + date-published filter (last 5 years).
      * ``rescue`` — required only (no method, no filter).
      * ``case_series_or_pharmacovigilance`` — required + case-series /
        pharmacovigilance filter.
      * ``pathophysiology_translational`` — required + mechanism filter.
      * ``prediction_validation`` — required + validation/calibration
        filter.
    """
    out: dict[str, str] = {}
    for source in ("pubmed", "europe_pmc", "openalex"):
        req_blocks = required_blocks_by_source.get(source, [])
        opt_blocks = optional_blocks_by_source.get(source, [])
        if not req_blocks:
            continue  # never ship an unanchored query
        # Choose which concept blocks to AND together per family.
        if family_name == "high_precision":
            blocks = [b for b in (req_blocks + opt_blocks) if b]
        else:
            blocks = list(req_blocks)
        # Method block.
        method_block = method_blocks_by_source.get(source, "")
        if family_name in {
            "high_recall", "high_precision", "secondary_evidence",
            "guideline_or_consensus", "case_series_or_pharmacovigilance",
            "pathophysiology_translational", "prediction_validation",
        } and method_block:
            blocks.append(method_block)
        # Study-type filter.
        filt = ""
        if family_name == "secondary_evidence":
            filt = _study_type_filter(
                source, "systematic_review_or_meta_analysis"
            )
        elif family_name == "guideline_or_consensus":
            filt = _study_type_filter(source, "guideline_or_consensus")
        elif family_name == "high_precision":
            filt = study_type_filters_by_source.get(source, "")
        elif family_name == "case_series_or_pharmacovigilance":
            filt = _study_type_filter(
                source, "case_series_or_pharmacovigilance"
            )
        elif family_name == "pathophysiology_translational":
            filt = _study_type_filter(
                source, "mechanism_or_review"
            )
        elif family_name == "prediction_validation":
            filt = _study_type_filter(
                source, "prediction_validation"
            )
        elif family_name == "recent":
            filt = _recent_filter(source)
        # Rescue and others — no filter.
        if filt:
            blocks.append(filt)
        # Drop empty blocks defensively.
        blocks = [b for b in blocks if b]
        if len(blocks) < 1:
            continue
        out[source] = " AND ".join(blocks)
    return out


def _study_type_filter(source: str, family: str) -> str:
    """Closed-set study-type filter fragments. Kept narrow and
    syntactically safe."""
    if source == "pubmed":
        return {
            "systematic_review_or_meta_analysis": (
                '("systematic review"[Publication Type] OR '
                '"meta-analysis"[Publication Type])'
            ),
            "guideline_or_consensus": (
                '("guideline"[Publication Type] OR '
                '"practice guideline"[Publication Type] OR '
                '"consensus development conference"[Publication Type])'
            ),
            "case_series_or_pharmacovigilance": (
                '("case reports"[Publication Type] OR '
                '"adverse drug reaction reporting systems"[MeSH Terms])'
            ),
            "mechanism_or_review": (
                '("review"[Publication Type] OR '
                '"pathology"[MeSH Subheading])'
            ),
            "prediction_validation": (
                '("validation study"[Publication Type] OR '
                '"clinical decision rules"[MeSH Terms])'
            ),
            "high_precision_default": (
                '("randomized controlled trial"[Publication Type] OR '
                '"systematic review"[Publication Type] OR '
                '"meta-analysis"[Publication Type])'
            ),
        }.get(family, "")
    if source == "europe_pmc":
        return {
            "systematic_review_or_meta_analysis": (
                '(PUB_TYPE:"systematic-review" OR PUB_TYPE:"Meta-Analysis")'
            ),
            "guideline_or_consensus": (
                '(PUB_TYPE:"Guideline" OR '
                'PUB_TYPE:"Practice Guideline" OR '
                'PUB_TYPE:"Consensus Development Conference")'
            ),
            "case_series_or_pharmacovigilance": (
                '(PUB_TYPE:"Case Reports")'
            ),
            "mechanism_or_review": (
                '(PUB_TYPE:"Review")'
            ),
            "prediction_validation": (
                '(PUB_TYPE:"Validation Study")'
            ),
            "high_precision_default": (
                '(PUB_TYPE:"Randomized Controlled Trial" OR '
                'PUB_TYPE:"systematic-review" OR PUB_TYPE:"Meta-Analysis")'
            ),
        }.get(family, "")
    # OpenAlex — no publication-type filter in this shadow scope.
    return ""


def _recent_filter(source: str) -> str:
    if source == "pubmed":
        return '("last 5 years"[PDat])'
    if source == "europe_pmc":
        return 'FIRST_PDATE:[2020-01-01 TO *]'
    return ""


def _resolve_high_precision_filter(intent: str, source: str) -> str:
    """Intent-specific high-precision filter."""
    if intent in {"therapy", "therapy_safety"}:
        return _study_type_filter(source, "high_precision_default")
    if intent in {
        "diagnosis_accuracy", "diagnostic_workup", "differential_diagnosis",
    }:
        if source == "pubmed":
            return (
                '("sensitivity and specificity"[MeSH Terms] OR '
                'sensitivity[Title/Abstract] OR specificity[Title/Abstract] '
                'OR "ROC curve"[MeSH Terms])'
            )
        if source == "europe_pmc":
            return (
                '(MESH:"Sensitivity and Specificity" OR '
                'TITLE_ABS:"sensitivity" OR TITLE_ABS:"specificity")'
            )
        return ""
    if intent in {
        "prognosis_natural_history", "prognostic_factor", "prediction_model",
    }:
        if source == "pubmed":
            return (
                '("cohort studies"[MeSH Terms] OR '
                '"follow-up studies"[MeSH Terms] OR '
                'cohort[Title/Abstract])'
            )
        if source == "europe_pmc":
            return (
                '(MESH:"Cohort Studies" OR MESH:"Follow-Up Studies" '
                'OR TITLE_ABS:"cohort")'
            )
        return ""
    if intent in {
        "risk_etiology_harm", "complication_risk", "device_complication",
        "procedure_complication", "drug_complication",
    }:
        if source == "pubmed":
            return (
                '("cohort studies"[MeSH Terms] OR '
                '"case-control studies"[MeSH Terms] OR '
                '"risk factors"[MeSH Terms])'
            )
        if source == "europe_pmc":
            return (
                '(MESH:"Cohort Studies" OR MESH:"Case-Control Studies" '
                'OR MESH:"Risk Factors")'
            )
        return ""
    if intent == "epidemiology_prevalence_incidence":
        if source == "pubmed":
            return (
                '("prevalence"[MeSH Terms] OR "incidence"[MeSH Terms] OR '
                '"epidemiologic studies"[MeSH Terms])'
            )
        if source == "europe_pmc":
            return (
                '(MESH:"Prevalence" OR MESH:"Incidence" OR '
                'MESH:"Epidemiologic Studies")'
            )
        return ""
    return ""


# ---------------------------------------------------------------------
# Method block (intent-specific) per source.
# ---------------------------------------------------------------------


def _build_method_block_by_source(intent: str) -> dict[str, str]:
    pack = _INTENT_METHOD_PACKS.get(intent) or []
    if not pack:
        return {}
    # Flatten the synonyms into a single OR block; resolve MeSH for
    # canonical method terms only.
    ta: list[str] = []
    mesh: list[str] = []
    for canon, syns in pack:
        if canon and canon not in ta:
            ta.append(canon)
        for s in syns:
            if s not in ta:
                ta.append(s)
    # MeSH for the canonical method anchor only.
    validated, _rejected = _validate_mesh_candidates(
        [p[0] for p in pack]
    )
    mesh = validated
    return {
        "pubmed": _pubmed_concept_block(ta_terms=ta, mesh_terms=mesh),
        "europe_pmc": _epmc_concept_block(ta_terms=ta, mesh_terms=mesh),
        "openalex": _openalex_concept_block(ta_terms=ta),
    }


# ---------------------------------------------------------------------
# Per-branch plan builder.
# ---------------------------------------------------------------------


def _plan_one_branch(
    branch: dict[str, Any],
    cqm: ClinicalQuestionMap | dict[str, Any],
) -> BranchQueryFamilies:
    intent = str(branch.get("intent") or "")
    framework = str(branch.get("framework") or "PICOTT")
    branch_id = str(branch.get("branch_id") or "primary_0")
    priority = int(branch.get("branch_priority") or 1)
    confidence = float(branch.get("branch_confidence") or 0.0)

    warnings: list[str] = []
    quality_checks: list[QueryQualityCheck] = []

    required_in = list(branch.get("required_concepts") or [])
    optional_in = list(branch.get("optional_concepts") or [])
    ranking_only_in = list(branch.get("ranking_only_concepts") or [])
    display_only_in = list(branch.get("display_only_concepts") or [])

    # K2.A — adjudication runs on required + optional. ranking_only and
    # display_only are surfaced unmodified (they never enter required
    # query blocks, by rule K2.B.6 / K2.B.7).
    req_adj = _adjudicate_slots(required_in, branch_intent=intent, warnings=warnings)
    opt_adj = _adjudicate_slots(optional_in, branch_intent=intent, warnings=warnings)

    # If adjudication demoted a required slot to display_only, route it
    # there explicitly.
    req_final: list[dict[str, Any]] = []
    extra_display: list[dict[str, Any]] = []
    for s in req_adj:
        if s.get("slot_search_policy") == "display_only":
            extra_display.append(s)
        else:
            req_final.append(s)

    # Build per-source concept blocks for required + optional slots.
    required_blocks_by_source: dict[str, list[str]] = {
        "pubmed": [], "europe_pmc": [], "openalex": [],
    }
    optional_blocks_by_source: dict[str, list[str]] = {
        "pubmed": [], "europe_pmc": [], "openalex": [],
    }
    all_validated_mesh: list[str] = []
    seen_validated: set[str] = set()
    all_rejected_mesh: list[UnvalidatedMeshEntry] = []
    seen_rejected: set[str] = set()
    all_ta_terms: list[str] = []
    seen_ta: set[str] = set()

    def _push_mesh(v: list[str], r: list[UnvalidatedMeshEntry]) -> None:
        for m in v:
            k = m.lower()
            if k in seen_validated:
                continue
            seen_validated.add(k)
            all_validated_mesh.append(m)
        for entry in r:
            ek = f"{entry.term.lower()}::{entry.reason}"
            if ek in seen_rejected:
                continue
            seen_rejected.add(ek)
            all_rejected_mesh.append(entry)

    def _push_ta(terms: list[str]) -> None:
        for t in terms:
            k = t.lower()
            if k in seen_ta:
                continue
            seen_ta.add(k)
            all_ta_terms.append(t)

    for slot in req_final:
        ta, validated, rejected = _build_concept_text_and_mesh(slot)
        _push_mesh(validated, rejected)
        _push_ta(ta)
        required_blocks_by_source["pubmed"].append(
            _pubmed_concept_block(ta_terms=ta, mesh_terms=validated)
        )
        required_blocks_by_source["europe_pmc"].append(
            _epmc_concept_block(ta_terms=ta, mesh_terms=validated)
        )
        required_blocks_by_source["openalex"].append(
            _openalex_concept_block(ta_terms=ta)
        )

    for slot in opt_adj:
        ta, validated, rejected = _build_concept_text_and_mesh(slot)
        _push_mesh(validated, rejected)
        _push_ta(ta)
        optional_blocks_by_source["pubmed"].append(
            _pubmed_concept_block(ta_terms=ta, mesh_terms=validated)
        )
        optional_blocks_by_source["europe_pmc"].append(
            _epmc_concept_block(ta_terms=ta, mesh_terms=validated)
        )
        optional_blocks_by_source["openalex"].append(
            _openalex_concept_block(ta_terms=ta)
        )

    # Drop empty blocks.
    for source in required_blocks_by_source:
        required_blocks_by_source[source] = [
            b for b in required_blocks_by_source[source] if b
        ]
        optional_blocks_by_source[source] = [
            b for b in optional_blocks_by_source[source] if b
        ]

    method_blocks = _build_method_block_by_source(intent)
    study_type_filters: dict[str, str] = {
        s: _resolve_high_precision_filter(intent, s)
        for s in ("pubmed", "europe_pmc", "openalex")
    }

    # MeSH subheadings — high-precision / intent-specific only. Closed
    # subset of recognised subheadings.
    mesh_subheadings: list[str] = []
    if intent in {"drug_safety", "drug_complication", "therapy_safety"}:
        mesh_subheadings = ["adverse effects"]
    elif intent in {"diagnosis_accuracy"}:
        mesh_subheadings = ["diagnosis"]
    elif intent in {"epidemiology_prevalence_incidence"}:
        mesh_subheadings = ["epidemiology"]
    elif intent in {"pathophysiology_mechanism"}:
        mesh_subheadings = ["pathology"]

    # Families.
    families: list[QueryFamily] = []
    source_specific: dict[str, str] = {}
    family_names = _INTENT_FAMILY_TABLE.get(
        intent, ("high_recall", "high_precision"),
    )

    for fname in family_names:
        per_source = _emit_family_query_strings(
            family_name=fname,
            intent=intent,
            required_blocks_by_source=required_blocks_by_source,
            optional_blocks_by_source=optional_blocks_by_source,
            method_blocks_by_source=method_blocks,
            study_type_filters_by_source=study_type_filters,
        )
        for source, qstring in per_source.items():
            check_ok = _qa_validate_query(
                qstring,
                source=source,
                required_concept_count=len(
                    required_blocks_by_source.get(source, [])
                ),
                family_name=fname,
                warnings=warnings,
                quality_checks=quality_checks,
            )
            families.append(QueryFamily(
                family_name=fname,
                source=source,  # type: ignore[arg-type]
                query_string=qstring if check_ok else "",
                qa_checks_passed=check_ok,
                notes=[]
                if check_ok
                else ["query_dropped_due_to_qa_failure"],
            ))
            if check_ok and qstring:
                # First passing family per source wins for the
                # ``source_specific_query_strings`` map (typically
                # high_recall) — diagnostic representative only.
                if source not in source_specific:
                    source_specific[source] = qstring

    # Inferred concept partition.
    inferred_used: list[ConceptSummary] = []
    inferred_not_required: list[ConceptSummary] = []
    for s in (req_final + opt_adj):
        if not bool(s.get("inferred")):
            continue
        cs = _concept_summary_from_slot(s)
        if s in req_final and float(cs.confidence) >= 0.85:
            inferred_used.append(cs)
        else:
            inferred_not_required.append(cs)

    # Aggregate display-only and ranking-only NOT-searched lists.
    display_not_searched = [
        _concept_summary_from_slot(s) for s in (display_only_in + extra_display)
    ]
    ranking_not_searched = [
        _concept_summary_from_slot(s) for s in ranking_only_in
    ]

    # Required / optional concept summaries.
    required_used = [
        _concept_summary_from_slot(
            s, sources=s.get("_planner_sources"),
        ) for s in req_final
    ]
    optional_used = [
        _concept_summary_from_slot(
            s, sources=s.get("_planner_sources"),
        ) for s in opt_adj
    ]

    return BranchQueryFamilies(
        branch_id=branch_id,
        intent=intent,
        framework=framework,
        branch_priority=priority,
        branch_confidence=confidence,
        required_concepts_used=required_used,
        optional_concepts_used=optional_used,
        ranking_only_concepts_not_searched=ranking_not_searched,
        display_only_concepts_not_searched=display_not_searched,
        inferred_concepts_used=inferred_used,
        inferred_concepts_not_required=inferred_not_required,
        validated_MeSH_terms=all_validated_mesh,
        unvalidated_MeSH_terms_rejected_or_pending=all_rejected_mesh,
        title_abstract_terms=all_ta_terms,
        MeSH_subheadings_used=mesh_subheadings,
        query_families=families,
        source_specific_query_strings=source_specific,
        query_warnings=warnings,
        query_quality_checks=quality_checks,
        active_for_retrieval=False,
        mode="shadow",
    )


# ---------------------------------------------------------------------
# Query QA validator.
# ---------------------------------------------------------------------


_FORBIDDEN_QUERY_SUBSTRINGS: tuple[str, ...] = (
    "SYSTEM_PROMPT", "Return JSON only", "You are not",
    "Confidence guidance",
)


def _qa_validate_query(
    query_string: str,
    *,
    source: str,
    required_concept_count: int,
    family_name: str,
    warnings: list[str],
    quality_checks: list[QueryQualityCheck],
) -> bool:
    """Run all closed-set QA checks. Returns True iff every check passes.

    Failed checks emit a ``QueryQualityCheck`` AND a closed-set
    ``query_warnings`` entry. The query string is NEVER stored when
    ``False`` is returned.
    """
    # Empty query.
    if not query_string or not query_string.strip():
        quality_checks.append(QueryQualityCheck(
            name="non_empty_query", passed=False,
            detail=f"family={family_name} source={source}",
        ))
        warnings.append(f"qa_dropped:empty_query:{family_name}:{source}")
        return False
    # Forbidden substrings.
    for sub in _FORBIDDEN_QUERY_SUBSTRINGS:
        if sub in query_string:
            quality_checks.append(QueryQualityCheck(
                name="no_forbidden_substrings", passed=False,
                detail=f"matched={sub}",
            ))
            warnings.append(f"qa_dropped:forbidden_substring:{family_name}:{source}")
            return False
    # Rescue families may be unanchored by design; otherwise required.
    if family_name != "rescue" and required_concept_count < 1:
        quality_checks.append(QueryQualityCheck(
            name="has_required_anchor", passed=False,
            detail=f"family={family_name}",
        ))
        warnings.append(f"qa_dropped:no_required_anchor:{family_name}:{source}")
        return False
    # Syntax (closed-set codes via the existing compiler validator).
    syntax_source = "pubmed" if source == "pubmed" else (
        "europe_pmc" if source == "europe_pmc" else "europe_pmc"
    )
    if source == "openalex":
        # OpenAlex doesn't use the PubMed/EuropePMC field-tag grammar;
        # we still validate balanced parens here defensively.
        if query_string.count("(") != query_string.count(")"):
            quality_checks.append(QueryQualityCheck(
                name="balanced_parens", passed=False,
                detail=f"family={family_name} source={source}",
            ))
            warnings.append(
                f"qa_dropped:unbalanced_parens:{family_name}:{source}"
            )
            return False
        quality_checks.append(QueryQualityCheck(
            name="openalex_concept_block_present", passed=True,
            detail="",
        ))
        return True
    syntax = validate_query_syntax(query_string, source=syntax_source)
    if not syntax.valid:
        for err in syntax.errors:
            quality_checks.append(QueryQualityCheck(
                name="syntax", passed=False,
                detail=f"family={family_name} source={source} err={err}",
            ))
        warnings.append(
            f"qa_dropped:invalid_syntax:{family_name}:{source}"
        )
        return False
    quality_checks.append(QueryQualityCheck(
        name="syntax", passed=True,
        detail=f"family={family_name} source={source}",
    ))
    return True


# ---------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------


def plan_branch_queries(
    cqm: ClinicalQuestionMap | dict[str, Any],
    *,
    mode: PlannerMode = "shadow",
) -> BranchQueryPlan:
    """Phase K2 — emit a shadow ``BranchQueryPlan`` for ``cqm``.

    Hard guard: ``mode="active"`` raises an ``AssertionError`` — K2 is
    SHADOW ONLY. The active retrieval pipeline continues to use
    ``BooleanQueryCompilerService`` exclusively; this planner does NOT
    participate in retrieval.
    """
    assert mode != "active", (
        "BranchAwareBooleanQueryPlanner is SHADOW-ONLY in K2. "
        "mode='active' is rejected at the API boundary."
    )
    if mode == "disabled":
        return BranchQueryPlan(
            mode="disabled",
            active_for_retrieval=False,
            generated_at=datetime.now(timezone.utc).isoformat(),
            branches=[],
        )
    # Accept either the Pydantic model or a closed-set dict — the
    # pipeline persists JSON.
    cqm_dict = cqm.model_dump(mode="json") if isinstance(
        cqm, ClinicalQuestionMap
    ) else dict(cqm or {})
    primary_intent = str(cqm_dict.get("primary_intent") or "")
    clarification_required = bool(
        cqm_dict.get("clarification_required")
    )
    branches_in = list(cqm_dict.get("branches") or [])

    # Deferred / clarification — no retrievable plan generated.
    if (
        clarification_required
        or primary_intent in DEFERRED_INTENTS
        or not branches_in
    ):
        return BranchQueryPlan(
            mode="shadow",
            active_for_retrieval=False,
            generated_at=datetime.now(timezone.utc).isoformat(),
            branches=[],
        )

    out_branches: list[BranchQueryFamilies] = []
    for br in branches_in:
        if not isinstance(br, dict):
            continue
        try:
            out_branches.append(_plan_one_branch(br, cqm_dict))
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseK2 branch planning failed for branch %s (non-fatal)",
                br.get("branch_id"),
            )
            continue
    return BranchQueryPlan(
        mode="shadow",
        active_for_retrieval=False,
        generated_at=datetime.now(timezone.utc).isoformat(),
        branches=out_branches,
    )


__all__ = [
    # Enums
    "PlannerMode",
    "QueryFamilyName",
    "QuerySource",
    "UNVALIDATED_MESH_REASONS",
    # Models
    "ConceptSummary",
    "UnvalidatedMeshEntry",
    "QueryFamily",
    "QueryQualityCheck",
    "BranchQueryFamilies",
    "BranchQueryPlan",
    # Entry point
    "plan_branch_queries",
]
