"""Phase K1 — ClinicalQuestionMap durable interpretation object.

K1 introduces a NEW durable interpretation object that sits ABOVE PICOTT.
The existing Phase J pipeline (PHI guard → deterministic fast paths →
bounded resolver fallback → PICOTT → boolean compiler) MUST NOT regress.
PICOTT becomes a derived view of the ClinicalQuestionMap for
therapy/intervention questions; non-therapy intents are representable
without requiring a complete PICOTT shape.

This module is purely additive in K1 — it ships:
  * Closed-set enums (ClinicalIntent, Framework, SlotRole, SlotSource,
    SlotSearchPolicy, RetrievalStrategyHint, PreferredEvidenceType,
    InterpretationProvenance).
  * Pydantic models (DeterministicSignalSummary, Slot, InferredConcept,
    Branch, ClinicalQuestionMap) with ``extra="forbid"`` enforcement.
  * A pure function ``clinical_question_map_to_picott`` that derives a
    PicottTermMapping-compatible payload from the map (used by the
    pipeline integration in K1.I).

Hard rules (mirror the K0/K1 brief):
  * No LLM calls, no external network calls, no PHI handling here.
  * Every Literal enum is CLOSED-SET. Pydantic enforces this on instantiation.
  * Deferred intents (``qualitative``, ``implementation``,
    ``health_services``, ``patient_experience``, ``barriers_preferences``,
    ``policy``) NEVER route to therapy. They are representable but the
    pipeline (K1.I) routes them via ``clarification_required`` provenance.
  * ``slot_search_policy == "required_for_retrieval"`` MUST equal
    ``required_for_search == True`` — a validator enforces the
    backward-compat invariant.
  * Branch admission policy: primary branch always present; secondary
    branches admitted at ``branch_confidence >= 0.7`` OR explicit
    multi-intent language; max 3 branches total. The schema enforces
    the cap (``max_length=3``); the K1.I pipeline integration owns the
    confidence gating.

Pure / sync / stateless. Safe to import at module-load time.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------
# K1.G — schema version + prompt version sentinel.
# ---------------------------------------------------------------------

#: ClinicalQuestionMap schema version. Bump on any schema change.
SCHEMA_VERSION: str = "1.0"

#: Resolver prompt-version label bumped by K1.G + Phase K1.1 intent
#: disambiguation fix. The ``ClinicalContextResolverService`` reports
#: the active prompt version under ``ClinicalQuestionMap.prompt_version``
#: when the resolver was the source.
PROMPT_VERSION_K1: str = "j-tail-v2-K1.1"


# ---------------------------------------------------------------------
# K1.B — ClinicalIntent closed-set enum.
# ---------------------------------------------------------------------
# Active intents are routed by the pipeline. Deferred intents are
# representable (so the resolver can structure them) but the pipeline
# MUST NOT route them to therapy — see ``is_deferred_intent`` below.

ActiveIntent = Literal[
    "therapy",
    "therapy_safety",
    "drug_safety",
    "diagnosis_accuracy",
    "diagnostic_workup",
    "differential_diagnosis",
    "prognosis_natural_history",
    "prognostic_factor",
    "prediction_model",
    "risk_etiology_harm",
    "complication_risk",
    "disease_complication",
    "drug_complication",
    "procedure_complication",
    "device_complication",
    "epidemiology_prevalence_incidence",
    "pathophysiology_mechanism",
    "monitoring_followup_surveillance",
    "stabilization_emergency_management",
    "guideline_recommendation",
    "background_overview",
    "prevention_screening",
]

DeferredIntent = Literal[
    "qualitative",
    "implementation",
    "health_services",
    "patient_experience",
    "barriers_preferences",
    "policy",
    "qualitative_deferred",
]

#: Closed-set union of active + deferred intent identifiers.
ClinicalIntent = Literal[
    "therapy",
    "therapy_safety",
    "drug_safety",
    "diagnosis_accuracy",
    "diagnostic_workup",
    "differential_diagnosis",
    "prognosis_natural_history",
    "prognostic_factor",
    "prediction_model",
    "risk_etiology_harm",
    "complication_risk",
    "disease_complication",
    "drug_complication",
    "procedure_complication",
    "device_complication",
    "epidemiology_prevalence_incidence",
    "pathophysiology_mechanism",
    "monitoring_followup_surveillance",
    "stabilization_emergency_management",
    "guideline_recommendation",
    "background_overview",
    "prevention_screening",
    "qualitative",
    "implementation",
    "health_services",
    "patient_experience",
    "barriers_preferences",
    "policy",
    "qualitative_deferred",
]

ACTIVE_INTENTS: frozenset[str] = frozenset({
    "therapy", "therapy_safety", "drug_safety",
    "diagnosis_accuracy", "diagnostic_workup", "differential_diagnosis",
    "prognosis_natural_history", "prognostic_factor", "prediction_model",
    "risk_etiology_harm",
    "complication_risk", "disease_complication", "drug_complication",
    "procedure_complication", "device_complication",
    "epidemiology_prevalence_incidence", "pathophysiology_mechanism",
    "monitoring_followup_surveillance",
    "stabilization_emergency_management",
    "guideline_recommendation", "background_overview",
    "prevention_screening",
})

DEFERRED_INTENTS: frozenset[str] = frozenset({
    "qualitative", "implementation", "health_services",
    "patient_experience", "barriers_preferences", "policy",
    "qualitative_deferred",
})


def is_deferred_intent(intent: str) -> bool:
    """Return True when ``intent`` is in the closed-set deferred list.

    The pipeline (K1.I) MUST NOT route deferred intents to therapy. The
    typical handling is ``clarification_required=True`` OR routing the
    branch through a deferred placeholder. NEVER auto-route deferred
    intents to active retrieval lanes.
    """
    return intent in DEFERRED_INTENTS


# ---------------------------------------------------------------------
# K1.C — Framework closed-set enum.
# ---------------------------------------------------------------------

ActiveFramework = Literal[
    "PICOTS",
    "PICOTT",
    "PIRD",
    "PIRT",
    "PFO",
    "PECOT",
    "PECO",
    "PEO",
    "CHARMS",
    "CoCoPop",
    "PCC",
    "problem_context_differential",
    "concept_mechanism",
    "guideline_stabilization",
]

ReservedFramework = Literal[
    "SPIDER",
    "SPICE",
    "ECLIPSE",
]

Framework = Literal[
    "PICOTS",
    "PICOTT",
    "PIRD",
    "PIRT",
    "PFO",
    "PECOT",
    "PECO",
    "PEO",
    "CHARMS",
    "CoCoPop",
    "PCC",
    "problem_context_differential",
    "concept_mechanism",
    "guideline_stabilization",
    "SPIDER",
    "SPICE",
    "ECLIPSE",
]

ACTIVE_FRAMEWORKS: frozenset[str] = frozenset({
    "PICOTS", "PICOTT", "PIRD", "PIRT", "PFO",
    "PECOT", "PECO", "PEO",
    "CHARMS", "CoCoPop", "PCC",
    "problem_context_differential", "concept_mechanism",
    "guideline_stabilization",
})

RESERVED_FRAMEWORKS: frozenset[str] = frozenset({
    "SPIDER", "SPICE", "ECLIPSE",
})


# ---------------------------------------------------------------------
# K1.D — Slot schema (slot_role + slot_search_policy).
# ---------------------------------------------------------------------

SlotRole = Literal[
    "population",
    "age_group",
    "sex",
    "pregnancy",
    "clinical_setting",
    "clinical_problem",
    "disease_or_condition",
    "symptom_or_sign",
    "intervention",
    "drug_or_drug_class",
    "exposure",
    "procedure_or_surgery",
    "device_or_implant",
    "index_test",
    "target_condition",
    "reference_standard",
    "comparator",
    "outcome",
    "complication_or_adverse_event",
    "prognostic_factor",
    "prediction_model_or_score",
    "prediction_horizon",
    "mechanism_or_pathway",
    "epidemiology_context",
    "geography",
    "timeframe",
    "urgency",
    "guideline_context",
]

SlotSource = Literal[
    "deterministic_parser",
    "deterministic_concept_pack",
    "bounded_resolver_fallback",
    "manual",
    "unknown",
]

SlotSearchPolicy = Literal[
    "required_for_retrieval",
    "optional_for_retrieval",
    "ranking_only",
    "display_only",
    "clarification_needed",
]


# Reused closed-set entity_type tokens from Phase J3 / Phase 6G.4P.
EntityType = Literal[
    "disease_or_problem",
    "drug_or_drug_class",
    "vaccine",
    "diagnostic_test",
    "procedure",
    "procedure_or_surgery",
    "device_or_implant",
    "perioperative_context",
    "postprocedure_context",
    "management_intervention",
    "exposure",
    "biomarker_or_lab",
    "population_group",
    "clinical_action",
    "complication_or_adverse_event",
    "clinical_setting",
    "unknown",
]


class Slot(BaseModel):
    """A single framework-neutral concept slot.

    Carries both the legacy ``required_for_search`` boolean and the
    new ``slot_search_policy`` (closed-set). The validator enforces that
    they agree — ``slot_search_policy == "required_for_retrieval"`` iff
    ``required_for_search == True``. This keeps the boolean compiler
    consumers (which still read ``required_for_search``) in lock-step
    with the new retrieval policy field.
    """

    model_config = ConfigDict(extra="forbid")

    original_phrase: str
    canonical_concept: str
    entity_type: EntityType
    slot_role: SlotRole

    # BACKWARD COMPAT — bool legacy field. K1.I integrates by reading
    # this from the existing PICOTT must-have entries.
    required_for_search: bool

    # NEW PRIMARY FIELD — closed-set retrieval policy.
    slot_search_policy: SlotSearchPolicy

    inferred: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    synonyms: List[str] = Field(default_factory=list)
    search_terms: List[str] = Field(default_factory=list)
    source: SlotSource = "unknown"

    @model_validator(mode="after")
    def _enforce_search_policy_required_consistency(self) -> "Slot":
        """``slot_search_policy == "required_for_retrieval"`` MUST equal
        ``required_for_search == True`` and vice versa. This guards
        downstream consumers that still read the legacy boolean.
        """
        if (
            self.slot_search_policy == "required_for_retrieval"
            and not self.required_for_search
        ):
            raise ValueError(
                "slot_search_policy=required_for_retrieval requires "
                "required_for_search=True (legacy field must agree)",
            )
        if (
            self.slot_search_policy != "required_for_retrieval"
            and self.required_for_search
        ):
            raise ValueError(
                "required_for_search=True requires "
                "slot_search_policy=required_for_retrieval",
            )
        return self


# ---------------------------------------------------------------------
# K1.E — Branch model.
# ---------------------------------------------------------------------

RetrievalStrategyHint = Literal[
    "standard",
    "guideline_first",
    "rare_or_complication_rescue",
]

PreferredEvidenceType = Literal[
    "systematic_review",
    "meta_analysis",
    "rct",
    "observational_cohort",
    "case_control",
    "registry",
    "diagnostic_accuracy_study",
    "prediction_model_validation",
    "guideline",
    "narrative_review",
    "mechanism_review",
    "basic_science_review",
    "case_series_for_rare",
]


class Branch(BaseModel):
    """One interpretation branch.

    A multi-intent question can carry up to 3 branches. The pipeline
    (K1.I) is responsible for assembling primary + secondary branches
    per the admission policy. The schema only enforces the upper-bound
    cap and closed-set field types.
    """

    model_config = ConfigDict(extra="forbid")

    branch_id: str
    intent: ClinicalIntent
    framework: Framework
    branch_priority: int = Field(ge=1, le=3)
    branch_confidence: float = Field(ge=0.0, le=1.0)
    branch_reason: str = ""

    required_concepts: List[Slot] = Field(default_factory=list)
    optional_concepts: List[Slot] = Field(default_factory=list)
    ranking_only_concepts: List[Slot] = Field(default_factory=list)
    display_only_concepts: List[Slot] = Field(default_factory=list)

    preferred_evidence_types: List[PreferredEvidenceType] = Field(
        default_factory=list,
    )

    retrieval_strategy_hint: RetrievalStrategyHint = "standard"
    guideline_first: bool = False
    rare_or_complication_rescue: bool = False


# ---------------------------------------------------------------------
# K1.A — Top-level ClinicalQuestionMap.
# ---------------------------------------------------------------------

InterpretationProvenance = Literal[
    "deterministic_parser",
    "deterministic_concept_pack",
    "bounded_resolver_fallback",
    "clarification_required",
]


class InferredConcept(BaseModel):
    """A concept derived (not literally stated) from the question.

    Mirrors the J3 ``inferred=True`` resolver-anchor convention but
    captured at the question-map level so non-anchor inferences (e.g.,
    "anticoagulation" implied by "ICH risk in AF") have a structured
    home. Pure metadata — does not gate retrieval directly.
    """

    model_config = ConfigDict(extra="forbid")

    concept: str
    inferred_from: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class DeterministicSignalSummary(BaseModel):
    """Closed-set lexical / structural signals detected by the
    deterministic router (K1.F). NOT the semantic interpretation — just
    routing hints used to decide ``needs_resolver``.
    """

    model_config = ConfigDict(extra="forbid")

    detected_signals: List[str] = Field(default_factory=list)
    possible_intents: List[ClinicalIntent] = Field(default_factory=list)
    possible_multi_intent: bool = False
    deterministic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_resolver: bool = False
    reason: str = ""


class ClinicalQuestionMap(BaseModel):
    """Phase K1 — top-level durable interpretation object.

    Sits ABOVE PICOTT. PICOTT becomes a derived view for the
    therapy / intervention path; non-therapy intents are representable
    here without forcing a PICOTT shape. The pipeline persists this
    object additively under ``search_sessions.clinical_question_map_json``;
    the existing PICOTT-based path remains untouched.
    """

    model_config = ConfigDict(extra="forbid")

    original_question: str
    normalized_question: str

    interpretation_provenance: InterpretationProvenance

    clinical_validity: bool = True
    clarification_required: bool = False
    clarification_reason: Optional[str] = None

    primary_intent: ClinicalIntent
    secondary_intents: List[ClinicalIntent] = Field(default_factory=list)

    frameworks: List[Framework] = Field(default_factory=list)

    branches: List[Branch] = Field(default_factory=list, max_length=3)

    framework_neutral_slots: List[Slot] = Field(default_factory=list)
    inferred_concepts: List[InferredConcept] = Field(default_factory=list)

    candidate_keywords: List[str] = Field(default_factory=list)
    candidate_MeSH_terms_unvalidated: List[str] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # K1.G — schema_version is hard-coded for compatibility with future
    # migration tooling. NEVER mutate after construction.
    schema_version: Literal["1.0"] = SCHEMA_VERSION

    # K1.G — prompt_version is set by the pipeline ONLY when the
    # resolver was the interpretation source. Deterministic-parser /
    # concept-pack / clarification paths leave this None.
    prompt_version: Optional[str] = None

    deterministic_signal_summary: DeterministicSignalSummary = Field(
        default_factory=DeterministicSignalSummary,
    )


# ---------------------------------------------------------------------
# K1.H — Pure PICOTT derivation mapper.
# ---------------------------------------------------------------------

# Closed-set intents for which a FULL PICOTT shape derivation is
# guaranteed. Other intents fall through to a best-effort PICOTT
# (population/intervention/outcome populated where possible; missing
# slots stay empty — never fabricated).
_FULL_PICOTT_INTENTS: frozenset[str] = frozenset({
    "therapy",
    "therapy_safety",
    "drug_safety",
    "procedure_complication",
    "device_complication",
    "drug_complication",
    # Closed-set legacy J3 markers that downstream callers may emit as
    # ``primary_intent`` shorthand. ``electrolyte_replacement`` and
    # ``ICU_treatment_question`` are NOT in the K1.B ClinicalIntent
    # enum so the schema rejects them — but the PICOTT mapper still
    # surfaces a full shape when the resolver explicitly tagged them
    # via secondary_intents-as-strings (defensive only — the Pydantic
    # layer will refuse construction of an invalid map).
})

# Mapping from K1 SlotRole to picott_term_mapping_service PICOTT_ROLES.
# Closed-set; missing keys fall through to "clinical_action" (the
# generic action slot in the term-mapping schema).
_SLOT_ROLE_TO_PICOTT_ROLE: dict[str, str] = {
    "population": "population_problem",
    "age_group": "population_problem",
    "sex": "population_problem",
    "pregnancy": "population_problem",
    "clinical_problem": "population_problem",
    "disease_or_condition": "population_problem",
    "symptom_or_sign": "population_problem",
    "intervention": "intervention_exposure_index_test",
    "drug_or_drug_class": "intervention_exposure_index_test",
    "exposure": "intervention_exposure_index_test",
    "procedure_or_surgery": "intervention_exposure_index_test",
    "device_or_implant": "intervention_exposure_index_test",
    "index_test": "intervention_exposure_index_test",
    "target_condition": "outcomes_target_diagnosis",
    "reference_standard": "comparator_reference_standard",
    "comparator": "comparator_reference_standard",
    "outcome": "outcomes_target_diagnosis",
    "complication_or_adverse_event": "outcomes_target_diagnosis",
    "prognostic_factor": "outcomes_target_diagnosis",
    "prediction_model_or_score": "intervention_exposure_index_test",
    "prediction_horizon": "time_horizon",
    "mechanism_or_pathway": "clinical_action",
    "epidemiology_context": "clinical_action",
    "geography": "setting",
    "clinical_setting": "setting",
    "timeframe": "time_horizon",
    "urgency": "clinical_action",
    "guideline_context": "clinical_action",
}


def clinical_question_map_to_picott(
    cqm: ClinicalQuestionMap,
) -> dict[str, Any]:
    """Pure derivation: project a ``ClinicalQuestionMap`` onto a
    PICOTT-shaped dict suitable for the existing downstream pipeline.

    Returns a dict (NOT a Pydantic model) so the pipeline can keep
    using its existing ``picott_payload`` dict shape unchanged.

    Behavior:
      * ``primary_intent`` in ``_FULL_PICOTT_INTENTS`` → full PICOTT
        derivation (population / intervention / comparator / outcome /
        time_horizon / setting / clinical_action all populated where
        the slots support it).
      * Other intents → best-effort PICOTT: populates only the role
        buckets that have matching K1 slots. Missing slots stay empty.

    NEVER fabricates concepts. NEVER calls an LLM. NEVER touches the
    network.
    """
    # Collect slots from EVERY branch + the framework-neutral slots so
    # the derivation reflects the full question map. Branch
    # ``required_concepts`` win over ``optional_concepts`` win over
    # ``ranking_only_concepts`` win over ``display_only_concepts`` —
    # the term-mapping side flattens them, but the must-have bucket
    # only takes the required slots.
    all_slots: List[Slot] = list(cqm.framework_neutral_slots)
    required_slots: List[Slot] = []
    for branch in cqm.branches:
        required_slots.extend(branch.required_concepts)
        all_slots.extend(branch.required_concepts)
        all_slots.extend(branch.optional_concepts)
        all_slots.extend(branch.ranking_only_concepts)

    # Project into PICOTT role buckets.
    fields: dict[str, list[str]] = {
        "population": [],
        "intervention_or_exposure_or_index_test": [],
        "comparator_or_reference_standard": [],
        "outcomes_or_target_diagnosis": [],
        "time_horizon": [],
        "setting": [],
    }
    must_haves: list[dict[str, Any]] = []
    ranking_concepts: list[dict[str, Any]] = []

    def _bucket(slot: Slot) -> Optional[str]:
        target = _SLOT_ROLE_TO_PICOTT_ROLE.get(slot.slot_role)
        if target == "population_problem":
            return "population"
        if target == "intervention_exposure_index_test":
            return "intervention_or_exposure_or_index_test"
        if target == "comparator_reference_standard":
            return "comparator_or_reference_standard"
        if target == "outcomes_target_diagnosis":
            return "outcomes_or_target_diagnosis"
        if target == "time_horizon":
            return "time_horizon"
        if target == "setting":
            return "setting"
        return None

    seen_per_bucket: dict[str, set[str]] = {k: set() for k in fields}
    seen_mh: set[str] = set()
    seen_rk: set[str] = set()

    for slot in all_slots:
        bucket = _bucket(slot)
        if bucket and slot.canonical_concept:
            key = slot.canonical_concept.lower()
            if key not in seen_per_bucket[bucket]:
                seen_per_bucket[bucket].add(key)
                fields[bucket].append(slot.canonical_concept)
        # Must-have wiring — required_for_search slots become must-have
        # entries with the closed-set legacy concept role names.
        if slot.required_for_search and slot.canonical_concept:
            key = slot.canonical_concept.lower()
            if key not in seen_mh:
                seen_mh.add(key)
                legacy_role = _legacy_concept_role_for_slot_role(slot.slot_role)
                must_haves.append({
                    "concept": slot.canonical_concept,
                    "role": legacy_role,
                    "synonyms": list(slot.synonyms),
                    "required_for_high_relevance": True,
                })

    # Ranking-only slots from branches become ranking concepts.
    for branch in cqm.branches:
        for slot in branch.ranking_only_concepts:
            if not slot.canonical_concept:
                continue
            key = slot.canonical_concept.lower()
            if key in seen_rk:
                continue
            seen_rk.add(key)
            ranking_concepts.append({
                "concept": slot.canonical_concept,
                "role": _legacy_concept_role_for_slot_role(slot.slot_role),
                "synonyms": list(slot.synonyms),
                "boost_weight": 3,
            })

    # K1.H — closed-set intent → legacy type_of_question mapping.
    type_of_question = _intent_to_legacy_type_of_question(cqm.primary_intent)
    question_subtype = _intent_to_legacy_subtype(cqm.primary_intent)
    question_scope = _intent_to_legacy_scope(cqm.primary_intent)
    specialty = ""

    # Phase K4A.2 — severity-marker overview detection. Closed-set,
    # deterministic, ADDITIVE: overrides ``question_subtype`` only
    # when the question matches the severity-marker family. All other
    # questions are unchanged.
    try:
        from services.severity_marker_overview import (
            is_severity_marker_question,
            extract_condition_anchor,
        )
        if is_severity_marker_question(cqm.original_question or ""):
            question_subtype = "severity_marker_overview"
            _severity_anchor = extract_condition_anchor(
                cqm.original_question or "",
            )
        else:
            _severity_anchor = None
    except Exception:  # noqa: BLE001
        _severity_anchor = None

    out: dict[str, Any] = {
        "question_scope": question_scope,
        "normalized_question": cqm.normalized_question,
        "original_question": cqm.original_question,
        "pico": fields,
        "type_of_question": type_of_question,
        "question_subtype": question_subtype,
        "specialty": specialty,
        "must_have_concepts": must_haves,
        "ranking_concepts": ranking_concepts,
        "should_have_concepts": [],
        "negative_or_penalty_concepts": [],
        "missing_elements": {},
        "interpretation_for_user": "",
        "rationale": (
            f"clinical_question_map source={cqm.interpretation_provenance} "
            f"intent={cqm.primary_intent}"
        ),
        "confidence": cqm.confidence,
    }
    # Best-effort vs full path tag — purely informational, used by the
    # transparency panel / tests to confirm the derivation tier.
    if cqm.primary_intent in _FULL_PICOTT_INTENTS or _is_full_picott_legacy_token(
        cqm.primary_intent,
    ):
        out["_k1_derivation_tier"] = "full"
    else:
        out["_k1_derivation_tier"] = "best_effort"
    # Phase K4A.2 — surface the severity-marker condition anchor on
    # the picott so the ranker + default_selection can read it without
    # re-parsing the question. None unless the question family matched.
    if _severity_anchor:
        out["severity_marker_condition_anchor"] = _severity_anchor

    # Phase K4A.2 — when the severity-marker overview route is active,
    # AUGMENT each outcome must-have's ``synonyms`` with the closed-set
    # severity-or-block synonyms list. This is the cap-fix: it turns
    # the query-compiler AND-block's outcome clause from a single
    # literal phrase (e.g., ``"severity of illness"``) into a broad
    # OR-block that ALSO accepts severity / severe / predictor /
    # marker / biomarker / scoring system / named scores (APACHE,
    # BISAP, Ranson, CTSI) / organ failure / mortality / etc.
    # Closed-set, deterministic, NO LLM. Only fires for this route.
    if question_subtype == "severity_marker_overview":
        try:
            from services.severity_marker_overview import (
                severity_or_block_synonyms,
            )
            or_block = severity_or_block_synonyms()
            for mh in out.get("must_have_concepts") or []:
                if not isinstance(mh, dict):
                    continue
                if (mh.get("role") or "").lower() != "outcome":
                    continue
                existing = [s for s in (mh.get("synonyms") or []) if s]
                seen_low = {s.lower() for s in existing}
                # Also include the canonical concept itself so the
                # union of {canonical} ∪ existing-synonyms ∪ OR-block
                # is searchable.
                canon = mh.get("concept") or ""
                if canon and canon.lower() not in seen_low:
                    existing.append(canon)
                    seen_low.add(canon.lower())
                for syn in or_block:
                    if syn.lower() not in seen_low:
                        existing.append(syn)
                        seen_low.add(syn.lower())
                mh["synonyms"] = existing
                # Outcome must-haves in this route should NOT clamp
                # recall to the exact phrase — flip the
                # ``required_for_high_relevance`` flag for the outcome
                # so the ranker keeps it as a ranking concept rather
                # than a relevance-cap gate. Closed-set; safe because
                # the directness classifier owns recall via its own
                # boost/cap table.
                mh["required_for_high_relevance"] = False
        except Exception:  # noqa: BLE001
            # Defensive — never break the derivation if the helper
            # is unavailable.
            pass

    return out


def _is_full_picott_legacy_token(intent: str) -> bool:
    """Closed-set guard for legacy resolver tokens that still want a full
    PICOTT shape (kept distinct from ``_FULL_PICOTT_INTENTS`` so the
    K1 Literal enum stays minimal).
    """
    return intent in {
        "electrolyte_replacement",
        "ICU_treatment_question",
    }


def _legacy_concept_role_for_slot_role(slot_role: str) -> str:
    """Map a K1 ``SlotRole`` to the legacy ``ConceptRole`` token used by
    the boolean compiler. Closed-set; falls through to ``"other"``.
    """
    if slot_role in {
        "population", "age_group", "sex", "pregnancy",
        "clinical_problem", "disease_or_condition", "symptom_or_sign",
    }:
        return "condition"
    if slot_role in {
        "intervention", "drug_or_drug_class", "exposure",
        "procedure_or_surgery", "device_or_implant",
        "prediction_model_or_score",
    }:
        return "intervention"
    if slot_role == "comparator":
        return "comparator"
    if slot_role == "index_test":
        return "index_test"
    if slot_role in {
        "outcome", "target_condition", "complication_or_adverse_event",
        "prognostic_factor",
    }:
        return "outcome"
    if slot_role == "reference_standard":
        return "comparator"
    if slot_role in {"clinical_setting", "geography"}:
        return "setting"
    if slot_role in {"timeframe", "prediction_horizon"}:
        return "time_horizon"
    return "other"


def _intent_to_legacy_type_of_question(intent: str) -> str:
    """Closed-set mapping from K1 intent → legacy ``type_of_question``
    token used by the existing study-type router. Falls back to
    ``"other"`` for unknown / deferred intents.
    """
    table = {
        "therapy": "therapy",
        "therapy_safety": "therapy",
        "drug_safety": "etiology_harm",
        "diagnosis_accuracy": "diagnosis",
        "diagnostic_workup": "diagnosis",
        "differential_diagnosis": "diagnosis",
        "prognosis_natural_history": "prognosis",
        "prognostic_factor": "prognosis",
        "prediction_model": "prognosis",
        "risk_etiology_harm": "etiology_harm",
        "complication_risk": "etiology_harm",
        "disease_complication": "etiology_harm",
        "drug_complication": "etiology_harm",
        "procedure_complication": "etiology_harm",
        "device_complication": "etiology_harm",
        "epidemiology_prevalence_incidence": "epidemiology",
        "pathophysiology_mechanism": "background",
        "monitoring_followup_surveillance": "therapy",
        "stabilization_emergency_management": "therapy",
        "guideline_recommendation": "guideline",
        "background_overview": "background",
        "prevention_screening": "prevention",
    }
    return table.get(intent, "other")


def _intent_to_legacy_subtype(intent: str) -> Optional[str]:
    """Map K1 intent → optional legacy subtype string. Returns None
    when the intent has no canonical subtype.
    """
    table = {
        "therapy_safety": "drug_safety",
        "drug_safety": "drug_safety",
        "drug_complication": "drug_safety",
        "device_complication": "device_complication_risk",
        "procedure_complication": "procedure_complication_risk",
        "diagnosis_accuracy": "diagnostic_accuracy",
        "diagnostic_workup": "diagnostic_workup",
        "differential_diagnosis": "differential_diagnosis",
        "prognosis_natural_history": "prognosis_natural_history",
        "prognostic_factor": "prognostic_factor",
        "prediction_model": "prediction_model",
        "background_overview": "disease_overview",
        "prevention_screening": "prevention_screening",
        "guideline_recommendation": "guideline_recommendation",
    }
    return table.get(intent)


def _intent_to_legacy_scope(intent: str) -> str:
    """K1 intent → legacy ``question_scope``. Closed-set."""
    if intent == "background_overview":
        return "background"
    if intent == "pathophysiology_mechanism":
        return "background"
    if intent == "epidemiology_prevalence_incidence":
        return "hybrid"
    return "foreground"


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "PROMPT_VERSION_K1",
    "ACTIVE_INTENTS",
    "DEFERRED_INTENTS",
    "ACTIVE_FRAMEWORKS",
    "RESERVED_FRAMEWORKS",
    # Enums (re-exported for tests + type stubs)
    "ActiveIntent",
    "DeferredIntent",
    "ClinicalIntent",
    "ActiveFramework",
    "ReservedFramework",
    "Framework",
    "SlotRole",
    "SlotSource",
    "SlotSearchPolicy",
    "EntityType",
    "RetrievalStrategyHint",
    "PreferredEvidenceType",
    "InterpretationProvenance",
    # Models
    "DeterministicSignalSummary",
    "Slot",
    "InferredConcept",
    "Branch",
    "ClinicalQuestionMap",
    # Helpers
    "is_deferred_intent",
    "clinical_question_map_to_picott",
    # K1.2 — slot semantics normalization.
    "DEMOGRAPHIC_QUALIFIER_LEXICON",
    "GENERIC_MODIFIER_DENY_LIST",
    "split_demographic_qualifier_from_disease_phrase",
    "is_generic_modifier_token",
    "normalize_slot_semantics",
]


# ---------------------------------------------------------------------
# Phase K1.2 — slot-semantics normalization (closed-set, deterministic).
#
# Two fixes:
#   Fix 1: When the deterministic anchor / concept pack folds a
#          demographic qualifier ("in adults", "in children",
#          "in older adults", ...) into the disease phrase, split it
#          back out into a SEPARATE population / age_group slot and
#          strip the qualifier from the disease slot's original_phrase
#          + canonical_concept.
#   Fix 2: Filter generic-modifier tokens ("benefits", "risks",
#          "outcomes", "complications" (generic), "recommendations",
#          ...) out of ``slot_role=population`` and ``required_for_search``
#          slots. Specific clinical entities ("infection", "mortality",
#          "bleeding") MUST still pass through.
#
# Both functions are pure / sync / stateless / no I/O / no LLM.
# Closed-set: every output enum value is in the K1 Literal sets above.
# ---------------------------------------------------------------------

import re as _re_k12  # noqa: E402  (kept local to the normalization section)


#: Closed-set demographic-qualifier lexicon (Fix 1). Each entry maps a
#: phrase pattern (case-insensitive, word-bounded) to its canonical
#: ``slot_role`` and ``canonical_concept``. Order matters — longer /
#: more specific phrases first so "older adults" wins over "adults".
DEMOGRAPHIC_QUALIFIER_LEXICON: tuple[tuple[str, str, str], ...] = (
    # (pattern, canonical_concept, slot_role)
    (r"older\s+adults?",        "older adults", "age_group"),
    (r"elderly\s+patients?",    "older adults", "age_group"),
    (r"\belderly\b",            "older adults", "age_group"),
    (r"\bgeriatric\b",          "older adults", "age_group"),
    (r"adult\s+patients?",      "adult",        "population"),
    (r"\badults?\b",            "adult",        "population"),
    (r"\bpediatric\b",          "pediatric",    "age_group"),
    (r"\bpaediatric\b",         "pediatric",    "age_group"),
    (r"pediatric\s+patients?",  "pediatric",    "age_group"),
    (r"\bchildren\b",           "children",     "age_group"),
    (r"\bchild\b",              "children",     "age_group"),
    (r"\badolescents?\b",       "adolescents",  "age_group"),
    (r"\bteenagers?\b",         "adolescents",  "age_group"),
    (r"\binfants?\b",           "infants",      "age_group"),
    (r"\bneonates?\b",          "neonates",     "age_group"),
    (r"\bnewborns?\b",          "neonates",     "age_group"),
    (r"pregnant\s+patients?",   "pregnancy",    "pregnancy"),
    (r"\bpregnancy\b",          "pregnancy",    "pregnancy"),
    (r"\bpregnant\b",           "pregnancy",    "pregnancy"),
)

# Pre-compile patterns at module load. ``_BOUND`` lets us recognise
# either a phrase-internal occurrence ("AF in adults") or a trailing
# occurrence ("AF in adults?"). The capturing group is the bare
# qualifier; the surrounding ``in\s+`` / leading-comma context is
# consumed via the ``_QUAL_WITH_LEAD`` pattern below.
_QUAL_BARE = [
    (_re_k12.compile(p, _re_k12.IGNORECASE), canon, role)
    for (p, canon, role) in DEMOGRAPHIC_QUALIFIER_LEXICON
]

#: Closed-set generic-modifier deny-list (Fix 2). These tokens MUST NOT
#: appear as ``slot_role=population`` slots and MUST NOT be
#: ``required_for_retrieval``. When the deterministic parser / resolver
#: hands back one of these as a population anchor, we either drop it,
#: relegate it to ``display_only``, or move it to ``ranking_only`` as a
#: generic outcome-domain marker.
#:
#: Overcorrection caveat: specific complications (``infection``,
#: ``bleeding``, ``mortality``, ``stroke``, ``lead dislodgement``, ...)
#: are NOT in this list — they remain legitimate outcome /
#: complication slots. The deny-list targets the GENERIC modifiers only
#: (word-bounded, case-insensitive, exact-token comparison).
GENERIC_MODIFIER_DENY_LIST: frozenset[str] = frozenset({
    "benefit",
    "benefits",
    "risk",
    "risks",
    "side effect",
    "side effects",
    "adverse effect",
    "adverse effects",
    "outcome",
    "outcomes",
    "complication",     # generic modifier ONLY (see overcorrection caveat)
    "complications",    # generic modifier ONLY (see overcorrection caveat)
    "recommendation",
    "recommendations",
    "management",
    "treatment",
})


def is_generic_modifier_token(text: str | None) -> bool:
    """Return True when ``text`` (after strip + lower-case) is in the
    closed-set generic-modifier deny-list. Comparison is exact-token
    (case-insensitive, whitespace-stripped). Multi-word entries match
    when the entire string equals the entry — i.e. "side effects"
    matches "Side Effects" but NOT "side effects of warfarin"
    (a longer phrase will still extract the SPECIFIC drug "warfarin"
    as the anchor; the literal modifier alone is what we deny).
    """
    if not text:
        return False
    low = " ".join(text.lower().split())
    return low in GENERIC_MODIFIER_DENY_LIST


def split_demographic_qualifier_from_disease_phrase(
    original_phrase: str,
) -> tuple[str, Optional[tuple[str, str, str]]]:
    """Pure helper for Fix 1.

    Given a candidate disease/clinical-problem phrase such as
    ``"prevalence of atrial fibrillation in adults"``, detect a trailing
    or embedded demographic qualifier from the closed-set lexicon and
    return ``(stripped_phrase, (qualifier_canonical, slot_role,
    qualifier_original))`` or ``(stripped_phrase, None)`` when no
    qualifier is present. ``stripped_phrase`` is the input with the
    qualifier (and an optional leading ``in `` / ``among `` / ``for ``
    / ``,``) removed and whitespace normalised. Never returns an
    empty stripped_phrase: if removing the qualifier would empty the
    phrase, we leave the phrase untouched and return ``None``.
    """
    s = (original_phrase or "").strip()
    if not s:
        return s, None
    # Match "<lead><qualifier>" where lead ∈ {in, among, for, ", "}
    # and qualifier is one of the lexicon entries. We process the
    # lexicon in order so "older adults" wins over "adults".
    for patt, canon, role in _QUAL_BARE:
        # Trailing form: ``... in adults`` / ``..., older adults`` /
        # ``... among children``.
        m = _re_k12.search(
            r"(?:\s+(?:in|among|for)\s+|\s*,\s+)(" + patt.pattern + r")\s*$",
            s,
            _re_k12.IGNORECASE,
        )
        if m:
            original_qualifier = m.group(1)
            stripped = s[: m.start()].rstrip(" ,;:")
            if stripped:
                return stripped, (canon, role, original_qualifier)
            return s, None
        # Embedded form: ``... in adults with X`` — only strip when
        # qualifier is at the end of a clinical noun phrase. We
        # conservatively decline to strip embedded mid-phrase to avoid
        # mangling specific clinical phrases (e.g., "Adult Onset
        # Diabetes" — adult is part of the canonical name there).
    return s, None


def _slot_from_demographic_qualifier(
    canonical: str,
    role: str,
    original: str,
    *,
    inferred_from_disease_slot: bool = False,
) -> dict[str, Any]:
    """Build a closed-set Slot dict for an extracted demographic
    qualifier. Always emitted as ``optional_for_retrieval`` so the
    boolean compiler does NOT force the demographic into the
    retrieval query (the K2 branch-aware compiler can promote it
    later) while still preserving it as structured CQM data and as a
    ranking signal.
    """
    return {
        "original_phrase": original,
        "canonical_concept": canonical,
        "entity_type": "population_group",
        "slot_role": role,
        "required_for_search": False,
        "slot_search_policy": "optional_for_retrieval",
        "inferred": bool(inferred_from_disease_slot),
        "confidence": 0.85,
        "synonyms": [],
        "search_terms": [],
        "source": "deterministic_parser",
    }


def normalize_slot_semantics(
    *,
    required_slots: List[dict[str, Any]],
    optional_slots: List[dict[str, Any]],
    display_only_slots: List[dict[str, Any]],
) -> tuple[
    List[dict[str, Any]], List[dict[str, Any]], List[dict[str, Any]]
]:
    """Phase K1.2 entry point.

    Applies Fix 1 (demographic-qualifier split) and Fix 2 (generic-
    modifier deny-list) to the closed-set slot lists produced by the
    pipeline. Pure / sync; never raises. Returns the three slot
    buckets after normalization.

    Behaviour:
      * Fix 1 — For every slot whose ``slot_role`` is a clinical
        anchor (disease_or_condition, clinical_problem, intervention,
        drug_or_drug_class, exposure, procedure_or_surgery,
        device_or_implant, index_test, target_condition), inspect its
        ``original_phrase`` for a trailing closed-set demographic
        qualifier ("in adults", "in older adults", "in children",
        "in pregnancy", ...). When found, strip the qualifier from the
        slot's ``original_phrase`` AND ``canonical_concept`` (the
        canonical concept is updated ONLY when the canonical equals
        the original — we never overwrite a real ontology-supplied
        canonical), and emit a NEW closed-set population / age_group /
        pregnancy slot for the qualifier (``optional_for_retrieval``,
        ``inferred=false``). Deduplicate against existing slots so the
        same demographic doesn't appear twice.
      * Fix 2 — Walk every required / optional / display-only slot.
        When a slot has ``slot_role=population`` AND its
        ``original_phrase`` (or canonical_concept) is in
        ``GENERIC_MODIFIER_DENY_LIST``, demote it to ``display_only``
        with ``required_for_search=false``. When such a slot is in the
        ``required_slots`` list, REMOVE it from required entirely.
        Specific complications / outcomes are untouched.
    """
    # ---- Fix 1 — demographic qualifier extraction. -----------------
    _CLINICAL_ANCHOR_ROLES: frozenset[str] = frozenset({
        "disease_or_condition", "clinical_problem", "target_condition",
        "intervention", "drug_or_drug_class", "exposure",
        "procedure_or_surgery", "device_or_implant", "index_test",
    })
    extracted_qualifiers: list[tuple[str, str, str]] = []  # (canon, role, original)
    seen_qualifier_keys: set[str] = set()
    for slot in (required_slots + optional_slots):
        if not isinstance(slot, dict):
            continue
        if slot.get("slot_role") not in _CLINICAL_ANCHOR_ROLES:
            continue
        original = str(slot.get("original_phrase") or "")
        canonical = str(slot.get("canonical_concept") or "")
        stripped, qual = split_demographic_qualifier_from_disease_phrase(
            original
        )
        if qual is None:
            continue
        canon, role, original_qual = qual
        key = f"{canon.lower()}::{role}"
        if key not in seen_qualifier_keys:
            seen_qualifier_keys.add(key)
            extracted_qualifiers.append((canon, role, original_qual))
        # Strip the qualifier off the slot. Only overwrite the
        # canonical_concept when it equals the original (i.e. the
        # parser used the raw phrase as canonical). Real ontology
        # canonicals (e.g. "atrial fibrillation") are untouched.
        if canonical.strip().lower() == original.strip().lower():
            slot["canonical_concept"] = stripped
        slot["original_phrase"] = stripped
    # Append the new demographic slots into ``optional_slots`` (per
    # the K1.2 spec) but dedupe against the existing population /
    # age_group / pregnancy slots already in any bucket.
    existing_demo_keys: set[str] = set()
    for s in (required_slots + optional_slots + display_only_slots):
        if isinstance(s, dict) and s.get("slot_role") in {
            "population", "age_group", "pregnancy", "sex",
        }:
            existing_demo_keys.add(
                (s.get("canonical_concept") or "").strip().lower()
            )
    for canon, role, original_qual in extracted_qualifiers:
        if canon.lower() in existing_demo_keys:
            continue
        existing_demo_keys.add(canon.lower())
        optional_slots.append(
            _slot_from_demographic_qualifier(
                canonical=canon,
                role=role,
                original=original_qual,
            ),
        )

    # ---- Fix 2 — generic-modifier deny-list. -----------------------
    def _is_generic(s: dict[str, Any]) -> bool:
        if not isinstance(s, dict):
            return False
        # Population-mis-tagged generic modifiers only.
        if s.get("slot_role") != "population":
            return False
        return (
            is_generic_modifier_token(s.get("original_phrase"))
            or is_generic_modifier_token(s.get("canonical_concept"))
        )

    new_required: list[dict[str, Any]] = []
    new_optional = list(optional_slots)
    new_display = list(display_only_slots)
    for s in required_slots:
        if _is_generic(s):
            # Drop from required entirely. NEVER appear as
            # display_only-population either — generic modifiers are
            # not population at all.
            continue
        new_required.append(s)
    new_optional = [s for s in new_optional if not _is_generic(s)]
    new_display = [s for s in new_display if not _is_generic(s)]
    return new_required, new_optional, new_display
