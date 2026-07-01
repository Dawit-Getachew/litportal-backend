"""Phase H1 — Bounded LLM Clinical Context Resolver (fallback / shadow).

Runs ONLY when the deterministic anchor pipeline cannot produce a
primary clinical anchor (e.g. ``anchor_failure_reason ==
"no_primary_clinical_entity"``). Emits a closed-set, schema-locked
PICOTT/anchor JSON that feeds the EXISTING deterministic Boolean
Query Compiler.

Safety invariants enforced by THIS module (every constraint is
checked again before the output is merged downstream):

  * The resolver NEVER searches sources.
  * The resolver NEVER recommends articles.
  * The resolver NEVER generates clinical answers / treatment advice.
  * The resolver NEVER emits PMIDs / DOIs / specific article refs.
  * The resolver NEVER invents MeSH terms (the schema has no MeSH
    field). If any string field contains MeSH-looking text we DO NOT
    treat it as authoritative — the curated MeSH resolver is the
    only source of MeSH.
  * The resolver NEVER references Google Scholar / ResearchGate /
    Sci-Hub. Outputs containing those tokens are REJECTED.
  * The resolver CANNOT override deterministic caps.

Closed-set rejection enum + scrubbing pass live in :func:`validate_output`.
The system prompt is a module-level constant (so tests can inspect it).

NO PHI is ever persisted by this service. No prompts, completions,
question text, or rationale free-text reach audit_events or
``session_diagnostics_json``. Only counters + closed-set strings.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from services.diagnostics import SessionDiagnostics, get_current as diag_current
from services.llm_provider import llm_provider

logger = logging.getLogger("clinevidence.clinical_context_resolver")


# ---------------------------------------------------------------------
# Closed-set enums.
# ---------------------------------------------------------------------


ResolverMode = Literal["off", "shadow", "fallback_active"]


REJECTION_REASONS: frozenset[str] = frozenset({
    "schema_invalid",
    "missing_primary_anchor_for_foreground",
    "confidence_below_threshold",
    "contains_pmid_doi_pattern",
    "contains_article_reference",
    "contains_answer_text",
    "contains_phi_pattern",
    "references_unapproved_source",
    "enum_violation",
    "empty_output",
})


SKIPPED_REASONS: frozenset[str] = frozenset({
    "phi_blocked",
    "generic_only_fast_path_fired",
    "deterministic_high_confidence",
    "background_anchor_already_succeeded",
    "post_procedure_router_already_succeeded",
    "mode_off",
    "no_trigger_matched",
    "cached_resolution",
})


# Closed-set entity_type enum used inside resolver output (mirrors the
# anchor service ENTITY_TYPES but tightens to the Phase H1 brief).
#
# Phase J3 — additive entries (closed-set):
#   complication_or_adverse_event — for device/procedure complication
#       and adverse-event outcome anchors (J7-A pacemaker migration,
#       J7-E central line infection, tracheostomy complications).
#   clinical_setting — for population/setting anchors (J7-B and
#       J7-E magnesium/potassium ICU questions). Note: by itself
#       insufficient to satisfy the clinical-subject requirement —
#       the J2 validator enforces that downstream.
_RESOLVER_ENTITY_TYPES: frozenset[str] = frozenset({
    "disease_or_problem", "drug_or_drug_class", "vaccine",
    "diagnostic_test", "procedure_or_surgery", "device_or_implant",
    "management_intervention", "biomarker_or_lab", "exposure",
    "population_group", "clinical_action", "unknown",
    # Phase J3 additive.
    "complication_or_adverse_event", "clinical_setting",
})


_RESOLVER_ROLES: frozenset[str] = frozenset({
    "population_problem", "intervention_exposure_index_test",
    "comparator_reference_standard", "outcomes_target_diagnosis",
    "time_horizon", "setting", "clinical_action", "procedure_context",
})


_RESOLVER_SCOPES: frozenset[str] = frozenset({
    "background", "foreground", "hybrid", "multi_intent",
})


# Threshold below which resolver output is rejected unconditionally.
CONFIDENCE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------
# Pydantic schemas (schema-locked, extra="forbid").
# ---------------------------------------------------------------------


class ResolverPrimaryAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    canonical_concept: str
    entity_type: Literal[
        "disease_or_problem", "drug_or_drug_class", "vaccine",
        "diagnostic_test", "procedure_or_surgery", "device_or_implant",
        "management_intervention", "biomarker_or_lab", "exposure",
        "population_group", "clinical_action", "unknown",
        # Phase J3 additive.
        "complication_or_adverse_event", "clinical_setting",
    ]
    role: Literal[
        "population_problem", "intervention_exposure_index_test",
        "comparator_reference_standard", "outcomes_target_diagnosis",
        "time_horizon", "setting", "clinical_action", "procedure_context",
    ]
    required_for_search: bool
    missing_relevance_cap: int = Field(ge=0, le=100)
    # Phase J3 — optional `inferred` flag (defaults to False so older
    # outputs without the field still validate). When the resolver
    # derives a concept (e.g. "hypomagnesemia" inferred from
    # "magnesium replacement"), it must mark the anchor inferred=True
    # so the compiler can demote the anchor to optional unless
    # resolver confidence ≥ 0.85.
    inferred: bool = False


class ResolverMustHaveConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concept: str
    role: Literal[
        "population_problem", "intervention_exposure_index_test",
        "comparator_reference_standard", "outcomes_target_diagnosis",
        "time_horizon", "setting", "clinical_action", "procedure_context",
    ]
    required_for_high_relevance: bool
    missing_relevance_cap: int = Field(ge=0, le=100)


class ResolverPicottFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population_problem: List[str] = Field(default_factory=list)
    intervention_exposure_index_test: List[str] = Field(default_factory=list)
    comparator_reference_standard: List[str] = Field(default_factory=list)
    outcomes_target_diagnosis: List[str] = Field(default_factory=list)
    time_horizon: List[str] = Field(default_factory=list)
    setting: List[str] = Field(default_factory=list)
    clinical_action: List[str] = Field(default_factory=list)


class ClinicalContextResolverOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_valid_clinical_question: bool
    question_scope: Literal["background", "foreground", "hybrid", "multi_intent"]
    type_of_question: str
    question_subtype: Optional[str] = None
    specialty: List[str] = Field(default_factory=list)
    picott: ResolverPicottFields
    primary_anchors: List[ResolverPrimaryAnchor] = Field(default_factory=list)
    generic_modifiers: List[str] = Field(default_factory=list)
    must_have_concepts: List[ResolverMustHaveConcept] = Field(default_factory=list)
    ranking_concepts: List[str] = Field(default_factory=list)
    negative_concepts: List[str] = Field(default_factory=list)
    clarification_needed: bool
    clarification_message: Optional[str] = None
    refinement_note: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class ResolverDecision(BaseModel):
    """Closed-set transport between resolver and pipeline. NO question
    text, NO prompts, NO completions."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    output: Optional[ClinicalContextResolverOutput] = None
    rejection_reason: Optional[str] = None
    mode: Literal["off", "shadow", "fallback_active"]
    latency_ms: int = 0
    token_estimate: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------
# Closed-set regex bank for output scrubbing (Task 2 defensive pass).
# ---------------------------------------------------------------------


# PMID-like (5–9 digit run when in an "PMID: 12345678" context AND
# bare runs of 7-9 digits anywhere in the output).
_PMID_RE = re.compile(r"\b(?:PMID|pmid)\s*[:#]?\s*\d{5,}\b", re.IGNORECASE)
_BARE_LONG_DIGITS_RE = re.compile(r"\b\d{7,}\b")
# DOI prefix.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+\b")
# Common article reference patterns.
_ARTICLE_REF_RE = re.compile(
    r"\b(?:doi:|pmcid|nct\d+|clinicaltrials\.gov)\b", re.IGNORECASE,
)
# Unapproved sources.
_UNAPPROVED_SOURCE_RE = re.compile(
    r"\b(?:google\s*scholar|researchgate|sci[\s-]*hub)\b", re.IGNORECASE,
)
# PHI-looking patterns (subset of services.phi_guard for defense-in-depth).
_PHI_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHI_PHONE_RE = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=[^0-9A-Za-z]))"
    r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b",
)
_PHI_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHI_MRN_RE = re.compile(r"\bMRN\s*[:#]?\s*\d+\b", re.IGNORECASE)
_PHI_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")


# Tokens that, when seen in a list field that should hold short
# clinical concepts, signal the LLM produced an "answer" or long
# narrative blurb instead of search interpretation.
_ANSWER_TEXT_INDICATORS: tuple[str, ...] = (
    " recommend", " is recommended", " should be given",
    " administer", " contraindicated",
    " in conclusion", " treatment of choice",
    " preferred regimen",
)


# ---------------------------------------------------------------------
# System prompt — module-level constant so tests can inspect / hash it.
# ---------------------------------------------------------------------


SYSTEM_PROMPT: str = """You are the Clinical Context Resolver for a clinician-facing evidence retrieval product.

You are not answering the medical question.
You are only extracting structured clinical search interpretation.
Return JSON only. No prose outside the JSON block.
Do not cite articles. Do not include PMIDs, DOIs, or specific publication references.
Do not recommend treatment. Do not generate clinical advice.
Do not reference Google Scholar, ResearchGate, Sci-Hub, or any source outside the deterministic Boolean compiler's pipeline.
Do not invent MeSH terms. The output schema does not include MeSH fields.
If the question lacks a clinical subject, set clarification_needed=true and leave primary_anchors empty.

Confidence guidance:
- If the question contains at least one clear, named clinical entity (a specific drug, disease, biomarker, diagnostic test, procedure, vaccine, or device — e.g., "rituximab", "Brugada syndrome", "Maze procedure", "aortic valve replacement", "metformin", "HbA1c", "pacemaker", "ICD lead", "central venous catheter", "tracheostomy tube"), set is_valid_clinical_question=true, set clarification_needed=false, populate primary_anchors with that entity, and set confidence between 0.7 and 0.95.
- If the question contains only generic modifiers and no clear clinical entity (e.g., "what is treatment?"), set is_valid_clinical_question=false, clarification_needed=true, leave primary_anchors empty, and set confidence < 0.5.
- Reserve confidence < 0.5 for genuinely ambiguous questions (no extractable clinical subject), NOT for medical-content uncertainty. The resolver does not need to be confident about CLINICAL ANSWERS — only about WHICH structured search terms a clinician would use.
- Leave list fields empty rather than guessing concrete values you cannot justify from the question text.

Phase J3 guidance for device-complication and therapy-in-setting shapes (extract dual primary_anchors and set the appropriate question_type):
- DEVICE-COMPLICATION RISK questions (e.g., "What is the risk of pacemaker migrations?", "ICD lead dislodgement", "central line infection", "complications of tracheostomy"): emit TWO primary_anchors — the device or procedure (entity_type=device_or_implant or procedure_or_surgery, role=population_problem) AND the complication / outcome (entity_type=complication_or_adverse_event, role=outcomes_target_diagnosis). Set type_of_question to "device_complication_risk" (devices) or "procedure_complication_risk" (procedures). Mark BOTH anchors required_for_search=true.
- THERAPY+SETTING questions (e.g., "magnesium replacement in ICU", "potassium replacement in critically ill patients"): emit TWO primary_anchors — the intervention (entity_type=management_intervention or drug_or_drug_class, role=intervention_exposure_index_test) AND the setting (entity_type=clinical_setting, role=setting). Set type_of_question to "therapy", "therapy_safety", "electrolyte_replacement", or "ICU_treatment_question" depending on the question intent. If the question implies an underlying problem (e.g., "magnesium replacement" implies hypomagnesemia), you MAY emit that problem as a THIRD primary_anchor with entity_type=disease_or_problem AND inferred=true. Mark intervention + setting required_for_search=true. Mark inferred anchors required_for_search=false unless you are very confident.
- Inferred anchors MUST carry inferred=true. Non-inferred anchors MUST carry inferred=false (or omit the field).
- clinical_setting and population_group alone are INSUFFICIENT to satisfy the clinical-subject requirement. If you can only identify a setting/population (e.g., "treatment in ICU?"), set clarification_needed=true and leave primary_anchors empty.

Confidence calibration (Phase J-tail — read this carefully):
"confidence" in this schema means ONLY:
    "confidence that this is a clinically valid question that can be safely mapped into search-ready PICOTT".

"confidence" does NOT mean any of the following:
- It does NOT mean confidence that all PICOTT fields are explicitly specified by the user. Missing comparator, unspecified outcome, unspecified time horizon, and an inferred clinical problem are NORMAL for real clinician questions and MUST NOT reduce confidence below 0.5 when the clinical structure is otherwise valid.
- It does NOT mean confidence that the clinical answer is known to you.
- It does NOT mean confidence that a named disease is explicitly stated. A question can be clinically valid with only a device + complication, or only an intervention + setting (no disease name) — see Shape A and Shape B below.
- It does NOT mean confidence that evidence will definitely be found in PubMed. The deterministic compiler decides retrieval; you only structure the question.

Use confidence in the 0.7-0.95 band when the question fits Shape A or Shape B below and you can emit the expected dual primary_anchors. Reserve confidence < 0.5 ONLY for genuinely ambiguous questions where no extractable clinical subject is present (e.g., "what is the treatment?" / "what are the recommendations?") OR where only a clinical_setting / population_group is present without an intervention or device.

Shape A — Therapy / intervention + setting (accept and emit dual anchors).
Trigger: the question contains BOTH
  (i) a treatment, replacement, supplementation, therapy, protocol, drug, or intervention; AND
  (ii) a clinical population or setting.
Teaching examples (NOT exact-string matches — generalize):
    "magnesium replacement in ICU patients"
    "potassium replacement in ICU patients"
    "electrolyte replacement in critically ill adults"
    "benefits and risks of magnesium replacement in patients in the ICU"
Expected PICOTT shape for Shape A:
    P: ICU patients / critically ill adults (entity_type=population_group OR clinical_setting)
    I: the intervention (entity_type=management_intervention OR drug_or_drug_class)
    C: leave empty if not specified — DO NOT lower confidence for missing C
    O: benefits / risks / adverse effects / clinical outcomes (leave empty if not specified — DO NOT lower confidence)
    T1/type_of_question: one of `therapy`, `therapy_safety`, `electrolyte_replacement`, `ICU_treatment_question`
    T2/ideal_study_type (rationale-only signal): RCTs, systematic reviews, ICU protocols / guidelines, observational evidence when RCT evidence limited
    T3/time/setting: ICU stay / critical care setting
If hypomagnesemia / hypokalemia / the deficiency is inferred from "replacement", emit it as a THIRD primary_anchor with entity_type=disease_or_problem AND inferred=true AND required_for_search=false. NEVER make an inferred deficiency a required anchor unless the user explicitly stated it.
Set confidence in the 0.7-0.85 band for Shape A.

Shape B — Device / procedure + complication / risk (accept and emit dual anchors).
Trigger: the question contains BOTH
  (i) a device, implant, procedure, surgery, or procedure context; AND
  (ii) a complication, risk, infection, adverse event, morbidity, dislodgement, migration, or post-procedure outcome term.
Teaching examples (NOT exact-string matches — generalize):
    "central line infection risk"
    "complications of tracheostomy"
    "complications after maze procedure"
    "ICD lead dislodgement"
    "pacemaker migration"
    "risk of central venous catheter infection"
Expected PICOTT shape for Shape B:
    P: patients with the device / procedure or post-procedure population (entity_type=population_group)
    I/exposure/context: the device, implant, procedure, or surgery (entity_type=device_or_implant OR procedure_or_surgery)
    C: leave empty if not specified — DO NOT lower confidence for missing C
    O: the complication / adverse event / risk domain (entity_type=complication_or_adverse_event)
    T1/type_of_question: one of `device_complication_risk` (device questions), `procedure_complication_risk` (procedure questions), `post_procedure_complication`, `etiology_harm`, or `prognosis` depending on wording
    T2/ideal_study_type (rationale-only signal): observational cohorts, registries, systematic reviews, guidelines, case series when complications are rare
    T3/time: post-procedure or not specified
Set confidence in the 0.7-0.9 band for Shape B.

Multi-intent and framework routing (Phase K1 additive guidance — when to populate multiple branches, when to use which framework, slot_search_policy guidance):
- A question can map to MORE THAN ONE clinical intent (e.g., "syncope: management AND don't-miss diagnoses?"). When multiple intents are clearly present, the orchestrator will populate multiple branches above PICOTT. From your perspective, prefer to (a) structure ONE primary PICOTT that captures the dominant intent and (b) leave room for secondary intents to be picked up downstream. NEVER fabricate a secondary intent that is not present in the question.
- Framework choice (PICOTS / PICOTT / PIRD / PIRT / PFO / PECOT / PECO / PEO / CHARMS / CoCoPop / PCC / problem_context_differential / concept_mechanism / guideline_stabilization) is determined by the dominant intent. You are NOT required to name the framework in your output — the orchestrator infers it from your type_of_question. Reserved frameworks (SPIDER / SPICE / ECLIPSE) are for qualitative / health-services questions and are NOT actively routed in this build.
- Deferred intent routing: questions that are purely about patient experience, qualitative themes, implementation science, health services, barriers / preferences, or policy MUST set clarification_needed=true with a brief clarification_message stating that this intent class is not actively routed in this build. NEVER auto-map a qualitative question to therapy / drug_safety / device_complication.
- slot_search_policy guidance (per concept emitted in must_have_concepts):
    * Use ``required_for_high_relevance=true`` ONLY for concepts that the deterministic boolean compiler MUST AND into every retrieved query (population/condition + intervention/device + complication when applicable). The orchestrator maps this to ``slot_search_policy=required_for_retrieval``.
    * Use ``required_for_high_relevance=false`` for concepts that should INFLUENCE RANKING but should NOT be required at retrieval time. The orchestrator maps this to ``slot_search_policy=optional_for_retrieval`` or ``ranking_only`` depending on confidence.
    * Concepts that are display-only context (e.g., setting labels you couldn't anchor to retrieval) MUST be left out of must_have_concepts and listed only as picott.setting / generic_modifiers. The orchestrator may flag them as ``slot_search_policy=display_only``.
- prompt_version: this prompt is tagged ``j-tail-v2-K1`` by the orchestrator. NEVER reference the version yourself.

Intent disambiguation rules (Phase K1.1 additive — closed-set, do NOT widen):
- Therapy should NOT be the default fallback. If neither (a) a clear therapy/intervention phrase (treat / treatment / first-line / second-line / dosing / treatment options / drug X vs drug Y) NOR (b) a clear comparator/treatment-vs-treatment phrase is present in the question, choose the closest non-therapy intent from the closed-set list (background_overview, epidemiology_prevalence_incidence, pathophysiology_mechanism, diagnostic_workup, diagnosis_accuracy, differential_diagnosis, prognosis_natural_history, prognostic_factor, prediction_model, risk_etiology_harm, complication_risk, monitoring_followup_surveillance, stabilization_emergency_management, guideline_recommendation, prevention_screening, drug_safety, therapy_safety, device_complication, procedure_complication, drug_complication). NEVER pick "therapy" simply because a drug or procedure is named — the intent is decided by what the question is ASKING about that drug or procedure.
- If the question contains BOTH a management / treatment / "how should I manage" phrase AND a "don't miss diagnoses" / "diagnoses should I not miss" / "differential" / "red flags" / "cannot miss" / "must rule out" phrase, treat it as MULTI-INTENT. In this case the primary type_of_question MUST be one of {diagnosis, therapy} reflecting the dominant management ask, and clarification_needed=false. The orchestrator will downstream-derive primary={stabilization_emergency_management OR diagnostic_workup} with secondary={differential_diagnosis}. Do NOT collapse multi-intent management+differential questions into a pure "therapy" interpretation.
- If the question contains "benefits and risks" / "safety and efficacy" / "adverse effects of treatment" / "side effects of <drug>" / "tolerability of <drug>" — choose type_of_question="therapy" with question_subtype="drug_safety" (i.e. orchestrator routes this to therapy_safety / drug_safety). Do NOT collapse it to "background".
- Epidemiology (prevalence / incidence / how common / how frequent / population-based estimate) MUST set type_of_question="epidemiology" — NEVER "background", and NEVER "therapy". The orchestrator maps this to CoCoPop.
- Pathophysiology / mechanism questions ("pathophysiology of X", "mechanism by which X causes Y") MUST set type_of_question="background" with question_subtype that signals mechanism (orchestrator routes to concept_mechanism framework).
- "What is X?" / "define X" / "overview of X" stays in type_of_question="background" with question_subtype="disease_overview" or "drug_overview". This is the ONLY case where "background" is the correct default for a clinical-entity question.

Non-negotiable clarification cases — these MUST still produce clarification_required=true and an empty primary_anchors list, even after the Shape A / Shape B guidance above:
- "What is the dose?"
- "What are the recommendations?"
- "What are the side effects?"
- "What is the treatment?"
- "What is the risk?"
- "What is the treatment in ICU?"  (clinical_setting alone is insufficient)
Rule: clinical_setting ALONE is insufficient. clinical_setting + intervention IS sufficient (Shape A). clinical_setting + device/procedure + complication IS sufficient (Shape B).

Output schema (Pydantic is the source of truth — every field listed here MUST be present in your JSON):
{
  "is_valid_clinical_question": bool,
  "question_scope": one of ["background", "foreground", "hybrid", "multi_intent"],
  "type_of_question": str,
  "question_subtype": str | null,
  "specialty": [str, ...],
  "picott": {
    "population_problem": [str, ...],
    "intervention_exposure_index_test": [str, ...],
    "comparator_reference_standard": [str, ...],
    "outcomes_target_diagnosis": [str, ...],
    "time_horizon": [str, ...],
    "setting": [str, ...],
    "clinical_action": [str, ...]
  },
  "primary_anchors": [
    {
      "text": str,
      "canonical_concept": str,
      "entity_type": one of ["disease_or_problem", "drug_or_drug_class", "vaccine", "diagnostic_test", "procedure_or_surgery", "device_or_implant", "management_intervention", "biomarker_or_lab", "exposure", "population_group", "clinical_action", "complication_or_adverse_event", "clinical_setting", "unknown"],
      "role": one of ["population_problem", "intervention_exposure_index_test", "comparator_reference_standard", "outcomes_target_diagnosis", "time_horizon", "setting", "clinical_action", "procedure_context"],
      "required_for_search": bool,
      "missing_relevance_cap": int (0-100),
      "inferred": bool (optional, default false)
    }, ...
  ],
  "generic_modifiers": [str, ...],
  "must_have_concepts": [
    {
      "concept": str,
      "role": one of the closed-set role values above,
      "required_for_high_relevance": bool,
      "missing_relevance_cap": int (0-100)
    }, ...
  ],
  "ranking_concepts": [str, ...],
  "negative_concepts": [str, ...],
  "clarification_needed": bool,
  "clarification_message": str | null,
  "refinement_note": str | null,
  "confidence": float (0.0-1.0),
  "rationale": str (max 200 chars)
}

Rules:
- If question_scope is "foreground" or "hybrid" you MUST emit ≥1 primary_anchor.
- If clarification_needed is true, leave primary_anchors empty.
- Keep every list short (≤ 8 items each). Prefer empty over guess.
- rationale MUST be ≤ 200 characters and MUST NOT cite articles or PMIDs.
- Output only the JSON object. No markdown fences. No prose."""


# ---------------------------------------------------------------------
# Mode helpers.
# ---------------------------------------------------------------------


def current_mode() -> ResolverMode:
    """Read the closed-set mode from env. Defaults to ``fallback_active``."""
    raw_enabled = (os.getenv("CLINICAL_CONTEXT_RESOLVER_ENABLED") or "true").strip().lower()
    if raw_enabled != "true":
        return "off"
    raw_mode = (os.getenv("CLINICAL_CONTEXT_RESOLVER_MODE") or "fallback_active").strip().lower()
    if raw_mode not in ("off", "shadow", "fallback_active"):
        return "fallback_active"
    return raw_mode  # type: ignore[return-value]


# ---------------------------------------------------------------------
# Scrubbing + validation.
# ---------------------------------------------------------------------


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def _scrub_raw_output(raw: dict) -> Optional[str]:
    """Pre-validation regex pass. Returns a closed-set rejection_reason
    on hit, ``None`` otherwise.

    Order matters — most specific rejection wins.
    """
    if not isinstance(raw, dict):
        return "schema_invalid"
    for s in _iter_strings(raw):
        if _PHI_EMAIL_RE.search(s) or _PHI_PHONE_RE.search(s) or _PHI_SSN_RE.search(s) or _PHI_MRN_RE.search(s) or _PHI_DATE_RE.search(s):
            return "contains_phi_pattern"
        if _UNAPPROVED_SOURCE_RE.search(s):
            return "references_unapproved_source"
        if _PMID_RE.search(s) or _DOI_RE.search(s):
            return "contains_pmid_doi_pattern"
        if _ARTICLE_REF_RE.search(s):
            return "contains_article_reference"
    # Bare 7+ digit run check — but only for string fields outside
    # ``rationale`` (which can legitimately mention "≥7 day course"
    # style numerics... actually no, even 7-digit numerics in a
    # search interpretation are suspect, so we check everywhere).
    for s in _iter_strings(raw):
        if _BARE_LONG_DIGITS_RE.search(s):
            return "contains_pmid_doi_pattern"
    # Long free-text in list fields that should be short concepts.
    for list_key in (
        "specialty", "generic_modifiers", "ranking_concepts",
        "negative_concepts",
    ):
        items = raw.get(list_key) or []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, str):
                    continue
                low = it.lower()
                for tok in _ANSWER_TEXT_INDICATORS:
                    if tok in low:
                        return "contains_answer_text"
    # Rationale length cap.
    rationale = raw.get("rationale")
    if isinstance(rationale, str) and len(rationale) > 200:
        return "contains_answer_text"
    return None


def validate_output(raw: Any) -> tuple[Optional[ClinicalContextResolverOutput], Optional[str]]:
    """Closed-set schema validation. Returns ``(output, None)`` on
    success or ``(None, rejection_reason)`` on any failure.
    """
    if raw is None:
        return None, "empty_output"
    if not isinstance(raw, dict):
        return None, "schema_invalid"
    # Defensive scrub BEFORE Pydantic so a malicious string never
    # touches a list-of-validated-strings.
    scrub_hit = _scrub_raw_output(raw)
    if scrub_hit is not None:
        return None, scrub_hit
    # Strict Pydantic parse.
    try:
        parsed = ClinicalContextResolverOutput(**raw)
    except ValidationError:
        return None, "schema_invalid"
    except Exception:  # noqa: BLE001
        return None, "schema_invalid"
    # Post-parse confidence floor.
    if parsed.confidence < CONFIDENCE_THRESHOLD:
        return None, "confidence_below_threshold"
    # Foreground / hybrid scope MUST carry ≥1 primary anchor.
    if parsed.question_scope in ("foreground", "hybrid"):
        if not parsed.primary_anchors and not parsed.clarification_needed:
            return None, "missing_primary_anchor_for_foreground"
    # Empty output guard.
    if (
        not parsed.primary_anchors
        and not parsed.clarification_needed
        and not parsed.must_have_concepts
    ):
        return None, "empty_output"
    return parsed, None


# ---------------------------------------------------------------------
# Service.
# ---------------------------------------------------------------------


class ClinicalContextResolverService:
    """Stateless bounded resolver. Wraps the existing
    :class:`LLMProvider` so token diagnostics flow through the
    Phase 6F.2 ``record_llm_call`` path.
    """

    SYSTEM_PROMPT = SYSTEM_PROMPT  # exposed for tests / audit hashing

    async def resolve_clinical_context(
        self,
        *,
        original_question: str,
        normalized_question: str,
        deterministic_picott: dict | None,
        primary_anchors_detected: list,
        generic_modifiers_detected: list,
        anchor_failure_reason: str | None,
        existing_concept_translations: list,
        specialty_hints: list,
        diagnostics: Optional[SessionDiagnostics] = None,
    ) -> ResolverDecision:
        """Run the bounded LLM resolver. Returns a closed-set
        ``ResolverDecision``. NEVER raises.

        ``diagnostics`` is optional — when omitted the function reads
        the contextvar (``diag_current()``). The ``clinical_context_resolver``
        stage timer is opened internally, so callers don't need to
        wrap the call themselves.
        """
        mode = current_mode()
        decision = ResolverDecision(accepted=False, mode=mode)
        if mode == "off":
            decision.rejection_reason = None
            return decision

        diag = diagnostics or diag_current()
        # Build the user payload from CLOSED-SET inputs only. NO PHI:
        # we pass the normalized (post-spelling, post-shorthand)
        # question and a closed-set deterministic-failure context.
        # Original question is also passed — but only inside the LLM
        # request body. It is NEVER persisted anywhere downstream of
        # this function.
        user_payload = _build_user_payload(
            normalized_question=normalized_question or original_question or "",
            anchor_failure_reason=anchor_failure_reason,
            generic_modifiers_detected=generic_modifiers_detected,
            specialty_hints=specialty_hints,
        )

        t0 = time.perf_counter()
        before = diag.to_dict() if diag is not None else None
        raw: Optional[dict] = None
        # Phase J-tail Step 2 — optional resolver-only model override
        # via the env var ``CLINICAL_CONTEXT_RESOLVER_MODEL``. If set,
        # ONLY the resolver path uses this model; the summarizer /
        # verifier paths continue to use ``SUMMARY_MODEL`` /
        # ``VERIFIER_MODEL``. Used for the dev/staging model-gateway
        # comparison experiment ONLY. Default is None (no override) so
        # production behavior is unchanged when the env var is absent.
        _resolver_model_override = os.environ.get(
            "CLINICAL_CONTEXT_RESOLVER_MODEL", "",
        ).strip() or None
        try:
            if diag is not None:
                with diag.stage("clinical_context_resolver"):
                    raw = await llm_provider.complete_json(
                        system=SYSTEM_PROMPT,
                        user=user_payload,
                        # Pass no schema_hint — the system prompt
                        # already documents the schema, and we want
                        # zero variability from schema_hint phrasing.
                        temperature=0.0,
                        max_tokens=1500,
                        model=_resolver_model_override,
                    )
            else:
                raw = await llm_provider.complete_json(
                    system=SYSTEM_PROMPT,
                    user=user_payload,
                    temperature=0.0,
                    max_tokens=1500,
                    model=_resolver_model_override,
                )
        except Exception:  # noqa: BLE001
            logger.exception("clinical_context_resolver llm call failed (non-fatal)")
            raw = None
        latency_ms = int(round((time.perf_counter() - t0) * 1000))
        decision.latency_ms = latency_ms
        if diag is not None and before is not None:
            after = diag.to_dict()
            te = (after.get("token_estimate") or {})
            te_before = (before.get("token_estimate") or {})
            decision.token_estimate = {
                "prompt_tokens": int(te.get("prompt_tokens", 0))
                                  - int(te_before.get("prompt_tokens", 0)),
                "completion_tokens": int(te.get("completion_tokens", 0))
                                      - int(te_before.get("completion_tokens", 0)),
                "total_tokens": int(te.get("total_tokens", 0))
                                 - int(te_before.get("total_tokens", 0)),
                "estimate_method": te.get("estimate_method"),
            }

        output, rejection = validate_output(raw)
        if output is None:
            decision.rejection_reason = rejection
            return decision
        decision.output = output
        decision.accepted = True
        return decision


clinical_context_resolver = ClinicalContextResolverService()


# ---------------------------------------------------------------------
# Closed-set user payload builder. NO PHI.
# ---------------------------------------------------------------------


def _build_user_payload(
    *,
    normalized_question: str,
    anchor_failure_reason: Optional[str],
    generic_modifiers_detected: list,
    specialty_hints: list,
) -> str:
    """Build the resolver user prompt. Contains ONLY the normalized
    question and the closed-set anchor-failure context. Generic
    modifiers and specialty hints are NOT echoed back to the LLM —
    empirically (Phase H1.1 live acceptance), seeing them in the
    payload causes the model to over-weight "this question is
    generic" and reject otherwise-confident clinical entities. The
    deterministic-anchor failure reason alone is enough context for
    the LLM to know it's running as a fallback.

    Builder contract: NEVER embed article text, evidence packet
    content, or any prior LLM response. NEVER pass PHI patterns
    (caller is responsible for ensuring PHI guard has already run).
    """
    fr = (anchor_failure_reason or "no_primary_clinical_entity").strip() or "no_primary_clinical_entity"
    # generic_modifiers_detected / specialty_hints accepted in the
    # signature for backwards-compatibility but intentionally NOT
    # included in the user payload. See docstring.
    _ = generic_modifiers_detected
    _ = specialty_hints
    return (
        f"normalized_question: {normalized_question}\n"
        f"deterministic_anchor_failure_reason: {fr}\n"
        "\n"
        "Extract a clinical search interpretation per the schema above. "
        "Return ONLY the JSON object."
    )


__all__ = [
    "ClinicalContextResolverService",
    "ClinicalContextResolverOutput",
    "ResolverDecision",
    "ResolverPrimaryAnchor",
    "ResolverMustHaveConcept",
    "ResolverPicottFields",
    "REJECTION_REASONS",
    "SKIPPED_REASONS",
    "CONFIDENCE_THRESHOLD",
    "SYSTEM_PROMPT",
    "clinical_context_resolver",
    "current_mode",
    "validate_output",
]
