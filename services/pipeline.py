"""Pipeline orchestrator — multi-source Governed Agentic RAG (Phase 2B).

Status machine: searching → ranking → awaiting_selection (terminal) | error.

Multi-source flow (post-PHI screen):
1. parse → 2. plan (per enabled source) → 3. retrieve in PARALLEL via asyncio.gather
   → 4. normalize (Europe PMC raw records only) → 5. dedupe across sources
   → 6. OpenAlex enrichment (top-N, metadata only) → 7. safety filter
   → 8. rank (with multi-source confidence boost) → 9. persist + evaluate.

PHI is screened in the route handler BEFORE the pipeline is kicked off — the
pipeline NEVER sees raw PHI text.

No HTTP call to any third-party occurs unless the corresponding
``source_configs`` row is ``enabled=true`` (with an exception for the
authoritative Europe PMC source which remains enabled by default).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from core.db import Collections, get_db
from services.audit import audit_service
from services.clinical_context_engine import build_clinical_context
from services.clinical_question_parser import clinical_question_parser
from services.concept_block_query_builder import concept_block_query_builder
from services import ai_query_expansion
from services.query_compiler import boolean_query_compiler
from services.connectors.openalex import openalex_connector
from services.deduplication import deduplication
from services.default_selection import (
    select_default_articles,
    recount_summary_after_rescue,
    _exclusion_reason_for,
)
from services.evidence_ranker import evidence_ranker
from services import evidence_lens
from services.full_text_access_broker import full_text_access_broker
from services.literature_retrieval import literature_retrieval
from services.metadata_normalizer import metadata_normalizer
from services.picott_ranker import picott_aware_ranker
from services.picott_term_mapping_service import build_picott_term_mapping
from services.primary_clinical_anchor import primary_clinical_anchor_service
from services.diagnostics import SessionDiagnostics, bind as diag_bind, reset as diag_reset
from services.clinical_question_map import (
    PROMPT_VERSION_K1 as _CQM_PROMPT_VERSION,
    SCHEMA_VERSION as _CQM_SCHEMA_VERSION,
    ACTIVE_INTENTS as _CQM_ACTIVE_INTENTS,
    DEFERRED_INTENTS as _CQM_DEFERRED_INTENTS,
    ACTIVE_FRAMEWORKS as _CQM_ACTIVE_FRAMEWORKS,
    Branch as _CQMBranch,
    ClinicalQuestionMap as _CQM,
    DeterministicSignalSummary as _CQMSignalSummary,
    InferredConcept as _CQMInferredConcept,
    Slot as _CQMSlot,
    normalize_slot_semantics as _cqm_normalize_slot_semantics,
)
from services.branch_aware_boolean_query_planner import (
    plan_branch_queries as _k2_plan_branch_queries,
)
from services.article_lane_classifier import (
    classify_articles as _k3a_classify_articles,
    _FORBIDDEN_DIAG_SUBSTRINGS as _K3A_FORBIDDEN_DIAG_SUBSTRINGS,
)
from services.clinical_signal_router import clinical_signal_router as _cqm_signal_router
from services.retrieval_quality_evaluator import retrieval_quality_evaluator
from services.safety_filter import safety_filter
from services.scholar_like_expansion import expansion_mode, scholar_like_expansion
from services.search_planning import search_planning
from services import recall_supplement
from services import anchor_dedup
from services.semantic_reranker import semantic_reranker

logger = logging.getLogger("clinevidence.pipeline")

# Keep references so tasks don't get GC'd.
_RUNNING: set[asyncio.Task] = set()

# Sources that the pipeline knows how to query. New connectors are added here.
_KNOWN_RETRIEVAL_SOURCES = ("europe_pmc", "pubmed")

# ---- Phase M4.2 — Legacy semantic-repair consolidation. ----
# The M0/M1/M2 ``semantic_anchor_repair`` shadow, the M3-preflight
# eligibility/projection, and the M3.2 active semantic-repair decision are
# DEPRECATED and superseded by the M4 full-question ClinicalQuestionMap
# interpreter, which is now the forward semantic path for degraded
# baselines (preview/dev). Their pipeline runtime hooks — including the
# duplicate bounded-LLM call and the legacy ``m3_2_session_decisions``
# sidecar writes — are gated OFF behind this single constant so the
# consolidation is reversible. The legacy service modules remain on disk
# for the read-only admin/benchmark surfaces and the service-level unit
# tests. M3.2 stays dormant + production-hard-off; M4 owns the behavior.
_LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED = False


# When a source row is missing or disabled, Europe PMC is always allowed since
# it is the authoritative open-access baseline. Other sources require explicit
# opt-in via the admin UI.
_DEFAULT_ENABLED = {"europe_pmc"}


async def _set_status(
    session_id: str,
    *,
    status: str,
    extras: Optional[dict[str, Any]] = None,
) -> None:
    db = get_db()
    update = {"status": status, "updated_at": datetime.now(timezone.utc)}
    if extras:
        update.update(extras)
    await db[Collections.SEARCH_SESSIONS].update_one({"id": session_id}, {"$set": update})


def _topic_label_from(normalized: str) -> str:
    s = (normalized or "").strip()
    return s[:80]


# ---------------------------------------------------------------------
# Phase K1.I — ClinicalQuestionMap builder.
#
# Pure / sync / no I/O. Builds a closed-set ``ClinicalQuestionMap`` from
# what the existing pipeline already produced (deterministic anchor
# output, picott payload, resolver decision, deterministic signal
# router output). The map is PERSISTED ADDITIVELY on the session
# document under ``clinical_question_map_json``; nothing downstream of
# the boolean compiler reads from it in K1.
#
# Hard rules:
#   * Returns a JSON-safe dict via ``model_dump(mode="json")``.
#   * Never raises — any structural mismatch is caught and a closed-set
#     ``clarification_required`` placeholder is returned so the
#     pipeline cannot fail because of K1 instrumentation.
#   * NO PHI, NO prompts, NO completions in the persisted dict (only
#     closed-set fields).
# ---------------------------------------------------------------------


def _k1_intent_for_picott(
    picott: dict[str, Any], anchors_out: dict[str, Any],
) -> str:
    """Map the existing closed-set PICOTT ``type_of_question`` /
    ``question_subtype`` + anchor entity-types onto the K1
    ``ClinicalIntent`` Literal. Falls back to ``background_overview``
    only as a last resort; never invents an intent.
    """
    subtype = (picott.get("question_subtype") or "").strip().lower()
    toq = (picott.get("type_of_question") or "").strip().lower()
    anchors = anchors_out.get("primary_anchors") or []
    entity_types = {
        (a.get("entity_type") or "").lower() for a in anchors
        if isinstance(a, dict)
    }

    # Most specific closed-set first.
    if subtype == "device_complication_risk":
        return "device_complication"
    if subtype == "procedure_complication_risk":
        return "procedure_complication"
    if subtype in {"drug_safety", "adverse_effects"}:
        return "drug_safety"
    if subtype in {"diagnostic_accuracy", "accuracy"}:
        return "diagnosis_accuracy"
    if subtype in {"differential_diagnosis", "diagnosis_workup"}:
        return "differential_diagnosis"
    if subtype in {"diagnostic_workup"}:
        return "diagnostic_workup"
    if subtype in {"prognosis_natural_history"}:
        return "prognosis_natural_history"
    if subtype in {"prognostic_factor"}:
        return "prognostic_factor"
    if subtype in {"prediction_model"}:
        return "prediction_model"
    if subtype in {"prevention_screening"}:
        return "prevention_screening"
    if subtype in {"guideline_recommendation"}:
        return "guideline_recommendation"
    if subtype in {"post_procedure_management"}:
        return "therapy_safety"
    if subtype in {"electrolyte_replacement", "icu_treatment_question"}:
        return "therapy_safety"
    if subtype in {"disease_overview", "drug_overview", "definition"}:
        return "background_overview"
    # PICOTT-level routing.
    if toq == "therapy":
        return "therapy"
    if toq == "diagnosis":
        return "diagnosis_accuracy"
    if toq == "prognosis":
        return "prognosis_natural_history"
    if toq in {"etiology_harm"}:
        if "complication_or_adverse_event" in entity_types:
            if "device_or_implant" in entity_types:
                return "device_complication"
            if "procedure_or_surgery" in entity_types or "procedure" in entity_types:
                return "procedure_complication"
            if "drug_or_drug_class" in entity_types:
                return "drug_complication"
            return "complication_risk"
        return "risk_etiology_harm"
    if toq == "prevention":
        return "prevention_screening"
    if toq == "epidemiology":
        return "epidemiology_prevalence_incidence"
    if toq == "guideline":
        return "guideline_recommendation"
    if toq == "background":
        return "background_overview"
    if toq == "pharmacology":
        return "drug_safety"
    if toq == "dosing_protocol":
        return "therapy"
    if toq == "clinical_exam":
        return "diagnostic_workup"
    # Last resort.
    return "background_overview"


def _k1_framework_for_intent(intent: str) -> str:
    """Closed-set intent → primary framework. Returns one of the
    ``ACTIVE_FRAMEWORKS``. Falls back to ``PICOTT``.
    """
    if intent == "therapy":
        return "PICOTT"
    if intent == "therapy_safety":
        return "PICOTT"
    if intent == "drug_safety":
        return "PECO"
    if intent in {
        "drug_complication", "device_complication",
        "procedure_complication", "complication_risk",
        "disease_complication",
    }:
        return "PECO"
    if intent == "risk_etiology_harm":
        return "PECO"
    if intent in {"diagnosis_accuracy", "diagnostic_workup"}:
        return "PIRD"
    if intent == "differential_diagnosis":
        return "problem_context_differential"
    if intent in {"prognosis_natural_history", "prognostic_factor"}:
        return "PFO"
    if intent == "prediction_model":
        return "CHARMS"
    if intent == "epidemiology_prevalence_incidence":
        return "CoCoPop"
    if intent == "pathophysiology_mechanism":
        return "concept_mechanism"
    if intent == "monitoring_followup_surveillance":
        return "PICOTS"
    if intent == "stabilization_emergency_management":
        return "guideline_stabilization"
    if intent == "guideline_recommendation":
        return "guideline_stabilization"
    if intent == "background_overview":
        return "concept_mechanism"
    if intent == "prevention_screening":
        return "PICOTS"
    return "PICOTT"


# Map K1 SlotRole based on anchor's existing closed-set ``entity_type``
# + ``role`` tags. Closed-set table.
_K1_ENTITY_TYPE_TO_SLOT_ROLE: dict[str, str] = {
    "disease_or_problem": "disease_or_condition",
    "drug_or_drug_class": "drug_or_drug_class",
    "vaccine": "drug_or_drug_class",
    "diagnostic_test": "index_test",
    "procedure": "procedure_or_surgery",
    "procedure_or_surgery": "procedure_or_surgery",
    "device_or_implant": "device_or_implant",
    "perioperative_context": "clinical_setting",
    "postprocedure_context": "clinical_setting",
    "management_intervention": "intervention",
    "exposure": "exposure",
    "biomarker_or_lab": "outcome",
    "population_group": "population",
    "clinical_action": "guideline_context",
    "complication_or_adverse_event": "complication_or_adverse_event",
    "clinical_setting": "clinical_setting",
    "unknown": "clinical_problem",
}


def _k1_anchor_to_slot(
    anchor: dict[str, Any],
    *,
    inferred: bool = False,
) -> Optional[dict[str, Any]]:
    """Translate a primary-anchor dict (Phase J3 schema) into the
    closed-set K1 ``Slot`` dict. Returns ``None`` on missing canonical.
    """
    canonical = (anchor.get("canonical_concept") or "").strip()
    if not canonical:
        return None
    original = (anchor.get("text") or canonical).strip()
    raw_et = (anchor.get("entity_type") or "unknown").lower()
    slot_role = _K1_ENTITY_TYPE_TO_SLOT_ROLE.get(raw_et, "clinical_problem")
    required = bool(anchor.get("required_for_search"))
    policy = (
        "required_for_retrieval" if required else "optional_for_retrieval"
    )
    source_tag = anchor.get("source") or "unknown"
    if source_tag == "llm_resolver_fallback":
        slot_source = "bounded_resolver_fallback"
    elif source_tag in {"ontology_pack", "drug_pack", "rxnorm_style", "mesh"}:
        slot_source = "deterministic_concept_pack"
    elif source_tag in {"rule"}:
        slot_source = "deterministic_parser"
    elif source_tag in {"llm_repair"}:
        slot_source = "bounded_resolver_fallback"
    else:
        slot_source = "unknown"
    return {
        "original_phrase": original,
        "canonical_concept": canonical,
        "entity_type": raw_et if raw_et in {
            "disease_or_problem", "drug_or_drug_class", "vaccine",
            "diagnostic_test", "procedure", "procedure_or_surgery",
            "device_or_implant", "perioperative_context",
            "postprocedure_context", "management_intervention",
            "exposure", "biomarker_or_lab", "population_group",
            "clinical_action", "complication_or_adverse_event",
            "clinical_setting", "unknown",
        } else "unknown",
        "slot_role": slot_role,
        "required_for_search": required,
        "slot_search_policy": policy,
        "inferred": bool(anchor.get("inferred") or inferred),
        "confidence": float(anchor.get("confidence") or 0.7),
        "synonyms": [
            s for s in (anchor.get("synonyms") or []) if isinstance(s, str)
        ],
        "search_terms": [],
        "source": slot_source,
    }


def _k1_build_clinical_question_map(
    *,
    normalized_question: str,
    original_question: str,
    picott_payload: dict[str, Any],
    anchors_out: dict[str, Any],
    resolver_diag: dict[str, Any],
    signal_summary: dict[str, Any],
    clarification_only: bool = False,
) -> dict[str, Any]:
    """Build a closed-set ``ClinicalQuestionMap`` JSON dict from the
    existing pipeline artifacts. Never raises; returns a
    ``clarification_required`` placeholder on any structural mismatch.
    """
    try:
        # Interpretation provenance — closed-set.
        anchor_failure = anchors_out.get("anchor_failure_reason")
        resolver_called = bool(resolver_diag.get("called"))
        resolver_accepted = bool(resolver_diag.get("accepted"))
        primary_anchors = anchors_out.get("primary_anchors") or []
        # Provenance rules:
        #   - clarification_required when no primary anchor or
        #     clarification_only flag is set
        #   - bounded_resolver_fallback when resolver was called AND
        #     accepted AND its anchors landed in primary_anchors
        #   - deterministic_concept_pack when any anchor.source ==
        #     "ontology_pack"/"drug_pack"/"mesh"/"rxnorm_style"
        #   - deterministic_parser otherwise
        if clarification_only or not primary_anchors or anchor_failure == "no_primary_clinical_entity":
            provenance = "clarification_required"
        elif resolver_called and resolver_accepted and any(
            (a.get("source") or "") == "llm_resolver_fallback"
            for a in primary_anchors if isinstance(a, dict)
        ):
            provenance = "bounded_resolver_fallback"
        elif any(
            (a.get("source") or "") in {
                "ontology_pack", "drug_pack", "mesh", "rxnorm_style",
            }
            for a in primary_anchors if isinstance(a, dict)
        ):
            provenance = "deterministic_concept_pack"
        else:
            provenance = "deterministic_parser"

        intent = _k1_intent_for_picott(picott_payload, anchors_out)

        # Phase K1.1 — apply the signal router's lexical signal as an
        # OVERRIDE when the picott-derived intent is a weak fallback.
        # The picott extractor / "What is X?" repair often defaults to
        # ``background_overview`` for any non-therapy/non-diagnosis
        # question; we let the deterministic signal router promote it
        # to the more specific closed-set intent when a strong lexical
        # signal is present. Rules (closed-set, ordered by specificity):
        #   1. Multi-intent management + differential: when the router
        #      detected BOTH a management signal (therapy /
        #      diagnostic_workup / stabilization) AND
        #      differential_diagnosis AND ``possible_multi_intent`` is
        #      True, the PRIMARY stays on the management side (prefer
        #      stabilization → diagnostic_workup → therapy in that
        #      order). The secondary-intents emission later picks up
        #      differential_diagnosis.
        #   2. If the router detected epidemiology / pathophysiology /
        #      prevention_screening / monitoring / stabilization /
        #      guideline / diagnosis_* / risk_* / complication tokens
        #      AND the picott intent is ``background_overview`` or
        #      ``therapy`` (the two known fallbacks), override to the
        #      most-specific router intent.
        signal_intents: list[str] = [
            i for i in (signal_summary or {}).get("possible_intents", [])
            if isinstance(i, str) and i in _CQM_ACTIVE_INTENTS
        ]
        possible_multi = bool((signal_summary or {}).get("possible_multi_intent"))
        _STRONG_OVERRIDE_INTENTS = {
            "epidemiology_prevalence_incidence",
            "pathophysiology_mechanism",
            "prevention_screening",
            "monitoring_followup_surveillance",
            "stabilization_emergency_management",
            "guideline_recommendation",
            "diagnosis_accuracy",
            "diagnostic_workup",
            "differential_diagnosis",
            "prognosis_natural_history",
            "prognostic_factor",
            "prediction_model",
            "risk_etiology_harm",
            "complication_risk",
            "device_complication",
            "procedure_complication",
            "drug_complication",
            "therapy_safety",
            "drug_safety",
        }
        _FALLBACK_INTENTS = {"background_overview", "therapy"}
        _MANAGEMENT_PRIMARY_PREFERENCE = (
            "stabilization_emergency_management",
            "diagnostic_workup",
            "diagnosis_accuracy",
        )
        signal_intent_set = set(signal_intents)
        # Rule 1 — multi-intent management + differential.
        if (
            possible_multi
            and "differential_diagnosis" in signal_intent_set
            and intent in _FALLBACK_INTENTS
        ):
            promoted = False
            for mgmt in _MANAGEMENT_PRIMARY_PREFERENCE:
                if mgmt in signal_intent_set:
                    intent = mgmt
                    promoted = True
                    break
            if not promoted and intent == "background_overview":
                # No explicit management signal but a differential
                # signal IS present — promote to diagnostic_workup so
                # the primary intent is still actionable.
                intent = "diagnostic_workup"
        # Rule 2 — single-signal fallback promotion.
        else:
            for cand in signal_intents:
                if cand in _STRONG_OVERRIDE_INTENTS and intent in _FALLBACK_INTENTS:
                    intent = cand
                    break

        # Phase K1.1 — multi-intent secondary-intent emission. When the
        # signal router detected ≥2 distinct active intents AND
        # ``possible_multi_intent`` is True (closed-set conjunction
        # signal), promote the second-strongest router intent to
        # ``secondary_intents``. Cap at 2 secondary entries (3-branch
        # total cap is enforced by the schema).
        secondary_intents_out: list[str] = []
        if possible_multi and len(signal_intents) >= 2:
            for cand in signal_intents:
                if cand == intent:
                    continue
                if cand in _FALLBACK_INTENTS:
                    continue
                if cand in secondary_intents_out:
                    continue
                secondary_intents_out.append(cand)
                if len(secondary_intents_out) >= 2:
                    break

        # Deferred-intent safety net (defensive — closed-set guard).
        if intent in _CQM_DEFERRED_INTENTS:
            provenance = "clarification_required"
        framework = _k1_framework_for_intent(intent)

        # Slots from primary anchors. Required anchors land in
        # required_concepts; rest in optional_concepts.
        required_slots: list[dict[str, Any]] = []
        optional_slots: list[dict[str, Any]] = []
        for a in primary_anchors:
            if not isinstance(a, dict):
                continue
            slot = _k1_anchor_to_slot(a)
            if slot is None:
                continue
            if slot["required_for_search"]:
                required_slots.append(slot)
            else:
                optional_slots.append(slot)

        # Phase K1.1 — Bridge ``generic_modifiers`` (e.g., "adults",
        # "older adults", "ICU patients") into the CQM as display-only
        # slots. These are filtered OUT of primary_anchors by the
        # anchor service (they're not eligible primary anchors) but
        # they're still meaningful structure for the clinician, and
        # they unblock the "framework_neutral_slots is empty" failure
        # the tester reported on the AF prevalence question.
        display_only_slots: list[dict[str, Any]] = []
        seen_display = {
            (s.get("canonical_concept") or "").strip().lower()
            for s in (required_slots + optional_slots)
            if isinstance(s, dict)
        }
        for gm in (anchors_out.get("generic_modifiers") or []):
            phrase = str(gm or "").strip()
            if not phrase:
                continue
            low = phrase.lower()
            if low in seen_display:
                continue
            seen_display.add(low)
            # Closed-set slot_role inference for known generic modifiers.
            gm_role = "population"
            if low in {"older adults", "older adult", "elderly",
                       "pediatric", "paediatric", "pediatrics",
                       "neonates", "neonate", "infants", "infant",
                       "children", "child"}:
                gm_role = "age_group"
            elif low in {"icu patients", "ed patients",
                         "outpatient", "outpatients",
                         "inpatient", "inpatients"}:
                gm_role = "clinical_setting"
            elif low in {"male", "men", "female", "women"}:
                gm_role = "sex"
            display_only_slots.append({
                "original_phrase": phrase,
                "canonical_concept": phrase,
                "entity_type": (
                    "population_group" if gm_role in {"population", "age_group", "sex"}
                    else "clinical_setting" if gm_role == "clinical_setting"
                    else "unknown"
                ),
                "slot_role": gm_role,
                "required_for_search": False,
                "slot_search_policy": "display_only",
                "inferred": False,
                "confidence": 0.7,
                "synonyms": [],
                "search_terms": [],
                "source": "deterministic_parser",
            })

        # Phase K1.2 — slot semantics normalization (closed-set, no I/O).
        # Fix 1: split demographic qualifiers ("in adults", "in older
        # adults", "in children", "in pregnancy", ...) out of the
        # disease/intervention slots and emit them as separate
        # population/age_group/pregnancy slots with
        # ``slot_search_policy=optional_for_retrieval``.
        # Fix 2: filter generic modifiers ("benefits", "risks",
        # "outcomes", "complications" generic, "recommendations",
        # ...) out of any ``slot_role=population`` slot.
        required_slots, optional_slots, display_only_slots = (
            _cqm_normalize_slot_semantics(
                required_slots=required_slots,
                optional_slots=optional_slots,
                display_only_slots=display_only_slots,
            )
        )
        inferred_concepts: list[dict[str, Any]] = []
        for a in primary_anchors:
            if isinstance(a, dict) and bool(a.get("inferred")):
                c = (a.get("canonical_concept") or "").strip()
                if c:
                    inferred_concepts.append({
                        "concept": c,
                        "inferred_from": (a.get("text") or "").strip(),
                        "confidence": float(a.get("confidence") or 0.7),
                        "rationale": "resolver_inferred",
                    })

        # Candidate keywords + unvalidated MeSH come from picott payload.
        keywords: list[str] = []
        for mh in (picott_payload.get("must_have_concepts") or []):
            if isinstance(mh, dict):
                c = (mh.get("concept") or "").strip()
                if c and c not in keywords:
                    keywords.append(c)
        # Top-level confidence — best-effort.
        if clarification_only:
            confidence = 0.0
        elif resolver_accepted:
            conf = resolver_diag.get("resolver_confidence")
            confidence = float(conf or 0.7) if isinstance(conf, (int, float)) else 0.7
        elif primary_anchors:
            confidence = max(
                (float(a.get("confidence") or 0.0)
                 for a in primary_anchors if isinstance(a, dict)),
                default=0.7,
            )
        else:
            confidence = 0.0

        # Phase K1.1 — Primary branch + optional secondary branch.
        # Schema cap = 3 branches; we emit at most 2 here (primary +
        # one secondary derived from the strongest non-primary signal
        # router intent). The boolean compiler does NOT consume
        # branches yet (K2+); they exist purely for transparency.
        primary_branch: dict[str, Any] = {
            "branch_id": "primary_0",
            "intent": intent,
            "framework": framework,
            "branch_priority": 1,
            "branch_confidence": float(confidence),
            "branch_reason": f"provenance={provenance}",
            "required_concepts": required_slots,
            "optional_concepts": optional_slots,
            "ranking_only_concepts": [],
            "display_only_concepts": display_only_slots,
            "preferred_evidence_types": [],
            "retrieval_strategy_hint": "standard",
            "guideline_first": (intent == "guideline_recommendation"),
            "rare_or_complication_rescue": (
                intent in {
                    "device_complication", "procedure_complication",
                    "drug_complication", "complication_risk",
                }
            ),
        }
        branches_list: list[dict[str, Any]] = [primary_branch]
        # Secondary branch — only emit when the primary anchor list is
        # non-empty (i.e. we have a real interpretation to anchor a
        # secondary intent on) AND we detected ≥1 secondary intent.
        if required_slots and secondary_intents_out:
            for i, sec_intent in enumerate(secondary_intents_out, start=1):
                sec_framework = _k1_framework_for_intent(sec_intent)
                # Secondary branches reuse the primary anchors as
                # required_concepts but route them through a different
                # framework / intent for transparency purposes.
                branches_list.append({
                    "branch_id": f"secondary_{i}",
                    "intent": sec_intent,
                    "framework": sec_framework,
                    "branch_priority": min(i + 1, 3),
                    "branch_confidence": max(0.6, float(confidence) - 0.1),
                    "branch_reason": "signal_router_multi_intent",
                    "required_concepts": required_slots,
                    "optional_concepts": optional_slots,
                    "ranking_only_concepts": [],
                    "display_only_concepts": display_only_slots,
                    "preferred_evidence_types": [],
                    "retrieval_strategy_hint": "standard",
                    "guideline_first": (
                        sec_intent == "guideline_recommendation"
                    ),
                    "rare_or_complication_rescue": (
                        sec_intent in {
                            "device_complication", "procedure_complication",
                            "drug_complication", "complication_risk",
                        }
                    ),
                })
                if len(branches_list) >= 3:
                    break
        # When the primary anchor list is empty, drop branches
        # entirely so the schema's "branches==0 OR branches>=1 with
        # populated required_concepts" invariant holds.
        if not required_slots:
            branches_list = []

        cqm_dict: dict[str, Any] = {
            "original_question": original_question,
            "normalized_question": normalized_question,
            "interpretation_provenance": provenance,
            "clinical_validity": not clarification_only,
            "clarification_required": bool(
                clarification_only or provenance == "clarification_required"
            ),
            "clarification_reason": (
                "no_primary_clinical_entity"
                if (clarification_only or anchor_failure == "no_primary_clinical_entity")
                else None
            ),
            "primary_intent": intent,
            "secondary_intents": secondary_intents_out,
            "frameworks": [framework] + [
                _k1_framework_for_intent(si)
                for si in secondary_intents_out
                if _k1_framework_for_intent(si) != framework
            ],
            "branches": branches_list if (
                not clarification_only and required_slots
            ) else [],
            "framework_neutral_slots": (
                required_slots + optional_slots + display_only_slots
            ),
            "inferred_concepts": inferred_concepts,
            "candidate_keywords": keywords[:20],
            "candidate_MeSH_terms_unvalidated": [],
            "confidence": float(confidence),
            "schema_version": _CQM_SCHEMA_VERSION,
            "prompt_version": (
                _CQM_PROMPT_VERSION if (resolver_called and resolver_accepted)
                else None
            ),
            "deterministic_signal_summary": dict(signal_summary or {}),
        }
        # Validate against the Pydantic schema — guarantees closed-set
        # field shapes before persistence. On validation failure, fall
        # back to a clarification_required placeholder (never raises).
        try:
            cqm_validated = _CQM(**cqm_dict)
            persisted = cqm_validated.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseK1 ClinicalQuestionMap pydantic validation failed (non-fatal)"
            )
            return _k1_clarification_required_placeholder(
                normalized_question=normalized_question,
                original_question=original_question,
                signal_summary=signal_summary,
            )

        # Phase K2 — additive shadow ``branch_query_plan``. Reads the
        # closed-set env flag ``CLINICAL_QUESTION_MAP_BRANCH_QUERIES_MODE``
        # (default ``"shadow"``). NEVER runs in active mode — the
        # planner raises AssertionError if asked for active. The plan
        # is persisted alongside the CQM as a sibling key so the
        # validated K1 CQM schema (which has ``extra="forbid"``) is
        # NOT mutated.
        try:
            _k2_mode_raw = os.environ.get(
                "CLINICAL_QUESTION_MAP_BRANCH_QUERIES_MODE", "shadow",
            ).strip().lower()
            if _k2_mode_raw not in {"disabled", "shadow"}:
                _k2_mode_raw = "shadow"
            _k2_plan = _k2_plan_branch_queries(
                cqm_validated, mode=_k2_mode_raw,
            ).model_dump(mode="json")
            persisted["branch_query_plan"] = _k2_plan
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseK2 branch_query_plan build failed (non-fatal)"
            )
            # Defensive: never let K2 break K1 persistence.
            persisted["branch_query_plan"] = {
                "schema_version": "K2.1.0",
                "mode": "disabled",
                "active_for_retrieval": False,
                "generated_at": "",
                "branches": [],
            }
        return persisted
    except Exception:  # noqa: BLE001
        logger.exception(
            "phaseK1 _k1_build_clinical_question_map failed (non-fatal)"
        )
        return _k1_clarification_required_placeholder(
            normalized_question=normalized_question,
            original_question=original_question,
            signal_summary=signal_summary,
        )


def _k1_clarification_required_placeholder(
    *,
    normalized_question: str,
    original_question: str,
    signal_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Closed-set placeholder used whenever the map cannot be built
    confidently. Always passes ``ClinicalQuestionMap`` validation.
    """
    placeholder = {
        "original_question": original_question,
        "normalized_question": normalized_question,
        "interpretation_provenance": "clarification_required",
        "clinical_validity": False,
        "clarification_required": True,
        "clarification_reason": "no_primary_clinical_entity",
        "primary_intent": "background_overview",
        "secondary_intents": [],
        "frameworks": ["PICOTT"],
        "branches": [],
        "framework_neutral_slots": [],
        "inferred_concepts": [],
        "candidate_keywords": [],
        "candidate_MeSH_terms_unvalidated": [],
        "confidence": 0.0,
        "schema_version": _CQM_SCHEMA_VERSION,
        "prompt_version": None,
        "deterministic_signal_summary": dict(signal_summary or {}),
    }
    return _CQM(**placeholder).model_dump(mode="json")


async def _upsert_article(art: dict[str, Any]) -> str:
    """Upsert by canonical_key; return the article id (existing or new).

    Phase 6D — also check pmid/doi_normalized/pmcid uniqueness BEFORE
    insert. If a doc with one of those identifiers already exists under
    a DIFFERENT canonical_key, reuse it (the legacy collection had a
    long tail of pmid-only-vs-doi-only duplicates before Phase 6D dedup).
    Handles ``DuplicateKeyError`` as a last-resort fallback.

    Phase H2-tail Task A — author/contact-email redaction is enforced
    at THIS centralized persistence boundary so every connector
    (PubMed bypasses MetadataNormalizer; Europe PMC / OpenAlex /
    Guidelines all funnel through here) writes a clean record.
    Idempotent.
    """
    from pymongo.errors import DuplicateKeyError
    from services.article_email_redactor import redact_article_emails
    # Centralized email redaction at the persistence boundary. Idempotent
    # (running it twice is a no-op) so it is safe even when the article
    # has already been scrubbed at the normalizer step.
    art = redact_article_emails(art)
    db = get_db()
    ck = art["canonical_key"]
    existing = await db[Collections.ARTICLES].find_one(
        {"canonical_key": ck}, {"_id": 0, "id": 1},
    )
    if not existing:
        # Look up by identifier — different canonical_key but same record.
        idents: list[dict[str, Any]] = []
        if art.get("pmid"):
            idents.append({"pmid": art["pmid"]})
        if art.get("doi_normalized"):
            idents.append({"doi_normalized": art["doi_normalized"]})
        if art.get("pmcid"):
            idents.append({"pmcid": art["pmcid"]})
        if idents:
            existing = await db[Collections.ARTICLES].find_one(
                {"$or": idents}, {"_id": 0, "id": 1, "canonical_key": 1},
            )
    if existing:
        to_set = {
            k: v for k, v in art.items()
            if k not in {"id", "created_at", "canonical_key"}
            and v not in (None, [], {}, "")
        }
        to_set["updated_at"] = datetime.now(timezone.utc)
        await db[Collections.ARTICLES].update_one(
            {"id": existing["id"]}, {"$set": to_set},
        )
        return existing["id"]
    doc = dict(art)
    doc["created_at"] = doc.get("created_at") or datetime.now(timezone.utc)
    doc["updated_at"] = doc.get("updated_at") or datetime.now(timezone.utc)
    for k in ("created_at", "updated_at"):
        v = doc[k]
        if hasattr(v, "tzinfo") and v.tzinfo is None:
            doc[k] = v.replace(tzinfo=timezone.utc)
    try:
        await db[Collections.ARTICLES].insert_one(doc)
    except DuplicateKeyError:
        # Race: another concurrent upsert (same persist batch) or a prior
        # request inserted a row sharing one of our unique identifiers
        # (canonical_key / pmid / doi_normalized / pmcid) between our lookup and
        # our insert. Re-resolve to the winner's id. The winning insert IS
        # committed (that is what raised E11000) but may not be visible on our
        # very first read under concurrent load, so retry a few times with a
        # short backoff before giving up.
        ors: list[dict[str, Any]] = [{"canonical_key": ck}]
        if doc.get("pmid"):
            ors.append({"pmid": doc["pmid"]})
        if doc.get("doi_normalized"):
            ors.append({"doi_normalized": doc["doi_normalized"]})
        if doc.get("pmcid"):
            ors.append({"pmcid": doc["pmcid"]})
        for _attempt in range(6):
            race = await db[Collections.ARTICLES].find_one(
                {"$or": ors}, {"_id": 0, "id": 1},
            )
            if race:
                return race["id"]
            await asyncio.sleep(0.05 * (_attempt + 1))
        raise
    return doc["id"]


async def _enabled_sources() -> list[str]:
    """Read source_configs and return list of enabled retrieval sources.

    Europe PMC is always returned (default baseline). Other sources are
    returned only if their row has ``enabled=true``.
    """
    db = get_db()
    rows = await db[Collections.SOURCE_CONFIGS].find(
        {"source_name": {"$in": list(_KNOWN_RETRIEVAL_SOURCES)}},
        {"_id": 0, "source_name": 1, "enabled": 1},
    ).to_list(length=20)
    by_name = {r["source_name"]: bool(r.get("enabled")) for r in rows}
    enabled: list[str] = []
    for s in _KNOWN_RETRIEVAL_SOURCES:
        if by_name.get(s) or s in _DEFAULT_ENABLED:
            enabled.append(s)
    return enabled


async def _openalex_enabled() -> bool:
    db = get_db()
    row = await db[Collections.SOURCE_CONFIGS].find_one(
        {"source_name": "openalex"}, {"_id": 0, "enabled": 1}
    )
    return bool(row and row.get("enabled"))


async def run_pipeline(
    session_id: str,
    user_id: str,
    sanitized_question: str,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
) -> None:
    """Post-PHI pipeline. Runs as a background task. Never raises."""
    # Phase 6F.2 — bind a session-scoped SessionDiagnostics accumulator.
    # Every LLM call recorded via services/llm_provider.py and every
    # ``with diag.stage(...)`` invocation inside this coroutine will
    # populate this object. At the end of run_pipeline we flush it
    # onto search_sessions.session_diagnostics_json so the GET response
    # surfaces real values. Best-effort: any diagnostics failure must
    # NEVER break the pipeline.
    diag = SessionDiagnostics()
    diag_token = diag_bind(diag)
    # Phase H1 — function-scoped accumulator for ClinicalContextResolver
    # diagnostics. Persisted in the ``finally`` flush below alongside the
    # Phase 6F.2 diagnostics block. Closed-set strings + ints only.
    #
    # Defaults match the "resolver was not exercised on this request"
    # contract: ``called=False``, ``mode`` = the env-configured mode
    # (not the misleading "off"), ``skipped_reason=None`` (only set
    # when an explicit closed-set skip applies).
    from services.clinical_context_resolver import current_mode as _h1_mode
    resolver_diag: dict[str, Any] = {
        "called": False,
        "mode": _h1_mode(),
        "accepted": False,
        "skipped_reason": None,
        "rejection_reason": None,
        "latency_ms": 0,
        "token_estimate": {},
        "interpretation_source": "deterministic",
    }
    # Phase 6F.2 — seed phi_guard latency from the pre-pipeline write
    # in routers/search_sessions.py::create_session. This survives the
    # final flush below because we re-emit it alongside the other
    # stage timings.
    try:
        _seed_doc = await get_db()[Collections.SEARCH_SESSIONS].find_one(
            {"id": session_id},
            {"session_diagnostics_json.latency_by_stage_json": 1, "_id": 0},
        )
        _seed_phi = (
            ((_seed_doc or {})
             .get("session_diagnostics_json") or {})
            .get("latency_by_stage_json") or {}
        ).get("phi_guard")
        if isinstance(_seed_phi, int) and _seed_phi >= 0:
            diag.latency_by_stage["phi_guard"] = int(_seed_phi)
    except Exception:  # noqa: BLE001
        logger.exception("phase6f2 phi_guard seed failed (non-fatal)")
    try:
        # Phase 6F.2 clean-up — deterministic pre-PICOTT anchor
        # pre-flight. Runs spelling + shorthand normalization (closed-
        # set dictionaries, no LLM) and then the closed-set generic-
        # only check. If the question carries NO candidate clinical
        # noun, we short-circuit to ``awaiting_clinical_subject``
        # BEFORE the LLM PICOTT extractor fires. Closed-set audit
        # event; closed-set diagnostics stages.
        try:
            with diag.stage("normalization"):
                from services.spelling_normalizer import spelling_normalizer
                from services.clinical_shorthand_normalizer import (
                    clinical_shorthand_normalizer,
                )
                _norm_spell = spelling_normalizer.normalize(sanitized_question or "")
                _norm_short = clinical_shorthand_normalizer.normalize(
                    _norm_spell.normalized
                )
                normalized_for_preflight = _norm_short.normalized
        except Exception:  # noqa: BLE001
            logger.exception("phase6f2 normalization (pre-flight) failed (non-fatal)")
            normalized_for_preflight = sanitized_question or ""
        try:
            with diag.stage("primary_clinical_anchor"):
                is_generic_only = primary_clinical_anchor_service.is_generic_only(
                    normalized_for_preflight
                )
        except Exception:  # noqa: BLE001
            logger.exception("phase6f2 anchor pre-flight failed (non-fatal)")
            is_generic_only = False

        if is_generic_only:
            from services.primary_clinical_anchor import (
                CLARIFICATION_MESSAGE,
                _detect_generic_modifiers,
            )
            anchors_out = {
                "primary_anchors": [],
                "generic_modifiers": _detect_generic_modifiers(
                    normalized_for_preflight
                ),
                "anchor_repair_applied": False,
                "anchor_failure_reason": "no_primary_clinical_entity",
            }
            await _set_status(
                session_id,
                status="awaiting_clinical_subject",
                extras={
                    "primary_clinical_anchors_json": dict(anchors_out),
                    "clarification_message_text": CLARIFICATION_MESSAGE,
                    "topic_label": _topic_label_from(normalized_for_preflight),
                },
            )
            try:
                await audit_service.log(
                    "search_session_anchor_clarification_required",
                    user_id=user_id,
                    search_session_id=session_id,
                    payload={
                        # Closed-set fields ONLY. No question text.
                        "anchor_failure_reason": "no_primary_clinical_entity",
                        "generic_modifier_count": len(
                            anchors_out.get("generic_modifiers") or []
                        ),
                        "anchor_repair_applied": False,
                        "phase": "phase6f2_preflight",
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "phase6f2 anchor_clarification (preflight) audit failed (non-fatal)"
                )
            # Terminate the pipeline here. PICOTT extractor DOES NOT
            # fire. No PubMed, no Europe PMC, no OpenAlex, no ontology
            # mapping, no scholar-like expansion. NO LLM calls.
            # Phase H1 — closed-set skip reason for the diagnostics
            # block. The resolver was not invoked because the
            # deterministic generic-only fast-path won.
            resolver_diag["skipped_reason"] = "generic_only_fast_path_fired"
            return

        with diag.stage("picott_parsing"):
            await audit_service.log("search_stage_parse_start", user_id=user_id, search_session_id=session_id)
            parsed = await clinical_question_parser.parse(sanitized_question)
            # AI Agent 1 — query categorizer (advisory; default OFF). Refines a
            # weak deterministic type ("other") before downstream type-aware
            # logic so e.g. a drug question is no longer mis-routed. Never raises.
            try:
                from agents.query_categorizer import maybe_categorize
                await maybe_categorize(session_id, user_id, sanitized_question, parsed)
            except Exception:  # noqa: BLE001
                logger.exception("ai_query_categorizer hook failed (non-fatal)")
            topic_label = _topic_label_from(parsed.get("normalized_question") or sanitized_question)
            picott_payload = parsed.get("picott_json") or {}

        # Phase 5F — Clinical Context Engine. Runs AFTER the LLM PICOTT
        # extractor and BEFORE the Boolean query compiler. Performs:
        #   1) ontology mapping over every PICOTT concept
        #   2) role repair (ontology evidence overrides mistagged LLM roles)
        #   3) role weighting by question_type/subtype
        #   4) multi-intent detection
        # The repaired PICOTT replaces ``picott_payload`` so all downstream
        # consumers (compiler, ranker, display normalizer) see the repaired
        # version. Never raises.
        try:
            context = build_clinical_context(parsed)
        except Exception:  # noqa: BLE001
            logger.exception("phase5f clinical_context_engine failed (non-fatal)")
            context = {
                "shorthand_substitutions": list(parsed.get("shorthand_substitutions") or []),
                "ontology_mappings": [], "role_repairs": [],
                "role_weights": {"type_of_question": "other",
                                  "question_subtype": None, "weights": {}},
                "multi_intent": {"is_multi_intent": False, "sub_questions": [], "rationale": ""},
                "repaired_picott": picott_payload,
            }
        # Replace the picott payload with the repaired version so the
        # compiler / ranker / display normalizer all use the same truth.
        if context.get("repaired_picott"):
            picott_payload = context["repaired_picott"]
            parsed["picott_json"] = picott_payload

        # AI Agent 2 — question understanding / ContextAugment (B3). Stores
        # interpretation diagnostics; never feeds the compiler/ranker/selection.
        # Default OFF, never raises.
        try:
            from agents.question_interpreter import maybe_interpret
            await maybe_interpret(session_id, user_id, sanitized_question, parsed, picott_payload)
        except Exception:  # noqa: BLE001
            logger.exception("ai_question_understanding hook failed (non-fatal)")

        # Phase K4A.2 — severity-marker overview deterministic override.
        # Runs AFTER repaired_picott so the compiler, ranker, and
        # default_selection all see the same question_subtype +
        # synonym-augmented outcome must-have. Closed-set, deterministic.
        # NO LLM, NO network.
        try:
            from services.severity_marker_overview import (
                is_severity_marker_question,
                extract_condition_anchor,
                severity_or_block_synonyms,
            )
            _q = sanitized_question or ""
            if is_severity_marker_question(_q):
                picott_payload["question_subtype"] = "severity_marker_overview"
                _anchor = extract_condition_anchor(_q)
                if _anchor:
                    picott_payload["severity_marker_condition_anchor"] = _anchor
                # Augment outcome must-haves with the closed-set OR-block
                # synonyms so the query compiler does not hard-AND a
                # single literal phrase like ``"severity of illness"``.
                _orblock = severity_or_block_synonyms()
                _has_condition_role = False
                _anchor_low = (_anchor or "").lower()
                for _mh in picott_payload.get("must_have_concepts") or []:
                    if not isinstance(_mh, dict):
                        continue
                    _role = (_mh.get("role") or "").lower()
                    if _role in {"condition", "population", "patient_group"}:
                        _has_condition_role = True
                        # Repair: if the condition concept does not
                        # contain the deterministic anchor (e.g., the
                        # PICOTT extractor glued the whole question into
                        # one "condition"), add the anchor as a synonym
                        # so the ranker / anchor service can match the
                        # canonical disease. We do NOT replace the
                        # original concept — only add the anchor as a
                        # synonym.
                        _concept_low = (_mh.get("concept") or "").lower()
                        if _anchor and _anchor_low not in _concept_low:
                            _syns = list(_mh.get("synonyms") or [])
                            _syns_low = {s.lower() for s in _syns}
                            if _anchor_low not in _syns_low:
                                _syns.append(_anchor)
                                _mh["synonyms"] = _syns
                    if _role != "outcome":
                        continue
                    _existing = [s for s in (_mh.get("synonyms") or []) if s]
                    _seen = {s.lower() for s in _existing}
                    _canon = _mh.get("concept") or ""
                    if _canon and _canon.lower() not in _seen:
                        _existing.append(_canon)
                        _seen.add(_canon.lower())
                    for _syn in _orblock:
                        if _syn.lower() not in _seen:
                            _existing.append(_syn)
                            _seen.add(_syn.lower())
                    _mh["synonyms"] = _existing
                    # Outcome must-have is no longer a hard cap — the
                    # severity-marker directness classifier owns recall.
                    _mh["required_for_high_relevance"] = False
                # If NO condition-role must_have exists at all but we
                # do have a deterministic anchor, inject a clean
                # condition must_have so the anchor service + ranker
                # can match the canonical disease. Closed-set, safe.
                if (
                    _anchor
                    and not _has_condition_role
                ):
                    _mh_list = list(picott_payload.get("must_have_concepts") or [])
                    _mh_list.append({
                        "concept": _anchor,
                        "role": "condition",
                        "synonyms": [_anchor],
                        "required_for_high_relevance": True,
                    })
                    picott_payload["must_have_concepts"] = _mh_list
        except Exception:  # noqa: BLE001
            # Defensive — never break the pipeline if the helper is
            # unavailable.
            logger.exception(
                "phaseK4A2 severity_marker_overview override failed (non-fatal)",
            )

        # Phase K4A.3 — diagnostic rule-out / test-utility deterministic
        # override. Mirrors the K4A.2 injection: when the question
        # matches the closed-set "Does a negative [test] rule out
        # [condition]?" shape (or equivalents — NPV / sensitivity /
        # de-escalation / stewardship), we override the picott_payload
        # so the downstream anchor service, query compiler, and ranker
        # all route the question as a diagnostic-accuracy / rule-out
        # test-utility question instead of falling through to
        # ``awaiting_clinical_subject``.
        # Closed-set, deterministic, NO LLM. NO publisher scraping.
        try:
            from services.rule_out_test_utility import (
                is_rule_out_test_utility_question,
                extract_rule_out_components,
                mrsa_or_block_synonyms,
                nasal_screening_or_block_synonyms,
                mrsa_pneumonia_or_block_synonyms,
                diagnostic_rule_out_or_block_synonyms,
            )
            _q = sanitized_question or ""
            if is_rule_out_test_utility_question(_q):
                _comp = extract_rule_out_components(_q)
                _idx_test = _comp.get("index_test") or "MRSA nasal screening"
                _tgt_cond = _comp.get("target_condition") or "MRSA pneumonia"
                _mrsa_route = bool(_comp.get("mrsa_route"))
                # Force a diagnosis-family route — never let an empty
                # PICOTT (LLM zero-output) drag the question down to
                # ``type_of_question='other'``.
                picott_payload["question_subtype"] = "rule_out_test_utility"
                picott_payload["type_of_question"] = "diagnosis"
                # Surface the closed-set anchors for ranker + display.
                picott_payload["rule_out_index_test_anchor"] = _idx_test
                picott_payload["rule_out_target_condition_anchor"] = _tgt_cond
                # Replace / augment must_have_concepts so the anchor
                # service has a real disease + index_test to work with.
                _existing_mh = list(picott_payload.get("must_have_concepts") or [])
                _existing_concepts_low = {
                    (m.get("concept") or "").strip().lower()
                    for m in _existing_mh if isinstance(m, dict)
                }
                # 1. Target-condition must-have (multi-word; survives
                #    K4A.2.1 generic denylist by exact-match check).
                if _tgt_cond and _tgt_cond.lower() not in _existing_concepts_low:
                    _existing_mh.append({
                        "concept": _tgt_cond,
                        "role": "condition",
                        "synonyms": list(mrsa_pneumonia_or_block_synonyms())
                                    if _mrsa_route else [_tgt_cond],
                        "required_for_high_relevance": True,
                    })
                # 2. Index-test must-have.
                if _idx_test and _idx_test.lower() not in _existing_concepts_low:
                    _existing_mh.append({
                        "concept": _idx_test,
                        "role": "index_test",
                        "synonyms": list(nasal_screening_or_block_synonyms()),
                        "required_for_high_relevance": True,
                    })
                # 3. MRSA organism must-have (only when MRSA route).
                if _mrsa_route and "mrsa" not in _existing_concepts_low:
                    _existing_mh.append({
                        "concept": "MRSA",
                        "role": "drug_or_drug_class",
                        "synonyms": list(mrsa_or_block_synonyms()),
                        # MRSA acronym is not a hard cap — the index-test
                        # + target-condition pair already gates retrieval.
                        "required_for_high_relevance": False,
                    })
                # 4. Outcome must-have (diagnostic rule-out cues) —
                #    ranking-only (NO hard AND on every emitted query).
                _existing_mh.append({
                    "concept": "rule out",
                    "role": "outcome",
                    "synonyms": list(diagnostic_rule_out_or_block_synonyms()),
                    "required_for_high_relevance": False,
                })
                picott_payload["must_have_concepts"] = _existing_mh
        except Exception:  # noqa: BLE001
            # Defensive — never break the pipeline if the helper is
            # unavailable.
            logger.exception(
                "phaseK4A3 rule_out_test_utility override failed (non-fatal)",
            )

        # Phase K4A.4 — incidence / risk epidemiology deterministic
        # override for the "What is the incidence/risk of OUTCOME in
        # patients with RISK_CONTEXT?" question family (e.g., aspiration
        # pneumonia + low GCS / depressed consciousness). Mirrors the
        # K4A.2 / K4A.3 injection pattern. Closed-set, deterministic,
        # NO LLM.
        #
        # Route mutual exclusivity (CRITICAL): at most ONE of
        # K4A.2 / K4A.3 / K4A.4 fires per session. Precedence (first
        # match wins; later routes abort):
        #   1. K4A.3 (rule-out / diagnostic accuracy)  — most specific
        #   2. K4A.2 (severity-marker overview)
        #   3. K4A.4 (epidemiology / incidence)
        #   4. default
        # We enforce this here by gating K4A.4 on
        # ``question_subtype NOT IN {severity_marker_overview,
        # rule_out_test_utility}`` (set by the blocks above).
        try:
            from services.incidence_risk_epidemiology import (
                is_incidence_risk_question,
                extract_incidence_risk_components,
                aspiration_pneumonia_or_block_synonyms,
                low_consciousness_or_block_synonyms,
                incidence_risk_or_block_synonyms,
            )
            _q = sanitized_question or ""
            _existing_subtype = (
                picott_payload.get("question_subtype") or ""
            ).strip().lower()
            _route_taken_by_prior_phase = _existing_subtype in {
                "severity_marker_overview", "rule_out_test_utility",
            }
            if (
                not _route_taken_by_prior_phase
                and is_incidence_risk_question(_q)
            ):
                _comp = extract_incidence_risk_components(_q)
                _outcome = _comp.get("outcome") or "aspiration pneumonia"
                _risk_context = (
                    _comp.get("risk_context") or "low Glasgow Coma Scale"
                )
                _gcs_route = bool(_comp.get("gcs_route"))
                # Force epidemiology route — never let an empty PICOTT
                # drag the question down to ``type_of_question='other'``.
                picott_payload["question_subtype"] = (
                    "epidemiology_incidence_complication_risk"
                )
                picott_payload["type_of_question"] = "epidemiology"
                # Surface closed-set anchors for ranker + display.
                picott_payload["incidence_outcome_anchor"] = _outcome
                picott_payload["incidence_risk_context_anchor"] = _risk_context
                # Filter out spurious concepts the LLM PICOTT extractor
                # or ontology lift may have introduced for this shape:
                #   • generic ``patient`` / ``patients`` mapped as
                #     ``clinical_action`` / population (display-only,
                #     never a primary clinical anchor)
                #   • literal-phrase population qualifiers like
                #     ``"patients with GCS 8 or less"`` (the canonical
                #     ``low Glasgow Coma Scale`` anchor is re-injected
                #     below)
                #   • ``complications prevention`` injected via the
                #     vaccine specialty pack's pneumonia-synonym match
                #     (fabricated — user did NOT ask about prevention)
                _STRIP_CONCEPTS_LOW = {
                    "patient", "patients",
                    "complications prevention",
                }
                _STRIP_CONCEPT_SUBSTRINGS = (
                    "patients with gcs",
                    "patient with gcs",
                    "patients with low gcs",
                    "patients with depressed gcs",
                    "patients with low glasgow coma",
                    "patients with depressed consciousness",
                    "patients with depressed level of consciousness",
                    "patients with decreased level of consciousness",
                    "patients with altered mental status",
                    "patients with loss of consciousness",
                    "patient with low gcs",
                    "patient with low glasgow coma",
                    "patient with depressed consciousness",
                    "complications prevention",
                )
                _existing_mh = []
                for _mh in (picott_payload.get("must_have_concepts") or []):
                    if not isinstance(_mh, dict):
                        continue
                    _concept_low = (_mh.get("concept") or "").strip().lower()
                    _role_low = (_mh.get("role") or "").strip().lower()
                    if _concept_low in _STRIP_CONCEPTS_LOW:
                        continue
                    if any(
                        sub in _concept_low
                        for sub in _STRIP_CONCEPT_SUBSTRINGS
                    ):
                        continue
                    # Also strip any clinical_action role pointing at
                    # generic population labels.
                    if (
                        _role_low == "clinical_action"
                        and _concept_low in {
                            "patient", "patients", "adults", "adult",
                        }
                    ):
                        continue
                    _existing_mh.append(_mh)
                _existing_concepts_low = {
                    (m.get("concept") or "").strip().lower()
                    for m in _existing_mh if isinstance(m, dict)
                }
                # 1. Outcome must-have (aspiration pneumonia) —
                #    required_for_high_relevance=True; multi-word
                #    survives the K4A.2.1 generic denylist by exact-
                #    match check.
                if _outcome and _outcome.lower() not in _existing_concepts_low:
                    _existing_mh.append({
                        "concept": _outcome,
                        "role": "condition",
                        "synonyms": list(aspiration_pneumonia_or_block_synonyms()),
                        "required_for_high_relevance": True,
                    })
                # 2. Risk-context exposure must-have (low GCS / Glasgow
                #    Coma Scale / depressed consciousness).
                if _risk_context and _risk_context.lower() not in _existing_concepts_low:
                    _existing_mh.append({
                        "concept": _risk_context,
                        "role": "exposure",
                        "synonyms": list(low_consciousness_or_block_synonyms()),
                        "required_for_high_relevance": True,
                    })
                # NOTE: ``incidence`` is intentionally NOT added as a
                # must_have_concept. Phase K4A.4.1 regression fix —
                # promoting it to a must_have made it become a primary
                # clinical anchor with ``entity_type=biomarker_or_lab``
                # + ``role=outcome``, which routed it into the
                # ``outcomes_target_diagnosis`` bucket of
                # ``picott_term_mapping_json`` and surfaced as the UI
                # "Outcome" label — masking ``aspiration pneumonia``
                # as the true outcome. The OR-block synonyms still
                # ride in ``pico.outcomes_or_target_diagnosis`` (added
                # below) for the legacy concept_block_query_builder
                # ranking pass, and the directness classifier reads
                # the article metadata directly — so retrieval +
                # ranking are unaffected.
                picott_payload["must_have_concepts"] = _existing_mh
                # Phase K4A.4 — ALSO clean the legacy ``pico`` dict that
                # the Phase 5A concept_block_query_builder reads from
                # directly (population / outcomes_or_target_diagnosis).
                # Without this scrub, the legacy queries still emit
                # literal phrases like ``"patients with GCS 8 or less"``
                # and ``"complications prevention"`` AS REQUIRED AND-
                # block clauses — even though the K4A.4 must_have_
                # concepts replacement above is clean.
                _legacy_pico = dict(picott_payload.get("pico") or {})
                # Replace ``population`` with the canonical low-GCS
                # anchor set so the legacy population OR-block surfaces
                # the right terms instead of the bare LLM literal.
                _legacy_pico["population"] = [_risk_context] + [
                    s for s in low_consciousness_or_block_synonyms()
                    if s.lower() != _risk_context.lower()
                ]
                # Strip ``"complications prevention"`` (and similar
                # fabricated prevention outcomes) from
                # outcomes_or_target_diagnosis. Replace with the
                # canonical incidence/risk OR-block so the legacy
                # broad-recall query carries the correct method
                # concept.
                _legacy_outcomes = []
                for _o in (_legacy_pico.get("outcomes_or_target_diagnosis") or []):
                    _o_low = (_o or "").strip().lower()
                    if not _o_low:
                        continue
                    if _o_low in {
                        "complications prevention",
                        "prevention of complications",
                    }:
                        continue
                    _legacy_outcomes.append(_o)
                # Always surface aspiration pneumonia + incidence
                # method concepts so the legacy broad-recall block has
                # the canonical anchors.
                for _t in (
                    [_outcome] + list(aspiration_pneumonia_or_block_synonyms())
                    + ["incidence"] + list(incidence_risk_or_block_synonyms())
                ):
                    if _t and _t.lower() not in {
                        x.lower() for x in _legacy_outcomes
                    }:
                        _legacy_outcomes.append(_t)
                _legacy_pico["outcomes_or_target_diagnosis"] = _legacy_outcomes
                picott_payload["pico"] = _legacy_pico
        except Exception:  # noqa: BLE001
            # Defensive — never break the pipeline if the helper is
            # unavailable.
            logger.exception(
                "phaseK4A4 incidence_risk_epidemiology override failed (non-fatal)",
            )

        # Phase M4.6 — deterministic prognosis/stage cancer retrieval rescue.
        # Mirrors the K4A.2/K4A.3/K4A.4 injection: runs AFTER repaired_picott
        # and BEFORE the primary-anchor derivation + Boolean compiler so the
        # CANCER becomes the condition anchor — NOT the glued full phrase
        # "prognosis of stage 3 prostate cancer", which matches no article and
        # caps the entire pool as missing_condition_anchor. Stage rides as a
        # high-precision ranking concept (never a hard must-have). Closed-set,
        # deterministic, NO LLM, NO network. Skips if a prior K4A phase owns
        # the route.
        try:
            from services.prognosis_stage_cancer_rescue import (
                apply_rescue as _m46_apply,
            )
            _m46_diag = _m46_apply(picott_payload, sanitized_question or "")
            if _m46_diag.get("applied"):
                parsed["picott_json"] = picott_payload
                logger.info(
                    "phaseM4_6_prognosis_stage_cancer_rescue_applied "
                    "subtype=%s has_stage=%s",
                    _m46_diag.get("question_subtype"),
                    bool(_m46_diag.get("stage")),
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseM4_6 prognosis_stage_cancer rescue failed (non-fatal)",
            )

        # Phase 6B.2 — attach Clinical Context Engine outputs onto
        # ``parsed`` so the downstream EvidenceRankerService can read
        # ``role_weights_json`` + ``multi_intent_json`` without a second
        # DB round-trip. These keys mirror the session document fields.
        parsed["role_weights_json"] = dict(context.get("role_weights") or {})
        parsed["multi_intent_json"] = dict(context.get("multi_intent") or {})
        parsed["ontology_mappings_json"] = list(context.get("ontology_mappings") or [])
        parsed["role_repairs_json"] = list(context.get("role_repairs") or [])
        parsed["ontology_lifts_json"] = list(context.get("ontology_lifts") or [])

        # Phase 6F — PrimaryClinicalEntityAnchorService.
        # Runs AFTER PHI guard + parser + clinical_context_engine,
        # BEFORE the Boolean query compiler. Walks the repaired PICOTT
        # + ontology mappings to derive closed-set primary clinical
        # entity anchors. The output drives the query compiler caps,
        # the picott_aware_ranker (wrong-drug / wrong-disease caps),
        # default-selection guardrails, and the evidence-packet
        # validator. NEVER raises.
        try:
            with diag.stage("primary_clinical_anchor"):
                anchors_out = primary_clinical_anchor_service.derive(
                    picott=picott_payload,
                    ontology_mappings=list(context.get("ontology_mappings") or []),
                    normalized_question=(
                        parsed.get("normalized_question") or sanitized_question or ""
                    ),
                )
        except Exception:  # noqa: BLE001
            logger.exception("phase6f primary_clinical_anchor failed (non-fatal)")
            anchors_out = {
                "primary_anchors": [],
                "generic_modifiers": [],
                "anchor_repair_applied": False,
                "anchor_failure_reason": None,
                "picott_overrides": {},
            }
        # Phase MedSafety-1 — deterministic medication subject rescue.
        # Runs ONLY when the normal anchor derivation found no primary
        # clinical entity. If the (normalized) question contains a
        # recognized medication (generic / brand / safe typo from the
        # curated lexicon), inject a deterministic drug anchor so a
        # clearly drug-centric dosing/safety question proceeds to
        # evidence retrieval instead of ``awaiting_clinical_subject``.
        # NO LLM. NO network. NO dosing advice. PHI guard already ran
        # first. Questions that already resolved an anchor are untouched.
        try:
            _no_subject = (
                anchors_out.get("anchor_failure_reason") == "no_primary_clinical_entity"
                or not (anchors_out.get("primary_anchors") or [])
            )
            if _no_subject:
                from services.medication_subject_rescue import (
                    build_medication_anchor,
                    detect_medication,
                    is_dosing_safety_question,
                )
                _med_q = (
                    parsed.get("normalized_question") or sanitized_question or ""
                )
                _med = detect_medication(_med_q)
                if _med is not None:
                    _med_anchor = build_medication_anchor(_med)
                    anchors_out["primary_anchors"] = [_med_anchor]
                    anchors_out["anchor_failure_reason"] = None
                    anchors_out["anchor_repair_applied"] = True
                    anchors_out["medication_subject_rescue_applied"] = True
                    # Make the Boolean compiler query the drug (not "dose"
                    # alone): synthesize a must_have_concept on the picott.
                    _mh = list(picott_payload.get("must_have_concepts") or [])
                    _have = {
                        (m.get("concept") or "").lower()
                        for m in _mh if isinstance(m, dict)
                    }
                    if _med_anchor["canonical_concept"].lower() not in _have:
                        _mh.append({
                            "concept": _med_anchor["canonical_concept"],
                            "role": "intervention",
                            "synonyms": list(_med_anchor.get("synonyms") or []),
                            "required_for_high_relevance": True,
                        })
                        picott_payload["must_have_concepts"] = _mh
                    # Route dosing/safety questions to the drug_safety intent.
                    if is_dosing_safety_question(_med_q):
                        picott_payload["question_subtype"] = "drug_safety"
                    logger.info(
                        "medsafety1_medication_subject_rescue_applied "
                        "generic=%s source=%s dosing_safety=%s",
                        _med.get("generic"), _med.get("source"),
                        is_dosing_safety_question(_med_q),
                    )
        except Exception:  # noqa: BLE001
            logger.exception("medsafety1 medication rescue failed (non-fatal)")
        # Phase TopicRescue — deterministic supplement / nutrient /
        # "benefit-of-<topic>" subject rescue. Runs ONLY when anchor
        # derivation AND the medication rescue both found no primary clinical
        # entity. Extracts a concrete searchable topic (e.g. "creatine
        # supplementation") and injects an intervention/exposure anchor so a
        # clearly answerable question runs a real search instead of
        # dead-ending at ``awaiting_clinical_subject``. NO LLM. NO network.
        try:
            _still_no_subject = (
                anchors_out.get("anchor_failure_reason") == "no_primary_clinical_entity"
                or not (anchors_out.get("primary_anchors") or [])
            )
            if _still_no_subject:
                from services.topic_subject_rescue import (
                    build_topic_anchor,
                    detect_topic_subject,
                )
                _topic_q = (
                    parsed.get("normalized_question") or sanitized_question or ""
                )
                _topic = detect_topic_subject(_topic_q)
                if _topic:
                    _topic_anchor = build_topic_anchor(_topic)
                    anchors_out["primary_anchors"] = [_topic_anchor]
                    anchors_out["anchor_failure_reason"] = None
                    anchors_out["anchor_repair_applied"] = True
                    anchors_out["topic_subject_rescue_applied"] = True
                    # Query the topic (not a generic modifier): synthesize a
                    # must_have_concept so the Boolean compiler anchors on it.
                    _mh = list(picott_payload.get("must_have_concepts") or [])
                    _have = {
                        (m.get("concept") or "").lower()
                        for m in _mh if isinstance(m, dict)
                    }
                    if _topic_anchor["canonical_concept"].lower() not in _have:
                        _mh.append({
                            "concept": _topic_anchor["canonical_concept"],
                            "role": _topic_anchor.get("role") or "intervention",
                            "synonyms": list(_topic_anchor.get("synonyms") or []),
                            "required_for_high_relevance": True,
                        })
                        picott_payload["must_have_concepts"] = _mh
                    logger.info(
                        "topic_subject_rescue_applied topic=%s entity=%s",
                        _topic_anchor.get("canonical_concept"),
                        _topic_anchor.get("entity_type"),
                    )
        except Exception:  # noqa: BLE001
            logger.exception("topic_subject_rescue failed (non-fatal)")
        # Phase 6G.1 — apply background-definition picott_overrides
        # produced by the anchor service when "What is X?" repair
        # succeeded. We only OVERWRITE the targeted keys
        # (``question_subtype``, ``type_of_question``,
        # ``question_scope``, ``specialty``) on the picott_payload so
        # the downstream query compiler / ranker / term-mapping
        # projection can route the question as ``disease_overview`` or
        # ``drug_overview``.
        picott_overrides = (anchors_out or {}).get("picott_overrides") or {}
        if picott_overrides:
            for key, val in picott_overrides.items():
                if val is None:
                    continue
                # Don't trash an existing concrete specialty with None.
                if key == "specialty" and not val and picott_payload.get(key):
                    continue
                picott_payload[key] = val
            # Synthesize a must_have_concept for the new primary anchor
            # so the boolean compiler sees it as a query anchor (the
            # compiler reads ``picott_payload["must_have_concepts"]``).
            mh_list = list(picott_payload.get("must_have_concepts") or [])
            existing_concepts = {
                (m.get("concept") or "").lower()
                for m in mh_list if isinstance(m, dict)
            }
            for a in (anchors_out.get("primary_anchors") or []):
                if not isinstance(a, dict):
                    continue
                c = (a.get("canonical_concept") or "").strip()
                if not c or c.lower() in existing_concepts:
                    continue
                mh_role = (
                    "condition" if a.get("entity_type") == "disease_or_problem"
                    else "intervention" if a.get("entity_type") == "drug_or_drug_class"
                    else "index_test" if a.get("entity_type") in {
                        "diagnostic_test", "biomarker_or_lab",
                    }
                    else "other"
                )
                mh_list.append({
                    "concept": c,
                    "role": mh_role,
                    "synonyms": list(a.get("synonyms") or []),
                    "required_for_high_relevance": True,
                })
                existing_concepts.add(c.lower())
            picott_payload["must_have_concepts"] = mh_list
        # Drop the overrides key before persistence — it's a transport
        # field, not a schema-stable session-document field.
        anchors_out_persist = {
            k: v for k, v in anchors_out.items() if k != "picott_overrides"
        }
        parsed["primary_clinical_anchors_json"] = dict(anchors_out_persist)

        # Phase 6F closeout (IMPL-2) — anchor-failure clarification gate.
        # If PrimaryClinicalEntityAnchorService could not derive ANY
        # primary clinical entity (and the ontology-repair pass also
        # failed), the session terminates at
        # ``status="awaiting_clinical_subject"`` WITHOUT touching
        # PubMed / Europe PMC / OpenAlex / scholar-like expansion / any
        # LLM. The clarification message is persisted on the session
        # so the GET response and the UI banner can render it.
        # Closed-set audit event with COUNTS only — never the raw
        # question text.
        #
        # Phase H1 — bounded LLM Clinical Context Resolver runs HERE
        # (before the clarification short-circuit) when triggered. The
        # resolver is closed-set, schema-locked, and cannot bypass
        # caps or the Boolean compiler. Closed-set diagnostics are
        # always populated even when the resolver is skipped.
        resolver_decision_dict: dict | None = None
        resolver_skipped_reason: str | None = None
        if anchors_out.get("anchor_failure_reason") == "no_primary_clinical_entity":
            from services.clinical_context_resolver import (
                clinical_context_resolver,
                current_mode as _resolver_current_mode,
            )
            _mode = _resolver_current_mode()
            if _mode == "off":
                resolver_skipped_reason = "mode_off"
            else:
                try:
                    decision = await clinical_context_resolver.resolve_clinical_context(
                        original_question=sanitized_question or "",
                        normalized_question=(
                            parsed.get("normalized_question")
                            or sanitized_question or ""
                        ),
                        deterministic_picott=picott_payload,
                        primary_anchors_detected=list(
                            anchors_out.get("primary_anchors") or []
                        ),
                        generic_modifiers_detected=list(
                            anchors_out.get("generic_modifiers") or []
                        ),
                        anchor_failure_reason="no_primary_clinical_entity",
                        existing_concept_translations=[],
                        specialty_hints=[],
                        diagnostics=diag,
                    )
                    resolver_decision_dict = {
                        "accepted": bool(decision.accepted),
                        "mode": decision.mode,
                        "rejection_reason": decision.rejection_reason,
                        "latency_ms": int(decision.latency_ms or 0),
                        "token_estimate": dict(decision.token_estimate or {}),
                        "resolver_confidence": (
                            float(decision.output.confidence)
                            if (decision.accepted and decision.output is not None)
                            else None
                        ),
                    }
                    try:
                        await audit_service.log(
                            "context_resolver_called",
                            user_id=user_id,
                            search_session_id=session_id,
                            payload={
                                "mode": decision.mode,
                            },
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug("phaseH1 audit context_resolver_called failed", exc_info=True)
                    # Phase H1 — merge resolver output into the
                    # canonical interpretation ONLY when:
                    #   - accepted=True
                    #   - mode == "fallback_active"
                    #   - output.is_valid_clinical_question == True
                    #   - output.clarification_needed == False
                    #   - output.primary_anchors non-empty
                    if (
                        decision.accepted
                        and decision.output is not None
                        and decision.mode == "fallback_active"
                        and decision.output.is_valid_clinical_question
                        and not decision.output.clarification_needed
                        and decision.output.primary_anchors
                    ):
                        # Merge into picott_payload + anchors_out.
                        new_anchors: list[dict] = []
                        merged_must_have: list[dict] = list(
                            picott_payload.get("must_have_concepts") or []
                        )
                        existing_concepts = {
                            (m.get("concept") or "").lower()
                            for m in merged_must_have
                            if isinstance(m, dict)
                        }
                        for ra in decision.output.primary_anchors:
                            new_anchors.append({
                                "text": ra.text,
                                "canonical_concept": ra.canonical_concept,
                                "entity_type": ra.entity_type,
                                "role": ra.role,
                                # Resolver-sourced anchors land at
                                # confidence 0.7 (deterministic floor)
                                # — they don't get the pack-sourced
                                # 0.92+ confidence.
                                "confidence": 0.7,
                                "source": "llm_resolver_fallback",
                                "synonyms": [ra.canonical_concept],
                                "required_for_search": bool(ra.required_for_search),
                                "missing_relevance_cap": int(ra.missing_relevance_cap),
                            })
                            c = (ra.canonical_concept or "").strip()
                            if c and c.lower() not in existing_concepts:
                                # Phase H1.1-tail — closed-set entity_type
                                # → compiler block role mapping. ONLY the
                                # 7 closed-set entity_types below get an
                                # AND-required role (condition /
                                # intervention / index_test); anything
                                # else falls through to "other" so the
                                # compiler treats it as context-only.
                                # This is the targeted fix for the
                                # resolver-sourced procedure_or_surgery
                                # anchor that previously landed as
                                # "other" and was silently dropped from
                                # every compiled query.
                                _et = ra.entity_type
                                if _et == "disease_or_problem":
                                    _role = "condition"
                                elif _et == "procedure_or_surgery":
                                    _role = "condition"
                                elif _et == "drug_or_drug_class":
                                    _role = "intervention"
                                elif _et == "vaccine":
                                    _role = "intervention"
                                elif _et == "management_intervention":
                                    _role = "intervention"
                                elif _et == "exposure":
                                    _role = "intervention"
                                elif _et in {
                                    "diagnostic_test", "biomarker_or_lab",
                                }:
                                    _role = "index_test"
                                else:
                                    _role = "other"
                                merged_must_have.append({
                                    "concept": c,
                                    "role": _role,
                                    "synonyms": [c],
                                    "required_for_high_relevance": True,
                                })
                                existing_concepts.add(c.lower())
                        # Term-sets of the anchors already present, so we can
                        # collapse resolver concepts that are just a narrower
                        # duplicate of an existing anchor (e.g. parent molecule
                        # "Glucagon-Like Peptide 1" vs drug class "GLP-1 receptor
                        # agonists") instead of adding them as a THIRD required
                        # anchor that wrongly caps on-topic articles.
                        _dedup_on = anchor_dedup.anchor_dedup_enabled()
                        _existing_termsets = [
                            [m.get("concept") or ""] + list(m.get("synonyms") or [])
                            for m in merged_must_have if isinstance(m, dict)
                        ] + [
                            [a.get("canonical_concept") or a.get("text") or ""]
                            + list(a.get("synonyms") or [])
                            for a in new_anchors
                        ]
                        for mh in decision.output.must_have_concepts:
                            c = (mh.concept or "").strip()
                            if not c or c.lower() in existing_concepts:
                                continue
                            # Skip a resolver concept that is a redundant
                            # duplicate of an anchor already present — it would
                            # otherwise become a hard anchor and reject relevant
                            # papers for "missing" it.
                            if _dedup_on and anchor_dedup.concept_is_redundant(
                                c, _existing_termsets
                            ):
                                existing_concepts.add(c.lower())
                                continue
                            merged_must_have.append({
                                "concept": c,
                                "role": "other",
                                "synonyms": [c],
                                "required_for_high_relevance": bool(
                                    mh.required_for_high_relevance,
                                ),
                            })
                            existing_concepts.add(c.lower())
                            _existing_termsets.append([c])
                        picott_payload["must_have_concepts"] = merged_must_have
                        # Surface resolver-sourced top-level fields.
                        if decision.output.question_subtype:
                            picott_payload["question_subtype"] = (
                                decision.output.question_subtype
                            )
                        if decision.output.type_of_question:
                            picott_payload["type_of_question"] = (
                                decision.output.type_of_question
                            )
                        if decision.output.question_scope:
                            picott_payload["question_scope"] = (
                                decision.output.question_scope
                            )
                        if decision.output.specialty:
                            picott_payload["specialty"] = (
                                decision.output.specialty[0]
                            )
                        # Update anchors_out so the persistence + UI
                        # see the resolver-sourced anchors as the
                        # authoritative primary_anchors list.
                        anchors_out["primary_anchors"] = (
                            new_anchors
                            + [a for a in (anchors_out.get("primary_anchors") or [])]
                        )
                        anchors_out["anchor_failure_reason"] = None
                        anchors_out["anchor_repair_applied"] = True
                        anchors_out["interpretation_source"] = (
                            "llm_resolver_fallback"
                        )
                        # Phase H1.1 — refresh the persisted snapshot
                        # so the resolver-sourced anchors land on the
                        # session document. The earlier persistence
                        # at line ~437 ran BEFORE the resolver merge.
                        parsed["primary_clinical_anchors_json"] = {
                            k: v for k, v in anchors_out.items()
                            if k != "picott_overrides"
                        }
                        # Also refresh picott_json — the merged
                        # must_have_concepts + question_subtype /
                        # type_of_question / specialty must reach the
                        # downstream Boolean compiler via the GET
                        # response shape.
                        parsed["picott_json"] = dict(picott_payload)
                        try:
                            await audit_service.log(
                                "context_resolver_accepted",
                                user_id=user_id,
                                search_session_id=session_id,
                                payload={
                                    "resolver_confidence": float(
                                        decision.output.confidence,
                                    ),
                                    "latency_ms": int(decision.latency_ms or 0),
                                    "total_tokens": int(
                                        (decision.token_estimate or {})
                                        .get("total_tokens") or 0
                                    ),
                                },
                            )
                        except Exception:  # noqa: BLE001
                            logger.debug("phaseH1 audit context_resolver_accepted failed", exc_info=True)
                    elif decision.accepted and decision.mode == "shadow":
                        # Record-only; do NOT mutate the search.
                        try:
                            await audit_service.log(
                                "context_resolver_shadow_recorded",
                                user_id=user_id,
                                search_session_id=session_id,
                                payload={
                                    "resolver_confidence": float(
                                        decision.output.confidence
                                        if decision.output else 0.0,
                                    ),
                                    "latency_ms": int(decision.latency_ms or 0),
                                },
                            )
                        except Exception:  # noqa: BLE001
                            logger.debug("phaseH1 audit shadow_recorded failed", exc_info=True)
                    elif not decision.accepted:
                        try:
                            await audit_service.log(
                                "context_resolver_rejected",
                                user_id=user_id,
                                search_session_id=session_id,
                                payload={
                                    "rejection_reason": decision.rejection_reason,
                                },
                            )
                        except Exception:  # noqa: BLE001
                            logger.debug("phaseH1 audit context_resolver_rejected failed", exc_info=True)
                except Exception:  # noqa: BLE001
                    logger.exception("phaseH1 resolver invocation failed (non-fatal)")
                    resolver_skipped_reason = "no_trigger_matched"
            # After resolver: if anchors_out still carries
            # ``anchor_failure_reason == "no_primary_clinical_entity"``
            # fall through to the existing closed-set clarification
            # path below (unchanged behavior).
        if (
            resolver_decision_dict is not None
            or resolver_skipped_reason is not None
        ):
            parsed.setdefault("session_diagnostics_json", {})
            _diag_mode = (
                (resolver_decision_dict or {}).get("mode")
                or ("off" if resolver_skipped_reason == "mode_off" else "fallback_active")
            )
            _interpretation_source = (
                "llm_resolver_fallback"
                if anchors_out.get("interpretation_source") == "llm_resolver_fallback"
                else (
                    "llm_resolver_shadow_recorded"
                    if _diag_mode == "shadow"
                    and (resolver_decision_dict or {}).get("accepted")
                    else "deterministic"
                )
            )
            resolver_diag.update({
                "called": bool(resolver_decision_dict is not None),
                "mode": _diag_mode,
                "accepted": bool((resolver_decision_dict or {}).get("accepted")),
                "skipped_reason": resolver_skipped_reason,
                "rejection_reason": (resolver_decision_dict or {}).get("rejection_reason"),
                "latency_ms": int((resolver_decision_dict or {}).get("latency_ms") or 0),
                "token_estimate": dict(
                    (resolver_decision_dict or {}).get("token_estimate") or {}
                ),
                "interpretation_source": _interpretation_source,
                "resolver_confidence": (
                    (resolver_decision_dict or {}).get("resolver_confidence")
                ),
            })
            parsed["clinical_context_resolver_diag"] = dict(resolver_diag)

        # ---------------------------------------------------------
        # Phase M0/M1/M2-shadow — Bounded Clinical Semantics &
        # Context Interpreter v2. Closed-set deterministic detector
        # decides whether to run the bounded LLM repair. The
        # candidate is validated by a separate deterministic
        # validator and persisted ONLY into session diagnostics —
        # active retrieval / ranking / default selection / evidence
        # packets / summaries / citation verifier are untouched.
        #
        # The detector skips when:
        #   • K4A.2 / K4A.3 / K4A.4 already fired (question_subtype
        #     is in the closed-set _K4A_FIRED_SUBTYPES)
        #   • anchor_failure_reason == "no_primary_clinical_entity"
        #     (the H1 resolver path covers that case)
        #
        # PHI dominance: this block is only reachable when the PHI
        # guard upstream did NOT block the session — the pipeline
        # short-circuits to ``status=blocked_phi`` long before
        # reaching here. As defense-in-depth the validator ALSO
        # scans every output field for PHI patterns.
        # ---------------------------------------------------------
        semantic_repair_block: dict[str, Any] = {
            "semantic_repair_triggered": False,
            "semantic_repair_trigger_reasons": [],
            "semantic_repair_candidate": None,
            "semantic_repair_validation_status": "not_attempted",
            "semantic_repair_rejection_reasons": [],
            "semantic_repair_diff": {
                "would_change_intent": False,
                "would_change_framework": False,
                "would_change_required_anchors": False,
                "would_change_branch_plan": False,
            },
            "semantic_repair_latency_ms": 0,
        }
        if _LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED:
            try:
                from services.semantic_anchor_repair import (
                    detect_repair_triggers as _sar_detect,
                    run_semantic_anchor_repair_shadow as _sar_run,
                    diff_repair_candidate as _sar_diff,
                )
                _sar_triggers = _sar_detect(
                    normalized_question=(
                        parsed.get("normalized_question")
                        or sanitized_question or ""
                    ),
                    question_subtype=picott_payload.get("question_subtype"),
                    type_of_question=picott_payload.get("type_of_question"),
                    primary_anchors=list(anchors_out.get("primary_anchors") or []),
                    generic_modifiers=list(anchors_out.get("generic_modifiers") or []),
                    anchor_failure_reason=anchors_out.get("anchor_failure_reason"),
                )
                if _sar_triggers:
                    semantic_repair_block["semantic_repair_triggered"] = True
                    semantic_repair_block["semantic_repair_trigger_reasons"] = (
                        list(_sar_triggers)
                    )
                    _sar_candidate, _sar_rejections, _sar_latency = await _sar_run(
                        normalized_question=(
                            parsed.get("normalized_question")
                            or sanitized_question or ""
                        ),
                        trigger_reasons=_sar_triggers,
                        deterministic_primary_anchors=list(
                            anchors_out.get("primary_anchors") or [],
                        ),
                        diagnostics=diag,
                    )
                    semantic_repair_block["semantic_repair_latency_ms"] = (
                        int(_sar_latency or 0)
                    )
                    if _sar_candidate is not None and not _sar_rejections:
                        semantic_repair_block["semantic_repair_candidate"] = (
                            _sar_candidate.model_dump(mode="json")
                        )
                        semantic_repair_block["semantic_repair_validation_status"] = (
                            "passed"
                        )
                        semantic_repair_block["semantic_repair_diff"] = _sar_diff(
                            candidate=_sar_candidate,
                            current_intent=picott_payload.get("type_of_question"),
                            current_frameworks=list(
                                picott_payload.get("framework_list_json") or []
                            ),
                            current_required_anchors=[
                                (a.get("canonical_concept")
                                 or a.get("text") or "")
                                for a in (anchors_out.get("primary_anchors") or [])
                                if isinstance(a, dict)
                                and a.get("required_for_search") is True
                            ],
                            current_branch_count=len(
                                (picott_payload.get("branches") or [])
                            ),
                        )
                    else:
                        semantic_repair_block["semantic_repair_validation_status"] = (
                            "rejected"
                        )
                        if _sar_rejections:
                            semantic_repair_block["semantic_repair_rejection_reasons"] = list(
                                _sar_rejections,
                            )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "phaseM semantic_anchor_repair shadow block failed (non-fatal)",
                )
                semantic_repair_block = {
                    "semantic_repair_triggered": False,
                    "semantic_repair_trigger_reasons": [],
                    "semantic_repair_candidate": None,
                    "semantic_repair_validation_status": "not_attempted",
                    "semantic_repair_rejection_reasons": [],
                    "semantic_repair_diff": {
                        "would_change_intent": False,
                        "would_change_framework": False,
                        "would_change_required_anchors": False,
                        "would_change_branch_plan": False,
                    },
                    "semantic_repair_latency_ms": 0,
                }
            parsed["semantic_anchor_repair_shadow"] = semantic_repair_block
            # Phase M3-preflight — compute the eligibility decision and
            # structural would-be projection. Diagnostics-only; never
            # promoted into active retrieval. The decision is wired even
            # when the validator rejected the candidate so the operator
            # can see WHY it would still be ineligible (operational +
            # clinical reasons).
            try:
                from services.semantic_repair_active_config import (
                    load_active_config as _m3_load_config,
                )
                from services.semantic_repair_active_eligibility import (
                    is_semantic_repair_active_eligible as _m3_eligibility,
                )
                from services.semantic_repair_active_projection import (
                    project_would_be_compiler_input as _m3_projection,
                )
                _m3_cfg = _m3_load_config()
                _m3_cand_dict = semantic_repair_block.get("semantic_repair_candidate")
                _m3_cand_validated = (
                    semantic_repair_block.get("semantic_repair_validation_status")
                    == "passed"
                )
                _m3_decision = _m3_eligibility(
                    session_blocked_phi=False,  # PHI path never reaches this code
                    detector_triggered=bool(
                        semantic_repair_block.get("semantic_repair_triggered")
                    ),
                    trigger_reasons=list(
                        semantic_repair_block.get("semantic_repair_trigger_reasons") or []
                    ),
                    candidate_validated=_m3_cand_validated,
                    candidate_rejection_reasons=list(
                        semantic_repair_block.get("semantic_repair_rejection_reasons") or []
                    ),
                    candidate_dict=_m3_cand_dict if isinstance(_m3_cand_dict, dict) else None,
                    deterministic_primary_anchors=list(
                        anchors_out.get("primary_anchors") or []
                    ),
                    question_subtype=picott_payload.get("question_subtype"),
                    latency_ms=int(
                        semantic_repair_block.get("semantic_repair_latency_ms") or 0
                    ),
                    config=_m3_cfg,
                )
                _m3_projection_block: dict[str, Any] = {}
                if _m3_cand_validated and isinstance(_m3_cand_dict, dict):
                    _m3_projection_block = _m3_projection(
                        candidate_dict=_m3_cand_dict,
                        current_primary_intent=picott_payload.get("type_of_question"),
                        current_frameworks=list(
                            picott_payload.get("framework_list_json") or []
                        ),
                        current_required_anchors=[
                            (a.get("canonical_concept") or a.get("text") or "")
                            for a in (anchors_out.get("primary_anchors") or [])
                            if isinstance(a, dict)
                            and a.get("required_for_search") is True
                        ],
                        current_branch_count=len(
                            (picott_payload.get("branches") or [])
                        ),
                    )
                semantic_repair_block["m3_preflight"] = {
                    "config": _m3_cfg.model_dump(mode="json"),
                    "eligibility": _m3_decision.to_diagnostics(),
                    "projection": _m3_projection_block,
                    # Hard invariant: M3-preflight NEVER promotes.
                    "active_retrieval_changed": False,
                }
            except Exception:  # noqa: BLE001
                logger.exception(
                    "phaseM3 preflight block failed (non-fatal)",
                )
                semantic_repair_block["m3_preflight"] = {
                    "config": None,
                    "eligibility": {
                        "eligible": False,
                        "pass_reasons": [],
                        "block_reasons": ["preflight_internal_error"],
                        "would_be_eligible_if_flags_enabled": False,
                        "shape": None,
                    },
                    "projection": {},
                    "active_retrieval_changed": False,
                }
        else:
            # Phase M4.2 — legacy M0/M1/M2 semantic_anchor_repair shadow +
            # M3-preflight runtime is DEPRECATED and superseded by the M4
            # full-question interpreter. The duplicate bounded-LLM call is
            # NOT made; a compact closed-set marker records the deprecation.
            semantic_repair_block = {
                "deprecated": True,
                "superseded_by": "m4_full_question_interpreter",
                "phase": "M4.2",
            }
            parsed["semantic_anchor_repair_shadow"] = semantic_repair_block
        # Persist into session_diagnostics_json so admin GETs surface
        # it deterministically. The closed-set field is namespaced
        # under ``semantic_anchor_repair`` to keep the parent dict
        # readable.
        parsed.setdefault("session_diagnostics_json", {})
        parsed["session_diagnostics_json"]["semantic_anchor_repair"] = (
            dict(semantic_repair_block)
        )

        # ---------------------------------------------------------
        # Phase M4-shadow — Full-Question LLM ClinicalQuestionMap
        # Interpreter. Broad bounded-LLM interpretation of the entire
        # PHI-cleared question into a full CQM candidate; deterministic
        # validation decides clean-ness. SHADOW ONLY, PREVIEW/DEV ONLY
        # (mode resolver forces ``disabled`` in production). NEVER
        # changes active retrieval / ranking / selection / summaries /
        # evidence packets / citation verifier. Recorded into session
        # diagnostics + an admin-only sidecar. PHI guard short-circuits
        # upstream; validator scans every output for PHI too.
        # ---------------------------------------------------------
        m4_block: dict[str, Any] = {
            "mode": "disabled",
            "triggered": False,
            "trigger_reasons": [],
            "validation_status": "not_attempted",
            "rejection_reasons": [],
            "baseline": None,
            "candidate": None,
            "would_be_compiler_input": {},
            "m4_0_1_projection": {},
            "accepted": False,
            "active_retrieval_changed": False,
            "latency_ms": 0,
        }
        try:
            from services.m4_full_question_interpreter import (
                resolve_m4_mode as _m4_mode,
                detect_m4_triggers as _m4_detect,
                run_m4_interpreter_shadow as _m4_run,
                project_m4_compiler_input as _m4_project,
            )
            from services.m4_compiler_input_projection import (
                build_m4_0_1_projection as _m4_0_1_build,
            )
            from services.m4_active_config import (
                load_m4_active_config as _m4_load_cfg,
            )
            _m4_resolved = _m4_mode()
            m4_block["mode"] = _m4_resolved
            _m4_baseline_required = [
                (a.get("canonical_concept") or a.get("text") or "")
                for a in (anchors_out.get("primary_anchors") or [])
                if isinstance(a, dict) and a.get("required_for_search") is True
            ]
            # Best-effort baseline confidence from anchor confidences.
            _m4_conf_vals = [
                a.get("confidence")
                for a in (anchors_out.get("primary_anchors") or [])
                if isinstance(a, dict) and isinstance(a.get("confidence"), (int, float))
            ]
            _m4_baseline_conf = max(_m4_conf_vals) if _m4_conf_vals else None
            m4_block["baseline"] = {
                "primary_intent": picott_payload.get("type_of_question"),
                "question_subtype": picott_payload.get("question_subtype"),
                "frameworks": list(picott_payload.get("framework_list_json") or []),
                "required_anchors": list(_m4_baseline_required),
                "branch_count": len(picott_payload.get("branches") or []),
                "anchor_failure_reason": anchors_out.get("anchor_failure_reason"),
            }
            if _m4_resolved == "shadow":
                # ---- Phase M4.1.5 — end-to-end M4 latency instrumentation
                # + hard timeout. Closed-set NUMERIC timings only (ms);
                # NO prompts / completions / PHI / evidence text recorded.
                try:
                    _m4_budget_ms = int(_m4_load_cfg().max_latency_ms)
                except Exception:  # noqa: BLE001
                    _m4_budget_ms = 4000
                _m4_budget_s = max(0.1, _m4_budget_ms / 1000.0)
                _m4_break = {
                    "trigger_detect_ms": 0, "llm_candidate_ms": 0,
                    "validation_ms": 0, "would_be_projection_ms": 0,
                    "compiler_projection_ms": 0,
                }
                _t_det0 = time.perf_counter()
                _m4_triggers = _m4_detect(
                    normalized_question=(
                        parsed.get("normalized_question")
                        or sanitized_question or ""
                    ),
                    baseline_primary_intent=picott_payload.get("type_of_question"),
                    baseline_subtype=picott_payload.get("question_subtype"),
                    primary_anchors=list(anchors_out.get("primary_anchors") or []),
                    generic_modifiers=list(anchors_out.get("generic_modifiers") or []),
                    anchor_failure_reason=anchors_out.get("anchor_failure_reason"),
                    baseline_confidence=_m4_baseline_conf,
                )
                _m4_break["trigger_detect_ms"] = int(
                    round((time.perf_counter() - _t_det0) * 1000)
                )
                if _m4_triggers:
                    m4_block["triggered"] = True
                    m4_block["trigger_reasons"] = list(_m4_triggers)
                    _m4_timed_out = False
                    _t_run0 = time.perf_counter()
                    try:
                        _m4_cand, _m4_rej, _m4_lat = await asyncio.wait_for(
                            _m4_run(
                                normalized_question=(
                                    parsed.get("normalized_question")
                                    or sanitized_question or ""
                                ),
                                trigger_reasons=_m4_triggers,
                                deterministic_primary_anchors=list(
                                    anchors_out.get("primary_anchors") or [],
                                ),
                                baseline_required_anchors=list(_m4_baseline_required),
                                diagnostics=diag,
                            ),
                            timeout=_m4_budget_s,
                        )
                    except (asyncio.TimeoutError, TimeoutError):
                        _m4_timed_out = True
                        _m4_cand, _m4_rej = None, ["m4_latency_timeout"]
                        _m4_lat = int(_m4_budget_ms)
                        logger.warning(
                            "phaseM4 bounded-LLM interpreter timed out "
                            "session_id=%s budget_ms=%d — falling back",
                            session_id, _m4_budget_ms,
                        )
                    _m4_run_total_ms = int(
                        round((time.perf_counter() - _t_run0) * 1000)
                    )
                    _m4_break["llm_candidate_ms"] = int(_m4_lat or 0)
                    _m4_break["validation_ms"] = max(
                        0, _m4_run_total_ms - int(_m4_lat or 0)
                    )
                    m4_block["latency_ms"] = int(_m4_lat or 0)
                    m4_block["timed_out"] = bool(_m4_timed_out)
                    if _m4_timed_out:
                        m4_block["validation_status"] = "timeout"
                        m4_block["accepted"] = False
                        m4_block["rejection_reasons"] = ["m4_latency_timeout"]
                    elif _m4_cand is not None and not _m4_rej:
                        m4_block["validation_status"] = "passed"
                        m4_block["accepted"] = True
                        m4_block["candidate"] = _m4_cand.model_dump(mode="json")
                        _t_proj0 = time.perf_counter()
                        m4_block["would_be_compiler_input"] = _m4_project(
                            candidate=_m4_cand,
                            baseline_primary_intent=picott_payload.get("type_of_question"),
                            baseline_frameworks=list(
                                picott_payload.get("framework_list_json") or []
                            ),
                            baseline_required_anchors=list(_m4_baseline_required),
                            baseline_branch_count=len(
                                picott_payload.get("branches") or []
                            ),
                        )
                        _m4_break["would_be_projection_ms"] = int(
                            round((time.perf_counter() - _t_proj0) * 1000)
                        )
                        # Phase M4.0.1 — full compiler-input contract projection
                        # (structured slots + dry-run query families + validation).
                        _t_cproj0 = time.perf_counter()
                        m4_block["m4_0_1_projection"] = _m4_0_1_build(
                            _m4_cand,
                            baseline_primary_intent=picott_payload.get("type_of_question"),
                            baseline_frameworks=list(
                                picott_payload.get("framework_list_json") or []
                            ),
                            baseline_required_anchors=list(_m4_baseline_required),
                        )
                        _m4_break["compiler_projection_ms"] = int(
                            round((time.perf_counter() - _t_cproj0) * 1000)
                        )
                    else:
                        m4_block["validation_status"] = "rejected"
                        m4_block["rejection_reasons"] = list(_m4_rej or [])
                m4_block["latency_breakdown"] = dict(_m4_break)
                m4_block["upstream_latency_ms"] = int(sum(_m4_break.values()))
        except Exception:  # noqa: BLE001
            logger.exception("phaseM4 full-question interpreter shadow failed (non-fatal)")
        parsed["session_diagnostics_json"]["m4_full_question_interpreter"] = dict(m4_block)
        # Admin-only sidecar (append-only, TTL 30d) — only when triggered.
        if m4_block.get("triggered"):
            try:
                from services.m4_shadow_sidecar import write_m4_sidecar
                await write_m4_sidecar(
                    session_id=session_id,
                    user_id=user_id,
                    m4_block=dict(m4_block),
                )
            except Exception:  # noqa: BLE001
                logger.exception("phaseM4 sidecar write failed (non-fatal)")

        # ---------------------------------------------------------
        # Phase M3.2 — Preview-only Active Semantic Repair decision.
        #
        # Runs AFTER the M-phase shadow block and AFTER the M3-preflight
        # eligibility/projection have been recorded. Decides whether
        # to PROMOTE the validator-accepted candidate into the
        # clinician-visible active retrieval pipeline. PREVIEW ONLY —
        # production fail-closed via the startup guard.
        #
        # Two-source kill switch (env OR mongo). K4A.2/3/4 routes are
        # always excluded. Hard timeout = config.max_latency_ms
        # (default 3500 ms). On any failure / timeout / non-eligible
        # → A-arm CQM is preserved unchanged.
        # ---------------------------------------------------------
        m3_2_decision: dict[str, Any] = {
            "applied": False,
            "decision": "deprecated_consolidated_into_m4",
            "shape": None,
            "timeout_status": "none",
            "fallback_status": "none",
            "latency_ms": 0,
            # Phase M4.2 — static DORMANT kill state so the M4 co-fire
            # guard continues to see M3.2 as dormant (combined_engaged)
            # without invoking the deprecated M3.2 runtime.
            "kill_switch_state": {
                "env_kill_switch_engaged": True,
                "mongo_kill_switch_engaged": True,
                "combined_engaged": True,
                "fail_closed_default": True,
            },
            "chip": None,
            "deprecated": True,
        }
        m3_2_cb_records: list[dict[str, Any]] = []
        if _LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED:
            try:
                from services.semantic_repair_active_pipeline import (
                    run_m3_2_active_decision as _m3_2_run,
                )
                # Snapshot the A-arm CQM (preview shape) and the deterministic
                # primary anchors for the audit sidecar BEFORE any
                # potential injection. The CQM payload here mirrors the
                # M3.1 serializer's ``cqm`` projection so the sidecar
                # stays comparable with the M3.1 benchmark surface.
                _a_arm_cqm_snapshot = {
                    "primary_intent": picott_payload.get("type_of_question"),
                    "question_subtype": picott_payload.get("question_subtype"),
                    "frameworks": list(
                        picott_payload.get("framework_list_json") or []
                    ),
                    "specialty": picott_payload.get("specialty"),
                    "required_anchors": [
                        (a.get("canonical_concept") or a.get("text") or "")
                        for a in (anchors_out.get("primary_anchors") or [])
                        if isinstance(a, dict)
                        and a.get("required_for_search") is True
                    ],
                }
                m3_2_decision = await _m3_2_run(
                    session_id=session_id,
                    user_id=user_id,
                    semantic_repair_block=semantic_repair_block,
                    picott_payload=picott_payload,
                    deterministic_primary_anchors=list(
                        anchors_out.get("primary_anchors") or [],
                    ),
                    sources=list(_KNOWN_RETRIEVAL_SOURCES),
                    a_arm_cqm=_a_arm_cqm_snapshot,
                    # A-arm boolean queries are not yet known at this
                    # point in the pipeline (cb_queries are built later);
                    # the sidecar's ``a_arm_boolean_queries`` is updated
                    # in-place after compilation completes. We start
                    # with an empty list so the field is always present.
                    a_arm_boolean_queries=[],
                )
                if m3_2_decision.get("applied"):
                    m3_2_cb_records = list(m3_2_decision.get("m_repaired_cb_records") or [])
            except Exception:  # noqa: BLE001
                logger.exception(
                    "phaseM3_2 active decision failed (non-fatal); A-arm preserved",
                )
                m3_2_decision = {
                    "applied": False,
                    "decision": "ineligible_unknown_clinical_block",
                    "shape": None,
                    "timeout_status": "none",
                    "fallback_status": "fallback_to_a_arm",
                    "latency_ms": 0,
                    "kill_switch_state": None,
                    "chip": None,
                }
                m3_2_cb_records = []

        # Persist the chip/diagnostic block under
        # ``session_diagnostics_json.m3_2_active_repair``. PHI / leak
        # discipline: only closed-set, deterministically-derived
        # fields land here; the LLM candidate completion / prompt
        # never enters this surface.
        parsed["session_diagnostics_json"]["m3_2_active_repair"] = {
            "applied": bool(m3_2_decision.get("applied")),
            "decision": m3_2_decision.get("decision"),
            "shape": m3_2_decision.get("shape"),
            "timeout_status": m3_2_decision.get("timeout_status"),
            "fallback_status": m3_2_decision.get("fallback_status"),
            "latency_ms": int(m3_2_decision.get("latency_ms") or 0),
            "kill_switch_state": m3_2_decision.get("kill_switch_state"),
            "chip": m3_2_decision.get("chip"),
            "route_owned_by_k4a_specialty": bool(
                m3_2_decision.get("route_owned_by_k4a_specialty")
            ),
            "sidecar_doc_id": m3_2_decision.get("sidecar_doc_id"),
            "deprecated": bool(m3_2_decision.get("deprecated")),
            "superseded_by": "m4_full_question_interpreter",
            "phase": "M3.2",
        }

        # ---------------------------------------------------------
        # Phase M4.1 / M4.1.1 — Preview-only ACTIVE M4.
        #
        # Two-step, deterministic: (1) HERE, while the baseline anchors
        # are in scope, compute the closed-set BASELINE-DEGRADATION
        # assessment (M4.1.1). (2) Below at the cb_queries swap — once
        # the baseline query families are known — finalize the active
        # PROMOTION decision so we can enforce the query-family
        # preservation guard (do NOT promote a clean baseline and
        # narrow its families). M4 still runs SHADOW regardless.
        #
        # PREVIEW/DEV ONLY — production hard-off via the mode resolver
        # (no candidate exists), the env-allowlist gate, and the
        # startup guard. Two-source kill switch; ships engaged. M3.2
        # stays dormant — if it applied OR is not dormant, M4.1 blocks
        # with ``m3_2_already_applied_or_not_dormant``. Any non-eligible
        # / failure path → baseline retrieval preserved unchanged.
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # Phase M4.1 — explicit K4A protected-route exclusion. Resolve
        # the route deterministically from the parser's question_subtype
        # (defense-in-depth: also honor an upstream M3 route hint). A
        # K4A-owned route NEVER promotes M4 active queries and the
        # decision/diagnostics surface ``route_owned_by_k4a_specialty``
        # explicitly — independent of M4 candidate acceptance.
        from services.m4_active_pipeline import (
            resolve_active_route as _m4_resolve_route,
        )
        _m4_route_info = _m4_resolve_route(
            question_subtype=(picott_payload or {}).get("question_subtype"),
            route_owned_hint=bool(
                m3_2_decision.get("route_owned_by_k4a_specialty")
            ),
        )
        _m4_route_owned = bool(_m4_route_info["route_owned_by_k4a_specialty"])
        m4_active_decision: dict[str, Any] = {
            "applied": False,
            "decision": (
                "route_owned_by_k4a_specialty" if _m4_route_owned
                else "ineligible_candidate_not_accepted"
            ),
            "kill_switch_state": None,
            "chip": None,
            "sidecar_doc_id": None,
            "active_route_id": _m4_route_info["active_route_id"],
            "protected_route_family": _m4_route_info["protected_route_family"],
            "route_owned_by_k4a_specialty": _m4_route_owned,
            "m4_block_reasons": (
                ["route_owned_by_k4a_specialty"] if _m4_route_owned
                else ["ineligible_candidate_not_accepted"]
            ),
            "active_query_rewritten": False,
        }
        m4_active_cb_records: list[dict[str, Any]] = []
        m4_baseline_degradation: dict[str, Any] = {
            "baseline_degraded": False, "reasons": [],
        }
        _m4_baseline_required_anchors = [
            (a.get("canonical_concept") or a.get("text") or "")
            for a in (anchors_out.get("primary_anchors") or [])
            if isinstance(a, dict) and a.get("required_for_search") is True
        ]
        try:
            if m4_block.get("accepted"):
                from services.m4_active_pipeline import (
                    assess_baseline_degradation as _m4_assess_degradation,
                )
                m4_baseline_degradation = _m4_assess_degradation(
                    baseline_required_anchors=_m4_baseline_required_anchors,
                    baseline_framework=list(
                        picott_payload.get("framework_list_json") or []
                    ),
                    baseline_intent=picott_payload.get("type_of_question"),
                    baseline_confidence=picott_payload.get("confidence"),
                    m4_candidate=m4_block.get("candidate") or {},
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseM4_1_1 baseline-degradation assessment failed (non-fatal)",
            )
            m4_baseline_degradation = {"baseline_degraded": False, "reasons": []}

        # Phase M4.1.3 — flag set when the no-primary-entity rescue has
        # already produced the active decision (so the swap below does
        # not recompute / double-write the sidecar).
        _m4_no_primary_rescue_decided = False

        if anchors_out.get("anchor_failure_reason") == "no_primary_clinical_entity":
            # -------------------------------------------------------
            # Phase M4.1.3 — No-Primary-Entity Rescue (PREVIEW only).
            # PHI / nonmedical / generic-only guards already ran
            # upstream. If an ACCEPTED + validated M4 candidate provides
            # a clean, non-generic clinical anchor for this clinically
            # meaningful question, let M4 active drive retrieval INSTEAD
            # of clarifying — but ONLY when the kill switch is disengaged
            # in preview (enforced inside run_m4_active_decision). Any
            # failure/ineligibility falls cleanly back to clarification.
            # -------------------------------------------------------
            try:
                if m4_block.get("accepted") and not _m4_route_owned:
                    from services.m4_active_pipeline import (
                        is_m4_no_primary_rescue_eligible as _m4_rescue_eligible,
                        run_m4_active_decision as _m4_active_run,
                    )
                    _resc_ok, _resc_reason = _m4_rescue_eligible(m4_block)
                    if _resc_ok:
                        _m3_2_kill_state = (
                            m3_2_decision.get("kill_switch_state") or {}
                        )
                        _m3_2_combined = (
                            bool(_m3_2_kill_state.get("combined_engaged"))
                            if isinstance(_m3_2_kill_state, dict) else None
                        )
                        _m4_a_arm_cqm = {
                            "primary_intent": picott_payload.get("type_of_question"),
                            "question_subtype": picott_payload.get("question_subtype"),
                            "frameworks": list(
                                picott_payload.get("framework_list_json") or []
                            ),
                            "specialty": picott_payload.get("specialty"),
                            "required_anchors": [],
                        }
                        m4_active_decision = await _m4_active_run(
                            session_id=session_id,
                            user_id=user_id,
                            m4_block=m4_block,
                            picott_payload=picott_payload,
                            sources=list(_KNOWN_RETRIEVAL_SOURCES),
                            a_arm_cqm=_m4_a_arm_cqm,
                            m3_2_applied=bool(m3_2_decision.get("applied")),
                            m3_2_kill_combined_engaged=_m3_2_combined,
                            baseline_degraded=True,
                            baseline_degradation_reasons=[
                                "no_clean_primary_entity_in_baseline",
                                _resc_reason,
                            ],
                            baseline_query_family_count=None,
                            a_arm_boolean_queries=[],
                            rescue_path="no_primary_entity",
                            normalized_question=(
                                parsed.get("normalized_question")
                                or sanitized_question or ""
                            ),
                            upstream_latency_ms=int(
                                m4_block.get("upstream_latency_ms") or 0
                            ),
                            upstream_latency_breakdown=dict(
                                m4_block.get("latency_breakdown") or {}
                            ),
                            upstream_timed_out=bool(m4_block.get("timed_out")),
                        )
                        if m4_active_decision.get("applied"):
                            m4_active_cb_records = list(
                                m4_active_decision.get("cb_records") or []
                            )
                            _m4_no_primary_rescue_decided = True
                            logger.info(
                                "phaseM4_1_3 no_primary rescue APPLIED "
                                "session_id=%s reason=%s n_queries=%d",
                                session_id, _resc_reason,
                                len(m4_active_cb_records),
                            )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "phaseM4_1_3 no_primary rescue failed (non-fatal); "
                    "clarifying",
                )
                _m4_no_primary_rescue_decided = False

            if _m4_no_primary_rescue_decided:
                # Rescue applies — record clinician-safe diag block and
                # FALL THROUGH to the retrieval path (do NOT clarify).
                parsed["session_diagnostics_json"]["m4_active"] = {
                    "applied": True,
                    "decision": m4_active_decision.get("decision"),
                    "chip": m4_active_decision.get("chip"),
                    "kill_switch_state": m4_active_decision.get("kill_switch_state"),
                    "sidecar_doc_id": m4_active_decision.get("sidecar_doc_id"),
                    "baseline_degraded": True,
                    "baseline_degradation_reasons": (
                        m4_active_decision.get("baseline_degradation_reasons")
                    ),
                    "rescue_path": "no_primary_entity",
                    "timed_out": False,
                    "fallback_reason": None,
                    "m4_total_latency_ms": int(
                        m4_active_decision.get("m4_total_latency_ms") or 0
                    ),
                    "latency_breakdown": dict(
                        m4_active_decision.get("latency_breakdown") or {}
                    ),
                    "m4_specificity": dict(
                        m4_active_decision.get("m4_specificity") or {}
                    ),
                    "phase": "M4.1.3",
                }

        if (anchors_out.get("anchor_failure_reason") == "no_primary_clinical_entity"
                and not _m4_no_primary_rescue_decided):
            from services.primary_clinical_anchor import CLARIFICATION_MESSAGE
            # Phase K1 — build ClinicalQuestionMap (clarification path).
            try:
                _cqm_signal = _cqm_signal_router.analyze(
                    normalized_for_preflight
                ).model_dump(mode="json")
            except Exception:  # noqa: BLE001
                _cqm_signal = {}
            try:
                _cqm_json = _k1_build_clinical_question_map(
                    normalized_question=(
                        parsed.get("normalized_question")
                        or sanitized_question or ""
                    ),
                    original_question=sanitized_question or "",
                    picott_payload=picott_payload,
                    anchors_out=anchors_out,
                    resolver_diag=resolver_diag,
                    signal_summary=_cqm_signal,
                    clarification_only=True,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "phaseK1 clarification CQM build failed (non-fatal)"
                )
                _cqm_json = None
            await _set_status(
                session_id,
                status="awaiting_clinical_subject",
                extras={
                    "pico_json": parsed,
                    "picott_json": picott_payload,
                    "clinical_question_type": parsed.get("clinical_question_type"),
                    "topic_label": topic_label,
                    "primary_clinical_anchors_json": dict(anchors_out),
                    "clarification_message_text": CLARIFICATION_MESSAGE,
                    # Persist the context-engine outputs too so the
                    # transparency panel still has provenance to show.
                    "ontology_mappings_json": list(context.get("ontology_mappings") or []),
                    "ontology_lifts_json": list(context.get("ontology_lifts") or []),
                    "role_repairs_json": list(context.get("role_repairs") or []),
                    "role_weights_json": dict(context.get("role_weights") or {}),
                    "multi_intent_json": dict(context.get("multi_intent") or {}),
                    "shorthand_substitutions_json": list(context.get("shorthand_substitutions") or []),
                    # Phase K1 — additive ClinicalQuestionMap.
                    "clinical_question_map_json": _cqm_json,
                },
            )
            try:
                await audit_service.log(
                    "search_session_anchor_clarification_required",
                    user_id=user_id,
                    search_session_id=session_id,
                    payload={
                        # Closed-set fields ONLY. No question text. No
                        # rationale. No PMIDs. No URLs.
                        "anchor_failure_reason": "no_primary_clinical_entity",
                        "generic_modifier_count": len(
                            anchors_out.get("generic_modifiers") or []
                        ),
                        "anchor_repair_applied": bool(
                            anchors_out.get("anchor_repair_applied")
                        ),
                        "phase": "phase6f_closeout",
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "phase6f anchor_clarification audit failed (non-fatal)"
                )
            # Terminate the pipeline here. NO PubMed, NO Europe PMC,
            # NO OpenAlex, NO scholar-like expansion, NO LLM.
            return

        # Determine enabled retrieval sources.
        sources = await _enabled_sources()
        oa_enrich = await _openalex_enabled()

        # Phase K1 — build ClinicalQuestionMap (searching path).
        try:
            _cqm_signal_search = _cqm_signal_router.analyze(
                normalized_for_preflight
            ).model_dump(mode="json")
        except Exception:  # noqa: BLE001
            _cqm_signal_search = {}
        try:
            _cqm_json_search = _k1_build_clinical_question_map(
                normalized_question=(
                    parsed.get("normalized_question") or sanitized_question or ""
                ),
                original_question=sanitized_question or "",
                picott_payload=picott_payload,
                anchors_out=anchors_out,
                resolver_diag=resolver_diag,
                signal_summary=_cqm_signal_search,
                clarification_only=False,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseK1 searching CQM build failed (non-fatal)"
            )
            _cqm_json_search = None

        await _set_status(
            session_id,
            status="searching",
            extras={
                "pico_json": parsed,
                "picott_json": picott_payload,
                "clinical_question_type": parsed.get("clinical_question_type"),
                "topic_label": topic_label,
                "sources_searched": sources,
                "openalex_enriched": False,
                # Phase 5F — context-engine outputs persisted on the
                # session document so GET /api/search-sessions/{id} can
                # return them to the transparency panel and the tester.
                "shorthand_substitutions_json": list(context.get("shorthand_substitutions") or []),
                "ontology_mappings_json": list(context.get("ontology_mappings") or []),
                "ontology_lifts_json": list(context.get("ontology_lifts") or []),
                "role_repairs_json": list(context.get("role_repairs") or []),
                "role_weights_json": dict(context.get("role_weights") or {}),
                "multi_intent_json": dict(context.get("multi_intent") or {}),
                # Phase 6F — primary clinical entity anchors + the
                # closed-set list of generic modifiers detected in the
                # question. Persisted so the transparency panel can
                # render the anchor list and so the picott_aware_ranker
                # can apply ``wrong_drug`` / ``wrong_condition`` caps
                # against the authoritative anchor list. Closed-set only.
                "primary_clinical_anchors_json": dict(
                    parsed.get("primary_clinical_anchors_json") or {}
                ),
                # Phase K1 — additive ClinicalQuestionMap.
                "clinical_question_map_json": _cqm_json_search,
            },
        )

        # ---- Phase 6B.1 — BooleanQueryCompiler is the PRIMARY query producer.
        # Legacy ``search_planning.plan()`` is now FALLBACK-ONLY, invoked per
        # source when (a) compiler emitted zero valid queries for that source,
        # OR (b) high-precision compiler queries returned zero results.
        # The retrieval_quality_evaluator's anchor-coverage check (criterion c)
        # is handled later via the existing rescue path (planner_source=
        # ``rescue``).
        db = get_db()
        role_weights_json = dict(context.get("role_weights") or {})

        # ---- Step 1 — Build all PRIMARY (compiler) queries ----
        cb_queries: list[dict[str, Any]] = []
        # Phase 5D — per-source counters for diagnostics. Retained for
        # backwards-compat with admin diagnostics that already read this key.
        phase5d_metrics: dict[str, Any] = {
            "legacy_queries_by_source": {s: 0 for s in sources},
            "phase5d_queries_emitted_by_source": {s: 0 for s in sources},
            "phase5d_queries_dropped_by_source": {s: 0 for s in sources},
            "compiler_yielded_no_queries_sources": [],
        }
        compiler_query_counts_by_source: dict[str, int] = {s: 0 for s in sources}

        # ---- AI clinical-synonym expansion (default-OFF; see
        #      services/ai_query_expansion). Broaden the primary anchor(s) into a
        #      controlled-vocabulary OR-group ("stage 4 prostate cancer" ->
        #      prostate cancer / metastatic / advanced / stage IV / CRPC / mCRPC)
        #      so an over-specific or staged phrase still retrieves the right
        #      papers. Terms ride as OR-within-concept ``synonyms`` that the
        #      builder already expands; AND-across-concept gating preserves
        #      precision. Fully gated by AI_QUERY_EXPANSION_ENABLED. ----
        if (picott_payload and picott_payload.get("must_have_concepts")
                and ai_query_expansion.is_enabled()):
            try:
                _expand_roles = {"condition", "population", "intervention", "exposure"}
                _expanded = 0
                for _mh in picott_payload["must_have_concepts"]:
                    if _expanded >= 3:
                        break
                    if not isinstance(_mh, dict):
                        continue
                    if (_mh.get("role") or "").lower() not in _expand_roles:
                        continue
                    _new = await ai_query_expansion.expand_concept_synonyms(
                        _mh.get("concept") or ""
                    )
                    if _new:
                        _have = {
                            s.strip().lower()
                            for s in (_mh.get("synonyms") or [])
                            if isinstance(s, str)
                        }
                        _mh["synonyms"] = list(_mh.get("synonyms") or []) + [
                            s for s in _new if s.strip().lower() not in _have
                        ]
                        _expanded += 1
                if _expanded:
                    logger.info(
                        "ai_query_expansion: expanded %d anchor concept(s) session=%s",
                        _expanded, session_id,
                    )
            except Exception:  # noqa: BLE001
                logger.exception("ai_query_expansion failed (non-fatal)")

        if picott_payload and picott_payload.get("must_have_concepts"):
            with diag.stage("boolean_query_compilation"):
                try:
                    for cbq in concept_block_query_builder.build(
                        picott_payload, sources=sources,
                    ):
                        rec = cbq.to_query_record()
                        cb_queries.append(rec)
                        phase5d_metrics["legacy_queries_by_source"][rec["source"]] = (
                            phase5d_metrics["legacy_queries_by_source"].get(rec["source"], 0) + 1
                        )
                except Exception:  # noqa: BLE001
                    logger.exception("concept-block query builder failed (non-fatal)")
                # ---- Phase 5D: 3 new families (mesh_keyword, clinical_queries,
                #      trial_acronym) on top of the 6 Phase 5A families. ----
                phase5d_emitted_by_src: dict[str, int] = {s: 0 for s in sources}
                try:
                    for compiled in boolean_query_compiler.compile(
                        picott_payload, sources=sources,
                    ):
                        if not compiled.syntax_validation.valid:
                            phase5d_metrics["phase5d_queries_dropped_by_source"][compiled.source] = (
                                phase5d_metrics["phase5d_queries_dropped_by_source"].get(compiled.source, 0) + 1
                            )
                            logger.info(
                                "phase5d_query_skipped",
                                extra={
                                    "source": compiled.source,
                                    "family": compiled.query_family,
                                    "errors": compiled.syntax_validation.errors,
                                },
                            )
                            continue
                        rec = compiled.to_concept_block_query().to_query_record()
                        rec["field_tags_used"] = list(compiled.field_tags_used)
                        rec["mesh_terms_used"] = list(compiled.mesh_terms_used)
                        if compiled.cq_category:
                            rec["cq_category"] = compiled.cq_category
                            rec["cq_scope"] = compiled.cq_scope
                        cb_queries.append(rec)
                        phase5d_emitted_by_src[compiled.source] = phase5d_emitted_by_src.get(compiled.source, 0) + 1
                except Exception:  # noqa: BLE001
                    logger.exception("phase5d boolean_query_compiler failed (non-fatal)")
            phase5d_metrics["phase5d_queries_emitted_by_source"] = phase5d_emitted_by_src

        # ---- Phase M3.2 — Active Semantic Repair query swap.
        # When the M3.2 decision returned ``applied=True``, the
        # validator-accepted candidate's M-repaired queries REPLACE
        # the A-arm ``cb_queries`` set for clinician-visible
        # retrieval. The A-arm queries are captured below into the
        # audit sidecar BEFORE the swap so the operator can audit
        # the diff at any time.
        # ----
        a_arm_boolean_queries_for_sidecar = sorted({
            q["query_string"] for q in cb_queries
            if q.get("query_string")
        })
        # ---- Phase M4.1 / M4.1.1 — finalize the ACTIVE promotion
        # decision now that the baseline query families are known. The
        # M4.1.1 guard inside run_m4_active_decision blocks promotion
        # when the baseline is NOT degraded (``baseline_already_clean``)
        # or when promotion would narrow query-family diversity without
        # a real baseline defect
        # (``query_family_diversity_would_decrease_without_baseline_defect``).
        # Runs only when M4 accepted; M3.2 precedence still applies. ----
        try:
            if m4_block.get("accepted") and not _m4_no_primary_rescue_decided and not _m4_route_owned:
                from services.m4_active_pipeline import (
                    run_m4_active_decision as _m4_active_run,
                )
                _m3_2_kill_state = m3_2_decision.get("kill_switch_state") or {}
                _m3_2_combined_engaged = (
                    bool(_m3_2_kill_state.get("combined_engaged"))
                    if isinstance(_m3_2_kill_state, dict) else None
                )
                _m4_a_arm_cqm = {
                    "primary_intent": picott_payload.get("type_of_question"),
                    "question_subtype": picott_payload.get("question_subtype"),
                    "frameworks": list(
                        picott_payload.get("framework_list_json") or []
                    ),
                    "specialty": picott_payload.get("specialty"),
                    "required_anchors": list(_m4_baseline_required_anchors),
                }
                _baseline_family_count = len({
                    (q.get("query_family") or q.get("query_type"))
                    for q in cb_queries
                    if (q.get("query_family") or q.get("query_type"))
                })
                m4_active_decision = await _m4_active_run(
                    session_id=session_id,
                    user_id=user_id,
                    m4_block=m4_block,
                    picott_payload=picott_payload,
                    sources=list(_KNOWN_RETRIEVAL_SOURCES),
                    a_arm_cqm=_m4_a_arm_cqm,
                    m3_2_applied=bool(m3_2_decision.get("applied")),
                    m3_2_kill_combined_engaged=_m3_2_combined_engaged,
                    baseline_degraded=bool(
                        m4_baseline_degradation.get("baseline_degraded")
                    ),
                    baseline_degradation_reasons=list(
                        m4_baseline_degradation.get("reasons") or []
                    ),
                    baseline_query_family_count=_baseline_family_count,
                    a_arm_boolean_queries=a_arm_boolean_queries_for_sidecar,
                    normalized_question=(
                        parsed.get("normalized_question")
                        or sanitized_question or ""
                    ),
                    upstream_latency_ms=int(
                        m4_block.get("upstream_latency_ms") or 0
                    ),
                    upstream_latency_breakdown=dict(
                        m4_block.get("latency_breakdown") or {}
                    ),
                    upstream_timed_out=bool(m4_block.get("timed_out")),
                )
                if m4_active_decision.get("applied"):
                    m4_active_cb_records = list(
                        m4_active_decision.get("cb_records") or []
                    )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseM4_1 active decision failed (non-fatal); baseline preserved",
            )
            m4_active_decision = {
                "applied": False,
                "decision": "ineligible_internal_error",
                "kill_switch_state": None,
                "chip": None,
                "sidecar_doc_id": None,
            }
            m4_active_cb_records = []
        # Clinician-safe + diagnostic block. The clinician GET further
        # restricts this to ``{applied, chip}`` only. NO prompts / NO
        # completions / NO PHI / NO LLM specifics. Skipped when the
        # M4.1.3 no-primary rescue already set this block above.
        if not _m4_no_primary_rescue_decided:
            _m4_timed_out_flag = bool(m4_block.get("timed_out"))
            _m4_decision_str = (
                "m4_latency_timeout"
                if (_m4_timed_out_flag and not _m4_route_owned)
                else m4_active_decision.get("decision")
            )
            parsed["session_diagnostics_json"]["m4_active"] = {
                "applied": bool(m4_active_decision.get("applied")),
                "decision": _m4_decision_str,
                "active_route_id": m4_active_decision.get("active_route_id"),
                "protected_route_family": m4_active_decision.get(
                    "protected_route_family"
                ),
                "route_owned_by_k4a_specialty": bool(
                    m4_active_decision.get("route_owned_by_k4a_specialty")
                ),
                "m4_block_reasons": list(
                    m4_active_decision.get("m4_block_reasons") or []
                ),
                "active_query_rewritten": bool(
                    m4_active_decision.get("active_query_rewritten")
                ),
                "chip": m4_active_decision.get("chip"),
                "kill_switch_state": m4_active_decision.get("kill_switch_state"),
                "sidecar_doc_id": m4_active_decision.get("sidecar_doc_id"),
                "baseline_degraded": bool(
                    m4_baseline_degradation.get("baseline_degraded")
                ),
                "baseline_degradation_reasons": list(
                    m4_baseline_degradation.get("reasons") or []
                ),
                "timed_out": _m4_timed_out_flag,
                "fallback_reason": (
                    "m4_latency_timeout" if _m4_timed_out_flag else None
                ),
                "m4_total_latency_ms": int(
                    m4_active_decision.get("m4_total_latency_ms")
                    if m4_active_decision.get("m4_total_latency_ms") is not None
                    else (m4_block.get("upstream_latency_ms") or 0)
                ),
                "latency_breakdown": dict(
                    m4_active_decision.get("latency_breakdown")
                    or m4_block.get("latency_breakdown") or {}
                ),
                "m4_specificity": dict(
                    m4_active_decision.get("m4_specificity") or {}
                ),
                "phase": "M4.1.1",
            }
        if m3_2_decision.get("applied") and m3_2_cb_records:
            logger.info(
                "phaseM3_2_active_apply session_id=%s shape=%s "
                "n_m_repaired_queries=%d n_a_arm_queries=%d "
                "decision=%s",
                session_id, m3_2_decision.get("shape"),
                len(m3_2_cb_records),
                len(cb_queries),
                m3_2_decision.get("decision"),
            )
            cb_queries = list(m3_2_cb_records)
        elif m4_active_decision.get("applied") and m4_active_cb_records:
            # ---- Phase M4.5 — ADDITIVE diversity-preserving rescue.
            # M4 no longer REPLACES the baseline families. The baseline
            # query families are PRESERVED and the validated M4 rescue
            # families are ADDED (deduped by source+query_string). Final
            # strategy = baseline + m4_rescue, so query-family diversity
            # is preserved or increased. (No-primary rescue: the baseline
            # built no families, so this reduces to the rescue families
            # only — unchanged from M4.1.3.) ----
            _mode = (
                m4_active_decision.get("application_mode") or "additive_rescue"
            )
            for _bq in cb_queries:
                _bq.setdefault("query_family_source", "baseline")
                _bq.setdefault("m4_rescue", False)
                _bq.setdefault("m4_rescue_reason", None)
            _seen_q = {
                (q["source"], q["query_string"]) for q in cb_queries
            }
            _n_baseline = len(cb_queries)
            _n_added = 0
            for _rec in m4_active_cb_records:
                _key = (_rec.get("source"), _rec.get("query_string"))
                if _key in _seen_q:
                    continue
                _seen_q.add(_key)
                cb_queries.append(_rec)
                _n_added += 1
            logger.info(
                "phaseM4_5_additive_rescue_apply session_id=%s mode=%s "
                "n_baseline=%d n_m4_rescue_added=%d decision=%s",
                session_id, _mode, _n_baseline, _n_added,
                m4_active_decision.get("decision"),
            )
        # Always update the M3.2 sidecar with the A-arm Boolean
        # query strings, regardless of applied/fallback. We use a
        # bounded ``$set`` on the just-inserted sidecar row so
        # subsequent audit/admin reads see the canonical A-arm
        # snapshot. Non-fatal on failure.
        _m3_2_sidecar_doc_id = m3_2_decision.get("sidecar_doc_id")
        if _m3_2_sidecar_doc_id:
            try:
                await db[Collections.M3_2_SESSION_DECISIONS].update_one(
                    {"id": _m3_2_sidecar_doc_id},
                    {"$set": {
                        "a_arm_boolean_queries": (
                            a_arm_boolean_queries_for_sidecar
                        ),
                    }},
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "m3_2_sidecar a_arm_boolean_queries update failed "
                    "(non-fatal) session_id=%s", session_id,
                )

        # ---- Step 2 — Persist PRIMARY queries (planner_source=
        # "boolean_compiler", is_primary_strategy=True) ----
        for q in cb_queries:
            src = q["source"]
            compiler_query_counts_by_source[src] = (
                compiler_query_counts_by_source.get(src, 0) + 1
            )
            await db[Collections.SEARCH_QUERIES].insert_one({
                "id": uuid.uuid4().hex,
                "search_session_id": session_id,
                "source": src,
                # Prefix the legacy ``query_type`` column so concept-block
                # queries don't collide with legacy planner families
                # (``high_precision`` is both a legacy and a cb name).
                "query_type": f"cb_{q['query_family']}",
                "query_family": q["query_family"],
                # Phase 5B Task 7 — candidate tier (closed set).
                "candidate_tier": q["query_family"],
                "query_string": q["query_string"],
                "rationale": q.get("rationale") or "",
                # Phase 5A legacy field name (kept for BC with older tests).
                "concept_blocks_json": q.get("concept_blocks_json") or {},
                # Phase 6B.1 — canonical ``*_used`` variants.
                "concept_blocks_used_json": q.get("concept_blocks_json") or {},
                "mandatory_anchors_json": q.get("mandatory_anchors") or [],
                "optional_blocks_json": q.get("optional_terms") or [],
                "optional_terms_json": q.get("optional_terms") or [],
                "role_weights_json": role_weights_json,
                # Phase 6B.1 — planner provenance.
                # Phase M3.2 — when M-repair active is applied, the
                # records arrive with ``planner_source="semantic_repair_active"``;
                # honor that. Default to ``boolean_compiler`` for the
                # standard A-arm path.
                "planner_source": q.get("planner_source") or "boolean_compiler",
                "is_primary_strategy": True,
                "is_fallback_strategy": False,
                "fallback_reason": None,
                # Phase M4.5 — additive-rescue provenance (closed-set).
                "query_family_source": q.get("query_family_source") or "baseline",
                "m4_rescue": bool(q.get("m4_rescue")),
                "m4_rescue_reason": q.get("m4_rescue_reason"),
                "max_results": int(q.get("max_results") or 50),
                # Phase 5D — field tags + MeSH provenance + Clinical Queries scope.
                "field_tags_used_json": q.get("field_tags_used") or [],
                "mesh_terms_used_json": q.get("mesh_terms_used") or [],
                "cq_category": q.get("cq_category"),
                "cq_scope": q.get("cq_scope"),
                "created_at": datetime.now(timezone.utc),
            })

        # ---- Step 3 — Retrieve in PARALLEL across sources (PRIMARY ONLY) ----
        await audit_service.log("search_stage_retrieve_start", user_id=user_id, search_session_id=session_id)

        plans_by_source: dict[str, list[dict[str, Any]]] = {s: [] for s in sources}
        for q in cb_queries:
            src = q["source"]
            if src in plans_by_source:
                plans_by_source[src].append({
                    "query_type": f"cb_{q['query_family']}",
                    "query_string": q["query_string"],
                })

        async def _retrieve(src: str, src_plans: list[dict[str, Any]]):
            # Phase 6F.2 — per-source latency under the closed-set stage
            # names ``pubmed_retrieval`` / ``europe_pmc_retrieval`` /
            # ``openalex_retrieval``. Sources outside this set are
            # measured but bucketed into ``pubmed_retrieval`` by default
            # to keep the key set closed (defensive — today only
            # pubmed/epmc/openalex are exercised).
            _stage = {
                "pubmed": "pubmed_retrieval",
                "europe_pmc": "europe_pmc_retrieval",
                "openalex": "openalex_retrieval",
            }.get(src, "pubmed_retrieval")
            try:
                with diag.stage(_stage):
                    return src, await literature_retrieval.retrieve(
                        src, src_plans,
                        client=http_client,
                        search_session_id=session_id,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("source %s failed", src)
                await audit_service.log(
                    "source_unavailable",
                    user_id=user_id,
                    search_session_id=session_id,
                    payload={"source": src, "error": str(exc)[:300]},
                )
                return src, []

        retrieval_results = await asyncio.gather(
            *[_retrieve(s, plans_by_source[s]) for s in sources]
        )

        # ---- Step 4 — Per-source fallback criteria check ----
        # Closed-set criteria:
        #   (a) compiler_yielded_zero_queries — compiler emitted no valid
        #       queries for this source
        #   (b) high_precision_returned_zero — the high-precision concept-
        #       block families (cb_high_precision / cb_mesh_keyword /
        #       cb_clinical_queries) returned zero hits for this source
        _HIGH_PRECISION_QTYPES = {
            "cb_high_precision", "cb_mesh_keyword", "cb_clinical_queries",
        }
        fallback_reasons_by_source: dict[str, list[str]] = {s: [] for s in sources}
        per_source_primary_counts: dict[str, int] = {}
        per_source_high_precision_hits: dict[str, int] = {}
        for src, pairs in retrieval_results:
            per_source_primary_counts[src] = len(pairs)
            hp_hits = sum(1 for qt, _ in pairs if qt in _HIGH_PRECISION_QTYPES)
            per_source_high_precision_hits[src] = hp_hits
            if compiler_query_counts_by_source.get(src, 0) == 0:
                fallback_reasons_by_source[src].append("compiler_yielded_zero_queries")
                phase5d_metrics["compiler_yielded_no_queries_sources"].append(src)
            else:
                had_hp = any(
                    p["query_type"] in _HIGH_PRECISION_QTYPES
                    for p in plans_by_source[src]
                )
                if had_hp and hp_hits == 0:
                    fallback_reasons_by_source[src].append("high_precision_returned_zero")

        fallback_sources: list[str] = [
            s for s, reasons in fallback_reasons_by_source.items() if reasons
        ]

        # ---- Step 5 — Legacy planner runs ONLY for fallback sources ----
        legacy_plans: list = []
        if fallback_sources:
            legacy_plans = search_planning.plan(parsed, sources=fallback_sources)
            for p in legacy_plans:
                fb_reasons = fallback_reasons_by_source.get(p.source) or []
                primary_fb_reason = fb_reasons[0] if fb_reasons else None
                await db[Collections.SEARCH_QUERIES].insert_one({
                    "id": uuid.uuid4().hex,
                    "search_session_id": session_id,
                    "source": p.source,
                    "query_type": p.query_type,
                    "query_family": p.query_type,
                    "query_string": p.query_string,
                    "rationale": p.rationale,
                    "concept_blocks_used_json": {},
                    "mandatory_anchors_json": [],
                    "optional_blocks_json": [],
                    "role_weights_json": role_weights_json,
                    # Phase 6B.1 — planner provenance.
                    "planner_source": "legacy_planner",
                    "is_primary_strategy": False,
                    "is_fallback_strategy": True,
                    "fallback_reason": primary_fb_reason,
                    "created_at": datetime.now(timezone.utc),
                })
            legacy_plans_by_source: dict[str, list[dict[str, Any]]] = {
                s: [] for s in fallback_sources
            }
            for p in legacy_plans:
                legacy_plans_by_source[p.source].append({
                    "query_type": p.query_type, "query_string": p.query_string,
                })
            legacy_retrieval_results = await asyncio.gather(
                *[_retrieve(s, legacy_plans_by_source[s]) for s in fallback_sources]
            )
            # Merge legacy pairs into retrieval_results so the rest of the
            # pipeline (dedupe, ranker, evaluator) sees a single combined set.
            # The picott_aware_ranker applies mandatory-anchor caps uniformly,
            # so fallback results CANNOT rank highly unless they pass anchor
            # gates — guardrail per the 6B.1 brief.
            by_src: dict[str, list] = {s: list(p) for s, p in retrieval_results}
            for src, pairs in legacy_retrieval_results:
                by_src.setdefault(src, []).extend(pairs)
            retrieval_results = [(s, by_src[s]) for s in by_src]
            await audit_service.log(
                "phase6b1_legacy_planner_fallback",
                user_id=user_id, search_session_id=session_id,
                payload={
                    "fallback_sources": fallback_sources,
                    "fallback_reasons_by_source": fallback_reasons_by_source,
                    "legacy_query_count": len(legacy_plans),
                },
            )

        # Phase 6B.1 — planner-strategy summary used by the transparency
        # endpoint, admin diagnostics, and the new tests.
        planner_strategy_summary: dict[str, Any] = {
            "primary_planner": "boolean_compiler",
            "primary_queries_by_source": compiler_query_counts_by_source,
            "primary_results_by_source": per_source_primary_counts,
            "primary_high_precision_hits_by_source":
                per_source_high_precision_hits,
            "legacy_planner_invoked": bool(fallback_sources),
            "legacy_planner_sources": fallback_sources,
            "fallback_reasons_by_source": fallback_reasons_by_source,
            "legacy_query_count": len(legacy_plans),
        }

        # Build the raw pairs by source so we can normalize Europe PMC records
        # and pass through already-normalized PubMed records. NOTE: the
        # closed-set ``normalization`` diagnostics stage is reserved for
        # the pre-PICOTT spelling + shorthand text normalization (see
        # PROOF A step 2). This post-retrieval metadata normalization
        # is inlined and not stage-timed (it's bounded by the dedupe
        # stage which immediately follows).
        all_pairs: list[tuple[str, dict[str, Any], str]] = []  # (qtype, article_dict, source)
        per_source_counts: dict[str, int] = {}
        for src, pairs in retrieval_results:
            count = 0
            for qtype, raw in pairs:
                if src == "europe_pmc":
                    try:
                        art = metadata_normalizer.normalize(raw)
                    except Exception:  # noqa: BLE001
                        logger.exception("normalize failed for one Europe PMC record")
                        continue
                    art["_retrieved_by"] = "europe_pmc"
                else:
                    # PubMed connector already normalizes — passthrough.
                    art = raw
                    if not art.get("_retrieved_by"):
                        art["_retrieved_by"] = src
                # Track sources_present on each article (set of sources)
                art.setdefault("sources_present", [])
                if src not in art["sources_present"]:
                    art["sources_present"].append(src)
                all_pairs.append((qtype, art, src))
                count += 1
            per_source_counts[src] = count

        # ---- normalize → dedupe (cross-source) ----
        await _set_status(session_id, status="ranking")
        norm_pairs_for_dedup = [(qt, a) for qt, a, _ in all_pairs]
        with diag.stage("deduplication"):
            unique_articles, qmap = deduplication.dedupe(norm_pairs_for_dedup)

        # ---- Thin-pool supplemental retrieval (additive recall) ----
        # When the deduped candidate pool is small even after the boolean
        # compiler + per-source legacy fallback, an over-specific PICOTT/anchor
        # set has starved retrieval. Run ONE broadened legacy-planner round for
        # the sources that weren't already broadened and merge + re-dedupe, so
        # the extra candidates flow through the SAME downstream guardrails
        # (OpenAlex enrich → safety filter → PICOTT anchor caps → ranking →
        # default-selection → rescue re-add). Strictly additive: dedupe collapses
        # overlaps and the anchor caps still gate what can rank/show, so this
        # never lowers precision — it only gives thin searches more to work with.
        # Default-ON (no deployment env needed); THIN_POOL_SUPPLEMENT_ENABLED=false
        # disables it. Non-fatal: any failure leaves the original pool intact.
        _thin_enabled, _thin_min = recall_supplement.thin_pool_settings()
        if recall_supplement.should_supplement(
            len(unique_articles), enabled=_thin_enabled, min_pool=_thin_min
        ):
            _broaden_sources = recall_supplement.sources_to_broaden(sources, fallback_sources)
            if _broaden_sources:
                try:
                    supp_plans = search_planning.plan(parsed, sources=_broaden_sources)
                    supp_by_source: dict[str, list[dict[str, Any]]] = {
                        s: [] for s in _broaden_sources
                    }
                    for p in supp_plans:
                        if p.source not in supp_by_source:
                            continue
                        supp_by_source[p.source].append(
                            {"query_type": p.query_type, "query_string": p.query_string}
                        )
                        await db[Collections.SEARCH_QUERIES].insert_one({
                            "id": uuid.uuid4().hex,
                            "search_session_id": session_id,
                            "source": p.source,
                            "query_type": p.query_type,
                            "query_family": p.query_type,
                            "query_string": p.query_string,
                            "rationale": p.rationale,
                            "concept_blocks_used_json": {},
                            "mandatory_anchors_json": [],
                            "optional_blocks_json": [],
                            "role_weights_json": role_weights_json,
                            "planner_source": "thin_pool_supplement",
                            "is_primary_strategy": False,
                            "is_fallback_strategy": True,
                            "fallback_reason": "thin_pool_supplement",
                            "created_at": datetime.now(timezone.utc),
                        })
                    supp_results = await asyncio.gather(
                        *[_retrieve(s, supp_by_source[s]) for s in _broaden_sources]
                    )
                    # Normalize supplemental pairs exactly like the main loop.
                    supp_norm_pairs: list[tuple[str, dict[str, Any]]] = []
                    for src, pairs in supp_results:
                        for qtype, raw in pairs:
                            if src == "europe_pmc":
                                try:
                                    art = metadata_normalizer.normalize(raw)
                                except Exception:  # noqa: BLE001
                                    continue
                                art["_retrieved_by"] = "europe_pmc"
                            else:
                                art = raw
                                if not art.get("_retrieved_by"):
                                    art["_retrieved_by"] = src
                            art.setdefault("sources_present", [])
                            if src not in art["sources_present"]:
                                art["sources_present"].append(src)
                            supp_norm_pairs.append((qtype, art))
                    if supp_norm_pairs:
                        _pool_before = len(unique_articles)
                        # Re-dedupe the combined set (not wrapped in the timed
                        # ``deduplication`` stage again to keep the closed-set
                        # stage timings single-entry).
                        unique_articles, qmap = deduplication.dedupe(
                            norm_pairs_for_dedup + supp_norm_pairs
                        )
                        logger.info(
                            "thin_pool_supplement: session=%s pool %d -> %d "
                            "(broadened %s)",
                            session_id, _pool_before, len(unique_articles), _broaden_sources,
                        )
                        await audit_service.log(
                            "thin_pool_supplement",
                            user_id=user_id, search_session_id=session_id,
                            payload={
                                "pool_before": _pool_before,
                                "pool_after": len(unique_articles),
                                "broaden_sources": _broaden_sources,
                                "min_pool": _thin_min,
                                "supplemental_queries": len(supp_plans),
                            },
                        )
                except Exception:  # noqa: BLE001
                    logger.exception("thin_pool_supplement failed (non-fatal)")

        # ---- OpenAlex enrichment (after dedup, metadata-only) ----
        oa_stats: dict[str, Any] = {
            "attempted": 0, "applied": 0, "status": "disabled",
        }
        if oa_enrich and unique_articles:
            # Use _pre_rank_score so OpenAlex prioritizes the obviously-relevant ones.
            for a in unique_articles:
                a["_pre_rank_score"] = (
                    100 if a.get("is_guideline") else
                    80 if a.get("article_type_classification") in {
                        "systematic_review", "meta_analysis", "rct",
                        "diagnostic_accuracy_study"
                    } else 50
                )
            # OpenAlex enrichment is best-effort metadata (citation counts, OA
            # links) and does up to 4 lookups per article with retry/backoff. A
            # slow or rate-limited OpenAlex must NEVER stall the whole search
            # for minutes, so cap it with a wall-clock budget and proceed with
            # whatever enriched in time — articles still rank via source-native
            # signals. Configurable via OPENALEX_ENRICH_BUDGET_S.
            try:
                stats = await asyncio.wait_for(
                    openalex_connector.enrich_many(
                        unique_articles, client=http_client, search_session_id=session_id,
                    ),
                    timeout=float(os.environ.get("OPENALEX_ENRICH_BUDGET_S", "30")),
                )
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning(
                    "openalex enrich_many exceeded budget (session=%s); "
                    "proceeding without full enrichment", session_id,
                )
                stats = {
                    "status": "budget_timeout",
                    "enrichments_attempted": 0,
                    "enrichments_applied": 0,
                }
            oa_stats = {
                "attempted": stats.get("enrichments_attempted", 0),
                "applied": stats.get("enrichments_applied", 0),
                "status": stats.get("status", "ok"),
                "error_message": stats.get("error_message"),
            }
            await _set_status(session_id, status="ranking", extras={"openalex_enriched": True})

        # ---- safety pre-rank filter ----
        with diag.stage("safety_filter"):
            kept, excluded_pre = safety_filter.pre_rank_filter(unique_articles)

        # ---- rank ----
        with diag.stage("ranking"):
            ranked = evidence_ranker.rank(parsed, kept, qmap)

        # ---- Phase 5A: PICOTT-aware post-processing (anchor gates + caps +
        # ranking-concept boosts + OA-text signals). No-op when picott has
        # no must_have_concepts and no negatives.
        family_map_by_ck: dict[str, list[str]] = {}
        for ck, q_types in (qmap or {}).items():
            family_map_by_ck[ck] = list(q_types)
        # Pull per-session/per-user OA text extracts (NEVER cross-session).
        oa_extracts_by_ck: dict[str, str] = {}
        try:
            cur = db[Collections.ARTICLE_TEXT_EXTRACTS].find(
                {
                    "search_session_id": session_id,
                    "text_source": {"$in": ["oa_full_text", "user_uploaded_full_text",
                                              "institution_authorized_full_text"]},
                    "include_in_summary": True,
                },
                {"_id": 0, "article_id": 1, "section_text": 1},
            )
            tmp_by_art: dict[str, list[str]] = {}
            async for row in cur:
                aid = row.get("article_id") or ""
                if not aid:
                    continue
                tmp_by_art.setdefault(aid, []).append(row.get("section_text") or "")
            # Map article_id → canonical_key.
            if tmp_by_art:
                art_lookup = await db[Collections.ARTICLES].find(
                    {"id": {"$in": list(tmp_by_art.keys())}},
                    {"_id": 0, "id": 1, "canonical_key": 1},
                ).to_list(length=len(tmp_by_art) + 1)
                for row in art_lookup:
                    ck = row.get("canonical_key") or ""
                    aid = row.get("id") or ""
                    if ck and aid in tmp_by_art:
                        oa_extracts_by_ck[ck] = "\n".join(tmp_by_art[aid])[:8000]
        except Exception:  # noqa: BLE001
            logger.exception("could not load article_text_extracts (non-fatal)")
        if picott_payload:
            ranked = picott_aware_ranker.postprocess(
                ranked, picott_payload,
                article_text_extracts_by_article=oa_extracts_by_ck,
                query_family_by_canonical=family_map_by_ck,
            )

        # ---- Phase 5B: Capture top-k snapshot BEFORE semantic rerank,
        # then annotate with semantic-rerank scores. Shadow mode does
        # not reorder — we still annotate so the diagnostics view shows
        # the signal. Active mode also re-sorts by semantic score after
        # cap-aware blending. ----
        top_before_rerank = [
            {
                "rank": idx + 1,
                "pmid": (r.get("_article") or {}).get("pmid"),
                "title": ((r.get("_article") or {}).get("title") or "")[:120],
                "overall_rank_score": int(r.get("overall_rank_score") or 0),
            }
            for idx, r in enumerate(ranked[:10])
        ]
        try:
            ranked = semantic_reranker.rerank(ranked, picott_payload)
        except Exception:  # noqa: BLE001 — never fail the pipeline
            logger.exception("semantic_reranker failed (non-fatal)")
        eff_expansion_mode = expansion_mode()
        if eff_expansion_mode == "active":
            # Blend cap-clamped overall_rank_score (70%) with semantic
            # score (30%) but never exceed the cap. Then re-sort.
            for r in ranked:
                base = float(r.get("overall_rank_score") or 0)
                sem = r.get("semantic_relevance_score")
                cap_applied = r.get("relevance_cap_applied")
                if isinstance(sem, (int, float)):
                    blended = 0.7 * base + 0.3 * float(sem)
                    if isinstance(cap_applied, (int, float)) and cap_applied < 100:
                        blended = min(blended, float(cap_applied))
                    r["overall_rank_score"] = int(round(max(0.0, min(100.0, blended))))
            ranked.sort(key=lambda r: r.get("overall_rank_score", 0), reverse=True)
        top_after_rerank = [
            {
                "rank": idx + 1,
                "pmid": (r.get("_article") or {}).get("pmid"),
                "title": ((r.get("_article") or {}).get("title") or "")[:120],
                "overall_rank_score": int(r.get("overall_rank_score") or 0),
                "semantic_relevance_score": r.get("semantic_relevance_score"),
            }
            for idx, r in enumerate(ranked[:10])
        ]
        await db[Collections.SEARCH_SESSIONS].update_one(
            {"id": session_id},
            {"$set": {
                "top_k_before_semantic_rerank_json": top_before_rerank,
                "top_k_after_semantic_rerank_json": top_after_rerank,
            }},
        )

        # ---- Phase 5B: Scholar-like candidate expansion (shadow by default).
        # Produces the expansion artifact and persists it on the session row.
        # The expansion service refuses on PHI-blocked sessions and audits
        # itself through ``audit_service.log`` with closed-set codes only. ----
        # Phase 6C — also persist closed-set per-session expansion metadata
        # (expansion_mode_used / active_expansion_applied / candidate_count /
        # latency_ms / reason) so admin diagnostics + tests can read them
        # without parsing the larger ``scholar_like_expansion_json`` blob.
        _exp_t0 = time.time()
        _exp_reason = "off_by_config"
        _exp_candidate_count = 0
        _exp_applied = False
        try:
            expansion_out = await scholar_like_expansion.expand(
                seed_articles=ranked[:5],
                picott=picott_payload,
                is_phi_blocked=False,
                mode=eff_expansion_mode,
            )
            _exp_candidate_count = int(
                expansion_out.get("candidate_count")
                or len(expansion_out.get("candidates") or [])
                or 0
            )
            # Closed-set ``active_expansion_reason`` per the Phase 6C brief.
            if eff_expansion_mode == "off":
                _exp_reason = "off_by_config"
            elif eff_expansion_mode == "shadow":
                _exp_reason = "shadow_collected_only"
            elif eff_expansion_mode == "active":
                # Pipeline reached this point with active mode resolved.
                # Per the Phase 5B+ shadow contract the candidates still
                # don't merge into ranking automatically — the semantic-
                # reranker blend above is the only active-mode effect.
                _exp_applied = True
                _exp_reason = "applied_to_session"
            else:
                _exp_reason = "off_by_config"
            await db[Collections.SEARCH_SESSIONS].update_one(
                {"id": session_id},
                {"$set": {
                    "scholar_like_expansion_json": expansion_out,
                    # Phase 6C per-session metadata.
                    "expansion_mode_used": eff_expansion_mode,
                    "active_expansion_applied": _exp_applied,
                    "active_expansion_candidate_count": _exp_candidate_count,
                    "active_expansion_latency_ms": int(
                        (time.time() - _exp_t0) * 1000
                    ),
                    "active_expansion_reason": _exp_reason,
                }},
            )
            await audit_service.log(
                "scholar_like_expansion_ran",
                user_id=user_id, search_session_id=session_id,
                payload={
                    "mode": expansion_out.get("mode"),
                    "per_tier_counts": expansion_out.get("per_tier_counts"),
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("scholar_like_expansion failed (non-fatal)")
            # Even on error persist the metadata so admin can audit.
            await db[Collections.SEARCH_SESSIONS].update_one(
                {"id": session_id},
                {"$set": {
                    "expansion_mode_used": eff_expansion_mode,
                    "active_expansion_applied": False,
                    "active_expansion_candidate_count": 0,
                    "active_expansion_latency_ms": int(
                        (time.time() - _exp_t0) * 1000
                    ),
                    "active_expansion_reason": "error_skipped",
                }},
            )

        # ---- persist articles + session_articles ----
        # Phase 5C — determine per-article candidate_tier (closed set) and
        # per-source attribution list (candidate_sources_json) from the
        # query families this article matched + the expansion provenance.
        def _candidate_tier_for(r: dict[str, Any]) -> tuple[str, list[str]]:
            families = list(r.get("query_family_matches") or r.get("query_types_matched") or [])
            sources_present = list((r.get("_article") or {}).get("sources_present") or [])
            sources: list[str] = []
            # Expansion-provenance attribution (added later when shadow/active mode merges).
            expansion_provenance = list(r.get("candidate_sources") or [])
            sources.extend(expansion_provenance)
            for s in sources_present:
                tag = f"direct_{s}"
                if tag not in sources:
                    sources.append(tag)
            # Tier priority for the primary chip:
            # benchmark_gold_match > semantic_reranker_promotion > expansion tiers
            #   > trial_acronym_match > mesh_keyword_match > direct_*.
            if r.get("benchmark_gold_match"):
                return "benchmark_gold_match", sources
            if r.get("semantic_reranker_promoted"):
                return "semantic_reranker_promotion", sources
            if any(s in {"openalex_cited_by", "europe_pmc_citations",
                          "pubmed_elink_cited_by"} for s in expansion_provenance):
                return "citation_expansion", sources
            if any(s in {"openalex_referenced", "europe_pmc_references",
                          "pubmed_elink_referenced"} for s in expansion_provenance):
                return "reference_expansion", sources
            if any(s in {"openalex_related", "europe_pmc_similar",
                          "pubmed_elink_related"} for s in expansion_provenance):
                return "related_work_expansion", sources
            if "trial_acronym_expansion" in families or any(
                s == "trial_acronym" for s in expansion_provenance
            ) or any(f in {"trial_acronym", "cb_trial_acronym"} for f in families):
                return "trial_acronym_match", sources
            if any(f in {"study_type_filtered", "guideline_review",
                          "mesh_keyword", "clinical_queries",
                          "cb_mesh_keyword", "cb_clinical_queries"}
                    for f in families):
                return "mesh_keyword_match", sources
            if "pubmed" in sources_present:
                return "direct_pubmed_boolean", sources
            if "europe_pmc" in sources_present:
                return "direct_europe_pmc_boolean", sources
            return "direct_europe_pmc_boolean", sources

        # ---- Phase L1: Explicit Evidence Lens + Population Focus ----
        # Clinician-stated SOFT preference layer. Runs AFTER all
        # deterministic ranking + cap enforcement + semantic rerank and
        # BEFORE default selection, so a (bounded, cap-clamped) boost can
        # gently re-prioritize already-retrieved evidence. Soft-boost /
        # shadow-first: never a hard filter, never hides evidence, never
        # overrides safety caps (every boost is re-clamped to the row's
        # relevance cap ceiling). Defaults (clinically_direct / any) are
        # a no-op, so baseline behavior is unchanged unless the clinician
        # explicitly picks a preference.
        try:
            with diag.stage("l1_evidence_lens"):
                _pref_doc = await db[Collections.SEARCH_SESSIONS].find_one(
                    {"id": session_id},
                    {"_id": 0, "evidence_lens": 1, "population_focus": 1},
                ) or {}
                _l1_lens = evidence_lens.normalize_lens(
                    _pref_doc.get("evidence_lens")
                )
                _l1_pop = evidence_lens.normalize_population(
                    _pref_doc.get("population_focus")
                )
                _l1_mode = evidence_lens.resolve_preference_mode()
                _l1_diag = evidence_lens.apply_evidence_lens(
                    ranked, lens=_l1_lens, population=_l1_pop, mode=_l1_mode,
                )
            await db[Collections.SEARCH_SESSIONS].update_one(
                {"id": session_id},
                {"$set": {
                    "evidence_lens": _l1_lens,
                    "population_focus": _l1_pop,
                    "preference_mode": _l1_mode,
                    "preference_provenance": evidence_lens.PREFERENCE_PROVENANCE,
                    "l1_preference_diagnostics_json": _l1_diag,
                    "updated_at": datetime.now(timezone.utc),
                }},
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseL1 evidence-lens application failed (non-fatal) "
                "session=%s", session_id,
            )

        # Phase 5D.1 — pre-upsert articles so each row has a stable
        # article_id BEFORE the default-selection function runs. This
        # lets the function key its rationale dict by article_id and
        # the persistence loop just looks the row up. Run bounded-concurrent:
        # each _upsert_article is an independent, race-safe find+write, and a
        # remote Atlas otherwise serializes 300+ round-trips into minutes.
        _persist_sem = asyncio.Semaphore(
            max(1, int(os.environ.get("ARTICLE_PERSIST_CONCURRENCY", "24")))
        )

        async def _persist_article(_r):
            async with _persist_sem:
                try:
                    _r["_article"]["id"] = await _upsert_article(_r["_article"])
                except Exception:  # noqa: BLE001 — one bad row (e.g. an
                    # unrecoverable duplicate-key race) must NOT abort the whole
                    # search; mirror the failure-isolation _persist_excluded has.
                    logger.exception(
                        "persist of ranked article failed; dropping it "
                        "(session=%s pmid=%s doi=%s)",
                        session_id,
                        _r["_article"].get("pmid"),
                        _r["_article"].get("doi_normalized"),
                    )
                    _r["_article"]["id"] = None

        await asyncio.gather(*[_persist_article(r) for r in ranked])
        # Drop candidates that could not be persisted, and collapse any that
        # resolved to the SAME persisted article (identifier-level duplicates
        # from different connectors, e.g. a PubMed record + an OpenAlex record
        # sharing a DOI). This prevents a single bad row from erroring the
        # session, dangling an SSA reference to a non-persisted article, or
        # surfacing the same paper twice.
        _ranked_before = len(ranked)
        _seen_article_ids: set[str] = set()
        _deduped_ranked = []
        for _r in ranked:
            _aid = (_r.get("_article") or {}).get("id")
            if not _aid or _aid in _seen_article_ids:
                continue
            _seen_article_ids.add(_aid)
            _deduped_ranked.append(_r)
        if len(_deduped_ranked) != _ranked_before:
            logger.warning(
                "ranked persist: kept %d of %d candidates (dropped unpersistable "
                "or duplicate rows) session=%s",
                len(_deduped_ranked), _ranked_before, session_id,
            )
        ranked = _deduped_ranked

        # Phase 5D.1 — default-selection module is the single source of
        # truth for ``included_by_default``. The PICOTT ranker already
        # populated cap_reasons / penalty_concepts on each row; we feed
        # those plus the picott payload into the pure ``select_default_articles``
        # function and override the legacy ``included_by_default`` flag.
        with diag.stage("default_selection"):
            default_sel = select_default_articles(picott_payload, ranked)
        selected_set = set(default_sel["selected_article_ids"])
        # Phase K4A.2 — selected/excluded display invariant: any article
        # in this set MUST NOT carry an exclusion_reason when persisted.
        cleared_excl_set = set(default_sel.get("cleared_exclusion_for_ids") or [])
        rationale_map = default_sel["selection_rationale"]
        # Persist the session-level summary so GET responses can surface
        # excluded-rationale counts without re-running selection.
        # Phase 6B.2 — merge the new ``default_selection_summary`` block
        # (selected_count, eligible_direct_count, etc.) returned by
        # ``select_default_articles`` into the persisted view.
        ds_summary = dict(default_sel.get("default_selection_summary") or {})
        ds_summary.update({
            "selected_article_ids": default_sel["selected_article_ids"],
            "excluded_count": default_sel["excluded_count"],
            "excluded_rationale_summary":
                default_sel["excluded_rationale_summary"],
        })
        await db[Collections.SEARCH_SESSIONS].update_one(
            {"id": session_id},
            {"$set": {
                "default_selection_summary": ds_summary,
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        for r in ranked:
            art = r["_article"]
            article_id = art["id"]
            if article_id in selected_set:
                r["included_by_default"] = True
                r["default_selection_rationale"] = rationale_map.get(article_id)
                # Phase 6B.2.1 / Phase K4A.2 — selected/excluded display
                # invariant: any article in ``cleared_exclusion_for_ids``
                # (== ``selected_article_ids``) MUST NOT carry an
                # exclusion_reason. We clear it unconditionally for
                # selected articles. ``select_default_articles`` only
                # ever picks rows that passed safety + caps, so this
                # never silences a real safety / cap exclusion (those
                # rows are filtered out of ``eligible`` before
                # ``selected_set`` is built).
                if article_id in cleared_excl_set and r.get("exclusion_reason"):
                    r["exclusion_reason"] = None
            else:
                r["included_by_default"] = False
                # Closed-set rejection reason if it applies (None = the
                # article was eligible but didn't make the top-N).
                rej = _exclusion_reason_for(r)
                r["default_selection_rationale"] = rej

        # Phase 6B.3 — multi-intent per-article tagging.
        # When ``multi_intent.is_multi_intent === true``, walk each
        # sub_question's anchor terms against the article's title +
        # abstract and tag the article with the matched sub-question(s).
        # This is a per-article post-rank pass; no separate retrieval
        # pass is added in this sub-phase. Lane balance is applied
        # below.
        mi = context.get("multi_intent") or {}
        is_multi = bool(mi.get("is_multi_intent"))
        sub_questions: list[dict[str, Any]] = list(mi.get("sub_questions") or [])
        if is_multi and sub_questions:
            import re as _re6b3
            # Synthesize a deterministic id for each sub-question.
            for i, sq in enumerate(sub_questions):
                if not sq.get("subquestion_id"):
                    sq["subquestion_id"] = (
                        f"sq_{i}_{(sq.get('subtype') or 'other').lower()}"
                    )
            # Persist the synthesized IDs back to context so the
            # session's multi_intent_json carries them downstream.
            mi["sub_questions"] = sub_questions
            context["multi_intent"] = mi
            # Phase 6B.3 — also write the synthesized IDs back to the
            # session document so the GET response's ``multi_intent_json``
            # matches the IDs persisted under
            # ``search_session_articles.subquestion_ids_matched``.
            await db[Collections.SEARCH_SESSIONS].update_one(
                {"id": session_id},
                {"$set": {
                    "multi_intent_json": dict(mi),
                    "updated_at": datetime.now(timezone.utc),
                }},
            )
            # Build per-sub-question anchor regex from the sub-question
            # text + its detected subtype must-haves (only words ≥ 3
            # chars to avoid noisy stopwords).
            anchor_terms_by_sq: dict[str, list[str]] = {}
            for sq in sub_questions:
                terms: list[str] = []
                text = (sq.get("question") or "").lower()
                # Curated stop-list reuse from the ontology extractor.
                from services.clinical_ontology_mapper import (  # noqa: WPS433
                    _EXTRACT_STOP_TOKENS, clinical_ontology_mapper,
                )
                for w in _re6b3.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text):
                    lw = w.lower()
                    if lw in _EXTRACT_STOP_TOKENS or len(lw) < 3:
                        continue
                    if lw not in terms:
                        terms.append(lw)
                # Also add ontology canonicals + their synonyms when
                # the sub-question maps cleanly.
                try:
                    hits = clinical_ontology_mapper.extract_from_text(
                        text, max_terms=4, min_confidence=0.85,
                    )
                    for h in hits:
                        c = (h.get("canonical_concept") or "").lower()
                        if c and c not in terms:
                            terms.append(c)
                        for s in (h.get("synonyms") or []):
                            sl = (s or "").lower()
                            if sl and sl not in terms:
                                terms.append(sl)
                except Exception:  # noqa: BLE001
                    pass
                anchor_terms_by_sq[sq["subquestion_id"]] = terms
            # Tag each article.
            for r in ranked:
                art = r["_article"]
                body = " ".join([
                    (art.get("title") or "").lower(),
                    (art.get("abstract") or "").lower(),
                ])
                matched_ids: list[str] = []
                roles_by_sq: dict[str, list[str]] = {}
                rel_by_sq: dict[str, int] = {}
                for sq in sub_questions:
                    sid = sq["subquestion_id"]
                    hits = sum(1 for t in anchor_terms_by_sq[sid]
                                if t and t in body)
                    if hits >= 2:
                        matched_ids.append(sid)
                        roles_by_sq[sid] = list(
                            (sq.get("role_weights") or {}).keys()
                        )[:5]
                        rel_by_sq[sid] = min(100, 50 + hits * 5)
                r["subquestion_ids_matched"] = matched_ids
                r["subquestion_roles_matched"] = roles_by_sq
                r["subquestion_relevance_scores"] = rel_by_sq
                # Badge derivation.
                if len(matched_ids) >= 2:
                    badge = "Both"
                elif len(matched_ids) == 1:
                    # Use the sub-question's subtype as a friendly badge.
                    subtype = next(
                        (sq.get("subtype") for sq in sub_questions
                         if sq["subquestion_id"] == matched_ids[0]), None,
                    )
                    if subtype:
                        # Title-case the subtype, but keep short labels.
                        badge = subtype.replace("_", " ").title()
                        if "schedule" in subtype:
                            badge = "Schedule"
                        elif (subtype.endswith("effectiveness")
                              or "benefit" in subtype):
                            badge = "Benefit"
                    else:
                        badge = "Related"
                else:
                    badge = "Related"
                r["article_subquestion_badge"] = badge
            # ---- Lane balance ----
            # If each lane has ≥3 candidates, top-10 must include ≥3 from
            # each. Apply by re-sorting the top of ``ranked`` so the
            # caller-side persistence loop sees the balanced order.
            if len(sub_questions) >= 2:
                lane_counts = {
                    sq["subquestion_id"]: sum(
                        1 for r in ranked
                        if sq["subquestion_id"] in (
                            r.get("subquestion_ids_matched") or []
                        )
                    ) for sq in sub_questions
                }
                ranked_top = ranked[:10]
                ranked_tail = ranked[10:]
                # Per-lane top-3 from the ranker order (anywhere in ranked).
                forced_top: list[dict] = []
                forced_keys: set[str] = set()
                if all(c >= 3 for c in lane_counts.values()):
                    for sq in sub_questions:
                        sid = sq["subquestion_id"]
                        for r in ranked:
                            if sid in (r.get("subquestion_ids_matched") or []):
                                k = r["_article"]["id"]
                                if k not in forced_keys:
                                    forced_keys.add(k)
                                    forced_top.append(r)
                            if sum(1 for x in forced_top
                                   if sid in (x.get("subquestion_ids_matched") or [])) >= 3:
                                break
                # Fill remaining slots from the original top-10 order.
                remaining_slots = max(0, 10 - len(forced_top))
                fillers: list[dict] = []
                for r in ranked_top:
                    if r["_article"]["id"] in forced_keys:
                        continue
                    fillers.append(r)
                    if len(fillers) >= remaining_slots:
                        break
                new_top = forced_top + fillers
                # Any ``ranked_top`` rows that were displaced shift down.
                displaced = [r for r in ranked_top
                             if r["_article"]["id"] not in
                             {x["_article"]["id"] for x in new_top}]
                ranked = new_top + displaced + ranked_tail

        # AI Agent 3b — rescued-article RE-ADD (flag-gated, default OFF).
        # Reviews the deterministically-rejected (soft-excluded) candidates with
        # ONE batched LLM call and, for high-confidence direct matches that pass
        # a central-anchor guard, re-adds them by clearing the soft
        # exclusion_reason on the ranked row IN PLACE — so the paper flows into
        # the SSA rows built just below with zero read-side/API change. This runs
        # BEFORE the SSA build on purpose; the advisory maybe_curate_evidence()
        # call further down runs AFTER the insert and can only shelve. The
        # function self-guards on the flag + prod opt-in and never raises.
        try:
            from agents.evidence_curator import maybe_rescue_readd
            await maybe_rescue_readd(
                session_id, user_id, sanitized_question, ranked, picott_payload,
            )
        except Exception:  # noqa: BLE001
            logger.exception("ai_rescue_readd hook failed (non-fatal)")

        # If the rescue re-added any rows, recompute the persisted
        # default_selection_summary so selected/excluded counts (first written
        # above, pre-rescue) reflect the re-added papers instead of undercounting
        # them. No-op + no extra write when nothing was rescued.
        if any(r.get("llm_rescued") for r in ranked):
            ds_summary = recount_summary_after_rescue(ds_summary, ranked)
            try:
                await db[Collections.SEARCH_SESSIONS].update_one(
                    {"id": session_id},
                    {"$set": {
                        "default_selection_summary": ds_summary,
                        "updated_at": datetime.now(timezone.utc),
                    }},
                )
            except Exception:  # noqa: BLE001
                logger.exception("ai_rescue_readd summary recount persist failed (non-fatal)")

        _ssa_rows: list[dict[str, Any]] = []
        for r in ranked:
            art = r["_article"]
            article_id = art["id"]
            candidate_tier, candidate_sources = _candidate_tier_for(r)
            sa = {
                "id": uuid.uuid4().hex,
                "search_session_id": session_id,
                "article_id": article_id,
                "relevance_score": float(r["relevance_score"]),
                "evidence_design_score": float(r["evidence_design_score"]),
                "overall_rank": int(r["overall_rank_score"]),
                "rank_explanation": r["rank_explanation"],
                "included_by_default": bool(r.get("included_by_default")),
                "user_included": bool(r.get("user_included")),
                "exclusion_reason": r.get("exclusion_reason"),
                "query_types_matched": r.get("query_types_matched") or [],
                "query_family_matches": r.get("query_family_matches") or list(r.get("query_types_matched") or []),
                "sources_present": list(art.get("sources_present") or []),
                "flags": r.get("flags") or [],
                # Phase 5A diagnostics — populated only when picott has anchors.
                "matched_must_have": r.get("matched_must_have") or [],
                "missing_must_have": r.get("missing_must_have") or [],
                "matched_should_have": r.get("matched_should_have") or [],
                "matched_ranking_concepts": r.get("matched_ranking") or [],
                "penalty_concepts": r.get("penalty_concepts") or [],
                "relevance_cap_applied": r.get("relevance_cap_applied"),
                "cap_reasons": r.get("cap_reasons") or [],
                # Phase 6G.3 — overview directness classifier verdict
                # (closed-set string). ``None`` when the question
                # subtype is not a disease_overview / definition
                # route. UI surfaces this as the
                # ``article-card-directness-badge``.
                "directness_label": r.get("directness_label"),
                # Phase 5D Fix 5 — structured boost provenance for the
                # admin diagnostics view + clinician "why ranked here?" UI.
                "ranking_boost_reasons": r.get("ranking_boost_reasons") or [],
                "ranking_boost_total": int(r.get("ranking_boost_total") or 0),
                # Phase 6B.2 — role-weighted ranking provenance.
                "role_match_score": int(r.get("role_match_score") or 0),
                "role_match_breakdown_json": r.get("role_match_breakdown_json") or [],
                "role_weights_used_json": dict(r.get("role_weights_used") or {}),
                # Phase 6B.3 — multi-intent per-article tagging.
                "subquestion_ids_matched": list(
                    r.get("subquestion_ids_matched") or []
                ),
                "subquestion_roles_matched": dict(
                    r.get("subquestion_roles_matched") or {}
                ),
                "subquestion_relevance_scores": dict(
                    r.get("subquestion_relevance_scores") or {}
                ),
                "article_subquestion_badge": r.get("article_subquestion_badge"),
                # Phase K5.2C-0 — PubMed-indexed Cochrane review lane (preview).
                # Closed-set provenance label + bool. NEVER implies Cochrane
                # Library full-text coverage; NO full-text ingestion.
                "cochrane_review_lane": bool(r.get("cochrane_review_lane")),
                "cochrane_review_provenance": r.get("cochrane_review_provenance"),
                "cochrane_review_shadow_boost": int(
                    r.get("cochrane_review_shadow_boost") or 0
                ),
                # Phase 5B — Semantic reranker outputs (per-article).
                "semantic_relevance_score": r.get("semantic_relevance_score"),
                "direct_answer_likelihood": r.get("direct_answer_likelihood"),
                "matched_canonical_concepts": r.get("matched_canonical_concepts") or [],
                "missing_canonical_concepts": r.get("missing_canonical_concepts") or [],
                "semantic_rationale": r.get("semantic_rationale") or "",
                # Phase 5C — candidate tier (primary chip on article card)
                # + per-source attribution list.
                "candidate_tier": candidate_tier,
                "candidate_sources_json": candidate_sources,
                # Phase 5D.1 — default-selection module rationale.
                "default_selection_rationale": r.get("default_selection_rationale"),
                # Agent 3b — provenance: True when this row was re-added by the
                # LLM rescue re-add pass (soft exclusion cleared). Additive;
                # the frontend ignores unknown fields.
                "llm_rescued": bool(r.get("llm_rescued")),
                "user_override_included": False,
                "user_override_acknowledged_at": None,
                "user_override_warning": None,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            # Phase L1 — per-article evidence-lens preference annotations
            # (closed-set; soft-boost/shadow). Drives the clinician
            # "lens match" badge + admin diagnostics. No PHI / no free text.
            sa["l1_preference_boost"] = float(r.get("l1_preference_boost") or 0.0)
            sa["l1_preference_matched"] = bool(r.get("l1_preference_matched"))
            sa["l1_preference_signals"] = list(r.get("l1_preference_signals") or [])
            sa["l1_preference_rank_delta"] = int(r.get("l1_preference_rank_delta") or 0)
            # Stable pre-L1 deterministic score so re-applying a lens via
            # the review-page control always boosts from baseline (never
            # compounds). Equals the cap-clamped ranker score.
            sa["base_rank_score"] = float(
                r.get("l1_pre_boost_rank_score", r.get("overall_rank_score")) or 0.0
            )
            _ssa_rows.append(sa)
        # Single bulk insert instead of one round-trip per article — these
        # rows are independent and a remote Atlas makes per-article inserts the
        # dominant ranking-stage cost.
        if _ssa_rows:
            await db[Collections.SEARCH_SESSION_ARTICLES].insert_many(
                _ssa_rows, ordered=False,
            )

        # AI Agent 3 — evidence curator / surfacing (B4 / B4a). Reviews the
        # deterministically-excluded (soft) candidates and writes advisory
        # surfacing rows to a SEPARATE shelf (ai_surfaced_articles). It never
        # mutates the ranked/excluded lists or reorders results. Default OFF,
        # never raises.
        try:
            from agents.evidence_curator import maybe_curate_evidence
            await maybe_curate_evidence(session_id, user_id, sanitized_question, ranked)
        except Exception:  # noqa: BLE001
            logger.exception("ai_evidence_surfacing hook failed (non-fatal)")

        # Phase K3B Wave 1 Step 2 — read enforcement mode ONCE so the
        # excluded_pre SSA-insert loop below can skip SSA persistence
        # for actively-excluded retracted articles (so they never
        # surface in the clinician's ``excluded_articles`` array).
        _k3b_step2_enforcement_mode = os.environ.get(
            "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "disabled",
        ).strip().lower()
        assert _k3b_step2_enforcement_mode in {
            "disabled", "shadow", "active",
        }, (
            "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE must be one of "
            "{'disabled', 'shadow', 'active'}. Got: "
            + repr(_k3b_step2_enforcement_mode)
        )
        # Closed-set list capturing the per-article diagnostic payload
        # for actively-excluded retracted articles. Used by:
        #   - the clinician banner projection (closed-set fields)
        #   - the admin observability aggregator
        #   - the per-article audit-event writer
        # Stays empty unless mode == "active" AND at least one
        # retracted-on-stable-join article is present.
        _k3b_step2_active_excluded: list[dict[str, Any]] = []
        # Pre-upsert the excluded candidates concurrently (same bounded pool)
        # so this loop's per-article Atlas writes don't serialize. The
        # canonical post-upsert id is back-stamped onto each ``ex`` dict (Phase
        # K3B Wave 1 Step 1.5) so the K3A classifier + K3B enforcement below
        # read the SAME id that was persisted to ``articles`` / SSA.
        async def _persist_excluded(_ex):
            async with _persist_sem:
                try:
                    _ex["id"] = await _upsert_article(_ex)
                except Exception:  # noqa: BLE001
                    _ex["id"] = None
        await asyncio.gather(*[_persist_excluded(ex) for ex in excluded_pre])

        for ex in excluded_pre:
            article_id = ex.get("id")
            if not article_id:
                continue
            # Phase K3B Wave 1 Step 2 — ACTIVE exclusion gate.
            # Eligibility (binding):
            #   1. mode == "active"
            #   2. ex.is_retracted == True (PubMed Retracted
            #      Publication publication-type signal — the
            #      ``pubmed_retracted_publication_type`` closed-set
            #      source signal)
            #   3. article_join_status is implicitly "stable" here
            #      because the article was just upserted in this same
            #      session and ``ex["id"]`` was back-stamped from the
            #      canonical post-upsert id. ``compute_article_join_
            #      status`` would return "stable" by definition.
            # Effects when gate fires:
            #   - SKIP the SSA insert below (so the article never
            #     reaches the clinician's ``excluded_articles`` array,
            #     ``articles`` array, ranking, default selection,
            #     evidence packets, summary, or citation verifier).
            #   - Append a closed-set diagnostic row to
            #     ``_k3b_step2_active_excluded`` for downstream banner /
            #     audit / observability surfacing.
            # Notes:
            #   - Retraction notices (``is_retraction_notice=True``)
            #     are NEVER actively excluded — they are evidence OF
            #     retraction, not themselves retracted.
            #   - EoC-only items are NEVER actively excluded.
            #   - Preprints are pre-rank-excluded by the safety
            #     filter on a different reason code; that's
            #     orthogonal and unchanged here.
            _k3b_should_actively_exclude = (
                _k3b_step2_enforcement_mode == "active"
                and bool(ex.get("is_retracted"))
                and not bool(ex.get("is_retraction_notice"))
            )
            if _k3b_should_actively_exclude:
                _k3b_step2_active_excluded.append({
                    "article_id": article_id,
                    "pmid": str(ex.get("pmid") or "") or None,
                    "doi": str(ex.get("doi") or "") or None,
                    "title": str(ex.get("title") or "")[:200],
                    "journal": str(ex.get("journal") or "")[:160],
                    "publication_year": ex.get("publication_year"),
                    "publication_date": (
                        str(ex.get("publication_date") or "")
                        if ex.get("publication_date") else None
                    ),
                    "source_signal": "pubmed_retracted_publication_type",
                    "source_signal_label": "PubMed retraction notice",
                    "cap_reason_codes": ["retracted"],
                    "enforcement_explanation": [
                        "k3b:wave1",
                        "active_exclusion",
                        "retracted_via:pubmed_retracted_publication_type",
                        "join:stable",
                    ],
                    "article_join_status": "stable",
                    "enforcement_mode": "active",
                })
                # Skip the SSA insert below — the article is gone.
                continue
            await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
                "id": uuid.uuid4().hex,
                "search_session_id": session_id,
                "article_id": article_id,
                "relevance_score": 0.0,
                "evidence_design_score": 0.0,
                "overall_rank": 0,
                "rank_explanation": None,
                "included_by_default": False,
                "user_included": False,
                "exclusion_reason": ex.get("exclusion_reason"),
                "query_types_matched": [],
                "sources_present": list(ex.get("sources_present") or []),
                "flags": [],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })

        # Phase K3A — shadow article lane labels + cap-candidate
        # detection. Closed-set, deterministic, NO LLM, NO network.
        # Diagnostics ONLY — NEVER alters ranking, default selection,
        # evidence packets, summaries, or citation verification. Hard
        # guard rejects ``mode="active"`` inside the classifier.
        try:
            _k3a_lane_mode = os.environ.get(
                "CLINICAL_ARTICLE_LANE_LABEL_MODE", "shadow",
            ).strip().lower()
            _k3a_cap_mode = os.environ.get(
                "CLINICAL_ARTICLE_CAP_DETECTION_MODE", "shadow",
            ).strip().lower()
            # Phase K3A.1 — separate enforcement flag scaffold. Phase
            # K3B Wave 1 Step 2 — widened to allow ``"active"`` per
            # operator approval (dev/staging only). When ``"active"``,
            # the pipeline performs ACTUAL exclusion of stable-join
            # PubMed-confirmed retracted articles BEFORE ranking.
            # ``"shadow"`` and ``"disabled"`` retain Step 1 / Step 1.6
            # behavior unchanged.
            _k3a_enforcement_mode = os.environ.get(
                "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "disabled",
            ).strip().lower()
            assert _k3a_enforcement_mode in {
                "disabled", "shadow", "active",
            }, (
                "Phase K3B Wave 1 Step 2 hard invariant: "
                "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE must be one of "
                "{'disabled', 'shadow', 'active'}. Got: "
                + repr(_k3a_enforcement_mode)
            )
            if _k3a_lane_mode not in {"disabled", "shadow"}:
                _k3a_lane_mode = "shadow"
            if _k3a_cap_mode not in {"disabled", "shadow"}:
                _k3a_cap_mode = "shadow"
            _k3a_branches = []
            try:
                # The CQM lives on the session document under
                # ``clinical_question_map_json`` — it is NOT a key on
                # the PICOTT ``parsed`` dict. Use the in-scope build
                # output ``_cqm_json_search`` directly.
                if isinstance(_cqm_json_search, dict):
                    _k3a_branches = list(
                        _cqm_json_search.get("branches") or []
                    )
            except Exception:  # noqa: BLE001
                _k3a_branches = []
            if (
                _k3a_lane_mode != "disabled"
                or _k3a_cap_mode != "disabled"
            ) and _k3a_branches:
                _k3a_article_inputs: list[dict[str, Any]] = []
                for r in ranked:
                    # ``ranked`` rows nest the article doc under
                    # ``_article``. Use that as the primary source for
                    # title / abstract / metadata; fall back to the
                    # top-level row for resiliency.
                    _art = r.get("_article") if isinstance(
                        r.get("_article"), dict
                    ) else {}
                    _k3a_article_inputs.append({
                        "id": (
                            _art.get("id")
                            or r.get("article_id")
                            or r.get("id")
                        ),
                        "title": (
                            _art.get("title") or r.get("title") or ""
                        ),
                        "abstract": (
                            _art.get("abstract") or r.get("abstract") or ""
                        ),
                        "publication_types_json": list(
                            _art.get("publication_types_json")
                            or r.get("publication_types_json") or []
                        ),
                        "mesh_terms_json": list(
                            _art.get("mesh_terms_json")
                            or r.get("mesh_terms_json") or []
                        ),
                        "is_guideline": bool(
                            _art.get("is_guideline")
                            or r.get("is_guideline")
                        ),
                        "is_preprint": bool(
                            _art.get("is_preprint")
                            or r.get("is_preprint")
                        ),
                        "is_retracted": bool(
                            _art.get("is_retracted")
                            or r.get("is_retracted")
                        ),
                        "has_expression_of_concern": bool(
                            _art.get("has_expression_of_concern")
                            or r.get("has_expression_of_concern")
                        ),
                        "article_type_classification": (
                            _art.get("article_type_classification")
                            or r.get("article_type_classification")
                        ),
                    })
                for ex in excluded_pre:
                    _k3a_article_inputs.append({
                        "id": ex.get("article_id") or ex.get("id"),
                        "title": ex.get("title") or "",
                        "abstract": ex.get("abstract") or "",
                        "publication_types_json": list(
                            ex.get("publication_types_json") or []
                        ),
                        "mesh_terms_json": list(
                            ex.get("mesh_terms_json") or []
                        ),
                        "is_guideline": bool(ex.get("is_guideline")),
                        "is_preprint": bool(ex.get("is_preprint")),
                        "is_retracted": bool(ex.get("is_retracted")),
                        "has_expression_of_concern": bool(
                            ex.get("has_expression_of_concern")
                        ),
                    })
                _k3a_batch = _k3a_classify_articles(
                    articles=_k3a_article_inputs,
                    branches=_k3a_branches,
                    lane_label_mode=_k3a_lane_mode,
                    cap_detection_mode=_k3a_cap_mode,
                ).model_dump(mode="json")
                # Defensive leak guard — strip any diagnostic that
                # contains a forbidden substring.
                _safe_diag = []
                for d in _k3a_batch.get("diagnostics") or []:
                    blob = str(d)
                    if any(
                        s in blob for s in _K3A_FORBIDDEN_DIAG_SUBSTRINGS
                    ):
                        continue
                    _safe_diag.append(d)
                _k3a_batch["diagnostics"] = _safe_diag
                # Phase K3A.3 — stamp ``article_join_status`` on every
                # per-article diagnostic. At classification time, the
                # article and search_session_articles rows should be
                # present (we just inserted them) — so this is
                # expected to be ``"stable"`` in virtually every case.
                # The field exists so K3B Wave 1 enforcement can
                # re-verify at enforcement time (when dedup / cleanup
                # may have shifted the join), and so the soak
                # collector can route unstable rows to ``unclear``
                # rather than ``false_positive``.
                try:
                    from services.cap_enforcement_guard import (
                        compute_article_join_status as _k3a3_join_status,
                    )
                    # Build a fast in-memory lookup of the in-flight
                    # ``ranked`` articles by id so we don't issue extra
                    # DB calls for the common stable case. Articles
                    # we didn't see in ``ranked`` (e.g. unmatched ids
                    # — should not happen, but defensive) fall back
                    # to a DB lookup.
                    _ranked_by_id: dict[str, dict[str, Any]] = {}
                    for _r in ranked:
                        _ra = _r.get("_article") if isinstance(
                            _r.get("_article"), dict
                        ) else {}
                        _aid = _ra.get("id") or _r.get("article_id")
                        if _aid:
                            _ranked_by_id[str(_aid)] = _ra
                    # Build the SSA lookup ONCE for this session.
                    _ssa_present_ids: set[str] = set()
                    async for _ssa_row in db[
                        Collections.SEARCH_SESSION_ARTICLES
                    ].find(
                        {"search_session_id": session_id},
                        {"_id": 0, "article_id": 1},
                    ):
                        _aid = _ssa_row.get("article_id")
                        if _aid:
                            _ssa_present_ids.add(str(_aid))
                    for d in _k3a_batch["diagnostics"]:
                        aid = str(d.get("article_id") or "")
                        if not aid:
                            d["article_join_status"] = (
                                "article_doc_missing"
                            )
                            continue
                        _live_article = _ranked_by_id.get(aid)
                        if _live_article is None:
                            _live_article = await db[
                                Collections.ARTICLES
                            ].find_one({"id": aid})
                        # SSA presence check (in-memory set lookup).
                        _ssa = (
                            {"id": aid}
                            if aid in _ssa_present_ids else None
                        )
                        d["article_join_status"] = _k3a3_join_status(
                            article_doc=_live_article,
                            search_session_article_doc=_ssa,
                            classification_canonical_key=(
                                _live_article.get("canonical_key")
                                if isinstance(_live_article, dict)
                                else None
                            ),
                        )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "phaseK3A.3 article_join_status enrichment "
                        "failed (non-fatal)"
                    )
                    for d in _k3a_batch["diagnostics"]:
                        d.setdefault(
                            "article_join_status", "stable",
                        )

                # Phase K3B Wave 1 Step 1 — stamp the shadow-
                # enforcement diagnostic fields on every per-article
                # diagnostic. NEVER touches active retrieval, ranking,
                # default selection, evidence packets, summaries, or
                # the citation verifier. Diagnostics-only — the fields
                # are projected on the clinician + admin GETs and
                # aggregated on /api/admin/cap-observability.
                #
                # Eligibility (binding — see report):
                #   1. "retracted" in cap_reason_codes (membership)
                #   2. article_join_status == "stable"
                #   3. enforcement_mode in {"shadow", "active"}
                #   4. cap_reason_codes set is in Wave 1 scope
                #
                # When mode == "disabled": ALL diagnostics carry
                # ``would_have_been_excluded_in_active=False`` with
                # ``enforcement_mode="disabled"`` — never positive.
                try:
                    from services.cap_enforcement_guard import (
                        should_enforce_cap_for_article as _k3b_guard,
                    )
                    _NON_WAVE1_REASONS = {
                        "missing_primary_anchor",
                        "cross_intent_drift",
                        "low_branch_alignment",
                        "wrong_framework_lane",
                        "animal_only_for_human_question",
                        "preprint_unverified",
                        "generic_modifier_only_match",
                        "generic_modifier_only",
                    }
                    for d in _k3a_batch["diagnostics"]:
                        codes = list(d.get("cap_reason_codes") or [])
                        codes_lower = {str(c).lower() for c in codes}
                        join = str(
                            d.get("article_join_status") or "stable"
                        )
                        has_retracted = "retracted" in codes_lower
                        has_eoc = (
                            "expression_of_concern" in codes_lower
                        )
                        only_non_wave1 = (
                            bool(codes_lower)
                            and codes_lower.issubset(_NON_WAVE1_REASONS)
                        )
                        # Compute the gate. ``_k3b_guard`` honors
                        # mode + join + retracted membership.
                        would_exclude = bool(_k3b_guard(
                            article_join_status=join,
                            cap_reason_codes=codes,
                            enforcement_mode=_k3a_enforcement_mode,
                        ))
                        skip_reasons: list[str] = []
                        if _k3a_enforcement_mode == "disabled":
                            # Never positive while disabled.
                            would_exclude = False
                        else:
                            if join != "stable" and has_retracted:
                                skip_reasons.append(
                                    "skipped_due_to_unstable_join"
                                )
                            if has_eoc and not has_retracted:
                                skip_reasons.append(
                                    "skipped_eoc_uncertified"
                                )
                            if only_non_wave1 and not has_retracted:
                                skip_reasons.append(
                                    "skipped_non_wave1_reason"
                                )
                        if would_exclude:
                            d["would_have_been_excluded_in_active"] = True
                            d["would_have_been_excluded_reason_codes"] = [
                                "retracted"
                            ]
                            d["enforcement_explanation_full"] = [
                                "k3b:wave1",
                                "retracted_via:pubmed_retracted_publication_type",
                                "join:stable",
                            ]
                        else:
                            d["would_have_been_excluded_in_active"] = False
                            d["would_have_been_excluded_reason_codes"] = []
                            _expl: list[str] = []
                            if _k3a_enforcement_mode == "disabled":
                                _expl.append(
                                    "skipped:enforcement_disabled"
                                )
                            for sk in skip_reasons:
                                _expl.append(f"skipped:{sk}")
                            if not _expl:
                                _expl.append("skipped:not_wave1_eligible")
                            d["enforcement_explanation_full"] = _expl
                        d["enforcement_candidate_type"] = "retracted_only"
                        d["enforcement_mode"] = _k3a_enforcement_mode
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "phaseK3B Wave 1 shadow-enforcement "
                        "enrichment failed (non-fatal)"
                    )
                    for d in _k3a_batch["diagnostics"]:
                        d.setdefault(
                            "would_have_been_excluded_in_active", False,
                        )
                        d.setdefault(
                            "would_have_been_excluded_reason_codes", [],
                        )
                        d.setdefault(
                            "enforcement_candidate_type",
                            "retracted_only",
                        )
                        d.setdefault(
                            "enforcement_mode", _k3a_enforcement_mode,
                        )
                        d.setdefault(
                            "enforcement_explanation_full",
                            ["skipped:enrichment_error"],
                        )

                # Persist additively under
                # session_diagnostics_json.article_lane_diagnostics.
                # Schema-locked: closed-set Pydantic model already validated.
                try:
                    await db[Collections.SEARCH_SESSIONS].update_one(
                        {"id": session_id},
                        {"$set": {
                            "session_diagnostics_json.article_lane_diagnostics":
                                _k3a_batch,
                        }},
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "phaseK3A persistence failed (non-fatal)"
                    )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseK3A article lane classification failed (non-fatal)"
            )

        # ---- Phase K3B Wave 1 Step 2 — persist k3b_active_exclusions
        # to session_diagnostics + write one audit event per actively-
        # excluded article. Idempotent within a session because the
        # excluded_pre upsert loop runs once and the SSA insert is
        # already skipped for these article_ids. Closed-set only.
        # ---------------------------------------------------------
        if _k3b_step2_active_excluded:
            try:
                # Leak-guard the per-article diagnostic blob before
                # persistence. Closed-set fields are all derived from
                # ``ex`` (PubMed metadata) — no PHI / prompts /
                # completions surface here, but we re-scrub each
                # string defensively against the K3A guard list.
                from services.article_lane_classifier import (
                    _FORBIDDEN_DIAG_SUBSTRINGS as _K3B_LG,
                )

                def _safe_str(v: Any) -> Any:
                    if not isinstance(v, str):
                        return v
                    return v if not any(
                        s in v for s in _K3B_LG
                    ) else "[REDACTED]"

                _safe_rows: list[dict[str, Any]] = []
                for row in _k3b_step2_active_excluded:
                    _safe_rows.append({
                        k: (
                            [
                                _safe_str(x) for x in v
                                if not isinstance(x, str)
                                or _safe_str(x) != "[REDACTED]"
                            ] if isinstance(v, list) else _safe_str(v)
                        )
                        for k, v in row.items()
                    })
                _k3b_active_block = {
                    "schema_version": "k3b.wave1.step2.v1",
                    "enforcement_mode": "active",
                    "enforcement_candidate_type": "retracted_only",
                    "exclusions": _safe_rows,
                    "total": len(_safe_rows),
                }
                await db[Collections.SEARCH_SESSIONS].update_one(
                    {"id": session_id},
                    {"$set": {
                        "session_diagnostics_json.k3b_active_exclusions":
                            _k3b_active_block,
                    }},
                )
                # One closed-set audit event per actively-excluded
                # article. Closed-set fields only; no PHI / prompts /
                # completions. The K2 / K3A leak guard above already
                # scrubbed strings, so the payload below is safe.
                for row in _safe_rows:
                    try:
                        await audit_service.log(
                            "k3b_wave1_active_exclusion",
                            user_id=user_id,
                            search_session_id=session_id,
                            payload={
                                "session_id": session_id,
                                "article_id": row.get("article_id"),
                                "pmid": row.get("pmid"),
                                "doi": row.get("doi"),
                                "cap_reason_codes": list(
                                    row.get("cap_reason_codes") or []
                                ),
                                "enforcement_explanation": list(
                                    row.get(
                                        "enforcement_explanation"
                                    ) or []
                                ),
                                "source_signal": row.get(
                                    "source_signal"
                                ) or "pubmed_retracted_publication_type",
                                "enforcement_mode": "active",
                                "article_join_status": "stable",
                            },
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "phaseK3B Wave 1 Step 2 audit event failed "
                            "(non-fatal)"
                        )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "phaseK3B Wave 1 Step 2 persistence failed "
                    "(non-fatal)"
                )

        # ---- evaluator (with source_coverage + Phase 5A picott_diagnostics) ----
        # Phase 6B.1 — feed the evaluator the FULL planner-query set (primary
        # compiler queries + any fallback legacy queries) so the source-coverage
        # block reflects everything that actually ran.
        evaluator_planner_queries: list[dict[str, Any]] = []
        for q in cb_queries:
            evaluator_planner_queries.append({
                "query_type": f"cb_{q['query_family']}",
                "query_string": q["query_string"],
            })
        for p in legacy_plans:
            evaluator_planner_queries.append({
                "query_type": p.query_type, "query_string": p.query_string,
            })
        evaluator = retrieval_quality_evaluator.evaluate(
            parsed, evaluator_planner_queries,
            ranked,
            raw_total=len(all_pairs),
            per_source_counts=per_source_counts,
            sources_searched=list(sources),
            picott=picott_payload or None,
        )
        evaluator["openalex_enrichment"] = oa_stats
        # Phase 5D — surface compiler metrics for admin diagnostics.
        evaluator["phase5d_query_compiler"] = phase5d_metrics
        # Phase 6B.1 — surface planner-strategy summary on the evaluator so
        # the transparency endpoint + admin diagnostics can render the
        # "Recommended Boolean Strategy" / "Fallback legacy queries" split.
        evaluator["planner_strategy"] = planner_strategy_summary

        # ---- Phase 5A — Rescue query generation (recorded in
        # search_queries; one-shot only). Triggered by the evaluator when
        # top-10 misses must-have anchors. We persist the rescue plan + its
        # generated terms so the diagnostics panel + benchmark harness can
        # show it. Actual rescue retrieval is gated behind the env flag
        # ``OA_RESCUE_RETRIEVAL_ENABLED`` (default false) to avoid doubling
        # API spend during this phase — recording the rescue plan is the
        # most important auditable artifact.
        if (
            picott_payload
            and (evaluator.get("picott_diagnostics") or {}).get("rescue_query_needed")
        ):
            rescue_must_haves = (
                evaluator.get("picott_diagnostics", {}).get("rescue_must_haves") or []
            )
            try:
                rescue_queries = concept_block_query_builder.build_rescue(
                    picott_payload, sources=sources,
                    rescue_must_haves=rescue_must_haves,
                )
                for q in rescue_queries:
                    rec = q.to_query_record()
                    await db[Collections.SEARCH_QUERIES].insert_one({
                        "id": uuid.uuid4().hex,
                        "search_session_id": session_id,
                        "source": rec["source"],
                        "query_type": f"cb_{rec['query_family']}",
                        "query_family": rec["query_family"],
                        "query_string": rec["query_string"],
                        "rationale": rec.get("rationale") or "",
                        "concept_blocks_json": rec.get("concept_blocks_json") or {},
                        "concept_blocks_used_json": rec.get("concept_blocks_json") or {},
                        "mandatory_anchors_json": rec.get("mandatory_anchors") or [],
                        "optional_blocks_json": rec.get("optional_terms") or [],
                        "optional_terms_json": rec.get("optional_terms") or [],
                        "role_weights_json": role_weights_json,
                        "max_results": int(rec.get("max_results") or 50),
                        "triggered_by": "retrieval_evaluator",
                        # Phase 6B.1 — planner provenance.
                        "planner_source": "rescue",
                        "is_primary_strategy": False,
                        "is_fallback_strategy": False,
                        "fallback_reason": "retrieval_evaluator_anchor_coverage",
                        "created_at": datetime.now(timezone.utc),
                    })
                await audit_service.log(
                    "rescue_query_generated",
                    user_id=user_id, search_session_id=session_id,
                    payload={"rescue_must_haves": rescue_must_haves,
                              "count": len(rescue_queries)},
                )
            except Exception:  # noqa: BLE001
                logger.exception("rescue query generation failed (non-fatal)")

        # ---- access broker (Phase 2C): OA-first cascade ----
        await _set_status(session_id, status="brokering_access")
        try:
            with diag.stage("full_text_access_broker"):
                access_summary = await full_text_access_broker.broker_for_session(
                    session_id, client=http_client, replace_existing=True,
                )
        except Exception as exc:  # noqa: BLE001 — broker never fails the pipeline.
            logger.exception("access broker failed (non-fatal)")
            access_summary = {
                "articles_with_oa_full_text": 0,
                "articles_with_oa_pdf": 0,
                "articles_with_free_full_text_link": 0,
                "articles_with_institutional_link": 0,
                "articles_abstract_only": 0,
                "articles_unknown_access": 0,
                "sources_status": {},
                "error": f"{type(exc).__name__}: {str(exc)[:120]}",
                "note": (
                    "Subscription full text is link-out only. ClinEvidence never stores "
                    "institutional credentials or ingests paywalled PDFs."
                ),
            }
            await audit_service.log(
                "access_broker_pipeline_error",
                user_id=user_id,
                search_session_id=session_id,
                payload={"error": str(exc)[:300]},
            )
        evaluator["access_broker"] = access_summary

        # ---- Phase 4A: OA full-text fetcher (non-fatal) ----
        oa_fetch_summary: dict[str, Any] = {
            "fetched": 0, "cache_hits": 0, "failed": 0,
            "skipped_ineligible": 0, "skipped_license": 0,
            "skipped_cap_exceeded": 0, "disabled": 0, "status": "skipped",
        }
        if os.environ.get("OA_FETCHER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                from services.oa_fetcher import oa_fetcher as _oa
                await _set_status(session_id, status="fetching_oa_full_text")
                await audit_service.log(
                    "oa_fetch_stage_start",
                    user_id=user_id,
                    search_session_id=session_id,
                )
                # Top-5 ranked + included_by_default, deduped.
                top_ids = [r["_article"].get("id") or "" for r in ranked[:5]]
                default_ids = [
                    r["_article"].get("id") or "" for r in ranked
                    if r.get("included_by_default")
                ]
                seen: set[str] = set()
                target_ids: list[str] = []
                for aid in (top_ids + default_ids):
                    if not aid or aid in seen:
                        continue
                    seen.add(aid)
                    target_ids.append(aid)
                if target_ids:
                    res = await _oa.fetch_for_articles(
                        article_ids=target_ids, session_id=session_id,
                        user_id=user_id, client=http_client,
                    )
                    oa_fetch_summary = {**res["summary"], "status": "ok",
                                        "target_count": len(target_ids)}
            except Exception as exc:  # noqa: BLE001
                logger.exception("oa fetcher stage failed (non-fatal)")
                oa_fetch_summary = {**oa_fetch_summary, "status": "error",
                                    "error_message": f"{type(exc).__name__}: {str(exc)[:120]}"}
                await audit_service.log(
                    "oa_fetch_stage_error",
                    user_id=user_id, search_session_id=session_id,
                    payload={"error": str(exc)[:200]},
                )
        else:
            oa_fetch_summary["status"] = "disabled"
            oa_fetch_summary["disabled"] = 1
        evaluator["oa_fetcher"] = oa_fetch_summary

        await _set_status(
            session_id,
            status="awaiting_selection",
            extras={"evaluator_json": evaluator},
        )
        # Phase 6G — persist the canonical PICOTT term mapping on the
        # session document at finalize time so reads are O(1). Pure
        # projection — no LLM, no external call, no behavior change.
        # Best-effort: failure here must NEVER mask the pipeline result.
        try:
            db_g = get_db()
            session_doc_g = await db_g[Collections.SEARCH_SESSIONS].find_one(
                {"id": session_id}, {"_id": 0},
            )
            if isinstance(session_doc_g, dict):
                qrows_g = await (
                    db_g[Collections.SEARCH_QUERIES]
                    .find({"search_session_id": session_id}, {"_id": 0})
                    .to_list(length=200)
                )
                mapping_g = build_picott_term_mapping(
                    session_doc_g, queries=qrows_g,
                )
                await db_g[Collections.SEARCH_SESSIONS].update_one(
                    {"id": session_id},
                    {"$set": {
                        "picott_term_mapping_json":
                            mapping_g.model_dump(mode="json"),
                        "updated_at": datetime.now(timezone.utc),
                    }},
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phase6g picott_term_mapping finalize-persist failed "
                "(non-fatal) session=%s",
                session_id,
            )
        await audit_service.log(
            "search_stage_complete",
            user_id=user_id,
            search_session_id=session_id,
            payload={"counts": evaluator.get("counts"), "sources": sources},
        )

        # ---- Phase R1A — shadow reranking simulation (async, non-blocking).
        # Hooked AFTER the clinician's ``awaiting_selection`` status is
        # committed and after the ``search_stage_complete`` audit, so it
        # CANNOT add latency to the live response (the API call has
        # already returned by the time this runs in the background
        # task). Wrapped end-to-end inside the simulator itself so a
        # shadow failure can never surface to the clinician.
        try:
            from services.r1a_shadow_simulator import (
                schedule_shadow_simulation,
            )
            _r1a_source_errors: dict[str, int] = {}
            try:
                _sources_status_block = (
                    evaluator.get("sources_status") or {}
                )
                for _src_name, _src_block in _sources_status_block.items():
                    if isinstance(_src_block, dict):
                        _err = int(_src_block.get("errors", 0) or 0)
                        if _err:
                            _r1a_source_errors[str(_src_name)] = _err
            except Exception:  # noqa: BLE001
                pass
            _r1a_expansion_terms: list[str] = []
            try:
                _r1a_expansion_terms = list(
                    (parsed.get("picott_json") or {})
                    .get("expansion_terms") or [],
                )
            except Exception:  # noqa: BLE001
                _r1a_expansion_terms = []
            schedule_shadow_simulation(
                session_id=session_id,
                picott_payload=parsed.get("picott_json") or {},
                query_expansion_terms_used=_r1a_expansion_terms,
                high_precision_query_required_both_anchors=False,
                source_errors=_r1a_source_errors,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseR1A shadow simulator hook failed (non-fatal) "
                "session=%s",
                session_id,
            )

    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline failed for session %s", session_id)
        await _set_status(
            session_id,
            status="error",
            extras={"error_message": f"pipeline error: {type(exc).__name__}: {str(exc)[:200]}"},
        )
        await audit_service.log(
            "search_stage_error",
            user_id=user_id,
            search_session_id=session_id,
            payload={"error": str(exc)[:500]},
        )
    finally:
        # Phase 6F.2 — flush the session-scoped diagnostics block to
        # MongoDB and unbind the contextvar. Best-effort: must NEVER
        # mask a pipeline failure. The persisted dict is closed-set
        # numerics + closed-set strings only — see
        # services/diagnostics.py::SessionDiagnostics.to_dict.
        try:
            diag_payload = diag.to_dict()
            db = get_db()
            # Phase H1 — fold the closed-set ClinicalContextResolver
            # diagnostics into the persisted session_diagnostics_json
            # block alongside the Phase 6F.2 fields. Closed-set keys
            # only — NO question text, prompt, completion, or PHI is
            # ever written here (verified by validate_output + the
            # service-level scrubbing pass).
            await db[Collections.SEARCH_SESSIONS].update_one(
                {"id": session_id},
                {"$set": {
                    "session_diagnostics_json.llm_calls_count":
                        diag_payload["llm_calls_count"],
                    "session_diagnostics_json.llm_calls_by_stage":
                        diag_payload["llm_calls_by_stage"],
                    "session_diagnostics_json.token_estimate":
                        diag_payload["token_estimate"],
                    "session_diagnostics_json.latency_by_stage_json":
                        diag_payload["latency_by_stage_json"],
                    "session_diagnostics_json.clinical_context_resolver_called":
                        bool(resolver_diag.get("called")),
                    "session_diagnostics_json.clinical_context_resolver_mode":
                        resolver_diag.get("mode") or "fallback_active",
                    "session_diagnostics_json.clinical_context_resolver_accepted":
                        bool(resolver_diag.get("accepted")),
                    "session_diagnostics_json.clinical_context_resolver_skipped_reason":
                        resolver_diag.get("skipped_reason"),
                    "session_diagnostics_json.resolver_confidence":
                        resolver_diag.get("resolver_confidence"),
                    "session_diagnostics_json.resolver_rejection_reason":
                        resolver_diag.get("rejection_reason"),
                    "session_diagnostics_json.resolver_latency_ms":
                        int(resolver_diag.get("latency_ms") or 0),
                    "session_diagnostics_json.resolver_token_estimate":
                        dict(resolver_diag.get("token_estimate") or {}),
                    "session_diagnostics_json.interpretation_source":
                        resolver_diag.get("interpretation_source")
                        or "deterministic",
                    # Phase M0/M1/M2-shadow — closed-set bounded semantic
                    # repair diagnostics. The block was scrubbed +
                    # pydantic-validated inside services/semantic_anchor_repair.py
                    # before reaching here (no PHI, no article refs,
                    # no unapproved sources, no answer text, no
                    # promoted-validated MeSH, no raw prompt /
                    # completion). Defensive `locals()` lookup —
                    # ``parsed`` is only defined on the full
                    # interpretation path; generic-only short-circuits
                    # never reach the M-phase block.
                    "session_diagnostics_json.semantic_anchor_repair":
                        dict(
                            (locals().get("parsed") or {})
                            .get("session_diagnostics_json", {})
                            .get("semantic_anchor_repair")
                            or {}
                        ),
                    # Phase M3.2 — Active Semantic Repair clinician
                    # diagnostic block. Mirrors the structure built
                    # in the post-M-phase block above. ``None`` is
                    # persisted as ``None`` (legacy / PHI-blocked /
                    # short-circuited sessions). NO prompts, NO
                    # completions, NO PHI — only closed-set decision
                    # / shape / latency / kill-switch state fields.
                    "session_diagnostics_json.m3_2_active_repair":
                        (
                            (locals().get("parsed") or {})
                            .get("session_diagnostics_json", {})
                            .get("m3_2_active_repair")
                        ),
                    # Phase M4.1 — Active M4 clinician diagnostic block.
                    # Clinician GET restricts this to ``{applied, chip}``.
                    # ``None`` for legacy / short-circuited sessions. NO
                    # prompts, NO completions, NO PHI, NO LLM specifics.
                    "session_diagnostics_json.m4_active":
                        (
                            (locals().get("parsed") or {})
                            .get("session_diagnostics_json", {})
                            .get("m4_active")
                        ),
                }},
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "phase6f2 diagnostics flush failed (non-fatal) for session %s",
                session_id,
            )
        diag_reset(diag_token)


def schedule(
    session_id: str,
    user_id: str,
    sanitized_question: str,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
) -> asyncio.Task:
    task = asyncio.create_task(
        run_pipeline(session_id, user_id, sanitized_question, http_client=http_client)
    )
    _RUNNING.add(task)
    task.add_done_callback(_RUNNING.discard)
    return task
