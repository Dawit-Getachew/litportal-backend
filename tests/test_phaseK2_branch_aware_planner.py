"""Phase K2 — Shadow Branch-Aware Boolean Query Planner tests.

All tests are pure / sync. They drive the planner directly with closed-
set ``ClinicalQuestionMap`` payloads so no live LLM / network is
required in CI. Live preview re-verification is captured separately
in the K2 memory report.

NO exact-question hardcoding. Every assertion uses generalized
closed-set membership checks (intent ∈ ACTIVE_INTENTS, family ∈
QueryFamilyName, slot_role ∈ closed-set, etc.).
"""
from __future__ import annotations

import os
import pytest

from services.branch_aware_boolean_query_planner import (
    UNVALIDATED_MESH_REASONS,
    BranchQueryPlan,
    plan_branch_queries,
)
from services.clinical_question_map import (
    ACTIVE_FRAMEWORKS,
    ACTIVE_INTENTS,
    Branch,
    ClinicalQuestionMap,
    Slot,
)


# ---------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------


def _slot(
    *,
    canonical: str,
    role: str,
    entity_type: str = "disease_or_problem",
    required: bool = True,
    source: str = "deterministic_concept_pack",
    inferred: bool = False,
    confidence: float = 0.9,
    synonyms: list[str] | None = None,
    original_phrase: str | None = None,
    policy: str | None = None,
) -> Slot:
    policy_str = policy or (
        "required_for_retrieval" if required else "optional_for_retrieval"
    )
    return Slot(
        original_phrase=original_phrase or canonical,
        canonical_concept=canonical,
        entity_type=entity_type,
        slot_role=role,
        required_for_search=required,
        slot_search_policy=policy_str,
        inferred=inferred,
        confidence=confidence,
        synonyms=list(synonyms or []),
        search_terms=[],
        source=source,
    )


def _branch(
    *,
    intent: str,
    framework: str,
    required: list[Slot],
    optional: list[Slot] | None = None,
    ranking: list[Slot] | None = None,
    display: list[Slot] | None = None,
    branch_id: str = "primary_0",
    priority: int = 1,
    confidence: float = 0.9,
    rescue: bool = False,
    guideline_first: bool = False,
) -> Branch:
    return Branch(
        branch_id=branch_id,
        intent=intent,
        framework=framework,
        branch_priority=priority,
        branch_confidence=confidence,
        branch_reason="test",
        required_concepts=required,
        optional_concepts=list(optional or []),
        ranking_only_concepts=list(ranking or []),
        display_only_concepts=list(display or []),
        preferred_evidence_types=[],
        retrieval_strategy_hint="standard",
        guideline_first=guideline_first,
        rare_or_complication_rescue=rescue,
    )


def _cqm(
    *,
    intent: str,
    framework: str,
    required: list[Slot],
    optional: list[Slot] | None = None,
    ranking: list[Slot] | None = None,
    display: list[Slot] | None = None,
    secondary_intents: list[str] | None = None,
    secondary_branches: list[Branch] | None = None,
    provenance: str = "deterministic_concept_pack",
    clarification: bool = False,
    primary_intent_override: str | None = None,
) -> ClinicalQuestionMap:
    primary = primary_intent_override or intent
    branches: list[Branch] = []
    if not clarification:
        branches.append(_branch(
            intent=intent, framework=framework,
            required=required, optional=optional or [],
            ranking=ranking or [], display=display or [],
        ))
        branches.extend(secondary_branches or [])
    return ClinicalQuestionMap(
        original_question="<test>",
        normalized_question="<test>",
        interpretation_provenance="clarification_required" if clarification else provenance,
        clinical_validity=not clarification,
        clarification_required=clarification,
        primary_intent=primary,
        secondary_intents=list(secondary_intents or []),
        frameworks=[framework]
        + ([f for f in (secondary_branches or [])[:1]
            and [secondary_branches[0].framework]] if secondary_branches else []),
        branches=branches,
        framework_neutral_slots=list(required) + list(optional or []),
        candidate_keywords=[s.canonical_concept for s in required],
        confidence=0.9,
    )


def _plan(cqm: ClinicalQuestionMap, mode: str = "shadow") -> BranchQueryPlan:
    return plan_branch_queries(cqm, mode=mode)


# ---------------------------------------------------------------------
# Hard guards: active mode rejected; closed-set schema.
# ---------------------------------------------------------------------


def test_planner_rejects_active_mode_with_assertion() -> None:
    cqm = _cqm(
        intent="therapy", framework="PICOTT",
        required=[_slot(canonical="warfarin", role="drug_or_drug_class",
                        entity_type="drug_or_drug_class")],
    )
    with pytest.raises(AssertionError):
        plan_branch_queries(cqm, mode="active")  # type: ignore[arg-type]


def test_planner_disabled_mode_returns_empty_plan() -> None:
    cqm = _cqm(
        intent="therapy", framework="PICOTT",
        required=[_slot(canonical="warfarin", role="drug_or_drug_class",
                        entity_type="drug_or_drug_class")],
    )
    plan = _plan(cqm, mode="disabled")
    assert plan.mode == "disabled"
    assert plan.active_for_retrieval is False
    assert plan.branches == []


def test_planner_clarification_required_emits_no_branches() -> None:
    cqm = _cqm(
        intent="background_overview", framework="PICOTT",
        required=[], clarification=True,
    )
    plan = _plan(cqm)
    assert plan.mode == "shadow"
    assert plan.active_for_retrieval is False
    assert plan.branches == []


def test_planner_schema_version_constant_stable() -> None:
    cqm = _cqm(
        intent="therapy", framework="PICOTT",
        required=[_slot(canonical="warfarin", role="drug_or_drug_class",
                        entity_type="drug_or_drug_class")],
    )
    plan = _plan(cqm)
    assert plan.schema_version == "K2.1.0"
    assert plan.active_for_retrieval is False
    for br in plan.branches:
        assert br.active_for_retrieval is False
        assert br.mode == "shadow"
        # Closed-set intent and framework.
        assert br.intent in ACTIVE_INTENTS
        assert br.framework in ACTIVE_FRAMEWORKS


def test_unvalidated_mesh_reasons_are_closed_set() -> None:
    expected = {
        "not_in_mesh_table", "keyword_only_concept",
        "below_confidence_floor", "generic_modifier_blocked",
        "phi_or_secret_pattern", "empty_or_invalid",
    }
    assert set(UNVALIDATED_MESH_REASONS) == expected


# ---------------------------------------------------------------------
# Case 1 — Therapy parity (magnesium ICU).
# ---------------------------------------------------------------------


def test_case_1_therapy_magnesium_icu_branch_plan_generated() -> None:
    cqm = _cqm(
        intent="therapy_safety", framework="PICOTT",
        required=[
            _slot(canonical="magnesium replacement",
                  role="intervention",
                  entity_type="management_intervention"),
            _slot(canonical="ICU", role="clinical_setting",
                  entity_type="clinical_setting"),
        ],
    )
    plan = _plan(cqm)
    assert len(plan.branches) == 1
    br = plan.branches[0]
    assert br.intent in {"therapy", "therapy_safety"}
    # benefits / risks NEVER appear as required.
    req_canon = [c.canonical_concept.lower() for c in br.required_concepts_used]
    assert "benefits" not in req_canon
    assert "risks" not in req_canon
    # Magnesium present as required (intervention).
    assert any("magnesium" in c.lower() for c in req_canon)
    # ICU present as required (setting).
    assert any("icu" == c.lower() for c in req_canon)
    # ``active_for_retrieval`` must be False.
    assert br.active_for_retrieval is False
    assert plan.active_for_retrieval is False


# ---------------------------------------------------------------------
# Case 2 — Prognosis (IPF).
# ---------------------------------------------------------------------


def test_case_2_prognosis_ipf_does_not_over_AND_outcome_or_timeframe() -> None:
    cqm = _cqm(
        intent="prognosis_natural_history", framework="PFO",
        required=[
            _slot(canonical="idiopathic pulmonary fibrosis",
                  role="disease_or_condition"),
        ],
        ranking=[
            _slot(canonical="median survival",
                  role="outcome", entity_type="biomarker_or_lab",
                  required=False, policy="ranking_only"),
        ],
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    # IPF in required; survival NOT in required, present in ranking_only.
    assert any("pulmonary fibrosis" in c.canonical_concept.lower()
               for c in br.required_concepts_used)
    rank_canon = [c.canonical_concept.lower()
                  for c in br.ranking_only_concepts_not_searched]
    assert any("survival" in r for r in rank_canon)
    # Families include prognosis-appropriate options.
    fam_names = {f.family_name for f in br.query_families}
    assert "high_recall" in fam_names
    # Prognosis method block referenced in TA terms.
    ta_lc = [t.lower() for t in br.title_abstract_terms]
    # IPF is in TA; method block words optional in title_abstract_terms
    # (we only require IPF here — method terms live in source-specific
    # query strings, not necessarily in title_abstract_terms).
    assert any("pulmonary fibrosis" in t for t in ta_lc)


# ---------------------------------------------------------------------
# Case 3 — Prognostic factor (troponin/sepsis) — role adjudication.
# ---------------------------------------------------------------------


def test_case_3_prognostic_factor_troponin_sepsis_role_adjudicated() -> None:
    cqm = _cqm(
        intent="prognostic_factor", framework="PFO",
        required=[
            _slot(canonical="sepsis", role="disease_or_condition"),
            # Mis-tagged on purpose — biomarker_or_lab + outcome role
            # should be adjudicated to prognostic_factor.
            _slot(
                canonical="troponin elevation",
                role="outcome",
                entity_type="biomarker_or_lab",
                required=True,
            ),
        ],
        optional=[
            _slot(canonical="mortality",
                  role="outcome",
                  entity_type="biomarker_or_lab",
                  required=False),
        ],
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    # Role adjudication warning surfaced.
    assert any(
        w.startswith("role_adjudicated:troponin elevation:prognostic_factor")
        for w in br.query_warnings
    ), br.query_warnings
    # sepsis + troponin elevation present in required.
    req_canon = [c.canonical_concept.lower() for c in br.required_concepts_used]
    assert "sepsis" in req_canon
    assert "troponin elevation" in req_canon
    # mortality NOT in required (optional/ranking).
    assert "mortality" not in req_canon


# ---------------------------------------------------------------------
# Case 4 — Prediction model (TIA / stroke).
# ---------------------------------------------------------------------


def test_case_4_prediction_model_emits_prediction_validation_family() -> None:
    cqm = _cqm(
        intent="prediction_model", framework="CHARMS",
        required=[
            _slot(canonical="transient ischemic attack",
                  role="disease_or_condition"),
            _slot(canonical="ABCD2 score",
                  role="prediction_model_or_score",
                  entity_type="diagnostic_test"),
        ],
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    fam_names = {f.family_name for f in br.query_families}
    assert "prediction_validation" in fam_names
    # Validation / calibration / discrimination method tokens land in
    # the prediction_validation family's query_string via the intent
    # method pack.
    pv = [f for f in br.query_families
          if f.family_name == "prediction_validation"
          and f.source == "pubmed" and f.qa_checks_passed]
    assert pv, "prediction_validation pubmed family was dropped"
    qs = pv[0].query_string.lower()
    assert any(w in qs for w in (
        "prediction model", "risk score", "validation",
        "calibration", "discrimination", "c-statistic", "auc",
    ))


# ---------------------------------------------------------------------
# Case 5 — Diagnostic accuracy (lung US / pneumonia).
# ---------------------------------------------------------------------


def test_case_5_diagnostic_accuracy_does_not_require_reference_standard() -> None:
    cqm = _cqm(
        intent="diagnosis_accuracy", framework="PIRD",
        required=[
            _slot(canonical="lung ultrasound", role="index_test",
                  entity_type="diagnostic_test"),
            _slot(canonical="pneumonia", role="target_condition",
                  entity_type="disease_or_problem"),
        ],
        optional=[
            _slot(canonical="adults", role="population",
                  entity_type="population_group", required=False),
            # Reference standard supplied as OPTIONAL — must NOT be in
            # the broad-recall AND chain.
            _slot(canonical="chest CT", role="reference_standard",
                  entity_type="diagnostic_test", required=False),
        ],
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    # high_recall + high_precision families both present.
    fam = {f.family_name: f for f in br.query_families}
    assert "high_recall" in fam
    assert "high_precision" in fam
    # Broad recall queries (high_recall, source=pubmed/europe_pmc) must
    # not AND the reference standard.
    for f in br.query_families:
        if f.family_name == "high_recall" and f.qa_checks_passed:
            assert "chest CT" not in f.query_string, f.query_string
    # High precision pulls in sens/spec terms (via _resolve_high_precision_filter).
    hp = [f for f in br.query_families
          if f.family_name == "high_precision" and f.source == "pubmed"
          and f.qa_checks_passed]
    assert hp
    assert "sensitivity" in hp[0].query_string.lower() or \
        "ROC" in hp[0].query_string


# ---------------------------------------------------------------------
# Case 6 — Diagnostic workup (syncope / older adults).
# ---------------------------------------------------------------------


def test_case_6_diagnostic_workup_emits_guideline_family() -> None:
    cqm = _cqm(
        intent="diagnostic_workup", framework="PIRD",
        required=[
            _slot(canonical="syncope", role="clinical_problem"),
        ],
        optional=[
            _slot(canonical="older adults", role="age_group",
                  entity_type="population_group", required=False),
        ],
    )
    plan = _plan(cqm)
    fam_names = {f.family_name for f in plan.branches[0].query_families}
    assert "guideline_or_consensus" in fam_names


# ---------------------------------------------------------------------
# Case 7 — Pathophysiology (APS thrombosis).
# ---------------------------------------------------------------------


def test_case_7_pathophysiology_emits_translational_family() -> None:
    cqm = _cqm(
        intent="pathophysiology_mechanism", framework="concept_mechanism",
        required=[
            _slot(canonical="antiphospholipid syndrome",
                  role="disease_or_condition"),
            _slot(canonical="thrombosis", role="mechanism_or_pathway",
                  entity_type="disease_or_problem"),
        ],
    )
    plan = _plan(cqm)
    fam_names = {f.family_name for f in plan.branches[0].query_families}
    assert "pathophysiology_translational" in fam_names


# ---------------------------------------------------------------------
# Case 8 — Epidemiology (AF prevalence) — intent-prefix strip.
# ---------------------------------------------------------------------


def test_case_8_epidemiology_strips_prevalence_prefix_from_disease() -> None:
    # Mimic the K1.2-live state: the deterministic parser emits a
    # disease slot whose canonical includes the intent prefix.
    cqm = _cqm(
        intent="epidemiology_prevalence_incidence", framework="CoCoPop",
        required=[
            _slot(
                canonical="prevalence of atrial fibrillation",
                role="disease_or_condition",
                entity_type="disease_or_problem",
                source="deterministic_parser",
                original_phrase="prevalence of atrial fibrillation",
                synonyms=["prevalence of atrial fibrillation in adults"],
            ),
        ],
        optional=[
            _slot(canonical="adult", role="population",
                  entity_type="population_group", required=False),
        ],
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    # Intent-prefix-strip warning emitted.
    assert any(
        w.startswith("intent_prefix_stripped:prevalence of")
        for w in br.query_warnings
    ), br.query_warnings
    # The required concept used MUST be "atrial fibrillation" (no
    # "prevalence of" prefix).
    req_canon = [c.canonical_concept.lower() for c in br.required_concepts_used]
    assert "atrial fibrillation" in req_canon
    for canon in req_canon:
        assert not canon.startswith("prevalence of")
    # Epidemiology method block referenced.
    ta = " ".join(br.title_abstract_terms).lower()
    assert any(w in ta for w in ("prevalence", "incidence", "epidemiology"))


# ---------------------------------------------------------------------
# Case 9 — Risk/etiology (smoking → pancreatic cancer).
# ---------------------------------------------------------------------


def test_case_9_risk_etiology_emits_cohort_family() -> None:
    cqm = _cqm(
        intent="risk_etiology_harm", framework="PECO",
        required=[
            _slot(canonical="smoking", role="exposure",
                  entity_type="exposure"),
            _slot(canonical="pancreatic cancer",
                  role="target_condition",
                  entity_type="disease_or_problem"),
        ],
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    # high_precision filter pulls in cohort / case-control / risk-factor.
    hp = [f for f in br.query_families
          if f.family_name == "high_precision" and f.source == "pubmed"
          and f.qa_checks_passed]
    assert hp
    s = hp[0].query_string.lower()
    assert any(w in s for w in ("cohort", "case-control", "risk factors"))


# ---------------------------------------------------------------------
# Case 10 — Complication (infection after pacemaker).
# ---------------------------------------------------------------------


def test_case_10_device_complication_emits_case_series_family() -> None:
    cqm = _cqm(
        intent="device_complication", framework="PECO",
        required=[
            _slot(canonical="pacemaker", role="device_or_implant",
                  entity_type="device_or_implant"),
            _slot(canonical="infection",
                  role="complication_or_adverse_event",
                  entity_type="complication_or_adverse_event"),
        ],
    )
    plan = _plan(cqm)
    fam_names = {f.family_name for f in plan.branches[0].query_families}
    assert "case_series_or_pharmacovigilance" in fam_names
    # "risk" itself is NOT a required block (was never extracted).
    req_canon = [c.canonical_concept.lower()
                 for c in plan.branches[0].required_concepts_used]
    assert "risk" not in req_canon


# ---------------------------------------------------------------------
# Case 11 — Multi-intent (syncope + don't-miss).
# ---------------------------------------------------------------------


def test_case_11_multi_intent_emits_separate_branches() -> None:
    syncope = _slot(canonical="syncope", role="clinical_problem")
    primary = _branch(
        intent="diagnostic_workup", framework="PIRD",
        required=[syncope], branch_id="primary_0", priority=1,
    )
    secondary = _branch(
        intent="differential_diagnosis",
        framework="problem_context_differential",
        required=[syncope], branch_id="secondary_1", priority=2,
    )
    cqm = ClinicalQuestionMap(
        original_question="<test>", normalized_question="<test>",
        interpretation_provenance="bounded_resolver_fallback",
        primary_intent="diagnostic_workup",
        secondary_intents=["differential_diagnosis"],
        frameworks=["PIRD", "problem_context_differential"],
        branches=[primary, secondary],
        framework_neutral_slots=[syncope],
        confidence=0.85,
    )
    plan = _plan(cqm)
    assert len(plan.branches) == 2
    intents = {br.intent for br in plan.branches}
    assert "diagnostic_workup" in intents
    assert "differential_diagnosis" in intents
    # No single over-constrained combined query — each branch's
    # source_specific_query_strings is independent.
    for br in plan.branches:
        for q in br.source_specific_query_strings.values():
            # Each branch's query DOES contain syncope.
            assert "syncope" in q.lower()


# ---------------------------------------------------------------------
# Case 12 — Deferred qualitative.
# ---------------------------------------------------------------------


def test_case_12_deferred_qualitative_no_retrievable_plan() -> None:
    cqm = ClinicalQuestionMap(
        original_question="<test>", normalized_question="<test>",
        interpretation_provenance="clarification_required",
        clinical_validity=False,
        clarification_required=True,
        primary_intent="qualitative_deferred",
        frameworks=["PICOTT"],
        branches=[],
        framework_neutral_slots=[],
        confidence=0.0,
    )
    plan = _plan(cqm)
    assert plan.active_for_retrieval is False
    assert plan.branches == []


# ---------------------------------------------------------------------
# Case 13 — Generic-only — clarification required.
# ---------------------------------------------------------------------


def test_case_13_generic_only_question_emits_empty_plan() -> None:
    cqm = ClinicalQuestionMap(
        original_question="<test>", normalized_question="<test>",
        interpretation_provenance="clarification_required",
        clinical_validity=False,
        clarification_required=True,
        primary_intent="background_overview",
        frameworks=["PICOTT"],
        branches=[],
        framework_neutral_slots=[],
        confidence=0.0,
    )
    plan = _plan(cqm)
    assert plan.branches == []
    assert plan.active_for_retrieval is False


# ---------------------------------------------------------------------
# Case 14 — PHI — defensive: a clarification_required CQM produces no
# retrievable plan even when the underlying question had PHI tokens.
# (PHI redaction happens before the CQM is built; the planner only
# guarantees that an empty/clarification CQM stays empty.)
# ---------------------------------------------------------------------


def test_case_14_phi_blocked_session_empty_plan() -> None:
    cqm = ClinicalQuestionMap(
        original_question="<test>", normalized_question="<test>",
        interpretation_provenance="clarification_required",
        clinical_validity=False,
        clarification_required=True,
        clarification_reason="phi_blocked",
        primary_intent="background_overview",
        frameworks=["PICOTT"],
        branches=[],
        framework_neutral_slots=[],
        confidence=0.0,
    )
    plan = _plan(cqm)
    assert plan.branches == []
    # Diagnostic dump must never contain SSN / email patterns.
    blob = plan.model_dump_json()
    import re
    assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", blob)
    assert "@gmail.com" not in blob


# ---------------------------------------------------------------------
# Generic-modifier deny-list demotion + dedup warnings.
# ---------------------------------------------------------------------


def test_generic_modifier_required_slot_demoted_with_warning() -> None:
    cqm = _cqm(
        intent="therapy_safety", framework="PICOTT",
        required=[
            _slot(canonical="magnesium replacement",
                  role="intervention",
                  entity_type="management_intervention"),
            # Defensive — "benefits" wrongly tagged as required population
            # gets demoted by the K2 adjudicator (back-stop on top of
            # K1.2). Slot_role is intervention so it's a defensive
            # extra layer beyond K1.2's population-only filter.
            _slot(canonical="benefits", role="intervention",
                  entity_type="management_intervention",
                  required=True),
        ],
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    # Closed-set warning emitted.
    assert any(
        w.startswith("generic_modifier_demoted:benefits")
        for w in br.query_warnings
    ), br.query_warnings
    req_canon = [c.canonical_concept.lower() for c in br.required_concepts_used]
    assert "benefits" not in req_canon


def test_specific_outcome_not_demoted_when_role_is_outcome() -> None:
    cqm = _cqm(
        intent="prognosis_natural_history", framework="PFO",
        required=[
            _slot(canonical="sepsis", role="disease_or_condition"),
            _slot(canonical="mortality", role="outcome",
                  entity_type="complication_or_adverse_event"),
        ],
    )
    plan = _plan(cqm)
    req_canon = [c.canonical_concept.lower()
                 for c in plan.branches[0].required_concepts_used]
    assert "mortality" in req_canon


def test_slot_deduplication_emits_warning_and_merges_sources() -> None:
    cqm = _cqm(
        intent="epidemiology_prevalence_incidence", framework="CoCoPop",
        required=[
            _slot(canonical="atrial fibrillation",
                  role="clinical_problem",
                  source="deterministic_concept_pack"),
            _slot(canonical="atrial fibrillation",
                  role="disease_or_condition",
                  source="deterministic_parser"),
        ],
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    # Dedup warning emitted.
    assert any(
        w.startswith("slot_deduplicated:atrial fibrillation")
        for w in br.query_warnings
    ), br.query_warnings
    # Only ONE AF entry survives.
    af_count = sum(
        1 for c in br.required_concepts_used
        if c.canonical_concept.lower() == "atrial fibrillation"
    )
    assert af_count == 1


# ---------------------------------------------------------------------
# QA validator — empty / unanchored queries are dropped.
# ---------------------------------------------------------------------


def test_qa_validator_drops_unanchored_queries() -> None:
    # A branch with NO required concepts produces NO retrievable
    # query family — every family is dropped by the
    # "has_required_anchor" check OR no required block is built.
    cqm = ClinicalQuestionMap(
        original_question="<test>", normalized_question="<test>",
        interpretation_provenance="bounded_resolver_fallback",
        primary_intent="therapy",
        frameworks=["PICOTT"],
        branches=[_branch(
            intent="therapy", framework="PICOTT",
            required=[], optional=[
                _slot(canonical="warfarin", role="drug_or_drug_class",
                      entity_type="drug_or_drug_class",
                      required=False),
            ],
        )],
        framework_neutral_slots=[],
        confidence=0.7,
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    # No required anchors → no source_specific query strings.
    assert br.source_specific_query_strings == {}


# ---------------------------------------------------------------------
# MeSH validation — resolver-suggested terms that fail the validator
# land in the unvalidated bucket.
# ---------------------------------------------------------------------


def test_unvalidated_mesh_terms_recorded_with_closed_set_reason() -> None:
    cqm = _cqm(
        intent="therapy", framework="PICOTT",
        required=[
            # canonical_concept that is NOT in the MeSH descriptor table
            # MUST land in unvalidated_MeSH_terms_rejected_or_pending.
            _slot(canonical="zzz nonexistent thing 9001",
                  role="drug_or_drug_class",
                  entity_type="drug_or_drug_class"),
        ],
    )
    plan = _plan(cqm)
    br = plan.branches[0]
    reasons = {entry.reason for entry in br.unvalidated_MeSH_terms_rejected_or_pending}
    assert reasons.issubset(UNVALIDATED_MESH_REASONS)
    assert "not_in_mesh_table" in reasons


# ---------------------------------------------------------------------
# Regression: existing PICOTT/therapy path is purely additive — the
# planner does not raise or mutate the CQM model. (Pipeline parity is
# guaranteed by the active retrieval test suite that K1 ships green.)
# ---------------------------------------------------------------------


def test_planner_is_pure_does_not_mutate_input_cqm() -> None:
    s = _slot(canonical="atrial fibrillation",
              role="clinical_problem",
              original_phrase="prevalence of atrial fibrillation")
    cqm = _cqm(
        intent="epidemiology_prevalence_incidence", framework="CoCoPop",
        required=[s],
    )
    before = cqm.model_dump(mode="json")
    _plan(cqm)
    after = cqm.model_dump(mode="json")
    assert before == after


# ---------------------------------------------------------------------
# Closed-set no-leak guard — diagnostic dump never contains forbidden
# substrings (system prompts / secrets).
# ---------------------------------------------------------------------


def test_diagnostic_dump_does_not_contain_forbidden_substrings() -> None:
    cqm = _cqm(
        intent="therapy_safety", framework="PICOTT",
        required=[
            _slot(canonical="magnesium replacement",
                  role="intervention",
                  entity_type="management_intervention"),
            _slot(canonical="ICU", role="clinical_setting",
                  entity_type="clinical_setting"),
        ],
    )
    blob = _plan(cqm).model_dump_json()
    for sub in (
        "SYSTEM_PROMPT", "Return JSON only", "You are not",
        "Confidence guidance", "JWT_ACCESS_SECRET",
        "JWT_REFRESH_SECRET", "ObjectId(", "BSON",
    ):
        assert sub not in blob


# ---------------------------------------------------------------------
# Env-flag wiring — default is shadow.
# ---------------------------------------------------------------------


def test_env_flag_default_is_shadow() -> None:
    """The pipeline reads ``CLINICAL_QUESTION_MAP_BRANCH_QUERIES_MODE``
    with default ``shadow``. The .env carries the explicit override on
    dev/staging. NEVER set to ``active``.
    """
    value = os.environ.get(
        "CLINICAL_QUESTION_MAP_BRANCH_QUERIES_MODE", "shadow"
    ).strip().lower()
    assert value in {"disabled", "shadow"}
    assert value != "active"
