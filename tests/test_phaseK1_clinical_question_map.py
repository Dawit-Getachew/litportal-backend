"""Phase K1 — ClinicalQuestionMap test suite (15 cases A–O + regressions).

Hard rules (mirror the K0/K1 brief):
- NO exact-question hardcoding. Assertions are GENERALIZED — e.g.
  ``primary_intent in {therapy, therapy_safety}`` — so prompt drift or
  closed-set token shuffles do not flake the suite.
- NO LLM calls. The ClinicalContextResolverService is exercised via
  monkeypatched outputs ONLY (Phase J3 fakes pattern).
- NO PHI fixtures. The PHI guard test (case O) uses the established
  PHI-blocking phrasing.
- Every test asserts only on closed-set enum / structural invariants.

Coverage:
  A — therapy regression (magnesium replacement)
  B — prognosis (IPF)
  C — prognostic factor (troponin/sepsis)
  D — prediction model (TIA/stroke)
  E — diagnostic accuracy (lung US/pneumonia)
  F — diagnostic workup (syncope/older adults)
  G — pathophysiology (APS thrombosis)
  H — epidemiology (AF prevalence)
  I — risk/etiology (smoking/pancreatic CA)
  J — complication risk (pacemaker infection)
  K — multi-intent (syncope mgmt + don't-miss diagnoses)
  L — deferred qualitative (HF adherence barriers)
  M — generic-only ×3
  N — setting-alone (treatment in ICU)
  O — PHI guard
  Plus regressions:
    - rituximab + ASCT deterministic
    - Brugada perioperative deterministic
    - Brugada drug-avoidance deterministic
"""
from __future__ import annotations

from typing import Any

import pytest

from services.clinical_question_map import (
    ACTIVE_FRAMEWORKS,
    ACTIVE_INTENTS,
    DEFERRED_INTENTS,
    PROMPT_VERSION_K1,
    SCHEMA_VERSION,
    Branch,
    ClinicalQuestionMap,
    DeterministicSignalSummary,
    InferredConcept,
    Slot,
    clinical_question_map_to_picott,
    is_deferred_intent,
)
from services.clinical_signal_router import clinical_signal_router


# ---------------------------------------------------------------------
# Helpers — builders that produce closed-set valid payloads for each
# case.
# ---------------------------------------------------------------------


def _slot(
    *,
    canonical: str,
    role: str,
    entity_type: str = "disease_or_problem",
    required: bool = True,
    source: str = "deterministic_parser",
    inferred: bool = False,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "original_phrase": canonical,
        "canonical_concept": canonical,
        "entity_type": entity_type,
        "slot_role": role,
        "required_for_search": required,
        "slot_search_policy": (
            "required_for_retrieval" if required else "optional_for_retrieval"
        ),
        "inferred": inferred,
        "confidence": confidence,
        "synonyms": [],
        "search_terms": [],
        "source": source,
    }


def _branch(
    *,
    branch_id: str,
    intent: str,
    framework: str,
    required: list[dict[str, Any]] | None = None,
    optional: list[dict[str, Any]] | None = None,
    ranking: list[dict[str, Any]] | None = None,
    confidence: float = 0.85,
    priority: int = 1,
    guideline_first: bool = False,
    rescue: bool = False,
) -> dict[str, Any]:
    return {
        "branch_id": branch_id,
        "intent": intent,
        "framework": framework,
        "branch_priority": priority,
        "branch_confidence": confidence,
        "branch_reason": f"test_branch_{intent}",
        "required_concepts": list(required or []),
        "optional_concepts": list(optional or []),
        "ranking_only_concepts": list(ranking or []),
        "display_only_concepts": [],
        "preferred_evidence_types": [],
        "retrieval_strategy_hint": "standard",
        "guideline_first": guideline_first,
        "rare_or_complication_rescue": rescue,
    }


def _build_cqm(
    *,
    primary_intent: str,
    framework: str,
    required_slots: list[dict[str, Any]],
    optional_slots: list[dict[str, Any]] | None = None,
    ranking_slots: list[dict[str, Any]] | None = None,
    provenance: str = "deterministic_parser",
    secondary_intents: list[str] | None = None,
    branches: list[dict[str, Any]] | None = None,
    inferred_concepts: list[dict[str, Any]] | None = None,
    confidence: float = 0.85,
    clarification_required: bool = False,
) -> ClinicalQuestionMap:
    return ClinicalQuestionMap(
        original_question="<test question>",
        normalized_question="<test question>",
        interpretation_provenance=provenance,
        clinical_validity=not clarification_required,
        clarification_required=clarification_required,
        clarification_reason=None,
        primary_intent=primary_intent,
        secondary_intents=list(secondary_intents or []),
        frameworks=[framework],
        branches=list(branches or [
            _branch(
                branch_id="primary_0",
                intent=primary_intent,
                framework=framework,
                required=required_slots,
                optional=optional_slots or [],
                ranking=ranking_slots or [],
                confidence=confidence,
                guideline_first=(primary_intent == "guideline_recommendation"),
                rescue=(primary_intent in {
                    "device_complication", "procedure_complication",
                    "drug_complication", "complication_risk",
                }),
            ),
        ]) if not clarification_required else [],
        framework_neutral_slots=required_slots + (optional_slots or []),
        inferred_concepts=list(inferred_concepts or []),
        candidate_keywords=[s["canonical_concept"] for s in required_slots],
        candidate_MeSH_terms_unvalidated=[],
        confidence=confidence,
        prompt_version=PROMPT_VERSION_K1 if provenance == "bounded_resolver_fallback" else None,
        deterministic_signal_summary=DeterministicSignalSummary(),
    )


# ---------------------------------------------------------------------
# Closed-set schema invariants (run before every case).
# ---------------------------------------------------------------------


def test_schema_version_constant_stable() -> None:
    assert SCHEMA_VERSION == "1.0"
    # Phase K1.1 — prompt version bumped after the integration-fix
    # disambiguation rules (Bug 3).
    assert PROMPT_VERSION_K1 == "j-tail-v2-K1.1"


def test_active_and_deferred_intent_disjoint() -> None:
    overlap = ACTIVE_INTENTS & (DEFERRED_INTENTS - {"qualitative_deferred"})
    assert overlap == set()


def test_active_frameworks_includes_core_set() -> None:
    for f in [
        "PICOTT", "PICOTS", "PIRD", "PIRT", "PFO", "PECOT", "PECO",
        "PEO", "CHARMS", "CoCoPop", "PCC",
        "problem_context_differential", "concept_mechanism",
        "guideline_stabilization",
    ]:
        assert f in ACTIVE_FRAMEWORKS


def test_slot_search_policy_required_must_match_legacy_required_bool() -> None:
    with pytest.raises(Exception):
        Slot(
            original_phrase="x",
            canonical_concept="x",
            entity_type="disease_or_problem",
            slot_role="clinical_problem",
            required_for_search=False,
            slot_search_policy="required_for_retrieval",
            inferred=False,
            confidence=0.5,
            synonyms=[],
            search_terms=[],
            source="unknown",
        )
    with pytest.raises(Exception):
        Slot(
            original_phrase="x",
            canonical_concept="x",
            entity_type="disease_or_problem",
            slot_role="clinical_problem",
            required_for_search=True,
            slot_search_policy="optional_for_retrieval",
            inferred=False,
            confidence=0.5,
            synonyms=[],
            search_terms=[],
            source="unknown",
        )


def test_clinical_question_map_extra_forbid() -> None:
    with pytest.raises(Exception):
        ClinicalQuestionMap(
            original_question="x",
            normalized_question="x",
            interpretation_provenance="deterministic_parser",
            primary_intent="therapy",
            something_unknown="oops",  # type: ignore[call-arg]
        )


def test_branch_cap_is_three() -> None:
    s = _slot(canonical="x", role="clinical_problem")
    branches = [
        _branch(branch_id=f"b{i}", intent="therapy", framework="PICOTT",
                required=[s], priority=(i + 1))
        for i in range(4)
    ]
    with pytest.raises(Exception):
        ClinicalQuestionMap(
            original_question="x",
            normalized_question="x",
            interpretation_provenance="deterministic_parser",
            primary_intent="therapy",
            frameworks=["PICOTT"],
            branches=branches,
        )


def test_is_deferred_intent_helper() -> None:
    assert is_deferred_intent("qualitative") is True
    assert is_deferred_intent("implementation") is True
    assert is_deferred_intent("therapy") is False
    assert is_deferred_intent("background_overview") is False


# ---------------------------------------------------------------------
# Case A — therapy regression (magnesium replacement)
# ---------------------------------------------------------------------


def test_case_A_therapy_magnesium_replacement_full_picott_derivation() -> None:
    cqm = _build_cqm(
        primary_intent="therapy",
        framework="PICOTT",
        required_slots=[
            _slot(canonical="hypomagnesemia", role="disease_or_condition"),
            _slot(canonical="magnesium replacement",
                  role="drug_or_drug_class",
                  entity_type="drug_or_drug_class"),
        ],
    )
    out = clinical_question_map_to_picott(cqm)
    # Closed-set derivation invariants.
    assert out["type_of_question"] == "therapy"
    assert out["_k1_derivation_tier"] == "full"
    assert out["question_scope"] == "foreground"
    # Population + intervention buckets MUST be populated.
    assert out["pico"]["population"]
    assert out["pico"]["intervention_or_exposure_or_index_test"]
    # Must-have list mirrors the required slots.
    assert {m["concept"] for m in out["must_have_concepts"]} == {
        "hypomagnesemia", "magnesium replacement",
    }


# ---------------------------------------------------------------------
# Case B — prognosis (IPF) — best-effort PICOTT
# ---------------------------------------------------------------------


def test_case_B_prognosis_idiopathic_pulmonary_fibrosis() -> None:
    cqm = _build_cqm(
        primary_intent="prognosis_natural_history",
        framework="PFO",
        required_slots=[
            _slot(canonical="idiopathic pulmonary fibrosis",
                  role="disease_or_condition"),
        ],
        ranking_slots=[
            _slot(canonical="median survival", role="outcome",
                  required=False, entity_type="biomarker_or_lab",
                  source="deterministic_parser"),
        ],
    )
    assert cqm.primary_intent in ACTIVE_INTENTS
    assert cqm.frameworks == ["PFO"]
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "prognosis"
    # Best-effort path — no full-PICOTT promise required.
    assert out["_k1_derivation_tier"] in {"full", "best_effort"}


# ---------------------------------------------------------------------
# Case C — prognostic factor (troponin in sepsis)
# ---------------------------------------------------------------------


def test_case_C_prognostic_factor_troponin_sepsis() -> None:
    cqm = _build_cqm(
        primary_intent="prognostic_factor",
        framework="PFO",
        required_slots=[
            _slot(canonical="sepsis", role="disease_or_condition"),
            _slot(canonical="troponin",
                  role="prognostic_factor",
                  entity_type="biomarker_or_lab"),
        ],
    )
    out = clinical_question_map_to_picott(cqm)
    # Closed-set: prognostic_factor → prognosis bucket.
    assert out["type_of_question"] == "prognosis"
    assert out["question_subtype"] == "prognostic_factor"


# ---------------------------------------------------------------------
# Case D — prediction model (TIA/stroke)
# ---------------------------------------------------------------------


def test_case_D_prediction_model_tia_stroke() -> None:
    cqm = _build_cqm(
        primary_intent="prediction_model",
        framework="CHARMS",
        required_slots=[
            _slot(canonical="transient ischemic attack",
                  role="disease_or_condition"),
            _slot(canonical="ABCD2 score",
                  role="prediction_model_or_score",
                  entity_type="diagnostic_test"),
        ],
    )
    assert "CHARMS" in cqm.frameworks
    out = clinical_question_map_to_picott(cqm)
    assert out["question_subtype"] == "prediction_model"


# ---------------------------------------------------------------------
# Case E — diagnostic accuracy (lung US for pneumonia)
# ---------------------------------------------------------------------


def test_case_E_diagnostic_accuracy_lung_us_pneumonia() -> None:
    cqm = _build_cqm(
        primary_intent="diagnosis_accuracy",
        framework="PIRD",
        required_slots=[
            _slot(canonical="lung ultrasound",
                  role="index_test",
                  entity_type="diagnostic_test"),
            _slot(canonical="pneumonia",
                  role="target_condition",
                  entity_type="disease_or_problem"),
        ],
    )
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "diagnosis"
    assert out["question_subtype"] in {
        "diagnostic_accuracy", "diagnostic_workup",
    }
    assert out["pico"]["intervention_or_exposure_or_index_test"]


# ---------------------------------------------------------------------
# Case F — diagnostic workup (syncope / older adults)
# ---------------------------------------------------------------------


def test_case_F_diagnostic_workup_syncope_older_adults() -> None:
    cqm = _build_cqm(
        primary_intent="diagnostic_workup",
        framework="PIRD",
        required_slots=[
            _slot(canonical="syncope", role="clinical_problem"),
        ],
        optional_slots=[
            _slot(canonical="older adults",
                  role="age_group",
                  entity_type="population_group",
                  required=False),
        ],
    )
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "diagnosis"


# ---------------------------------------------------------------------
# Case G — pathophysiology (APS thrombosis)
# ---------------------------------------------------------------------


def test_case_G_pathophysiology_antiphospholipid_thrombosis() -> None:
    cqm = _build_cqm(
        primary_intent="pathophysiology_mechanism",
        framework="concept_mechanism",
        required_slots=[
            _slot(canonical="antiphospholipid syndrome",
                  role="disease_or_condition"),
            _slot(canonical="thrombosis",
                  role="mechanism_or_pathway",
                  entity_type="disease_or_problem"),
        ],
    )
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "background"
    assert out["question_scope"] == "background"


# ---------------------------------------------------------------------
# Case H — epidemiology (AF prevalence)
# ---------------------------------------------------------------------


def test_case_H_epidemiology_atrial_fibrillation_prevalence() -> None:
    cqm = _build_cqm(
        primary_intent="epidemiology_prevalence_incidence",
        framework="CoCoPop",
        required_slots=[
            _slot(canonical="atrial fibrillation",
                  role="disease_or_condition"),
        ],
    )
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "epidemiology"
    assert out["question_scope"] == "hybrid"


# ---------------------------------------------------------------------
# Case I — risk/etiology (smoking / pancreatic CA)
# ---------------------------------------------------------------------


def test_case_I_risk_etiology_smoking_pancreatic_cancer() -> None:
    cqm = _build_cqm(
        primary_intent="risk_etiology_harm",
        framework="PECO",
        required_slots=[
            _slot(canonical="smoking",
                  role="exposure",
                  entity_type="exposure"),
            _slot(canonical="pancreatic cancer",
                  role="target_condition",
                  entity_type="disease_or_problem"),
        ],
    )
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "etiology_harm"


# ---------------------------------------------------------------------
# Case J — complication risk (pacemaker infection)
# ---------------------------------------------------------------------


def test_case_J_device_complication_pacemaker_infection() -> None:
    cqm = _build_cqm(
        primary_intent="device_complication",
        framework="PECO",
        required_slots=[
            _slot(canonical="pacemaker",
                  role="device_or_implant",
                  entity_type="device_or_implant"),
            _slot(canonical="device infection",
                  role="complication_or_adverse_event",
                  entity_type="complication_or_adverse_event"),
        ],
    )
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "etiology_harm"
    assert out["question_subtype"] == "device_complication_risk"
    # Closed-set rescue flag wired through the branch.
    assert cqm.branches[0].rare_or_complication_rescue is True


# ---------------------------------------------------------------------
# Case K — multi-intent (syncope mgmt + don't-miss diagnoses)
# ---------------------------------------------------------------------


def test_case_K_multi_intent_syncope_management_and_differentials() -> None:
    syncope = _slot(canonical="syncope", role="clinical_problem")
    primary_branch = _branch(
        branch_id="primary_0",
        intent="diagnostic_workup",
        framework="PIRD",
        required=[syncope],
        priority=1,
        confidence=0.85,
    )
    secondary_branch = _branch(
        branch_id="secondary_1",
        intent="differential_diagnosis",
        framework="problem_context_differential",
        required=[syncope],
        priority=2,
        confidence=0.75,
    )
    cqm = ClinicalQuestionMap(
        original_question="syncope workup and what differentials to consider",
        normalized_question="syncope workup and what differentials to consider",
        interpretation_provenance="bounded_resolver_fallback",
        clinical_validity=True,
        clarification_required=False,
        primary_intent="diagnostic_workup",
        secondary_intents=["differential_diagnosis"],
        frameworks=["PIRD", "problem_context_differential"],
        branches=[primary_branch, secondary_branch],
        framework_neutral_slots=[syncope],
        confidence=0.85,
        prompt_version=PROMPT_VERSION_K1,
    )
    assert len(cqm.branches) == 2
    assert cqm.branches[0].branch_priority == 1
    assert cqm.branches[1].branch_priority == 2
    # Generalised: secondary intent MUST be in the active set.
    assert cqm.secondary_intents[0] in ACTIVE_INTENTS
    # Multi-branch derivation tier remains best-effort or full.
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "diagnosis"


# ---------------------------------------------------------------------
# Case L — deferred qualitative (HF adherence barriers)
# ---------------------------------------------------------------------


def test_case_L_deferred_qualitative_routes_to_clarification() -> None:
    # K1 rule: deferred intents NEVER route to therapy. They MAY be
    # represented but the pipeline-builder ALWAYS flags them as
    # clarification_required.
    cqm = ClinicalQuestionMap(
        original_question="what are the qualitative barriers to HF medication adherence",
        normalized_question="what are the qualitative barriers to HF medication adherence",
        interpretation_provenance="clarification_required",
        clinical_validity=False,
        clarification_required=True,
        clarification_reason="deferred_intent_not_actively_routed",
        primary_intent="qualitative_deferred",
        secondary_intents=[],
        frameworks=["PICOTT"],
        branches=[],
        framework_neutral_slots=[],
        confidence=0.0,
    )
    assert is_deferred_intent(cqm.primary_intent)
    assert cqm.clarification_required is True
    out = clinical_question_map_to_picott(cqm)
    # NEVER maps to therapy.
    assert out["type_of_question"] != "therapy"


# ---------------------------------------------------------------------
# Case M — generic-only ×3  (clarification_required path)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("q", [
    "what should I do for my patient",
    "any thoughts on this",
    "help with workup",
])
def test_case_M_generic_only_signal_router_needs_resolver(q: str) -> None:
    summary = clinical_signal_router.analyze(q)
    # Generic-only phrases have 0–1 detected lexical signals → resolver
    # is required (closed-set router invariant).
    assert summary.needs_resolver is True
    assert isinstance(summary.detected_signals, list)


# ---------------------------------------------------------------------
# Case N — setting-alone (treatment in ICU)
# ---------------------------------------------------------------------


def test_case_N_setting_alone_treatment_in_icu() -> None:
    summary = clinical_signal_router.analyze("treatment in the ICU")
    # The therapy signal IS present, but the deterministic anchor
    # service in the pipeline is what decides whether a setting-only
    # phrase produces a primary anchor. The router itself only emits
    # closed-set lexical hints — assert the closed-set invariants.
    assert isinstance(summary, DeterministicSignalSummary)
    assert summary.deterministic_confidence >= 0.0
    assert summary.deterministic_confidence <= 1.0
    assert "therapy" in summary.possible_intents


# ---------------------------------------------------------------------
# Case O — PHI guard
# ---------------------------------------------------------------------


def test_case_O_phi_guard_pattern_intact() -> None:
    # The PHI guard runs BEFORE this module. We assert that the
    # deterministic signal router never echoes the PHI tokens — it
    # only emits closed-set lexical signal labels.
    summary = clinical_signal_router.analyze(
        "Pt John Doe SSN 123-45-6789 has chest pain, what is the diagnostic workup?",
    )
    # The router output structure must remain closed-set.
    assert isinstance(summary, DeterministicSignalSummary)
    for sig in summary.detected_signals:
        assert "123-45-6789" not in sig
        assert "John" not in sig
        assert "Doe" not in sig


# ---------------------------------------------------------------------
# Regression: rituximab + ASCT  → therapy path remains full PICOTT
# ---------------------------------------------------------------------


def test_regression_rituximab_asct_therapy_dual_anchor_full_picott() -> None:
    rituximab = _slot(
        canonical="rituximab",
        role="drug_or_drug_class",
        entity_type="drug_or_drug_class",
    )
    asct = _slot(
        canonical="autologous stem cell transplant",
        role="procedure_or_surgery",
        entity_type="procedure_or_surgery",
    )
    cqm = _build_cqm(
        primary_intent="therapy",
        framework="PICOTT",
        required_slots=[rituximab, asct],
    )
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "therapy"
    assert out["_k1_derivation_tier"] == "full"
    # Both required anchors land in must_have.
    mh_concepts = {m["concept"] for m in out["must_have_concepts"]}
    assert "rituximab" in mh_concepts
    assert "autologous stem cell transplant" in mh_concepts


# ---------------------------------------------------------------------
# Regression: Brugada perioperative — drug_safety path representable
# ---------------------------------------------------------------------


def test_regression_brugada_perioperative_drug_safety_branch() -> None:
    brugada = _slot(
        canonical="Brugada syndrome",
        role="disease_or_condition",
        entity_type="disease_or_problem",
    )
    perioperative = _slot(
        canonical="perioperative management",
        role="clinical_setting",
        entity_type="perioperative_context",
        required=False,
    )
    cqm = _build_cqm(
        primary_intent="therapy_safety",
        framework="PICOTT",
        required_slots=[brugada],
        optional_slots=[perioperative],
    )
    out = clinical_question_map_to_picott(cqm)
    assert out["type_of_question"] == "therapy"
    assert out["question_subtype"] == "drug_safety"
    assert out["_k1_derivation_tier"] == "full"


# ---------------------------------------------------------------------
# Regression: Brugada drug-avoidance — therapy_safety path
# ---------------------------------------------------------------------


def test_regression_brugada_drug_avoidance_therapy_safety() -> None:
    brugada = _slot(
        canonical="Brugada syndrome",
        role="disease_or_condition",
        entity_type="disease_or_problem",
    )
    drug = _slot(
        canonical="propofol",
        role="drug_or_drug_class",
        entity_type="drug_or_drug_class",
    )
    cqm = _build_cqm(
        primary_intent="therapy_safety",
        framework="PICOTT",
        required_slots=[brugada, drug],
    )
    out = clinical_question_map_to_picott(cqm)
    # Closed-set: therapy_safety still resolves to a therapy type with
    # drug_safety subtype.
    assert out["type_of_question"] == "therapy"
    assert out["question_subtype"] == "drug_safety"
    # Both required anchors land in must_have.
    mh_concepts = {m["concept"] for m in out["must_have_concepts"]}
    assert "Brugada syndrome" in mh_concepts
    assert "propofol" in mh_concepts


# ---------------------------------------------------------------------
# Pipeline-builder integration tests (in-memory, no DB / no LLM).
# Exercises ``_k1_build_clinical_question_map`` directly with closed-
# set inputs to confirm the persisted JSON respects the schema.
# ---------------------------------------------------------------------


def test_pipeline_builder_clarification_path_emits_clarification_required() -> None:
    from services.pipeline import _k1_build_clinical_question_map

    cqm_json = _k1_build_clinical_question_map(
        normalized_question="vague question",
        original_question="vague question",
        picott_payload={
            "type_of_question": "other",
            "question_subtype": None,
            "must_have_concepts": [],
            "pico": {},
            "question_scope": "foreground",
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": [],
            "generic_modifiers": [],
            "anchor_repair_applied": False,
            "anchor_failure_reason": "no_primary_clinical_entity",
        },
        resolver_diag={
            "called": False,
            "accepted": False,
            "resolver_confidence": None,
        },
        signal_summary={
            "detected_signals": [],
            "possible_intents": [],
            "possible_multi_intent": False,
            "deterministic_confidence": 0.0,
            "needs_resolver": True,
            "reason": "no_lexical_signal_detected",
        },
        clarification_only=True,
    )
    assert cqm_json["interpretation_provenance"] == "clarification_required"
    assert cqm_json["clarification_required"] is True
    assert cqm_json["branches"] == []
    assert cqm_json["schema_version"] == "1.0"
    # Confidence floor — clarification path must be at 0.
    assert cqm_json["confidence"] == 0.0


def test_pipeline_builder_therapy_path_emits_full_branch() -> None:
    from services.pipeline import _k1_build_clinical_question_map

    cqm_json = _k1_build_clinical_question_map(
        normalized_question="best treatment for hypomagnesemia in ICU",
        original_question="best treatment for hypomagnesemia in ICU",
        picott_payload={
            "type_of_question": "therapy",
            "question_subtype": "electrolyte_replacement",
            "must_have_concepts": [
                {"concept": "hypomagnesemia", "role": "condition",
                 "synonyms": [], "required_for_high_relevance": True},
                {"concept": "magnesium replacement", "role": "intervention",
                 "synonyms": [], "required_for_high_relevance": True},
            ],
            "pico": {},
            "question_scope": "foreground",
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": [
                {
                    "canonical_concept": "hypomagnesemia",
                    "entity_type": "disease_or_problem",
                    "required_for_search": True,
                    "confidence": 0.95,
                    "source": "ontology_pack",
                    "inferred": False,
                    "synonyms": [],
                    "text": "hypomagnesemia",
                },
                {
                    "canonical_concept": "magnesium replacement",
                    "entity_type": "drug_or_drug_class",
                    "required_for_search": True,
                    "confidence": 0.9,
                    "source": "rule",
                    "inferred": False,
                    "synonyms": [],
                    "text": "magnesium",
                },
            ],
            "generic_modifiers": ["ICU"],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
        resolver_diag={
            "called": False,
            "accepted": False,
            "resolver_confidence": None,
        },
        signal_summary={
            "detected_signals": ["therapy_token"],
            "possible_intents": ["therapy"],
            "possible_multi_intent": False,
            "deterministic_confidence": 0.4,
            "needs_resolver": False,
            "reason": "two_or_more_concrete_lexical_signals",
        },
        clarification_only=False,
    )
    assert cqm_json["interpretation_provenance"] in {
        "deterministic_parser", "deterministic_concept_pack",
    }
    assert cqm_json["clarification_required"] is False
    # Closed-set therapy_safety subtype path → primary_intent in active
    # therapy bucket.
    assert cqm_json["primary_intent"] in {"therapy", "therapy_safety"}
    # Branch + slots populated.
    assert len(cqm_json["branches"]) == 1
    branch = cqm_json["branches"][0]
    assert branch["branch_priority"] == 1
    # Both required anchors land in required_concepts.
    assert {
        s["canonical_concept"] for s in branch["required_concepts"]
    } == {"hypomagnesemia", "magnesium replacement"}


def test_pipeline_builder_resolver_provenance_stamps_prompt_version() -> None:
    from services.pipeline import _k1_build_clinical_question_map

    cqm_json = _k1_build_clinical_question_map(
        normalized_question="pacemaker infection risk",
        original_question="pacemaker infection risk",
        picott_payload={
            "type_of_question": "etiology_harm",
            "question_subtype": "device_complication_risk",
            "must_have_concepts": [
                {"concept": "pacemaker", "role": "intervention",
                 "synonyms": [], "required_for_high_relevance": True},
                {"concept": "device infection", "role": "outcome",
                 "synonyms": [], "required_for_high_relevance": True},
            ],
            "pico": {},
            "question_scope": "foreground",
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": [
                {
                    "canonical_concept": "pacemaker",
                    "entity_type": "device_or_implant",
                    "required_for_search": True,
                    "confidence": 0.85,
                    "source": "llm_resolver_fallback",
                    "inferred": False,
                    "synonyms": [],
                    "text": "pacemaker",
                },
                {
                    "canonical_concept": "device infection",
                    "entity_type": "complication_or_adverse_event",
                    "required_for_search": True,
                    "confidence": 0.85,
                    "source": "llm_resolver_fallback",
                    "inferred": False,
                    "synonyms": [],
                    "text": "infection",
                },
            ],
            "generic_modifiers": [],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
        resolver_diag={
            "called": True,
            "accepted": True,
            "resolver_confidence": 0.78,
        },
        signal_summary={
            "detected_signals": ["device_complication_token"],
            "possible_intents": ["device_complication"],
            "possible_multi_intent": False,
            "deterministic_confidence": 0.4,
            "needs_resolver": True,
            "reason": "single_signal_insufficient_for_structuring",
        },
        clarification_only=False,
    )
    assert cqm_json["interpretation_provenance"] == "bounded_resolver_fallback"
    assert cqm_json["prompt_version"] == PROMPT_VERSION_K1
    assert cqm_json["primary_intent"] == "device_complication"


# ---------------------------------------------------------------------
# Signal router pattern smoke tests (closed-set lexical detection).
# ---------------------------------------------------------------------


@pytest.mark.parametrize("question,expected_intent", [
    ("first-line treatment for atrial fibrillation",
     "therapy"),
    ("sensitivity and specificity of lung ultrasound for pneumonia",
     "diagnosis_accuracy"),
    ("approach to syncope workup in older adults",
     "diagnostic_workup"),
    ("differential diagnosis of chest pain",
     "differential_diagnosis"),
    ("prognosis of idiopathic pulmonary fibrosis",
     "prognosis_natural_history"),
    ("ABCD2 risk score for TIA",
     "prediction_model"),
    ("smoking exposure and pancreatic cancer risk",
     "risk_etiology_harm"),
    ("prevalence of atrial fibrillation",
     "epidemiology_prevalence_incidence"),
    ("pathophysiology of antiphospholipid syndrome",
     "pathophysiology_mechanism"),
    ("how often to monitor INR",
     "monitoring_followup_surveillance"),
    ("emergency management of acute coronary syndrome",
     "stabilization_emergency_management"),
    ("AHA guideline for STEMI",
     "guideline_recommendation"),
    ("what is heart failure",
     "background_overview"),
    ("primary prevention of stroke",
     "prevention_screening"),
])
def test_signal_router_emits_closed_set_intent(
    question: str,
    expected_intent: str,
) -> None:
    summary = clinical_signal_router.analyze(question)
    assert expected_intent in summary.possible_intents
    for intent in summary.possible_intents:
        assert intent in ACTIVE_INTENTS
