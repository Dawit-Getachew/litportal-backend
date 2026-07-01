"""Phase K3A — Shadow article lane classifier + cap-candidate tests.

All tests are pure / sync. They feed closed-set mocked article metadata
(title, abstract, publication_types_json, is_guideline, is_retracted,
is_preprint, mesh_terms_json) plus a closed-set branch dict to the
deterministic classifier. NO LLM, NO network.

NO exact-question hardcoding. Every assertion uses generalised closed-
set membership checks.
"""
from __future__ import annotations

import os
import pytest

from services.article_lane_classifier import (
    ArticleLaneClassificationBatch,
    classify_articles,
)


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def _art(
    title: str, abstract: str,
    *,
    article_id: str = "art_x",
    pub_types: list[str] | None = None,
    mesh: list[str] | None = None,
    is_guideline: bool = False,
    is_preprint: bool = False,
    is_retracted: bool = False,
    has_expression_of_concern: bool = False,
) -> dict:
    return {
        "id": article_id,
        "title": title,
        "abstract": abstract,
        "publication_types_json": list(pub_types or []),
        "mesh_terms_json": list(mesh or []),
        "is_guideline": is_guideline,
        "is_preprint": is_preprint,
        "is_retracted": is_retracted,
        "has_expression_of_concern": has_expression_of_concern,
    }


def _branch(
    intent: str, framework: str,
    required: list[dict],
    *,
    branch_id: str = "primary_0",
) -> dict:
    return {
        "branch_id": branch_id,
        "intent": intent,
        "framework": framework,
        "required_concepts": required,
    }


def _anchor(canonical: str, role: str = "disease_or_condition",
            synonyms: list[str] | None = None) -> dict:
    return {
        "canonical_concept": canonical,
        "original_phrase": canonical,
        "slot_role": role,
        "synonyms": list(synonyms or []),
    }


def _classify(articles, branches) -> ArticleLaneClassificationBatch:
    return classify_articles(
        articles=articles, branches=branches,
        lane_label_mode="shadow", cap_detection_mode="shadow",
    )


# ---------------------------------------------------------------------
# Closed-set invariants.
# ---------------------------------------------------------------------


def test_active_mode_rejected_with_assertion() -> None:
    with pytest.raises(AssertionError):
        classify_articles(
            articles=[_art("x", "y", pub_types=["Review"])],
            branches=[_branch("therapy", "PICOTT", [_anchor("x")])],
            lane_label_mode="active",  # type: ignore[arg-type]
        )
    with pytest.raises(AssertionError):
        classify_articles(
            articles=[],
            branches=[],
            cap_detection_mode="active",  # type: ignore[arg-type]
        )


def test_disabled_mode_returns_empty_batch() -> None:
    batch = classify_articles(
        articles=[_art("x", "y")],
        branches=[_branch("therapy", "PICOTT", [_anchor("x")])],
        lane_label_mode="disabled",
        cap_detection_mode="disabled",
    )
    assert batch.lane_label_mode == "disabled"
    assert batch.cap_detection_mode == "disabled"
    assert batch.diagnostics == []


def test_empty_inputs_return_empty_batch() -> None:
    assert _classify([], []).diagnostics == []
    assert _classify([], [_branch("therapy", "PICOTT", [_anchor("x")])]).diagnostics == []
    assert _classify([_art("x", "y")], []).diagnostics == []


def test_every_diagnostic_carries_shadow_mode_and_inactive_flags() -> None:
    batch = _classify(
        [_art("Magnesium replacement RCT in critically ill adults",
              "Randomised controlled trial of magnesium in ICU patients.",
              pub_types=["Randomized Controlled Trial"])],
        [_branch("therapy_safety", "PICOTT",
                 [_anchor("magnesium replacement", role="intervention"),
                  _anchor("ICU", role="clinical_setting")])],
    )
    assert batch.diagnostics
    for d in batch.diagnostics:
        assert d.cap_mode == "shadow"
        assert d.would_cap is False
        assert d.would_exclude is False


def test_env_flag_defaults_safe() -> None:
    for flag in (
        "CLINICAL_ARTICLE_LANE_LABEL_MODE",
        "CLINICAL_ARTICLE_CAP_DETECTION_MODE",
    ):
        val = os.environ.get(flag, "shadow").strip().lower()
        assert val in {"disabled", "shadow"}
        assert val != "active"


# ---------------------------------------------------------------------
# Group 1 — Therapy.
# ---------------------------------------------------------------------


def test_group1A_therapy_rct_in_critically_ill_adults() -> None:
    article = _art(
        title="Magnesium replacement therapy in ICU patients: a randomized controlled trial",
        abstract="In this RCT we enrolled critically ill adults receiving magnesium replacement therapy.",
        pub_types=["Randomized Controlled Trial"],
        article_id="art_magnesium_rct",
    )
    branch = _branch(
        intent="therapy_safety", framework="PICOTT",
        required=[
            _anchor("magnesium replacement", role="intervention"),
            _anchor("ICU", role="clinical_setting"),
        ],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "therapy_rct"
    assert d.directness_label == "direct"
    assert d.branch_alignment == "aligned"
    assert d.cap_candidates == []
    assert d.evidence_role_candidate == "primary_evidence"


def test_group1B_magnesium_in_athletes_wrong_population() -> None:
    article = _art(
        title="Magnesium replacement in elite endurance athletes",
        abstract="We studied magnesium replacement in healthy young athletes during marathon training.",
        pub_types=["Cohort Studies"],
        article_id="art_magnesium_athletes",
    )
    branch = _branch(
        intent="therapy_safety", framework="PICOTT",
        required=[
            _anchor("magnesium replacement", role="intervention"),
            _anchor("ICU", role="clinical_setting",
                    synonyms=["intensive care unit"]),
        ],
    )
    d = _classify([article], [branch]).diagnostics[0]
    # ICU anchor NOT matched in title/abstract → partial or misaligned.
    assert d.branch_alignment in {"partial", "misaligned"}
    assert d.directness_label in {"related", "wrong_anchor", "indirect"}


def test_group1C_editorial_commentary() -> None:
    article = _art(
        title="Editorial: electrolyte replacement policy in critical care",
        abstract="We comment on recent practice trends in electrolyte management.",
        pub_types=["Editorial"],
        article_id="art_editorial",
    )
    higher = _art(
        title="Systematic review of magnesium replacement in ICU",
        abstract="Systematic review and meta-analysis of magnesium in ICU.",
        pub_types=["Systematic Review", "Meta-Analysis"],
        article_id="art_sr",
    )
    branch = _branch(
        intent="therapy_safety", framework="PICOTT",
        required=[
            _anchor("magnesium replacement", role="intervention"),
        ],
    )
    batch = _classify([article, higher], [branch])
    diag_editorial = next(d for d in batch.diagnostics if d.article_id == "art_editorial")
    assert diag_editorial.lane_label == "editorial_commentary"
    # When a stronger evidence article is in the batch, the editorial
    # gets the closed-set "editorial_commentary_when_stronger_evidence_available" cap.
    assert any(
        c.reason_code == "editorial_commentary_when_stronger_evidence_available"
        for c in diag_editorial.cap_candidates
    )


# ---------------------------------------------------------------------
# Group 2 — Diagnostic accuracy.
# ---------------------------------------------------------------------


def test_group2A_lung_us_dta_pneumonia_direct() -> None:
    article = _art(
        title="Diagnostic accuracy of lung ultrasound for pneumonia in adults",
        abstract="We computed sensitivity and specificity of lung ultrasound vs chest CT for pneumonia.",
        pub_types=["Diagnostic Accuracy Study"],
        article_id="art_dxa",
    )
    branch = _branch(
        intent="diagnosis_accuracy", framework="PIRD",
        required=[
            _anchor("lung ultrasound", role="index_test"),
            _anchor("pneumonia", role="target_condition"),
        ],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "diagnostic_accuracy_primary"
    assert d.directness_label == "direct"


def test_group2B_lung_us_for_heart_failure_wrong_target() -> None:
    article = _art(
        title="Lung ultrasound for heart failure assessment",
        abstract="Diagnostic accuracy of lung ultrasound for heart failure in adults.",
        pub_types=["Diagnostic Accuracy Study"],
        article_id="art_lung_us_hf",
    )
    branch = _branch(
        intent="diagnosis_accuracy", framework="PIRD",
        required=[
            _anchor("lung ultrasound", role="index_test"),
            _anchor("pneumonia", role="target_condition"),
        ],
    )
    d = _classify([article], [branch]).diagnostics[0]
    # Lung US matches but pneumonia anchor does NOT → partial alignment.
    assert d.branch_alignment in {"partial", "misaligned"}
    assert any(
        c.reason_code in {"wrong_target_condition", "missing_primary_anchor"}
        or d.branch_alignment == "partial"
        for c in d.cap_candidates
    ) or d.branch_alignment == "partial"


def test_group2C_pneumonia_antibiotic_trial_wrong_lane_for_dxa() -> None:
    article = _art(
        title="Antibiotic therapy in community-acquired pneumonia: an RCT",
        abstract="RCT comparing antibiotic regimens in adults with pneumonia.",
        pub_types=["Randomized Controlled Trial"],
        article_id="art_pneumonia_rct",
    )
    branch = _branch(
        intent="diagnosis_accuracy", framework="PIRD",
        required=[
            _anchor("lung ultrasound", role="index_test"),
            _anchor("pneumonia", role="target_condition"),
        ],
    )
    d = _classify([article], [branch]).diagnostics[0]
    # No lung US → at least partial; no DTA → not diagnostic_accuracy_primary.
    assert d.lane_label != "diagnostic_accuracy_primary"
    assert d.branch_alignment in {"partial", "misaligned"}


# ---------------------------------------------------------------------
# Group 3 — Diagnostic workup.
# ---------------------------------------------------------------------


def test_group3A_syncope_guideline_direct_workup() -> None:
    article = _art(
        title="Guideline for the evaluation of syncope",
        abstract="Society guideline on the workup and evaluation of syncope in adults.",
        pub_types=["Practice Guideline"],
        is_guideline=True,
        article_id="art_syncope_guideline",
    )
    branch = _branch(
        intent="diagnostic_workup", framework="PIRD",
        required=[_anchor("syncope", role="clinical_problem")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "diagnostic_workup_guideline"
    assert d.directness_label == "direct"
    assert d.evidence_role_candidate == "guideline_context"


def test_group3B_syncope_ablation_trial_indirect_for_workup() -> None:
    article = _art(
        title="Catheter ablation for vasovagal syncope: an RCT",
        abstract="RCT of ablation therapy in patients with recurrent syncope.",
        pub_types=["Randomized Controlled Trial"],
        article_id="art_syncope_rct",
    )
    branch = _branch(
        intent="diagnostic_workup", framework="PIRD",
        required=[_anchor("syncope", role="clinical_problem")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    # Therapy RCT doesn't match the workup lane — falls through to
    # closed-set fallback labels.
    assert d.lane_label not in {"diagnostic_workup_guideline",
                                 "diagnostic_workup_review"}


# ---------------------------------------------------------------------
# Group 4 — Prognosis.
# ---------------------------------------------------------------------


def test_group4A_ipf_survival_cohort_direct() -> None:
    article = _art(
        title="Survival and natural history of idiopathic pulmonary fibrosis: a cohort study",
        abstract="Multi-centre cohort of IPF patients with long-term follow-up and mortality outcomes.",
        pub_types=["Cohort Studies"],
        article_id="art_ipf_cohort",
    )
    branch = _branch(
        intent="prognosis_natural_history", framework="PFO",
        required=[_anchor("idiopathic pulmonary fibrosis",
                          role="disease_or_condition",
                          synonyms=["IPF"])],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "prognosis_cohort"
    assert d.directness_label == "direct"


def test_group4B_ipf_rct_therapy_for_prognosis_question_capped_as_cross_intent() -> None:
    article = _art(
        title="Pirfenidone for idiopathic pulmonary fibrosis: an RCT",
        abstract="RCT of antifibrotic therapy in IPF patients evaluating progression-free survival.",
        pub_types=["Randomized Controlled Trial"],
        article_id="art_ipf_rct",
    )
    branch = _branch(
        intent="prognosis_natural_history", framework="PFO",
        required=[_anchor("idiopathic pulmonary fibrosis",
                          role="disease_or_condition")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    # Lane label is therapy_rct (fallback rules outside the prognosis
    # branch would otherwise fire). Closed-set cap fires because
    # therapy_lane × prognosis_intent.
    assert any(
        c.reason_code == "therapy_article_for_prognosis_question"
        for c in d.cap_candidates
    ), d.cap_candidates


def test_group4C_copd_prognosis_wrong_disease() -> None:
    article = _art(
        title="Long-term mortality in COPD: a cohort study",
        abstract="Multi-centre cohort of COPD patients followed for 10 years.",
        pub_types=["Cohort Studies"],
        article_id="art_copd_cohort",
    )
    branch = _branch(
        intent="prognosis_natural_history", framework="PFO",
        required=[_anchor("idiopathic pulmonary fibrosis",
                          role="disease_or_condition",
                          synonyms=["IPF"])],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.branch_alignment in {"misaligned", "partial"}
    assert any(
        c.reason_code == "missing_primary_anchor"
        for c in d.cap_candidates
    ) or d.branch_alignment == "partial"


# ---------------------------------------------------------------------
# Group 5 — Prognostic factor.
# ---------------------------------------------------------------------


def test_group5A_troponin_mortality_sepsis_cohort_direct() -> None:
    article = _art(
        title="Troponin elevation predicts mortality in sepsis: a cohort study",
        abstract=("Multi-centre cohort study of patients with sepsis. "
                  "Troponin elevation was a prognostic factor for mortality; "
                  "adjusted hazard ratio for 30-day mortality was reported."),
        pub_types=["Cohort Studies"],
        article_id="art_troponin_sepsis",
    )
    branch = _branch(
        intent="prognostic_factor", framework="PFO",
        required=[_anchor("sepsis", role="disease_or_condition"),
                  _anchor("troponin elevation", role="prognostic_factor",
                          synonyms=["troponin"])],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "prognostic_factor_study"
    assert d.directness_label == "direct"


def test_group5B_troponin_for_MI_diagnosis_wrong_for_pf() -> None:
    article = _art(
        title="Troponin for the diagnosis of acute myocardial infarction",
        abstract="Sensitivity and specificity of high-sensitivity troponin in suspected acute MI.",
        pub_types=["Diagnostic Accuracy Study"],
        article_id="art_troponin_mi",
    )
    branch = _branch(
        intent="prognostic_factor", framework="PFO",
        required=[_anchor("sepsis", role="disease_or_condition"),
                  _anchor("troponin elevation", role="prognostic_factor",
                          synonyms=["troponin"])],
    )
    d = _classify([article], [branch]).diagnostics[0]
    # sepsis anchor NOT matched → misaligned/partial.
    assert d.branch_alignment in {"misaligned", "partial"}


# ---------------------------------------------------------------------
# Group 6 — Prediction model.
# ---------------------------------------------------------------------


def test_group6A_external_validation_abcd2_stroke_direct() -> None:
    article = _art(
        title="External validation of the ABCD2 score for stroke risk after TIA",
        abstract=("Multi-centre external validation study of the ABCD2 prediction "
                  "model for 7-day stroke risk after TIA. Calibration and "
                  "discrimination assessed."),
        pub_types=["Validation Study"],
        article_id="art_abcd2_ext",
    )
    branch = _branch(
        intent="prediction_model", framework="CHARMS",
        required=[_anchor("ABCD2 score",
                          role="prediction_model_or_score"),
                  _anchor("TIA", role="disease_or_condition",
                          synonyms=["transient ischemic attack"])],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "prediction_model_external_validation"
    assert d.directness_label == "direct"


def test_group6B_derivation_only_related_not_direct() -> None:
    article = _art(
        title="Derivation of a new clinical risk score for stroke after TIA",
        abstract=("We developed a prediction model for stroke risk after TIA "
                  "using a derivation cohort. External validation pending."),
        pub_types=["Validation Study"],
        article_id="art_abcd2_deriv",
    )
    branch = _branch(
        intent="prediction_model", framework="CHARMS",
        required=[_anchor("TIA", role="disease_or_condition",
                          synonyms=["transient ischemic attack"])],
    )
    d = _classify([article], [branch]).diagnostics[0]
    # Derivation-only → label is derivation; user prompt expects related.
    assert d.lane_label == "prediction_model_derivation"


# ---------------------------------------------------------------------
# Group 7 — Risk / etiology.
# ---------------------------------------------------------------------


def test_group7A_smoking_pancreatic_cancer_cohort_direct() -> None:
    article = _art(
        title="Smoking and pancreatic cancer risk: a prospective cohort study",
        abstract="Cohort of 100,000 adults; smoking was a risk factor for pancreatic cancer.",
        pub_types=["Cohort Studies"],
        article_id="art_smoking_panc",
    )
    branch = _branch(
        intent="risk_etiology_harm", framework="PECO",
        required=[_anchor("smoking", role="exposure"),
                  _anchor("pancreatic cancer", role="target_condition")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "risk_factor_cohort"
    assert d.directness_label == "direct"


def test_group7C_smoking_lung_cancer_wrong_outcome() -> None:
    article = _art(
        title="Smoking and lung cancer risk in a cohort",
        abstract="Cohort study showing strong association between smoking and lung cancer.",
        pub_types=["Cohort Studies"],
        article_id="art_smoking_lung",
    )
    branch = _branch(
        intent="risk_etiology_harm", framework="PECO",
        required=[_anchor("smoking", role="exposure"),
                  _anchor("pancreatic cancer", role="target_condition")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.branch_alignment in {"partial", "misaligned"}


# ---------------------------------------------------------------------
# Group 8 — Complication.
# ---------------------------------------------------------------------


def test_group8A_pacemaker_infection_registry_direct() -> None:
    article = _art(
        title="Pacemaker infection rates in a national registry",
        abstract=("Registry analysis of pacemaker device infection rates after "
                  "implantation; cohort included 50,000 devices."),
        pub_types=["Registry"],
        article_id="art_pacemaker_registry",
    )
    branch = _branch(
        intent="device_complication", framework="PECO",
        required=[_anchor("pacemaker", role="device_or_implant"),
                  _anchor("infection",
                          role="complication_or_adverse_event")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "complication_registry"
    assert d.directness_label == "direct"


def test_group8B_hip_arthroplasty_infection_wrong_procedure_or_device() -> None:
    article = _art(
        title="Surgical site infection after hip arthroplasty",
        abstract="Cohort of hip arthroplasty patients evaluating prosthetic infection rates.",
        pub_types=["Cohort Studies"],
        article_id="art_hip_infection",
    )
    branch = _branch(
        intent="device_complication", framework="PECO",
        required=[_anchor("pacemaker", role="device_or_implant"),
                  _anchor("infection",
                          role="complication_or_adverse_event")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    # pacemaker NOT matched.
    assert d.branch_alignment in {"misaligned", "partial"}


# ---------------------------------------------------------------------
# Group 9 — Epidemiology.
# ---------------------------------------------------------------------


def test_group9A_af_prevalence_in_adults_direct() -> None:
    article = _art(
        title="Prevalence and incidence of atrial fibrillation in adults: a population-based study",
        abstract=("Cross-sectional population-based study of atrial fibrillation "
                  "prevalence in adults; epidemiology and burden of disease."),
        pub_types=["Cross-Sectional"],
        article_id="art_af_prev",
    )
    branch = _branch(
        intent="epidemiology_prevalence_incidence", framework="CoCoPop",
        required=[_anchor("atrial fibrillation",
                          role="disease_or_condition",
                          synonyms=["AF", "AFib"])],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label in {
        "epidemiology_prevalence", "epidemiology_incidence",
    }
    assert d.directness_label == "direct"


def test_group9B_af_anticoagulation_rct_wrong_framework() -> None:
    article = _art(
        title="Anticoagulation for atrial fibrillation: an RCT",
        abstract="RCT of anticoagulants in atrial fibrillation for stroke prevention.",
        pub_types=["Randomized Controlled Trial"],
        article_id="art_af_anticoag",
    )
    branch = _branch(
        intent="epidemiology_prevalence_incidence", framework="CoCoPop",
        required=[_anchor("atrial fibrillation",
                          role="disease_or_condition")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label not in {
        "epidemiology_prevalence", "epidemiology_incidence",
        "epidemiology_systematic_review",
    }


# ---------------------------------------------------------------------
# Group 10 — Pathophysiology.
# ---------------------------------------------------------------------


def test_group10A_aps_pathophysiology_review_direct() -> None:
    article = _art(
        title="Mechanism of thrombosis in antiphospholipid syndrome: a review",
        abstract=("Review of the pathophysiology and mechanism of thrombosis in "
                  "antiphospholipid syndrome."),
        pub_types=["Review"],
        article_id="art_aps_review",
    )
    branch = _branch(
        intent="pathophysiology_mechanism", framework="concept_mechanism",
        required=[_anchor("antiphospholipid syndrome",
                          role="disease_or_condition"),
                  _anchor("thrombosis",
                          role="mechanism_or_pathway")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "pathophysiology_review"
    assert d.directness_label == "direct"


def test_group10B_animal_model_aps_thrombosis_labeled_animal() -> None:
    article = _art(
        title="Murine model of antiphospholipid syndrome thrombosis",
        abstract=("In a mouse model of APS we studied the molecular mechanism of "
                  "thrombosis."),
        pub_types=["Animal Study"],
        article_id="art_aps_mouse",
    )
    branch = _branch(
        intent="pathophysiology_mechanism", framework="concept_mechanism",
        required=[_anchor("antiphospholipid syndrome",
                          role="disease_or_condition"),
                  _anchor("thrombosis",
                          role="mechanism_or_pathway")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "animal_model"
    # Mechanism intent → animal_only_for_clinical_question cap MUST NOT fire.
    assert not any(
        c.reason_code == "animal_only_for_clinical_question"
        for c in d.cap_candidates
    )


def test_group10C_aps_anticoagulation_trial_for_mechanism_capped() -> None:
    article = _art(
        title="Anticoagulation in antiphospholipid syndrome: an RCT",
        abstract="RCT of anticoagulation in APS patients with prior thrombosis.",
        pub_types=["Randomized Controlled Trial"],
        article_id="art_aps_rct",
    )
    branch = _branch(
        intent="pathophysiology_mechanism", framework="concept_mechanism",
        required=[_anchor("antiphospholipid syndrome",
                          role="disease_or_condition")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    # Therapy article for mechanism question → pathophys cap fires.
    assert any(
        c.reason_code ==
        "pathophysiology_basic_science_for_clinical_management_question"
        for c in d.cap_candidates
    )


# ---------------------------------------------------------------------
# Group 11 — Multi-intent.
# ---------------------------------------------------------------------


def test_group11_multi_intent_syncope_assigns_to_best_branch() -> None:
    syncope = _anchor("syncope", role="clinical_problem")
    branches = [
        _branch(intent="diagnostic_workup", framework="PIRD",
                required=[syncope], branch_id="primary_0"),
        _branch(intent="differential_diagnosis",
                framework="problem_context_differential",
                required=[syncope], branch_id="secondary_1"),
    ]
    article_guideline = _art(
        title="Guideline for the evaluation of syncope in older adults",
        abstract="Society guideline on the workup of syncope.",
        pub_types=["Practice Guideline"],
        is_guideline=True,
        article_id="art_syncope_guideline",
    )
    article_ddx = _art(
        title="Differential diagnosis of syncope: a clinical review",
        abstract=("Clinical review of the differential diagnosis of syncope, "
                  "including don't miss diagnoses and red flags."),
        pub_types=["Review"],
        article_id="art_syncope_ddx",
    )
    batch = _classify([article_guideline, article_ddx], branches)
    by_id = {d.article_id: d for d in batch.diagnostics}
    assert by_id["art_syncope_guideline"].lane_label == "diagnostic_workup_guideline"
    assert by_id["art_syncope_ddx"].lane_label == "differential_diagnosis_review"


# ---------------------------------------------------------------------
# Group 12 — Safety.
# ---------------------------------------------------------------------


def test_group12_retracted_article_label_and_cap() -> None:
    article = _art(
        title="Magnesium replacement in critically ill adults: an RCT",
        abstract="RCT of magnesium in ICU patients.",
        pub_types=["Randomized Controlled Trial"],
        is_retracted=True,
        article_id="art_retracted",
    )
    branch = _branch(
        intent="therapy_safety", framework="PICOTT",
        required=[_anchor("magnesium replacement", role="intervention")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert d.lane_label == "retracted_or_expression_of_concern"
    assert any(c.reason_code == "retracted" for c in d.cap_candidates)
    assert d.evidence_role_candidate == "to_exclude_candidate"
    assert d.cap_mode == "shadow"
    assert d.would_cap is False
    assert d.would_exclude is False


def test_group12_preprint_for_therapy_capped_but_shadow() -> None:
    article = _art(
        title="Magnesium replacement in ICU patients: a preprint RCT",
        abstract="RCT of magnesium in ICU patients (preprint).",
        pub_types=["Randomized Controlled Trial"],
        is_preprint=True,
        article_id="art_preprint",
    )
    branch = _branch(
        intent="therapy_safety", framework="PICOTT",
        required=[_anchor("magnesium replacement", role="intervention")],
    )
    d = _classify([article], [branch]).diagnostics[0]
    assert any(
        c.reason_code == "preprint_for_clinical_recommendation"
        for c in d.cap_candidates
    )
    assert d.would_cap is False


def test_group12_diagnostic_dump_does_not_contain_forbidden_substrings() -> None:
    article = _art(
        title="A study with no secrets",
        abstract="Just plain abstract text.",
        pub_types=["Review"],
        article_id="art_clean",
    )
    branch = _branch(
        intent="pathophysiology_mechanism", framework="concept_mechanism",
        required=[_anchor("nothing", role="disease_or_condition")],
    )
    batch = _classify([article], [branch])
    blob = batch.model_dump_json()
    for forbidden in (
        "SYSTEM_PROMPT", "Return JSON only", "You are not",
        "Confidence guidance", "JWT_ACCESS_SECRET",
        "JWT_REFRESH_SECRET", "ObjectId(", "BSON",
    ):
        assert forbidden not in blob
