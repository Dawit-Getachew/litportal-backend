"""Phase 5C — provisional benchmark cases (parse-only).

These 18 cases extend the existing 7-case set to a 25-case suite named
``phase5c_v1``. Each is marked ``provisional=True`` because we have not
sourced confident gold PMIDs at build time. Gold-PMID provenance lives
in ``/app/memory/phase5c_benchmark_provenance.md``.

The benchmark runner exercises parser + concept-block query generation
on these (catching translation/scope/specialty regressions across all
covered specialties) but excludes them from retrieval pass/fail
aggregates.
"""
from __future__ import annotations

# Phase Q1 — import ``BenchmarkCase`` from the shared common module
# (not from ``services.search_benchmark``) to break the circular
# dependency. Behavior is unchanged.
from services.search_benchmark_common import BenchmarkCase


# Phase 5C.1 — provisional cases carry ``status="needs_gold_set"`` so
# the curation workflow can graduate them later by PATCH-ing the case.
PHASE5C_PROVISIONAL_STATUS = "needs_gold_set"


PHASE5C_PROVISIONAL_CASES: list[BenchmarkCase] = [
    # ---- Cardiology (in addition to AMI transfusion which is non-provisional) ----
    BenchmarkCase(
        id="hfpef_sglt2",
        question="What is the evidence for SGLT2 inhibitors in heart failure with preserved ejection fraction?",
        expected_type_of_question={"therapy", "pharmacology"},
        expected_specialty="cardiology",
        must_have_anchors=["SGLT2 inhibitor", "heart failure"],
        provisional=True, specialty_tag="cardiology",
    ),
    BenchmarkCase(
        id="afib_doac_vs_warfarin_elderly",
        question="In elderly patients with non-valvular atrial fibrillation, how do direct oral anticoagulants compare with warfarin for stroke prevention and major bleeding?",
        expected_type_of_question={"therapy"},
        expected_specialty="cardiology",
        must_have_anchors=["atrial fibrillation", "DOAC", "warfarin"],
        provisional=True, specialty_tag="cardiology",
    ),
    BenchmarkCase(
        id="acs_dapt_duration",
        question="What is the optimal duration of dual antiplatelet therapy after acute coronary syndrome with stenting?",
        expected_type_of_question={"therapy"},
        expected_specialty="cardiology",
        must_have_anchors=["acute coronary syndrome", "dual antiplatelet therapy"],
        provisional=True, specialty_tag="cardiology",
    ),
    # ---- Critical care (milrinone/SAH is non-provisional) ----
    BenchmarkCase(
        id="septic_shock_balanced_vs_saline",
        question="In adult sepsis, do balanced crystalloids reduce mortality compared with normal saline?",
        expected_type_of_question={"therapy"},
        expected_specialty="critical_care",
        must_have_anchors=["sepsis", "balanced crystalloids", "normal saline"],
        provisional=True, specialty_tag="critical_care",
    ),
    BenchmarkCase(
        id="ards_prone_positioning",
        question="In adults with severe acute respiratory distress syndrome, does prone positioning reduce mortality?",
        expected_type_of_question={"therapy"},
        expected_specialty="critical_care",
        must_have_anchors=["ARDS", "prone positioning"],
        provisional=True, specialty_tag="critical_care",
    ),
    BenchmarkCase(
        id="vasopressor_norepi_vs_dopamine",
        question="In septic shock, does norepinephrine reduce mortality compared with dopamine?",
        expected_type_of_question={"therapy"},
        expected_specialty="critical_care",
        must_have_anchors=["septic shock", "norepinephrine", "dopamine"],
        provisional=True, specialty_tag="critical_care",
    ),
    # ---- Infectious disease ----
    BenchmarkCase(
        id="cap_oral_vs_iv_antibiotics",
        question="In hospitalized adults with non-severe community-acquired pneumonia, do oral antibiotics produce equivalent outcomes to intravenous antibiotics?",
        expected_type_of_question={"therapy"},
        expected_specialty="infectious_disease",
        must_have_anchors=["community-acquired pneumonia"],
        provisional=True, specialty_tag="infectious_disease",
    ),
    BenchmarkCase(
        id="uti_short_vs_long_course",
        question="In women with uncomplicated urinary tract infection, is a short course of antibiotics non-inferior to a longer course?",
        expected_type_of_question={"therapy"},
        expected_specialty="infectious_disease",
        must_have_anchors=["urinary tract infection"],
        provisional=True, specialty_tag="infectious_disease",
    ),
    BenchmarkCase(
        id="hiv_initial_art",
        question="What is the recommended initial antiretroviral regimen for treatment-naive adults with HIV?",
        expected_type_of_question={"guideline", "therapy"},
        expected_specialty="infectious_disease",
        must_have_anchors=["HIV", "antiretroviral therapy"],
        provisional=True, specialty_tag="infectious_disease",
    ),
    # ---- Nephrology ----
    BenchmarkCase(
        id="ckd_sglt2",
        question="What is the evidence for SGLT2 inhibitors in chronic kidney disease without diabetes?",
        expected_type_of_question={"therapy", "pharmacology"},
        expected_specialty="nephrology",
        must_have_anchors=["SGLT2 inhibitor", "chronic kidney disease"],
        provisional=True, specialty_tag="nephrology",
    ),
    BenchmarkCase(
        id="hyperkalemia_management",
        question="What is the most effective acute management of severe hyperkalemia in patients with chronic kidney disease?",
        expected_type_of_question={"therapy"},
        expected_specialty="nephrology",
        must_have_anchors=["hyperkalemia"],
        provisional=True, specialty_tag="nephrology",
    ),
    BenchmarkCase(
        id="contrast_aki_prevention",
        question="In patients undergoing coronary angiography, does intravenous saline prevent contrast-associated acute kidney injury?",
        expected_type_of_question={"prevention"},
        expected_specialty="nephrology",
        must_have_anchors=["contrast-associated acute kidney injury"],
        provisional=True, specialty_tag="nephrology",
    ),
    # ---- Pulmonology ----
    BenchmarkCase(
        id="copd_triple_therapy",
        question="In patients with severe COPD, does triple inhaler therapy reduce exacerbations compared with dual bronchodilator therapy?",
        expected_type_of_question={"therapy"},
        expected_specialty="pulmonology",
        must_have_anchors=["COPD", "triple therapy"],
        provisional=True, specialty_tag="pulmonology",
    ),
    BenchmarkCase(
        id="asthma_biologics",
        question="What is the evidence for biologic therapy in severe eosinophilic asthma?",
        expected_type_of_question={"therapy", "pharmacology"},
        expected_specialty="pulmonology",
        must_have_anchors=["eosinophilic asthma"],
        provisional=True, specialty_tag="pulmonology",
    ),
    BenchmarkCase(
        id="pulmonary_fibrosis_antifibrotic",
        question="In idiopathic pulmonary fibrosis, do antifibrotic therapies slow disease progression?",
        expected_type_of_question={"therapy"},
        expected_specialty="pulmonology",
        must_have_anchors=["idiopathic pulmonary fibrosis"],
        provisional=True, specialty_tag="pulmonology",
    ),
    # ---- Neurology / neurocritical care (milrinone/SAH covers some) ----
    BenchmarkCase(
        id="stroke_thrombectomy_window",
        question="In ischemic stroke, does mechanical thrombectomy improve outcomes beyond 6 hours from onset?",
        expected_type_of_question={"therapy"},
        expected_specialty="neurology",
        must_have_anchors=["ischemic stroke", "mechanical thrombectomy"],
        provisional=True, specialty_tag="neurology",
    ),
    BenchmarkCase(
        id="status_epilepticus_first_line",
        question="What is the first-line pharmacologic treatment for adults with convulsive status epilepticus?",
        expected_type_of_question={"therapy"},
        expected_specialty="neurology",
        must_have_anchors=["status epilepticus"],
        provisional=True, specialty_tag="neurology",
    ),
    # ---- General internal medicine ----
    BenchmarkCase(
        id="osa_cpap_cv_outcomes",
        question="In adults with moderate-to-severe obstructive sleep apnea, does CPAP reduce cardiovascular events?",
        expected_type_of_question={"therapy"},
        expected_specialty="general_internal_medicine",
        must_have_anchors=["obstructive sleep apnea", "CPAP"],
        provisional=True, specialty_tag="general_internal_medicine",
    ),
    # ---- Phase 5F — Cardiometabolic / lipidology (GLP-1/LDL).
    # Provisional retrieval (no confidently-sourced gold PMIDs at build
    # time) but parser+query expectations are exercised.
    BenchmarkCase(
        id="glp1_ldl_cholesterol",
        question="how do glp1s impact ldl cholesterole?",
        expected_type_of_question={"pharmacology", "therapy"},
        expected_specialty="endocrinology",
        must_have_anchors=["GLP-1 receptor agonists", "LDL-C"],
        provisional=True, specialty_tag="endocrinology",
    ),
    # ---- Phase 5F refinement #8 — ARDS steroid timing/dose.
    BenchmarkCase(
        id="ards_steroid_timing_dose",
        question="What is the right timing and dose to administer steroids for patients with ARDS?",
        expected_type_of_question={"therapy", "pharmacology", "dosing_protocol"},
        expected_specialty="critical_care",
        must_have_anchors=["acute respiratory distress syndrome", "corticosteroids"],
        provisional=True, specialty_tag="critical_care",
    ),
    # ---- Phase 5F refinement #8 — varicella vaccine schedule + benefit
    # (multi-intent: immunization_schedule + vaccine_effectiveness_benefit).
    BenchmarkCase(
        id="varicella_schedule_and_benefit",
        question="What is the recommended age to get varicella vaccination? What is its benefit?",
        expected_type_of_question={"prevention", "therapy"},
        expected_specialty="preventive_medicine",
        must_have_anchors=["varicella vaccine"],
        provisional=True, specialty_tag="preventive_medicine",
    ),
]

# Propagate the closed-set ``status`` tag so the curation workflow can
# graduate any provisional case via PATCH later.
for _case in PHASE5C_PROVISIONAL_CASES:
    if _case.provisional and _case.status == "active_benchmark":
        _case.status = PHASE5C_PROVISIONAL_STATUS
