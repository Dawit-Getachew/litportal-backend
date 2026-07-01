"""Phase 5A — Mocked source fixtures for the benchmark harness.

Per case, we ship a small deterministic corpus of articles in the same
shape the ranker expects (``_article`` + pre-rank ``relevance_score`` /
``overall_rank_score``). The fixtures NEVER contain PHI. They include:

- the case's gold PMIDs with realistic titles/abstracts that mention the
  required must-have anchors;
- a handful of decoys that match the case's ``negative_title_patterns``
  but NOT the required intervention, to exercise the cap/penalty layer.

The benchmark harness applies the picott-aware ranker post-process to
this corpus and emits the retrieval-side metrics required by the spec.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helper to keep fixture definitions terse.
# ---------------------------------------------------------------------------
def _mk(idx: int, *, pmid: str, title: str, abstract: str,
        article_type: str = "cohort_study", base: int = 75) -> dict[str, Any]:
    return {
        "_article": {
            "id": f"fx-{pmid}",
            "canonical_key": f"pmid:{pmid}",
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "article_type_classification": article_type,
            "publication_year": 2018 + (idx % 5),
        },
        "relevance_score": base,
        "evidence_design_score": 65,
        "overall_rank_score": base,
        # The legacy planner family is "high_recall" by default; the
        # post-processor only boosts "high_precision" / "study_type_filtered".
        "query_types_matched": ["high_precision"],
    }


def _decoy(idx: int, *, pmid: str, title: str, abstract: str,
            base: int = 80) -> dict[str, Any]:
    return {
        "_article": {
            "id": f"fx-decoy-{pmid}",
            "canonical_key": f"pmid:{pmid}",
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "article_type_classification": "cohort_study",
            "publication_year": 2019 + (idx % 5),
        },
        "relevance_score": base,
        "evidence_design_score": 60,
        "overall_rank_score": base,
        "query_types_matched": ["broad_recall"],
    }


# ---------------------------------------------------------------------------
# Per-case fixtures.
# ---------------------------------------------------------------------------

MILRINONE_SAH_FIXTURE: list[dict[str, Any]] = [
    # 9 gold articles (PMIDs from spec).
    _mk(0, pmid="33480639", base=85,
        title="Intra-arterial milrinone for cerebral vasospasm after aneurysmal SAH",
        abstract="We administered IA milrinone 0.5 mcg/kg/min in subarachnoid hemorrhage patients with vasospasm. "
                  "Cohort of 22 patients showed angiographic improvement."),
    _mk(1, pmid="22528278", base=84,
        title="Continuous intravenous milrinone infusion in delayed cerebral ischemia",
        abstract="IV milrinone bolus + continuous infusion at 0.75 mcg/kg/min for delayed cerebral ischemia in aSAH."),
    _mk(2, pmid="27842615", base=83,
        title="Cisternal milrinone administration after subarachnoid hemorrhage",
        abstract="Intra-cisternal milrinone protocol in patients with refractory vasospasm after subarachnoid hemorrhage."),
    _mk(3, pmid="33470361", base=82,
        title="Milrinone dose-finding for cerebral vasospasm in aneurysmal SAH",
        abstract="Dose-titration study (0.25–1.0 mcg/kg/min) of milrinone in aSAH with vasospasm."),
    _mk(4, pmid="33682040", base=81,
        title="Outcomes of milrinone protocol in subarachnoid hemorrhage",
        abstract="Single-center cohort comparing milrinone IV + IA in subarachnoid hemorrhage delayed cerebral ischemia."),
    _mk(5, pmid="18239182", base=80,
        title="Intra-arterial milrinone after aSAH: case series and dose schedule",
        abstract="IA milrinone dosing schedule and bolus titration in aSAH."),
    _mk(6, pmid="11322432", base=79,
        title="Continuous milrinone for delayed cerebral ischemia",
        abstract="Continuous IV milrinone infusion at 0.5 mcg/kg/min for delayed cerebral ischemia post-SAH."),
    _mk(7, pmid="34478028", base=78,
        title="Milrinone vs nimodipine for vasospasm in SAH",
        abstract="Comparison of milrinone (IV, 0.5 mcg/kg/min) vs nimodipine in cerebral vasospasm after SAH."),
    _mk(8, pmid="32217799", base=77,
        title="High-dose milrinone protocol in refractory SAH vasospasm",
        abstract="Loading bolus 50 mcg/kg followed by 1.0 mcg/kg/min infusion of milrinone in refractory aSAH vasospasm."),

    # 5 decoys — SAH topic but NO milrinone. Should be capped at 30.
    _decoy(0, pmid="77001",
        title="Bone-subtraction CT angiography for aneurysmal SAH",
        abstract="Diagnostic accuracy of bone-subtraction CTA in detecting cerebral aneurysms in subarachnoid hemorrhage."),
    _decoy(1, pmid="77002",
        title="CT angiography in acute subarachnoid hemorrhage",
        abstract="CTA performance for aneurysm detection in suspected SAH patients."),
    _decoy(2, pmid="77003",
        title="Digital subtraction angiography for SAH vasospasm",
        abstract="DSA for assessing vasospasm severity in subarachnoid hemorrhage."),
    _decoy(3, pmid="77004",
        title="Transcranial Doppler monitoring after aneurysmal SAH",
        abstract="TCD-based vasospasm screening in subarachnoid hemorrhage."),
    _decoy(4, pmid="77005",
        title="MR angiography for follow-up of SAH aneurysms",
        abstract="MRA versus DSA for SAH aneurysm surveillance."),
]


APIXABAN_AFIB_FIXTURE: list[dict[str, Any]] = [
    _mk(0, pmid="55001", base=85, article_type="rct",
        title="Apixaban versus warfarin in nonvalvular atrial fibrillation (ARISTOTLE)",
        abstract="Apixaban reduced stroke vs warfarin in nonvalvular atrial fibrillation; large RCT."),
    _mk(1, pmid="55002", base=80, article_type="systematic_review",
        title="DOAC versus warfarin for stroke prevention in atrial fibrillation",
        abstract="Systematic review of apixaban, rivaroxaban, dabigatran versus warfarin in AF."),
    _decoy(0, pmid="55101",
        title="Warfarin monitoring in NVAF",
        abstract="INR monitoring strategies for warfarin in nonvalvular atrial fibrillation (no apixaban)."),
]


NSAID_AKI_FIXTURE: list[dict[str, Any]] = [
    _mk(0, pmid="61001", base=82, article_type="cohort_study",
        title="NSAID use and risk of acute kidney injury in older adults with CKD",
        abstract="Cohort of older adults with chronic kidney disease; NSAID exposure raised AKI risk."),
    _mk(1, pmid="61002", base=80, article_type="case_control",
        title="NSAIDs and AKI in CKD: case-control",
        abstract="Case-control of NSAID exposure and acute kidney injury in patients with chronic kidney disease."),
]


SEPTIC_SHOCK_FIXTURE: list[dict[str, Any]] = [
    _mk(0, pmid="71001", base=78, article_type="narrative_review",
        title="Septic shock: definition, pathophysiology and management overview",
        abstract="Comprehensive review of septic shock pathophysiology and current treatment paradigms."),
    _mk(1, pmid="71002", base=75, article_type="guideline",
        title="Surviving Sepsis Campaign guidelines: septic shock",
        abstract="Practice guidelines for sepsis and septic shock management."),
]


COLORECTAL_SCREENING_FIXTURE: list[dict[str, Any]] = [
    _mk(0, pmid="81001", base=82, article_type="guideline",
        title="Colorectal cancer screening guidelines for average-risk adults (USPSTF)",
        abstract="Recommendations for colorectal cancer screening modalities and age ranges in average-risk adults."),
    _mk(1, pmid="81002", base=78, article_type="systematic_review",
        title="Systematic review of colorectal cancer screening strategies",
        abstract="Comparison of colonoscopy, FIT, and stool DNA for colorectal cancer screening."),
]


DDIMER_PE_FIXTURE: list[dict[str, Any]] = [
    _mk(0, pmid="91001", base=83, article_type="diagnostic_accuracy_study",
        title="D-dimer for pulmonary embolism in low-risk outpatients",
        abstract="Sensitivity and specificity of D-dimer for ruling out PE in low-pretest-probability adults."),
    _mk(1, pmid="91002", base=80, article_type="systematic_review",
        title="Diagnostic accuracy of D-dimer for PE: a systematic review",
        abstract="Meta-analysis of D-dimer diagnostic accuracy for pulmonary embolism."),
]


# ---------------------------------------------------------------------------
# Phase 5B — AMI hemoglobin / transfusion threshold case fixture.
# Gold primary: 37952133 (MINT), 33560322 (REALITY), 39714935, 39475958.
# Gold secondary: 15824203, 38408702, 39221566.
# Decoys: cardiac surgery transfusion, ICU transfusion, trauma transfusion,
#         GI bleeding transfusion, generic hemoglobin (no MI), generic AMI
#         (no anemia), hemoglobin A1c.
# ---------------------------------------------------------------------------
AMI_HEMOGLOBIN_TRANSFUSION_FIXTURE: list[dict[str, Any]] = [
    # ---- Gold primary: direct transfusion-threshold RCTs in AMI ----
    _mk(0, pmid="37952133", base=92, article_type="rct",
        title="Restrictive or Liberal Transfusion Strategy in Myocardial Infarction and Anemia (MINT)",
        abstract=(
            "Multicenter randomized trial in 3504 patients with acute myocardial infarction "
            "and anemia (hemoglobin <10 g/dL). Restrictive transfusion strategy (hemoglobin "
            "threshold ~7-8 g/dL) vs liberal transfusion strategy (hemoglobin threshold "
            "~10 g/dL). Primary composite outcome: death or recurrent MI at 30 days. "
            "MINT trial; red blood cell transfusion."
        )),
    _mk(1, pmid="33560322", base=90, article_type="rct",
        title="Restrictive vs Liberal Transfusion Strategy in Acute Myocardial Infarction (REALITY)",
        abstract=(
            "Open-label randomized trial in 668 patients with acute myocardial infarction "
            "and anemia. Restrictive transfusion (hemoglobin trigger 8 g/dL) vs liberal "
            "transfusion (hemoglobin trigger 10 g/dL). Major adverse cardiovascular events "
            "at 30 days; REALITY trial; non-inferiority. red blood cell transfusion strategy."
        )),
    _mk(2, pmid="39714935", base=88, article_type="meta_analysis",
        title=(
            "Liberal versus restrictive red blood cell transfusion strategies in "
            "patients with acute myocardial infarction: systematic review and meta-analysis"
        ),
        abstract=(
            "Systematic review and meta-analysis of randomized trials of liberal versus "
            "restrictive transfusion strategies in acute myocardial infarction and anemia. "
            "Pooled estimates for death, recurrent MI, and major cardiovascular events."
        )),
    _mk(3, pmid="39475958", base=87, article_type="systematic_review",
        title=(
            "Hemoglobin transfusion thresholds in acute coronary syndromes: a "
            "systematic review and updated meta-analysis"
        ),
        abstract=(
            "Updated systematic review including MINT, REALITY, CRIT and other RCTs of "
            "hemoglobin transfusion thresholds in acute coronary syndromes and "
            "acute myocardial infarction. Restrictive vs liberal strategies."
        )),
    # ---- Gold secondary: hemoglobin/anemia prognosis in ACS/AMI ----
    _mk(4, pmid="15824203", base=76, article_type="cohort_study",
        title=(
            "Anemia and hemoglobin levels predict outcomes in acute myocardial "
            "infarction: registry cohort"
        ),
        abstract=(
            "Observational cohort of patients with acute myocardial infarction; "
            "anemia and lower hemoglobin associated with worse mortality. CRIT trial "
            "registry data on red blood cell transfusion and hemoglobin levels."
        )),
    _mk(5, pmid="38408702", base=74, article_type="cohort_study",
        title=(
            "Transfusion practice and outcomes in acute coronary syndrome: "
            "observational analysis"
        ),
        abstract=(
            "Observational analysis of red blood cell transfusion patterns in acute "
            "coronary syndrome patients with anemia, evaluating in-hospital mortality "
            "and recurrent myocardial infarction."
        )),
    _mk(6, pmid="39221566", base=72, article_type="cohort_study",
        title="Hemoglobin levels and major cardiovascular events in AMI cohort",
        abstract=(
            "Retrospective cohort of acute myocardial infarction patients evaluating "
            "the relationship between baseline hemoglobin, anemia, and major "
            "adverse cardiovascular events."
        )),

    # ---- Decoys: wrong-population transfusion-threshold articles ----
    _decoy(0, pmid="92001", base=84,
        title="Restrictive vs liberal transfusion threshold in cardiac surgery (TITRe2)",
        abstract=(
            "Cardiac surgery transfusion threshold trial; restrictive 7.5 g/dL vs "
            "liberal 9 g/dL hemoglobin threshold. No AMI/ACS subgroup."
        )),
    _decoy(1, pmid="92002", base=82,
        title="ICU transfusion threshold trial",
        abstract=(
            "ICU transfusion threshold RCT in critically ill adults — restrictive vs "
            "liberal red blood cell transfusion; no MI cohort."
        )),
    _decoy(2, pmid="92003", base=81,
        title="Trauma transfusion threshold and outcomes",
        abstract=(
            "Trauma transfusion strategy review; restrictive vs liberal hemoglobin "
            "targets in trauma resuscitation."
        )),
    _decoy(3, pmid="92004", base=80,
        title="GI bleeding transfusion threshold RCT",
        abstract=(
            "Upper gastrointestinal bleeding transfusion threshold trial; restrictive "
            "hemoglobin 7 g/dL vs liberal hemoglobin 9 g/dL."
        )),
    # ---- Decoy: generic hemoglobin without MI/ACS ----
    _decoy(4, pmid="92005", base=78,
        title="Hemoglobin levels in chronic kidney disease anemia",
        abstract=(
            "Cohort evaluating hemoglobin targets for erythropoiesis-stimulating "
            "agents in chronic kidney disease anemia. No MI/ACS."
        )),
    # ---- Decoy: generic AMI without anemia/transfusion ----
    _decoy(5, pmid="92006", base=77,
        title="Acute myocardial infarction reperfusion strategies update",
        abstract=(
            "Review of primary PCI versus fibrinolysis in acute myocardial infarction. "
            "No transfusion, no anemia, no hemoglobin threshold."
        )),
    # ---- Decoy: hemoglobin A1c / diabetes (must NOT match without ACS+anemia+transfusion) ----
    _decoy(6, pmid="92007", base=76,
        title="Hemoglobin A1c targets in type 2 diabetes",
        abstract=(
            "Glycated hemoglobin (HbA1c) targets for type 2 diabetes management. "
            "No AMI, no transfusion."
        )),
]


# Map case_id → fixture corpus. Cases without a fixture only run the parser
# stage in the harness.
FIXTURES_BY_CASE: dict[str, list[dict[str, Any]]] = {
    "milrinone_sah": MILRINONE_SAH_FIXTURE,
    "apixaban_warfarin_afib": APIXABAN_AFIB_FIXTURE,
    "nsaid_aki_ckd_older": NSAID_AKI_FIXTURE,
    "septic_shock_background": SEPTIC_SHOCK_FIXTURE,
    "colorectal_screening_guideline": COLORECTAL_SCREENING_FIXTURE,
    "ddimer_pe_lowrisk": DDIMER_PE_FIXTURE,
    "ami_hemoglobin_transfusion": AMI_HEMOGLOBIN_TRANSFUSION_FIXTURE,
}
