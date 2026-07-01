"""Phase 5A — StudyTypeRouter.

Single source of truth for:
- Legacy ``clinical_question_type`` ↔ new ``type_of_question`` mapping.
- Per-type pathway data: preferred + acceptable study types, MeSH/PubMed
  publication-type filters, query focus, ranking boosts, common penalties.

Pathway data is consumed by:
- The concept-block query builder (publication-type filter blocks).
- The evidence ranker (per-type design table + boosts + penalties).
- The retrieval quality evaluator (target study-type coverage).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Legacy ↔ new bidirectional mapping.
# ---------------------------------------------------------------------------
LEGACY_TO_NEW: dict[str, str] = {
    "treatment": "therapy",
    "diagnosis": "diagnosis",
    "prognosis": "prognosis",
    "harm": "etiology_harm",
    "screening": "prevention",        # screening is a prevention question
    "prevention": "prevention",
    "etiology": "etiology_harm",
    "guideline": "guideline",
    "epidemiology": "epidemiology",
    "pharmacology": "pharmacology",
    "other": "other",
}

NEW_TO_LEGACY: dict[str, str] = {
    "therapy": "treatment",
    "diagnosis": "diagnosis",
    "prognosis": "prognosis",
    "etiology_harm": "harm",         # collapse to "harm" by default
    "prevention": "prevention",
    "clinical_exam": "diagnosis",    # clinical exam is a kind of diagnostic
    "cost": "other",                 # no legacy bucket; route to other
    "pharmacology": "pharmacology",
    "dosing_protocol": "pharmacology",
    "guideline": "guideline",
    "epidemiology": "epidemiology",
    "background": "other",
    "other": "other",
}


def legacy_to_new(legacy: str) -> str:
    return LEGACY_TO_NEW.get((legacy or "").lower().strip(), "other")


def new_to_legacy(new: str) -> str:
    return NEW_TO_LEGACY.get((new or "").lower().strip(), "other")


# ---------------------------------------------------------------------------
# Pathway data per type_of_question.
# ---------------------------------------------------------------------------

@dataclass
class StudyTypePathway:
    """Pathway configuration for a single ``type_of_question``."""
    type_of_question: str
    # Top-tier study types — highest design weight, included by default.
    preferred_study_types: list[str] = field(default_factory=list)
    # Also-acceptable study types (lower weight, still high-relevance candidates).
    acceptable_study_types: list[str] = field(default_factory=list)
    # PubMed [Publication Type] tag values.
    pubmed_pub_types: list[str] = field(default_factory=list)
    # Europe PMC PUB_TYPE values (case-insensitive).
    europe_pmc_pub_types: list[str] = field(default_factory=list)
    # Free-text focus hints used by the concept-block query builder.
    query_focus: list[str] = field(default_factory=list)
    # Per-design ranking boosts applied on top of the relevance baseline.
    ranking_boosts: dict[str, int] = field(default_factory=dict)
    # Penalty codes (closed set). Used by ranker + UI explanations.
    common_penalties: list[str] = field(default_factory=list)
    # Optional human-friendly subtype label exposed in PICOTT.
    subtype_label: Optional[str] = None


# A compact set of pathway configurations covering all 11 ``type_of_question``
# values that the spec asked us to implement.
_PATHWAYS: dict[str, StudyTypePathway] = {
    "therapy": StudyTypePathway(
        type_of_question="therapy",
        preferred_study_types=["systematic_review", "meta_analysis", "rct", "guideline"],
        acceptable_study_types=["cohort_study", "narrative_review"],
        pubmed_pub_types=[
            "Randomized Controlled Trial", "Systematic Review",
            "Meta-Analysis", "Practice Guideline", "Guideline",
        ],
        europe_pmc_pub_types=[
            "randomized controlled trial", "systematic review",
            "meta analysis", "practice guideline", "guideline",
        ],
        query_focus=["intervention", "comparator", "outcome"],
        ranking_boosts={"rct": 10, "systematic_review": 12, "meta_analysis": 12, "guideline": 8},
        common_penalties=["case_report_when_higher_evidence_exists",
                           "diagnostic_only_for_therapy"],
    ),
    "diagnosis": StudyTypePathway(
        type_of_question="diagnosis",
        preferred_study_types=["diagnostic_accuracy_study", "systematic_review", "meta_analysis"],
        acceptable_study_types=["cohort_study", "case_control", "cross_sectional"],
        pubmed_pub_types=[
            "Validation Study", "Comparative Study", "Systematic Review",
            "Meta-Analysis", "Practice Guideline",
        ],
        europe_pmc_pub_types=[
            "diagnostic accuracy", "systematic review", "meta analysis",
            "validation study", "practice guideline",
        ],
        query_focus=["index test", "reference standard", "target condition"],
        ranking_boosts={"diagnostic_accuracy_study": 12, "systematic_review": 10, "meta_analysis": 10},
        common_penalties=["therapy_only_for_diagnosis", "animal_only_for_clinical_dx"],
    ),
    "prognosis": StudyTypePathway(
        type_of_question="prognosis",
        preferred_study_types=["cohort_study", "systematic_review", "meta_analysis"],
        acceptable_study_types=["case_control", "rct", "narrative_review"],
        pubmed_pub_types=["Observational Study", "Cohort Studies",
                          "Systematic Review", "Meta-Analysis"],
        europe_pmc_pub_types=["cohort studies", "observational study",
                              "systematic review", "meta analysis"],
        query_focus=["population", "outcome", "time horizon"],
        ranking_boosts={"cohort_study": 12, "systematic_review": 8, "meta_analysis": 8},
        common_penalties=["acute_intervention_paper_for_long_term_prognosis"],
    ),
    "etiology_harm": StudyTypePathway(
        type_of_question="etiology_harm",
        preferred_study_types=["cohort_study", "case_control",
                                "systematic_review", "meta_analysis"],
        acceptable_study_types=["cross_sectional", "rct"],
        pubmed_pub_types=["Observational Study", "Case-Control Studies",
                          "Cohort Studies", "Systematic Review", "Meta-Analysis"],
        europe_pmc_pub_types=["observational study", "case-control studies",
                              "cohort studies", "systematic review",
                              "meta analysis"],
        query_focus=["exposure", "harm/outcome", "population"],
        ranking_boosts={"cohort_study": 10, "case_control": 10,
                         "systematic_review": 8, "meta_analysis": 8},
        common_penalties=["case_report_for_known_harm"],
    ),
    "prevention": StudyTypePathway(
        type_of_question="prevention",
        preferred_study_types=["rct", "systematic_review", "meta_analysis", "guideline"],
        acceptable_study_types=["cohort_study"],
        pubmed_pub_types=["Randomized Controlled Trial", "Systematic Review",
                          "Meta-Analysis", "Practice Guideline"],
        europe_pmc_pub_types=["randomized controlled trial", "systematic review",
                              "meta analysis", "practice guideline"],
        query_focus=["preventive intervention", "population at risk", "outcome"],
        ranking_boosts={"rct": 10, "systematic_review": 12, "meta_analysis": 12, "guideline": 10},
        common_penalties=["treatment_only_paper_for_prevention"],
    ),
    "clinical_exam": StudyTypePathway(
        type_of_question="clinical_exam",
        preferred_study_types=["diagnostic_accuracy_study", "systematic_review", "meta_analysis"],
        acceptable_study_types=["cohort_study", "cross_sectional"],
        pubmed_pub_types=["Validation Study", "Systematic Review",
                          "Meta-Analysis", "Comparative Study"],
        europe_pmc_pub_types=["diagnostic accuracy", "validation study",
                              "systematic review", "meta analysis"],
        query_focus=["physical sign / maneuver", "target condition", "sensitivity/specificity"],
        ranking_boosts={"diagnostic_accuracy_study": 12, "systematic_review": 8},
        common_penalties=["therapy_only_for_clinical_exam"],
    ),
    "cost": StudyTypePathway(
        type_of_question="cost",
        preferred_study_types=["systematic_review", "meta_analysis", "cost_effectiveness_study"],
        acceptable_study_types=["cohort_study", "rct"],
        pubmed_pub_types=["Cost-Benefit Analysis", "Systematic Review",
                          "Meta-Analysis", "Comparative Study"],
        europe_pmc_pub_types=["cost-benefit analysis", "systematic review",
                              "meta analysis"],
        query_focus=["intervention", "comparator", "cost-effectiveness / QALY"],
        ranking_boosts={"systematic_review": 10, "meta_analysis": 10},
        common_penalties=["clinical_only_paper_for_cost"],
    ),
    "pharmacology": StudyTypePathway(
        type_of_question="pharmacology",
        preferred_study_types=["systematic_review", "meta_analysis", "rct",
                                "pharmacokinetic_study"],
        acceptable_study_types=["narrative_review", "guideline"],
        pubmed_pub_types=["Randomized Controlled Trial", "Systematic Review",
                          "Meta-Analysis", "Review", "Practice Guideline"],
        europe_pmc_pub_types=["randomized controlled trial",
                              "systematic review", "meta analysis", "review",
                              "practice guideline"],
        query_focus=["drug", "mechanism / kinetics", "indication / patient population"],
        ranking_boosts={"rct": 10, "systematic_review": 10, "meta_analysis": 10},
        common_penalties=["imaging_only_for_pharmacology",
                           "animal_only_for_clinical_pharm"],
    ),
    "dosing_protocol": StudyTypePathway(
        type_of_question="dosing_protocol",
        preferred_study_types=["rct", "systematic_review", "meta_analysis", "guideline"],
        acceptable_study_types=["cohort_study", "case_series", "narrative_review"],
        pubmed_pub_types=["Randomized Controlled Trial", "Systematic Review",
                          "Meta-Analysis", "Practice Guideline", "Clinical Trial",
                          "Observational Study"],
        europe_pmc_pub_types=["randomized controlled trial", "systematic review",
                              "meta analysis", "practice guideline",
                              "observational study"],
        query_focus=["drug", "dose / infusion / bolus / route",
                     "population / indication"],
        ranking_boosts={"rct": 10, "systematic_review": 12, "meta_analysis": 12,
                         "guideline": 12, "cohort_study": 6, "case_series": 4},
        common_penalties=["imaging_only_for_pharmacology",
                           "case_report_when_higher_evidence_exists",
                           "diagnostic_only_for_dosing"],
        subtype_label="dosing or protocol question",
    ),
    "guideline": StudyTypePathway(
        type_of_question="guideline",
        preferred_study_types=["guideline", "systematic_review", "meta_analysis"],
        acceptable_study_types=["narrative_review", "rct"],
        pubmed_pub_types=["Practice Guideline", "Guideline",
                          "Consensus Development Conference",
                          "Systematic Review", "Meta-Analysis"],
        europe_pmc_pub_types=["practice guideline", "guideline",
                              "consensus development conference",
                              "systematic review", "meta analysis"],
        query_focus=["society / organization", "recommendation", "indication"],
        ranking_boosts={"guideline": 15, "systematic_review": 8, "meta_analysis": 8},
        common_penalties=["non_guideline_for_guideline"],
    ),
    "epidemiology": StudyTypePathway(
        type_of_question="epidemiology",
        preferred_study_types=["cross_sectional", "cohort_study",
                                "systematic_review", "meta_analysis"],
        acceptable_study_types=["case_control"],
        pubmed_pub_types=["Cross-Sectional Studies", "Cohort Studies",
                          "Systematic Review", "Meta-Analysis"],
        europe_pmc_pub_types=["cross-sectional studies", "cohort studies",
                              "systematic review", "meta analysis"],
        query_focus=["prevalence / incidence", "population", "geography"],
        ranking_boosts={"cross_sectional": 10, "cohort_study": 10,
                         "systematic_review": 8, "meta_analysis": 8},
        common_penalties=[],
    ),
    "background": StudyTypePathway(
        type_of_question="background",
        preferred_study_types=["narrative_review", "guideline",
                                "systematic_review"],
        acceptable_study_types=["meta_analysis", "rct", "cohort_study"],
        pubmed_pub_types=["Review", "Practice Guideline", "Guideline",
                          "Systematic Review"],
        europe_pmc_pub_types=["review", "practice guideline", "guideline",
                              "systematic review"],
        query_focus=["concept overview", "pathophysiology", "definition"],
        ranking_boosts={"narrative_review": 8, "guideline": 10,
                         "systematic_review": 6},
        common_penalties=["case_report_for_background"],
    ),
    "other": StudyTypePathway(
        type_of_question="other",
        preferred_study_types=["systematic_review", "meta_analysis",
                                "rct", "guideline"],
        acceptable_study_types=["cohort_study", "narrative_review"],
        pubmed_pub_types=["Systematic Review", "Review", "Meta-Analysis"],
        europe_pmc_pub_types=["systematic review", "review", "meta analysis"],
        query_focus=["topic"],
        ranking_boosts={"systematic_review": 8, "meta_analysis": 8},
        common_penalties=[],
    ),
}


class StudyTypeRouterService:
    def pathway(self, type_of_question: str) -> StudyTypePathway:
        key = (type_of_question or "other").lower().strip()
        return _PATHWAYS.get(key, _PATHWAYS["other"])

    def all_types(self) -> list[str]:
        return list(_PATHWAYS.keys())


study_type_router = StudyTypeRouterService()
