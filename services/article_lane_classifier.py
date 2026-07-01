"""Phase K3A — Shadow article lane classifier + cap-candidate detector.

Deterministic, closed-set, NO LLM. Consumes a ``ClinicalQuestionMap``
(K1), the article's metadata (title, abstract, journal, publication
year, MeSH terms, publication types, retraction / expression-of-concern
/ preprint flags), and the branch's intent + required anchors, and
emits:

  * ``lane_label``         (closed-set Literal)
  * ``lane_confidence``    (float 0..1)
  * ``branch_alignment``   (closed-set Literal: aligned/partial/misaligned)
  * ``matched_branch_ids`` (list of branch_id strings)
  * ``article_intent_fit`` (closed-set Literal: full/partial/none)
  * ``directness_label``   (closed-set Literal)
  * ``cap_candidates``     (list of ``CapCandidate``)
  * ``cap_reason_codes``   (list of closed-set codes)
  * ``cap_severity``       (closed-set Literal: low/medium/high)
  * ``cap_confidence``     (float 0..1)
  * ``cap_mode``           ("shadow")
  * ``would_cap``          (HARD False in K3A)
  * ``would_exclude``      (HARD False in K3A)
  * ``explanation``        (closed-set rule-id list — NOT free text)
  * ``evidence_role_candidate`` (closed-set Literal)
  * ``qa_warnings``        (closed-set warning sentinels)

Hard rules (mirror the K3A brief):
  * NEVER activates retrieval, caps, or exclusions in K3A. The
    ``classify_articles`` entry-point hard-rejects ``mode="active"``.
  * NEVER calls an LLM. NEVER touches the network. Pure / sync.
  * NEVER stores raw prompts, completions, PHI, evidence text,
    uploaded content, emails, credentials, or secrets in diagnostics.
  * Closed-set ``lane_label`` + ``cap_reason_codes`` enums only.
  * Caps NEVER modify ``search_session_articles``; the diagnostics
    live alongside under ``session_diagnostics_json.article_lane_diagnostics``.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


LaneClassifierMode = Literal["disabled", "shadow", "active"]


# ---------------------------------------------------------------------
# Closed-set Literal enums.
# ---------------------------------------------------------------------


LaneLabel = Literal[
    # General
    "direct_answer", "related_answer", "indirect_answer", "background_only",
    "wrong_framework_lane", "wrong_disease", "wrong_population",
    "wrong_intervention", "wrong_exposure", "wrong_test",
    "wrong_procedure", "wrong_device", "wrong_outcome", "wrong_setting",
    "animal_only", "in_vitro_basic_science", "preclinical_basic_science",
    "editorial_commentary",
    "retracted_or_expression_of_concern",
    "unsafe_or_ineligible",
    # Therapy
    "therapy_guideline", "therapy_systematic_review",
    "therapy_meta_analysis", "therapy_rct",
    "therapy_comparative_effectiveness", "therapy_observational",
    "therapy_case_series",
    # Diagnostic
    "diagnostic_accuracy_systematic_review", "diagnostic_accuracy_primary",
    "diagnostic_accuracy_guideline",
    "diagnostic_workup_guideline", "diagnostic_workup_review",
    "differential_diagnosis_review",
    # Prognosis
    "prognosis_systematic_review", "prognosis_cohort",
    "prognosis_survival_study",
    "prognostic_factor_study", "prognostic_factor_systematic_review",
    # Prediction
    "prediction_model_derivation", "prediction_model_internal_validation",
    "prediction_model_external_validation",
    "prediction_model_impact_study", "prediction_model_systematic_review",
    # Risk/etiology/harm
    "risk_factor_cohort", "risk_factor_case_control",
    "risk_factor_registry", "risk_factor_systematic_review",
    "adverse_effect_pharmacovigilance",
    # Complication
    "complication_systematic_review", "complication_cohort",
    "complication_registry", "complication_case_series",
    "complication_case_report",
    "pharmacovigilance_signal",
    # Epidemiology
    "epidemiology_prevalence", "epidemiology_incidence",
    "epidemiology_surveillance", "epidemiology_systematic_review",
    # Pathophysiology
    "pathophysiology_review", "mechanism_review", "human_translational",
    "biomarker_mechanism", "animal_model", "in_vitro_mechanism",
    # Guideline
    "guideline_pubmed_indexed", "consensus_statement",
    "society_guideline", "guideline_review",
]

DirectnessLabel = Literal[
    "direct", "related", "indirect", "aspect_specific",
    "background_only", "wrong_lane", "wrong_anchor", "unsafe",
]

CapReasonCode = Literal[
    "missing_primary_anchor",
    "wrong_disease", "wrong_population", "wrong_intervention",
    "wrong_exposure", "wrong_index_test", "wrong_target_condition",
    "wrong_procedure", "wrong_device", "wrong_complication",
    "wrong_outcome", "wrong_setting",
    "therapy_article_for_prognosis_question",
    "prognosis_article_for_therapy_question",
    "pathophysiology_basic_science_for_clinical_management_question",
    "animal_only_for_clinical_question",
    "in_vitro_only_for_clinical_question",
    "case_report_when_higher_evidence_available",
    "editorial_commentary_when_stronger_evidence_available",
    "retracted",
    "expression_of_concern",
    "preprint_for_clinical_recommendation",
    "generic_modifier_only_match",
    "aspect_specific_when_general_overview_requested",
    "management_only_without_procedure_context",
    "related_procedure_not_target_procedure",
]

CapSeverity = Literal["low", "medium", "high"]

BranchAlignment = Literal["aligned", "partial", "misaligned", "unknown"]

ArticleIntentFit = Literal["full", "partial", "none"]

EvidenceRoleCandidate = Literal[
    "primary_evidence", "supporting_evidence", "background_context",
    "mechanism_context", "comparator_context", "harm_context",
    "guideline_context", "case_example", "to_exclude_candidate",
    "unknown",
]


# ---------------------------------------------------------------------
# Pydantic schemas.
# ---------------------------------------------------------------------


class CapCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason_code: CapReasonCode
    severity: CapSeverity
    explanation_id: str  # closed-set rule id, NOT free text


class ArticleLaneDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    article_id: str
    search_session_article_id: Optional[str] = None
    lane_label: LaneLabel
    lane_confidence: float = Field(ge=0.0, le=1.0)
    branch_alignment: BranchAlignment
    matched_branch_ids: List[str] = Field(default_factory=list)
    article_intent_fit: ArticleIntentFit
    directness_label: DirectnessLabel
    cap_candidates: List[CapCandidate] = Field(default_factory=list)
    cap_reason_codes: List[CapReasonCode] = Field(default_factory=list)
    cap_severity: CapSeverity = "low"
    cap_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cap_mode: Literal["shadow", "disabled"] = "shadow"
    would_cap: Literal[False] = False
    would_exclude: Literal[False] = False
    explanation: List[str] = Field(default_factory=list)
    evidence_role_candidate: EvidenceRoleCandidate = "unknown"
    qa_warnings: List[str] = Field(default_factory=list)


class ArticleLaneClassificationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["K3A.1.0"] = "K3A.1.0"
    lane_label_mode: Literal["disabled", "shadow"] = "shadow"
    cap_detection_mode: Literal["disabled", "shadow"] = "shadow"
    diagnostics: List[ArticleLaneDiagnostic] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Closed-set lexicon used by the classifier.
# ---------------------------------------------------------------------


# Publication-type → category. ``publication_types`` on articles is a
# free-form list; we normalize to closed-set categories.
_PUB_TYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "systematic_review": re.compile(
        r"\bsystematic[\s-]?review\b", re.IGNORECASE,
    ),
    "meta_analysis": re.compile(r"\bmeta[\s-]?analysis\b", re.IGNORECASE),
    "rct": re.compile(
        r"\brandomi[sz]ed\s+controlled\s+trial\b|\brct\b", re.IGNORECASE,
    ),
    "controlled_trial": re.compile(
        r"\bcontrolled\s+clinical\s+trial\b", re.IGNORECASE,
    ),
    "guideline": re.compile(
        r"\b(?:practice\s+)?guideline\b|\bguidance\b", re.IGNORECASE,
    ),
    "consensus_statement": re.compile(
        r"\bconsensus\s+(?:statement|development)\b", re.IGNORECASE,
    ),
    "cohort": re.compile(r"\bcohort\s+study\b|\bcohort\b", re.IGNORECASE),
    "case_control": re.compile(r"\bcase[\s-]control\s+study\b", re.IGNORECASE),
    "case_series": re.compile(r"\bcase[\s-]series\b|\bcase\s+report\b", re.IGNORECASE),
    "case_report": re.compile(r"\bcase\s+report\b", re.IGNORECASE),
    "validation_study": re.compile(r"\bvalidation\s+study\b", re.IGNORECASE),
    "editorial": re.compile(r"\beditorial\b|\bletter\b|\bcomment\b", re.IGNORECASE),
    "review": re.compile(r"\breview\b", re.IGNORECASE),
    "diagnostic_accuracy": re.compile(
        r"\bdiagnostic\s+accuracy\b|\bsensitivity\s+and\s+specificity\b",
        re.IGNORECASE,
    ),
    "registry": re.compile(r"\bregistry\b", re.IGNORECASE),
    "narrative_review": re.compile(r"\bnarrative\s+review\b", re.IGNORECASE),
}


_TITLE_ABSTRACT_SIGNALS: dict[str, re.Pattern[str]] = {
    "diagnostic_accuracy_terms": re.compile(
        r"\b(?:sensitivity|specificity|likelihood\s+ratio|"
        r"diagnostic\s+accuracy|ROC\s+curve|AUC|"
        r"sensitivity\s+and\s+specificity)\b",
        re.IGNORECASE,
    ),
    "differential_diagnosis_terms": re.compile(
        r"\b(?:differential\s+diagnos[ei]s|red\s+flags?|"
        r"don[-'’]?t\s+miss|mimics|must\s+rule\s+out)\b",
        re.IGNORECASE,
    ),
    "prognosis_terms": re.compile(
        r"\b(?:prognosis|natural\s+history|survival|mortality|"
        r"follow[-\s]?up|recurrence|long[-\s]?term\s+outcome)\b",
        re.IGNORECASE,
    ),
    "prognostic_factor_terms": re.compile(
        r"\b(?:prognostic\s+factor|prognostic\s+marker|"
        r"adjusted\s+hazard\s+ratio|adjusted\s+HR|"
        r"prognostic\s+role)\b",
        re.IGNORECASE,
    ),
    "prediction_model_terms": re.compile(
        r"\b(?:prediction\s+model|risk\s+score|risk\s+prediction|"
        r"clinical\s+score|calibration|discrimination|"
        r"c[-\s]?statistic|AUC|external\s+validation|"
        r"internal\s+validation|impact\s+study)\b",
        re.IGNORECASE,
    ),
    "risk_etiology_terms": re.compile(
        r"\b(?:risk\s+factor|exposure|etiology|association|"
        r"causation|cohort|case[-\s]control)\b",
        re.IGNORECASE,
    ),
    "complication_terms": re.compile(
        r"\b(?:complication|adverse\s+event|infection|"
        r"bleeding|dislodg(?:e?ment|ed)|migration|"
        r"thrombosis|post[-\s]operative)\b",
        re.IGNORECASE,
    ),
    "epidemiology_terms": re.compile(
        r"\b(?:prevalence|incidence|epidemiology|"
        r"population[-\s]based|surveillance|burden\s+of\s+disease)\b",
        re.IGNORECASE,
    ),
    "pathophysiology_terms": re.compile(
        r"\b(?:pathophysiology|mechanism|pathogenesis|"
        r"physiopathology|molecular\s+basis)\b",
        re.IGNORECASE,
    ),
    "animal_model_terms": re.compile(
        r"\b(?:mouse|mice|rat|rats|murine|rodent|"
        r"canine|porcine|animal\s+model|in\s+vivo\s+mouse)\b",
        re.IGNORECASE,
    ),
    "in_vitro_terms": re.compile(
        r"\b(?:in\s+vitro|cell\s+line|HEK293|HeLa|cultured\s+cells|"
        r"cell\s+culture|knockout\s+cell)\b",
        re.IGNORECASE,
    ),
    "pharmacovigilance_terms": re.compile(
        r"\b(?:pharmacovigilance|spontaneous\s+report|FAERS|"
        r"VAERS|adverse\s+drug\s+reaction\s+report)\b",
        re.IGNORECASE,
    ),
    "external_validation_terms": re.compile(
        r"\bexternal\s+validation\b", re.IGNORECASE,
    ),
    "internal_validation_terms": re.compile(
        r"\binternal\s+validation\b", re.IGNORECASE,
    ),
    "derivation_terms": re.compile(
        r"\b(?:derivation\s+cohort|model\s+derivation|"
        r"developed\s+a\s+(?:new\s+)?(?:prediction|risk|clinical)\s+"
        r"(?:model|score)|"
        r"we\s+developed|developed\s+(?:and|the))\b",
        re.IGNORECASE,
    ),
    "guideline_terms": re.compile(
        r"\b(?:guideline|recommendation|consensus|society\s+statement)\b",
        re.IGNORECASE,
    ),
    "workup_terms": re.compile(
        r"\b(?:workup|work[-\s]?up|evaluation|approach\s+to|algorithm)\b",
        re.IGNORECASE,
    ),
}


_FORBIDDEN_DIAG_SUBSTRINGS: tuple[str, ...] = (
    "SYSTEM_PROMPT", "Return JSON only", "You are not",
    "Confidence guidance", "JWT_ACCESS_SECRET", "JWT_REFRESH_SECRET",
    # Phase K3A.1 — additional universal sentinels for prompt residue
    # (``system:`` / ``assistant:`` role headers), API-key prefixes
    # (``sk-`` for OpenAI-style keys), and bare email addresses
    # (``@`` sentinel covers ``@gmail.com`` / ``@yahoo.com`` etc).
    # The diagnostic blob NEVER legitimately contains any of these in
    # K3A: rule IDs are closed-set ASCII identifiers (`therapy:rct`,
    # `safety:retracted`), so a hit is always a leak.
    "system:", "assistant:", "sk-", "@gmail.com", "@yahoo.com",
    "@outlook.com", "@hotmail.com", "@protonmail.com",
)


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def _norm_list(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    return [str(x).strip() for x in xs if isinstance(x, str) and str(x).strip()]


def _detect_pub_categories(article: dict[str, Any]) -> set[str]:
    pubs = _norm_list(article.get("publication_types_json")) + _norm_list(
        article.get("publication_types")
    )
    blob = " ".join(pubs)
    out: set[str] = set()
    for cat, patt in _PUB_TYPE_PATTERNS.items():
        if patt.search(blob):
            out.add(cat)
    # Article-type classification (if present from upstream).
    cls = str(article.get("article_type_classification") or "").lower()
    if "rct" in cls:
        out.add("rct")
    if "systematic" in cls:
        out.add("systematic_review")
    if "meta-analysis" in cls or "meta_analysis" in cls:
        out.add("meta_analysis")
    if "guideline" in cls:
        out.add("guideline")
    if "cohort" in cls:
        out.add("cohort")
    if "case-control" in cls:
        out.add("case_control")
    if "review" in cls:
        out.add("review")
    if article.get("is_guideline"):
        out.add("guideline")
    return out


def _detect_signals(article: dict[str, Any]) -> set[str]:
    """Return closed-set signal labels triggered by the article's title +
    abstract. NEVER stores the raw text.
    """
    blob = " ".join([
        str(article.get("title") or ""),
        str(article.get("abstract") or ""),
    ])
    out: set[str] = set()
    for label, patt in _TITLE_ABSTRACT_SIGNALS.items():
        if patt.search(blob):
            out.add(label)
    return out


def _anchor_matches(
    article: dict[str, Any],
    anchors: List[dict[str, Any]],
) -> tuple[set[str], List[str]]:
    """Return ``(matched_canonical_concepts_lower, unmatched_canonical_concepts)``.

    Matching is whole-token + case-insensitive on title + abstract +
    MeSH terms. Synonyms count.
    """
    blob = " ".join([
        str(article.get("title") or "").lower(),
        str(article.get("abstract") or "").lower(),
        " ".join(_norm_list(article.get("mesh_terms_json"))).lower(),
        " ".join(_norm_list(article.get("mesh_terms"))).lower(),
    ])
    matched: set[str] = set()
    unmatched: list[str] = []
    for a in anchors:
        if not isinstance(a, dict):
            continue
        canon = str(a.get("canonical_concept") or "").strip().lower()
        original = str(a.get("original_phrase") or "").strip().lower()
        syns = [
            str(s).strip().lower() for s in (a.get("synonyms") or [])
            if isinstance(s, str)
        ]
        candidates = [c for c in [canon, original, *syns] if c]
        if not candidates:
            continue
        hit = any(c in blob for c in candidates)
        if hit:
            matched.add(canon)
        else:
            unmatched.append(canon)
    return matched, unmatched


# ---------------------------------------------------------------------
# Lane classification — closed-set intent table.
# ---------------------------------------------------------------------


def _classify_lane(
    *,
    intent: str,
    pub_cats: set[str],
    signals: set[str],
    is_guideline: bool,
    is_preprint: bool,
    is_retracted: bool,
    has_eoc: bool,
    matched_required: set[str],
    unmatched_required: List[str],
) -> tuple[LaneLabel, float, List[str]]:
    """Return ``(lane_label, lane_confidence, explanation_rule_ids)``.

    Closed-set rule cascade. Higher-priority rules short-circuit. Every
    return path also records the rule id(s) that fired.
    """
    rules: list[str] = []
    # Safety overrides (always apply before intent-specific routing).
    if is_retracted or has_eoc:
        rules.append("safety:retracted_or_eoc")
        return "retracted_or_expression_of_concern", 0.99, rules
    if "editorial" in pub_cats:
        rules.append("safety:editorial_commentary")
        return "editorial_commentary", 0.85, rules
    if "animal_model_terms" in signals and not (
        "human" in signals or "human_translational" in signals
    ) and "rct" not in pub_cats and "cohort" not in pub_cats and (
        intent != "pathophysiology_mechanism"
    ):
        rules.append("safety:animal_only")
        return "animal_only", 0.7, rules
    if (
        "in_vitro_terms" in signals
        and "rct" not in pub_cats and "cohort" not in pub_cats
        and intent != "pathophysiology_mechanism"
    ):
        rules.append("safety:in_vitro_only")
        return "in_vitro_basic_science", 0.7, rules

    # Therapy
    if intent in {"therapy", "therapy_safety", "drug_safety"}:
        if is_guideline or "guideline" in pub_cats or "consensus_statement" in pub_cats:
            rules.append("therapy:guideline")
            return "therapy_guideline", 0.95, rules
        if "systematic_review" in pub_cats:
            rules.append("therapy:sr")
            return "therapy_systematic_review", 0.9, rules
        if "meta_analysis" in pub_cats:
            rules.append("therapy:ma")
            return "therapy_meta_analysis", 0.9, rules
        if "rct" in pub_cats or "controlled_trial" in pub_cats:
            rules.append("therapy:rct")
            return "therapy_rct", 0.9, rules
        if "cohort" in pub_cats:
            rules.append("therapy:observational")
            return "therapy_observational", 0.65, rules
        if "case_series" in pub_cats or "case_report" in pub_cats:
            rules.append("therapy:case_series")
            return "therapy_case_series", 0.55, rules

    # Cross-intent therapy detection — when the article is clearly a
    # therapy-shaped trial (RCT / SR / MA / guideline) but the BRANCH
    # intent is NOT therapy, we still label it with the therapy lane
    # so downstream cap detection can fire the closed-set
    # ``therapy_article_for_prognosis_question`` /
    # ``pathophysiology_basic_science_for_clinical_management_question``
    # cap. Without this fallthrough, mismatched-intent therapy articles
    # would land in ``related_answer`` and the cap detector would
    # never see a ``therapy_*`` lane to compare against the intent.
    cross_intent_therapy = (
        intent not in {"therapy", "therapy_safety", "drug_safety"}
        and (
            "rct" in pub_cats or "controlled_trial" in pub_cats
            or ("systematic_review" in pub_cats and intent not in {
                "diagnosis_accuracy", "diagnostic_workup",
                "differential_diagnosis", "prognosis_natural_history",
                "prognostic_factor", "prediction_model",
                "risk_etiology_harm", "complication_risk",
                "epidemiology_prevalence_incidence",
            })
        )
    )

    # Diagnostic accuracy
    if intent == "diagnosis_accuracy":
        if "systematic_review" in pub_cats and (
            "diagnostic_accuracy" in pub_cats
            or "diagnostic_accuracy_terms" in signals
        ):
            rules.append("dxa:sr")
            return "diagnostic_accuracy_systematic_review", 0.92, rules
        if is_guideline or "guideline" in pub_cats:
            rules.append("dxa:guideline")
            return "diagnostic_accuracy_guideline", 0.88, rules
        if (
            "diagnostic_accuracy" in pub_cats
            or "diagnostic_accuracy_terms" in signals
        ):
            rules.append("dxa:primary")
            return "diagnostic_accuracy_primary", 0.85, rules

    # Diagnostic workup
    if intent == "diagnostic_workup":
        if is_guideline or "guideline" in pub_cats:
            rules.append("workup:guideline")
            return "diagnostic_workup_guideline", 0.9, rules
        if (
            "review" in pub_cats or "narrative_review" in pub_cats
            or "workup_terms" in signals
        ):
            rules.append("workup:review")
            return "diagnostic_workup_review", 0.78, rules

    # Differential diagnosis
    if intent == "differential_diagnosis":
        if "differential_diagnosis_terms" in signals:
            rules.append("ddx:differential_terms")
            return "differential_diagnosis_review", 0.85, rules
        if "review" in pub_cats:
            rules.append("ddx:review")
            return "differential_diagnosis_review", 0.7, rules

    # Prognosis
    if intent == "prognosis_natural_history":
        if "systematic_review" in pub_cats:
            rules.append("prog:sr")
            return "prognosis_systematic_review", 0.9, rules
        if "cohort" in pub_cats or (
            "prognosis_terms" in signals and "rct" not in pub_cats
        ):
            rules.append("prog:cohort")
            return "prognosis_cohort", 0.82, rules

    # Prognostic factor
    if intent == "prognostic_factor":
        if "systematic_review" in pub_cats:
            rules.append("pf:sr")
            return "prognostic_factor_systematic_review", 0.9, rules
        if "prognostic_factor_terms" in signals or "cohort" in pub_cats:
            rules.append("pf:study")
            return "prognostic_factor_study", 0.82, rules

    # Prediction model
    if intent == "prediction_model":
        if "systematic_review" in pub_cats:
            rules.append("pm:sr")
            return "prediction_model_systematic_review", 0.9, rules
        # External validation outranks internal/derivation ONLY when
        # the derivation signal is NOT also present — abstracts that
        # describe a derivation cohort with "External validation
        # pending" should be labelled as derivation, not external.
        if "external_validation_terms" in signals and (
            "derivation_terms" not in signals
        ):
            rules.append("pm:ext_val")
            return "prediction_model_external_validation", 0.92, rules
        if "derivation_terms" in signals:
            rules.append("pm:derivation")
            return "prediction_model_derivation", 0.85, rules
        if "external_validation_terms" in signals:
            rules.append("pm:ext_val")
            return "prediction_model_external_validation", 0.85, rules
        if "internal_validation_terms" in signals or "validation_study" in pub_cats:
            rules.append("pm:int_val")
            return "prediction_model_internal_validation", 0.8, rules
        if "prediction_model_terms" in signals:
            rules.append("pm:generic_pm")
            return "prediction_model_derivation", 0.7, rules

    # Risk / etiology / harm
    if intent == "risk_etiology_harm":
        if "systematic_review" in pub_cats:
            rules.append("risk:sr")
            return "risk_factor_systematic_review", 0.9, rules
        if "cohort" in pub_cats:
            rules.append("risk:cohort")
            return "risk_factor_cohort", 0.85, rules
        if "case_control" in pub_cats:
            rules.append("risk:case_control")
            return "risk_factor_case_control", 0.8, rules
        if "registry" in pub_cats:
            rules.append("risk:registry")
            return "risk_factor_registry", 0.78, rules

    # Complication
    if intent in {
        "complication_risk", "disease_complication",
        "drug_complication", "procedure_complication",
        "device_complication",
    }:
        if "systematic_review" in pub_cats:
            rules.append("complication:sr")
            return "complication_systematic_review", 0.9, rules
        if "registry" in pub_cats:
            rules.append("complication:registry")
            return "complication_registry", 0.88, rules
        if "cohort" in pub_cats:
            rules.append("complication:cohort")
            return "complication_cohort", 0.8, rules
        if "pharmacovigilance_terms" in signals:
            rules.append("complication:pharmacovigilance")
            return "pharmacovigilance_signal", 0.82, rules
        if "case_series" in pub_cats:
            rules.append("complication:case_series")
            return "complication_case_series", 0.65, rules
        if "case_report" in pub_cats:
            rules.append("complication:case_report")
            return "complication_case_report", 0.55, rules

    # Epidemiology
    if intent == "epidemiology_prevalence_incidence":
        if "systematic_review" in pub_cats:
            rules.append("epi:sr")
            return "epidemiology_systematic_review", 0.9, rules
        if "epidemiology_terms" in signals:
            # Prevalence vs incidence sub-label — derived from
            # presence of "prevalence" in pub-type strings (rare) or
            # the article's title/abstract. Defaults to prevalence.
            if any("prevalence" in p for p in pub_cats):
                rules.append("epi:prevalence")
                return "epidemiology_prevalence", 0.85, rules
            rules.append("epi:prevalence")
            return "epidemiology_prevalence", 0.82, rules

    # Pathophysiology / mechanism
    if intent == "pathophysiology_mechanism":
        if "animal_model_terms" in signals:
            rules.append("mech:animal")
            return "animal_model", 0.85, rules
        if "in_vitro_terms" in signals:
            rules.append("mech:in_vitro")
            return "in_vitro_mechanism", 0.85, rules
        if "pathophysiology_terms" in signals or "review" in pub_cats:
            rules.append("mech:review")
            return "pathophysiology_review", 0.82, rules
        # Mechanism asked, no mechanism signal — keep the fallback at
        # ``mechanism_review`` so the cap detector can fire
        # ``pathophysiology_basic_science_for_clinical_management_question``
        # via the cross-intent fallthrough below.
        # (Drop into the cross-intent block by NOT returning here.)

    # Monitoring / stabilization / guideline
    if intent in {
        "monitoring_followup_surveillance",
        "stabilization_emergency_management",
        "guideline_recommendation",
    }:
        if is_guideline or "guideline" in pub_cats:
            rules.append("guideline:society_or_indexed")
            return "society_guideline", 0.92, rules
        if "consensus_statement" in pub_cats:
            rules.append("guideline:consensus")
            return "consensus_statement", 0.88, rules
        if "review" in pub_cats:
            rules.append("guideline:review")
            return "guideline_review", 0.75, rules

    # Cross-intent therapy fallthrough — when an obvious therapy-shaped
    # article appears under a non-therapy intent, label with the
    # therapy lane so the cap detector can fire the closed-set
    # cross-intent cap (e.g. ``therapy_article_for_prognosis_question``,
    # ``pathophysiology_basic_science_for_clinical_management_question``).
    if cross_intent_therapy:
        if is_guideline or "guideline" in pub_cats:
            rules.append("cross:therapy_guideline")
            return "therapy_guideline", 0.7, rules
        if "systematic_review" in pub_cats:
            rules.append("cross:therapy_sr")
            return "therapy_systematic_review", 0.7, rules
        if "rct" in pub_cats:
            rules.append("cross:therapy_rct")
            return "therapy_rct", 0.7, rules

    # Pathophysiology fallback — only reached when intent =
    # pathophysiology_mechanism and no mechanism signal fired.
    if intent == "pathophysiology_mechanism":
        rules.append("mech:default_review")
        return "mechanism_review", 0.6, rules

    # Generic / background
    if intent == "background_overview":
        rules.append("background:fallback")
        return "background_only", 0.6, rules

    # Fallback when no rule fires — be honest about it.
    if matched_required and unmatched_required:
        rules.append("fallback:partial_anchor_match")
        return "related_answer", 0.5, rules
    if matched_required:
        rules.append("fallback:full_anchor_match")
        return "related_answer", 0.55, rules
    rules.append("fallback:no_anchor_match")
    return "indirect_answer", 0.4, rules


# ---------------------------------------------------------------------
# Cap-candidate detection.
# ---------------------------------------------------------------------


def _detect_cap_candidates(
    *,
    intent: str,
    lane_label: LaneLabel,
    pub_cats: set[str],
    signals: set[str],
    is_preprint: bool,
    is_retracted: bool,
    has_eoc: bool,
    matched_required: set[str],
    unmatched_required: List[str],
    has_higher_evidence_available: bool,
) -> List[CapCandidate]:
    candidates: List[CapCandidate] = []

    # Safety
    if is_retracted:
        candidates.append(CapCandidate(
            reason_code="retracted", severity="high",
            explanation_id="safety:retracted",
        ))
    if has_eoc:
        candidates.append(CapCandidate(
            reason_code="expression_of_concern", severity="high",
            explanation_id="safety:expression_of_concern",
        ))
    if is_preprint and intent in {
        "therapy", "therapy_safety", "drug_safety",
        "guideline_recommendation",
        "stabilization_emergency_management",
    }:
        candidates.append(CapCandidate(
            reason_code="preprint_for_clinical_recommendation",
            severity="medium",
            explanation_id="safety:preprint_for_clinical_recommendation",
        ))

    # Anchor coverage
    if unmatched_required and not matched_required:
        candidates.append(CapCandidate(
            reason_code="missing_primary_anchor", severity="high",
            explanation_id="anchor:none_matched",
        ))

    # Intent × lane mismatches.
    therapy_lanes = {
        "therapy_guideline", "therapy_systematic_review",
        "therapy_meta_analysis", "therapy_rct",
        "therapy_comparative_effectiveness", "therapy_observational",
        "therapy_case_series",
    }
    prognosis_lanes = {
        "prognosis_systematic_review", "prognosis_cohort",
        "prognosis_survival_study",
    }
    if intent == "prognosis_natural_history" and lane_label in therapy_lanes:
        candidates.append(CapCandidate(
            reason_code="therapy_article_for_prognosis_question",
            severity="medium",
            explanation_id="intent:therapy_for_prognosis",
        ))
    if intent in {"therapy", "therapy_safety"} and lane_label in prognosis_lanes:
        candidates.append(CapCandidate(
            reason_code="prognosis_article_for_therapy_question",
            severity="medium",
            explanation_id="intent:prognosis_for_therapy",
        ))
    if intent in {
        "therapy", "therapy_safety", "diagnosis_accuracy",
        "diagnostic_workup", "differential_diagnosis",
        "prognosis_natural_history", "prognostic_factor",
        "prediction_model", "risk_etiology_harm", "complication_risk",
        "device_complication", "procedure_complication",
        "drug_complication", "monitoring_followup_surveillance",
        "stabilization_emergency_management", "guideline_recommendation",
    } and lane_label == "animal_model":
        candidates.append(CapCandidate(
            reason_code="animal_only_for_clinical_question", severity="high",
            explanation_id="lane:animal_only_clinical",
        ))
    if intent != "pathophysiology_mechanism" and lane_label == "in_vitro_mechanism":
        candidates.append(CapCandidate(
            reason_code="in_vitro_only_for_clinical_question", severity="high",
            explanation_id="lane:in_vitro_clinical",
        ))
    if (
        intent in {"therapy", "complication_risk", "device_complication",
                   "procedure_complication"}
        and lane_label in {
            "complication_case_report", "therapy_case_series",
            "complication_case_series",
        }
        and has_higher_evidence_available
    ):
        candidates.append(CapCandidate(
            reason_code="case_report_when_higher_evidence_available",
            severity="low",
            explanation_id="evidence:case_report_when_higher_available",
        ))
    if (
        lane_label == "editorial_commentary"
        and has_higher_evidence_available
    ):
        candidates.append(CapCandidate(
            reason_code="editorial_commentary_when_stronger_evidence_available",
            severity="low",
            explanation_id="evidence:editorial_when_stronger_available",
        ))
    if (
        intent == "pathophysiology_mechanism"
        and lane_label in therapy_lanes
    ):
        # Asked about mechanism, got a therapy paper — pathophys-not-management.
        candidates.append(CapCandidate(
            reason_code=(
                "pathophysiology_basic_science_for_clinical_management_question"
            ),
            severity="low",
            explanation_id="intent:pathophys_for_clinical_management",
        ))
    return candidates


# ---------------------------------------------------------------------
# Per-article classification.
# ---------------------------------------------------------------------


def _classify_article_for_branch(
    article: dict[str, Any],
    branch: dict[str, Any],
    *,
    has_higher_evidence_available_in_batch: bool,
) -> ArticleLaneDiagnostic:
    intent = str(branch.get("intent") or "")
    required_anchors = list(branch.get("required_concepts") or branch.get(
        "required_concepts_used"
    ) or [])
    pub_cats = _detect_pub_categories(article)
    signals = _detect_signals(article)
    is_retracted = bool(article.get("is_retracted"))
    has_eoc = bool(article.get("has_expression_of_concern"))
    is_preprint = bool(article.get("is_preprint"))
    is_guideline = bool(article.get("is_guideline"))

    matched_set, unmatched_list = _anchor_matches(article, required_anchors)
    matched_required = matched_set
    unmatched_required = unmatched_list

    lane_label, lane_conf, rule_ids = _classify_lane(
        intent=intent,
        pub_cats=pub_cats,
        signals=signals,
        is_guideline=is_guideline,
        is_preprint=is_preprint,
        is_retracted=is_retracted,
        has_eoc=has_eoc,
        matched_required=matched_required,
        unmatched_required=unmatched_required,
    )

    cap_candidates = _detect_cap_candidates(
        intent=intent,
        lane_label=lane_label,
        pub_cats=pub_cats,
        signals=signals,
        is_preprint=is_preprint,
        is_retracted=is_retracted,
        has_eoc=has_eoc,
        matched_required=matched_required,
        unmatched_required=unmatched_required,
        has_higher_evidence_available=has_higher_evidence_available_in_batch,
    )

    # Branch alignment.
    branch_alignment: BranchAlignment
    if not required_anchors:
        branch_alignment = "unknown"
    elif unmatched_required and not matched_required:
        branch_alignment = "misaligned"
    elif unmatched_required:
        branch_alignment = "partial"
    else:
        branch_alignment = "aligned"

    if branch_alignment == "aligned":
        article_intent_fit: ArticleIntentFit = "full"
    elif branch_alignment == "partial":
        article_intent_fit = "partial"
    else:
        article_intent_fit = "none"

    # Directness.
    if is_retracted or has_eoc:
        directness: DirectnessLabel = "unsafe"
    elif branch_alignment == "aligned" and not cap_candidates:
        directness = "direct"
    elif branch_alignment == "partial":
        directness = "related"
    elif branch_alignment == "misaligned":
        directness = "wrong_anchor"
    elif lane_label == "background_only":
        directness = "background_only"
    else:
        directness = "indirect"

    # Evidence-role candidate (closed-set).
    if lane_label in {
        "therapy_guideline", "diagnostic_accuracy_guideline",
        "society_guideline", "consensus_statement", "guideline_pubmed_indexed",
        "diagnostic_workup_guideline",
    }:
        evidence_role: EvidenceRoleCandidate = "guideline_context"
    elif lane_label in {
        "animal_model", "in_vitro_mechanism", "biomarker_mechanism",
        "pathophysiology_review", "mechanism_review",
    }:
        evidence_role = "mechanism_context"
    elif lane_label in {
        "complication_case_report", "therapy_case_series",
        "complication_case_series",
    }:
        evidence_role = "case_example"
    elif lane_label == "editorial_commentary":
        evidence_role = "background_context"
    elif lane_label == "retracted_or_expression_of_concern":
        evidence_role = "to_exclude_candidate"
    elif directness == "direct":
        evidence_role = "primary_evidence"
    elif directness == "related":
        evidence_role = "supporting_evidence"
    else:
        evidence_role = "unknown"

    cap_codes = [c.reason_code for c in cap_candidates]
    if cap_candidates:
        severity_rank = {"low": 1, "medium": 2, "high": 3}
        cap_severity: CapSeverity = max(
            (c.severity for c in cap_candidates),
            key=lambda s: severity_rank[s],
        )
    else:
        cap_severity = "low"
    cap_confidence = 0.85 if cap_candidates else 0.0

    qa_warnings: list[str] = []
    if not required_anchors:
        qa_warnings.append("no_required_anchors_on_branch")
    if lane_conf < 0.55:
        qa_warnings.append("low_lane_confidence")
    if not matched_required and required_anchors:
        qa_warnings.append("zero_required_anchor_matches")

    article_id = str(article.get("id") or article.get("article_id") or "")
    branch_id = str(branch.get("branch_id") or "primary_0")

    return ArticleLaneDiagnostic(
        article_id=article_id,
        search_session_article_id=str(article.get("search_session_article_id"))
        if article.get("search_session_article_id") else None,
        lane_label=lane_label,
        lane_confidence=lane_conf,
        branch_alignment=branch_alignment,
        matched_branch_ids=[branch_id] if branch_alignment in {
            "aligned", "partial",
        } else [],
        article_intent_fit=article_intent_fit,
        directness_label=directness,
        cap_candidates=cap_candidates,
        cap_reason_codes=cap_codes,
        cap_severity=cap_severity,
        cap_confidence=cap_confidence,
        cap_mode="shadow",
        would_cap=False,
        would_exclude=False,
        explanation=rule_ids,
        evidence_role_candidate=evidence_role,
        qa_warnings=qa_warnings,
    )


# ---------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------


def classify_articles(
    *,
    articles: List[dict[str, Any]],
    branches: List[dict[str, Any]],
    lane_label_mode: LaneClassifierMode = "shadow",
    cap_detection_mode: LaneClassifierMode = "shadow",
) -> ArticleLaneClassificationBatch:
    """Phase K3A — closed-set deterministic lane classification.

    Hard guards: ``mode="active"`` raises an ``AssertionError`` for
    either flag. K3A is SHADOW ONLY.
    """
    assert lane_label_mode != "active", (
        "ArticleLaneClassifier lane_label_mode='active' is rejected in K3A."
    )
    assert cap_detection_mode != "active", (
        "ArticleLaneClassifier cap_detection_mode='active' is rejected in K3A."
    )
    if lane_label_mode == "disabled" and cap_detection_mode == "disabled":
        return ArticleLaneClassificationBatch(
            lane_label_mode="disabled",
            cap_detection_mode="disabled",
            diagnostics=[],
        )
    if not articles or not branches:
        return ArticleLaneClassificationBatch(
            lane_label_mode="shadow" if lane_label_mode != "disabled" else "disabled",
            cap_detection_mode="shadow" if cap_detection_mode != "disabled" else "disabled",
            diagnostics=[],
        )

    # Determine whether the batch contains "higher evidence" — used to
    # gate the "case_report_when_higher_evidence_available" cap.
    has_higher_evidence_in_batch = False
    for a in articles:
        pc = _detect_pub_categories(a)
        if (
            "systematic_review" in pc
            or "meta_analysis" in pc
            or "rct" in pc
            or "guideline" in pc
        ):
            has_higher_evidence_in_batch = True
            break

    diagnostics: List[ArticleLaneDiagnostic] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        # Choose the best-matching branch — multi-intent routing:
        #   1. Compute candidate diagnostics against every branch.
        #   2. Pick the branch whose ``directness_label`` is most
        #      favourable (direct > related > indirect > misaligned).
        #   3. Tie-break by branch_priority.
        chosen_diag: Optional[ArticleLaneDiagnostic] = None
        chosen = branches[0]
        if len(branches) > 1:
            _RANK = {
                "direct": 4, "related": 3, "background_only": 2,
                "indirect": 2, "aspect_specific": 2,
                "wrong_anchor": 1, "wrong_lane": 1, "unsafe": 0,
            }
            # Intent-specific tie-breaker — when two branches both
            # produce ``direct`` matches, prefer the branch whose
            # intent has the strongest lexical alignment with the
            # article's title/abstract signals.
            _INTENT_SIGNAL_WEIGHTS: dict[str, set[str]] = {
                "differential_diagnosis": {
                    "differential_diagnosis_terms",
                },
                "diagnostic_workup": {"workup_terms"},
                "diagnosis_accuracy": {"diagnostic_accuracy_terms"},
                "prognosis_natural_history": {"prognosis_terms"},
                "prognostic_factor": {"prognostic_factor_terms"},
                "prediction_model": {
                    "prediction_model_terms",
                    "external_validation_terms",
                    "internal_validation_terms",
                    "derivation_terms",
                },
                "risk_etiology_harm": {"risk_etiology_terms"},
                "complication_risk": {"complication_terms"},
                "epidemiology_prevalence_incidence": {
                    "epidemiology_terms",
                },
                "pathophysiology_mechanism": {
                    "pathophysiology_terms", "animal_model_terms",
                    "in_vitro_terms",
                },
            }
            best_rank = -1
            best_signal_score = -1
            for br in branches:
                try:
                    cand = _classify_article_for_branch(
                        article, br,
                        has_higher_evidence_available_in_batch=(
                            has_higher_evidence_in_batch
                        ),
                    )
                except Exception:  # noqa: BLE001
                    continue
                rank = _RANK.get(cand.directness_label, 0)
                br_signals = _detect_signals(article) & (
                    _INTENT_SIGNAL_WEIGHTS.get(str(br.get("intent") or ""), set())
                )
                signal_score = len(br_signals)
                if (
                    rank > best_rank
                    or (rank == best_rank and signal_score > best_signal_score)
                ):
                    best_rank = rank
                    best_signal_score = signal_score
                    chosen_diag = cand
                    chosen = br
        try:
            if chosen_diag is not None:
                diagnostics.append(chosen_diag)
                continue
            diag = _classify_article_for_branch(
                article, chosen,
                has_higher_evidence_available_in_batch=(
                    has_higher_evidence_in_batch
                ),
            )
            diagnostics.append(diag)
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseK3A lane classification failed for article %s (non-fatal)",
                article.get("id"),
            )
            continue
    return ArticleLaneClassificationBatch(
        lane_label_mode="disabled" if lane_label_mode == "disabled" else "shadow",
        cap_detection_mode="disabled" if cap_detection_mode == "disabled" else "shadow",
        diagnostics=diagnostics,
    )


__all__ = [
    # Enums
    "LaneClassifierMode",
    "LaneLabel",
    "DirectnessLabel",
    "CapReasonCode",
    "CapSeverity",
    "BranchAlignment",
    "ArticleIntentFit",
    "EvidenceRoleCandidate",
    # Models
    "CapCandidate",
    "ArticleLaneDiagnostic",
    "ArticleLaneClassificationBatch",
    # Entry point
    "classify_articles",
    # Constants (for diagnostic projection / leak guard)
    "_FORBIDDEN_DIAG_SUBSTRINGS",
]
