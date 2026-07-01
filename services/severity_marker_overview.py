"""Phase K4A.2 — Severity-marker overview detection + directness classifier.

This module is the deterministic recognizer for the
"What are the markers of severity of illness in [disease]?" question
family, plus the closed-set directness classifier that ranks articles
for this question subtype.

HARD CONSTRAINTS (do NOT relax in subclasses):
  - NO LLM. NO model-memory generation.
  - NO publisher scraping. NO new external sources.
  - All classification is DETERMINISTIC from already-available article
    metadata (title, abstract, publication_types_json).
  - Closed-set directness label enum — see ``DirectnessLabel``.
  - This module fires ONLY when the question matches the
    severity-marker-overview pattern. Other questions see no change.
  - The base ``primary_intent`` stays at ``prognostic_factor`` (kept
    inside the existing K1 ClinicalIntent enum so schema-lock holds).
    The signal that the special directness route is active is the
    closed-set ``question_subtype = "severity_marker_overview"`` on
    the picott dict — additive only.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Literal, Optional

# Closed-set directness labels — exactly the labels in the operator spec.
DirectnessLabel = Literal[
    "general_severity_marker_overview",
    "guideline_severity_assessment",
    "multi_marker_systematic_review",
    "severity_score_review",
    "single_marker_systematic_review",
    "single_marker_primary_study",
    "acute_pancreatitis_general_management",
    "treatment_specific_article",
    "wrong_framework_lane",
    "wrong_disease",
]

# Lower index = higher directness (top of the list).
DIRECTNESS_RANKED: tuple[str, ...] = (
    "general_severity_marker_overview",
    "guideline_severity_assessment",
    "multi_marker_systematic_review",
    "severity_score_review",
    "single_marker_systematic_review",
    "single_marker_primary_study",
    "acute_pancreatitis_general_management",
    "treatment_specific_article",
    "wrong_framework_lane",
    "wrong_disease",
)

# Closed-set boost per directness label. Boosts are added on top of the
# base score; caps (below) are the ceiling, lowest-wins.
DIRECTNESS_BOOSTS: dict[str, int] = {
    "general_severity_marker_overview": 20,
    "guideline_severity_assessment": 18,
    "multi_marker_systematic_review": 16,
    "severity_score_review": 14,
    "single_marker_systematic_review": 10,
    "single_marker_primary_study": 0,
    "acute_pancreatitis_general_management": 0,
    "treatment_specific_article": 0,
    "wrong_framework_lane": 0,
    "wrong_disease": 0,
}

# Closed-set cap per directness label. Cap of None means no clamp.
DIRECTNESS_CAPS: dict[str, Optional[int]] = {
    "general_severity_marker_overview": None,
    "guideline_severity_assessment": None,
    "multi_marker_systematic_review": None,
    "severity_score_review": None,
    "single_marker_systematic_review": None,
    "single_marker_primary_study": None,
    "acute_pancreatitis_general_management": 45,
    "treatment_specific_article": 35,
    "wrong_framework_lane": 25,
    "wrong_disease": 15,
}

DIRECTNESS_CAP_REASONS: dict[str, str] = {
    "acute_pancreatitis_general_management": (
        "cap_reason_general_management_no_severity_focus"
    ),
    "treatment_specific_article": (
        "cap_reason_treatment_specific_not_severity_marker"
    ),
    "wrong_framework_lane": "cap_reason_wrong_framework_lane",
    "wrong_disease": "cap_reason_wrong_disease",
}


# ---------------------------------------------------------------------
# Severity-concept synonyms — accepted as equivalent to the literal
# phrase "severity of illness" for cap-coverage purposes.
# ---------------------------------------------------------------------
SEVERITY_CONCEPT_SYNONYMS: tuple[str, ...] = (
    "severity",
    "severe",
    "severity of illness",
    "severe acute pancreatitis",
    "severity prediction",
    "severity assessment",
    "severity score",
    "severity scores",
    "severity marker",
    "severity markers",
    "disease severity",
    "risk stratification",
    "prognostic marker",
    "prognostic markers",
    "prognostic factor",
    "prognostic factors",
    "biomarker",
    "biomarkers",
    "laboratory marker",
    "laboratory markers",
    "inflammatory marker",
    "inflammatory markers",
    "predictor",
    "predictors",
    "prediction",
    "predictive value",
    "predicting severity",
    "predict severity",
    "scoring system",
    "scoring systems",
    "risk score",
    "risk scores",
    "apache",
    "bisap",
    "ranson",
    "ctsi",
    "balthazar",
    "organ failure",
    "persistent organ failure",
    "mortality",
    "icu admission",
    "complications",
    "outcome prediction",
)

# Concepts that strongly indicate a TREATMENT focus (downrank candidates).
_TREATMENT_FOCUS_TERMS: tuple[str, ...] = (
    "treatment", "therapy", "therapies", "management of",
    "probiotic", "probiotics",
    "fluid", "fluid resuscitation", "fluid therapy",
    "octreotide", "ulinastatin", "somatostatin",
    "antibiotic", "antibiotics", "antimicrobial",
    "nutrition", "nasojejunal", "enteral feeding",
    "ercp", "early ercp",
    "drug therapy", "pharmacotherapy",
    "intervention", "operative", "surgery",
)

# Concepts that indicate a SYSTEMATIC REVIEW or META-ANALYSIS or
# OVERVIEW article shape (used to upgrade a "single-marker" article
# to a "review" variant).
_REVIEW_SHAPE_TERMS: tuple[str, ...] = (
    "systematic review", "meta-analysis", "meta analysis",
    "narrative review", "scoping review",
    "umbrella review", "review of",
    "an overview", "overview of",
    "narrative",
)

# Concepts that mark an article as multi-marker (vs single-marker).
_MULTI_MARKER_TERMS: tuple[str, ...] = (
    "markers", "biomarkers", "predictors", "scoring systems",
    "risk scores", "severity scores",
    "prognostic markers", "prognostic factors",
    "laboratory markers", "inflammatory markers",
    "scores", "indices",
)

# Concepts that mark an article as a severity SCORE review specifically.
_SEVERITY_SCORE_TERMS: tuple[str, ...] = (
    "apache", "apache ii", "apache-ii",
    "bisap", "ranson", "ctsi", "balthazar",
    "scoring system", "scoring systems",
    "severity score", "severity scores",
    "risk score", "risk scores",
    "marshall score", "sofa", "qsofa",
    "japan severity score", "atlanta classification",
)


# ---------------------------------------------------------------------
# Question recognition — deterministic regex over the original question.
# ---------------------------------------------------------------------
# Severity-marker QUESTION patterns. Each pattern matches the entire
# question shape. The condition anchor is extracted afterwards.
_SEVERITY_MARKER_QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 1. "What are the markers of severity (of illness)? in [condition]"
    re.compile(
        r"\bwhat\s+are\s+(?:the\s+)?(?:markers|predictors|biomarkers|"
        r"laboratory\s+markers|inflammatory\s+markers|scoring\s+systems?|"
        r"severity\s+scores?|risk\s+scores?|prognostic\s+(?:markers?|factors?))"
        r"\s+(?:of|for|that\s+predict)\s+(?:severity(?:\s+of\s+illness)?|"
        r"severe(?:\s+disease)?|severe\s+\w+|severity\s+(?:prediction|assessment))"
        r"(?:\s+in\s+(?P<cond1>[a-z0-9 ,'\-]+))?",
        re.IGNORECASE,
    ),
    # 2. "What [markers|biomarkers|...] predict (the) (severity|outcome[s]|
    #    mortality|organ failure|disease activity) in [condition]"
    re.compile(
        r"\bwhat\s+(?:[a-z]+\s+)?(?:markers|predictors|biomarkers|laboratory\s+markers|"
        r"inflammatory\s+markers|scoring\s+systems?|severity\s+scores?|"
        r"risk\s+scores?|prognostic\s+(?:markers?|factors?))"
        r"\s+predict\s+(?:the\s+)?(?:severity|severe(?:\s+disease)?|outcomes?|"
        r"mortality|organ\s+failure|disease\s+activity|disease\s+progression)"
        r"(?:\s+in\s+(?P<cond2>[a-z0-9 ,'\-]+))?",
        re.IGNORECASE,
    ),
    # 3. "What scores or markers predict severe [condition]"
    re.compile(
        r"\bwhat\s+(?:scores?\s+or\s+markers?|markers?\s+or\s+scores?|"
        r"predictors?\s+or\s+scores?)\s+predict\s+severe\s+"
        r"(?P<cond3>[a-z0-9 ,'\-]+)",
        re.IGNORECASE,
    ),
    # 4. "How is severity predicted/assessed in [condition]"
    re.compile(
        r"\bhow\s+is\s+(?:severity|severe\s+disease|outcome|prognosis)\s+"
        r"(?:predicted|assessed|determined|stratified)\s+in\s+"
        r"(?P<cond4>[a-z0-9 ,'\-]+)",
        re.IGNORECASE,
    ),
    # 5. "Predictors of severe [condition]" / "Markers for predicting
    #    severity..." — captures both `of [cond]` AND bare
    #    `severe [cond]` (multi-word disease after `severe`).
    re.compile(
        r"\b(?:predictors|markers|biomarkers|scoring\s+systems?|risk\s+scores?)\s+"
        r"(?:of|for(?:\s+predicting)?)\s+(?:severe|severity)"
        r"(?:\s+(?:of|in)\s+(?P<cond5>[a-z0-9 ,'\-]+))?",
        re.IGNORECASE,
    ),
    # 6. Phase K4A.2.1 — "What are predictors of severe [disease]"
    #    (bare multi-word disease right after ``severe`` — no "in/of").
    re.compile(
        r"\bwhat\s+(?:are\s+|is\s+)?(?:the\s+)?(?:predictors?|markers?|"
        r"biomarkers?|scoring\s+systems?|severity\s+scores?|risk\s+scores?|"
        r"laboratory\s+markers?|inflammatory\s+markers?|prognostic\s+"
        r"(?:markers?|factors?))\s+(?:of|for)\s+severe\s+(?P<cond6>[a-z0-9 ,'\-]+)",
        re.IGNORECASE,
    ),
    # 7. Phase K4A.2.1 — "What [X] markers predict [disease] severity"
    #    (disease appears BEFORE the severity keyword).
    re.compile(
        r"\bwhat\s+(?:[a-z]+\s+)?(?:markers?|predictors?|biomarkers?|"
        r"scores?|scoring\s+systems?|severity\s+scores?|risk\s+scores?|"
        r"prognostic\s+(?:markers?|factors?))\s+predict\s+"
        r"(?P<cond7>[a-z][a-z0-9 ,'\-]+?)\s+(?:severity|"
        r"severe\s+disease|outcomes?|mortality)",
        re.IGNORECASE,
    ),
)


def is_severity_marker_question(question: str) -> bool:
    """Pure boolean — does this question match the severity-marker
    overview family? Used by the picott dict builder + tests."""
    q = (question or "").strip()
    if not q:
        return False
    for pat in _SEVERITY_MARKER_QUESTION_PATTERNS:
        if pat.search(q):
            return True
    return False


def extract_condition_anchor(question: str) -> Optional[str]:
    """Best-effort closed-set extraction of the condition anchor
    (e.g. ``"acute pancreatitis"``). Returns ``None`` when no anchor
    was found — caller decides whether to fall back."""
    q = (question or "").strip()
    if not q:
        return None
    for pat in _SEVERITY_MARKER_QUESTION_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        # First non-empty named group wins.
        for name in ("cond1", "cond2", "cond3", "cond4", "cond5",
                     "cond6", "cond7"):
            try:
                v = m.group(name)
            except IndexError:
                v = None
            if v:
                cleaned = re.sub(r"\s+", " ", v.strip().lower()).strip(",.?! ")
                # Stop at common boundary words.
                cleaned = re.split(
                    r"\b(?:and|or|with|in|among)\b", cleaned,
                )[0].strip()
                return cleaned or None
    return None


# ---------------------------------------------------------------------
# Article directness classifier — deterministic from metadata.
# ---------------------------------------------------------------------
def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).lower())


def _any_in(haystack: str, needles: Iterable[str]) -> bool:
    return any(n in haystack for n in needles if n)


def _has_severity_concept(text: str) -> bool:
    """Closed-set severity-concept presence check. The cap-fix lives here:
    we accept ANY severity-concept synonym, not just the literal phrase
    'severity of illness'."""
    if not text:
        return False
    return _any_in(text, SEVERITY_CONCEPT_SYNONYMS)


def _has_condition_anchor(
    text: str,
    condition_anchor: str,
    condition_synonyms: Iterable[str] = (),
) -> bool:
    if not text:
        return False
    needles = [condition_anchor.lower()] + [str(s).lower() for s in condition_synonyms]
    return _any_in(text, [n for n in needles if n])


def classify_severity_marker_directness(
    article: dict[str, Any],
    *,
    condition_anchor: str,
    condition_synonyms: Iterable[str] = (),
) -> DirectnessLabel:
    """Closed-set directness label for the severity-marker overview
    question family. Pure function on article metadata. NO LLM.

    Decision tree (deterministic, evaluated top-to-bottom):

      1. wrong_disease         — condition anchor not present anywhere
      2. wrong_framework_lane  — disease present BUT clearly off-topic
                                  framework (e.g., antimicrobial-AMR
                                  guideline that mentions pancreatitis
                                  in passing)
      3. guideline_severity_assessment
                                — is_guideline=True AND severity-concept
                                  present in title or abstract
      4. general_severity_marker_overview
                                — title explicitly frames the article
                                  as an overview of severity markers /
                                  predictors / risk stratification for
                                  the condition
      5. multi_marker_systematic_review
                                — systematic-review/meta-analysis shape
                                  AND multi-marker terms present
      6. severity_score_review  — review shape AND focused on severity
                                  scores (APACHE/BISAP/Ranson/CTSI/...)
      7. single_marker_systematic_review
                                — systematic-review shape AND severity
                                  context BUT specific single marker
      8. single_marker_primary_study
                                — primary study with severity concept
                                  but ONE marker
      9. treatment_specific_article
                                — treatment focus without severity
                                  marker framing
     10. acute_pancreatitis_general_management
                                — generic disease management article
                                  without severity-marker framing
    """
    title = _normalize_text(article.get("title"))
    abstract = _normalize_text(article.get("abstract"))
    blob = f"{title}  {abstract}"
    pub_types = [
        _normalize_text(p) for p in (article.get("publication_types_json") or [])
    ]
    is_guideline = bool(article.get("is_guideline")) or any(
        ("guideline" in p or "practice guideline" in p) for p in pub_types
    )
    is_sr = any("systematic review" in p for p in pub_types) or (
        "systematic review" in blob
    )
    is_meta = any("meta-analysis" in p or "meta analysis" in p for p in pub_types) or (
        "meta-analysis" in blob or "meta analysis" in blob
    )
    # Bare "review" pubtype (PubMed publication type "Review" on a
    # comparison / scoring paper still qualifies as review-shape for
    # this classifier, since the directness verdict cares about the
    # "is this a synthesis vs a primary study" question — APACHE-vs-
    # BISAP comparison reviews live here).
    is_bare_review_pubtype = any(
        p.strip() == "review" or p.strip().startswith("review ")
        for p in pub_types
    )
    is_review_shape = (
        is_sr or is_meta
        or is_bare_review_pubtype
        or _any_in(blob, _REVIEW_SHAPE_TERMS)
    )

    has_condition = _has_condition_anchor(
        blob, condition_anchor, condition_synonyms,
    )
    has_severity = _has_severity_concept(blob)
    has_multi_marker = _any_in(blob, _MULTI_MARKER_TERMS)
    has_score_focus = _any_in(blob, _SEVERITY_SCORE_TERMS)
    has_treatment_focus = _any_in(blob, _TREATMENT_FOCUS_TERMS)

    # 1. wrong_disease
    if not has_condition:
        return "wrong_disease"

    # 2. wrong_framework_lane — disease mentioned but framework is
    # clearly off-target (e.g., antimicrobial / IAI guideline that
    # cites pancreatitis as one of many indications).
    off_topic_framework = any(
        marker in blob for marker in (
            "antimicrobial stewardship", "stewardship program",
            "intra-abdominal infection", "intra abdominal infection",
            "surgical site infection",
            "antibiotic prophylaxis review",
            "amr report", "antimicrobial resistance",
        )
    )
    if off_topic_framework and not has_severity:
        return "wrong_framework_lane"

    # 3. guideline_severity_assessment
    if is_guideline and has_severity:
        return "guideline_severity_assessment"

    # 4. general_severity_marker_overview — title-framed overview
    title_overview_signal = any(
        kw in title for kw in (
            "markers of severity",
            "predictors of severity",
            "predictors of severe",
            "severity markers",
            "severity prediction",
            "severity assessment",
            "risk stratification",
            "biomarkers in",
            "biomarkers of",
            "predicting severity",
            "predict severity",
            "predicting severe",
            "prognostic markers",
            "prognostic factors",
        )
    )
    if title_overview_signal and has_condition and has_severity:
        return "general_severity_marker_overview"

    # 5. severity_score_review — review shape + named-score focus
    # (APACHE/BISAP/Ranson/CTSI/...). More specific than the generic
    # multi-marker review bucket, so it wins when both could apply.
    if is_review_shape and has_score_focus and has_severity:
        return "severity_score_review"

    # 6. multi_marker_systematic_review
    if is_review_shape and has_multi_marker and has_severity:
        return "multi_marker_systematic_review"

    # 7. single_marker_systematic_review
    if is_review_shape and has_severity:
        return "single_marker_systematic_review"

    # 8. single_marker_primary_study (primary study + severity concept)
    if has_severity and not is_review_shape and not has_treatment_focus:
        return "single_marker_primary_study"

    # 9. treatment_specific_article
    if has_treatment_focus and not has_severity:
        return "treatment_specific_article"
    # Also treat treatment-focus + severity-mention as treatment_specific
    # when there is no review shape or score focus.
    if has_treatment_focus and has_severity and not (is_review_shape or has_score_focus):
        return "treatment_specific_article"

    # 10. acute_pancreatitis_general_management — disease management
    # article without severity framing.
    return "acute_pancreatitis_general_management"


# ---------------------------------------------------------------------
# Query family generator — deterministic strings the planner can use
# directly OR expand into source-specific syntax. Kept identifier-free.
# ---------------------------------------------------------------------
def severity_or_block_synonyms() -> list[str]:
    """Closed-set OR-block synonyms used to broaden the outcome must-have
    when the severity-marker overview route is active.

    The query compiler builds ``(condition) AND (outcome_terms)``;
    augmenting the outcome must-have's ``synonyms`` with this list
    turns the AND-block's second clause from a single phrase like
    ``"severity of illness"`` into a broad OR-block that ALSO accepts
    ``severity``, ``severe``, ``predictor``, ``marker``, ``biomarker``,
    closed-set named scores (APACHE/BISAP/Ranson/CTSI), ``organ
    failure``, ``mortality``, etc. This is the cap-fix the K4A.2 spec
    requires: equivalent severity concepts are accepted in lieu of the
    literal phrase. Closed-set, deterministic, no LLM.
    """
    # Curated subset of SEVERITY_CONCEPT_SYNONYMS — names that pull
    # double-duty as searchable identifiers in MEDLINE / Europe PMC.
    # Closed-set + deduped (preserve insertion order so the compiler
    # output is deterministic across runs).
    closed: list[str] = [
        "severity",
        "severe",
        "severity of illness",
        "severity prediction",
        "severity assessment",
        "severity score",
        "severity scores",
        "severity marker",
        "severity markers",
        "disease severity",
        "risk stratification",
        "prognostic marker",
        "prognostic markers",
        "prognostic factor",
        "prognostic factors",
        "biomarker",
        "biomarkers",
        "laboratory marker",
        "laboratory markers",
        "inflammatory marker",
        "inflammatory markers",
        "predictor",
        "predictors",
        "prediction",
        "predictive value",
        "scoring system",
        "scoring systems",
        "risk score",
        "risk scores",
        "APACHE",
        "BISAP",
        "Ranson",
        "CTSI",
        "Balthazar",
        "organ failure",
        "persistent organ failure",
        "mortality",
        "ICU admission",
        "complications",
        "outcome prediction",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for s in closed:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def query_families_for(condition_anchor: str) -> list[dict[str, str]]:
    """Return the closed-set query families the planner should run for
    a severity-marker question. Each family carries a human-readable
    rationale string for the diagnostics panel."""
    cond = condition_anchor.strip()
    if not cond:
        return []
    or_block = (
        "severity OR severe OR predictor* OR prediction OR marker* OR "
        "biomarker* OR risk stratification OR scoring system* OR "
        "severity score* OR APACHE OR BISAP OR Ranson OR CTSI OR "
        "organ failure"
    )
    return [
        {
            "name": "broad_recall",
            "query": f'"{cond}" AND ({or_block})',
            "rationale": (
                "Broad recall — accepts any severity / predictor / score concept."
            ),
        },
        {
            "name": "review_overview",
            "query": (
                f'"{cond}" AND (severity prediction OR severity assessment OR '
                f'risk stratification OR prognostic markers OR biomarkers OR '
                f'predictors) AND (review OR systematic review OR meta-analysis '
                f'OR guideline OR consensus)'
            ),
            "rationale": "Prefer reviews, systematic reviews, guidelines.",
        },
        {
            "name": "title_focused_overview",
            "query": (
                f'("{cond}"[Title] OR "{cond}"[Abstract]) AND '
                f'(markers[Title] OR predictors[Title] OR severity[Title] OR '
                f'"risk stratification"[Title])'
            ),
            "rationale": "Title/abstract framing — overview shape.",
        },
        {
            "name": "score_or_marker_branch",
            "query": (
                f'"{cond}" AND (APACHE OR BISAP OR Ranson OR CTSI OR '
                f'biomarker* OR marker* OR predictor*)'
            ),
            "rationale": "Score / marker branch for completeness.",
        },
    ]
