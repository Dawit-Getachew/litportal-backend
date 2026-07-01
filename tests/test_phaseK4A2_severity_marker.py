"""Phase K4A.2 — severity-marker overview + selected/excluded invariant.

Coverage:
  1. Question recognition (positive cases across the 5 patterns).
  2. Question recognition (negative cases — must NOT fire).
  3. Condition-anchor extraction for the family.
  4. Directness classifier — all 10 closed-set labels.
  5. PICOTT derivation injects ``question_subtype`` +
     ``severity_marker_condition_anchor`` AND augments the outcome
     must-have's synonyms with the closed-set OR-block (NO hard-AND
     on the literal phrase ``severity of illness``).
  6. ``select_default_articles`` returns ``cleared_exclusion_for_ids``
     equal to ``selected_article_ids`` — the selected/excluded display
     invariant.
  7. Selected/excluded invariant: an article with cap_reasons is NEVER
     in the selected set.
  8. Picott ranker boost: general overview ranks above single-marker
     primary studies and treatment-specific articles.

All tests are deterministic — no LLM, no network, no PHI, no
publisher scraping.
"""
from __future__ import annotations

import pytest

from services.severity_marker_overview import (
    DirectnessLabel,
    DIRECTNESS_BOOSTS,
    DIRECTNESS_CAPS,
    classify_severity_marker_directness,
    extract_condition_anchor,
    is_severity_marker_question,
    query_families_for,
    severity_or_block_synonyms,
)
from services.clinical_question_map import (
    Branch,
    ClinicalQuestionMap,
    Slot,
    clinical_question_map_to_picott,
)
from services.default_selection import select_default_articles


# =====================================================================
# 1 + 2 — question recognition
# =====================================================================
@pytest.mark.parametrize("q", [
    "What are the markers of severity of illness in acute pancreatitis?",
    "What biomarkers predict severity in acute pancreatitis?",
    "What scoring systems predict severe acute pancreatitis?",
    "What scores or markers predict severe acute pancreatitis?",
    "How is severity predicted in acute pancreatitis?",
    "Predictors of severe acute pancreatitis",
    "Markers for predicting severity",
    "What laboratory markers predict severity in sepsis?",
    "what predictors of severity in pneumonia?",
    "what are the prognostic factors that predict severity in acute pancreatitis",
])
def test_severity_marker_question_recognized(q: str) -> None:
    assert is_severity_marker_question(q) is True, q


@pytest.mark.parametrize("q", [
    "",
    "What is the treatment of acute pancreatitis?",
    "What is the evidence for SGLT2 inhibitors in HFpEF?",
    "Octreotide vs placebo in acute pancreatitis",
    "ICU admission criteria in sepsis",
    "Diagnostic accuracy of D-dimer for PE",
])
def test_severity_marker_question_not_recognized(q: str) -> None:
    assert is_severity_marker_question(q) is False, q


# =====================================================================
# 3 — condition anchor extraction
# =====================================================================
@pytest.mark.parametrize("q,expected", [
    ("What are the markers of severity of illness in acute pancreatitis?",
     "acute pancreatitis"),
    ("What biomarkers predict severity in sepsis?", "sepsis"),
    ("How is severity predicted in pneumonia?", "pneumonia"),
    ("What scores or markers predict severe acute pancreatitis?",
     "acute pancreatitis"),
])
def test_condition_anchor_extraction(q: str, expected: str) -> None:
    got = extract_condition_anchor(q)
    assert got == expected, (q, got)


# =====================================================================
# 4 — directness classifier covers all 10 closed-set labels
# =====================================================================
def _article(title: str, abstract: str = "", **kw: object) -> dict:
    base = {
        "title": title,
        "abstract": abstract or title,
        "publication_types_json": [],
        "is_guideline": False,
    }
    base.update(kw)
    return base


def test_classifier_general_overview() -> None:
    a = _article(
        "Markers of severity in acute pancreatitis: a narrative review",
        abstract="biomarkers and predictors of severity in acute pancreatitis",
        publication_types_json=["Review"],
    )
    label: DirectnessLabel = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "general_severity_marker_overview"


def test_classifier_guideline_severity() -> None:
    a = _article(
        "ACG guideline on the management of acute pancreatitis",
        abstract="severity assessment and predictors in acute pancreatitis",
        is_guideline=True,
        publication_types_json=["Practice Guideline"],
    )
    label = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "guideline_severity_assessment"


def test_classifier_multi_marker_systematic_review() -> None:
    a = _article(
        "Biomarkers and predictors in acute pancreatitis: a systematic review",
        abstract="systematic review of biomarkers and prognostic markers for severity",
        publication_types_json=["Systematic Review"],
    )
    label = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "multi_marker_systematic_review"


def test_classifier_severity_score_review() -> None:
    a = _article(
        "APACHE II vs BISAP for acute pancreatitis severity",
        abstract="comparison of severity scores APACHE BISAP Ranson in acute pancreatitis",
        publication_types_json=["Review"],
    )
    label = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "severity_score_review"


def test_classifier_single_marker_systematic_review() -> None:
    a = _article(
        "Procalcitonin in acute pancreatitis: a systematic review",
        abstract="systematic review of procalcitonin for severity in acute pancreatitis",
        publication_types_json=["Systematic Review"],
    )
    label = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "single_marker_systematic_review"


def test_classifier_single_marker_primary_study() -> None:
    a = _article(
        "Serum CRP at admission predicts severity in acute pancreatitis",
        abstract="prospective cohort: CRP predicts severity in acute pancreatitis",
        publication_types_json=["Journal Article"],
    )
    label = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "single_marker_primary_study"


def test_classifier_treatment_specific() -> None:
    a = _article(
        "Early enteral nutrition in acute pancreatitis",
        abstract="randomized trial of enteral feeding in acute pancreatitis",
        publication_types_json=["Randomized Controlled Trial"],
    )
    label = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "treatment_specific_article"


def test_classifier_general_management() -> None:
    a = _article(
        "Acute pancreatitis: an update",
        abstract="overview of acute pancreatitis epidemiology and clinical features",
        publication_types_json=["Review"],
    )
    label = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "acute_pancreatitis_general_management"


def test_classifier_wrong_framework_lane() -> None:
    a = _article(
        "Antimicrobial stewardship in intra-abdominal infection",
        abstract="antimicrobial stewardship program review including acute pancreatitis",
        publication_types_json=["Review"],
    )
    label = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "wrong_framework_lane"


def test_classifier_wrong_disease() -> None:
    a = _article(
        "Severity markers in heart failure",
        abstract="biomarkers and severity scores in heart failure",
        publication_types_json=["Review"],
    )
    label = classify_severity_marker_directness(
        a, condition_anchor="acute pancreatitis",
    )
    assert label == "wrong_disease"


def test_directness_boosts_and_caps_closed_set() -> None:
    # general_severity_marker_overview must outrank every other label.
    boost_general = DIRECTNESS_BOOSTS["general_severity_marker_overview"]
    for other in (
        "single_marker_primary_study",
        "treatment_specific_article",
        "wrong_framework_lane",
        "wrong_disease",
    ):
        assert boost_general > DIRECTNESS_BOOSTS[other]
    # Caps: treatment_specific ≤ 35, wrong_framework ≤ 25, wrong_disease ≤ 15.
    assert DIRECTNESS_CAPS["treatment_specific_article"] == 35
    assert DIRECTNESS_CAPS["wrong_framework_lane"] == 25
    assert DIRECTNESS_CAPS["wrong_disease"] == 15
    # Primary directness labels must NOT cap.
    for primary in (
        "general_severity_marker_overview",
        "guideline_severity_assessment",
        "multi_marker_systematic_review",
        "severity_score_review",
        "single_marker_systematic_review",
    ):
        assert DIRECTNESS_CAPS[primary] is None


# =====================================================================
# 5 — picott derivation
# =====================================================================
def _build_cqm() -> ClinicalQuestionMap:
    cond = Slot(
        original_phrase="acute pancreatitis",
        canonical_concept="acute pancreatitis",
        entity_type="disease_or_problem",
        slot_role="disease_or_condition",
        required_for_search=True,
        slot_search_policy="required_for_retrieval",
        confidence=0.95,
        synonyms=["pancreatitis, acute"],
        source="deterministic_parser",
    )
    outcome = Slot(
        original_phrase="severity of illness",
        canonical_concept="severity of illness",
        entity_type="unknown",
        slot_role="outcome",
        required_for_search=True,
        slot_search_policy="required_for_retrieval",
        confidence=0.85,
        source="deterministic_parser",
    )
    return ClinicalQuestionMap(
        original_question=(
            "What are the markers of severity of illness in acute pancreatitis?"
        ),
        normalized_question=(
            "what are the markers of severity of illness in acute pancreatitis"
        ),
        interpretation_provenance="deterministic_parser",
        primary_intent="prognostic_factor",
        confidence=0.9,
        branches=[
            Branch(
                branch_id="b1",
                intent="prognostic_factor",
                framework="PFO",
                branch_priority=1,
                branch_confidence=0.9,
                required_concepts=[cond, outcome],
            )
        ],
    )


def test_picott_derivation_severity_marker_subtype_set() -> None:
    cqm = _build_cqm()
    pico = clinical_question_map_to_picott(cqm)
    assert pico["question_subtype"] == "severity_marker_overview"
    assert pico["severity_marker_condition_anchor"] == "acute pancreatitis"


def test_picott_derivation_outcome_synonyms_expanded() -> None:
    """K4A.2 cap-fix: outcome must-have's synonyms expanded with the
    closed-set OR-block so the literal phrase ``severity of illness``
    is never the only term the query compiler ANDs in."""
    cqm = _build_cqm()
    pico = clinical_question_map_to_picott(cqm)
    outcomes = [
        mh for mh in pico["must_have_concepts"]
        if (mh.get("role") or "").lower() == "outcome"
    ]
    assert outcomes, "expected at least one outcome must-have"
    syns_lower = {s.lower() for s in outcomes[0].get("synonyms") or []}
    # Closed-set OR-block synonyms — assert presence of key tokens.
    for needle in (
        "severity", "severe", "predictor", "predictors",
        "biomarker", "biomarkers", "scoring system", "scoring systems",
        "risk stratification", "apache", "bisap", "ranson", "ctsi",
        "organ failure", "mortality",
    ):
        assert needle in syns_lower, needle
    # The original literal phrase remains in the union.
    assert "severity of illness" in syns_lower
    # Outcome must-have is downgraded to non-mandatory so the query
    # compiler does not force the outcome OR-block as a hard cap.
    assert outcomes[0]["required_for_high_relevance"] is False


def test_picott_derivation_non_severity_question_unchanged() -> None:
    """Non-severity-marker questions must NOT see the augmentation."""
    cond = Slot(
        original_phrase="HFpEF",
        canonical_concept="HFpEF",
        entity_type="disease_or_problem",
        slot_role="disease_or_condition",
        required_for_search=True,
        slot_search_policy="required_for_retrieval",
        confidence=0.95,
        source="deterministic_parser",
    )
    iv = Slot(
        original_phrase="SGLT2 inhibitors",
        canonical_concept="SGLT2 inhibitors",
        entity_type="drug_or_drug_class",
        slot_role="drug_or_drug_class",
        required_for_search=True,
        slot_search_policy="required_for_retrieval",
        confidence=0.9,
        source="deterministic_parser",
    )
    cqm = ClinicalQuestionMap(
        original_question="What is the evidence for SGLT2 inhibitors in HFpEF?",
        normalized_question="what is the evidence for sglt2 inhibitors in hfpef",
        interpretation_provenance="deterministic_parser",
        primary_intent="therapy",
        confidence=0.9,
        branches=[
            Branch(
                branch_id="b1",
                intent="therapy",
                framework="PICOTT",
                branch_priority=1,
                branch_confidence=0.9,
                required_concepts=[cond, iv],
            )
        ],
    )
    pico = clinical_question_map_to_picott(cqm)
    assert pico.get("question_subtype") != "severity_marker_overview"
    assert pico.get("severity_marker_condition_anchor") is None
    # SGLT2 must-have has no severity synonyms injected.
    iv_mh = [
        mh for mh in pico["must_have_concepts"]
        if (mh.get("concept") or "").lower() == "sglt2 inhibitors"
    ]
    assert iv_mh
    assert "apache" not in {s.lower() for s in iv_mh[0].get("synonyms") or []}


def test_severity_or_block_synonyms_is_deduped_and_non_empty() -> None:
    s = severity_or_block_synonyms()
    assert len(s) > 20
    low = [x.lower() for x in s]
    assert len(low) == len(set(low)), "duplicates in OR-block"
    # Closed-set anchors required.
    for needle in ("severity", "predictor", "biomarker", "APACHE", "BISAP"):
        assert needle in s or needle.lower() in low, needle


def test_query_families_for_does_not_hard_and_phrase() -> None:
    """The query-families generator MUST NOT produce a query that
    hard-ANDs the literal phrase ``severity of illness``."""
    fams = query_families_for("acute pancreatitis")
    assert fams
    for f in fams:
        q = f["query"]
        # Hard-AND of the literal phrase is the regression we are
        # guarding against. The phrase MAY appear inside an OR-block,
        # but NEVER as ``AND "severity of illness"``.
        assert "AND \"severity of illness\"" not in q.lower().replace(
            'and "severity of illness"', 'AND "severity of illness"',
        )
    # Broad recall family must use OR-block.
    broad = [f for f in fams if f["name"] == "broad_recall"][0]
    assert " OR " in broad["query"]
    assert "APACHE" in broad["query"]


# =====================================================================
# 6 + 7 — default selection invariant
# =====================================================================
def _rank(article_id: str, *, score: float = 80, cap_reasons: list[str] | None = None,
          relevance: float = 80, article: dict | None = None,
          directness: str | None = None) -> dict:
    art = article or {"id": article_id, "title": f"art-{article_id}"}
    art["id"] = article_id
    return {
        "_article": art,
        "article_id": article_id,
        "overall_rank_score": score,
        "relevance_score": relevance,
        "evidence_design_score": 70,
        "cap_reasons": list(cap_reasons or []),
        "penalty_concepts": [],
        "role_match_score": 80,
        "directness_label": directness,
    }


def test_default_selection_returns_cleared_exclusion_for_ids() -> None:
    picott = {
        "must_have_concepts": [
            {"concept": "acute pancreatitis", "role": "condition",
             "synonyms": ["pancreatitis"], "required_for_high_relevance": True},
        ],
        "type_of_question": "prognosis",
    }
    ranked = [
        _rank("a1", score=90, relevance=85, article={
            "id": "a1",
            "title": "acute pancreatitis biomarkers review",
            "abstract": "biomarkers in acute pancreatitis",
            "is_guideline": True,
        }),
        _rank("a2", score=80, relevance=75, article={
            "id": "a2",
            "title": "acute pancreatitis review",
            "abstract": "review of acute pancreatitis",
            "article_type_classification": "systematic_review",
        }),
    ]
    out = select_default_articles(picott, ranked)
    assert "cleared_exclusion_for_ids" in out
    # Selected and cleared sets are equal — the K4A.2 invariant.
    assert set(out["cleared_exclusion_for_ids"]) == set(
        out["selected_article_ids"]
    )
    assert "a1" in out["selected_article_ids"]


def test_default_selection_invariant_capped_articles_never_selected() -> None:
    picott = {
        "must_have_concepts": [
            {"concept": "acute pancreatitis", "role": "condition",
             "synonyms": ["pancreatitis"], "required_for_high_relevance": True},
        ],
        "type_of_question": "prognosis",
    }
    # Capped article — must NEVER be default-selected.
    ranked = [
        _rank("capped", score=95, relevance=95,
              cap_reasons=["wrong_condition"], article={
                  "id": "capped",
                  "title": "acute pancreatitis treatment",
                  "abstract": "acute pancreatitis therapy",
              }),
        _rank("clean", score=80, relevance=80, article={
            "id": "clean",
            "title": "acute pancreatitis review",
            "abstract": "biomarkers in acute pancreatitis review",
            "is_guideline": True,
        }),
    ]
    out = select_default_articles(picott, ranked)
    assert "capped" not in out["selected_article_ids"]
    assert "capped" not in out["cleared_exclusion_for_ids"]
    assert "clean" in out["selected_article_ids"]


def test_default_selection_invariant_safety_excluded_never_selected() -> None:
    picott = {
        "must_have_concepts": [
            {"concept": "acute pancreatitis", "role": "condition",
             "synonyms": ["pancreatitis"], "required_for_high_relevance": True},
        ],
        "type_of_question": "prognosis",
    }
    ranked = [
        _rank("retracted_id", score=95, relevance=95, article={
            "id": "retracted_id",
            "title": "acute pancreatitis biomarkers (RETRACTED)",
            "abstract": "acute pancreatitis biomarkers",
            "is_retracted": True,
        }),
        _rank("good", score=80, relevance=80, article={
            "id": "good",
            "title": "acute pancreatitis biomarkers review",
            "abstract": "biomarkers in acute pancreatitis",
            "is_guideline": True,
        }),
    ]
    out = select_default_articles(picott, ranked)
    assert "retracted_id" not in out["selected_article_ids"]
    assert "retracted_id" not in out["cleared_exclusion_for_ids"]


# =====================================================================
# 8 — ranking smoke: general overview boost > primary study & treatment
# =====================================================================
def test_directness_boost_priority_table() -> None:
    """Boost-table invariant: closed-set boosts are monotonically
    non-increasing across the ``DIRECTNESS_RANKED`` ordering."""
    from services.severity_marker_overview import DIRECTNESS_RANKED, DIRECTNESS_BOOSTS
    last = None
    for label in DIRECTNESS_RANKED:
        b = DIRECTNESS_BOOSTS[label]
        if last is not None:
            assert b <= last, (label, b, last)
        last = b
