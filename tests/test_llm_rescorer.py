"""Tests for the relevance scorer fix + LLM rescorer.

  * relevance_terms.concept_term_set — pulls the FULL PICOTT synonym set.
  * evidence_ranker._relevance — an on-topic paper phrasing the concept via a
    synonym now scores much higher (was excluded before).
  * llm_rescorer.apply_judgments — only LIFTS relevant excluded rows, clears
    the low_relevance exclusion at the floor, never touches safety/cap rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.relevance_terms import concept_term_set  # noqa: E402
from services import evidence_ranker  # noqa: E402
from services import llm_rescorer  # noqa: E402


# ------------------------------------------------------------- concept_term_set
def _parsed_glp1():
    return {
        "key_concepts": ["GLP-1 agonists", "acute coronary syndrome"],
        "mesh_terms": ["Glucagon-Like Peptide-1"],
        "synonyms": ["GLP-1 receptor agonists"],
        "drug_names_and_classes": ["GLP-1 agonists"],
        "picott_json": {
            "must_have_concepts": [
                {"concept": "GLP-1 receptor agonists", "synonyms": [
                    "GLP-1RA", "semaglutide", "liraglutide", "incretin mimetic",
                ]},
                {"concept": "acute myocardial infarction", "synonyms": [
                    "ACS", "STEMI", "NSTEMI",
                ]},
            ],
        },
    }


def test_concept_term_set_includes_picott_synonyms():
    terms = [t.lower() for t in concept_term_set(_parsed_glp1())]
    for expected in ["semaglutide", "glp-1ra", "nstemi", "acs", "liraglutide"]:
        assert expected in terms
    # de-duped: "GLP-1 receptor agonists" appears once despite being in two places
    assert terms.count("glp-1 receptor agonists") == 1


def test_relevance_boosted_by_synonym_match(monkeypatch):
    parsed = _parsed_glp1()
    # An on-topic paper that phrases everything via synonyms the sparse term set
    # (GLP-1 receptor agonists / Glucagon-Like Peptide-1 / GLP-1 agonists) does
    # NOT contain — semaglutide, GLP-1RA, NSTEMI, ACS.
    art = {
        "title": "Semaglutide, a GLP-1RA, and NSTEMI/ACS outcomes",
        "abstract": (
            "Semaglutide (a GLP-1RA) in patients with NSTEMI and ACS. "
            "GLP-1RA therapy reduced ACS events after NSTEMI."
        ),
    }
    monkeypatch.setenv("RELEVANCE_USE_PICOTT_SYNONYMS", "true")
    enriched = evidence_ranker._relevance(art, parsed, ["high_precision"])
    monkeypatch.setenv("RELEVANCE_USE_PICOTT_SYNONYMS", "false")
    sparse = evidence_ranker._relevance(art, parsed, ["high_precision"])
    assert sparse <= 25             # sparse term set matches nothing -> floored
    assert enriched > sparse        # synonym matching lifts the on-topic paper
    assert enriched >= 40


# ------------------------------------------------------------- apply_judgments
def _row(aid, rel, overall, excl="low_relevance"):
    return {
        "_article": {"id": aid, "title": "t", "abstract": "a"},
        "relevance_score": rel,
        "overall_rank_score": overall,
        "exclusion_reason": excl,
    }


def _result(items):
    return llm_rescorer.RescoreResult(items=[
        llm_rescorer.RescoreItem(index=i, label=lbl, confidence=conf)
        for (i, lbl, conf) in items
    ])


def test_apply_lifts_relevant_and_clears_exclusion():
    ranked = [_row("a1", 40, 45)]
    cands = [{"index": 0, "article_id": "a1"}]
    lifted = llm_rescorer.apply_judgments(
        ranked, cands, _result([(0, "relevant", 0.9)]), min_confidence=0.7
    )
    assert lifted == 1
    assert ranked[0]["overall_rank_score"] == 72   # lifted to 'relevant' target
    assert ranked[0]["relevance_score"] == 72.0
    assert ranked[0]["exclusion_reason"] is None    # un-hidden (>= floor 60)
    assert ranked[0]["llm_rescored"] is True


def test_apply_marginal_boosts_but_stays_hidden():
    ranked = [_row("a1", 20, 30)]
    cands = [{"index": 0, "article_id": "a1"}]
    lifted = llm_rescorer.apply_judgments(
        ranked, cands, _result([(0, "marginally_relevant", 0.9)]), min_confidence=0.7
    )
    assert lifted == 1
    assert ranked[0]["overall_rank_score"] == 55       # boosted
    assert ranked[0]["exclusion_reason"] == "low_relevance"  # still < 60 floor


def test_apply_skips_off_topic_and_low_confidence():
    ranked = [_row("a1", 40, 45), _row("a2", 40, 45)]
    cands = [{"index": 0, "article_id": "a1"}, {"index": 1, "article_id": "a2"}]
    lifted = llm_rescorer.apply_judgments(
        ranked, cands,
        _result([(0, "off_topic", 0.99), (1, "relevant", 0.5)]),  # low conf
        min_confidence=0.7,
    )
    assert lifted == 0
    assert ranked[0]["exclusion_reason"] == "low_relevance"
    assert ranked[1]["exclusion_reason"] == "low_relevance"


def test_apply_never_touches_safety_exclusion():
    ranked = [_row("a1", 40, 45, excl="excluded_retracted_or_preprint_or_unsafe")]
    cands = [{"index": 0, "article_id": "a1"}]
    lifted = llm_rescorer.apply_judgments(
        ranked, cands, _result([(0, "highly_relevant", 0.99)]), min_confidence=0.7
    )
    assert lifted == 0
    assert ranked[0]["exclusion_reason"] == "excluded_retracted_or_preprint_or_unsafe"


def test_apply_never_lowers_a_strong_deterministic_score():
    ranked = [_row("a1", 90, 92, excl=None)]  # already high + shown
    cands = [{"index": 0, "article_id": "a1"}]
    lifted = llm_rescorer.apply_judgments(
        ranked, cands, _result([(0, "relevant", 0.9)]), min_confidence=0.7
    )
    assert lifted == 0  # target 72 < current 92 -> no change
    assert ranked[0]["overall_rank_score"] == 92


def test_collect_candidates_only_low_relevance_sorted():
    ranked = [
        _row("a1", 50, 55, excl="low_relevance"),
        _row("a2", 10, 20, excl="low_relevance"),
        _row("a3", 90, 92, excl=None),               # shown -> not a candidate
        _row("a4", 30, 40, excl="excluded_retracted_or_preprint_or_unsafe"),  # safety
    ]
    cands = llm_rescorer.collect_rescore_candidates(ranked, cap=40)
    ids = [c["article_id"] for c in cands]
    assert ids == ["a1", "a2"]        # only low_relevance, highest score first
