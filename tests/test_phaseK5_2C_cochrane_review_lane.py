"""Phase K5.2C-0 — PubMed-indexed Cochrane Review lane (unit tests).

Covers: ta/jour field-tag query QA, Cochrane detection from PubMed metadata,
source-registry permission defaults, PubMed-only cochrane_review query family
(applicability + non-applicability), and the BOUNDED, cap-limited,
directness-gated ranker boost (caps always dominate; wrong-disease /
missing-anchor Cochrane reviews receive NO rescue).
"""
from __future__ import annotations

import copy

from models.guideline_sources import GuidelineSourceDoc
from services.cochrane_lane import (
    COCHRANE_LANE_PROVENANCE,
    article_cochrane_provenance,
    cochrane_boost_points,
    cochrane_boost_tier,
    cochrane_query_applicable,
)
from services.query_compiler import (
    boolean_query_compiler,
    validate_query_syntax,
)
from services.picott_ranker import picott_aware_ranker


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _therapy_picott() -> dict:
    return {
        "type_of_question": "therapy",
        "question_subtype": "",
        "normalized_question": "anticoagulation for atrial fibrillation",
        "must_have_concepts": [
            {"concept": "atrial fibrillation", "role": "condition",
             "synonyms": ["af"], "required_for_high_relevance": True},
            {"concept": "anticoagulation", "role": "intervention",
             "synonyms": ["anticoagulant"], "required_for_high_relevance": True},
        ],
        "should_have_concepts": [],
        "ranking_concepts": [],
        "negative_or_penalty_concepts": [],
        "pico": {},
    }


def _ranked_row(*, ck, title, abstract, journal="", doi_normalized="",
                base=55) -> dict:
    return {
        "_article": {
            "canonical_key": ck,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "doi_normalized": doi_normalized,
            "article_type_classification": "systematic_review",
        },
        "overall_rank_score": base,
        "relevance_score": float(base),
        "evidence_design_score": 50.0,
        "rank_explanation": "",
    }


# --------------------------------------------------------------------------
# A) Query QA — ta / jour field tags now pass
# --------------------------------------------------------------------------
def test_ta_field_tag_passes_pubmed_query_qa():
    qs = ('"Cochrane Database Syst Rev"[ta] AND '
          '("Atrial Fibrillation"[MeSH Terms] OR "atrial fibrillation"[tiab])')
    v = validate_query_syntax(qs, source="pubmed")
    assert v.valid is True, v.errors


def test_jour_field_tag_passes_pubmed_query_qa():
    v = validate_query_syntax('"cochrane"[jour] AND "stroke"[tiab]', source="pubmed")
    assert v.valid is True, v.errors


def test_unknown_pubmed_tag_still_rejected():
    v = validate_query_syntax('"x"[bogusfield]', source="pubmed")
    assert v.valid is False
    assert "invalid_field_tag" in v.errors


# --------------------------------------------------------------------------
# B) Cochrane detection from PubMed metadata
# --------------------------------------------------------------------------
def test_detect_by_full_journal_name():
    art = {"journal": "Cochrane Database of Systematic Reviews"}
    assert article_cochrane_provenance(art) == COCHRANE_LANE_PROVENANCE


def test_detect_by_journal_abbreviation():
    art = {"journal": "Cochrane Database Syst Rev"}
    assert article_cochrane_provenance(art) == COCHRANE_LANE_PROVENANCE


def test_detect_by_journal_abbrev_trailing_period():
    art = {"journal": "Cochrane Database Syst Rev."}
    assert article_cochrane_provenance(art) == COCHRANE_LANE_PROVENANCE


def test_detect_by_validated_doi_prefix():
    art = {"journal": "", "doi_normalized": "10.1002/14651858.cd004816.pub3"}
    assert article_cochrane_provenance(art) == COCHRANE_LANE_PROVENANCE


def test_not_cochrane_for_other_journal():
    art = {"journal": "BMJ", "doi_normalized": "10.1136/bmj.x"}
    assert article_cochrane_provenance(art) is None


def test_not_cochrane_for_generic_cochrane_mention():
    # A non-CDSR journal that merely contains the word "cochrane" should NOT
    # trip (requires "cochrane database" + systematic-review token).
    art = {"journal": "Cochrane Reviews in Cardiology"}
    assert article_cochrane_provenance(art) is None


def test_detect_handles_non_dict():
    assert article_cochrane_provenance(None) is None
    assert article_cochrane_provenance("nope") is None


# --------------------------------------------------------------------------
# C) Applicability + boost tiers (closed set, bounded)
# --------------------------------------------------------------------------
def test_applicable_for_therapy_strong():
    p = _therapy_picott()
    assert cochrane_query_applicable(p) is True
    assert cochrane_boost_tier(p) == "strong"
    assert cochrane_boost_points(p) == 8


def test_diagnosis_strong():
    p = _therapy_picott(); p["type_of_question"] = "diagnosis"
    assert cochrane_boost_tier(p) == "strong"
    assert cochrane_query_applicable(p) is True


def test_drug_safety_subtype_tier():
    p = _therapy_picott()
    p["type_of_question"] = "etiology_harm"
    p["question_subtype"] = "drug_safety"
    assert cochrane_boost_tier(p) == "drug_safety"
    assert cochrane_boost_points(p) == 7
    assert cochrane_query_applicable(p) is True


def test_prognosis_moderate():
    p = _therapy_picott(); p["type_of_question"] = "prognosis"
    assert cochrane_boost_tier(p) == "moderate"
    assert cochrane_boost_points(p) == 5
    assert cochrane_query_applicable(p) is True


def test_background_weak_and_not_query_applicable():
    p = _therapy_picott(); p["type_of_question"] = "background"
    assert cochrane_boost_tier(p) == "weak"
    assert cochrane_boost_points(p) == 2
    # weak lane: boost may still apply if a Cochrane review is otherwise
    # retrieved, but we do NOT actively emit a Cochrane query family.
    assert cochrane_query_applicable(p) is False


def test_guideline_and_cost_none():
    for t in ("guideline", "cost", "other"):
        p = _therapy_picott(); p["type_of_question"] = t
        assert cochrane_boost_tier(p) == "none"
        assert cochrane_boost_points(p) == 0
        assert cochrane_query_applicable(p) is False


def test_boost_is_bounded():
    # Hard upper bound across all lanes is the strong tier (8).
    for t in ("therapy", "prevention", "diagnosis", "prognosis",
              "etiology_harm", "background", "guideline", "cost", "other"):
        p = _therapy_picott(); p["type_of_question"] = t
        assert 0 <= cochrane_boost_points(p) <= 8


# --------------------------------------------------------------------------
# D) PubMed-only cochrane_review query family
# --------------------------------------------------------------------------
def test_cochrane_family_emitted_for_therapy_pubmed():
    out = boolean_query_compiler.compile(_therapy_picott(), sources=["pubmed"])
    fam = [q for q in out if q.query_family == "cochrane_review"]
    assert len(fam) == 1
    q = fam[0]
    assert q.source == "pubmed"
    assert '"Cochrane Database Syst Rev"[ta]' in q.query_string
    assert q.syntax_validation.valid is True
    assert "ta" in q.field_tags_used


def test_cochrane_family_not_emitted_for_europe_pmc():
    out = boolean_query_compiler.compile(_therapy_picott(), sources=["europe_pmc"])
    assert not [q for q in out if q.query_family == "cochrane_review"]


def test_cochrane_family_not_emitted_for_background():
    p = _therapy_picott(); p["type_of_question"] = "background"
    out = boolean_query_compiler.compile(p, sources=["pubmed"])
    assert not [q for q in out if q.query_family == "cochrane_review"]


def test_cochrane_family_not_emitted_without_anchors():
    p = _therapy_picott(); p["must_have_concepts"] = []
    out = boolean_query_compiler.compile(p, sources=["pubmed"])
    assert not [q for q in out if q.query_family == "cochrane_review"]


def test_compile_returns_empty_when_phi_blocked():
    out = boolean_query_compiler.compile(
        _therapy_picott(), sources=["pubmed"], is_phi_blocked=True,
    )
    assert out == []


# --------------------------------------------------------------------------
# E) Source-registry permission defaults
# --------------------------------------------------------------------------
def _cochrane_registry_entry() -> dict:
    from server import GUIDELINE_SOURCE_REGISTRY
    entries = [e for e in GUIDELINE_SOURCE_REGISTRY
               if e["source_name"] == "cochrane_pubmed_indexed"]
    assert len(entries) == 1, "cochrane_pubmed_indexed must be registered once"
    return entries[0]


def test_registry_entry_permissions():
    e = _cochrane_registry_entry()
    assert e["ai_ingestion_allowed"] is False        # HARD invariant
    assert e["retrieval_allowed"] is True            # via PubMed
    assert e["link_out_allowed"] is True
    assert e["display_snippet_allowed"] == "verify"
    assert e["requires_api_key"] is False
    assert e["connector_status"] == "live_pubmed_indexed"
    assert e["source_type"] == "systematic_review_bibliographic_lane"


def test_registry_entry_constructs_model():
    e = _cochrane_registry_entry()
    doc = GuidelineSourceDoc(**e).model_dump()
    assert doc["ai_ingestion_allowed"] is False


def test_all_registry_entries_ai_ingestion_false():
    from server import GUIDELINE_SOURCE_REGISTRY
    for e in GUIDELINE_SOURCE_REGISTRY:
        assert e["ai_ingestion_allowed"] is False, e["source_name"]


# --------------------------------------------------------------------------
# F) Ranker boost — bounded, cap-limited, directness-gated, lane-matched
# --------------------------------------------------------------------------
def test_directly_relevant_cochrane_review_gets_boost():
    p = _therapy_picott()
    body = "Anticoagulation reduces stroke in atrial fibrillation patients."
    cochrane = _ranked_row(
        ck="c1", title="Anticoagulants for atrial fibrillation",
        abstract=body, journal="Cochrane Database Syst Rev",
    )
    non_cochrane = _ranked_row(
        ck="n1", title="Anticoagulants for atrial fibrillation",
        abstract=body, journal="BMJ",
    )
    out = picott_aware_ranker.postprocess([cochrane, non_cochrane], p)
    c = next(r for r in out if r["_article"]["canonical_key"] == "c1")
    n = next(r for r in out if r["_article"]["canonical_key"] == "n1")
    assert c["cochrane_review_lane"] is True
    assert n["cochrane_review_lane"] is False
    assert any(b.startswith("cochrane_review_") for b in c["ranking_boost_reasons"])
    # Boosted Cochrane review ranks at or above the otherwise-identical
    # non-Cochrane article.
    assert c["overall_rank_score"] >= n["overall_rank_score"]


def test_wrong_disease_cochrane_review_gets_no_rescue():
    # Question is about atrial fibrillation; the Cochrane review is about a
    # different disease (no AF anchor in title/abstract) → missing condition
    # anchor cap → NO cochrane boost, even though it is a Cochrane review.
    p = _therapy_picott()
    cochrane = _ranked_row(
        ck="c2",
        title="Anticoagulation for venous thromboembolism",
        abstract="Anticoagulants in deep vein thrombosis patients.",
        journal="Cochrane Database Syst Rev",
        base=90,
    )
    out = picott_aware_ranker.postprocess([cochrane], p)
    c = out[0]
    # Labeled (badge still shows) but NOT boosted; cap dominates.
    assert c["cochrane_review_lane"] is True
    assert not any(
        b.startswith("cochrane_review_") for b in c["ranking_boost_reasons"]
    )
    assert c["cap_reasons"], "wrong-disease article must carry a cap reason"
    # The cap clamps the final score (caps dominate citation/SR status).
    assert c["overall_rank_score"] <= 50


def test_prognosis_cochrane_boost_is_moderate_not_strong():
    p = _therapy_picott()
    p["type_of_question"] = "prognosis"
    body = "Anticoagulation and atrial fibrillation prognosis outcomes."
    coch = _ranked_row(ck="c3", title="atrial fibrillation anticoagulation",
                       abstract=body, journal="Cochrane Database Syst Rev")
    out = picott_aware_ranker.postprocess([coch], p)
    reasons = out[0]["ranking_boost_reasons"]
    assert any("cochrane_review_moderate(+5)" == b for b in reasons), reasons


def test_capped_cochrane_review_stays_at_or_below_cap():
    # Even with the journal label, a capped article never exceeds its cap.
    p = _therapy_picott()
    coch = _ranked_row(
        ck="c4", title="A different topic entirely",
        abstract="Unrelated content with no anchors.",
        journal="Cochrane Database Syst Rev", base=100,
    )
    out = picott_aware_ranker.postprocess([coch], p)
    cap = out[0].get("relevance_cap_applied")
    if cap is not None:
        assert out[0]["overall_rank_score"] <= cap


def test_non_cochrane_article_has_no_cochrane_fields_set_true():
    p = _therapy_picott()
    body = "Anticoagulation in atrial fibrillation."
    row = _ranked_row(ck="n2", title="AF anticoagulation", abstract=body,
                      journal="Lancet")
    out = picott_aware_ranker.postprocess([row], p)
    assert out[0]["cochrane_review_lane"] is False
    assert out[0]["cochrane_review_provenance"] is None


# --------------------------------------------------------------------------
# G) Production gate — boost is shadow-only (not applied) in production
#    unless COCHRANE_PROD_BOOST_ALLOWED=true.
# --------------------------------------------------------------------------
def _coch_row():
    p = _therapy_picott()
    body = "Anticoagulation reduces stroke in atrial fibrillation patients."
    row = _ranked_row(ck="g1", title="Anticoagulants for atrial fibrillation",
                      abstract=body, journal="Cochrane Database Syst Rev")
    return p, row


def test_prod_boost_disabled_by_default(monkeypatch):
    from services import cochrane_lane
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("COCHRANE_PROD_BOOST_ALLOWED", raising=False)
    assert cochrane_lane.cochrane_boost_enabled() is False
    p, row = _coch_row()
    out = picott_aware_ranker.postprocess([row], p)
    r = out[0]
    # Detection/labeling still works; shadow value recorded; NOT applied.
    assert r["cochrane_review_lane"] is True
    assert r["cochrane_review_shadow_boost"] == 8
    assert not any(
        b.startswith("cochrane_review_strong(") for b in r["ranking_boost_reasons"]
    )
    assert any(
        b.startswith("cochrane_review_shadow_only") for b in r["ranking_boost_reasons"]
    )


def test_prod_boost_applied_when_flag_true(monkeypatch):
    from services import cochrane_lane
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("COCHRANE_PROD_BOOST_ALLOWED", "true")
    assert cochrane_lane.cochrane_boost_enabled() is True
    p, row = _coch_row()
    out = picott_aware_ranker.postprocess([row], p)
    r = out[0]
    assert r["cochrane_review_shadow_boost"] == 8
    assert any(
        b.startswith("cochrane_review_strong(+8)") for b in r["ranking_boost_reasons"]
    )


def test_non_prod_boost_active(monkeypatch):
    from services import cochrane_lane
    monkeypatch.setenv("APP_ENV", "development")
    assert cochrane_lane.cochrane_boost_enabled() is True
