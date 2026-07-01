"""Phase 2A — search planner + safety filter + dedupe + ranker tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.deduplication import DeduplicationService  # noqa: E402
from services.evidence_ranker import EvidenceRankerService  # noqa: E402
from services.metadata_normalizer import MetadataNormalizerService  # noqa: E402
from services.safety_filter import SafetyFilterService  # noqa: E402
from services.search_planning import SearchPlanningService  # noqa: E402


# ---------------- planner ----------------
def _parsed_treatment():
    return {
        "normalized_question": "Empagliflozin vs placebo in HFpEF",
        "clinical_question_type": "treatment",
        "specialty": "cardiology",
        "population": "patients with HFpEF",
        "intervention_or_exposure": "empagliflozin",
        "comparator": "placebo",
        "outcomes": ["mortality", "hospitalization"],
        "setting": "outpatient",
        "time_horizon": "",
        "key_concepts": ["empagliflozin", "hfpef", "heart failure"],
        "mesh_terms": [],
        "synonyms": ["heart failure with preserved ejection fraction"],
        "drug_names_and_classes": ["empagliflozin", "sglt2_inhibitor"],
        "exclusions": [],
        "rationale": "source=deterministic_fallback",
    }


def test_planner_emits_4_variants_all_excluding_preprints():
    plans = SearchPlanningService().plan(_parsed_treatment())
    qtypes = [p.query_type for p in plans]
    assert qtypes == ["high_recall", "high_precision", "evidence_filtered", "guideline_focused"]
    for p in plans:
        assert "NOT SRC:PPR" in p.query_string


def test_evidence_filtered_changes_pubtypes_with_qtype():
    treatment_plans = SearchPlanningService().plan(_parsed_treatment())
    ef_t = next(p for p in treatment_plans if p.query_type == "evidence_filtered").query_string
    assert "randomized controlled trial" in ef_t

    parsed_dx = {**_parsed_treatment(), "clinical_question_type": "diagnosis"}
    ef_dx = next(
        p for p in SearchPlanningService().plan(parsed_dx) if p.query_type == "evidence_filtered"
    ).query_string
    assert "diagnostic accuracy" in ef_dx


def test_guideline_focused_contains_guideline_pubtypes():
    plans = SearchPlanningService().plan(_parsed_treatment())
    gf = next(p for p in plans if p.query_type == "guideline_focused").query_string
    assert 'PUB_TYPE:"practice guideline"' in gf or 'PUB_TYPE:"guideline"' in gf


# ---------------- normalizer ----------------
def _raw_med(**overrides):
    base = {
        "id": "MED123",
        "source": "MED",
        "title": "Empagliflozin in patients with heart failure with preserved ejection fraction",
        "abstractText": "RCT of empagliflozin vs placebo in HFpEF.",
        "journalInfo": {"journal": {"title": "NEJM"}, "yearOfPublication": "2021"},
        "pubYear": "2021",
        "firstPublicationDate": "2021-08-27",
        "doi": "10.1056/NEJMoa2107038",
        "pmid": "34449189",
        "pmcid": "PMC8500551",
        "pubTypeList": {"pubType": ["Journal Article", "Randomized Controlled Trial"]},
        "authorList": {"author": [{"fullName": "Anker SD", "lastName": "Anker"}]},
        "fullTextUrlList": {"fullTextUrl": [{"url": "https://example.org/x", "availabilityCode": "OA"}]},
        "inEPMC": "Y",
    }
    base.update(overrides)
    return base


def test_normalizer_detects_rct_and_canonical_key():
    norm = MetadataNormalizerService().normalize(_raw_med())
    assert norm["article_type_classification"] == "rct"
    assert norm["is_preprint"] is False
    assert norm["is_guideline"] is False
    assert norm["full_text_status"] == "full_text_available"
    assert norm["canonical_key"] == "pmid:34449189"


def test_normalizer_detects_preprint_via_source_ppr():
    norm = MetadataNormalizerService().normalize(_raw_med(source="PPR", pmid="", pmcid=""))
    assert norm["is_preprint"] is True


def test_normalizer_detects_guideline():
    norm = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Practice Guideline"]}
    ))
    assert norm["is_guideline"] is True
    assert norm["article_type_classification"] == "guideline"


def test_normalizer_detects_retraction_and_eoc():
    retracted = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Retracted Publication"]}
    ))
    assert retracted["is_retracted"] is True
    eoc = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Expression of Concern", "Journal Article"]}
    ))
    assert eoc["has_expression_of_concern"] is True
    notice = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Retraction of Publication"]}
    ))
    assert notice["is_retraction_notice"] is True


# ---------------- dedupe ----------------
def test_dedupe_by_pmid_merges_query_types():
    a = MetadataNormalizerService().normalize(_raw_med())
    b = MetadataNormalizerService().normalize(_raw_med())  # same pmid
    unique, qmap = DeduplicationService().dedupe([("high_recall", a), ("high_precision", b)])
    assert len(unique) == 1
    assert set(qmap[unique[0]["canonical_key"]]) == {"high_recall", "high_precision"}


def test_dedupe_prefers_longer_abstract():
    short = MetadataNormalizerService().normalize(_raw_med(abstractText="short"))
    longer = MetadataNormalizerService().normalize(_raw_med(abstractText="A much longer detailed abstract."))
    unique, _ = DeduplicationService().dedupe([
        ("high_recall", short), ("high_precision", longer),
    ])
    assert unique[0]["abstract"] == "A much longer detailed abstract."


# ---------------- safety filter ----------------
def test_safety_filter_excludes_preprints_and_retractions():
    preprint = MetadataNormalizerService().normalize(_raw_med(source="PPR", pmid="", pmcid=""))
    retracted = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Retracted Publication"]}, pmid="22222"))
    notice = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Retraction of Publication"]}, pmid="33333"))
    ok = MetadataNormalizerService().normalize(_raw_med(pmid="44444"))
    kept, excluded = SafetyFilterService().pre_rank_filter([preprint, retracted, notice, ok])
    assert {e["exclusion_reason"] for e in excluded} == {
        "preprint_excluded_by_policy",
        "retracted_publication",
        "retraction_notice",
    }
    assert len(kept) == 1
    assert kept[0]["canonical_key"] == "pmid:44444"


def test_safety_filter_flags_eoc_but_keeps():
    eoc = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Expression of Concern", "Journal Article"]},
        pmid="55555",
    ))
    kept, excluded = SafetyFilterService().pre_rank_filter([eoc])
    assert not excluded
    assert "expression_of_concern" in (kept[0].get("flags") or [])


# ---------------- ranker ----------------
def test_ranker_prefers_design_appropriate_to_qtype():
    rct = MetadataNormalizerService().normalize(_raw_med(pmid="111"))
    obs = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Cohort Studies"]}, pmid="222"))
    guideline = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Practice Guideline"]}, pmid="333"))
    qmap = {
        "pmid:111": ["high_precision"],
        "pmid:222": ["high_recall"],
        "pmid:333": ["guideline_focused", "high_precision"],
    }
    parsed = _parsed_treatment()
    ranked = EvidenceRankerService().rank(parsed, [obs, rct, guideline], qmap)
    # Guideline should be top for treatment qtype (95) ahead of RCT (80) and cohort (60).
    top_keys = [r["_article"]["canonical_key"] for r in ranked if not r.get("exclusion_reason")][:3]
    assert top_keys[0] == "pmid:333"


def test_ranker_diagnosis_prefers_diagnostic_accuracy():
    dx_study = MetadataNormalizerService().normalize(_raw_med(
        pubTypeList={"pubType": ["Diagnostic Accuracy"]}, pmid="dx1"))
    rct = MetadataNormalizerService().normalize(_raw_med(pmid="rct1"))
    parsed = {**_parsed_treatment(), "clinical_question_type": "diagnosis"}
    qmap = {"pmid:dx1": ["high_precision"], "pmid:rct1": ["high_recall"]}
    ranked = EvidenceRankerService().rank(parsed, [rct, dx_study], qmap)
    top = next(r for r in ranked if not r.get("exclusion_reason"))
    assert top["_article"]["canonical_key"] == "pmid:dx1"


def test_ranker_low_relevance_gets_excluded():
    # Irrelevant title/abstract → low relevance → excluded
    irrelevant = MetadataNormalizerService().normalize(_raw_med(
        title="Knee osteoarthritis surgical outcomes",
        abstractText="Unrelated content.",
        pmid="irr1",
    ))
    ranked = EvidenceRankerService().rank(_parsed_treatment(), [irrelevant], {"pmid:irr1": ["high_recall"]})
    assert ranked[0]["exclusion_reason"] == "low_relevance"
