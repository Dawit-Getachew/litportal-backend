"""Phase 2B — cross-source deduplication, multi-source planner, evaluator."""
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
from services.retrieval_quality_evaluator import RetrievalQualityEvaluatorService  # noqa: E402
from services.search_planning import SearchPlanningService  # noqa: E402


def _parsed():
    return {
        "normalized_question": "Empagliflozin vs placebo in HFpEF",
        "clinical_question_type": "treatment",
        "specialty": "cardiology",
        "population": "patients with HFpEF",
        "intervention_or_exposure": "empagliflozin",
        "comparator": "placebo",
        "outcomes": ["mortality"],
        "setting": "outpatient",
        "time_horizon": "",
        "key_concepts": ["empagliflozin", "hfpef"],
        "mesh_terms": [],
        "synonyms": ["heart failure with preserved ejection fraction"],
        "drug_names_and_classes": ["empagliflozin", "SGLT2 inhibitors"],
        "exclusions": [],
        "rationale": "source=deterministic_fallback",
    }


# ---------- Multi-source planner ----------
def test_planner_emits_eight_plans_for_two_sources():
    plans = SearchPlanningService().plan(_parsed(), sources=["europe_pmc", "pubmed"])
    by_src = {}
    for p in plans:
        by_src.setdefault(p.source, []).append(p.query_type)
    assert set(by_src["europe_pmc"]) == {"high_recall", "high_precision", "evidence_filtered", "guideline_focused"}
    assert set(by_src["pubmed"]) == {"high_recall", "high_precision", "evidence_filtered", "guideline_focused"}


def test_pubmed_plans_use_pubmed_syntax_and_exclude_preprints():
    plans = SearchPlanningService().plan(_parsed(), sources=["pubmed"])
    for p in plans:
        assert p.source == "pubmed"
        # NCBI [tiab] field tag must be used.
        assert "[tiab]" in p.query_string
        # No Europe PMC SRC:PPR token.
        assert "SRC:PPR" not in p.query_string
        # Preprints excluded via Publication Type.
        assert '"preprint"[Publication Type]' in p.query_string


def test_pubmed_evidence_filtered_uses_publication_type_for_qtype():
    parsed = {**_parsed(), "clinical_question_type": "treatment"}
    plans = SearchPlanningService().plan(parsed, sources=["pubmed"])
    ef = next(p for p in plans if p.query_type == "evidence_filtered")
    assert '"randomized controlled trial"[Publication Type]' in ef.query_string

    parsed_dx = {**_parsed(), "clinical_question_type": "diagnosis"}
    ef_dx = next(p for p in SearchPlanningService().plan(parsed_dx, sources=["pubmed"]) if p.query_type == "evidence_filtered")
    # Diagnosis qtype prefers SR/MA pub types.
    assert '"systematic review"[Publication Type]' in ef_dx.query_string


def test_default_sources_is_europe_pmc_only_for_backwards_compat():
    plans = SearchPlanningService().plan(_parsed())
    assert all(p.source == "europe_pmc" for p in plans)
    assert len(plans) == 4


# ---------- Cross-source deduplication ----------
def _raw_med(**overrides):
    base = {
        "id": "MED123",
        "source": "MED",
        "title": "Empagliflozin in HFpEF",
        "abstractText": "Short abstract from Europe PMC.",
        "journalInfo": {"journal": {"title": "NEJM"}, "yearOfPublication": "2021"},
        "pubYear": "2021",
        "doi": "10.1056/NEJMoa2107038",
        "pmid": "34449189",
        "pmcid": "PMC8500551",
        "pubTypeList": {"pubType": ["Journal Article", "Randomized Controlled Trial"]},
        "authorList": {"author": [{"fullName": "Anker SD", "lastName": "Anker"}]},
        "fullTextUrlList": {"fullTextUrl": []},
    }
    base.update(overrides)
    return base


def test_dedup_merges_same_pmid_across_two_sources():
    """Europe PMC and PubMed both return the same article (same PMID) — must collapse."""
    epmc = MetadataNormalizerService().normalize(_raw_med())
    epmc["sources_present"] = ["europe_pmc"]
    # Simulate PubMed-normalized article (slightly different abstract).
    pubmed_art = dict(epmc)
    pubmed_art["abstract"] = "Much longer detailed PubMed abstract with extra information about empagliflozin trial."
    pubmed_art["sources_present"] = ["pubmed"]
    pubmed_art["source_ids_json"] = {"pubmed": "34449189", "doi": "10.1056/NEJMoa2107038"}

    unique, qmap = DeduplicationService().dedupe([
        ("high_recall", epmc), ("high_precision", pubmed_art),
    ])
    assert len(unique) == 1
    assert set(unique[0]["sources_present"]) == {"europe_pmc", "pubmed"}
    # Longer abstract wins.
    assert "longer detailed" in unique[0]["abstract"]
    assert set(qmap[unique[0]["canonical_key"]]) == {"high_recall", "high_precision"}


def test_dedup_merges_across_canonical_keys_by_doi():
    """Two records sharing a DOI but keyed differently must collapse."""
    # Record A: has BOTH PMID and DOI (keyed by PMID).
    a = MetadataNormalizerService().normalize(_raw_med(pmid="34449189", pmcid="", doi="10.1056/NEJMoa2107038"))
    a["sources_present"] = ["europe_pmc"]
    assert a["canonical_key"] == "pmid:34449189"
    # Record B: same DOI, NO PMID → keyed by DOI.
    b = MetadataNormalizerService().normalize(_raw_med(pmid="", pmcid="", doi="10.1056/NEJMoa2107038"))
    b["sources_present"] = ["openalex"]
    assert b["canonical_key"].startswith("doi:")
    unique, qmap = DeduplicationService().dedupe([
        ("high_recall", a), ("high_precision", b),
    ])
    assert len(unique) == 1
    assert set(unique[0]["sources_present"]) == {"europe_pmc", "openalex"}


# ---------- Multi-source ranker boost ----------
def test_ranker_applies_multisource_boost():
    rct = MetadataNormalizerService().normalize(_raw_med(
        pmid="r1", abstractText="Empagliflozin vs placebo in HFpEF for mortality."
    ))
    rct["sources_present"] = ["europe_pmc", "pubmed"]  # 2 sources → +3
    single = MetadataNormalizerService().normalize(_raw_med(
        pmid="s1", abstractText="Empagliflozin vs placebo in HFpEF for mortality."
    ))
    single["sources_present"] = ["europe_pmc"]
    qmap = {"pmid:r1": ["high_recall"], "pmid:s1": ["high_recall"]}
    ranked = EvidenceRankerService().rank(_parsed(), [single, rct], qmap)
    top = next(r for r in ranked if not r.get("exclusion_reason"))
    # Boosted record should be top.
    assert top["_article"]["canonical_key"] == "pmid:r1"
    assert "Cross-source confirmation" in top["rank_explanation"]


# ---------- Evaluator source_coverage ----------
def test_evaluator_emits_source_coverage_block():
    parsed = _parsed()
    ranked_items = [
        {
            "_article": {
                "is_guideline": False, "publication_year": 2022,
                "sources_present": ["europe_pmc", "pubmed"],
            },
            "overall_rank_score": 85,
        },
        {
            "_article": {
                "is_guideline": False, "publication_year": 2021,
                "sources_present": ["europe_pmc"],
            },
            "overall_rank_score": 70,
        },
    ]
    out = RetrievalQualityEvaluatorService().evaluate(
        parsed, [], ranked_items,
        raw_total=30,
        per_source_counts={"europe_pmc": 20, "pubmed": 10},
        sources_searched=["europe_pmc", "pubmed"],
    )
    sc = out["source_coverage"]
    assert sc["sources_searched"] == ["europe_pmc", "pubmed"]
    assert sc["per_source_raw_counts"] == {"europe_pmc": 20, "pubmed": 10}
    assert sc["cross_source_confirmed_articles"] == 1
    assert out["counts"]["cross_source_confirmed"] == 1


def test_evaluator_flags_single_source_when_multi_searched():
    parsed = _parsed()
    ranked = [
        {"_article": {"sources_present": ["europe_pmc"]}, "overall_rank_score": 70},
    ]
    out = RetrievalQualityEvaluatorService().evaluate(
        parsed, [], ranked,
        raw_total=10,
        per_source_counts={"europe_pmc": 10, "pubmed": 0},
        sources_searched=["europe_pmc", "pubmed"],
    )
    assert "single_source_only" in out["problems"]
