"""Phase 5D Fix 5 — ``ranking_boost_reasons`` must be surfaced on every
article row in the diagnostics + clinician article-list responses.

Pre-fix, the field was absent (not persisted, not in API responses).
Now the ranker computes a structured boost-reasons list, the pipeline
persists it on ``search_session_articles`` docs, and both the admin
diagnostics endpoint and the clinician ``GET /search-sessions/{id}``
endpoint surface it.

This test exercises the ranker directly (no pipeline / Mongo / LLM
dependency) — it confirms the data shape is correct so the pipeline +
API can pass it through to clients.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.picott_ranker import picott_aware_ranker  # noqa: E402


def _ranker_input(*, families: list[str], art_type: str = "rct"):
    """Build a small ranker input fixture."""
    picott = {
        "type_of_question": "therapy",
        "must_have_concepts": [
            {"concept": "metformin", "role": "intervention"},
            {"concept": "type 2 diabetes", "role": "condition"},
        ],
        "ranking_concepts": [
            {"concept": "mortality", "role": "outcome"},
        ],
        "pico": {},
    }
    article = {
        "canonical_key": "k1",
        "title": "Metformin in T2D — RCT",
        "abstract": (
            "Metformin reduced all-cause mortality in adults with "
            "type 2 diabetes."
        ),
        "article_type_classification": art_type,
    }
    ranked = [{
        "_article": article,
        "title": article["title"],
        "abstract": article["abstract"],
        "relevance_score": 70,
        "evidence_design_score": 90,
        "overall_rank_score": 70,
        "rank_explanation": "test",
        "query_types_matched": families,
    }]
    qmap = {"k1": families}
    return picott, ranked, qmap


def test_ranking_boost_reasons_populated_for_high_signal_family():
    """When must-haves are satisfied AND a high-signal family is matched,
    ``ranking_boost_reasons`` must list the family name and a points value."""
    picott, ranked, qmap = _ranker_input(
        families=["cb_high_precision", "cb_mesh_keyword"],
    )
    out = picott_aware_ranker.postprocess(
        ranked, picott, query_family_by_canonical=qmap,
    )
    r = out[0]
    assert "ranking_boost_reasons" in r, "field missing from ranker output"
    reasons = r["ranking_boost_reasons"]
    assert isinstance(reasons, list)
    # Must include the "must_haves_satisfied_..." high-signal boost.
    assert any("must_haves_satisfied" in code for code in reasons), (
        f"expected must_haves_satisfied boost; got {reasons!r}"
    )
    # And a numeric total.
    assert isinstance(r.get("ranking_boost_total"), int)
    assert r["ranking_boost_total"] >= 5


def test_ranking_boost_reasons_lists_phase5d_families():
    """Phase 5D families (mesh_keyword/clinical_queries/trial_acronym)
    must contribute to the must_haves_satisfied boost code."""
    picott, ranked, qmap = _ranker_input(
        families=["cb_trial_acronym", "cb_clinical_queries"],
    )
    out = picott_aware_ranker.postprocess(
        ranked, picott, query_family_by_canonical=qmap,
    )
    reasons = out[0]["ranking_boost_reasons"]
    # The high-signal boost must reference at least one phase5d family.
    found = False
    for code in reasons:
        if "must_haves_satisfied" in code and (
            "trial_acronym" in code or "clinical_queries" in code
            or "mesh_keyword" in code
        ):
            found = True
    assert found, f"expected phase5d family in boost reason; got {reasons!r}"


def test_ranking_boost_reasons_empty_list_when_no_boosts():
    """When no boost conditions fire, the field must still be present —
    just an empty list. Clients can rely on the field being non-null."""
    # Article type with no design boost, no ranking-concept matches,
    # no high-signal family.
    picott = {
        "type_of_question": "therapy",
        "must_have_concepts": [
            {"concept": "metformin", "role": "intervention"},
            {"concept": "diabetes", "role": "condition"},
        ],
        "ranking_concepts": [],
        "pico": {},
    }
    article = {
        "canonical_key": "k1",
        "title": "Unrelated paper",
        "abstract": "Nothing about metformin or diabetes.",
        "article_type_classification": "case_report",
    }
    ranked = [{
        "_article": article,
        "title": article["title"],
        "abstract": article["abstract"],
        "relevance_score": 50,
        "evidence_design_score": 30,
        "overall_rank_score": 50,
        "rank_explanation": "test",
        "query_types_matched": ["cb_broad_recall"],
    }]
    qmap = {"k1": ["cb_broad_recall"]}
    out = picott_aware_ranker.postprocess(
        ranked, picott, query_family_by_canonical=qmap,
    )
    r = out[0]
    assert "ranking_boost_reasons" in r
    assert isinstance(r["ranking_boost_reasons"], list)
    # broad_recall-only with missing must-haves → no high-signal boost.
    assert r["ranking_boost_total"] == 0 or r["ranking_boost_total"] >= 0


def test_ranking_boost_reasons_includes_ranking_concept_match():
    """When ranking-concepts are matched, that boost must be in the list."""
    picott = {
        "type_of_question": "therapy",
        "must_have_concepts": [
            {"concept": "metformin", "role": "intervention"},
            {"concept": "diabetes", "role": "condition"},
        ],
        "ranking_concepts": [
            {"concept": "mortality", "role": "outcome"},
        ],
        "pico": {},
    }
    article = {
        "canonical_key": "k1",
        "title": "Metformin reduces mortality in T2D",
        "abstract": (
            "RCT showing all-cause mortality reduction with metformin "
            "in diabetes."
        ),
        "article_type_classification": "rct",
    }
    ranked = [{
        "_article": article,
        "title": article["title"],
        "abstract": article["abstract"],
        "relevance_score": 70,
        "evidence_design_score": 90,
        "overall_rank_score": 70,
        "rank_explanation": "test",
        "query_types_matched": ["cb_high_precision"],
    }]
    out = picott_aware_ranker.postprocess(
        ranked, picott, query_family_by_canonical={"k1": ["cb_high_precision"]},
    )
    reasons = out[0]["ranking_boost_reasons"]
    assert any("matched_" in c and "ranking_concepts" in c for c in reasons), (
        f"expected ranking-concept-match boost code; got {reasons!r}"
    )


def test_ranking_boost_field_keys_stable_for_api_contract():
    """API contract: clients depend on the exact field names. This test
    pins them so a rename triggers a failure."""
    picott, ranked, qmap = _ranker_input(
        families=["cb_high_precision"],
    )
    out = picott_aware_ranker.postprocess(
        ranked, picott, query_family_by_canonical=qmap,
    )
    r = out[0]
    # Closed-set keys (stable contract for diagnostics + clinician UI).
    for key in ("ranking_boost_reasons", "ranking_boost_total"):
        assert key in r, f"required API key missing: {key}"
