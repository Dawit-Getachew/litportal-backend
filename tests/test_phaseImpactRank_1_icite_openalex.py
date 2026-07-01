"""Phase ImpactRank-1 — iCite/OpenAlex citation-impact enrichment tests.

Deterministic (NO network — iCite is mocked). Proves the bounded,
shadow-only impact boost respects clinical directness + safety caps +
lane relevance, never penalizes missing metrics, keeps wrong-lane
high-citation papers capped, and that the rank simulation moves
clinically-direct high-impact papers up WITHOUT any clinician-visible
change.
"""
from __future__ import annotations

import asyncio

import pytest

from services.impact_enrichment import (
    IMPACT_APPLIED,
    IMPACT_REASONS,
    NOT_APPLIED_NO_METRICS,
    NOT_APPLIED_NON_ARTICLE,
    NOT_APPLIED_NOT_DIRECT,
    NOT_APPLIED_POLICY_EXCLUDED,
    NOT_APPLIED_WRONG_DISEASE,
    NOT_APPLIED_WRONG_LANE,
    assess_boost_eligibility,
    compute_impact_factor,
    compute_impact_for_article,
    max_boost,
    run_impact_enrichment_shadow,
)


_HIGH_METRICS = {
    "relative_citation_ratio": 4.0, "nih_percentile": 95.0, "apt": 0.9,
    "cited_by_clin_count": 12, "citation_count": 600, "is_clinical": True,
    "provisional": False,
}
_LOW_METRICS = {
    "relative_citation_ratio": 0.2, "nih_percentile": 10.0, "apt": 0.05,
    "cited_by_clin_count": 0, "citation_count": 3, "provisional": False,
}


# ---------------------------------------------------------------------
# Group A — impact factor (deterministic, bounded).
# ---------------------------------------------------------------------
def test_factor_high_metrics_near_one():
    f = compute_impact_factor(_HIGH_METRICS)
    assert f["has_metrics"] is True
    assert 0.85 <= f["impact_factor"] <= 1.0


def test_factor_missing_metrics_zero_no_penalty():
    f = compute_impact_factor({})
    assert f["has_metrics"] is False
    assert f["impact_factor"] == 0.0


def test_factor_clamped_to_one():
    f = compute_impact_factor({
        "relative_citation_ratio": 50.0, "nih_percentile": 100.0,
        "apt": 1.0, "cited_by_clin_count": 100, "citation_count": 9999,
    })
    assert f["impact_factor"] == 1.0


# ---------------------------------------------------------------------
# Group B — eligibility: directness / caps / policy dominate.
# ---------------------------------------------------------------------
def test_policy_excluded_dominates():
    ok, reason = assess_boost_eligibility(
        article_doc={"is_retracted": True}, sa_row={}, r2a_label=None,
    )
    assert ok is False and reason == NOT_APPLIED_POLICY_EXCLUDED


def test_r2a_wrong_disease_blocks():
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={},
        r2a_label={"rescue_label": "direct_relevance_candidate", "relevance_reason": "wrong_disease"},
    )
    assert ok is False and reason == NOT_APPLIED_WRONG_DISEASE


def test_r2a_screening_lane_blocks():
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={},
        r2a_label={"rescue_label": "indirect_relevance_candidate", "relevance_reason": "screening_not_prognosis"},
    )
    assert ok is False and reason == NOT_APPLIED_WRONG_LANE


def test_r2a_direct_prognosis_eligible():
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={},
        r2a_label={"rescue_label": "direct_relevance_candidate", "relevance_reason": "stage_specific_prognosis"},
    )
    assert ok is True and reason == IMPACT_APPLIED


def test_cap_token_wrong_lane_blocks_active_article():
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={"cap_reasons": ["off_target_screening"]}, r2a_label=None,
    )
    assert ok is False and reason == NOT_APPLIED_WRONG_LANE


def test_no_disqualifier_direct_eligible():
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={"overall_rank": 88, "cap_reasons": []}, r2a_label=None,
    )
    assert ok is True and reason == IMPACT_APPLIED


# ---------------------------------------------------------------------
# Group C — per-article shadow boost is bounded + reason-tagged.
# ---------------------------------------------------------------------
def test_eligible_high_impact_bounded_boost():
    out = compute_impact_for_article(
        article_doc={"id": "a1", "pmid": "111", "citation_count": 600},
        sa_row={"article_id": "a1", "overall_rank": 80, "cap_reasons": []},
        icite_metrics=_HIGH_METRICS,
        r2a_label={"rescue_label": "direct_relevance_candidate", "relevance_reason": "stage_specific_prognosis"},
    )
    assert out["impact_applied"] is True
    assert 0 < out["impact_boost_simulated"] <= max_boost()
    assert out["impact_reason"] == IMPACT_APPLIED
    assert out["openalex_cited_by_count"] == 600


def test_eligible_but_no_metrics_zero_boost():
    out = compute_impact_for_article(
        article_doc={"id": "a2", "pmid": "222"},
        sa_row={"article_id": "a2", "overall_rank": 80},
        icite_metrics={},
        r2a_label={"rescue_label": "direct_relevance_candidate", "relevance_reason": "survival_outcome"},
    )
    assert out["impact_applied"] is False
    assert out["impact_boost_simulated"] == 0.0
    assert out["impact_reason"] == NOT_APPLIED_NO_METRICS


def test_wrong_lane_high_citation_not_boosted():
    out = compute_impact_for_article(
        article_doc={"id": "a3", "pmid": "333", "citation_count": 2000},
        sa_row={"article_id": "a3", "overall_rank": 70},
        icite_metrics=_HIGH_METRICS,
        r2a_label={"rescue_label": "indirect_relevance_candidate", "relevance_reason": "screening_not_prognosis"},
    )
    assert out["impact_applied"] is False
    assert out["impact_boost_simulated"] == 0.0
    assert out["impact_reason"] == NOT_APPLIED_WRONG_LANE


def test_all_reasons_closed_set():
    for r in (NOT_APPLIED_NON_ARTICLE, NOT_APPLIED_WRONG_DISEASE, IMPACT_APPLIED):
        assert r in IMPACT_REASONS


# ---------------------------------------------------------------------
# Group D — orchestrator (mocked iCite) — prostate stage III prognosis.
# ---------------------------------------------------------------------
class _FakeIcite:
    def __init__(self, metrics):
        self._m = metrics

    async def fetch_metrics(self, pmids, *, client=None, db=None):
        return {
            "metrics": {p: self._m[p] for p in pmids if p in self._m},
            "stats": {"requested": len(pmids), "from_cache": 0, "fetched": len(pmids), "batches": 1, "status": "ok", "error_message": None},
        }


def test_orchestrator_prostate_prognosis(monkeypatch):
    import services.connectors.icite as icite_mod

    metrics = {
        "1": _HIGH_METRICS,           # direct prognosis, high impact
        "2": _HIGH_METRICS,           # screening-only, high citation (must stay capped)
        "3": _HIGH_METRICS,           # wrong disease, high citation (must stay capped)
        "4": _LOW_METRICS,            # direct prognosis, low impact
    }
    monkeypatch.setattr(icite_mod, "icite_connector", _FakeIcite(metrics))

    articles_by_id = {
        "a1": {"id": "a1", "pmid": "1", "title": "Prognosis of stage III prostate cancer", "journal": "JCO", "citation_count": 500},
        "a2": {"id": "a2", "pmid": "2", "title": "PSA screening guidelines", "journal": "NEJM", "citation_count": 3000},
        "a3": {"id": "a3", "pmid": "3", "title": "Breast cancer survival", "journal": "Lancet", "citation_count": 2500},
        "a4": {"id": "a4", "pmid": "4", "title": "Recurrence after prostatectomy stage III", "journal": "Eur Urol"},
        "a5": {"id": "a5", "pmid": "5", "title": "Retracted prostate paper", "journal": "X", "is_retracted": True},
    }
    sa_rows = [
        {"article_id": "a1", "overall_rank": 70, "cap_reasons": []},
        {"article_id": "a2", "overall_rank": 80, "cap_reasons": []},
        {"article_id": "a3", "overall_rank": 62, "cap_reasons": []},
        {"article_id": "a4", "overall_rank": 75, "cap_reasons": []},
        {"article_id": "a5", "overall_rank": 40, "cap_reasons": ["retracted"]},
    ]
    r2a_labels = {
        "a1": {"rescue_label": "direct_relevance_candidate", "relevance_reason": "stage_specific_prognosis"},
        "a2": {"rescue_label": "indirect_relevance_candidate", "relevance_reason": "screening_not_prognosis"},
        "a3": {"rescue_label": "keep_rejected", "relevance_reason": "wrong_disease"},
        "a4": {"rescue_label": "direct_relevance_candidate", "relevance_reason": "recurrence_outcome"},
    }

    out = asyncio.run(run_impact_enrichment_shadow(
        sa_rows=sa_rows, articles_by_id=articles_by_id,
        r2a_candidates_by_article_id=r2a_labels, db=None,
    ))

    assert out["mode"] == "shadow"
    assert out["no_clinician_visible_change"] is True
    assert out["max_impact_boost"] == max_boost()

    by_id = {r["article_id"]: r for r in out["impact_enriched_articles"]}
    # Direct prognosis high-impact → boosted.
    assert by_id["a1"]["impact_applied"] is True and by_id["a1"]["impact_boost_simulated"] > 0
    # Screening-only high-citation → NOT boosted.
    assert by_id["a2"]["impact_applied"] is False and by_id["a2"]["impact_boost_simulated"] == 0.0
    assert by_id["a2"]["impact_reason"] == NOT_APPLIED_WRONG_LANE
    # Wrong-disease high-citation → NOT boosted.
    assert by_id["a3"]["impact_applied"] is False
    assert by_id["a3"]["impact_reason"] == NOT_APPLIED_WRONG_DISEASE
    # Retracted → policy excluded.
    assert by_id["a5"]["impact_reason"] == NOT_APPLIED_POLICY_EXCLUDED

    # Wrong-lane high-citation papers reported as remaining capped.
    capped_ids = {c["article_id"] for c in out["shadow_rank_simulation"]["wrong_lane_high_citation_remain_capped"]}
    assert {"a2", "a3"} <= capped_ids

    # Direct high-impact (a1) should move up vs current order in simulation.
    moved_ids = {m["article_id"] for m in out["shadow_rank_simulation"]["clinically_direct_high_impact_moved_up"]}
    assert "a1" in moved_ids


def test_orchestrator_empty_pool_safe():
    out = asyncio.run(run_impact_enrichment_shadow(
        sa_rows=[], articles_by_id={}, r2a_candidates_by_article_id={}, db=None,
    ))
    assert out["articles_total"] == 0
    assert out["boost_eligible_count"] == 0
    assert out["no_clinician_visible_change"] is True


@pytest.mark.parametrize("v,expected", [("8", 8.0), ("6", 6.0), ("7", 7.0), ("12", 8.0), ("2", 6.0)])
def test_max_boost_bounds(monkeypatch, v, expected):
    monkeypatch.setenv("IMPACTRANK_MAX_BOOST", v)
    assert max_boost() == expected


# ---------------------------------------------------------------------
# Group H — Phase ImpactRank-1.1 gate tuning.
#   (1) extended cap/rejection block tokens;
#   (2) "blocked unless a positive directness signal exists".
# ---------------------------------------------------------------------
@pytest.mark.parametrize("cap,expected", [
    ("missing_condition_anchor", NOT_APPLIED_WRONG_DISEASE),
    ("wrong_condition", NOT_APPLIED_WRONG_DISEASE),
    ("missing_primary_anchor", NOT_APPLIED_WRONG_DISEASE),
    ("missing_disease_anchor", NOT_APPLIED_WRONG_DISEASE),
    ("wrong_framework_lane", NOT_APPLIED_WRONG_LANE),
    ("wrong_population", NOT_APPLIED_WRONG_LANE),
    ("policy_excluded", NOT_APPLIED_POLICY_EXCLUDED),
    ("non_article_record", NOT_APPLIED_NON_ARTICLE),
])
def test_new_cap_tokens_block_even_with_positive_signal(cap, expected):
    # overall_rank=88 is a positive signal, yet a blocking cap dominates.
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={"overall_rank": 88, "cap_reasons": [cap]}, r2a_label=None,
    )
    assert ok is False and reason == expected


def test_flip_no_positive_signal_now_blocked():
    # Previously "eligible unless blocked"; now requires a positive signal.
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={"overall_rank": 30, "cap_reasons": []}, r2a_label=None,
    )
    assert ok is False and reason == NOT_APPLIED_NOT_DIRECT


def test_positive_signal_included_by_default():
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={"overall_rank": 30, "included_by_default": True, "cap_reasons": []}, r2a_label=None,
    )
    assert ok is True and reason == IMPACT_APPLIED


def test_positive_signal_score_at_floor():
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={"overall_rank": 60, "cap_reasons": []}, r2a_label=None,
    )
    assert ok is True and reason == IMPACT_APPLIED


def test_positive_signal_directness_label():
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={"overall_rank": 30, "directness_label": "direct", "cap_reasons": []}, r2a_label=None,
    )
    assert ok is True and reason == IMPACT_APPLIED


def test_positive_signal_lane_role_token():
    ok, reason = assess_boost_eligibility(
        article_doc={},
        sa_row={"overall_rank": 30, "cap_reasons": [],
                "subquestion_roles_matched": {"sq1": ["survival_outcome"]}},
        r2a_label=None,
    )
    assert ok is True and reason == IMPACT_APPLIED


def test_r2a_borderline_candidate_eligible():
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={"overall_rank": 30, "cap_reasons": []},
        r2a_label={"rescue_label": "borderline_relevance_candidate", "relevance_reason": "stage_specific_prognosis"},
    )
    assert ok is True and reason == IMPACT_APPLIED


def test_r2a_indirect_blocks_even_with_high_score():
    # An explicit R2A "indirect" verdict overrides the score>=floor proxy.
    ok, reason = assess_boost_eligibility(
        article_doc={}, sa_row={"overall_rank": 100, "cap_reasons": []},
        r2a_label={"rescue_label": "indirect_relevance_candidate", "relevance_reason": "related_topic"},
    )
    assert ok is False and reason == NOT_APPLIED_NOT_DIRECT


def test_compute_missing_anchor_high_citation_zero_boost():
    # off-disease / un-anchored high-citation paper must get boost 0.
    out = compute_impact_for_article(
        article_doc={"id": "x", "pmid": "9", "title": "IDSA COVID-19 guideline", "citation_count": 600},
        sa_row={"article_id": "x", "overall_rank": 30,
                "cap_reasons": ["missing_condition_anchor", "wrong_condition", "missing_primary_anchor"]},
        icite_metrics=_HIGH_METRICS,
        r2a_label=None,
    )
    assert out["impact_applied"] is False
    assert out["impact_boost_simulated"] == 0.0
    assert out["impact_reason"] == NOT_APPLIED_WRONG_DISEASE
