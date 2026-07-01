"""Phase 5D.1 — default-selection module + override endpoint tests.

Test Groups D + E from the spec:
  D. Default selection: capped articles not default-selected; CKD high-Hb-
     target NOT default-selected for AMI; direct AMI guideline/SR/RCT CAN
     be default-selected; SAH-without-milrinone NOT default-selected;
     direct milrinone/SAH CAN be default-selected.
  E. Evidence packet override: capped articles need explicit override;
     warning string surfaced verbatim; override metadata persisted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.default_selection import (  # noqa: E402
    EXCLUSION_REASONS,
    select_default_articles,
)


# =============================================================================
# Fixtures
# =============================================================================

def _ami_picott() -> dict:
    return {
        "type_of_question": "therapy",
        "question_subtype": "transfusion_threshold_comparison",
        "specialty": "cardiology",
        "must_have_concepts": [
            {"concept": "acute myocardial infarction", "role": "condition",
             "synonyms": ["AMI", "acute coronary syndrome", "ACS"]},
            {"concept": "anemia", "role": "condition",
             "synonyms": ["low hemoglobin", "low hematocrit"]},
            {"concept": "transfusion threshold", "role": "intervention",
             "synonyms": ["liberal transfusion", "restrictive transfusion"]},
        ],
        "ranking_concepts": [{"concept": "mortality", "role": "outcome"}],
    }


def _milrinone_picott() -> dict:
    return {
        "type_of_question": "dosing_protocol",
        "specialty": "neurology",
        "must_have_concepts": [
            {"concept": "milrinone", "role": "intervention"},
            {"concept": "subarachnoid hemorrhage", "role": "condition",
             "synonyms": ["SAH", "aSAH"]},
        ],
        "ranking_concepts": [
            {"concept": "cerebral vasospasm", "role": "outcome"},
        ],
    }


def _make_article(
    *,
    aid: str,
    title: str,
    abstract: str = "",
    art_type: str = "rct",
    is_guideline: bool = False,
    is_retracted: bool = False,
    is_preprint: bool = False,
    has_eoc: bool = False,
) -> dict:
    return {
        "id": aid,
        "title": title,
        "abstract": abstract or title,
        "article_type_classification": art_type,
        "is_guideline": is_guideline,
        "is_retracted": is_retracted,
        "is_preprint": is_preprint,
        "has_expression_of_concern": has_eoc,
    }


def _rank_row(
    article: dict,
    *,
    relevance_score: int = 75,
    overall_rank_score: int = 75,
    cap_reasons: list[str] | None = None,
    penalty_concepts: list[str] | None = None,
) -> dict:
    return {
        "_article": article,
        "title": article["title"],
        "abstract": article["abstract"],
        "relevance_score": relevance_score,
        "overall_rank_score": overall_rank_score,
        "cap_reasons": cap_reasons or [],
        "penalty_concepts": penalty_concepts or [],
    }


# =============================================================================
# D. Default selection — AMI session
# =============================================================================

class TestGroupD_AMIDefaultSelection:

    def _ami_ranked_articles(self) -> list[dict]:
        """Mix of:
          - a1: AABB transfusion guideline (should be selected)
          - a2: MINT trial (should be selected — RCT)
          - a3: REALITY trial (should be selected — RCT)
          - a4: large SR on transfusion thresholds (should be selected)
          - a5: CKD high-hemoglobin-target paper (MUST be EXCLUDED:
                missing AMI condition anchor — capped)
          - a6: AMI without anemia or transfusion (MUST be EXCLUDED:
                missing intervention anchor)
          - a7: low-relevance paper (relevance < 60)
          - a8: retracted paper (safety excluded)
        """
        return [
            _rank_row(
                _make_article(
                    aid="a1",
                    title="AABB clinical practice guidelines on red cell transfusion thresholds in patients with acute myocardial infarction and anemia",
                    abstract="Guideline panel review of liberal vs restrictive transfusion strategies in AMI patients with anemia. Recommends restrictive threshold for hemodynamically stable patients.",
                    art_type="guideline",
                    is_guideline=True,
                ),
                relevance_score=95, overall_rank_score=95,
            ),
            _rank_row(
                _make_article(
                    aid="a2",
                    title="The MINT trial: liberal versus restrictive red cell transfusion in acute myocardial infarction with anemia",
                    abstract="Randomized trial of 3504 adults with acute myocardial infarction and anemia. Liberal vs restrictive transfusion thresholds. Primary outcome: death or recurrent MI at 30 days.",
                    art_type="rct",
                ),
                relevance_score=92, overall_rank_score=92,
            ),
            _rank_row(
                _make_article(
                    aid="a3",
                    title="REALITY trial: restrictive transfusion strategy in patients with acute myocardial infarction and anemia",
                    abstract="Open-label RCT comparing liberal vs restrictive transfusion in 668 patients with acute myocardial infarction and anemia.",
                    art_type="rct",
                ),
                relevance_score=90, overall_rank_score=90,
            ),
            _rank_row(
                _make_article(
                    aid="a4",
                    title="Systematic review and meta-analysis of red cell transfusion thresholds in acute coronary syndrome with anemia",
                    abstract="Pooled analysis of MINT, REALITY, CRIT, TRACS, MINT-pilot. Restrictive strategy non-inferior for mortality and recurrent MI in acute myocardial infarction.",
                    art_type="systematic_review",
                ),
                relevance_score=89, overall_rank_score=89,
            ),
            _rank_row(
                # CKD high-Hb-target paper — does NOT mention AMI/MI/ACS.
                _make_article(
                    aid="a5",
                    title="Higher hemoglobin targets with erythropoietin in chronic kidney disease patients on dialysis",
                    abstract="RCT of erythropoietin dosing to high vs low hemoglobin targets in 1432 CKD patients on hemodialysis. Outcomes: cardiovascular mortality.",
                    art_type="rct",
                ),
                relevance_score=72, overall_rank_score=42,
                cap_reasons=["missing_condition_anchor"],
            ),
            _rank_row(
                # AMI without anemia or transfusion focus.
                _make_article(
                    aid="a6",
                    title="Acute myocardial infarction in elderly adults: outcomes review",
                    abstract="Review of long-term outcomes after acute myocardial infarction in patients over 75. No mention of transfusion or hemoglobin.",
                    art_type="review",
                ),
                relevance_score=65, overall_rank_score=35,
                cap_reasons=["missing_intervention_anchor"],
            ),
            _rank_row(
                _make_article(aid="a7", title="Unrelated paper about hypertension"),
                relevance_score=55, overall_rank_score=55,
            ),
            _rank_row(
                _make_article(
                    aid="a8",
                    title="[RETRACTED] Transfusion in AMI with anemia",
                    abstract="Transfusion strategies in acute myocardial infarction.",
                    is_retracted=True,
                ),
                relevance_score=85, overall_rank_score=85,
            ),
        ]

    def test_aabb_guideline_default_selected(self):
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        assert "a1" in out["selected_article_ids"]

    def test_mint_trial_default_selected(self):
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        assert "a2" in out["selected_article_ids"]

    def test_reality_trial_default_selected(self):
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        assert "a3" in out["selected_article_ids"]

    def test_sr_default_selected(self):
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        assert "a4" in out["selected_article_ids"]

    def test_ckd_high_Hb_NOT_default_selected(self):
        """CKD high-Hb-target paper must NOT be default-selected for AMI
        session — it's capped for missing the AMI condition anchor."""
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        assert "a5" not in out["selected_article_ids"], (
            "CKD article (capped for missing_condition_anchor) leaked into "
            "AMI default selection"
        )

    def test_AMI_no_transfusion_NOT_default_selected(self):
        """AMI review without transfusion is capped for missing
        intervention anchor."""
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        assert "a6" not in out["selected_article_ids"]

    def test_low_relevance_NOT_default_selected(self):
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        assert "a7" not in out["selected_article_ids"]

    def test_retracted_NOT_default_selected(self):
        """Retracted articles excluded via safety filter (Phase 1)."""
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        assert "a8" not in out["selected_article_ids"]

    def test_max_5_default_selected(self):
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        assert len(out["selected_article_ids"]) <= 5

    def test_excluded_summary_closed_set_keys(self):
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        for reason in out["excluded_rationale_summary"]:
            assert reason in EXCLUSION_REASONS, (
                f"non-closed-set reason in summary: {reason!r}"
            )

    def test_excluded_count_matches_summary(self):
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        total = sum(out["excluded_rationale_summary"].values())
        assert total == out["excluded_count"]

    def test_tier_order_guideline_first(self):
        """The guideline should appear before the SRs/RCTs in the
        selection rationale (tier 1 → tier 2 → tier 3)."""
        out = select_default_articles(_ami_picott(), self._ami_ranked_articles())
        ids = out["selected_article_ids"]
        # a1 is guideline; should be at index 0.
        assert ids.index("a1") < ids.index("a4")  # guideline before SR
        assert ids.index("a4") < ids.index("a2")  # SR before RCT (tier 2 < 3)


# =============================================================================
# D. Default selection — Milrinone/SAH
# =============================================================================

class TestGroupD_MilrinoneDefaultSelection:

    def _milrinone_ranked_articles(self) -> list[dict]:
        return [
            _rank_row(
                _make_article(
                    aid="m1",
                    title="Systematic review of milrinone for cerebral vasospasm after subarachnoid hemorrhage",
                    abstract="Pooled analysis of intra-arterial and IV milrinone for aSAH-related vasospasm and DCI prevention.",
                    art_type="systematic_review",
                ),
                relevance_score=92, overall_rank_score=92,
            ),
            _rank_row(
                _make_article(
                    aid="m2",
                    title="Intra-arterial milrinone protocol for cerebral vasospasm in aneurysmal subarachnoid hemorrhage",
                    abstract="Protocol paper describing milrinone dosing for cerebral vasospasm in aSAH.",
                    art_type="protocol_paper",
                ),
                relevance_score=88, overall_rank_score=88,
            ),
            _rank_row(
                _make_article(
                    aid="m3",
                    title="Milrinone for cerebral vasospasm: prospective cohort in aSAH ICU patients",
                    abstract="Cohort study of milrinone for aSAH vasospasm. Mortality and DCI outcomes.",
                    art_type="cohort_study",
                ),
                relevance_score=82, overall_rank_score=82,
            ),
            _rank_row(
                # SAH imaging review WITHOUT milrinone — capped.
                _make_article(
                    aid="m4",
                    title="Imaging modalities for cerebral vasospasm following subarachnoid hemorrhage",
                    abstract="CT angiography, MR angiography, and DSA for vasospasm in aSAH. No pharmacotherapy discussed.",
                    art_type="review",
                ),
                relevance_score=68, overall_rank_score=30,
                cap_reasons=["missing_intervention_anchor"],
            ),
            _rank_row(
                # Milrinone in HF — wrong condition.
                _make_article(
                    aid="m5",
                    title="Milrinone in chronic heart failure: efficacy and safety",
                    abstract="IV milrinone in acute decompensated HF. No SAH.",
                    art_type="rct",
                ),
                relevance_score=70, overall_rank_score=30,
                cap_reasons=["missing_condition_anchor"],
            ),
        ]

    def test_milrinone_sr_default_selected(self):
        out = select_default_articles(_milrinone_picott(), self._milrinone_ranked_articles())
        assert "m1" in out["selected_article_ids"]

    def test_milrinone_protocol_default_selected(self):
        out = select_default_articles(_milrinone_picott(), self._milrinone_ranked_articles())
        assert "m2" in out["selected_article_ids"]

    def test_SAH_imaging_without_milrinone_NOT_default_selected(self):
        out = select_default_articles(_milrinone_picott(), self._milrinone_ranked_articles())
        assert "m4" not in out["selected_article_ids"]

    def test_milrinone_HF_wrong_condition_NOT_default_selected(self):
        out = select_default_articles(_milrinone_picott(), self._milrinone_ranked_articles())
        assert "m5" not in out["selected_article_ids"]


# =============================================================================
# Default-selection edge cases
# =============================================================================

class TestDefaultSelectionEdgeCases:

    def test_empty_input_returns_empty_selection(self):
        out = select_default_articles(_ami_picott(), [])
        assert out["selected_article_ids"] == []
        assert out["excluded_count"] == 0

    def test_no_picott_falls_through_to_eligibility_only(self):
        """When picott is None or empty, the must-have anchor check is
        skipped — every non-capped article is eligible."""
        articles = [
            _rank_row(_make_article(aid="x1", title="Anything"), relevance_score=70,
                       overall_rank_score=70),
        ]
        out = select_default_articles(None, articles)
        assert "x1" in out["selected_article_ids"]

    def test_negative_concept_excludes(self):
        articles = [
            _rank_row(
                _make_article(aid="n1", title="Milrinone in SAH"),
                relevance_score=80, overall_rank_score=80,
                penalty_concepts=["off-label"],
                cap_reasons=["negative_concept_fired"],
            ),
        ]
        out = select_default_articles(_milrinone_picott(), articles)
        assert "n1" not in out["selected_article_ids"]
        assert out["excluded_rationale_summary"].get("negative_concept_fired") == 1

    def test_preprint_excluded_by_default(self):
        articles = [
            _rank_row(
                _make_article(
                    aid="p1",
                    title="Milrinone SAH preprint",
                    abstract="milrinone subarachnoid hemorrhage",
                    is_preprint=True,
                ),
                relevance_score=80, overall_rank_score=80,
            ),
        ]
        out = select_default_articles(_milrinone_picott(), articles)
        assert "p1" not in out["selected_article_ids"]
        assert out["excluded_rationale_summary"].get(
            "excluded_retracted_or_preprint_or_unsafe"
        ) == 1

    def test_under_5_candidates_returns_fewer(self):
        """Spec: if fewer than 5 candidates pass, return fewer — DO NOT
        pad with capped articles."""
        articles = [
            _rank_row(
                _make_article(
                    aid="o1",
                    title="MINT trial — liberal vs restrictive transfusion in AMI anemia",
                    abstract="acute myocardial infarction anemia liberal transfusion threshold restrictive",
                    art_type="rct",
                ),
                relevance_score=92, overall_rank_score=92,
            ),
            # All other 4 articles are capped — must NOT be padded into selection.
            _rank_row(
                _make_article(aid="c1", title="Unrelated capped"),
                relevance_score=70, overall_rank_score=40,
                cap_reasons=["missing_condition_anchor"],
            ),
            _rank_row(
                _make_article(aid="c2", title="Another unrelated capped"),
                relevance_score=70, overall_rank_score=40,
                cap_reasons=["missing_intervention_anchor"],
            ),
        ]
        out = select_default_articles(_ami_picott(), articles)
        assert out["selected_article_ids"] == ["o1"], (
            f"expected only o1; got {out['selected_article_ids']!r}"
        )
        assert "c1" not in out["selected_article_ids"]
        assert "c2" not in out["selected_article_ids"]


# =============================================================================
# E. Override endpoint (via test client)
# =============================================================================

import asyncio
from httpx import AsyncClient, ASGITransport


async def _login_and_get_token(client, email, password):
    r = await client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_include_override_requires_acknowledgement():
    """POST without override_acknowledged=true must return 400."""
    from server import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Use the seeded clinician account.
        r = await client.post(
            "/api/auth/login",
            json={"email": "clinician@test.com", "password": "ClinicianTest123!"},
        )
        if r.status_code != 200:
            pytest.skip(f"clinician seed account not available: {r.status_code}")
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        # Use a known session/article id — even a 404 is fine for this contract test
        # because we're proving the 400 fires BEFORE the 404 lookup. Test the explicit
        # body validation only.
        r2 = await client.post(
            "/api/search-sessions/nonexistent-sid/articles/nonexistent-aid/include-override",
            json={"override_acknowledged": False},
            headers=headers,
        )
        # We expect 400 from the body-validation branch (precedes the
        # session lookup), confirming the contract.
        assert r2.status_code == 400, r2.text
        body = r2.json()
        assert "override_acknowledged=true is required" in body.get("detail", "")
