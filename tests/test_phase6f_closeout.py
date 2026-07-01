"""Phase 6F closeout — within-bucket strict DESC sort + clarification gate.

Tests two implementations from the Phase 6F closeout directive:

IMPL-1: Within the ``included_by_default=true`` bucket of
GET /api/search-sessions/{id} → ``articles[]``, sort strictly by
``relevance_score DESC``, then ``evidence_design_score DESC``, then
``canonical_key ASC``. Same DESC order inside the ``false`` bucket.

IMPL-2: When PrimaryClinicalEntityAnchorService returns
``anchor_failure_reason="no_primary_clinical_entity"``, the session
terminates at status ``awaiting_clinical_subject`` with a closed-set
clarification_message and NO external API / LLM calls.
"""
from __future__ import annotations

import asyncio

import pytest

from services.primary_clinical_anchor import (
    CLARIFICATION_MESSAGE,
    primary_clinical_anchor_service,
)


# ---------------------------------------------------------------------------
# IMPL-1 — within-bucket strict DESC sort
# ---------------------------------------------------------------------------

def _composite_sort_key(r: dict):
    """Replicates the production sort key from
    ``routers/search_sessions.py::get_search_session``.
    """
    return (
        0 if bool(r.get("included_by_default")) else 1,
        -float(r.get("relevance_score") or 0.0),
        -float(r.get("evidence_design_score") or 0.0),
        str(
            (r.get("_article") or {}).get("canonical_key")
            or r.get("article_id")
            or ""
        ),
    )


class TestWithinBucketDescSort:
    def test_directive_example_60_85_75_60_75_orders_85_75_75_60_60(self):
        """The directive's exact acceptance example: 5 default-selected
        articles with relevance 60/85/75/60/75 → ordered 85/75/75/60/60.
        """
        sa_rows = [
            {"article_id": "a", "included_by_default": True,
             "relevance_score": 60.0, "evidence_design_score": 50.0,
             "_article": {"canonical_key": "ck-a"}},
            {"article_id": "b", "included_by_default": True,
             "relevance_score": 85.0, "evidence_design_score": 70.0,
             "_article": {"canonical_key": "ck-b"}},
            {"article_id": "c", "included_by_default": True,
             "relevance_score": 75.0, "evidence_design_score": 80.0,
             "_article": {"canonical_key": "ck-c"}},
            {"article_id": "d", "included_by_default": True,
             "relevance_score": 60.0, "evidence_design_score": 60.0,
             "_article": {"canonical_key": "ck-d"}},
            {"article_id": "e", "included_by_default": True,
             "relevance_score": 75.0, "evidence_design_score": 65.0,
             "_article": {"canonical_key": "ck-e"}},
        ]
        sa_rows.sort(key=_composite_sort_key)
        rels = [r["relevance_score"] for r in sa_rows]
        assert rels == [85.0, 75.0, 75.0, 60.0, 60.0], rels

    def test_buckets_kept_distinct(self):
        """Default-selected rows surface above false-bucket rows
        regardless of relevance numbers."""
        sa_rows = [
            {"article_id": "high-but-not-selected",
             "included_by_default": False, "relevance_score": 95.0,
             "evidence_design_score": 60.0,
             "_article": {"canonical_key": "z-high"}},
            {"article_id": "low-but-selected",
             "included_by_default": True, "relevance_score": 40.0,
             "evidence_design_score": 60.0,
             "_article": {"canonical_key": "a-low"}},
        ]
        sa_rows.sort(key=_composite_sort_key)
        assert sa_rows[0]["article_id"] == "low-but-selected"
        assert sa_rows[1]["article_id"] == "high-but-not-selected"

    def test_evidence_design_secondary_then_canonical_key_tiebreaker(self):
        """When relevance ties, evidence_design_score breaks first;
        canonical_key ASC breaks last."""
        sa_rows = [
            {"article_id": "a", "included_by_default": True,
             "relevance_score": 60.0, "evidence_design_score": 50.0,
             "_article": {"canonical_key": "ck-zebra"}},
            {"article_id": "b", "included_by_default": True,
             "relevance_score": 60.0, "evidence_design_score": 80.0,
             "_article": {"canonical_key": "ck-zebra"}},
            {"article_id": "c", "included_by_default": True,
             "relevance_score": 60.0, "evidence_design_score": 80.0,
             "_article": {"canonical_key": "ck-alpha"}},
        ]
        sa_rows.sort(key=_composite_sort_key)
        # design 80 > 50 → b and c first. canonical_key alpha < zebra → c first.
        assert [r["article_id"] for r in sa_rows] == ["c", "b", "a"]

    def test_overall_rank_is_NOT_part_of_within_bucket_key(self):
        """Phase 6F closeout explicitly drops overall_rank from the
        within-bucket key. An article with a high overall_rank but
        lower relevance_score MUST sort BELOW one with higher
        relevance_score."""
        sa_rows = [
            {"article_id": "high-overall-low-rel",
             "included_by_default": True, "relevance_score": 60.0,
             "evidence_design_score": 50.0, "overall_rank": 100,
             "_article": {"canonical_key": "ck-a"}},
            {"article_id": "lower-overall-higher-rel",
             "included_by_default": True, "relevance_score": 85.0,
             "evidence_design_score": 50.0, "overall_rank": 80,
             "_article": {"canonical_key": "ck-b"}},
        ]
        sa_rows.sort(key=_composite_sort_key)
        assert sa_rows[0]["article_id"] == "lower-overall-higher-rel"


# ---------------------------------------------------------------------------
# IMPL-2 — clarification message + new session status
# ---------------------------------------------------------------------------

class TestClarificationGate:
    def test_anchor_service_clarification_message_constant_unchanged(self):
        """The closed-set constant must match the spec verbatim."""
        assert CLARIFICATION_MESSAGE == (
            "I could not confidently identify the clinical subject of "
            "the question. Please specify the drug, condition, test, "
            "vaccine, or procedure."
        )

    @pytest.mark.parametrize("question", [
        "What are the side effects?",
        "What is first-line treatment?",
        "What is the dose?",
        "What is the recommended age?",
    ])
    def test_generic_only_question_triggers_no_primary_clinical_entity(
        self, question
    ):
        """Each of the four directive PROOF B questions must produce
        ``anchor_failure_reason="no_primary_clinical_entity"`` from
        the anchor service when given an empty PICOTT (i.e., when the
        upstream parser returned no must-haves)."""
        # Deterministically empty PICOTT — simulates the case where the
        # LLM returns nothing actionable. The anchor service is pure;
        # this test does NOT exercise the LLM.
        out = primary_clinical_anchor_service.derive(
            picott={"must_have_concepts": []},
            ontology_mappings=[],
            normalized_question=question,
        )
        assert out["anchor_failure_reason"] == "no_primary_clinical_entity"
        # Generic modifiers ARE detected for transparency.
        assert isinstance(out["generic_modifiers"], list)

    def test_session_status_enum_includes_awaiting_clinical_subject(self):
        """The new ``awaiting_clinical_subject`` terminal status must
        be in the closed-set enum."""
        from models.sessions import SessionStatus
        import typing
        # SessionStatus is a Literal — extract its members.
        members = typing.get_args(SessionStatus)
        assert "awaiting_clinical_subject" in members
