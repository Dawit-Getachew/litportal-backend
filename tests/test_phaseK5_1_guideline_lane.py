"""Phase K5.1 — PubMed guideline lane tests.

Validates the deterministic helpers + the additive K4 evidence-packet
surfacing, with NO change to retrieval, ranking, selection, summaries,
or citation verification.
"""
from __future__ import annotations

import uuid

import pytest

from core.db import Collections, get_db
from models.common import utcnow
from services.guideline_lane import (
    GUIDELINE_LANE_LABEL,
    GUIDELINE_LANE_PROVENANCE,
    article_guideline_provenance,
    classify_guideline_intent,
    run_guideline_query_qa,
)
from services.k4_evidence_packet_builder import build_evidence_packet
from services.search_planning import SearchPlanningService


# =====================================================================
# 1. Intent gate — positives.
# =====================================================================
@pytest.mark.parametrize("q", [
    "What are the guidelines for colorectal cancer screening in average-risk adults?",
    "How should syncope be evaluated in older adults?",
    "How should unstable SVT be managed?",
    "How often should colonoscopy be repeated after polypectomy?",
])
def test_K5_1_intent_gate_positive(q):
    res = classify_guideline_intent(question=q)
    assert res["applicable"] is True, q


# =====================================================================
# 2. Intent gate — negatives (incl. K4A-protected + mechanism + generic).
# =====================================================================
@pytest.mark.parametrize("q", [
    "Does a negative MRSA nasal swab rule out MRSA pneumonia?",
    "What are the markers of severity in acute pancreatitis?",
    "What is the incidence of aspiration pneumonia in patients with GCS 8 or less?",
    "What is the mechanism of pruritus in cholestatic liver disease?",
    "What is the dose?",
])
def test_K5_1_intent_gate_negative(q):
    res = classify_guideline_intent(question=q)
    assert res["applicable"] is False, q


@pytest.mark.parametrize("intent,expected", [
    ("diagnostic_workup", True),
    ("stabilization_emergency_management", True),
    ("monitoring_followup_surveillance", True),
    ("prevention_screening", True),
    ("pathophysiology_mechanism", False),
    ("risk_etiology_harm", False),
])
def test_K5_1_intent_gate_by_cqm_intent(intent, expected):
    res = classify_guideline_intent(primary_intent=intent)
    assert res["applicable"] is expected, intent


# =====================================================================
# 3. Article provenance.
# =====================================================================
def test_K5_1_article_provenance():
    assert article_guideline_provenance({"is_guideline": True}) == GUIDELINE_LANE_PROVENANCE
    assert article_guideline_provenance({"article_type": "guideline"}) == GUIDELINE_LANE_PROVENANCE
    assert article_guideline_provenance(
        {"publication_types": ["Practice Guideline", "Journal Article"]}
    ) == GUIDELINE_LANE_PROVENANCE
    assert article_guideline_provenance({"publication_types": ["Journal Article"]}) is None
    assert article_guideline_provenance({"title": "a cohort study"}) is None


# =====================================================================
# 4. Query QA.
# =====================================================================
def test_K5_1_query_qa_valid_passes():
    q = '(asthma[tiab]) AND ("practice guideline"[Publication Type]) NOT preprint[Publication Type]'
    assert run_guideline_query_qa(q)["passed"] is True


@pytest.mark.parametrize("q,code", [
    ("   ", "empty_query"),
    ("(asthma[tiab] AND guideline[pt]", "unbalanced_parentheses"),
    ("asthma[tiab] AND ()", "empty_paren_block"),
    ("asthma[tiab] AND", "trailing_boolean_operator"),
    ("asthma[]", "empty_field_tag"),
])
def test_K5_1_query_qa_structural_failures(q, code):
    res = run_guideline_query_qa(q)
    assert res["passed"] is False
    assert code in res["warnings"]


def test_K5_1_planner_guideline_queries_pass_qa():
    """The guideline_focused queries the live planner emits must pass QA
    (query QA before dispatch)."""
    parsed = {
        "key_concepts": ["colorectal cancer", "screening"],
        "synonyms": ["colorectal neoplasms"],
        "clinical_question_type": "prevention_screening",
        "population": "average-risk adults",
        "intervention_or_exposure": "screening",
        "normalized_question": "guidelines for colorectal cancer screening in average-risk adults",
    }
    plans = SearchPlanningService().plan(parsed, sources=["pubmed", "europe_pmc"])
    gl = [p for p in plans if p.query_type == "guideline_focused"]
    assert gl, "expected at least one guideline_focused query"
    for p in gl:
        res = run_guideline_query_qa(p.query_string)
        assert res["passed"] is True, (p.source, p.query_string, res["warnings"])


# =====================================================================
# 5. Packet surfacing (integration).
# =====================================================================
async def _seed_session(*, user_id, primary_intent, question, articles):
    db = get_db()
    session_id = str(uuid.uuid4())
    bqp = {
        "schema_version": "1.0",
        "mode": "branch_aware",
        "branches": [{
            "branch_id": "b_main",
            "intent": primary_intent,
            "framework": primary_intent,
            "query_families": [{"family_name": "high_recall"}],
            "required_concepts_used": [],
        }],
    }
    cqm = {
        "primary_intent": primary_intent,
        "framework": primary_intent,
        "branch_query_plan": bqp,
        "anchors": [], "required_anchors": [],
    }
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": session_id, "user_id": user_id,
        "original_question": question, "sanitized_question": question,
        "status": "awaiting_selection",
        "clinical_question_type": primary_intent,
        "clinical_question_map_json": cqm,
        "created_at": utcnow(), "updated_at": utcnow(),
        "expires_at": utcnow(), "deleted_at": None,
    })
    await db[Collections.ARTICLES].insert_many(articles)
    for i, art in enumerate(articles, start=1):
        await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
            "id": f"sa-{i}-{session_id}",
            "search_session_id": session_id,
            "article_id": art["id"],
            "overall_rank": 100 - i,
            "included_by_default": (i == 1),
            "user_included": False,
            "exclusion_reason": None,
            "query_types_matched": ["high_recall"],
            "matched_branch_ids": ["b_main"],
            "sources_present": ["pubmed"],
            "created_at": utcnow(), "updated_at": utcnow(),
        })
    return session_id


async def _cleanup(session_id):
    db = get_db()
    await db[Collections.SEARCH_SESSIONS].delete_one({"id": session_id})
    await db[Collections.SEARCH_SESSION_ARTICLES].delete_many(
        {"search_session_id": session_id})
    await db[Collections.EVIDENCE_PACKETS].delete_one(
        {"search_session_id": session_id})


@pytest.mark.asyncio
async def test_K5_1_packet_surfaces_guideline_lane_for_guideline_question():
    arts = [
        {"id": f"k51-g1-{uuid.uuid4().hex[:6]}", "pmid": "10", "doi": "10.1/g",
         "title": "Colorectal cancer screening guideline",
         "journal": "Gastro", "publication_year": 2024, "source": "pubmed",
         "full_text_status": "abstract_only", "is_guideline": True,
         "publication_types_json": ["Practice Guideline"]},
        {"id": f"k51-g2-{uuid.uuid4().hex[:6]}", "pmid": "11", "doi": "10.1/h",
         "title": "Colorectal cancer screening cohort",
         "journal": "Gut", "publication_year": 2023, "source": "pubmed",
         "full_text_status": "abstract_only",
         "publication_types_json": ["Journal Article"]},
    ]
    sid = await _seed_session(
        user_id="k51-user",
        primary_intent="prevention_screening",
        question="What are the guidelines for colorectal cancer screening in average-risk adults?",
        articles=arts,
    )
    try:
        packet = await build_evidence_packet(search_session_id=sid)
        assert packet["evidence_packet_version"].startswith("k5")
        br = packet["branches"][0]
        assert br["guideline_lane_active"] is True
        assert br["guideline_lane_label"] == GUIDELINE_LANE_LABEL
        provs = [a.get("guideline_lane") for a in br["articles"]]
        assert GUIDELINE_LANE_PROVENANCE in provs   # the guideline article
        assert None in provs                        # the cohort article
    finally:
        await _cleanup(sid)


@pytest.mark.asyncio
async def test_K5_1_packet_no_guideline_lane_for_mechanism_question():
    """Mechanism question must NOT surface the guideline lane even if a
    guideline-pub-type article is present (no inappropriate override)."""
    arts = [
        {"id": f"k51-m1-{uuid.uuid4().hex[:6]}", "pmid": "20", "doi": "10.2/a",
         "title": "Mechanism of pruritus in cholestasis",
         "journal": "Nat Rev", "publication_year": 2023, "source": "pubmed",
         "full_text_status": "abstract_only", "is_guideline": True,
         "publication_types_json": ["Practice Guideline"]},
    ]
    sid = await _seed_session(
        user_id="k51-user",
        primary_intent="pathophysiology_mechanism",
        question="What is the mechanism of pruritus in cholestatic liver disease?",
        articles=arts,
    )
    try:
        packet = await build_evidence_packet(search_session_id=sid)
        br = packet["branches"][0]
        assert br["guideline_lane_active"] is False
        for a in br["articles"]:
            assert a.get("guideline_lane") is None
    finally:
        await _cleanup(sid)
