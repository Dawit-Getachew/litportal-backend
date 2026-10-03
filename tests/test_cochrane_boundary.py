"""B2 — Cochrane boundary, proven end-to-end through the summary pipeline.

Unit policy (resolver + eligibility helper) is covered in
test_source_permissions.py. This test closes the B2 acceptance gap: a Cochrane
record the user explicitly SELECTED for a session must NOT enter the AI summary
context, and must be surfaced as ineligible with the closed-set reason — while a
co-selected PubMed record summarizes normally.

Drives the real ``routers.summary._build_summary_inputs`` (no LLM). Uses the
session-scoped ``client`` fixture so the app/Mongo lifespan is active.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from core.db import Collections, get_db
from routers.summary import _build_summary_inputs

ABSTRACT = "RCT: the intervention reduced the primary composite endpoint vs placebo."


async def _seed_article(db, source: str, *, journal: str = "NEJM", pmid: bool = True,
                        sources: bool = True) -> str:
    aid = f"cb-{source}-" + uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc)
    await db[Collections.ARTICLES].insert_one({
        "id": aid, "canonical_key": aid, "title": f"Boundary test ({source})",
        "abstract": ABSTRACT, "journal": journal, "publication_year": 2024,
        "pmid": str(uuid.uuid4().int)[:8] if pmid else None, "doi": f"10.1/{aid}", "doi_normalized": f"10.1/{aid}",
        "sources_present": [source] if sources else [], "source_ids_json": {},
        "full_text_status": "abstract_only",
        "is_retracted": False, "is_preprint": False, "created_at": now, "updated_at": now,
    })
    return aid


async def _seed_ssa(db, sid: str, aid: str) -> None:
    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
        "id": uuid.uuid4().hex, "search_session_id": sid, "article_id": aid,
        "user_included": True, "included_by_default": True, "exclusion_reason": None,
        "cap_reasons": [], "relevance_score": 90.0, "evidence_design_score": 80.0,
        "overall_rank": 95, "created_at": now, "updated_at": now,
    })


async def test_selected_cochrane_record_excluded_from_ai_summary(client):
    db = get_db()
    uid = "cb-u-" + uuid.uuid4().hex[:8]
    sid = "cb-s-" + uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": uid, "status": "awaiting_selection",
        "original_question": "does the intervention help?",
        "sanitized_question": "does the intervention help?",
        "deleted_at": None, "created_at": now, "updated_at": now,
    })
    pubmed = await _seed_article(db, "pubmed")
    cochrane = await _seed_article(db, "cochrane")
    await _seed_ssa(db, sid, pubmed)
    await _seed_ssa(db, sid, cochrane)

    session = await db[Collections.SEARCH_SESSIONS].find_one({"id": sid}, {"_id": 0})
    selected_studies, articles_by_id, extracts, packet = await _build_summary_inputs(
        session=session, user_id=uid, db=db,
    )
    selected_ids = {s["article_id"] for s in selected_studies}

    # PubMed selected; Cochrane excluded from the AI context entirely.
    assert pubmed in selected_ids
    assert cochrane not in selected_ids
    assert cochrane not in extracts  # never reaches the summarizer's evidence

    # Cochrane surfaced as ineligible with the closed-set source-policy reason.
    assert packet["ineligible_count"] == 1
    ineligible = {a["article_id"]: a for a in packet["ineligible_articles"]}
    assert cochrane in ineligible
    assert ineligible[cochrane]["source"] == "cochrane"
    assert ineligible[cochrane]["reason"] == "cochrane_ai_ingestion_restricted"


async def test_pubmed_indexed_cochrane_and_unknown_source_records_fail_closed(client):
    """A Cochrane review that arrived through PubMed is still Cochrane content, and
    a record with no known source is not summarized; a PMID-only record is PubMed."""
    db = get_db()
    uid = "cb-u-" + uuid.uuid4().hex[:8]
    sid = "cb-s-" + uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": uid, "status": "awaiting_selection",
        "original_question": "does the intervention help?",
        "sanitized_question": "does the intervention help?",
        "deleted_at": None, "created_at": now, "updated_at": now,
    })
    indexed_cochrane = await _seed_article(db, "pubmed", journal="Cochrane Database of Systematic Reviews")
    unknown = await _seed_article(db, "none", pmid=False, sources=False)
    pmid_only = await _seed_article(db, "pmidonly", sources=False)
    for aid in (indexed_cochrane, unknown, pmid_only):
        await _seed_ssa(db, sid, aid)

    session = await db[Collections.SEARCH_SESSIONS].find_one({"id": sid}, {"_id": 0})
    selected_studies, _, extracts, packet = await _build_summary_inputs(session=session, user_id=uid, db=db)
    assert {s["article_id"] for s in selected_studies} == {pmid_only}
    assert indexed_cochrane not in extracts and unknown not in extracts
    ineligible = {a["article_id"]: a for a in packet["ineligible_articles"]}
    assert ineligible[indexed_cochrane]["reason"] == "cochrane_ai_ingestion_restricted"
    assert ineligible[unknown]["reason"] == "no_source_info_fail_closed"
