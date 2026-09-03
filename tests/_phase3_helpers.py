"""Shared helpers for Phase 3 tests.

Provides:
  - _signup, _login (auth)
  - _make_packet_session: build a fully wired session with N articles,
    N include_in_summary article_text_extracts, optionally user-uploaded.
  - patch_llm: context manager / fixture returning a mock complete_json.
"""
from __future__ import annotations

import contextlib
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.db import Collections, get_db  # noqa: E402


async def signup(client, email: str, password: str = "Pass1234567!") -> tuple[str, str]:
    r = await client.post(
        "/api/auth/signup", json={"email": email, "password": password, "name": "T"}
    )
    if r.status_code not in (200, 201):
        r = await client.post(
            "/api/auth/login", json={"email": email, "password": password}
        )
    j = r.json()
    return j["access_token"], j["user"]["id"]


async def login(client, email: str, password: str) -> str:
    r = await client.post("/api/auth/login", json={"email": email, "password": password})
    return r.json()["access_token"]


async def make_packet_session(
    client,
    headers,
    *,
    articles: list[dict],
    upload_extract_for: Optional[list[int]] = None,  # indices that get user-uploaded extracts
) -> tuple[str, str, list[str]]:
    """Create a session with N articles + extracts that the summarizer will see.

    `articles` is a list of dicts with keys: title, abstract, doi, pmid, pmcid,
    year, journal, study_type, is_preprint, is_retracted, user_included,
    exclusion_reason.

    Returns (session_id, user_id, [article_ids]).
    """
    db = get_db()
    me = await client.get("/api/auth/me", headers=headers)
    user_id = me.json()["id"]
    now = datetime.now(timezone.utc)
    session_id = uuid.uuid4().hex
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": session_id,
        "user_id": user_id,
        "status": "awaiting_selection",
        "original_question": "What is the evidence for SGLT2 inhibitors in HFpEF?",
        "sanitized_question": "What is the evidence for SGLT2 inhibitors in HFpEF?",
        "pico_json": {
            "population": "Adults with HFpEF",
            "intervention": "SGLT2 inhibitors",
            "comparator": "Placebo",
            "outcomes": "Heart failure hospitalization",
            "clinical_question_type": "treatment_efficacy",
        },
        "deleted_at": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    article_ids: list[str] = []
    upload_extract_for = set(upload_extract_for or [])
    for i, a in enumerate(articles):
        aid = uuid.uuid4().hex
        article_ids.append(aid)
        # Phase 6D — replace_one with upsert handles the unique
        # constraints on (pmid, pmcid, doi_normalized, canonical_key)
        # so test fixtures can be reused safely across multiple tests
        # in the same DB session.
        pmid_val = a.get("pmid")
        doi_val = a.get("doi")
        pmcid_val = a.get("pmcid")
        canonical = (
            f"pmid:{pmid_val}" if pmid_val else f"id:{aid}"
        )
        # Look up an existing article by any of the unique identifiers
        # so the test can reuse the prior id (subsequent calls to this
        # helper return the existing article_id under that pmid/pmcid).
        ors: list[dict] = [{"canonical_key": canonical}]
        if pmid_val:
            ors.append({"pmid": pmid_val})
        if pmcid_val:
            ors.append({"pmcid": pmcid_val})
        existing = await db[Collections.ARTICLES].find_one(
            {"$or": ors}, {"_id": 0, "id": 1},
        )
        if existing:
            aid = existing["id"]
            article_ids[-1] = aid
            # Phase 6D — sync mutable fields (abstract/title/is_preprint/
            # is_retracted/has_expression_of_concern/journal/year) so the
            # test's spec wins over the prior-test fixture.
            await db[Collections.ARTICLES].update_one(
                {"id": aid},
                {"$set": {
                    "abstract": a.get("abstract") or "",
                    "title": a["title"],
                    "is_preprint": bool(a.get("is_preprint")),
                    "is_retracted": bool(a.get("is_retracted")),
                    "has_expression_of_concern": bool(a.get("has_expression_of_concern")),
                    "journal": a.get("journal") or "Test Journal",
                    "publication_year": a.get("year") or 2024,
                    "article_type_classification": a.get("study_type") or "rct",
                    "full_text_status": "unknown",
                    "updated_at": now,
                }},
            )
        else:
            await db[Collections.ARTICLES].insert_one({
                "id": aid,
                "canonical_key": canonical,
                "title": a["title"],
                "abstract": a.get("abstract") or "",
                "doi": doi_val,
                "pmid": pmid_val,
                "pmcid": pmcid_val,
                "journal": a.get("journal") or "Test Journal",
                "publication_year": a.get("year") or 2024,
                "article_type_classification": a.get("study_type") or "rct",
                "authors_json": [], "source_ids_json": {}, "source_urls_json": [],
                "is_preprint": bool(a.get("is_preprint")),
                "is_retracted": bool(a.get("is_retracted")),
                "is_retraction_notice": False,
                "has_expression_of_concern": False,
                "publication_types_json": [],
                "full_text_status": "unknown",
                "created_at": now, "updated_at": now,
            })
        await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
            "id": uuid.uuid4().hex,
            "search_session_id": session_id,
            "article_id": aid,
            "user_included": a.get("user_included", True),
            "included_by_default": a.get(
                "included_by_default",
                a.get("user_included", True),
            ),
            "exclusion_reason": a.get("exclusion_reason"),
            "relevance_score": 80.0, "evidence_design_score": 80.0,
            "overall_rank": 90 - i, "rank_explanation": "test",
            "query_types_matched": [], "sources_present": ["europe_pmc"], "flags": [],
            "created_at": now, "updated_at": now,
        })
        # Build at least an abstract extract for selected articles that have abstract text.
        if a.get("abstract"):
            await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one({
                "id": uuid.uuid4().hex,
                "article_id": aid,
                "search_session_id": session_id,
                "section_name": "abstract",
                "section_text": a["abstract"],
                "token_count": max(1, len(a["abstract"].split())),
                "include_in_summary": True,
                "text_source": "abstract",
                "evidence_text_level": "abstract",
                "created_at": now,
            })
        if i in upload_extract_for:
            await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one({
                "id": uuid.uuid4().hex,
                "article_id": aid,
                "search_session_id": session_id,
                "user_id": user_id,
                "section_name": "results",
                "section_text": (
                    "Results: SGLT2 inhibitors reduced HF hospitalizations vs placebo "
                    "with consistent direction across trials. No new safety signals."
                ),
                "token_count": 30,
                "include_in_summary": True,
                "text_source": "user_uploaded_full_text",
                "evidence_text_level": "user_uploaded_full_text",
                "created_at": now,
            })
    return session_id, user_id, article_ids


def good_apixaban_articles() -> list[dict]:
    return [
        {
            "title": "Apixaban versus warfarin in patients with atrial fibrillation",
            "abstract": "ARISTOTLE: apixaban reduced major bleeding and stroke vs warfarin.",
            "doi": "10.1056/NEJMoa1107039",
            "pmid": "21870978",
            "year": 2011, "journal": "N Engl J Med", "study_type": "rct",
        },
        {
            "title": "Rivaroxaban versus warfarin in non-valvular atrial fibrillation",
            "abstract": "ROCKET-AF: rivaroxaban non-inferior to warfarin for stroke; similar major bleeding.",
            "doi": "10.1056/NEJMoa1009638",
            "pmid": "21830957",
            "year": 2011, "journal": "N Engl J Med", "study_type": "rct",
        },
        {
            "title": "Dabigatran versus warfarin in patients with atrial fibrillation",
            "abstract": "RE-LY: dabigatran 150 mg reduced stroke; bleeding similar at higher dose.",
            "doi": "10.1056/NEJMoa0905561",
            "pmid": "19717844",
            "year": 2009, "journal": "N Engl J Med", "study_type": "rct",
        },
    ]


# ----------------------------------------------------------------------------
# Canned LLM responses
# ----------------------------------------------------------------------------


def good_apixaban_draft(span_ids_by_article: dict[str, list[str]]) -> dict:
    a1, a2, a3 = list(span_ids_by_article.keys())[:3]
    return {
        "clinical_bottom_line": "DOACs reduce stroke vs warfarin with at-least-similar bleeding.",
        "evidence_summary": "Three RCTs showed favorable efficacy and safety profiles.",
        "applicability": "Applies to non-valvular AF adults eligible for anticoagulation.",
        "certainty": "low",
        "certainty_rationale": "Only abstract-level data available.",
        "selected_studies": [
            {
                "article_id": aid,
                "citation_label": f"PMID:{(span_ids_by_article[aid][0].split(':')[1])[:7]}" if False else "PMID:?",
                "pmid": None, "doi": None, "pmcid": None,
                "title": "study",
                "journal": "j", "publication_year": 2011, "study_type": "rct",
                "evidence_text_level": "abstract",
                "key_finding": "Reduced events.",
                "limitations": "Industry-funded.",
                "supporting_evidence_span_ids": [span_ids_by_article[aid][0]],
            } for aid in span_ids_by_article
        ],
        "claim_map": [
            {
                "claim_id": "c1",
                "claim_text": "DOACs reduce stroke vs warfarin.",
                "claim_type": "effect",
                "supporting_evidence_span_ids": [
                    span_ids_by_article[a1][0],
                    span_ids_by_article[a2][0],
                ],
                "supporting_article_ids": [a1, a2],
            },
        ],
        "conflicting_evidence": "",
        "important_caveats": "Abstract-only; full-text not parsed.",
        "what_was_not_answered": "Long-term mortality not reported here.",
    }


def good_verifier_passed() -> dict:
    return {
        "verification_status": "passed",
        "final_answer_safe_to_show": True,
        "unsupported_claims": [],
        "incorrect_citations": [],
        "required_revisions": [],
        "verifier_notes": "All claims supported.",
        "verified_summary_json": None,  # echoed by code
    }


def good_verifier_failed() -> dict:
    return {
        "verification_status": "failed",
        "final_answer_safe_to_show": False,
        "unsupported_claims": [
            {"claim_id": "c1", "claim_text": "x", "reason": "Span does not support claim."},
        ],
        "incorrect_citations": [],
        "required_revisions": ["Remove unsupported claim c1."],
        "verifier_notes": "Rejected.",
        "verified_summary_json": None,
    }


# ----------------------------------------------------------------------------
# Provider patching helpers
# ----------------------------------------------------------------------------


@contextlib.contextmanager
def patch_llm_configured(provider="emergent", summary_model="gpt-5.4-mini",
                          verifier_model="gpt-5.4-mini"):
    """Force `llm_provider.is_configured()` to return True for the duration.

    Also captures and restores `complete_json` so tests that monkey-patch it
    don't leak into subsequent tests.
    """
    from services.llm_provider import llm_provider
    saved = (
        llm_provider.provider, llm_provider._emergent_key, llm_provider._summary_model,
        llm_provider._verifier_model, llm_provider._openai_key,
        llm_provider.complete_json,
    )
    llm_provider.provider = provider
    llm_provider._emergent_key = "test-emergent-key"
    if provider == "openai":
        llm_provider._openai_key = "test-openai-key"
    llm_provider._summary_model = summary_model
    llm_provider._verifier_model = verifier_model
    try:
        yield llm_provider
    finally:
        (llm_provider.provider, llm_provider._emergent_key,
         llm_provider._summary_model, llm_provider._verifier_model,
         llm_provider._openai_key, llm_provider.complete_json) = saved


@contextlib.contextmanager
def patch_llm_disabled():
    from services.llm_provider import llm_provider
    saved = (
        llm_provider.provider, llm_provider._emergent_key, llm_provider._openai_key,
        llm_provider.complete_json,
    )
    llm_provider.provider = "none"
    llm_provider._emergent_key = ""
    llm_provider._openai_key = ""
    try:
        yield llm_provider
    finally:
        (llm_provider.provider, llm_provider._emergent_key,
         llm_provider._openai_key, llm_provider.complete_json) = saved
