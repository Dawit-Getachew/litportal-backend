"""B4a (part 3) — Parallel AI search executor.

When the EvidenceCurator recommends it AND ``AI_RANKER_PARALLEL_SEARCH_ENABLED``
is on, run a bounded supplemental PubMed search (built from the curator's
broader-guideline topics, falling back to the question), dedupe against the
session's existing results, and route the new candidates onto the SAME
``ai_surfaced_articles`` shelf — never into the deterministic ranking.

Governance: PubMed only (permission-checked, no scraping); bounded by
``MAX_RANKER_PARALLEL_AI_CANDIDATES``; advisory shelf only; fail-closed.
"""
from __future__ import annotations

import logging

from core.config import get_settings
from core.db import Collections, get_db
from models.common import new_id, utcnow
from services.audit import audit_service
from services.downstream_eligibility import can_use_article_for_tool

logger = logging.getLogger("litportal.agents.parallel_search")

PROMPT_VERSION = "parallel-v1"


async def maybe_run_parallel_search(
    session_id: str,
    user_id: str,
    sanitized_question: str,
    broader_topics: list[str],
    existing_pmids: set[str],
    existing_dois: set[str],
) -> int:
    """Run the supplemental search (if enabled). Returns # surfaced. Never raises."""
    s = get_settings()
    if not (s.AI_EVIDENCE_SURFACING_ENABLED and s.AI_RANKER_PARALLEL_SEARCH_ENABLED):
        return 0

    topics = [t for t in (broader_topics or []) if t][:5]
    query_string = " OR ".join(f'"{t}"' for t in topics) if topics else (sanitized_question or "").strip()
    if not query_string:
        return 0

    try:
        from services.connectors.pubmed import PubMedConnector
        results = await PubMedConnector().search(
            [{"query_type": "parallel_ai_search", "query_string": query_string}],
            search_session_id=session_id,
        )
    except Exception:  # noqa: BLE001
        logger.exception("parallel_ai_search pubmed query failed (non-fatal)")
        return 0

    cap = s.MAX_RANKER_PARALLEL_AI_CANDIDATES
    db = get_db()
    now = utcnow()
    surfaced = 0
    for _qtype, norm in results:
        if surfaced >= cap:
            break
        pmid = (norm.get("pmid") or "").strip() or None
        doi = (norm.get("doi") or norm.get("doi_normalized") or "").strip().lower() or None
        if (pmid and pmid in existing_pmids) or (doi and doi in existing_dois):
            continue  # already in the deterministic results
        # PubMed is AI-permitted, but check policy defensively (source aware).
        if not can_use_article_for_tool({"sources_present": ["pubmed"]}, "ranker_optimization")["allowed"]:
            break
        try:
            from services.pipeline import _upsert_article
            article_id = await _upsert_article(norm)
        except Exception:  # noqa: BLE001
            continue
        if not article_id:
            continue
        exists = await db[Collections.AI_SURFACED_ARTICLES].find_one(
            {"search_session_id": session_id, "article_id": article_id}, {"_id": 0, "id": 1},
        )
        if exists:
            continue
        try:
            await db[Collections.AI_SURFACED_ARTICLES].insert_one({
                "id": new_id(),
                "search_session_id": session_id,
                "user_id": user_id,
                "article_id": article_id,
                "advisory_label": "parallel_ai_search",
                "confidence": 0.6,
                "rationale": "Supplemental AI search candidate (not in deterministic results).",
                "original_exclusion_reason": "not_in_deterministic_results",
                "source": "parallel_ai_search",
                "prompt_version": PROMPT_VERSION,
                "created_at": now,
                "updated_at": now,
            })
        except Exception:  # noqa: BLE001
            continue
        if pmid:
            existing_pmids.add(pmid)
        if doi:
            existing_dois.add(doi)
        surfaced += 1

    await audit_service.log(
        "ai_parallel_search_decision",
        user_id=user_id,
        search_session_id=session_id,
        payload={
            "query_preview": query_string[:160],
            "surfaced_count": surfaced,
            "prompt_version": PROMPT_VERSION,
        },
    )
    return surfaced
