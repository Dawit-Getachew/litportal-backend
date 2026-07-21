"""AI Agent 3 — Evidence Curator / Surfacing + Ranker Optimization (B4 / B4a).

Problem it solves: the deterministic ranker sometimes excludes genuinely good
papers (e.g. a strong review whose title/abstract lacks the literal query terms),
and some excluded results are in fact relevant. This agent reviews the
deterministically-EXCLUDED (capped / low-relevance) candidates and surfaces the
genuinely-relevant ones onto a SEPARATE "AI-surfaced" shelf, with an advisory
label (direct / borderline / indirect) and a broader-guideline-topic mapping.

Hard governance limits enforced here:
  * Operates only on already-retrieved, safety-filtered articles (title+abstract
    metadata). It NEVER calls a source.
  * It NEVER modifies the deterministic included/excluded lists, never reorders,
    never selects. It only writes advisory rows to a separate collection.
  * Retracted / preprint / unsafe / non-article exclusions are NEVER surfaced.
  * The model references candidates by integer index only — it cannot fabricate
    PMIDs/DOIs.
Feature-flagged (default OFF), fail-closed. Confidence-gated by
MIN_AI_SURFACING_CONFIDENCE.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.base import run_json_agent, store_agent_decision
from agents.schemas import EvidenceCurationResult
from core.config import get_settings
from core.db import Collections, get_db
from models.common import new_id, utcnow
from services.audit import audit_service

logger = logging.getLogger("litportal.agents.evidence_curator")

PROMPT_VERSION = "evcurate-v1"
PROMPT_VERSION_RESCUE = "evrescue-v1"

# Exclusion reasons that must NEVER be surfaced (safety / integrity), no matter
# what the model thinks. Everything else (cap_reason_*, relevance_below_60,
# negative_concept_fired) is a "soft" exclusion eligible for advisory surfacing.
_HARD_EXCLUSIONS = {
    "excluded_retracted_or_preprint_or_unsafe",
    "retracted_publication",
    "retraction_notice",
    "preprint_excluded_by_policy",
    "non_article_record",
}

_SURFACE_LABELS = {"direct_candidate", "borderline_candidate", "indirect_candidate"}

_SYSTEM = (
    "You are an evidence CURATOR for a clinical literature search engine. You review "
    "papers the deterministic ranker EXCLUDED and decide which are actually relevant "
    "to the clinician's question and should be SURFACED for review. "
    "You DO NOT retrieve sources, you DO NOT select or re-rank the main results, and "
    "you DO NOT invent PMIDs/DOIs/citations — you only refer to candidates by their "
    "given integer index. Return STRICT JSON only.\n\n"
    "For each candidate, output an advisory_label: 'direct_candidate' (clearly "
    "answers the question), 'borderline_candidate' (plausibly relevant), "
    "'indirect_candidate' (related background), or 'keep_excluded' (not relevant). "
    "Give a confidence in [0,1] and a one-sentence rationale (no identifiers). "
    "Also map the question to broader_guideline_topics (e.g. 'sepsis management', "
    "'anticoagulation') and set parallel_ai_search_recommended=true ONLY if the "
    "excluded set is weak and a supplemental search would clearly help."
)

_SCHEMA_HINT = {
    "surfaced": [{"index": 0, "advisory_label": "direct_candidate|borderline_candidate|indirect_candidate|keep_excluded", "confidence": 0.0, "rationale": "string"}],
    "broader_guideline_topics": ["string"],
    "parallel_ai_search_recommended": False,
    "reason_codes": ["string"],
    "confidence": 0.0,
}


def _collect_candidates(ranked: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Soft-excluded candidates (capped/low-relevance), capped at ``cap``.

    Sources whose policy blocks AI ranker-optimization (e.g. Cochrane, HHS) are
    excluded from advisory decisions per governance (B1/B16 tie-in)."""
    from services.downstream_eligibility import can_use_article_for_tool

    candidates: list[dict[str, Any]] = []
    for r in ranked:
        excl = r.get("exclusion_reason")
        if not excl or excl in _HARD_EXCLUSIONS:
            continue
        art = r.get("_article") or {}
        if not art.get("id"):
            continue
        if not can_use_article_for_tool(art, "ranker_optimization")["allowed"]:
            continue  # source policy blocks AI ranker-optimization
        candidates.append({
            "index": len(candidates),
            "article_id": art["id"],
            "title": (art.get("title") or "")[:300],
            "abstract": (art.get("abstract") or "")[:1500],
            "journal": art.get("journal") or "",
            "year": art.get("publication_year"),
            "exclusion_reason": excl,
        })
        if len(candidates) >= cap:
            break
    return candidates


def _candidates_block(candidates: list[dict[str, Any]]) -> str:
    lines = []
    for c in candidates:
        lines.append(
            f"[{c['index']}] ({c.get('year') or 'n/a'}) {c['title']}\n"
            f"    journal: {c['journal']} | excluded_because: {c['exclusion_reason']}\n"
            f"    abstract: {c['abstract'] or '(no abstract)'}"
        )
    return "\n".join(lines)


async def maybe_curate_evidence(
    session_id: str,
    user_id: str,
    sanitized_question: str,
    ranked: list[dict[str, Any]],
) -> None:
    """Run the evidence curator (if enabled) over deterministically-excluded
    candidates and persist advisory surfacing rows. Never mutates ``ranked`` and
    never raises."""
    settings = get_settings()
    if not settings.AI_EVIDENCE_SURFACING_ENABLED:
        return

    candidates = _collect_candidates(ranked, settings.MAX_AI_SURFACING_CANDIDATES)
    if not candidates:
        return

    user_prompt = (
        f"Clinical question:\n{sanitized_question}\n\n"
        f"Excluded candidates (review each by index):\n{_candidates_block(candidates)}"
    )
    model, reason = await run_json_agent(
        system=_SYSTEM,
        user=user_prompt,
        schema_hint=_SCHEMA_HINT,
        validator=EvidenceCurationResult.model_validate,
        max_tokens=1600,
        session_id=session_id,
    )

    surfaced_count = 0
    if model is not None:
        min_conf = settings.MIN_AI_SURFACING_CONFIDENCE
        db = get_db()
        now = utcnow()
        for sc in model.surfaced:
            if sc.advisory_label not in _SURFACE_LABELS:
                continue
            if sc.confidence < min_conf:
                continue
            if sc.index < 0 or sc.index >= len(candidates):
                continue
            cand = candidates[sc.index]
            try:
                await db[Collections.AI_SURFACED_ARTICLES].insert_one({
                    "id": new_id(),
                    "search_session_id": session_id,
                    "user_id": user_id,
                    "article_id": cand["article_id"],
                    "advisory_label": sc.advisory_label,
                    "confidence": sc.confidence,
                    "rationale": sc.rationale,
                    "original_exclusion_reason": cand["exclusion_reason"],
                    "prompt_version": PROMPT_VERSION,
                    "created_at": now,
                    "updated_at": now,
                })
                surfaced_count += 1
            except Exception:  # noqa: BLE001
                logger.exception("ai_surfaced insert failed (non-fatal)")

    # Session-level advisory (broader topics + parallel-search recommendation).
    await store_agent_decision(
        agent="evidence_curator",
        session_id=session_id,
        user_id=user_id,
        decision=(model.model_dump() if model is not None else None),
        applied=False,  # advisory shelf only — never alters deterministic results
        failure_reason=reason,
    )
    await audit_service.log(
        "ai_evidence_surfacing_decision",
        user_id=user_id,
        search_session_id=session_id,
        payload={
            "candidates_reviewed": len(candidates),
            "surfaced_count": surfaced_count,
            "broader_topics": (model.broader_guideline_topics if model else []),
            "parallel_ai_search_recommended": (model.parallel_ai_search_recommended if model else False),
            "failure_reason": reason,
            "prompt_version": PROMPT_VERSION,
        },
    )

    # B4a part 3 — when recommended AND enabled, run a bounded supplemental
    # PubMed search and route new candidates onto the same advisory shelf.
    if (
        model is not None
        and model.parallel_ai_search_recommended
        and settings.AI_RANKER_PARALLEL_SEARCH_ENABLED
    ):
        try:
            from agents.parallel_search import maybe_run_parallel_search

            existing_pmids: set[str] = set()
            existing_dois: set[str] = set()
            for r in ranked:
                a = r.get("_article") or {}
                if a.get("pmid"):
                    existing_pmids.add(str(a["pmid"]).strip())
                d = a.get("doi_normalized") or a.get("doi") or ""
                if d:
                    existing_dois.add(str(d).strip().lower())
            await maybe_run_parallel_search(
                session_id, user_id, sanitized_question,
                model.broader_guideline_topics, existing_pmids, existing_dois,
            )
        except Exception:  # noqa: BLE001
            logger.exception("parallel_ai_search hook failed (non-fatal)")


# ── Agent 3b: rescued-article RE-ADD (active re-inclusion) ──────────────────
#
# The evidence curator above only writes to an advisory shelf. This function is
# the missing write-back half the surfacing shelf never performs: it reuses the
# SAME candidate collection + SAME single batched LLM call, but for each
# high-confidence ``direct_candidate`` that also passes a central-anchor guard,
# it MUTATES the matching ``ranked`` row in place — clearing the SOFT
# exclusion_reason so the paper flows into body['articles'] when the SSA rows
# are built. Governance is preserved end-to-end:
#   * _collect_candidates already drops hard/safety/policy exclusions and
#     source-policy-blocked papers BEFORE the LLM ever sees them, so a
#     retracted/preprint/non-article paper can never be re-added.
#   * We only ever CLEAR a soft exclusion; we never set a new one and never
#     touch a hard exclusion.
#   * The central-anchor guard blocks wrong-disease promotion.
#   * Bounded by MIN_CONFIDENCE (0.75, stricter than surfacing) and MAX_ADDS.
# Feature-flagged (default OFF), prod requires a second opt-in, fail-closed.


def _central_anchor_terms(picott: dict[str, Any] | None) -> list[str]:
    """Lowercased condition/exposure/index_test anchor terms from PICOTT.

    An empty list means we could not derive a disease anchor → the guard is
    permissive (mirrors default_selection, which does not block when it cannot
    validate)."""
    terms: list[str] = []
    for mh in ((picott or {}).get("must_have_concepts") or []):
        if not isinstance(mh, dict):
            continue
        role = (mh.get("role") or "").strip().lower()
        if role not in {"condition", "exposure", "index_test"}:
            continue
        for t in [mh.get("concept") or ""] + list(mh.get("synonyms") or []):
            t = (t or "").strip().lower()
            if len(t) >= 3:
                terms.append(t)
    return terms


def _passes_central_anchor(article: dict[str, Any], anchor_terms: list[str]) -> bool:
    """True if the article title/abstract mentions at least one central
    (disease/exposure/index-test) anchor — the wrong-disease guard."""
    if not anchor_terms:
        return True
    hay = (
        (article.get("title") or "") + " " + (article.get("abstract") or "")
    ).lower()
    return any(t in hay for t in anchor_terms)


async def maybe_rescue_readd(
    session_id: str,
    user_id: str,
    sanitized_question: str,
    ranked: list[dict[str, Any]],
    picott: dict[str, Any] | None = None,
) -> int:
    """Re-add high-confidence deterministically-rejected papers to the returned
    list (if enabled). MUTATES ``ranked`` in place and returns the number of
    rows re-added. Never raises; a no-op when the flag is off, in production
    without the prod opt-in, or when the LLM is unconfigured."""
    settings = get_settings()
    if not settings.AI_RESCUE_READD_ENABLED:
        return 0
    # Production requires an explicit SECOND opt-in — this is the only AI path
    # that changes the clinician-visible article list.
    is_prod = (getattr(settings, "APP_ENV", "development") or "").strip().lower() == "production"
    if is_prod and not settings.AI_RESCUE_READD_PROD_ALLOWED:
        return 0

    # Optional cost/latency lever: skip when the strict pipeline already
    # produced a healthy shown set. Default 0 → always run (every search).
    skip_if_atleast = int(getattr(settings, "AI_RESCUE_READD_SKIP_IF_INCLUDED_ATLEAST", 0) or 0)
    if skip_if_atleast > 0:
        already_shown = sum(1 for r in ranked if not r.get("exclusion_reason"))
        if already_shown >= skip_if_atleast:
            return 0

    candidates = _collect_candidates(ranked, settings.AI_RESCUE_READD_MAX_CANDIDATES)
    if not candidates:
        return 0

    user_prompt = (
        f"Clinical question:\n{sanitized_question}\n\n"
        f"Excluded candidates (review each by index):\n{_candidates_block(candidates)}"
    )
    model, reason = await run_json_agent(
        system=_SYSTEM,
        user=user_prompt,
        schema_hint=_SCHEMA_HINT,
        validator=EvidenceCurationResult.model_validate,
        max_tokens=1600,
        session_id=session_id,
    )
    if model is None:
        return 0

    # Map article_id → the mutable ranked row.
    row_by_article_id: dict[str, dict[str, Any]] = {}
    for r in ranked:
        aid = (r.get("_article") or {}).get("id")
        if aid:
            row_by_article_id[aid] = r

    min_conf = float(settings.AI_RESCUE_READD_MIN_CONFIDENCE)
    max_adds = int(settings.AI_RESCUE_READD_MAX_ADDS)
    autocheck = bool(getattr(settings, "AI_RESCUE_READD_AUTOCHECK", True))
    anchor_terms = _central_anchor_terms(picott)

    added = 0
    # Highest-confidence first so the MAX_ADDS budget goes to the best rows.
    for sc in sorted(model.surfaced, key=lambda s: -float(s.confidence or 0.0)):
        if added >= max_adds:
            break
        if sc.advisory_label != "direct_candidate":
            continue
        if sc.confidence < min_conf:
            continue
        if sc.index < 0 or sc.index >= len(candidates):
            continue
        cand = candidates[sc.index]
        r = row_by_article_id.get(cand["article_id"])
        if r is None:
            continue
        excl = r.get("exclusion_reason")
        # Only ever CLEAR a SOFT exclusion; never touch a hard/safety exclusion.
        if not excl or excl in _HARD_EXCLUSIONS:
            continue
        art = r.get("_article") or {}
        if not _passes_central_anchor(art, anchor_terms):
            continue
        # --- Re-inclusion mutation (same shape as the manual PATCH include
        # path in routers/search_sessions.py): clear the soft exclusion so the
        # row lands in body['articles'], optionally pre-check it, and stamp
        # provenance so it is auditable and UI-distinguishable. ---
        r["exclusion_reason"] = None
        if autocheck:
            r["included_by_default"] = True
        r["llm_rescued"] = True
        r["default_selection_rationale"] = "llm_rescued"
        added += 1

    if added:
        logger.info(
            "ai_rescue_readd: session=%s re-added %d of %d rejected candidates",
            session_id, added, len(candidates),
        )
    try:
        await audit_service.log(
            "ai_rescue_readd_decision",
            user_id=user_id,
            search_session_id=session_id,
            payload={
                "candidates_reviewed": len(candidates),
                "re_added_count": added,
                "min_confidence": min_conf,
                "max_adds": max_adds,
                "autocheck": autocheck,
                "failure_reason": reason,
                "prompt_version": PROMPT_VERSION_RESCUE,
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("ai_rescue_readd audit failed (non-fatal)")
    return added
