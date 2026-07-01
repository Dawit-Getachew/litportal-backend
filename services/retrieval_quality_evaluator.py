"""RetrievalQualityEvaluatorService — Phase 5A: PICOTT-aware anchor coverage,
study-type target coverage, and rescue-query trigger."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any


_GUIDELINE_QTYPES = {"treatment", "prevention", "screening"}


def _matches_any(haystack: str, needles: list[str]) -> bool:
    if not haystack:
        return False
    s = haystack.lower()
    for n in needles:
        t = (n or "").strip().lower()
        if not t:
            continue
        if re.search(rf"\b{re.escape(t)}\b", s):
            return True
    return False


def _anchor_coverage_at_k(
    articles: list[dict[str, Any]], picott: dict[str, Any], k: int
) -> dict[str, Any]:
    """Return ``{coverage_pct, all_anchors_covered, missing_anchors_in_topk}``
    for the top-k by overall_rank_score.
    """
    must = picott.get("must_have_concepts") or []
    anchors = [(m.get("concept") or "", [m.get("concept") or ""] + list(m.get("synonyms") or []))
                for m in must]
    if not anchors:
        return {"k": k, "coverage_pct": 100.0, "all_anchors_covered": True,
                 "missing_anchors_in_topk": []}
    topk = sorted(articles, key=lambda r: r.get("overall_rank_score") or 0, reverse=True)[:k]
    covered: list[str] = []
    for concept, terms in anchors:
        # An anchor is "covered" in top-k if at least one top-k article
        # mentions it (title + abstract).
        for r in topk:
            art = r.get("_article") or {}
            body = (art.get("title") or "") + " " + (art.get("abstract") or "")
            if _matches_any(body, terms):
                covered.append(concept)
                break
    missing = [c for c, _ in anchors if c not in covered]
    return {
        "k": k,
        "coverage_pct": 100.0 * len(covered) / max(1, len(anchors)),
        "all_anchors_covered": len(missing) == 0,
        "missing_anchors_in_topk": missing,
    }


class RetrievalQualityEvaluatorService:
    def evaluate(
        self,
        parsed: dict[str, Any],
        queries: list[dict[str, Any]],
        articles_after_filter: list[dict[str, Any]],
        raw_total: int = 0,
        *,
        per_source_counts: dict[str, int] | None = None,
        sources_searched: list[str] | None = None,
        picott: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        problems: list[str] = []
        qtype = (parsed.get("clinical_question_type") or "other").lower()
        per_source_counts = per_source_counts or {}
        sources_searched = sources_searched or []

        high_or_moderate = [
            a for a in articles_after_filter
            if (a.get("overall_rank_score") or 0) >= 60 and not a.get("exclusion_reason")
        ]
        n_relevant = len(high_or_moderate)
        n_guideline = sum(1 for a in articles_after_filter if a.get("_article", {}).get("is_guideline"))

        # Sources actually contributing evidence (i.e. articles with that source listed).
        sources_with_evidence: set[str] = set()
        cross_source_confirmed = 0
        for a in articles_after_filter:
            art = a.get("_article", {})
            srcs = list(art.get("sources_present") or [])
            for s in srcs:
                sources_with_evidence.add(s)
            if len(srcs) >= 2:
                cross_source_confirmed += 1

        sufficient = True
        action = "continue"
        reason = "Initial retrieval produced sufficient ranked evidence."

        if n_relevant < 3:
            sufficient = False
            problems.append("fewer_than_three_high_or_moderate_relevance")
            action = "broaden_query"
            reason = "Fewer than 3 high/moderate-relevance articles after one retrieval round."

        if n_guideline == 0 and qtype in _GUIDELINE_QTYPES:
            problems.append("no_guideline_found")
            if action == "continue":
                action = "search_guidelines_later"
                reason = (
                    "No practice guidelines surfaced; later phases will search dedicated "
                    "guideline portals (USPSTF/NICE/WHO/CDC)."
                )

        years = []
        for a in articles_after_filter:
            art = a.get("_article") or {}
            y = art.get("publication_year")
            if isinstance(y, int):
                years.append(y)
        if years:
            current_year = datetime.utcnow().year
            if all((current_year - y) > 10 for y in years):
                problems.append("no_recent_evidence")
                if action == "continue":
                    action = "broaden_query"
                    reason = "All retrieved articles are older than 10 years."

        if raw_total > 200 and len(articles_after_filter) < 5:
            problems.append("noisy_retrieval_low_inclusion")
            action = "narrow_query"
            reason = "Large raw hit count (>200) but few records survived dedup + safety."

        # Single-source warning when more than one source was attempted.
        if len(sources_searched) > 1 and len(sources_with_evidence) <= 1:
            problems.append("single_source_only")

        # ---- Phase 5A — PICOTT-aware diagnostics ----
        anchor_top10: dict[str, Any] = {}
        anchor_top20: dict[str, Any] = {}
        rescue_needed = False
        rescue_must_haves: list[str] = []
        broad_recall_only_dominates = False
        if picott and picott.get("must_have_concepts"):
            anchor_top10 = _anchor_coverage_at_k(articles_after_filter, picott, 10)
            anchor_top20 = _anchor_coverage_at_k(articles_after_filter, picott, 20)

            # Rescue is needed when top-10 misses any must-have anchor.
            if not anchor_top10.get("all_anchors_covered", True):
                rescue_needed = True
                rescue_must_haves = list(anchor_top10.get("missing_anchors_in_topk") or [])
                problems.append("top10_missing_must_have_anchors")
                sufficient = False
                if action in ("continue", "broaden_query", "search_guidelines_later"):
                    action = "trigger_rescue_query"
                    reason = (
                        "Top-10 results miss at least one mandatory anchor concept "
                        "(" + ", ".join(rescue_must_haves[:3]) + "). Running a rescue "
                        "query that AND-joins must-have concepts with key synonyms."
                    )

            # Detect broad-recall-only domination — if every top-20 article was
            # retrieved ONLY via broad_recall or legacy high_recall, the
            # ranking has lost focus.
            top20 = sorted(articles_after_filter, key=lambda r: r.get("overall_rank_score") or 0, reverse=True)[:20]
            if top20:
                broad_only = sum(
                    1 for r in top20 if set(r.get("query_family_matches") or [])
                    <= {"broad_recall", "high_recall"}
                )
                if broad_only / len(top20) > 0.8:
                    broad_recall_only_dominates = True
                    problems.append("broad_recall_dominates_topk")

            # Study-type target coverage — does any top-10 article use one of
            # the pathway-preferred study types?
            preferred = set((picott.get("ideal_study_types") or [])[:5])
            if preferred:
                top10 = sorted(articles_after_filter, key=lambda r: r.get("overall_rank_score") or 0, reverse=True)[:10]
                target_hits = sum(
                    1 for r in top10
                    if (r.get("_article") or {}).get("article_type_classification") in preferred
                )
                if target_hits == 0:
                    problems.append("no_preferred_study_type_in_top10")

        return {
            "retrieval_sufficient": sufficient,
            "reason": reason,
            "problems": problems,
            "recommended_next_action": action,
            "counts": {
                "raw_total": raw_total,
                "after_filter_total": len(articles_after_filter),
                "high_or_moderate_relevance": n_relevant,
                "guideline_articles": n_guideline,
                "cross_source_confirmed": cross_source_confirmed,
            },
            "source_coverage": {
                "sources_searched": list(sources_searched),
                "per_source_raw_counts": dict(per_source_counts),
                "sources_with_evidence": sorted(sources_with_evidence),
                "cross_source_confirmed_articles": cross_source_confirmed,
            },
            "picott_diagnostics": {
                "anchor_coverage_top10": anchor_top10,
                "anchor_coverage_top20": anchor_top20,
                "rescue_query_needed": rescue_needed,
                "rescue_must_haves": rescue_must_haves,
                "broad_recall_only_dominates_topk": broad_recall_only_dominates,
            },
        }


retrieval_quality_evaluator = RetrievalQualityEvaluatorService()
