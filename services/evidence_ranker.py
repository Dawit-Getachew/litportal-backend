"""EvidenceRankerService — deterministic rank with optional LLM tweak (off).

Composite score:
    overall_rank = 0.55 * relevance + 0.35 * design + 0.05 * recency + 0.05 * full_text

Buckets: high (>=80), moderate (60–79), low (<60). Low are EXCLUDED with
reason 'low_relevance' (post-rank step). Recency:
    last 5y → 100, 5–10y → 70, 10–15y → 40, else 20.
Full-text:
    full_text_available 100, free_full_text_link 70, abstract_only 50, unknown 30.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# Per-question-type design weights → score for each article_type_classification.
_DESIGN_TABLE: dict[str, dict[str, int]] = {
    "treatment": {
        "guideline": 95, "systematic_review": 90, "meta_analysis": 90,
        "rct": 80, "diagnostic_accuracy_study": 50,
        "cohort_study": 60, "case_control": 50, "cross_sectional": 40,
        "narrative_review": 40, "other": 30,
    },
    "prevention": {
        "guideline": 95, "systematic_review": 90, "meta_analysis": 90,
        "rct": 80, "cohort_study": 65, "case_control": 50,
        "cross_sectional": 40, "narrative_review": 40, "other": 30,
        "diagnostic_accuracy_study": 50,
    },
    "screening": {
        "guideline": 95, "systematic_review": 90, "meta_analysis": 90,
        "rct": 80, "cohort_study": 65, "diagnostic_accuracy_study": 75,
        "case_control": 50, "cross_sectional": 40, "narrative_review": 40,
        "other": 30,
    },
    "diagnosis": {
        "guideline": 90, "systematic_review": 90, "meta_analysis": 90,
        "diagnostic_accuracy_study": 85,
        "rct": 60, "cohort_study": 65, "case_control": 60,
        "cross_sectional": 55, "narrative_review": 40, "other": 30,
    },
    "harm": {
        "guideline": 80, "systematic_review": 85, "meta_analysis": 85,
        "rct": 70, "cohort_study": 75, "case_control": 65,
        "cross_sectional": 50, "narrative_review": 40, "other": 30,
        "diagnostic_accuracy_study": 45,
    },
    "prognosis": {
        "guideline": 75, "systematic_review": 85, "meta_analysis": 85,
        "cohort_study": 75, "case_control": 60, "rct": 65,
        "cross_sectional": 50, "narrative_review": 40, "other": 30,
        "diagnostic_accuracy_study": 45,
    },
    "etiology": {
        "guideline": 70, "systematic_review": 80, "meta_analysis": 80,
        "cohort_study": 70, "case_control": 70, "rct": 60,
        "cross_sectional": 55, "narrative_review": 40, "other": 30,
        "diagnostic_accuracy_study": 45,
    },
    "guideline": {
        "guideline": 100, "systematic_review": 75, "meta_analysis": 75,
        "rct": 55, "cohort_study": 50, "narrative_review": 35, "other": 25,
        "diagnostic_accuracy_study": 50, "case_control": 40, "cross_sectional": 35,
    },
    "epidemiology": {
        "guideline": 60, "systematic_review": 80, "meta_analysis": 80,
        "cross_sectional": 70, "cohort_study": 75, "case_control": 60,
        "rct": 50, "narrative_review": 40, "other": 30,
        "diagnostic_accuracy_study": 40,
    },
    "pharmacology": {
        "guideline": 75, "systematic_review": 85, "meta_analysis": 85,
        "rct": 80, "cohort_study": 55, "narrative_review": 45, "other": 30,
        "case_control": 45, "cross_sectional": 40, "diagnostic_accuracy_study": 40,
    },
    "other": {
        "guideline": 80, "systematic_review": 80, "meta_analysis": 80,
        "rct": 70, "cohort_study": 60, "case_control": 50,
        "cross_sectional": 45, "narrative_review": 40, "other": 30,
        "diagnostic_accuracy_study": 50,
    },
}


def _recency_score(year: int | None) -> int:
    if not year:
        return 30
    age = datetime.utcnow().year - year
    if age <= 5:
        return 100
    if age <= 10:
        return 70
    if age <= 15:
        return 40
    return 20


_FULL_TEXT_SCORE = {
    "full_text_available": 100,
    "free_full_text_link": 70,
    "abstract_only": 50,
    "unknown": 30,
}


def _hits(text: str, terms: list[str]) -> int:
    if not text or not terms:
        return 0
    s = text.lower()
    n = 0
    for t in terms:
        t = (t or "").strip().lower()
        if not t:
            continue
        if re.search(rf"\b{re.escape(t)}\b", s):
            n += 1
    return n


def _relevance(article: dict[str, Any], parsed: dict[str, Any], qtypes: list[str]) -> int:
    title = article.get("title") or ""
    abstract = article.get("abstract") or ""
    key = parsed.get("key_concepts") or []
    mesh = parsed.get("mesh_terms") or []
    syn = parsed.get("synonyms") or []
    drugs = parsed.get("drug_names_and_classes") or []
    intervention = (parsed.get("intervention_or_exposure") or "").lower()
    comparator = (parsed.get("comparator") or "").lower()

    # Hit counts (scaled so a well-matched article saturates near 100).
    score = 0
    score += 10 * _hits(title, key)
    score += 5 * _hits(abstract, key)
    score += 6 * _hits(title, mesh + syn + drugs)
    score += 3 * _hits(abstract, mesh + syn + drugs)

    # Query-variant boost
    if "high_precision" in qtypes:
        score += 5
    if "evidence_filtered" in qtypes:
        score += 3
    if "high_recall" in qtypes:
        score += 2
    if "guideline_focused" in qtypes:
        score += 2

    # Bonus: both intervention + comparator present
    full = (title + " " + abstract).lower()
    if intervention and comparator and intervention in full and comparator in full:
        score += 15

    # Floor: if the article was matched by any planned query AND has a non-empty
    # abstract, it must score at least 25 (otherwise design-only top picks like
    # guidelines / SR-MAs without the exact concept tokens get excluded).
    if qtypes and (abstract or title):
        score = max(score, 25)

    return max(0, min(100, score))


def _design(article: dict[str, Any], qtype: str) -> int:
    table = _DESIGN_TABLE.get(qtype) or _DESIGN_TABLE["other"]
    return int(table.get(article.get("article_type_classification") or "other", 30))


def _humanize_qtype_from_picott(picott: dict[str, Any] | None,
                                  legacy_qtype: str,
                                  role_weights: dict | None = None,
                                  multi_intent: dict | None = None) -> str:
    """Phase 6B.2 — delegate to the canonical interpretation helper so
    the rank_explanation "Question type: …" string respects the full
    Clinical Context Engine cascade (PICOTT → role_weights →
    multi_intent → legacy).
    """
    from services.interpretation_canonical import (  # noqa: WPS433
        canonical_label_for_rank_explanation,
    )
    return canonical_label_for_rank_explanation(
        picott=picott,
        role_weights=role_weights,
        multi_intent=multi_intent,
        legacy_qtype=legacy_qtype,
    )


def _explain(
    article: dict[str, Any], relevance: int, design: int, qtype_label: str,
) -> str:
    drivers: list[str] = []
    if article.get("is_guideline"):
        drivers.append("guideline document")
    atype = article.get("article_type_classification") or "other"
    if atype in {"systematic_review", "meta_analysis"}:
        drivers.append(atype.replace("_", " "))
    elif atype == "rct":
        drivers.append("randomized controlled trial")
    elif atype == "diagnostic_accuracy_study":
        drivers.append("diagnostic accuracy study")
    elif atype.endswith("_study") or atype in {"case_control", "cross_sectional"}:
        drivers.append(atype.replace("_", " "))
    if relevance >= 80:
        drivers.append("strong concept match")
    elif relevance >= 60:
        drivers.append("moderate concept match")
    if article.get("full_text_status") == "full_text_available":
        drivers.append("full text available")
    if not drivers:
        drivers.append("background reference")
    return (
        f"Question type: {qtype_label}. "
        + "; ".join(drivers)
        + f". design={design}, relevance={relevance}."
    )


class EvidenceRankerService:
    def rank(
        self,
        parsed: dict[str, Any],
        articles: list[dict[str, Any]],
        query_types_matched_map: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """Return a list of dicts shaped for search_session_articles + a
        view-only ``_article`` reference dict (caller separates persistence)."""
        qtype = parsed.get("clinical_question_type") or "other"
        # Phase 6B.2 — pull PICOTT + role_weights + multi_intent off the
        # parsed payload (pipeline.py attaches them under the same keys
        # the session document carries). Delegate to the canonical
        # helper so the rank_explanation header NEVER displays
        # "Question type: other" when the Clinical Context Engine has
        # a concrete classification.
        picott = parsed.get("picott_json") or {}
        role_weights_attached = (
            parsed.get("role_weights_json")
            or parsed.get("role_weights")
            or {}
        )
        multi_intent_attached = (
            parsed.get("multi_intent_json")
            or parsed.get("multi_intent")
            or {}
        )
        qtype_label = _humanize_qtype_from_picott(
            picott, qtype,
            role_weights=role_weights_attached,
            multi_intent=multi_intent_attached,
        )
        ranked: list[dict[str, Any]] = []
        for art in articles:
            ckey = art.get("canonical_key", "")
            qts = query_types_matched_map.get(ckey, [])
            relevance = _relevance(art, parsed, qts)
            design = _design(art, qtype)
            recency = _recency_score(art.get("publication_year"))
            ft = _FULL_TEXT_SCORE.get(art.get("full_text_status") or "unknown", 30)
            overall = round(0.55 * relevance + 0.35 * design + 0.05 * recency + 0.05 * ft)

            # Phase 2B: multi-source confidence boost. An article retrieved by
            # 2 sources gets +3; 3+ sources gets +5. The boost is additive on
            # top of the composite. We CAP overall at 100 to keep the bucket
            # semantics intact.
            sources_present = list(art.get("sources_present") or [])
            multi_source_boost = 0
            if len(sources_present) >= 3:
                multi_source_boost = 5
            elif len(sources_present) == 2:
                multi_source_boost = 3
            overall_boosted = min(100, overall + multi_source_boost)

            explanation = _explain(art, relevance, design, qtype_label)
            if multi_source_boost:
                explanation += (
                    f" Cross-source confirmation: {len(sources_present)} sources (+{multi_source_boost})."
                )
            ranked.append({
                "_article": art,
                "query_types_matched": qts,
                "flags": list(art.get("flags") or []),
                "relevance_score": float(relevance),
                "evidence_design_score": float(design),
                "overall_rank_score": int(overall_boosted),
                "rank_explanation": explanation,
            })
        # Sort descending by overall_rank_score
        ranked.sort(key=lambda r: r["overall_rank_score"], reverse=True)

        # Apply default selection per spec.
        ranked = self._apply_default_selection(ranked, qtype)

        # Apply post-rank low-relevance exclusion.
        for r in ranked:
            if r.get("exclusion_reason"):
                continue
            if r["overall_rank_score"] < 60:
                r["exclusion_reason"] = "low_relevance"
                r["included_by_default"] = False
                r["user_included"] = False
        return ranked

    @staticmethod
    def _apply_default_selection(ranked: list[dict[str, Any]], qtype: str) -> list[dict[str, Any]]:
        """Mark best guideline, best SR/MA, best primary study for the qtype."""
        primary_type_by_qtype = {
            "treatment": "rct",
            "prevention": "rct",
            "screening": "rct",
            "diagnosis": "diagnostic_accuracy_study",
            "harm": "cohort_study",
            "prognosis": "cohort_study",
            "etiology": "cohort_study",
            "epidemiology": "cohort_study",
            "pharmacology": "rct",
            "guideline": "guideline",
            "other": "rct",
        }
        primary = primary_type_by_qtype.get(qtype, "rct")
        chosen_ids: set[str] = set()

        def _pick_first(predicate) -> dict[str, Any] | None:
            for r in ranked:
                if r.get("exclusion_reason"):
                    continue
                if r["overall_rank_score"] < 60:
                    continue
                a = r["_article"]
                if predicate(a, r):
                    return r
            return None

        # Best guideline
        g = _pick_first(lambda a, r: a.get("is_guideline"))
        if g is not None:
            g["included_by_default"] = True
            chosen_ids.add(id(g))

        # Best SR/MA
        sr = _pick_first(lambda a, r: a.get("article_type_classification") in {"systematic_review", "meta_analysis"} and id(r) not in chosen_ids)
        if sr is not None:
            sr["included_by_default"] = True
            chosen_ids.add(id(sr))

        # Best primary study of the qtype-appropriate kind
        ps = _pick_first(lambda a, r: a.get("article_type_classification") == primary and id(r) not in chosen_ids)
        if ps is not None:
            ps["included_by_default"] = True
            chosen_ids.add(id(ps))

        # Default user_included == included_by_default
        for r in ranked:
            r.setdefault("included_by_default", False)
            r["user_included"] = bool(r.get("included_by_default"))

        return ranked


evidence_ranker = EvidenceRankerService()
