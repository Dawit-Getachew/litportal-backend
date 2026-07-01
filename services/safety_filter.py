"""SafetyFilterService — pre-rank exclusions + flag annotations.

Phase 5F refinement #4 — closed-set non-article publication types:
EDITORIALS, LETTERS, and COMMENTS are NOT excluded any more. They are
kept (a clinician should still see substantive commentary on an ARDS-
corticosteroid debate) but flagged so the ranker can demote them below
RCTs / SRs / guidelines.

True non-article records that REMAIN excluded:
- patent
- grant / funding announcement
- erratum / published-erratum
- corrected-and-republished-article
- retraction-notice / retraction-of-publication
- dataset
"""
from __future__ import annotations

from typing import Any


# Phase 5F refinement #4 — closed-set. Only TRUE non-article records.
_NON_ARTICLE_PUB_TYPES = {
    "patent",
    "grant",
    "erratum",
    "published-erratum",
    "published erratum",
    "corrected-and-republished-article",
    "corrected and republished article",
    "retraction-notice",
    "retraction-of-publication",
    "retraction notice",
    "retraction of publication",
    "dataset",
}

# Phase 5F refinement #4 — soft-flag categories. NOT excluded; the ranker
# demotes these below RCT / SR / guideline tiers via existing logic.
_EDITORIAL_OR_COMMENTARY_PUB_TYPES = {
    "editorial",
    "letter",
    "comment",
}


class SafetyFilterService:
    def pre_rank_filter(
        self,
        articles: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split into (kept_with_flags, excluded). Both lists carry the original
        article dict; excluded items get ``exclusion_reason``; kept items may
        get ``flags`` populated. The articles are not mutated in place."""
        kept: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for a in articles:
            reason = self._exclusion_reason(a)
            if reason:
                a2 = dict(a)
                a2["exclusion_reason"] = reason
                excluded.append(a2)
                continue
            a2 = dict(a)
            flags = self._flags(a)
            if flags:
                a2["flags"] = flags
            kept.append(a2)
        return kept, excluded

    @staticmethod
    def _exclusion_reason(a: dict[str, Any]) -> str | None:
        if a.get("is_preprint"):
            return "preprint_excluded_by_policy"
        if a.get("is_retracted"):
            return "retracted_publication"
        if a.get("is_retraction_notice"):
            return "retraction_notice"
        pts_raw = a.get("publication_types_json") or []
        # Normalize: case-insensitive matching against the closed-set
        # non-article list. Mirrors PubMed's mixed-case publication-type
        # strings (e.g., "Published Erratum", "Corrected and Republished
        # Article").
        pts_lower = {(p or "").strip().lower() for p in pts_raw}
        if pts_lower & _NON_ARTICLE_PUB_TYPES:
            return "non_article_record"
        return None

    @staticmethod
    def _flags(a: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        if a.get("has_expression_of_concern"):
            flags.append("expression_of_concern")
        if not a.get("abstract"):
            flags.append("missing_abstract")
        # Phase 5F — editorial/letter/comment annotation. Kept (not
        # excluded); the ranker tier-demotes via existing rules.
        pts_lower = {
            (p or "").strip().lower()
            for p in (a.get("publication_types_json") or [])
        }
        if pts_lower & _EDITORIAL_OR_COMMENTARY_PUB_TYPES:
            flags.append("editorial_or_commentary")
        return flags


safety_filter = SafetyFilterService()
