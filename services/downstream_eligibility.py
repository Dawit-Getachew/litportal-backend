"""B16 — Downstream eligibility helper.

``can_use_article_for_tool(article, tool)`` answers, for a given article and a
downstream tool (LitScholar, LitStream, summary, forum draft, link-out, snippet
display, ranker optimization, semantic screening), whether the article's
source policy permits it — using the shared B1 source-permission resolver.

Returns a closed-set verdict ``{allowed, reason, source, allowed_text_scope}``.
An article present in multiple sources is governed by the MOST RESTRICTIVE one.
"""
from __future__ import annotations

from typing import Any

from services.source_permission_resolver import resolve_for_sources

# Tool → the permission plane it requires.
_TOOL_PLANE: dict[str, str] = {
    "summary": "ai_summary_ingestion_allowed",
    "litscholar": "ai_summary_ingestion_allowed",
    "litstream": "ai_summary_ingestion_allowed",
    "forum_draft": "ai_summary_ingestion_allowed",
    "ai_summary_ingestion": "ai_summary_ingestion_allowed",
    "ranker_optimization": "ai_ranker_optimization_allowed",
    "ai_semantic_screening": "ai_semantic_screening_allowed",
    "link_out": "link_out_allowed",
    "display_snippet": "display_snippet_allowed",
}

VALID_TOOLS = frozenset(_TOOL_PLANE)


def article_sources(article: dict[str, Any]) -> list[str]:
    """Best-effort extraction of an article's source identifiers."""
    if not isinstance(article, dict):
        return []
    srcs = article.get("sources_present")
    if isinstance(srcs, list) and srcs:
        return [str(s) for s in srcs if s]
    sid = article.get("source_ids_json")
    if isinstance(sid, dict) and sid:
        return [str(k) for k in sid.keys()]
    if isinstance(sid, list) and sid:
        return [str(s) for s in sid if s]
    return []


def can_use_article_for_tool(article: dict[str, Any], tool: str) -> dict[str, Any]:
    """Return ``{allowed, reason, source, allowed_text_scope, source_policy_reason}``."""
    plane = _TOOL_PLANE.get(tool)
    if plane is None:
        return {
            "allowed": False, "tool": tool, "reason": "unknown_tool",
            "source": "(n/a)", "allowed_text_scope": "none", "source_policy_reason": "unknown_tool",
        }
    perm = resolve_for_sources(article_sources(article))
    if plane == "display_snippet_allowed":
        allowed = perm.display_snippet_allowed == "yes"
        reason = "snippet_permitted" if allowed else f"snippet_{perm.display_snippet_allowed}"
    else:
        allowed = bool(getattr(perm, plane))
        reason = f"{tool}_permitted" if allowed else perm.source_policy_reason
    return {
        "allowed": allowed,
        "tool": tool,
        "reason": reason,
        "source": perm.source,
        "allowed_text_scope": perm.allowed_text_scope,
        "source_policy_reason": perm.source_policy_reason,
    }
