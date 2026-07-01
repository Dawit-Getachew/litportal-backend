"""B1 — Shared Source Permission Resolver.

Returns the full permission set for any source (general, not Cochrane-only):
``retrieval_allowed``, ``link_out_allowed``, ``display_snippet_allowed``,
``ai_semantic_screening_allowed``, ``ai_summary_ingestion_allowed``,
``ai_ranker_optimization_allowed``, ``allowed_text_scope``, ``source_policy_reason``.

Policy (Northstar §12.1, fail-closed):
  * PubMed + PubMed-indexed guidelines → AI permitted at ``title_plus_abstract``.
  * Cochrane, MyHealthfinder/HHS → AI ingestion BLOCKED (metadata + link-out only).
  * Everything else → fail-closed for AI (no AI use) until explicitly allowed.

``resolve()`` is deterministic and synchronous (no network/DB). ``resolve_async()``
overlays an explicit override row from the GUIDELINE_SOURCES registry when present.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("litportal.source_permissions")

_AI_SCOPES = {"none", "title_only", "title_plus_abstract", "full_text"}


class SourcePermission(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source: str
    retrieval_allowed: bool = False
    link_out_allowed: bool = True
    display_snippet_allowed: str = "verify"  # yes | no | verify
    ai_semantic_screening_allowed: bool = False
    ai_summary_ingestion_allowed: bool = False
    ai_ranker_optimization_allowed: bool = False
    allowed_text_scope: str = "none"
    source_policy_reason: str = "default_fail_closed_ai"


def _ai_allowed(scope: str = "title_plus_abstract", reason: str = "") -> dict:
    return {
        "retrieval_allowed": True,
        "link_out_allowed": True,
        "display_snippet_allowed": "yes",
        "ai_semantic_screening_allowed": True,
        "ai_summary_ingestion_allowed": True,
        "ai_ranker_optimization_allowed": True,
        "allowed_text_scope": scope,
        "source_policy_reason": reason,
    }


def _ai_blocked(reason: str, *, retrieval: bool = True, snippet: str = "verify") -> dict:
    return {
        "retrieval_allowed": retrieval,
        "link_out_allowed": True,
        "display_snippet_allowed": snippet,
        "ai_semantic_screening_allowed": False,
        "ai_summary_ingestion_allowed": False,
        "ai_ranker_optimization_allowed": False,
        "allowed_text_scope": "none",
        "source_policy_reason": reason,
    }


# Built-in policy. Keys are normalized source identifiers.
# PubMed + PubMed-indexed content and the biomedical abstract aggregators are
# AI-permitted at title+abstract; Cochrane + HHS are blocked; unknown sources
# fail closed.
_POLICY: dict[str, dict] = {
    "pubmed": _ai_allowed(reason="pubmed_title_abstract_permitted"),
    "pubmed_guidelines": _ai_allowed(reason="pubmed_guidelines_title_abstract_permitted"),
    "europepmc": _ai_allowed(reason="europepmc_title_abstract_permitted"),
    "openalex": _ai_allowed(reason="openalex_title_abstract_permitted"),
    "icite": _ai_allowed(reason="icite_title_abstract_permitted"),
    "crossref": _ai_allowed(reason="crossref_title_abstract_permitted"),
    "semantic_scholar": _ai_allowed(reason="semantic_scholar_title_abstract_permitted"),
    "cochrane": _ai_blocked("cochrane_ai_ingestion_restricted"),
    "myhealthfinder": _ai_blocked("hhs_ai_ingestion_restricted"),
    "hhs": _ai_blocked("hhs_ai_ingestion_restricted"),
}

# Source-name aliases → canonical policy key.
_ALIASES: dict[str, str] = {
    "ncbi": "pubmed",
    "pubmed_central": "pubmed",
    "pmc": "pubmed",
    "medline": "pubmed",
    "pubmed_guideline_lane": "pubmed_guidelines",
    "guidelines": "pubmed_guidelines",
    "cochrane_review": "cochrane",
    "cochrane_library": "cochrane",
    "health.gov": "hhs",
    "healthfinder": "myhealthfinder",
    "odphp": "hhs",
    "europe_pmc": "europepmc",
    "openalex_works": "openalex",
    "s2": "semantic_scholar",
}


def _normalize(source_name: str) -> str:
    s = (source_name or "").strip().lower().replace(" ", "_")
    return _ALIASES.get(s, s)


def resolve(source_name: str) -> SourcePermission:
    """Resolve a source's permissions from the built-in policy (no DB/network)."""
    key = _normalize(source_name)
    if key in _POLICY:
        return SourcePermission(source=key, **_POLICY[key])
    # Unknown source → strict fail-closed (no AI; link-out only).
    return SourcePermission(source=key, **_ai_blocked(
        "unknown_source_fail_closed", retrieval=True, snippet="verify",
    ))


def merge_most_restrictive(perms: list[SourcePermission]) -> SourcePermission:
    """AND the AI/snippet planes across multiple sources (most restrictive wins).

    Used for an article present in several sources — the strictest source governs
    AI use. ``allowed_text_scope`` collapses to the lowest scope present.
    """
    if not perms:
        return SourcePermission(source="(none)")
    scope_rank = {"none": 0, "title_only": 1, "title_plus_abstract": 2, "full_text": 3}
    snippet_rank = {"no": 0, "verify": 1, "yes": 2}
    out = SourcePermission(
        source="+".join(p.source for p in perms),
        retrieval_allowed=all(p.retrieval_allowed for p in perms),
        link_out_allowed=all(p.link_out_allowed for p in perms),
        ai_semantic_screening_allowed=all(p.ai_semantic_screening_allowed for p in perms),
        ai_summary_ingestion_allowed=all(p.ai_summary_ingestion_allowed for p in perms),
        ai_ranker_optimization_allowed=all(p.ai_ranker_optimization_allowed for p in perms),
        allowed_text_scope=min(perms, key=lambda p: scope_rank.get(p.allowed_text_scope, 0)).allowed_text_scope,
        display_snippet_allowed=min(perms, key=lambda p: snippet_rank.get(p.display_snippet_allowed, 1)).display_snippet_allowed,
        source_policy_reason="; ".join(
            sorted({p.source_policy_reason for p in perms if p.source_policy_reason})
        ) or "merged",
    )
    return out


def resolve_for_sources(source_names: list[str]) -> SourcePermission:
    """Resolve + merge (most-restrictive) for a list of source identifiers."""
    perms = [resolve(s) for s in (source_names or []) if s]
    if not perms:
        # No source info → fail-closed for AI.
        return SourcePermission(source="(none)", **_ai_blocked("no_source_info_fail_closed"))
    return merge_most_restrictive(perms)


async def resolve_async(source_name: str) -> SourcePermission:
    """Policy resolve, overlaid with an explicit GUIDELINE_SOURCES override row
    when one exists for this source (granular flags win over policy)."""
    base = resolve(source_name)
    try:
        from core.db import Collections, get_db
        row = await get_db()[Collections.GUIDELINE_SOURCES].find_one(
            {"source_name": _normalize(source_name)}, {"_id": 0},
        )
    except Exception:  # noqa: BLE001 — registry is an optional overlay
        return base
    if not row:
        return base
    merged = base.model_dump()
    for fld in (
        "retrieval_allowed", "link_out_allowed", "display_snippet_allowed",
        "ai_semantic_screening_allowed", "ai_summary_ingestion_allowed",
        "ai_ranker_optimization_allowed", "allowed_text_scope", "source_policy_reason",
    ):
        if fld in row and row[fld] not in (None, ""):
            merged[fld] = row[fld]
    # A registry row with ai_ingestion_allowed=False forces AI planes off.
    if row.get("ai_ingestion_allowed") is False:
        merged["ai_semantic_screening_allowed"] = False
        merged["ai_summary_ingestion_allowed"] = False
        merged["ai_ranker_optimization_allowed"] = False
        merged["allowed_text_scope"] = "none"
    merged["source"] = _normalize(source_name)
    return SourcePermission(**merged)
