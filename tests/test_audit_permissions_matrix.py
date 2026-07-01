"""SCOPE AUDIT — B1 / B2 / B16 permission + eligibility + governance matrix.

Pure-logic, no app boot / DB / network → hundreds of fast parametrized cases.

Independently re-derives the expected policy (a second source of truth) and
asserts the resolver/eligibility helper agree across:
  * every policy source, every alias, and many unknown sources × every plane (B1)
  * every source × every downstream tool (B16)
  * most-restrictive merges across source combinations (B1)
  * the non-negotiable governance guardrails (Cochrane/HHS/unknown AI-blocked).
"""
from __future__ import annotations

import itertools

import pytest

from services.downstream_eligibility import (
    VALID_TOOLS,
    article_sources,
    can_use_article_for_tool,
)
from services.source_permission_resolver import (
    resolve,
    resolve_for_sources,
)

# ── Independent re-derivation of the intended policy (second source of truth) ──
AI_ALLOWED = {
    "pubmed", "pubmed_guidelines", "europepmc", "openalex",
    "icite", "crossref", "semantic_scholar",
}
BLOCKED = {"cochrane", "myhealthfinder", "hhs"}
ALIASES = {
    "ncbi": "pubmed", "pubmed_central": "pubmed", "pmc": "pubmed", "medline": "pubmed",
    "pubmed_guideline_lane": "pubmed_guidelines", "guidelines": "pubmed_guidelines",
    "cochrane_review": "cochrane", "cochrane_library": "cochrane",
    "health.gov": "hhs", "healthfinder": "myhealthfinder", "odphp": "hhs",
    "europe_pmc": "europepmc", "openalex_works": "openalex", "s2": "semantic_scholar",
}
PLANE_FIELDS = [
    "retrieval_allowed", "link_out_allowed", "display_snippet_allowed",
    "ai_semantic_screening_allowed", "ai_summary_ingestion_allowed",
    "ai_ranker_optimization_allowed", "allowed_text_scope",
]
UNKNOWN_SOURCES = [f"unknown_src_{i}" for i in range(40)]


def _canon(source: str) -> str:
    s = (source or "").strip().lower().replace(" ", "_")
    return ALIASES.get(s, s)


def _expected_profile(source: str) -> dict:
    key = _canon(source)
    if key in AI_ALLOWED:
        return {
            "retrieval_allowed": True, "link_out_allowed": True,
            "display_snippet_allowed": "yes",
            "ai_semantic_screening_allowed": True,
            "ai_summary_ingestion_allowed": True,
            "ai_ranker_optimization_allowed": True,
            "allowed_text_scope": "title_plus_abstract",
        }
    # blocked + unknown → identical fail-closed plane profile
    return {
        "retrieval_allowed": True, "link_out_allowed": True,
        "display_snippet_allowed": "verify",
        "ai_semantic_screening_allowed": False,
        "ai_summary_ingestion_allowed": False,
        "ai_ranker_optimization_allowed": False,
        "allowed_text_scope": "none",
    }


ALL_SOURCES = (
    sorted(AI_ALLOWED | BLOCKED) + sorted(ALIASES.keys()) + UNKNOWN_SOURCES
)

# ════════════════════════════════════════════════════════════════════
# GROUP A — B1 resolver: every source × every plane field
# ════════════════════════════════════════════════════════════════════
_A_CASES = [(s, f) for s in ALL_SOURCES for f in PLANE_FIELDS]


@pytest.mark.parametrize("source,field", _A_CASES)
def test_b1_resolver_plane(source, field):
    perm = resolve(source)
    assert getattr(perm, field) == _expected_profile(source)[field], (
        f"{source}.{field}"
    )


@pytest.mark.parametrize("source", ALL_SOURCES)
def test_b1_resolver_normalizes_source_name(source):
    # resolve() always reports the canonical (normalized) source id.
    assert resolve(source).source == _canon(source)


# ════════════════════════════════════════════════════════════════════
# GROUP B — B16 eligibility: every source × every tool
# ════════════════════════════════════════════════════════════════════
ELIG_SOURCES = sorted(AI_ALLOWED | BLOCKED) + sorted(ALIASES.keys()) + ["unknown_x", "unknown_y"]
AI_TOOLS = {
    "summary", "litscholar", "litstream", "forum_draft",
    "ai_summary_ingestion", "ranker_optimization", "ai_semantic_screening",
}
_B_CASES = [(s, t) for s in ELIG_SOURCES for t in sorted(VALID_TOOLS)]


@pytest.mark.parametrize("source,tool", _B_CASES)
def test_b16_eligibility(source, tool):
    verdict = can_use_article_for_tool({"sources_present": [source]}, tool)
    key = _canon(source)
    if tool in AI_TOOLS:
        expected = key in AI_ALLOWED
    elif tool == "link_out":
        expected = True  # link-out always allowed (incl. blocked + unknown)
    elif tool == "display_snippet":
        expected = key in AI_ALLOWED  # snippet "yes" only for AI-allowed
    else:
        expected = False
    assert verdict["allowed"] is expected, f"{source}/{tool} -> {verdict}"


@pytest.mark.parametrize("source", ELIG_SOURCES)
@pytest.mark.parametrize("tool", ["bogus_tool", "", "DROP TABLE", "summary2"])
def test_b16_unknown_tool_always_blocked(source, tool):
    v = can_use_article_for_tool({"sources_present": [source]}, tool)
    assert v["allowed"] is False
    assert v["reason"] == "unknown_tool"


@pytest.mark.parametrize("tool", sorted(VALID_TOOLS))
def test_b16_no_source_fails_closed_for_ai(tool):
    v = can_use_article_for_tool({"sources_present": []}, tool)
    if tool == "link_out":
        # No-source fail-closed blocks AI planes but link-out stays allowed
        # (link-out is never an AI operation).
        assert v["allowed"] is True
    elif tool in AI_TOOLS or tool == "display_snippet":
        assert v["allowed"] is False


# ════════════════════════════════════════════════════════════════════
# GROUP C — B1 most-restrictive merge across source combinations
# ════════════════════════════════════════════════════════════════════
_MERGE_POOL = ["pubmed", "europepmc", "openalex", "cochrane", "hhs", "unknown_z", "crossref"]
_MERGE_COMBOS = (
    list(itertools.combinations(_MERGE_POOL, 2))
    + list(itertools.combinations(_MERGE_POOL, 3))
)


@pytest.mark.parametrize("combo", _MERGE_COMBOS)
def test_b1_merge_is_most_restrictive(combo):
    merged = resolve_for_sources(list(combo))
    all_ai_ok = all(_canon(s) in AI_ALLOWED for s in combo)
    # AI planes are the AND across sources; one restricted source poisons them.
    assert merged.ai_summary_ingestion_allowed is all_ai_ok
    assert merged.ai_ranker_optimization_allowed is all_ai_ok
    assert merged.ai_semantic_screening_allowed is all_ai_ok
    if not all_ai_ok:
        assert merged.allowed_text_scope == "none"


@pytest.mark.parametrize("combo", _MERGE_COMBOS)
def test_b1_merge_can_use_for_summary_matches_and(combo):
    art = {"sources_present": list(combo)}
    expected = all(_canon(s) in AI_ALLOWED for s in combo)
    assert can_use_article_for_tool(art, "summary")["allowed"] is expected


# ════════════════════════════════════════════════════════════════════
# GROUP I — non-negotiable governance guardrails
# ════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("src", ["cochrane", "cochrane_review", "cochrane_library",
                                 "hhs", "myhealthfinder", "health.gov", "odphp", "healthfinder"])
@pytest.mark.parametrize("tool", sorted(AI_TOOLS))
def test_governance_restricted_sources_never_ai(src, tool):
    """Cochrane + HHS/MyHealthfinder are NEVER usable for any AI tool (B2)."""
    assert can_use_article_for_tool({"sources_present": [src]}, tool)["allowed"] is False


@pytest.mark.parametrize("src", ["cochrane", "hhs", "myhealthfinder", "unknown_q"])
def test_governance_restricted_sources_still_link_out(src):
    """Restricted sources are retrievable + link-out-able (just not AI)."""
    perm = resolve(src)
    assert perm.retrieval_allowed is True
    assert perm.link_out_allowed is True
    assert perm.ai_summary_ingestion_allowed is False


@pytest.mark.parametrize("src", UNKNOWN_SOURCES[:10])
def test_governance_unknown_fails_closed(src):
    perm = resolve(src)
    assert perm.ai_summary_ingestion_allowed is False
    assert perm.ai_ranker_optimization_allowed is False
    assert perm.source_policy_reason == "unknown_source_fail_closed"


@pytest.mark.parametrize("src", sorted(AI_ALLOWED))
def test_governance_pubmed_family_ai_permitted_title_abstract(src):
    perm = resolve(src)
    assert perm.ai_summary_ingestion_allowed is True
    assert perm.allowed_text_scope == "title_plus_abstract"


# article_sources() extraction variants.
@pytest.mark.parametrize("article,expected", [
    ({"sources_present": ["pubmed"]}, ["pubmed"]),
    ({"sources_present": ["pubmed", "cochrane"]}, ["pubmed", "cochrane"]),
    ({"source_ids_json": {"pubmed": "1", "doi": "x"}}, ["pubmed", "doi"]),
    ({"source_ids_json": ["pubmed"]}, ["pubmed"]),
    ({}, []),
    ({"sources_present": []}, []),
    (None, []),
    ("not-a-dict", []),
])
def test_article_sources_extraction(article, expected):
    assert sorted(article_sources(article)) == sorted(expected)
