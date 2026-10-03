"""B1 source-permission resolver + B16 downstream-eligibility unit tests (pure)."""
from __future__ import annotations

from services.downstream_eligibility import can_use_article_for_tool
from services.source_permission_resolver import (
    resolve,
    resolve_for_sources,
)


# ── B1 resolver ─────────────────────────────────────────────────────
def test_pubmed_ai_permitted_title_abstract():
    p = resolve("pubmed")
    assert p.ai_summary_ingestion_allowed
    assert p.ai_ranker_optimization_allowed
    assert p.allowed_text_scope == "title_plus_abstract"


def test_cochrane_ai_blocked_but_linkout_ok():
    p = resolve("cochrane")
    assert not p.ai_summary_ingestion_allowed
    assert not p.ai_ranker_optimization_allowed
    assert p.link_out_allowed  # link-out still allowed


def test_hhs_ai_blocked():
    assert not resolve("myhealthfinder").ai_summary_ingestion_allowed
    assert not resolve("hhs").ai_summary_ingestion_allowed


def test_unknown_source_fail_closed():
    p = resolve("acme_random_source")
    assert not p.ai_summary_ingestion_allowed
    assert p.source_policy_reason == "unknown_source_fail_closed"


def test_alias_normalization():
    assert resolve("NCBI").ai_summary_ingestion_allowed       # ncbi -> pubmed
    assert resolve("europe_pmc").ai_summary_ingestion_allowed  # alias -> europepmc


def test_most_restrictive_merge():
    merged = resolve_for_sources(["pubmed", "cochrane"])
    assert not merged.ai_ranker_optimization_allowed   # cochrane blocks it
    assert merged.allowed_text_scope == "none"          # collapses to lowest


def test_no_sources_fail_closed():
    assert not resolve_for_sources([]).ai_summary_ingestion_allowed


# ── B16 eligibility ─────────────────────────────────────────────────
def test_eligibility_pubmed_summary_allowed():
    assert can_use_article_for_tool({"sources_present": ["pubmed"]}, "summary")["allowed"]


def test_eligibility_cochrane_summary_blocked_linkout_ok():
    assert not can_use_article_for_tool({"sources_present": ["cochrane"]}, "summary")["allowed"]
    assert can_use_article_for_tool({"sources_present": ["cochrane"]}, "link_out")["allowed"]


def test_eligibility_ranker_optimization():
    assert can_use_article_for_tool({"sources_present": ["pubmed"]}, "ranker_optimization")["allowed"]
    assert not can_use_article_for_tool({"sources_present": ["hhs"]}, "ranker_optimization")["allowed"]


def test_eligibility_unknown_tool_and_no_source():
    assert not can_use_article_for_tool({"sources_present": ["pubmed"]}, "bogus_tool")["allowed"]
    assert not can_use_article_for_tool({"sources_present": []}, "summary")["allowed"]


# ── Cochrane whichever index it came through; PMID = PubMed record ──
def test_pubmed_indexed_cochrane_review_blocked_by_journal():
    art = {"sources_present": ["pubmed"], "journal": "The Cochrane Database of Systematic Reviews", "pmid": "35150000"}
    v = can_use_article_for_tool(art, "summary")
    assert not v["allowed"]
    assert v["reason"] == "cochrane_ai_ingestion_restricted"
    assert can_use_article_for_tool(art, "link_out")["allowed"]


def test_pubmed_indexed_cochrane_review_blocked_by_doi_prefix():
    art = {"source_ids_json": {"europepmc": "x"}, "journal": "", "doi": "10.1002/14651858.CD002243.pub4"}
    assert not can_use_article_for_tool(art, "summary")["allowed"]
    assert not can_use_article_for_tool(art, "ranker_optimization")["allowed"]


def test_record_with_pmid_but_no_recorded_source_is_pubmed():
    assert can_use_article_for_tool({"pmid": "29347874", "journal": "N Engl J Med"}, "summary")["allowed"]


def test_record_without_source_or_pmid_fails_closed():
    v = can_use_article_for_tool({"source_ids_json": {}, "doi": "10.1/unknown"}, "summary")
    assert not v["allowed"]
    assert v["reason"] == "no_source_info_fail_closed"
