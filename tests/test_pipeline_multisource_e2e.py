"""Phase 2B — multi-source pipeline integration tests with mocked retrieval."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


async def _noop_enrich(*args, **kwargs):
    """No-op OpenAlex enrichment for isolation. Returns the stats shape
    that ``services.connectors.openalex.openalex_connector.enrich_many``
    returns so the pipeline can read it without crashing."""
    return {
        "enrichments_attempted": 0,
        "enrichments_applied": 0,
        "status": "ok",
        "error_message": None,
    }


async def _noop_oa_fetch(*args, **kwargs):
    """No-op OA full-text fetcher. Skips real Europe PMC / PMC HTTP calls."""
    return {
        "fetched": 0, "cache_hits": 0, "failed": 0,
        "skipped_ineligible": 0, "skipped_license": 0,
        "skipped_cap_exceeded": 0, "disabled": 0, "status": "skipped",
    }


# Common mocks to keep the e2e pipeline deterministic (no external HTTP).
def _pipeline_isolation_patches(retrieve_fn):
    """Return the list of patch() context managers needed to isolate the
    pipeline e2e tests from real network: literature retrieval, OpenAlex
    enrichment, and the Phase 4A OA full-text fetcher."""
    return [
        patch("services.pipeline.literature_retrieval.retrieve", new=retrieve_fn),
        patch("services.pipeline.openalex_connector.enrich_many", new=_noop_enrich),
        patch("services.oa_fetcher.oa_fetcher.fetch_for_articles", new=_noop_oa_fetch),
    ]


_EPMC_RAW = {
    "id": "34449189",
    "source": "MED",
    "title": "Empagliflozin in heart failure with preserved ejection fraction",
    "abstractText": "Europe PMC abstract: RCT of empagliflozin vs placebo in HFpEF.",
    "journalInfo": {"journal": {"title": "NEJM"}, "yearOfPublication": "2021"},
    "pubYear": "2021",
    "doi": "10.1056/NEJMoa2107038",
    "pmid": "34449189",
    "pmcid": "PMC8500551",
    "pubTypeList": {"pubType": ["Journal Article", "Randomized Controlled Trial"]},
    "authorList": {"author": [{"fullName": "Anker SD", "lastName": "Anker"}]},
    "fullTextUrlList": {"fullTextUrl": []},
    "inEPMC": "Y",
}

# Same article from PubMed (normalized shape, includes _retrieved_by).
_PUBMED_NORM = {
    "id": "pubmed-uuid",
    "canonical_key": "pmid:34449189",
    "title": "Empagliflozin in heart failure with preserved ejection fraction",
    "abstract": (
        "BACKGROUND: This RCT evaluated empagliflozin versus placebo in patients "
        "with heart failure with preserved ejection fraction (HFpEF). "
        "METHODS: Randomized double-blind comparison of empagliflozin and placebo. "
        "RESULTS: Empagliflozin reduced hospitalization for heart failure compared "
        "with placebo in HFpEF."
    ),
    "journal": "NEJM",
    "publication_year": 2021,
    "publication_date": "2021 Aug 27",
    "authors_json": [{"fullName": "Anker SD", "lastName": "Anker"}],
    "doi": "10.1056/NEJMoa2107038",
    "pmid": "34449189",
    "pmcid": "PMC8500551",
    "source_ids_json": {"pubmed": "34449189", "doi": "10.1056/NEJMoa2107038"},
    "publication_types_json": ["journal article", "randomized controlled trial"],
    "article_type_classification": "rct",
    "is_guideline": False,
    "is_preprint": False,
    "is_retracted": False,
    "is_retraction_notice": False,
    "has_expression_of_concern": False,
    "full_text_status": "full_text_available",
    "source_urls_json": ["https://pubmed.ncbi.nlm.nih.gov/34449189/"],
    "_retrieved_by": "pubmed",
}


async def _signup(client, email):
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!", "name": "T"},
    )
    return r.json()["access_token"]


async def _wait_ready(client, sid, headers, max_iters=200):
    for _ in range(max_iters):
        await asyncio.sleep(0.1)
        r = await client.get(f"/api/search-sessions/{sid}/status", headers=headers)
        body = r.json()
        if body.get("articles_ready") or body["status"] == "error":
            return body
    return body


@pytest.mark.asyncio
async def test_multi_source_pipeline_dedupes_and_marks_sources_present(client):
    """With both europe_pmc and pubmed enabled, the same article must be
    de-duped and the search_session_article must carry both sources_present."""
    token = await _signup(client, "ms1@test.com")
    headers = {"Authorization": f"Bearer {token}"}

    # Enable pubmed in source_configs for the test DB.
    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": True}}
    )

    async def fake_retrieve(source, queries, **kwargs):
        if source == "europe_pmc":
            return [("high_recall", _EPMC_RAW)]
        if source == "pubmed":
            return [("high_precision", dict(_PUBMED_NORM))]
        return []

    with patch("services.pipeline.literature_retrieval.retrieve", new=fake_retrieve), \
         patch("services.pipeline.openalex_connector.enrich_many", new=_noop_enrich), \
         patch("services.oa_fetcher.oa_fetcher.fetch_for_articles", new=_noop_oa_fetch):
        r2 = await client.post(
            "/api/search-sessions/",
            json={"question": "Is empagliflozin effective in HFpEF compared to placebo?"},
            headers=headers,
        )
        assert r2.status_code == 200
        sid = r2.json()["id"]
        status = await _wait_ready(client, sid, headers)
        assert status["status"] == "awaiting_selection"

    full = (await client.get(f"/api/search-sessions/{sid}", headers=headers)).json()
    # Should be exactly one article (dedup successful).
    assert len(full["articles"]) == 1
    article_row = full["articles"][0]
    assert set(article_row["sources_present"]) == {"europe_pmc", "pubmed"}
    # Multi-source boost text appears in rank explanation.
    assert "Cross-source confirmation" in (article_row["rank_explanation"] or "")
    # Evaluator must report source coverage with both sources.
    ev = full["evaluator_json"]
    assert "source_coverage" in ev
    assert set(ev["source_coverage"]["sources_searched"]) == {"europe_pmc", "pubmed"}
    assert ev["source_coverage"]["cross_source_confirmed_articles"] == 1

    # Transparency endpoint must list both sources searched + queries from both.
    trans = (await client.get(f"/api/search-sessions/{sid}/transparency", headers=headers)).json()
    assert set(trans["sources_searched"]) == {"europe_pmc", "pubmed"}
    qsources = {q["source"] for q in trans["queries"]}
    assert "europe_pmc" in qsources and "pubmed" in qsources
    # Phase 5A — legacy planner still emits 4 per source (8 total). The
    # concept-block builder may add more on top; assert the legacy four
    # families are present per source.
    legacy_families = {"high_recall", "high_precision", "evidence_filtered", "guideline_focused"}
    legacy_qs = [q for q in trans["queries"] if (q.get("query_type") in legacy_families)]
    assert len(legacy_qs) == 8


@pytest.mark.asyncio
async def test_pubmed_only_source_handles_pubmed_disabled(client):
    """When pubmed is disabled (default), only Europe PMC should be queried."""
    token = await _signup(client, "ms2@test.com")
    headers = {"Authorization": f"Bearer {token}"}

    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": False}}
    )

    called_sources: list[str] = []

    async def fake_retrieve(source, queries, **kwargs):
        called_sources.append(source)
        if source == "europe_pmc":
            return [("high_recall", _EPMC_RAW)]
        return []

    with patch("services.pipeline.literature_retrieval.retrieve", new=fake_retrieve):
        r2 = await client.post(
            "/api/search-sessions/",
            json={"question": "Is empagliflozin effective in HFpEF?"},
            headers=headers,
        )
        sid = r2.json()["id"]
        await _wait_ready(client, sid, headers)

    # Phase 6B.1 — when the compiler is the primary planner and the
    # mocked retrieval returns zero ``cb_high_precision`` hits, the
    # legacy planner fires as a per-source fallback. That doubles the
    # retrieve() calls for the enabled source but pubmed must still
    # be untouched.
    assert "pubmed" not in called_sources
    assert set(called_sources) == {"europe_pmc"}


@pytest.mark.asyncio
async def test_pubmed_failure_does_not_block_europe_pmc(client):
    """If PubMed fails, Europe PMC results must still survive."""
    token = await _signup(client, "ms3@test.com")
    headers = {"Authorization": f"Bearer {token}"}

    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": True}}
    )

    async def fake_retrieve(source, queries, **kwargs):
        if source == "pubmed":
            raise RuntimeError("simulated pubmed outage")
        return [("high_recall", _EPMC_RAW)]

    with patch("services.pipeline.literature_retrieval.retrieve", new=fake_retrieve), \
         patch("services.pipeline.openalex_connector.enrich_many", new=_noop_enrich), \
         patch("services.oa_fetcher.oa_fetcher.fetch_for_articles", new=_noop_oa_fetch):
        r2 = await client.post(
            "/api/search-sessions/",
            json={"question": "Is empagliflozin effective in HFpEF compared to placebo for mortality?"},
            headers=headers,
        )
        sid = r2.json()["id"]
        status = await _wait_ready(client, sid, headers)

    # Pipeline must NOT be in error state — Europe PMC succeeded.
    assert status["status"] == "awaiting_selection"
    full = (await client.get(f"/api/search-sessions/{sid}", headers=headers)).json()
    # Either included or excluded — but it must be there. The point is the
    # pipeline didn't crash on PubMed failure.
    all_titles = (
        [a["article"]["title"] for a in full["articles"]]
        + [a["article"]["title"] for a in full["excluded_articles"]]
    )
    assert any("Empagliflozin" in t for t in all_titles)


@pytest.mark.asyncio
async def test_phi_blocked_never_hits_any_retrieval_source(client):
    """Regression: PHI must short-circuit ALL retrieval sources (not just Europe PMC)."""
    token = await _signup(client, "ms4@test.com")
    headers = {"Authorization": f"Bearer {token}"}

    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": True}}
    )
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "openalex"}, {"$set": {"enabled": True}}
    )

    with patch("services.literature_retrieval.literature_retrieval.retrieve", new=AsyncMock()) as mock_retrieve, \
         patch("services.connectors.openalex.openalex_connector.enrich_many", new=AsyncMock()) as mock_oa, \
         patch("services.connectors.pubmed.pubmed_connector.search", new=AsyncMock()) as mock_pm:
        r = await client.post(
            "/api/search-sessions/",
            json={"question": "Mr. John Doe DOB 03/14/1979, MRN 12345678 — best HTN treatment?"},
            headers=headers,
        )
        assert r.status_code == 200
        assert r.json()["status"] == "blocked_phi"
        mock_retrieve.assert_not_called()
        mock_oa.assert_not_called()
        mock_pm.assert_not_called()
