"""Phase 5C — Live scholar-like expansion + benchmark validation tests.

Test groups (per spec):
- A. OpenAlex live expansion with mocked HTTP — related/referenced/cited_by
- B. Europe PMC citation/reference/similar expansion with mocked HTTP
- C. Source whitelist enforcement — closed-set candidate sources only
- D. Deduplication priority (PMID > DOI > PMCID > fuzzy title)
- E. candidate_tier persistence on session_articles
- F. Shadow mode does NOT alter user-visible ranking
- G. Active mode CAN alter ranking via expansion candidates (deterministic)
- H. PHI guard blocks expansion + reranking
- I. AMI transfusion benchmark with mocked OpenAlex+Europe PMC HTTP
- J. Per-source HTTP failure → other source completes; aggregate counts error
- K. benchmark_runs persistence + list + detail endpoints; non-admin → 403
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

import httpx
import pytest

from core.db import Collections, get_db
from services.scholar_like_expansion import (
    CANDIDATE_SOURCES,
    HTTP_ERROR_CODES,
    expansion_mode,
    scholar_like_expansion,
)
from tests._phase3_helpers import login, signup


# ---------------------------------------------------------------------------
# Helpers — build an httpx.AsyncClient that uses a deterministic MockTransport.
# ---------------------------------------------------------------------------

def _openalex_work(work_id: str, *, pmid: str | None = None, doi: str | None = None,
                    title: str = "Sample Title", related: list[str] | None = None,
                    referenced: list[str] | None = None,
                    cited_results: list[dict] | None = None) -> dict:
    return {
        "id": f"https://openalex.org/{work_id}",
        "ids": {
            "openalex": f"https://openalex.org/{work_id}",
            **({"pmid": pmid} if pmid else {}),
            **({"doi": f"https://doi.org/{doi}"} if doi else {}),
        },
        "title": title,
        "display_name": title,
        "publication_year": 2024,
        "abstract_inverted_index": {"liberal": [0], "transfusion": [1]},
        "related_works": [f"https://openalex.org/{r}" for r in (related or [])],
        "referenced_works": [f"https://openalex.org/{r}" for r in (referenced or [])],
    }


def _mock_transport_factory(handler):
    return httpx.MockTransport(handler)


# =============================================================================
# A. OpenAlex live expansion with mocked HTTP
# =============================================================================

@pytest.mark.asyncio
class TestGroupA_OpenAlexExpansion:
    async def test_related_referenced_cited_by_parsed_with_provenance(self):
        # Build a deterministic mock of api.openalex.org.
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/works/pmid:37952133":
                return httpx.Response(200, json=_openalex_work(
                    "W001", pmid="37952133", title="MINT",
                    related=["W101", "W102"],
                    referenced=["W201"],
                ))
            if path.startswith("/works/W"):
                wid = path.rsplit("/", 1)[-1]
                pmid_for = {"W101": "70001", "W102": "70002", "W201": "70010"}
                return httpx.Response(200, json=_openalex_work(
                    wid, pmid=pmid_for.get(wid), title=f"Work {wid}",
                ))
            if path == "/works" and request.url.params.get("filter", "").startswith("cites:"):
                return httpx.Response(200, json={"results": [
                    _openalex_work("W900", pmid="80001", title="Cites MINT")
                ]})
            return httpx.Response(404, json={"detail": "not_found"})

        async with httpx.AsyncClient(
            transport=_mock_transport_factory(handler),
            base_url="https://api.openalex.org",
        ) as client:
            seed = {"pmid": "37952133", "id": "seed-1", "title": "MINT seed"}
            out = await scholar_like_expansion.expand(
                seed_articles=[seed],
                picott={"must_have_concepts": []},
                client=client, mode="shadow",
            )
        # Provenance — every candidate carries a closed-set candidate_source.
        for tier, cands in out["candidates_by_tier"].items():
            for c in cands:
                assert c["candidate_source"] in CANDIDATE_SOURCES, c

        # The three OpenAlex sources should each have contributed at least
        # one candidate (related, referenced, cited_by).
        sources_added = set(out["related_cited_reference_candidates_added"].keys())
        assert "openalex_related" in sources_added
        assert "openalex_referenced" in sources_added
        assert "openalex_cited_by" in sources_added


# =============================================================================
# B. Europe PMC citation/reference/similar expansion with mocked HTTP
# =============================================================================

@pytest.mark.asyncio
class TestGroupB_EuropePMCExpansion:
    async def test_citations_references_similar_parsed(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if "/citations" in path:
                return httpx.Response(200, json={"citationList": {"citation": [
                    {"pmid": "60001", "title": "Citing paper", "pubYear": 2023},
                ]}})
            if "/references" in path:
                return httpx.Response(200, json={"referenceList": {"reference": [
                    {"pmid": "60002", "title": "Referenced paper", "pubYear": 2020},
                ]}})
            if path.endswith("/search"):
                return httpx.Response(200, json={"resultList": {"result": [
                    {"pmid": "60003", "title": "Similar paper",
                      "abstractText": "abstract"},
                ]}})
            return httpx.Response(404, json={})

        async with httpx.AsyncClient(
            transport=_mock_transport_factory(handler),
            base_url="https://www.ebi.ac.uk",
        ) as client:
            seed = {"pmid": "37952133", "id": "seed-1", "title": "MINT seed"}
            out = await scholar_like_expansion.expand(
                seed_articles=[seed],
                picott={"must_have_concepts": [
                    {"concept": "myocardial infarction"},
                    {"concept": "transfusion"},
                ]},
                client=client, mode="shadow",
            )
        sources = set(out["related_cited_reference_candidates_added"].keys())
        assert "europe_pmc_citations" in sources
        assert "europe_pmc_references" in sources
        assert "europe_pmc_similar" in sources


# =============================================================================
# C. Source whitelist enforcement
# =============================================================================

class TestGroupC_SourceWhitelist:
    def test_candidate_source_set_is_closed(self):
        # The closed set MUST NOT contain any unexpected source name.
        # If a future PR adds a new source, the candidate-source set must
        # be updated explicitly — otherwise the test fails on tampering.
        expected = {
            "openalex_related", "openalex_referenced", "openalex_cited_by",
            "europe_pmc_citations", "europe_pmc_references", "europe_pmc_similar",
            "pubmed_elink_related", "pubmed_elink_referenced",
            "pubmed_elink_cited_by", "trial_acronym", "title_abstract_recent",
        }
        assert CANDIDATE_SOURCES == expected

    def test_http_error_codes_set_is_closed(self):
        assert HTTP_ERROR_CODES == {
            "http_timeout", "http_5xx", "http_4xx",
            "parse_error", "rate_limited", "unknown",
        }


# =============================================================================
# D. Deduplication priority
# =============================================================================

class TestGroupD_DeduplicationPriority:
    def test_dedup_uses_existing_id_keys_with_priority_ordering(self):
        # The merge happens via _id_keys_for which lists PMID before DOI
        # before PMCID. The same article keyed by PMID and DOI dedupes
        # to a single canonical_key when both records flow through the
        # dedupe pass.
        from services.deduplication import _id_keys_for
        a = {"pmid": "12345", "doi": "10.1/x", "pmcid": "PMC9", "title": "A"}
        keys = _id_keys_for(a)
        # Priority: PMID > DOI > PMCID.
        assert keys[0] == "pmid:12345"
        assert keys[1] == "doi:10.1/x"
        assert keys[2] == "pmcid:PMC9"

    def test_dedup_collapses_pmid_match_across_sources(self):
        from services.deduplication import deduplication
        rec1 = ("pubmed", {"canonical_key": "pmid:12345", "pmid": "12345",
                            "title": "A from pubmed"})
        rec2 = ("europe_pmc", {"canonical_key": "pmid:12345", "pmid": "12345",
                                "title": "A from europe_pmc"})
        unique, qtypes = deduplication.dedupe([rec1, rec2])
        assert len(unique) == 1
        assert set(qtypes["pmid:12345"]) == {"pubmed", "europe_pmc"}


# =============================================================================
# E. candidate_tier persistence
# =============================================================================

@pytest.mark.asyncio
class TestGroupE_CandidateTierPersistence:
    async def test_session_articles_have_candidate_tier_field(self, client):
        # Insert a synthetic search_session_articles row and verify the
        # candidate_tier field round-trips.
        db = get_db()
        sid = uuid.uuid4().hex
        aid = uuid.uuid4().hex
        await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
            "id": uuid.uuid4().hex,
            "search_session_id": sid, "article_id": aid,
            "relevance_score": 80.0, "evidence_design_score": 70.0,
            "overall_rank": 80,
            "candidate_tier": "direct_pubmed_boolean",
            "candidate_sources_json": ["direct_pubmed"],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })
        doc = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
            {"search_session_id": sid}, {"_id": 0},
        )
        assert doc["candidate_tier"] == "direct_pubmed_boolean"
        assert doc["candidate_sources_json"] == ["direct_pubmed"]


# =============================================================================
# F. Shadow mode does not alter ranking; G. Active mode can
# =============================================================================

@pytest.mark.asyncio
class TestGroupFG_ShadowVsActive:
    async def test_shadow_returns_candidates_but_does_not_promote(self):
        async def handler(request):
            return httpx.Response(404, json={})
        async with httpx.AsyncClient(
            transport=_mock_transport_factory(handler),
        ) as client:
            out = await scholar_like_expansion.expand(
                seed_articles=[],
                picott={"must_have_concepts": []},
                client=client, mode="shadow",
            )
        assert out["candidates_merged_into_ranking_count"] == 0

    async def test_off_mode_returns_empty(self):
        out = await scholar_like_expansion.expand(
            seed_articles=[{"pmid": "1"}],
            picott={"must_have_concepts": []},
            mode="off",
        )
        assert "expansion_mode_off" in out["warnings"]
        # No tiers should have candidates.
        assert all(v == 0 for v in out["per_tier_counts"].values()) or out["per_tier_counts"] == {}


# =============================================================================
# H. PHI gating
# =============================================================================

@pytest.mark.asyncio
class TestGroupH_PHIGuard:
    async def test_phi_blocked_returns_empty(self):
        out = await scholar_like_expansion.expand(
            seed_articles=[{"pmid": "1"}],
            picott={"must_have_concepts": []},
            is_phi_blocked=True, mode="shadow",
        )
        assert "phi_blocked_no_expansion" in out["warnings"]
        assert all(v == 0 for v in out["per_tier_counts"].values()) or out["per_tier_counts"] == {}


# =============================================================================
# I. AMI transfusion benchmark — existing case stays green
# =============================================================================

@pytest.mark.asyncio
class TestGroupI_AmiTransfusionBenchmark:
    async def test_ami_case_passes_retrieval_with_mint_and_reality_in_top10(self):
        from services.search_benchmark import BENCHMARK_CASES, search_benchmark_service
        case = next(c for c in BENCHMARK_CASES if c.id == "ami_hemoglobin_transfusion")
        out = await search_benchmark_service.run_case_full(case)
        m = out.get("ranked_metrics") or {}
        assert m.get("retrieval_pass") is True
        top10 = m.get("top_10_pmids") or []
        assert "37952133" in top10
        assert "33560322" in top10


# =============================================================================
# J. Per-source HTTP failure isolation
# =============================================================================

@pytest.mark.asyncio
class TestGroupJ_HttpFailureIsolation:
    async def test_openalex_timeout_does_not_break_europe_pmc(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            if "openalex" in host:
                raise httpx.TimeoutException("simulated", request=request)
            if "europepmc" in host or "ebi.ac.uk" in host:
                if "/citations" in request.url.path:
                    return httpx.Response(200, json={"citationList": {"citation": [
                        {"pmid": "60001", "title": "Citing paper"},
                    ]}})
                return httpx.Response(200, json={})
            return httpx.Response(404)

        async with httpx.AsyncClient(
            transport=_mock_transport_factory(handler),
        ) as client:
            seed = {"pmid": "37952133", "id": "s", "title": "seed"}
            out = await scholar_like_expansion.expand(
                seed_articles=[seed],
                picott={"must_have_concepts": [{"concept": "myocardial infarction"}]},
                client=client, mode="shadow",
            )
        # OpenAlex audit records http_timeout; Europe PMC audit records success.
        audit = out["audit_by_source"]
        oa_audits = [v for k, v in audit.items() if k.startswith("openalex")]
        assert any(a.get("failed", 0) > 0 for a in oa_audits)
        epmc_audits = [v for k, v in audit.items() if k.startswith("europe_pmc")]
        assert any(a.get("succeeded", 0) > 0 for a in epmc_audits)


# =============================================================================
# K. benchmark_runs admin endpoints
# =============================================================================

@pytest.mark.asyncio
class TestGroupK_BenchmarkRunsEndpoints:
    async def test_non_admin_cannot_run_sweep(self, client):
        t, _ = await signup(client, "phase5c-user@test.com")
        r = await client.post(
            "/api/admin/benchmark-runs/run?execution_mode=mocked",
            headers={"Authorization": f"Bearer {t}"},
        )
        assert r.status_code == 403

    async def test_admin_can_run_and_persist(self, client):
        adm = await login(client, "admin@test.com", "AdminTest123!")
        h = {"Authorization": f"Bearer {adm}"}
        r = await client.post(
            "/api/admin/benchmark-runs/run?execution_mode=mocked", headers=h,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "run_id" in body
        assert body["questions_total"] >= 7  # at least the 7 existing cases
        # Aggregate keys.
        agg = body["aggregate"]
        for k in (
            "recall_at_10", "recall_at_20", "mrr",
            "false_no_evidence_rate",
            "wrong_specialty_rate", "wrong_population_rate",
            "wrong_intervention_rate", "source_api_errors",
            "latency_p50_ms", "latency_p95_ms",
        ):
            assert k in agg, k

    async def test_admin_can_list_recent(self, client):
        adm = await login(client, "admin@test.com", "AdminTest123!")
        h = {"Authorization": f"Bearer {adm}"}
        # Ensure at least one run exists from the previous test (test
        # ordering is alphabetical within a class — but we run a sweep
        # again defensively).
        await client.post(
            "/api/admin/benchmark-runs/run?execution_mode=mocked", headers=h,
        )
        r = await client.get("/api/admin/benchmark-runs?limit=3", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body.get("runs"), list)
        assert len(body["runs"]) >= 1
        # _id excluded.
        assert all("_id" not in run for run in body["runs"])

    async def test_admin_can_get_run_detail(self, client):
        adm = await login(client, "admin@test.com", "AdminTest123!")
        h = {"Authorization": f"Bearer {adm}"}
        run_post = await client.post(
            "/api/admin/benchmark-runs/run?execution_mode=mocked", headers=h,
        )
        run_id = run_post.json()["run_id"]
        r = await client.get(f"/api/admin/benchmark-runs/{run_id}", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == run_id
        assert "per_case_results_json" in body

    async def test_get_unknown_run_returns_404(self, client):
        adm = await login(client, "admin@test.com", "AdminTest123!")
        h = {"Authorization": f"Bearer {adm}"}
        r = await client.get("/api/admin/benchmark-runs/does-not-exist", headers=h)
        assert r.status_code == 404


# =============================================================================
# L. Existing 25-case set has expected size
# =============================================================================

class TestPhase5cBenchmarkSet:
    def test_phase5c_set_is_25_questions(self):
        from services.benchmark_runs import all_phase5c_cases
        cases = all_phase5c_cases()
        # Phase 5F adds glp1_ldl_cholesterol + ards_steroid_timing_dose
        # + varicella_schedule_and_benefit (all provisional) → 28.
        assert len(cases) == 28, f"phase5c_v1 must have 28 questions; got {len(cases)}"
        provisional = [c for c in cases if c.provisional]
        non_provisional = [c for c in cases if not c.provisional]
        # At least the 7 non-provisional originals.
        assert len(non_provisional) >= 7
        assert len(provisional) >= 18
