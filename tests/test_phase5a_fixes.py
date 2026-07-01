"""Phase 5A — tester-flagged spec deviation fixes.

Fix 1: hard-ceiling cap enforcement on BOTH relevance_score and overall_rank_score
Fix 2: PubMed concept-block queries appear in /diagnostics
Fix 3: benchmark response includes retrieval-side fields
Fix 4: PHI-blocked diagnostics returns 403
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from core.db import Collections, get_db
from services.picott_ranker import picott_aware_ranker
from services.search_benchmark import (
    BENCHMARK_CASES, search_benchmark_service,
)
from tests._phase3_helpers import login, signup


def _milrinone_picott() -> dict:
    return {
        "type_of_question": "dosing_protocol",
        "pico": {},
        "must_have_concepts": [
            {"concept": "milrinone", "role": "intervention",
             "synonyms": ["IV milrinone", "IA milrinone"],
             "missing_relevance_cap": 30},
            {"concept": "subarachnoid hemorrhage", "role": "condition",
             "synonyms": ["SAH", "aSAH"],
             "missing_relevance_cap": 50},
        ],
        "ranking_concepts": [
            {"concept": "dose", "role": "context", "synonyms": []},
            {"concept": "infusion", "role": "context", "synonyms": []},
        ],
        "negative_or_penalty_concepts": [
            {"concept": "CT angiography", "penalty_reason": "imaging_only",
             "unless_concepts_present": ["milrinone"], "penalty_cap": 40},
            {"concept": "CTA", "penalty_reason": "imaging_only",
             "unless_concepts_present": ["milrinone"], "penalty_cap": 40},
        ],
    }


# ---------------------------------------------------------------------------
# Fix 1 — relevance_score and overall_rank_score clamped to cap.
# ---------------------------------------------------------------------------

class TestFix1HardCeilingOnRelevance:
    def test_no_milrinone_article_relevance_capped_at_30(self):
        picott = _milrinone_picott()
        ranked = [
            {
                "_article": {
                    "id": "a1", "canonical_key": "ck1",
                    "title": "Bone-subtraction CTA for aneurysmal SAH",
                    "abstract": "Diagnostic accuracy of CTA for aneurysm detection in subarachnoid hemorrhage.",
                    "article_type_classification": "cohort_study",
                },
                "overall_rank_score": 85,
                "relevance_score": 80.0,
                "evidence_design_score": 70.0,
                "query_types_matched": ["broad_recall"],
            },
            {
                "_article": {
                    "id": "a2", "canonical_key": "ck2",
                    "title": "Nimodipine vs placebo in SAH vasospasm",
                    "abstract": "Nimodipine in subarachnoid hemorrhage with delayed cerebral ischemia.",
                    "article_type_classification": "rct",
                },
                "overall_rank_score": 80,
                "relevance_score": 75.0,
                "evidence_design_score": 80.0,
                "query_types_matched": ["high_precision"],
            },
        ]
        out = picott_aware_ranker.postprocess(ranked, picott)
        for r in out:
            mh = [c.lower() for c in (r.get("matched_must_have") or [])]
            assert "milrinone" not in mh, "must_have milrinone unexpectedly matched"
            assert r["overall_rank_score"] <= 30, (
                f"overall_rank_score {r['overall_rank_score']} exceeds cap 30 "
                f"for non-milrinone article (cap_reasons={r.get('cap_reasons')})"
            )
            assert r["relevance_score"] <= 30, (
                f"relevance_score {r['relevance_score']} exceeds cap 30 "
                f"for non-milrinone article (cap_reasons={r.get('cap_reasons')})"
            )

    def test_imaging_decoy_article_relevance_capped_at_40_or_lower(self):
        picott = _milrinone_picott()
        # This article DOES have milrinone (waiver applies), but if it
        # doesn't waive — e.g., a bare CTA paper — the cap of 40 fires.
        # We construct an article missing milrinone so the imaging penalty
        # fires, then assert cap ≤ 40.
        ranked = [
            {
                "_article": {
                    "id": "a3", "canonical_key": "ck3",
                    "title": "CT angiography for cerebral aneurysm detection",
                    "abstract": "CTA performance for detecting cerebral aneurysms after SAH.",
                    "article_type_classification": "cohort_study",
                },
                "overall_rank_score": 90,
                "relevance_score": 85.0,
                "query_types_matched": ["broad_recall"],
            },
        ]
        out = picott_aware_ranker.postprocess(ranked, picott)
        r = out[0]
        assert "imaging_only_for_pharmacology" in r["cap_reasons"]
        assert r["relevance_score"] <= 40, (
            f"relevance_score {r['relevance_score']} exceeds imaging cap 40 "
            f"(cap_reasons={r['cap_reasons']})"
        )
        assert r["overall_rank_score"] <= 40

    def test_full_match_article_keeps_high_relevance(self):
        picott = _milrinone_picott()
        ranked = [
            {
                "_article": {
                    "id": "a4", "canonical_key": "ck4",
                    "title": "IA milrinone for cerebral vasospasm in aSAH",
                    "abstract": "Continuous milrinone infusion at 0.5 mcg/kg/min in subarachnoid hemorrhage.",
                    "article_type_classification": "rct",
                },
                "overall_rank_score": 80,
                "relevance_score": 80.0,
                "query_types_matched": ["high_precision"],
            },
        ]
        out = picott_aware_ranker.postprocess(ranked, picott)
        # Fully matched — no cap. relevance_score is NOT clamped DOWN.
        # Phase 6F.1 — the new ``anchor_match_full`` boost (+35) may
        # push relevance_score above its initial value, capped at 100.
        # The pre-6F.1 invariant was ``relevance_score == 80``;
        # the Phase 6F.1 invariant is ``relevance_score >= 80``.
        assert out[0]["relevance_cap_applied"] is None
        assert out[0]["relevance_score"] >= 80.0, (
            f"relevance_score regressed below initial value: "
            f"{out[0]['relevance_score']}"
        )
        assert out[0]["relevance_score"] <= 100.0
        assert out[0]["overall_rank_score"] >= 80

    def test_via_benchmark_fixture_all_non_milrinone_articles_capped(self):
        """End-to-end: run the milrinone_sah benchmark with the fixture
        corpus and assert EVERY article in the ranked top-50 whose
        ``matched_must_have`` does NOT contain milrinone has relevance ≤ 30.

        Implemented as an async test below
        (``test_benchmark_no_milrinone_articles_capped``); this sync
        placeholder is kept as a docstring anchor only.
        """

    @pytest.mark.asyncio
    async def test_benchmark_no_milrinone_articles_capped(self):
        case = next(c for c in BENCHMARK_CASES if c.id == "milrinone_sah")
        out = await search_benchmark_service.run_case_full(case)
        # Re-run the ranker INSIDE the test to inspect every article.
        from services.search_benchmark_fixtures import MILRINONE_SAH_FIXTURE
        ranked = [dict(r, _article=dict(r["_article"])) for r in MILRINONE_SAH_FIXTURE]
        ranked = picott_aware_ranker.postprocess(ranked, out["picott"])
        # Every article missing milrinone must be capped ≤ 30.
        for r in ranked:
            mh = [c.lower() for c in (r.get("matched_must_have") or [])]
            if "milrinone" not in mh:
                assert r["overall_rank_score"] <= 30, (
                    f"non-milrinone article '{r['_article']['title']}' "
                    f"overall_rank_score {r['overall_rank_score']} > 30"
                )
                assert r["relevance_score"] <= 30, (
                    f"non-milrinone article relevance_score "
                    f"{r['relevance_score']} > 30"
                )

    @pytest.mark.asyncio
    async def test_benchmark_any_imaging_cap_reason_means_relevance_le_40(self):
        case = next(c for c in BENCHMARK_CASES if c.id == "milrinone_sah")
        out = await search_benchmark_service.run_case_full(case)
        from services.search_benchmark_fixtures import MILRINONE_SAH_FIXTURE
        ranked = [dict(r, _article=dict(r["_article"])) for r in MILRINONE_SAH_FIXTURE]
        ranked = picott_aware_ranker.postprocess(ranked, out["picott"])
        for r in ranked:
            if "imaging_only_for_pharmacology" in (r.get("cap_reasons") or []):
                assert r["overall_rank_score"] <= 40
                assert r["relevance_score"] <= 40


# ---------------------------------------------------------------------------
# Fix 2 — PubMed concept-block queries appear in /diagnostics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFix2PubMedQueriesInDiagnostics:
    async def test_pipeline_emits_pubmed_concept_block_queries(self, client):
        """When both PubMed and Europe PMC are enabled, the diagnostics
        endpoint must return concept-block queries (query_family in the
        Phase 5A canonical set) from BOTH sources."""
        db = get_db()
        # Ensure both sources are enabled.
        for src in ("europe_pmc", "pubmed"):
            await db[Collections.SOURCE_CONFIGS].update_one(
                {"source_name": src},
                {"$set": {"enabled": True}},
                upsert=True,
            )
        # Sign up + submit a milrinone-SAH question.
        t, _ = await signup(client, "phase5a-fix2-pubmed@test.com")
        h_user = {"Authorization": f"Bearer {t}"}
        r = await client.post(
            "/api/search-sessions/", headers=h_user,
            json={"question": ("What is the dose of milrinone for patients "
                                "with acute subarachnoid hemorrhage?")},
        )
        assert r.status_code == 200, r.text
        sid = r.json()["id"]

        # Wait briefly for the pipeline to record queries (it may still be
        # ranking, but search_queries are written upfront).
        import asyncio
        for _ in range(20):
            await asyncio.sleep(0.25)
            q_count = await db[Collections.SEARCH_QUERIES].count_documents(
                {"search_session_id": sid}
            )
            if q_count >= 4:
                break

        # Admin reads diagnostics.
        adm = await login(client, "admin@test.com", "AdminTest123!")
        h_adm = {"Authorization": f"Bearer {adm}"}
        d = await client.get(
            f"/api/admin/search-sessions/{sid}/diagnostics", headers=h_adm,
        )
        assert d.status_code == 200, d.text
        body = d.json()
        sources_seen = {q["source"] for q in body["queries_by_family"]}
        assert "europe_pmc" in sources_seen
        assert "pubmed" in sources_seen, (
            f"PubMed queries missing from diagnostics; sources seen: {sources_seen}"
        )
        # Phase 5A canonical families must appear for at least one source.
        cb_families = {q["query_family"] for q in body["queries_by_family"]
                        if str(q.get("query_type") or "").startswith("cb_")}
        assert "high_precision" in cb_families
        assert "broad_recall" in cb_families


# ---------------------------------------------------------------------------
# Fix 3 — benchmark response includes retrieval-side fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFix3BenchmarkRetrievalFields:
    async def test_milrinone_retrieval_pass_against_fixture(self):
        case = next(c for c in BENCHMARK_CASES if c.id == "milrinone_sah")
        out = await search_benchmark_service.run_case_full(case)
        rm = out["ranked_metrics"]
        assert rm is not None
        # All required Phase 5A fields populated.
        for field in (
            "top_10_pmids", "top_20_pmids",
            "gold_retrieved_count", "gold_missed_count",
            "missed_gold_pmids",
            "negatives_in_top_10_count",
            "retrieval_pass", "diagnostic_notes",
        ):
            assert field in rm, f"missing required field {field!r}"
        # Concrete expectations against the deterministic fixture.
        assert rm["retrieval_pass"] is True
        assert rm["gold_retrieved_count"] == 9
        assert rm["gold_missed_count"] == 0
        assert len(rm["top_10_pmids"]) == 10
        assert rm["negatives_in_top_10_count"] <= case.max_negative_in_top_10
        # Diagnostic notes are closed-set strings.
        for note in rm["diagnostic_notes"]:
            assert isinstance(note, str)
            assert len(note) < 500  # no PHI leak / no raw text dumps

    async def test_admin_retrieval_mode_endpoint_returns_metrics(self, client):
        tok = await login(client, "admin@test.com", "AdminTest123!")
        h = {"Authorization": f"Bearer {tok}"}
        r = await client.post(
            "/api/admin/search-benchmarks/run?mode=retrieval", headers=h,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Summary includes retrieval_passed when mode=retrieval.
        assert "retrieval_passed" in body["summary"]
        # All 6 cases present, milrinone case has retrieval_metrics.
        ms = next(c for c in body["cases"] if c["case_id"] == "milrinone_sah")
        rm = ms["ranked_metrics"]
        assert rm["retrieval_pass"] is True
        assert rm["gold_retrieved_count"] >= case_min_top10(ms)


def case_min_top10(case_dict):
    # The thresholds aren't in the response, so resolve via the in-memory cases.
    c = next(c for c in BENCHMARK_CASES if c.id == case_dict["case_id"])
    return c.minimum_gold_in_top_10


# ---------------------------------------------------------------------------
# Fix 4 — PHI-blocked diagnostics returns 403
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFix4PhiBlockedDiagnosticsReturns403:
    async def test_phi_blocked_session_diagnostics_returns_403(self, client):
        # Sign up clinician + submit a PHI-laden question to create a
        # blocked_phi session.
        t, _ = await signup(client, "phase5a-fix4-phi@test.com")
        h_user = {"Authorization": f"Bearer {t}"}
        r = await client.post(
            "/api/search-sessions/", headers=h_user,
            json={"question": ("Patient John MRN 12345678 DOB 03/14/1979 "
                                "best tx for HTN? jane@x.com")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "blocked_phi"
        sid = body["session_id"]

        adm = await login(client, "admin@test.com", "AdminTest123!")
        h_adm = {"Authorization": f"Bearer {adm}"}
        d = await client.get(
            f"/api/admin/search-sessions/{sid}/diagnostics", headers=h_adm,
        )
        assert d.status_code == 403
        body = d.json()
        # Only the safe detail string is exposed.
        assert "Diagnostics unavailable" in (body.get("detail") or "")
        assert "PHI" in (body.get("detail") or "")
