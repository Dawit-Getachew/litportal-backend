"""Phase 6C — benchmark sweep + scholar-like expansion contract tests.

Covers:
- Execution modes (direct_baseline / shadow / active_simulation).
- Recommendation engine (GO / CONTINUE_SHADOW / NO_GO branches).
- Admin dashboard authz + redacted JSON.
- PHI / cap-dominance invariants for the expansion path.
"""
from __future__ import annotations

import asyncio

import pytest


async def _signup(client, email: str, *, role: str | None = None) -> dict:
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!",
              "name": email.split("@")[0]},
    )
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    if role == "admin":
        from core.db import Collections, get_db
        await get_db()[Collections.USERS].update_one(
            {"email": email}, {"$set": {"role": "admin"}},
        )
        re = await client.post(
            "/api/auth/login",
            json={"email": email, "password": "Pass1234567!"},
        )
        headers = {"Authorization": f"Bearer {re.json()['access_token']}"}
    return headers


# ---------------------------------------------------------------------------
# A. Execution modes
# ---------------------------------------------------------------------------
def test_direct_baseline_three_mode_report_shape():
    """``_build_three_mode_report`` emits the Phase 6C schema even on a
    minimal fixture-only case_result list."""
    from services.benchmark_runs import _build_three_mode_report
    from services.search_benchmark import search_benchmark_service
    cases = search_benchmark_service.list_all()
    fake_case_results = [{
        "id": cases[0].id,
        "gold_pmids": list(cases[0].gold_pmids or []),
        "ranked_metrics": {
            "top_10_pmids": ["X1", "X2"],
            "top_20_pmids": ["X1", "X2"],
            "negatives_in_top_10_count": 0,
            "latency_p50_ms": 1234,
            "latency_p95_ms": 1500,
        },
        "provisional": False,
        "parser_pass": True,
    }]
    rpt = _build_three_mode_report(fake_case_results, cases)
    assert rpt["schema_version"] == "phase6c_v1"
    row = rpt["per_case_three_mode"][0]
    # Phase 6C schema required fields.
    for k in (
        "top_10_baseline_json", "top_10_active_simulation_json",
        "expansion_candidates_json", "promoted_candidates_json",
        "demoted_candidates_json", "source_api_errors_json",
        "latency_json",
    ):
        assert k in row, (k, row)
    assert row["source_api_errors_json"]["pubmed"]["rate_limited"] is False
    assert row["latency_json"]["direct_baseline"]["p50_ms"] == 1234


def test_active_simulation_promotes_gold_candidates_only():
    """Candidates that ARE in ``gold_pmids`` get promoted; non-gold
    candidates are recorded in ``expansion_candidates_json`` but NOT
    promoted (we cannot verify anchors from PMID alone)."""
    from services.benchmark_runs import _build_three_mode_report
    from services.search_benchmark import search_benchmark_service
    cases = search_benchmark_service.list_all()
    case = next(c for c in cases if c.id == "milrinone_sah")
    gold = list(case.gold_pmids or [])
    # Pick a gold pmid that's NOT in the baseline top-10 to give the
    # simulation something to promote.
    baseline_top10 = ["FAKE1", "FAKE2", "FAKE3"]
    gold_outside_top10 = gold[0]
    case_result = {
        "id": case.id,
        "gold_pmids": gold,
        "ranked_metrics": {
            "top_10_pmids": baseline_top10,
            "top_20_pmids": baseline_top10,
            "negatives_in_top_10_count": 0,
            "latency_p50_ms": 1000,
        },
        "scholar_like_expansion_json": {
            "candidates_by_tier": {
                "citing_top_seeds": [
                    {"pmid": gold_outside_top10, "title": "Promoted gold hit",
                     "candidate_source": "openalex",
                     "candidate_tier": "citing_top_seeds"},
                    {"pmid": "NOT_GOLD_999", "title": "Not gold; not promoted",
                     "candidate_source": "openalex",
                     "candidate_tier": "citing_top_seeds"},
                ],
            },
        },
    }
    rpt = _build_three_mode_report([case_result], cases)
    row = rpt["per_case_three_mode"][0]
    promoted_pmids = {p["pmid"] for p in row["promoted_candidates_json"]}
    assert gold_outside_top10 in promoted_pmids
    assert "NOT_GOLD_999" not in promoted_pmids
    # Active simulation gold_in_top_10 strictly improved.
    assert (row["by_mode"]["active_simulation"]["gold_in_top_10"]
            > row["by_mode"]["off"]["gold_in_top_10"])
    # Baseline unchanged.
    assert row["by_mode"]["off"]["gold_in_top_10"] == 0
    # Shadow IDENTICAL to off.
    assert (row["by_mode"]["shadow"]["gold_in_top_10"]
            == row["by_mode"]["off"]["gold_in_top_10"])
    # All non-gold candidates recorded.
    cand_pmids = {c["pmid"] for c in row["expansion_candidates_json"]}
    assert "NOT_GOLD_999" in cand_pmids


def test_shadow_expansion_never_alters_user_ranking():
    """Schema invariant: shadow numbers == off numbers for every metric."""
    from services.benchmark_runs import _build_three_mode_report
    from services.search_benchmark import search_benchmark_service
    cases = search_benchmark_service.list_all()
    case_result = {
        "id": cases[0].id,
        "gold_pmids": ["G1", "G2"],
        "ranked_metrics": {
            "top_10_pmids": ["G1", "X"],
            "top_20_pmids": ["G1", "X", "G2"],
            "negatives_in_top_10_count": 0,
        },
    }
    rpt = _build_three_mode_report([case_result], cases)
    row = rpt["per_case_three_mode"][0]
    assert row["by_mode"]["off"] == row["by_mode"]["shadow"]


# ---------------------------------------------------------------------------
# B. Recommendation engine
# ---------------------------------------------------------------------------
def test_recommendation_continue_shadow_when_provisional_dominates():
    """Closed-set: if fewer than 5 populated cases, force
    CONTINUE_SHADOW. Provisional + placeholder_gold cases are NOT
    populated."""
    from services.benchmark_recommendation import compute_recommendation
    pooled_cases = [
        {"benchmark_case_id": f"sq_{i}", "provisional": True,
         "by_mode": {
             "off": {"gold_in_top_10": 0, "gold_in_top_20": 0,
                      "recall_at_10": 0, "recall_at_20": 0,
                      "mrr": 0, "false_no_evidence_rate": 1.0,
                      "negatives_in_top_10": 0,
                      "wrong_specialty_rate": 0,
                      "wrong_population_rate": 0,
                      "wrong_intervention_rate": 0,
                      "latency_p95_ms": 0, "source_api_errors": 0},
             "shadow": {"gold_in_top_10": 0, "gold_in_top_20": 0,
                         "recall_at_10": 0, "recall_at_20": 0,
                         "mrr": 0, "false_no_evidence_rate": 1.0,
                         "negatives_in_top_10": 0,
                         "wrong_specialty_rate": 0,
                         "wrong_population_rate": 0,
                         "wrong_intervention_rate": 0,
                         "latency_p95_ms": 0, "source_api_errors": 0},
             "active_simulation": {"gold_in_top_10": 0, "gold_in_top_20": 0,
                                    "recall_at_10": 0, "recall_at_20": 0,
                                    "mrr": 0, "false_no_evidence_rate": 1.0,
                                    "negatives_in_top_10": 0,
                                    "wrong_specialty_rate": 0,
                                    "wrong_population_rate": 0,
                                    "wrong_intervention_rate": 0,
                                    "latency_p95_ms": 0,
                                    "source_api_errors": 0},
         }} for i in range(10)
    ]
    rec = compute_recommendation(
        pooled_cases=pooled_cases,
        safety_signals={
            "phi_expansion_triggered": False,
            "non_whitelisted_source_detected": False,
            "raw_phi_in_output": False, "raw_prompts_in_output": False,
            "raw_evidence_text_in_output": False,
            "unaudited_ranking_change": False,
        },
        run_ids=["r1"],
    )
    d = rec.to_dict()
    assert d["recommendation"] == "CONTINUE_SHADOW"
    assert d["verdict"] == d["recommendation"]
    assert "populate_more_gold_sets" in d["recommended_fixes"]


def test_recommendation_placeholder_gold_treated_as_not_populated():
    """Phase 6C — placeholder_gold cases excluded from the
    ``populated`` set so they don't trip a false GO with synthetic
    PMIDs."""
    from services.benchmark_recommendation import compute_recommendation
    pooled_cases = [
        {"benchmark_case_id": f"ph_{i}",
         "provisional": False, "placeholder_gold": True,
         "by_mode": {
             "off": {"gold_in_top_10": 0, "gold_in_top_20": 0,
                      "recall_at_10": 0, "recall_at_20": 0, "mrr": 0,
                      "false_no_evidence_rate": 1.0,
                      "negatives_in_top_10": 0,
                      "wrong_specialty_rate": 0,
                      "wrong_population_rate": 0,
                      "wrong_intervention_rate": 0,
                      "latency_p95_ms": 0, "source_api_errors": 0},
             "shadow": {"gold_in_top_10": 0, "gold_in_top_20": 0,
                         "recall_at_10": 0, "recall_at_20": 0, "mrr": 0,
                         "false_no_evidence_rate": 1.0,
                         "negatives_in_top_10": 0,
                         "wrong_specialty_rate": 0,
                         "wrong_population_rate": 0,
                         "wrong_intervention_rate": 0,
                         "latency_p95_ms": 0, "source_api_errors": 0},
             "active_simulation": {"gold_in_top_10": 0, "gold_in_top_20": 0,
                                    "recall_at_10": 0, "recall_at_20": 0,
                                    "mrr": 0,
                                    "false_no_evidence_rate": 1.0,
                                    "negatives_in_top_10": 0,
                                    "wrong_specialty_rate": 0,
                                    "wrong_population_rate": 0,
                                    "wrong_intervention_rate": 0,
                                    "latency_p95_ms": 0,
                                    "source_api_errors": 0},
         }} for i in range(8)
    ]
    rec = compute_recommendation(
        pooled_cases=pooled_cases,
        safety_signals={
            "phi_expansion_triggered": False,
            "non_whitelisted_source_detected": False,
            "raw_phi_in_output": False, "raw_prompts_in_output": False,
            "raw_evidence_text_in_output": False,
            "unaudited_ranking_change": False,
        },
        run_ids=["r1"],
    )
    assert rec.to_dict()["recommendation"] == "CONTINUE_SHADOW"


def test_recommendation_safety_failure_overrides_perf_gains():
    """Safety signals always pin to NO_GO."""
    from services.benchmark_recommendation import compute_recommendation
    rec = compute_recommendation(
        pooled_cases=[],
        safety_signals={
            "phi_expansion_triggered": True,
            "non_whitelisted_source_detected": False,
            "raw_phi_in_output": False, "raw_prompts_in_output": False,
            "raw_evidence_text_in_output": False,
            "unaudited_ranking_change": False,
        },
        run_ids=["r1"],
    )
    d = rec.to_dict()
    assert d["recommendation"] == "NO_GO"
    assert any("safety" in s for s in d["safety_failures"])


# ---------------------------------------------------------------------------
# C. Redacted JSON endpoint
# ---------------------------------------------------------------------------
def test_redacted_json_excludes_phi_evidence_prompts_secrets_uploads():
    """The redaction allowlist drops every field NOT explicitly named.
    We feed in a doc with synthetic PHI / synthetic prompts / synthetic
    secrets / synthetic uploaded text and verify none of those strings
    survive."""
    from services.benchmark_redaction import redact_benchmark_run
    import json as _json
    doc = {
        "id": "run_test_123",
        "schema_version": "phase6c_v1",
        "execution_mode": "live",
        "expansion_mode": "shadow",
        # Disallowed fields — must be stripped.
        "phi_payload": "John Doe MRN 12345678 DOB 03/14/1979",
        "raw_prompt": "<system>You are a clinician...</system>",
        "summarizer_prompt": "system: produce a JSON object...",
        "secrets": {"openai_api_key": "sk-XXXXXXXXXXXXXXXXXXXX",
                     "jwt_secret": "Bearer eyJabc.def"},
        "uploaded_user_text": "Patient evidence text that must never leak.",
        "evidence_text": "Some abstract content that must never leak.",
        "comparison_report": {
            "schema_version": "phase6c_v1",
            "modes_evaluated": ["off", "shadow", "active_simulation"],
            "per_case_three_mode": [
                {"benchmark_case_id": "milrinone_sah",
                  "by_mode": {"off": {"gold_in_top_10": 1}},
                  "expansion_candidates_json": [{
                      "pmid": "12345678",
                      "title": ("X" * 500),
                      "candidate_source": "openalex",
                      "candidate_tier": "citing_top_seeds",
                      "abstract_text": "should-be-stripped",
                      "raw_phi": "Jane Roe phone 555-1212",
                  }],
                  "promoted_candidates_json": [],
                  "demoted_candidates_json": [],
                  "top_10_baseline_json": ["A", "B"],
                  "top_10_active_simulation_json": ["A", "B"],
                  "status": "active_benchmark",
                  "raw_evidence_text": "should-be-stripped",
                  },
            ],
        },
    }
    out = redact_benchmark_run(doc)
    blob = _json.dumps(out)
    assert "MRN" not in blob
    assert "DOB" not in blob
    assert "<system>" not in blob
    assert "PROMPT" not in blob.upper() or "summarizer_prompt" not in blob
    assert "sk-XXXX" not in blob
    assert "Bearer" not in blob
    assert "uploaded_user_text" not in blob
    assert "Patient evidence text" not in blob
    assert "should-be-stripped" not in blob
    assert "raw_phi" not in blob
    # Title was truncated to 200 chars.
    candidate = out["comparison_report"]["per_case_three_mode"][0][
        "expansion_candidates_json"
    ][0]
    assert len(candidate["title"]) == 200
    # Allowlisted fields survived.
    assert out["execution_mode"] == "live"
    assert out["_redacted"] is True


# ---------------------------------------------------------------------------
# D. Admin authz
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_admin_can_view_benchmark_dashboard(client):
    admin = await _signup(client, "p6c_admin@test.com", role="admin")
    r = await client.get("/api/admin/benchmark-runs?limit=3", headers=admin)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_clinician_cannot_view_benchmark_dashboard(client):
    clin = await _signup(client, "p6c_clin@test.com")
    r = await client.get("/api/admin/benchmark-runs?limit=3", headers=clin)
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# E. PHI + expansion isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_phi_blocked_session_does_not_invoke_expansion(client):
    headers = await _signup(client, "p6c_phi@test.com")
    phi_q = ("45M with MRN 12345678 DOB 03/14/1979 email john@x.com -- "
              "best tx for HTN?")
    r = await client.post(
        "/api/search-sessions/",
        json={"question": phi_q}, headers=headers,
    )
    assert r.status_code == 200
    body = r.json()
    sid = body.get("session_id") or body.get("id")
    assert sid, body
    assert body["status"] == "blocked_phi"
    sess = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    # Phase 6C — closed-set per-session metadata for the PHI path.
    assert sess.get("active_expansion_applied") is False
    assert sess.get("active_expansion_candidate_count") == 0
    assert sess.get("active_expansion_latency_ms") == 0
    assert sess.get("active_expansion_reason") == "phi_blocked"
    # Scholar-like expansion never invoked → no payload persisted.
    assert sess.get("scholar_like_expansion_json") is None


# ---------------------------------------------------------------------------
# F. Cap-dominance over expansion signals
# ---------------------------------------------------------------------------
def test_expansion_promoted_article_still_capped_when_anchors_missing():
    """The active_simulation promotion gate only admits gold-positive
    candidates. We verify that a NON-gold candidate with an arbitrarily
    high citation/expansion score is NOT promoted — i.e. expansion
    signal alone cannot bypass the anchor / gold gate."""
    from services.benchmark_runs import _build_three_mode_report
    from services.search_benchmark import search_benchmark_service
    cases = search_benchmark_service.list_all()
    case = cases[0]
    case_result = {
        "id": case.id,
        "gold_pmids": ["GOLD_A", "GOLD_B"],
        "ranked_metrics": {
            "top_10_pmids": ["NEUTRAL_X"],
            "top_20_pmids": ["NEUTRAL_X"],
            "negatives_in_top_10_count": 0,
        },
        "scholar_like_expansion_json": {
            "candidates_by_tier": {
                "citing_top_seeds": [
                    # High-citation but NOT in gold — must not promote.
                    {"pmid": "HIGH_CITATION_BUT_NOT_GOLD",
                     "title": "very cited", "candidate_source": "openalex",
                     "candidate_tier": "citing_top_seeds",
                     "citation_score": 9999},
                ],
            },
        },
    }
    row = _build_three_mode_report([case_result], cases)["per_case_three_mode"][0]
    promoted_pmids = {p["pmid"] for p in row["promoted_candidates_json"]}
    assert "HIGH_CITATION_BUT_NOT_GOLD" not in promoted_pmids
    # Active-simulation gold count unchanged from off.
    assert (row["by_mode"]["active_simulation"]["gold_in_top_10"]
            == row["by_mode"]["off"]["gold_in_top_10"])


# ---------------------------------------------------------------------------
# G. Aggregate placeholder_gold flag on real cases
# ---------------------------------------------------------------------------
def test_real_benchmark_cases_marked_placeholder_or_real():
    """Sanity: at minimum the 5 known-synthetic-PMID cases carry
    ``placeholder_gold=True``; ``milrinone_sah`` does not."""
    from services.search_benchmark import search_benchmark_service
    by_id = {c.id: c for c in search_benchmark_service.list_all()}
    for cid in ("ddimer_pe_lowrisk", "apixaban_warfarin_afib",
                 "nsaid_aki_ckd_older", "septic_shock_background",
                 "colorectal_screening_guideline"):
        assert by_id[cid].placeholder_gold is True, cid
    assert by_id["milrinone_sah"].placeholder_gold is False
