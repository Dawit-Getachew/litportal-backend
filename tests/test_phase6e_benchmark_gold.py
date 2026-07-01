"""Phase 6E — Benchmark gold curation + governance tests.

Covers A-G per the Phase 6E brief:
- A: Add candidate (closed-set role/lane/identifier validation; idempotency).
- B: Seed import (dry-run + idempotent; trusted-domain handling).
- C: Accept / reject candidate status transitions + closed-set audit events.
- D: Activation rules — case with no gold cannot become active_benchmark.
- E: Trusted-domain validation (cdc.gov passes; random domain → needs_review).
- F: Clinician 403 on all curation endpoints.
- G: Recommendation engine still returns CONTINUE_SHADOW when populated < 5.
"""
from __future__ import annotations

import uuid

import pytest

from core.db import Collections, get_db


async def _admin_login(client):
    r = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    if r.status_code != 200:
        pytest.skip("Admin seed credentials not present.")
    return r.json()["access_token"]


async def _clinician_signup(client) -> str:
    email = f"clin_p6e_{uuid.uuid4().hex[:8]}@p6e-test.com"
    r = await client.post(
        "/api/auth/signup",
        json={
            "email": email, "password": "P6eTesting1234!",
            "name": "Clin P6E",
        },
    )
    assert r.status_code in (200, 201)
    return r.json()["access_token"]


# ============================================================
# A — Add candidate (validation + idempotency)
# ============================================================
@pytest.mark.asyncio
async def test_add_candidate_validates_pmid_shape(client):
    token = await _admin_login(client)
    r = await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "varicella_schedule_and_benefit",
            "identifier_type": "pmid",
            "identifier_value": "not-a-pmid",
            "relevance_role": "primary_gold",
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"] == "identifier_invalid:invalid_pmid_shape"


@pytest.mark.asyncio
async def test_add_candidate_validates_doi_shape(client):
    token = await _admin_login(client)
    r = await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "varicella_schedule_and_benefit",
            "identifier_type": "doi",
            "identifier_value": "totally-not-a-doi",
            "relevance_role": "primary_gold",
        },
    )
    assert r.status_code == 422
    assert r.json()["detail"] == "identifier_invalid:invalid_doi_shape"


@pytest.mark.asyncio
async def test_add_candidate_unknown_case_404(client):
    token = await _admin_login(client)
    r = await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "does_not_exist",
            "identifier_type": "pmid",
            "identifier_value": "12345",
            "relevance_role": "primary_gold",
        },
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "benchmark_case_not_found"


@pytest.mark.asyncio
async def test_add_candidate_idempotent(client):
    token = await _admin_login(client)
    pmid = str(10000 + uuid.uuid4().int % 10000)
    payload = {
        "benchmark_case_id": "varicella_schedule_and_benefit",
        "identifier_type": "pmid",
        "identifier_value": pmid,
        "relevance_role": "primary_gold",
        "expected_lane": "schedule",
        "rationale": "Idempotency test fixture.",
    }
    headers = {"Authorization": f"Bearer {token}"}
    r1 = await client.post(
        "/api/admin/benchmark-gold/candidates", headers=headers, json=payload,
    )
    r2 = await client.post(
        "/api/admin/benchmark-gold/candidates", headers=headers, json=payload,
    )
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["id"] == r2.json()["id"]


@pytest.mark.asyncio
async def test_add_candidate_rejects_invalid_relevance_role(client):
    token = await _admin_login(client)
    r = await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "varicella_schedule_and_benefit",
            "identifier_type": "pmid",
            "identifier_value": "12345",
            "relevance_role": "landmark",  # NOT in closed set
        },
    )
    assert r.status_code == 422  # pydantic rejects


# ============================================================
# B — Seed import (dry-run + idempotent + trusted-domain)
# ============================================================
@pytest.mark.asyncio
async def test_seed_import_dry_run_creates_no_docs(client):
    token = await _admin_login(client)
    db = get_db()
    before = await db[Collections.BENCHMARK_GOLD_CANDIDATES].count_documents({})
    body = {
        "schema_version": "phase6e_v1",
        "cases": [
            {
                "benchmark_case_id": "varicella_schedule_and_benefit",
                "candidates": [
                    {
                        "identifier_type": "official_source_url",
                        "identifier_value":
                            "https://www.cdc.gov/vaccines/vpd/varicella/hcp/recommendations.html",
                        "relevance_role": "official_guideline",
                        "expected_lane": "guideline",
                    },
                ],
            },
        ],
    }
    r = await client.post(
        "/api/admin/benchmark-gold/import?dry_run=true",
        headers={"Authorization": f"Bearer {token}"},
        json=body,
    )
    assert r.status_code == 200
    rr = r.json()
    assert rr["dry_run"] is True
    assert rr["n_proposed_total"] == 1
    after = await db[Collections.BENCHMARK_GOLD_CANDIDATES].count_documents({})
    assert after == before, "Dry-run must not write."


@pytest.mark.asyncio
async def test_seed_import_idempotent_real_run(client):
    token = await _admin_login(client)
    body = {
        "schema_version": "phase6e_v1",
        "cases": [
            {
                "benchmark_case_id": "septic_shock_background",
                "candidates": [
                    {
                        "identifier_type": "official_source_url",
                        "identifier_value":
                            "https://www.sccm.org/SurvivingSepsisCampaign/Guidelines",
                        "relevance_role": "official_guideline",
                        "expected_lane": "background",
                    },
                ],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {token}"}
    r1 = await client.post(
        "/api/admin/benchmark-gold/import",
        headers=headers, json=body,
    )
    r2 = await client.post(
        "/api/admin/benchmark-gold/import",
        headers=headers, json=body,
    )
    assert r1.status_code == 200 and r2.status_code == 200
    j2 = r2.json()
    # Second call MUST skip the candidate as existing.
    assert j2["n_skipped_existing"] >= 1
    assert j2["n_created"] == 0


@pytest.mark.asyncio
async def test_seed_import_unknown_case_skipped(client):
    token = await _admin_login(client)
    body = {
        "schema_version": "phase6e_v1",
        "cases": [
            {
                "benchmark_case_id": "definitely_not_a_real_case",
                "candidates": [
                    {
                        "identifier_type": "pmid",
                        "identifier_value": "12345",
                        "relevance_role": "primary_gold",
                    },
                ],
            },
        ],
    }
    r = await client.post(
        "/api/admin/benchmark-gold/import",
        headers={"Authorization": f"Bearer {token}"},
        json=body,
    )
    assert r.status_code == 200
    j = r.json()
    assert j["n_skipped_unknown_case"] == 1
    assert j["n_created"] == 0


# ============================================================
# C — Accept / reject candidate status transitions
# ============================================================
@pytest.mark.asyncio
async def test_candidate_accept_reject_audits(client):
    token = await _admin_login(client)
    db = get_db()
    cand = (await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "varicella_schedule_and_benefit",
            "identifier_type": "pmid",
            "identifier_value": str(20000 + uuid.uuid4().int % 10000),
            "relevance_role": "primary_gold",
        },
    )).json()
    before_accept = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "benchmark_gold_candidate_accepted"},
    )
    r = await client.patch(
        f"/api/admin/benchmark-gold/candidates/{cand['id']}",
        headers={"Authorization": f"Bearer {token}"},
        json={"status": "accepted"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"
    after_accept = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "benchmark_gold_candidate_accepted"},
    )
    assert after_accept == before_accept + 1
    # Reject path.
    r2 = await client.patch(
        f"/api/admin/benchmark-gold/candidates/{cand['id']}",
        headers={"Authorization": f"Bearer {token}"},
        json={"status": "rejected", "reviewer_notes": "duplicate of #12345"},
    )
    assert r2.status_code == 200
    assert r2.json()["status"] == "rejected"


# ============================================================
# D — Activation rule enforcement
# ============================================================
@pytest.mark.asyncio
async def test_activation_blocked_when_no_gold(client):
    token = await _admin_login(client)
    # apixaban_warfarin_afib has placeholder gold and no real gold.
    # Wipe candidates first to make this deterministic.
    db = get_db()
    await db[Collections.BENCHMARK_GOLD_CANDIDATES].delete_many(
        {"benchmark_case_id": "apixaban_warfarin_afib"},
    )
    r = await client.post(
        "/api/admin/benchmark-gold/cases/apixaban_warfarin_afib/activate",
        headers={"Authorization": f"Bearer {token}"},
        json={"activate_reason": "test"},
    )
    assert r.status_code == 409
    assert "activation_blocked" in r.json()["detail"]
    assert "no_accepted_gold" in r.json()["detail"]


@pytest.mark.asyncio
async def test_activation_succeeds_after_gold_accepted(client):
    token = await _admin_login(client)
    # Use varicella_schedule_and_benefit which has must_have_anchors +
    # expected_type_of_question + accepted candidates from the seed.
    cand = (await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "varicella_schedule_and_benefit",
            "identifier_type": "pmid",
            "identifier_value": str(30000 + uuid.uuid4().int % 10000),
            "relevance_role": "primary_gold",
        },
    )).json()
    await client.patch(
        f"/api/admin/benchmark-gold/candidates/{cand['id']}",
        headers={"Authorization": f"Bearer {token}"},
        json={"status": "accepted"},
    )
    r = await client.post(
        "/api/admin/benchmark-gold/cases/varicella_schedule_and_benefit/activate",
        headers={"Authorization": f"Bearer {token}"},
        json={"activate_reason": "test_activate"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["governance_status"] == "active_benchmark"
    # The case should now show active_benchmark in /cases.
    cases = (await client.get(
        "/api/admin/benchmark-gold/cases",
        headers={"Authorization": f"Bearer {token}"},
    )).json()
    case = next(
        c for c in cases if c["id"] == "varicella_schedule_and_benefit"
    )
    assert case["governance_status"] == "active_benchmark"
    # Cleanup — re-retire so other tests aren't affected.
    await client.post(
        "/api/admin/benchmark-gold/cases/varicella_schedule_and_benefit/retire",
        headers={"Authorization": f"Bearer {token}"},
        json={"retire_reason": "test_cleanup"},
    )


# ============================================================
# E — Trusted-domain validation
# ============================================================
@pytest.mark.asyncio
async def test_trusted_domain_url_accepted(client):
    token = await _admin_login(client)
    r = await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "colorectal_screening_guideline",
            "identifier_type": "official_source_url",
            "identifier_value":
                "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/colorectal-cancer-screening",
            "relevance_role": "official_guideline",
            "expected_lane": "guideline",
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "pending_review"
    assert body["domain_trust"] == "trusted"
    assert body["domain_host"].endswith("uspreventiveservicestaskforce.org")


@pytest.mark.asyncio
async def test_untrusted_url_flagged_needs_review(client):
    token = await _admin_login(client)
    r = await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "colorectal_screening_guideline",
            "identifier_type": "official_source_url",
            "identifier_value": "https://random-evil-blog.example.com/screening",
            "relevance_role": "official_guideline",
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "needs_review"
    assert body["reviewer_notes"] == "untrusted_domain"
    assert body["domain_trust"] == "untrusted"


@pytest.mark.asyncio
async def test_http_only_url_rejected_for_official(client):
    token = await _admin_login(client)
    # HTTP (not HTTPS) → marked untrusted because allowlist requires
    # HTTPS.
    r = await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "varicella_schedule_and_benefit",
            "identifier_type": "official_source_url",
            "identifier_value": "http://www.cdc.gov/some/page.html",
            "relevance_role": "official_guideline",
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "needs_review"
    assert body["domain_trust"] == "untrusted"


# ============================================================
# F — Clinician 403
# ============================================================
@pytest.mark.asyncio
async def test_clinician_403_on_candidates_post(client):
    token = await _clinician_signup(client)
    r = await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "varicella_schedule_and_benefit",
            "identifier_type": "pmid",
            "identifier_value": "12345",
            "relevance_role": "primary_gold",
        },
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_clinician_403_on_import(client):
    token = await _clinician_signup(client)
    r = await client.post(
        "/api/admin/benchmark-gold/import",
        headers={"Authorization": f"Bearer {token}"},
        json={"schema_version": "phase6e_v1", "cases": []},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_clinician_403_on_cases_list(client):
    token = await _clinician_signup(client)
    r = await client.get(
        "/api/admin/benchmark-gold/cases",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


# ============================================================
# G — Recommendation engine still CONTINUE_SHADOW when populated < 5
# ============================================================
@pytest.mark.asyncio
async def test_recommendation_stays_continue_shadow(client):
    """Phase 6E — Adding gold candidates to one case does NOT change
    the recommendation engine's sufficiency floor. Until 5 different
    cases graduate to active_benchmark, the verdict stays CONTINUE_SHADOW.
    """
    token = await _admin_login(client)
    r = await client.get(
        "/api/admin/benchmark-runs/recommendation",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    # Either CONTINUE_SHADOW or NO_GO (safety failure) — never GO with
    # the current state of the curation set.
    assert body.get("recommendation") in (
        "CONTINUE_SHADOW", "NO_GO",
    ), body


# ============================================================
# Trusted-domain pure-function unit tests
# ============================================================
def test_trusted_domain_helper_pure():
    from services.trusted_domains import (
        classify_official_url, is_trusted_official_url,
    )
    assert is_trusted_official_url(
        "https://www.cdc.gov/vaccines/vpd/varicella/hcp/recommendations.html",
    )
    assert is_trusted_official_url(
        "https://www.who.int/news-room/fact-sheets",
    )
    assert not is_trusted_official_url(
        "https://random-evil-blog.example.com/x",
    )
    assert not is_trusted_official_url(
        "http://www.cdc.gov/some/page.html",
    )
    assert not is_trusted_official_url(None)
    assert not is_trusted_official_url("")
    c = classify_official_url("https://www.cdc.gov/x.html")
    assert c["is_trusted"] is True
    assert c["host"] == "www.cdc.gov"
    c2 = classify_official_url("http://www.cdc.gov/x.html")
    assert c2["is_trusted"] is False
    assert c2["reason"] == "not_https"
    c3 = classify_official_url("https://attacker.example.com/x")
    assert c3["is_trusted"] is False
    assert c3["reason"] == "untrusted_domain"


# ============================================================
# Audit safety — verify payloads carry no raw URL paths
# ============================================================
@pytest.mark.asyncio
async def test_audit_payloads_carry_no_url_paths(client):
    token = await _admin_login(client)
    cand = (await client.post(
        "/api/admin/benchmark-gold/candidates",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "benchmark_case_id": "varicella_schedule_and_benefit",
            "identifier_type": "official_source_url",
            "identifier_value":
                "https://www.cdc.gov/vaccines/vpd/varicella/hcp/recommendations.html",
            "relevance_role": "official_guideline",
            "expected_lane": "guideline",
            # Rationale stays under the 200-char limit; the audit
            # payload should NOT carry it at all (only closed-set
            # keys).
            "rationale": "ACIP canonical reference for varicella vaccine.",
        },
    )).json()
    db = get_db()
    audit = await db[Collections.AUDIT_EVENTS].find_one(
        {
            "event_type": "benchmark_official_source_added",
            "event_payload_json.candidate_id": cand["id"],
        },
    )
    assert audit, "Expected audit event not found"
    payload = audit["event_payload_json"]
    # Must contain only the host portion, not the path.
    assert payload.get("domain_host") == "www.cdc.gov"
    blob = str(payload)
    assert "/vaccines/" not in blob
    assert "/recommendations.html" not in blob
    # Rationale never carried in the closed-set audit payload.
    assert "ACIP" not in blob
    assert "rationale" not in payload
