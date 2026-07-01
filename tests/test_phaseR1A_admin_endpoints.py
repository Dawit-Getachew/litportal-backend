"""Phase R1A — Part 8 — admin endpoint integration tests.

Validates:
  * ``GET /api/admin/r1a/shadow-decisions`` requires admin auth.
  * Shape filter narrows results.
  * ``GET /api/admin/r1a/benchmark-cases`` returns the 4 seeded
    closed-set cases.
  * ``POST /api/admin/r1a/replay/{case_id}`` is synchronous,
    returns a sidecar_doc_id, writes one row.
  * Unknown case_id → 404.
"""
from __future__ import annotations

import pytest


async def _login_admin(client) -> str:
    res = await client.post(
        "/api/auth/login",
        json={
            "email": "admin@test.com",
            "password": "AdminTest123!",
        },
    )
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


async def _login_clinician(client) -> str:
    res = await client.post(
        "/api/auth/login",
        json={
            "email": "clinician@test.com",
            "password": "ClinicianTest123!",
        },
    )
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


# =====================================================================
# Auth — admin required, clinician rejected.
# =====================================================================
@pytest.mark.asyncio
async def test_R1A_shadow_decisions_requires_admin(client):
    res = await client.get("/api/admin/r1a/shadow-decisions")
    assert res.status_code in (401, 403)


@pytest.mark.asyncio
async def test_R1A_shadow_decisions_clinician_token_rejected(client):
    token = await _login_clinician(client)
    res = await client.get(
        "/api/admin/r1a/shadow-decisions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_R1A_benchmark_cases_returns_four_seeded_cases(client):
    token = await _login_admin(client)
    res = await client.get(
        "/api/admin/r1a/benchmark-cases",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 4
    case_ids = {c["case_id"] for c in data["cases"]}
    assert case_ids == {
        "r1a-case-A-cholestatic-pruritus",
        "r1a-case-B-pcos-insulin-resistance",
        "r1a-case-C-amyloid-diastolic-dysfunction",
        "r1a-case-D-cvc-pneumothorax",
    }


@pytest.mark.asyncio
async def test_R1A_shadow_decisions_shape_filter_narrows(client):
    token = await _login_admin(client)
    # Trigger one replay so we have a row.
    rep = await client.post(
        "/api/admin/r1a/replay/r1a-case-A-cholestatic-pruritus",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert rep.status_code == 200
    # All shapes filter.
    res_all = await client.get(
        "/api/admin/r1a/shadow-decisions?limit=50",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_all.status_code == 200
    all_rows = res_all.json()["rows"]
    # Patho filter.
    res_p = await client.get(
        "/api/admin/r1a/shadow-decisions?limit=50&shape=pathophysiology_mechanism",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_p.status_code == 200
    patho_rows = res_p.json()["rows"]
    assert all(r["shape"] == "pathophysiology_mechanism" for r in patho_rows)
    # Filter narrows or equals.
    assert len(patho_rows) <= len(all_rows)


@pytest.mark.asyncio
async def test_R1A_replay_unknown_case_returns_404(client):
    token = await _login_admin(client)
    res = await client.post(
        "/api/admin/r1a/replay/does-not-exist",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 404
    body = res.json()
    assert body["detail"]["status"] == "unknown_case_id"


@pytest.mark.asyncio
async def test_R1A_replay_known_case_returns_sidecar_doc_id(client):
    token = await _login_admin(client)
    res = await client.post(
        "/api/admin/r1a/replay/r1a-case-D-cvc-pneumothorax",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "sidecar_doc_id" in data and isinstance(data["sidecar_doc_id"], str)
    assert data["case_id"] == "r1a-case-D-cvc-pneumothorax"
    assert data["shape"] == "procedure_complication_risk"


@pytest.mark.asyncio
async def test_R1A_replay_writes_row_visible_in_list(client):
    token = await _login_admin(client)
    rep = await client.post(
        "/api/admin/r1a/replay/r1a-case-C-amyloid-diastolic-dysfunction",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert rep.status_code == 200
    sid = rep.json()["sidecar_doc_id"]
    res = await client.get(
        "/api/admin/r1a/shadow-decisions?limit=200",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    rows = res.json()["rows"]
    assert any(r["id"] == sid for r in rows)
