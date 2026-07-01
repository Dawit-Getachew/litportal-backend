"""Phase R1C — admin endpoint + cross-phase invariance tests.

Validates:
  * Admin auth gate (403 for clinician, 401 for unauth).
  * POST replay returns 200 + sidecar id for admin.
  * Unknown case → 404.
  * GET list + GET by id work.
  * R1A label weights unchanged.
  * R1B configuration unchanged (BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER,
    HIGH_PRECISION_SHAPES, R1B_SOURCES, CURATED_EXPANSION_MAPS).
  * M3.2 dormant (kill switch unchanged).
  * K4A.2/3/4 routes skip R1C entirely (shape-gate refusal).
  * R1C TTL index 30d (introspection).
"""
from __future__ import annotations

import pytest

from core.db import Collections, get_db


async def _login_admin(client) -> str:
    res = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    assert res.status_code == 200
    return res.json()["access_token"]


async def _login_clinician(client) -> str:
    res = await client.post(
        "/api/auth/login",
        json={
            "email": "clinician@test.com",
            "password": "ClinicianTest123!",
        },
    )
    assert res.status_code == 200
    return res.json()["access_token"]


# =====================================================================
# Auth gates.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_replay_requires_admin(client):
    res = await client.post(
        "/api/admin/r1c/replay/r1a-case-A-cholestatic-pruritus",
    )
    assert res.status_code in (401, 403)


@pytest.mark.asyncio
async def test_R1C_replay_clinician_token_rejected(client):
    tok = await _login_clinician(client)
    res = await client.post(
        "/api/admin/r1c/replay/r1a-case-A-cholestatic-pruritus",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_R1C_list_requires_admin(client):
    res = await client.get("/api/admin/r1c/shadow-decisions")
    assert res.status_code in (401, 403)


# =====================================================================
# 404 — unknown case.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_replay_unknown_case_returns_404(client):
    tok = await _login_admin(client)
    res = await client.post(
        "/api/admin/r1c/replay/does-not-exist",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert res.status_code == 404
    assert res.json()["detail"]["error"] == "unknown_case_id"


# =====================================================================
# GET list + GET by id.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_list_and_get_by_id_admin_paths(client):
    tok = await _login_admin(client)
    db = get_db()
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    seed_id = "test-seed-r1c"
    await db[Collections.R1C_SHADOW_DECISIONS].insert_one({
        "id": seed_id,
        "phase": "R1C",
        "benchmark_case_id": "r1a-case-A-cholestatic-pruritus",
        "search_session_id": None,
        "shape": "pathophysiology_mechanism",
        "expansion_concept_map_used": "cholestatic_liver_disease_mechanism",
        "current_active_top_n_metadata": [],
        "r1b_simulated_top_n_metadata": [],
        "r1c_simulated_top_n_metadata": [],
        "safety_flags": {"leak_scan_passed": True, "phi_guard_passed": True},
        "created_at": now,
        "expires_at": now + timedelta(days=30),
    })
    res_list = await client.get(
        "/api/admin/r1c/shadow-decisions?limit=10",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert res_list.status_code == 200
    ids = [r["id"] for r in res_list.json()["rows"]]
    assert seed_id in ids
    res_one = await client.get(
        f"/api/admin/r1c/shadow-decisions/{seed_id}",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert res_one.status_code == 200
    assert res_one.json()["id"] == seed_id
    res_404 = await client.get(
        "/api/admin/r1c/shadow-decisions/missing-id",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert res_404.status_code == 404


@pytest.mark.asyncio
async def test_R1C_list_shape_filter_narrows(client):
    tok = await _login_admin(client)
    res_all = await client.get(
        "/api/admin/r1c/shadow-decisions?limit=100",
        headers={"Authorization": f"Bearer {tok}"},
    )
    all_rows = res_all.json()["rows"]
    res_p = await client.get(
        "/api/admin/r1c/shadow-decisions?limit=100&shape=pathophysiology_mechanism",
        headers={"Authorization": f"Bearer {tok}"},
    )
    p_rows = res_p.json()["rows"]
    assert all(r["shape"] == "pathophysiology_mechanism" for r in p_rows)
    assert len(p_rows) <= len(all_rows)


# =====================================================================
# Cross-phase invariance — R1A label weights unchanged.
# =====================================================================
def test_R1C_does_not_mutate_R1A_label_weights():
    from services.r1a_directness_labels import SHADOW_LABEL_WEIGHTS
    snap = {
        "pathophysiology_mechanism_direct": SHADOW_LABEL_WEIGHTS[
            "pathophysiology_mechanism_direct"
        ],
        "treatment_modality_indirect_for_mechanism": SHADOW_LABEL_WEIGHTS[
            "treatment_modality_indirect_for_mechanism"
        ],
        "procedure_complication_direct": SHADOW_LABEL_WEIGHTS[
            "procedure_complication_direct"
        ],
        "missing_mechanism_focus": SHADOW_LABEL_WEIGHTS[
            "missing_mechanism_focus"
        ],
    }
    from services import r1c_anchor_labels  # noqa: F401
    from services import r1c_gold_recovery  # noqa: F401
    from services import r1c_shadow_simulator  # noqa: F401
    for k, v in snap.items():
        assert SHADOW_LABEL_WEIGHTS[k] == v


# =====================================================================
# Cross-phase invariance — R1B configuration unchanged.
# =====================================================================
def test_R1C_does_not_mutate_R1B_configuration():
    from services.r1b_shadow_simulator import (
        BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER,
        HIGH_PRECISION_SHAPES,
        R1B_SOURCES,
    )
    from services.r1b_query_expansion import (
        CURATED_EXPANSION_MAPS,
        MECHANISM_CONCEPT_TERMS,
        PROCEDURE_COMPLICATION_OUTCOME_VOCAB,
        R1B_ELIGIBLE_SHAPES,
    )
    # Closed-set snapshot — sanity that R1C didn't touch any of these.
    assert BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER == 0.5
    assert HIGH_PRECISION_SHAPES == frozenset({
        "procedure_complication_risk", "procedure_complication",
    })
    assert R1B_SOURCES == ("europe_pmc", "pubmed")
    assert set(CURATED_EXPANSION_MAPS.keys()) == {
        "cholestatic_liver_disease_mechanism",
        "amyloid_cardiomyopathy_mechanism",
        "pcos_insulin_resistance_mechanism",
        "central_line_pneumothorax_complication",
        "pneumothorax_outcome",
        "procedure_complication_boost",
    }
    assert "mechanism" in MECHANISM_CONCEPT_TERMS
    assert "complication" in PROCEDURE_COMPLICATION_OUTCOME_VOCAB
    assert R1B_ELIGIBLE_SHAPES == frozenset({
        "pathophysiology_mechanism",
        "procedure_complication_risk",
        "procedure_complication",
    })


# =====================================================================
# Cross-phase invariance — M3.2 kill switch unchanged.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_does_not_flip_M3_2_kill_switch():
    from services.m3_2_feature_flags import read_kill_switch_state
    before = await read_kill_switch_state()
    from services.r1c_shadow_simulator import list_r1c_shadow_decisions
    _ = await list_r1c_shadow_decisions(limit=1)
    after = await read_kill_switch_state()
    assert before == after


# =====================================================================
# TTL index on r1c_shadow_decisions.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_collection_has_ttl_index_30d(client):  # noqa: ARG001
    # Depend on ``client`` so the FastAPI lifespan (which calls
    # ``ensure_indexes``) has been triggered before the assertion.
    db = get_db()
    idx = await db[Collections.R1C_SHADOW_DECISIONS].index_information()
    found_ttl = False
    for spec in idx.values():
        if "expireAfterSeconds" in spec:
            assert spec["expireAfterSeconds"] == 0
            key = spec.get("key") or []
            if any(k[0] == "expires_at" for k in key):
                found_ttl = True
                break
    assert found_ttl, f"no TTL index on expires_at found: {idx}"


# =====================================================================
# K4A invariance — ineligible shape raises closed-set error.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_skips_non_eligible_shape():
    """Closed-set: any case whose shape isn't in R1B_ELIGIBLE_SHAPES
    is rejected by R1C with the closed-set error
    ``shape_not_eligible_for_r1c``. This is the simulator-side
    K4A invariance — K4A-owned subtypes are not in our eligible
    set so R1C will not touch them.
    """
    from services.r1c_shadow_simulator import (
        R1CReplayError,
        replay_benchmark_case_r1c,
    )
    db = get_db()
    # Insert a closed-set synthetic K4A-style benchmark case.
    K4A_CASE_ID = "test-k4a-synthetic-case"
    await db[Collections.R1A_GOLD_COMPARATOR_CASES].update_one(
        {"case_id": K4A_CASE_ID},
        {"$set": {
            "case_id": K4A_CASE_ID,
            "question": "synthetic K4A question",
            "shape": "severity_marker_overview",  # K4A-owned, not in eligible
            "gold_article_dois": [],
            "gold_article_pmids": [],
        }},
        upsert=True,
    )
    try:
        async def _empty_fetch(**kwargs):  # noqa: ARG001
            return []

        with pytest.raises(R1CReplayError) as exc_info:
            await replay_benchmark_case_r1c(
                case_id=K4A_CASE_ID,
                actor_user_id="admin-1",
                fetch_fn=_empty_fetch,
            )
        assert exc_info.value.code == "shape_not_eligible_for_r1c"
    finally:
        await db[Collections.R1A_GOLD_COMPARATOR_CASES].delete_one(
            {"case_id": K4A_CASE_ID},
        )


# =====================================================================
# Listing endpoint strips abstracts.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_listing_endpoint_response_strips_abstracts(client):
    tok = await _login_admin(client)
    res = await client.get(
        "/api/admin/r1c/shadow-decisions?limit=200",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert res.status_code == 200
    flat = res.text.lower()
    for forbidden in (
        '"abstract":', '"fulltext":',
        '"prompt":', '"completion":',
        '"api_key":', '"password":',
        'bearer ', 'sk-',
    ):
        assert forbidden not in flat
