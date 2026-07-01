"""Phase K5.0 — Guideline Source Registry + permission model tests.

Validates:
  * GuidelineSourceDoc safe defaults (ai_ingestion_allowed=False).
  * Registry seeded with 6 closed-set sources; only pubmed_guidelines live.
  * Every guideline source has ai_ingestion_allowed=False.
  * Admin-only read endpoint returns the registry; non-admin is rejected.
"""
from __future__ import annotations

import uuid

import pytest

from core.db import Collections, get_db
from models.guideline_sources import GuidelineSourceDoc
from server import GUIDELINE_SOURCE_REGISTRY, _seed_guideline_sources

_EXPECTED = {
    "pubmed_guidelines", "uspstf", "cdc_hhs", "nice", "who", "specialty_society",
}


def test_K5_0_model_safe_defaults():
    doc = GuidelineSourceDoc(source_name="x")
    assert doc.ai_ingestion_allowed is False
    assert doc.retrieval_allowed is False
    assert doc.link_out_allowed is True
    assert doc.display_snippet_allowed == "verify"
    assert doc.connector_status == "scaffolded"
    assert doc.enabled is False


def test_K5_0_registry_seed_list_all_ai_ingestion_false():
    names = {e["source_name"] for e in GUIDELINE_SOURCE_REGISTRY}
    assert _EXPECTED.issubset(names)
    for e in GUIDELINE_SOURCE_REGISTRY:
        assert e["ai_ingestion_allowed"] is False, e["source_name"]  # hard invariant
    by_name = {e["source_name"]: e for e in GUIDELINE_SOURCE_REGISTRY}
    # Live + retrieval-allowed: pubmed_guidelines (K5.1) and the no-key
    # official connector myhealthfinder_hhs (K5.2A) if present.
    for live_src in ("pubmed_guidelines",):
        assert by_name[live_src]["connector_status"] == "live"
        assert by_name[live_src]["retrieval_allowed"] is True
        assert by_name[live_src]["enabled"] is True
    if "myhealthfinder_hhs" in by_name:
        assert by_name["myhealthfinder_hhs"]["connector_status"] == "live"
        assert by_name["myhealthfinder_hhs"]["requires_api_key"] is False
        assert by_name["myhealthfinder_hhs"]["ai_ingestion_allowed"] is False
    # Keyed / uncertain official sources stay scaffolded.
    for n in ("uspstf", "cdc_hhs", "nice", "who", "specialty_society"):
        assert by_name[n]["connector_status"] == "scaffolded"
        assert by_name[n]["retrieval_allowed"] is False
        assert by_name[n]["enabled"] is False


@pytest.mark.asyncio
async def test_K5_0_registry_seeded_in_db():
    await _seed_guideline_sources()  # idempotent
    db = get_db()
    rows = await db[Collections.GUIDELINE_SOURCES].find(
        {}, {"_id": 0},
    ).to_list(length=50)
    names = {r["source_name"] for r in rows}
    assert _EXPECTED.issubset(names)
    for r in rows:
        if r["source_name"] in _EXPECTED:
            assert r["ai_ingestion_allowed"] is False, r["source_name"]


@pytest.mark.asyncio
async def test_K5_0_admin_endpoint_returns_registry(client):
    r = await client.post("/api/auth/login", json={
        "email": "admin@test.com", "password": "AdminTest123!",
    })
    token = r.json()["access_token"]
    resp = await client.get(
        "/api/admin/source-configs-readonly/guideline-sources",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    names = {row["source_name"] for row in body}
    assert _EXPECTED.issubset(names)
    for row in body:
        if row["source_name"] in _EXPECTED:
            assert row["ai_ingestion_allowed"] is False


@pytest.mark.asyncio
async def test_K5_0_admin_endpoint_rejects_non_admin(client):
    email = f"k5user-{uuid.uuid4().hex[:8]}@k5-test.com"
    await client.post("/api/auth/signup", json={
        "email": email, "password": "K5SuperTest!23", "name": "K5",
    })
    r = await client.post("/api/auth/login", json={
        "email": email, "password": "K5SuperTest!23",
    })
    token = r.json()["access_token"]
    resp = await client.get(
        "/api/admin/source-configs-readonly/guideline-sources",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code in (401, 403)
