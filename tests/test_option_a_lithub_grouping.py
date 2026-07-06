"""Option A — surface LitPortal-saved papers in LitPulse as one "LitPortal Saved"
group, with sub-folders drillable via LitArchive.

Backend pieces implemented in litportal-backend:
  1. A DURABLE origin marker (`portal_engine_record_id = "litportal:<article_id>"`)
     on BOTH LitHub mirror payloads (library + research-collection), so LitPulse
     can group reliably even after a cross-app re-save overwrites `source`.
  2. `GET /api/research-collections/tree` — the full collection hierarchy + each
     collection's papers' canonical ids, for LitArchive to render the sub-folders.

Uses the session-scoped ``client`` fixture + seeded clinician. asyncio_mode=auto.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest_asyncio

from core.config import get_settings
from core.db import Collections, get_db
from services.research_collection_service import research_collection_service
from services.user_library import library_service


async def _login(client) -> dict[str, str]:
    r = await client.post("/api/auth/login", json={
        "email": "clinician@test.com", "password": "ClinicianTest123!"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest_asyncio.fixture()
async def auth(client):
    return await _login(client)


async def _me_id(client, auth) -> str:
    return (await client.get("/api/auth/me", headers=auth)).json()["id"]


async def _seed_article(db, **over) -> str:
    aid = "oa-art-" + uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc)
    doc = {
        "id": aid, "canonical_key": "pmid:" + str(uuid.uuid4().int)[:8],
        "title": "Paper " + aid, "journal": "NEJM", "publication_year": 2024,
        "pmid": str(uuid.uuid4().int)[:8], "doi": f"10.1/{aid}",
        "doi_normalized": f"10.1/{aid}", "abstract": "abs",
        "authors_json": [{"fullName": "Doe J"}], "sources_present": ["pubmed"],
        "created_at": now, "updated_at": now,
    }
    doc.update(over)
    await db[Collections.ARTICLES].insert_one(doc)
    return aid


# ── 1. Durable origin marker on the mirror payloads ──────────────────
async def test_library_mirror_item_carries_durable_marker(client):
    db = get_db()
    aid = await _seed_article(db)
    item = await library_service._article_lithub_item(aid)
    assert item is not None
    assert item["portal_engine_record_id"] == f"litportal:{aid}"
    assert item["source"].startswith("litportal")


async def test_collection_mirror_sends_durable_marker(client, monkeypatch):
    """The research-collection LitHub mirror must include the durable marker."""
    db = get_db()
    # Enable dual-write + a base url on the settings singleton.
    s = get_settings()
    monkeypatch.setattr(s, "LITPORTAL_DUAL_WRITE_LITHUB", True)
    monkeypatch.setattr(s, "LITHUB_BASE_URL", "http://lithub.test")
    # A user WITH an Identity sub (mirror only fires for central users).
    uid = "oa-u-" + uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc)
    await db[Collections.USERS].insert_one({
        "id": uid, "identity_id": "sub-" + uuid.uuid4().hex[:8],
        "email": f"{uid}@test.com", "name": uid, "hashed_password": "",
        "role": "user", "created_at": now, "updated_at": now,
    })
    aid = await _seed_article(db)
    # Capture the item handed to LitHub.
    captured = {}

    async def fake_save(self, user_id, item):  # noqa: ANN001
        captured["item"] = item
        return {"paper_id": "P-" + uuid.uuid4().hex[:6]}

    from clients.lithub_client import LitHubClient
    monkeypatch.setattr(LitHubClient, "internal_save_paper", fake_save)

    paper_id = await research_collection_service._mirror_item_to_lithub(uid, aid)
    assert captured.get("item"), "mirror did not call internal_save_paper"
    assert captured["item"]["portal_engine_record_id"] == f"litportal:{aid}"
    assert captured["item"]["source"].startswith("litportal")
    assert paper_id and paper_id.startswith("P-")


# ── 2. The /tree endpoint ────────────────────────────────────────────
async def test_collections_tree_endpoint(client, auth):
    db = get_db()
    uid = await _me_id(client, auth)
    # parent collection + child + one paper each.
    r = await client.post("/api/research-collections", json={"name": "Cardiology"}, headers=auth)
    parent = r.json()["id"]
    r = await client.post("/api/research-collections",
                          json={"name": "HFpEF", "parent_id": parent}, headers=auth)
    child = r.json()["id"]
    aid = await _seed_article(db)
    ra = await client.post(f"/api/research-collections/{child}/records",
                           json={"records": [{"article_id": aid}]}, headers=auth)
    assert ra.status_code == 201, ra.text

    rt = await client.get("/api/research-collections/tree", headers=auth)
    assert rt.status_code == 200, rt.text
    tree = rt.json()["collections"]
    by_id = {c["id"]: c for c in tree}
    assert parent in by_id and child in by_id
    # hierarchy preserved
    assert by_id[child]["parent_id"] == parent
    # the child carries the paper with canonical ids for matching to LitHub
    papers = by_id[child]["papers"]
    assert len(papers) == 1
    p = papers[0]
    assert p["article_id"] == aid
    assert p["pmid"] and p["doi"] and p["canonical_key"]
    assert by_id[child]["item_count"] == 1


async def test_tree_route_not_shadowed_by_collection_id(client, auth):
    """/tree must resolve to the tree endpoint, not GET /{collection_id}."""
    r = await client.get("/api/research-collections/tree", headers=auth)
    assert r.status_code == 200
    assert "collections" in r.json()  # tree shape, not a single-collection 404


async def test_tree_requires_auth(client):
    r = await client.get("/api/research-collections/tree")
    assert r.status_code in (401, 403)
