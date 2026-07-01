"""Integration tests for the Research Collections + LitArchive APIs.

Uses the session-scoped ``client`` fixture (ASGI over a real test Mongo) and the
seeded clinician account. asyncio_mode=auto.
"""
from __future__ import annotations

import uuid

import pytest_asyncio

from core.db import Collections, get_db


async def _login(client) -> dict[str, str]:
    r = await client.post(
        "/api/auth/login",
        json={"email": "clinician@test.com", "password": "ClinicianTest123!"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest_asyncio.fixture()
async def auth(client):
    return await _login(client)


@pytest_asyncio.fixture()
async def seeded_article():
    db = get_db()
    aid = "rc-art-" + uuid.uuid4().hex[:8]
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    await db[Collections.ARTICLES].insert_one({
        "id": aid, "title": "SGLT2i in HFpEF", "journal": "NEJM",
        "publication_year": 2024, "pmid": "39222333", "doi": "10.1056/rc",
        "sources_present": ["pubmed"], "full_text_status": "available",
        "study_design": "rct", "created_at": now, "updated_at": now,
    })
    yield aid
    await db[Collections.ARTICLES].delete_one({"id": aid})


async def test_collection_crud_and_records(client, auth, seeded_article):
    # create
    r = await client.post("/api/research-collections", json={"name": "My HFpEF"}, headers=auth)
    assert r.status_code == 201, r.text
    cid = r.json()["id"]

    # default collection auto-creates
    rd = await client.get("/api/research-collections/default", headers=auth)
    assert rd.status_code == 200 and rd.json()["name"] == "LitPortal Imports"

    # add a record
    ra = await client.post(
        f"/api/research-collections/{cid}/records",
        json={"records": [{"article_id": seeded_article, "notes": "great"}]},
        headers=auth,
    )
    assert ra.status_code == 201, ra.text
    assert ra.json()["added_count"] == 1

    # list records (hydrated columns + provenance)
    rl = await client.get(f"/api/research-collections/{cid}/records", headers=auth)
    assert rl.status_code == 200
    items = rl.json()
    assert len(items) == 1
    assert items[0]["article"]["journal"] == "NEJM"
    assert items[0]["source_provenance"] == ["pubmed"]
    assert items[0]["summary_eligible"] is True

    # dedup re-add
    ra2 = await client.post(
        f"/api/research-collections/{cid}/records",
        json={"records": [{"article_id": seeded_article}]}, headers=auth,
    )
    assert ra2.json()["skipped_count"] == 1

    # litarchive shows it
    rla = await client.get("/api/litarchive", headers=auth)
    assert rla.status_code == 200
    assert any(it["article_id"] == seeded_article for it in rla.json()["items"])
    rla_f = await client.get("/api/litarchive?source=pubmed&ai_summary_eligible=true", headers=auth)
    assert any(it["article_id"] == seeded_article for it in rla_f.json()["items"])

    # cleanup
    rdel = await client.delete(f"/api/research-collections/{cid}", headers=auth)
    assert rdel.status_code == 200


async def test_collection_requires_auth(client):
    r = await client.get("/api/research-collections")
    assert r.status_code in (401, 403)
