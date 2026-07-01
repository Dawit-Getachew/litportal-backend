"""Phase K5.2A — official guideline connector (MyHealthfinder, no-key) tests.

Validates the deterministic no-key official-source connector + registry +
endpoint. NO live network in tests (HTTP is monkeypatched / pure parsing).
"""
from __future__ import annotations

import uuid

import pytest

from core.db import Collections, get_db
from models.common import utcnow
from services.connectors.myhealthfinder import (
    MYHF_PROVENANCE,
    MYHF_SOURCE_ID,
    MyHealthfinderConnector,
    build_keyword,
    myhealthfinder_connector,
)


# =====================================================================
# 1. Registry — myhealthfinder_hhs is the one no-key LIVE source;
#    ai_ingestion_allowed=false; keyed/uncertain sources stay scaffolded.
# =====================================================================
@pytest.mark.asyncio
async def test_K5_2A_registry_myhealthfinder_live_no_key(client):  # noqa: ARG001
    db = get_db()
    row = await db[Collections.GUIDELINE_SOURCES].find_one(
        {"source_name": "myhealthfinder_hhs"}, {"_id": 0},
    )
    assert row is not None
    assert row["connector_status"] == "live"
    assert row["retrieval_allowed"] is True
    assert row["requires_api_key"] is False
    assert row["ai_ingestion_allowed"] is False  # hard invariant


@pytest.mark.asyncio
async def test_K5_2A_keyed_sources_remain_scaffolded(client):  # noqa: ARG001
    db = get_db()
    for name in ("uspstf", "nice", "who", "specialty_society"):
        row = await db[Collections.GUIDELINE_SOURCES].find_one(
            {"source_name": name}, {"_id": 0},
        )
        assert row is not None and row["connector_status"] == "scaffolded", name
        assert row["retrieval_allowed"] is False, name
        assert row["ai_ingestion_allowed"] is False, name


# =====================================================================
# 2. Connector — pure parsing + closed-set status + metadata/link-out only.
# =====================================================================
_FAKE_PAYLOAD = {
    "Result": {"Resources": {"Resource": [
        {
            "Title": "Get Screened for Colorectal Cancer",
            "AccessibleVersion": "https://odphp.health.gov/myhealthfinder/x",
            "MyHFDescription": "<p>Talk to your doctor about screening.</p>",
            "Categories": "Colorectal Cancer, Screening Tests",
        },
        {"Title": "", "AccessibleVersion": "https://x"},  # dropped (no title)
        {"Title": "No URL", "AccessibleVersion": ""},      # dropped (no url)
    ]}},
}


def test_K5_2A_connector_parse_metadata_link_out_only():
    items = MyHealthfinderConnector._parse(_FAKE_PAYLOAD, limit=8)
    assert len(items) == 1
    it = items[0]
    assert it["source"] == MYHF_SOURCE_ID
    assert it["provenance"] == MYHF_PROVENANCE
    assert it["url"].startswith("https://")
    assert it["full_text_available"] is False        # invariant
    assert it["ai_ingestion_eligible"] is False       # invariant
    assert "<p>" not in it["snippet"]                 # HTML stripped


@pytest.mark.asyncio
async def test_K5_2A_connector_no_keyword_status():
    r = await myhealthfinder_connector.search("")
    assert r["status"] == "no_keyword"
    assert r["results"] == []
    assert r["ai_ingestion"] is False


@pytest.mark.asyncio
async def test_K5_2A_connector_ok_and_empty(monkeypatch):
    class _Resp:
        status_code = 200
        def __init__(self, payload):
            self._p = payload
        def json(self):
            return self._p

    class _Client:
        def __init__(self, payload):
            self._p = payload
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, *a, **k):
            # No API key / no Authorization header ever sent.
            assert "headers" in k
            assert "Authorization" not in (k.get("headers") or {})
            assert "key" not in (k.get("params") or {}) and "api_key" not in (k.get("params") or {})
            return _Resp(self._p)

    import services.connectors.myhealthfinder as mod
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda *a, **k: _Client(_FAKE_PAYLOAD))
    r = await myhealthfinder_connector.search("colorectal cancer")
    assert r["status"] == "ok" and len(r["results"]) == 1

    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda *a, **k: _Client({"Result": {}}))
    r2 = await myhealthfinder_connector.search("nothing")
    assert r2["status"] == "empty" and r2["results"] == []


def test_K5_2A_build_keyword_prefers_anchor():
    assert build_keyword(primary_anchor="colorectal cancer", question="x") == "colorectal cancer"
    kw = build_keyword(primary_anchor=None, question="What is the prognosis of IPF?")
    assert isinstance(kw, str) and kw  # non-empty deterministic fallback


# =====================================================================
# 3. Endpoint — owner-scoped, PHI-blocked skip, provenance label.
# =====================================================================
async def _seed(uid, *, status="awaiting_selection", anchor="colorectal cancer"):
    db = get_db()
    sid = f"k52a-{uuid.uuid4().hex[:8]}"
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": uid,
        "original_question": None if status == "blocked_phi" else "q",
        "sanitized_question": None if status == "blocked_phi" else "screening question",
        "status": status,
        "primary_clinical_anchors_json": {"primary_anchors": [{"canonical_concept": anchor}]},
        "clinical_question_map_json": {},
        "created_at": utcnow(), "updated_at": utcnow(),
        "expires_at": utcnow(), "deleted_at": None,
    })
    return sid


@pytest.mark.asyncio
async def test_K5_2A_endpoint_ok_and_owner_scoped(client, monkeypatch):
    # Canned connector result (no network).
    async def _fake_search(self, keyword, **k):  # noqa: ANN001
        return {
            "source": MYHF_SOURCE_ID, "organization": "HHS / ODPHP (MyHealthfinder)",
            "provenance": MYHF_PROVENANCE, "status": "ok", "keyword": keyword,
            "ai_ingestion": False,
            "results": [{
                "title": "Get Screened", "url": "https://odphp.health.gov/x",
                "snippet": "talk to your doctor", "categories": "Colorectal Cancer",
                "source": MYHF_SOURCE_ID, "organization": "HHS / ODPHP (MyHealthfinder)",
                "provenance": MYHF_PROVENANCE, "ai_ingestion_eligible": False,
                "full_text_available": False,
            }],
        }
    monkeypatch.setattr(MyHealthfinderConnector, "search", _fake_search)

    e1 = f"k52a-{uuid.uuid4().hex[:8]}@k52a.com"
    e2 = f"k52a-{uuid.uuid4().hex[:8]}@k52a.com"
    for e in (e1, e2):
        await client.post("/api/auth/signup", json={"email": e, "password": "K52aSuperTest!23", "name": "K"})
    r1 = await client.post("/api/auth/login", json={"email": e1, "password": "K52aSuperTest!23"})
    uid1 = r1.json()["user"]["id"]
    tok1 = r1.json()["access_token"]
    sid = await _seed(uid1)

    resp = await client.get(
        f"/api/search-sessions/{sid}/official-guideline-sources",
        headers={"Authorization": f"Bearer {tok1}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["lane_kind"] == "official_source_metadata_link_out"
    assert body["ai_ingestion_allowed"] is False
    assert body["status"] == "ok" and len(body["results"]) == 1
    assert body["results"][0]["full_text_available"] is False

    # Non-owner rejected.
    r2 = await client.post("/api/auth/login", json={"email": e2, "password": "K52aSuperTest!23"})
    resp2 = await client.get(
        f"/api/search-sessions/{sid}/official-guideline-sources",
        headers={"Authorization": f"Bearer {r2.json()['access_token']}"},
    )
    assert resp2.status_code in (403, 404)


@pytest.mark.asyncio
async def test_K5_2A_endpoint_phi_blocked_skips_connector(client, monkeypatch):
    # If the connector were called it would raise — proves it is NOT called.
    async def _boom(self, keyword, **k):  # noqa: ANN001
        raise AssertionError("connector must not be called for PHI-blocked session")
    monkeypatch.setattr(MyHealthfinderConnector, "search", _boom)

    e = f"k52a-{uuid.uuid4().hex[:8]}@k52a.com"
    await client.post("/api/auth/signup", json={"email": e, "password": "K52aSuperTest!23", "name": "K"})
    r = await client.post("/api/auth/login", json={"email": e, "password": "K52aSuperTest!23"})
    uid = r.json()["user"]["id"]
    tok = r.json()["access_token"]
    sid = await _seed(uid, status="blocked_phi")
    resp = await client.get(
        f"/api/search-sessions/{sid}/official-guideline-sources",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "skipped_phi_blocked"
    assert resp.json()["results"] == []
