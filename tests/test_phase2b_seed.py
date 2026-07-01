"""Phase 2B — startup seed for source_configs."""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.db import Collections, get_db  # noqa: E402
from server import _seed_source_configs  # noqa: E402


_DEFAULT_ON = {
    "pubmed", "europe_pmc", "openalex", "pubmed_guidelines",
    "pmc_oa", "europe_pmc_oa", "semantic_scholar_oa",
    "semantic_scholar_institutional", "openathens", "openurl_resolver",
}
_GUIDELINE_OFF = {"uspstf", "nice", "who", "cdc", "specialty_society"}
_ALL_SEEDED = {
    "pubmed", "europe_pmc", "openalex", "pubmed_guidelines",
    "uspstf", "nice", "who", "cdc", "specialty_society",
    "pmc_oa", "europe_pmc_oa", "unpaywall", "semantic_scholar_oa",
    "semantic_scholar_institutional", "crossref", "openathens",
    "openurl_resolver", "libkey_getftr",
    # Phase 4A — OA full-text fetcher
    "europe_pmc_oa_fetcher", "pmc_oa_fetcher", "oa_license_gate",
}


async def _wipe(coll):
    await coll.delete_many({})


@pytest.mark.asyncio
async def test_seed_fresh_db_enables_phase2b_defaults():
    db = get_db()
    await _wipe(db[Collections.SOURCE_CONFIGS])
    await _seed_source_configs()
    rows = await db[Collections.SOURCE_CONFIGS].find({}, {"_id": 0}).to_list(length=50)
    by_name = {r["source_name"]: r for r in rows}
    # All 17 sources present.
    assert set(by_name.keys()) == _ALL_SEEDED
    # Phase 2B defaults are ON.
    for name in _DEFAULT_ON:
        assert by_name[name]["enabled"] is True, f"{name} should be enabled by default"
    # Guideline scaffolds remain OFF.
    for name in _GUIDELINE_OFF:
        assert by_name[name]["enabled"] is False, f"{name} should be disabled"
    # specialty_society is present + requires API key.
    spec = by_name["specialty_society"]
    assert spec["requires_api_key"] is True
    assert "scaffolded" in spec["status_message"].lower()
    # PubMed status reflects key state.
    msg = by_name["pubmed"]["status_message"]
    assert ("API key" in msg) and ("req/s" in msg)
    # OpenAlex marked enrichment-only.
    assert "Enrichment-only" in by_name["openalex"]["status_message"]


@pytest.mark.asyncio
async def test_seed_is_idempotent_and_migrates_legacy_phase1_rows():
    db = get_db()
    coll = db[Collections.SOURCE_CONFIGS]
    await _wipe(coll)
    now = datetime.now(timezone.utc)
    # Pre-populate a legacy-style row for pubmed (Phase 1 marker, enabled=false).
    await coll.insert_one({
        "id": "legacy-pubmed",
        "source_name": "pubmed",
        "enabled": False,
        "requires_api_key": True,
        "status_message": "seeded; integration disabled in Phase 1",
        "last_success_at": None,
        "last_error_at": None,
        "last_error_message": None,
        "created_at": now,
        "updated_at": now,
    })
    # Pre-populate an admin-deliberately-disabled row for openalex
    # (status_message lacks the legacy marker; seeder must leave it alone).
    await coll.insert_one({
        "id": "admin-openalex",
        "source_name": "openalex",
        "enabled": False,
        "requires_api_key": False,
        "status_message": "Admin disabled — review pending.",
        "last_success_at": None,
        "last_error_at": None,
        "last_error_message": None,
        "created_at": now,
        "updated_at": now,
    })

    await _seed_source_configs()

    pubmed = await coll.find_one({"source_name": "pubmed"})
    assert pubmed["enabled"] is True  # Legacy row migrated to enabled
    assert "Phase 1" not in pubmed["status_message"]

    openalex = await coll.find_one({"source_name": "openalex"})
    assert openalex["enabled"] is False  # Admin's deliberate disable preserved
    assert openalex["status_message"] == "Admin disabled — review pending."


@pytest.mark.asyncio
async def test_seed_never_flips_enabled_true_to_false():
    """If a row is already enabled=true, the seeder must never disable it,
    regardless of whether its status_message matches the legacy marker."""
    db = get_db()
    coll = db[Collections.SOURCE_CONFIGS]
    await _wipe(coll)
    now = datetime.now(timezone.utc)
    # An admin enabled a guideline portal manually — must stay on.
    await coll.insert_one({
        "id": "admin-uspstf",
        "source_name": "uspstf",
        "enabled": True,
        "requires_api_key": False,
        "status_message": "Admin enabled — using internal proxy.",
        "last_success_at": None,
        "last_error_at": None,
        "last_error_message": None,
        "created_at": now,
        "updated_at": now,
    })
    await _seed_source_configs()
    uspstf = await coll.find_one({"source_name": "uspstf"})
    assert uspstf["enabled"] is True
    assert uspstf["status_message"] == "Admin enabled — using internal proxy."


@pytest.mark.asyncio
async def test_seed_pubmed_status_reflects_api_key(monkeypatch):
    db = get_db()
    await _wipe(db[Collections.SOURCE_CONFIGS])
    monkeypatch.setenv("NCBI_API_KEY", "fake-key-for-test")
    await _seed_source_configs()
    pubmed = await db[Collections.SOURCE_CONFIGS].find_one({"source_name": "pubmed"})
    assert "Using NCBI API key" in pubmed["status_message"]

    await _wipe(db[Collections.SOURCE_CONFIGS])
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("PUBMED_API_KEY", raising=False)
    await _seed_source_configs()
    pubmed = await db[Collections.SOURCE_CONFIGS].find_one({"source_name": "pubmed"})
    assert "No NCBI API key" in pubmed["status_message"]


@pytest.mark.asyncio
async def test_specialty_society_connector_returns_not_configured():
    from services.connectors.guidelines import specialty_society_connector
    out = await specialty_society_connector.search({"normalized_question": "test"})
    assert out["status"] == "not_configured"
    assert out["source"] == "specialty_society"
    assert out["results"] == []
