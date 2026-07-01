"""Phase 4A — OA fetch audit-payload safety (Task 11 / H)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from core.db import Collections, get_db
from services.oa_fetcher import oa_fetcher
from tests.fixtures_oa_xml import JATS_SAMPLE


class _MockResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


def _client_returning(*responses):
    cli = AsyncMock(spec=httpx.AsyncClient)
    cli.get = AsyncMock(side_effect=list(responses))
    return cli


async def _seed():
    db = get_db()
    sid = uuid.uuid4().hex
    aid = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": "u-audit", "status": "awaiting_selection",
        "original_question": "q", "sanitized_question": "q", "pico_json": {},
        "deleted_at": None, "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    await db[Collections.ARTICLES].insert_one({
        "id": aid, "canonical_key": f"id:{aid}", "title": "Audit test article",
        "abstract": "abs", "doi": None, "pmid": "12345",
        "pmcid": "PMC1234567", "journal": "J", "publication_year": 2024,
        "article_type_classification": "rct",
        "authors_json": [], "source_ids_json": {"inEPMC": "Y"},
        "source_urls_json": [],
        "is_preprint": False, "is_retracted": False, "is_retraction_notice": False,
        "has_expression_of_concern": False, "publication_types_json": [],
        "full_text_status": "unknown", "created_at": now, "updated_at": now,
    })
    return sid, aid


async def _oa_events(sid):
    db = get_db()
    return await db[Collections.AUDIT_EVENTS].find(
        {"search_session_id": sid, "event_type": {"$regex": r"^oa_fetch_"}},
        {"_id": 0},
    ).to_list(length=100)


def _dumps(events):
    return json.dumps(events, default=str)


@pytest.mark.asyncio
async def test_oa_fetch_audit_payload_no_raw_text():
    sid, aid = await _seed()
    cli = _client_returning(_MockResp(200, JATS_SAMPLE.encode("utf-8")))
    await oa_fetcher.fetch_for_article(
        article_id=aid, session_id=sid, user_id="u-audit", client=cli,
    )
    events = await _oa_events(sid)
    serialized = _dumps(events).lower()
    # Article body phrases must NEVER appear in audit payloads.
    for marker in (
        "apixaban reduced", "atrial fibrillation is the most common",
        "randomized, double-blind", "stroke or systemic embolism",
        "creativecommons.org",
    ):
        assert marker not in serialized, f"audit payload leaked: {marker}"


@pytest.mark.asyncio
async def test_oa_fetch_audit_payload_only_closed_reason_codes():
    """Every emitted audit event must carry a known event_type."""
    sid, aid = await _seed()
    cli = _client_returning(_MockResp(200, JATS_SAMPLE.encode("utf-8")))
    await oa_fetcher.fetch_for_article(
        article_id=aid, session_id=sid, user_id="u-audit", client=cli,
    )
    events = await _oa_events(sid)
    assert events, "no oa_fetch_* events emitted"
    allowed = {
        "oa_fetch_attempt", "oa_fetch_success", "oa_fetch_failed",
        "oa_fetch_cache_hit", "oa_fetch_skipped_ineligible",
        "oa_fetch_skipped_license",
    }
    for e in events:
        assert e["event_type"] in allowed, f"unexpected event_type: {e['event_type']}"
        payload = e.get("event_payload_json") or {}
        # Whatever payload keys we use must be from the closed set documented
        # in /app/memory/phase4a_notes.md.
        allowed_keys = {
            "article_id", "source", "license_category",
            "sections_count", "token_count", "error_code",
            "latency_ms",
        }
        unexpected = set(payload.keys()) - allowed_keys
        assert not unexpected, f"unexpected payload keys: {unexpected}"


@pytest.mark.asyncio
async def test_oa_fetch_audit_does_not_leak_secrets():
    sid, aid = await _seed()
    cli = _client_returning(_MockResp(200, JATS_SAMPLE.encode("utf-8")))
    await oa_fetcher.fetch_for_article(
        article_id=aid, session_id=sid, user_id="u-audit", client=cli,
    )
    events = await _oa_events(sid)
    serialized = _dumps(events).lower()
    for marker in ("sk-emergent-", "sk-proj-", "bearer ", "api_key", "openai_api_key"):
        assert marker not in serialized, f"audit payload leaked secret marker: {marker}"
