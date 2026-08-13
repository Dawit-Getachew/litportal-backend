"""fetch_for_articles must isolate a per-article crash (register §1a A2 prereq):
fetch_for_article is documented never-raises, but if that contract breaks, one
bad article must surface as status='failed' — not void the whole batch."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from services.oa_fetcher.service import OAFetcherService


def _ok(article_id):
    return {
        "article_id": article_id, "status": "fetched",
        "source": "europepmc", "license": "cc-by", "license_category": "open",
        "can_ingest_for_summary": True, "sections_created": 2,
        "token_count": 100, "evidence_text_level": "full_text",
        "error_code": None,
    }


@pytest.mark.asyncio
async def test_one_crashing_article_does_not_void_the_batch():
    svc = OAFetcherService()

    async def fake_fetch(*, article_id, **kwargs):
        if article_id == "art-bad":
            raise RuntimeError("contract breach")
        return _ok(article_id)

    with patch.object(OAFetcherService, "fetch_for_article", side_effect=fake_fetch):
        out = await svc.fetch_for_articles(
            article_ids=["art-1", "art-bad", "art-2"],
            session_id="ses-1", user_id="usr-1",
        )

    by_id = {r["article_id"]: r for r in out["results"]}
    assert by_id["art-1"]["status"] == "fetched"
    assert by_id["art-2"]["status"] == "fetched"
    assert by_id["art-bad"]["status"] == "failed"
    assert by_id["art-bad"]["error_code"] == "unexpected_exception"
    assert out["summary"]["failed"] == 1
    assert out["summary"]["fetched"] == 2


@pytest.mark.asyncio
async def test_results_keep_original_order():
    svc = OAFetcherService()

    async def fake_fetch(*, article_id, **kwargs):
        return _ok(article_id)

    with patch.object(OAFetcherService, "fetch_for_article", side_effect=fake_fetch):
        out = await svc.fetch_for_articles(
            article_ids=["a", "b", "c"], session_id="s", user_id="u",
        )
    assert [r["article_id"] for r in out["results"]] == ["a", "b", "c"]
