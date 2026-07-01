"""Phase 3 — evidence packet validation (Task 11 / A)."""
from __future__ import annotations

import pytest

from tests._phase3_helpers import (  # noqa: E402
    make_packet_session, signup, good_apixaban_articles,
)


@pytest.mark.asyncio
async def test_zero_selected_articles_rejected(client):
    token, _ = await signup(client, "pkt0@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = good_apixaban_articles()
    # Mark every article as not user_included so the packet has zero selected.
    for a in arts:
        a["user_included"] = False
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    assert r.status_code == 200
    p = r.json()
    assert p["selected_articles_count"] == 0
    assert p["ready_for_summary"] is False
    assert p["block_reason"] in ("no_selected_articles", "no_usable_evidence")


@pytest.mark.asyncio
async def test_selected_article_with_abstract_extract_allowed(client):
    token, _ = await signup(client, "pkt1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, _, _ = await make_packet_session(client, h, articles=good_apixaban_articles())
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p = r.json()
    assert p["usable_articles_count"] >= 1
    assert p["abstract_only_count"] >= 1
    assert p["ready_for_summary"] is True
    assert p["block_reason"] in (None, "")


@pytest.mark.asyncio
async def test_selected_article_with_user_uploaded_full_text_allowed(client):
    token, _ = await signup(client, "pkt2@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, _, _ = await make_packet_session(
        client, h, articles=good_apixaban_articles(),
        upload_extract_for=[0],
    )
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p = r.json()
    assert p["user_uploaded_full_text_count"] >= 1
    assert p["ready_for_summary"] is True


@pytest.mark.asyncio
async def test_excluded_article_not_in_packet_even_if_user_included(client):
    token, _ = await signup(client, "pkt3@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = good_apixaban_articles()
    arts[0]["exclusion_reason"] = "preprint"
    arts[0]["is_preprint"] = True
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p = r.json()
    # Excluded article should drop out of the selected set.
    assert p["selected_articles_count"] == 2
    assert p["usable_articles_count"] == 2


@pytest.mark.asyncio
async def test_unselected_article_not_in_packet(client):
    token, _ = await signup(client, "pkt4@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = good_apixaban_articles()
    arts[0]["user_included"] = False
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p = r.json()
    assert p["selected_articles_count"] == 2


@pytest.mark.asyncio
async def test_all_metadata_only_blocks_with_clear_message(client):
    token, _ = await signup(client, "pkt5@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = [
        {**a, "abstract": None} for a in good_apixaban_articles()
    ]
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p = r.json()
    assert p["usable_articles_count"] == 0
    assert p["ready_for_summary"] is False
    assert p["block_reason"] in ("no_usable_evidence", "no_selected_articles")


@pytest.mark.asyncio
async def test_abstract_only_count_correct(client):
    token, _ = await signup(client, "pkt6@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, _, _ = await make_packet_session(client, h, articles=good_apixaban_articles())
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p = r.json()
    assert p["abstract_only_count"] == 3
    assert p["oa_full_text_count"] == 0
    assert p["user_uploaded_full_text_count"] == 0


@pytest.mark.asyncio
async def test_user_uploaded_count_correct(client):
    token, _ = await signup(client, "pkt7@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, _, _ = await make_packet_session(
        client, h, articles=good_apixaban_articles(),
        upload_extract_for=[0, 2],
    )
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p = r.json()
    assert p["user_uploaded_full_text_count"] == 2


@pytest.mark.asyncio
async def test_ready_for_summary_true_when_at_least_one_usable(client):
    token, _ = await signup(client, "pkt8@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, _, _ = await make_packet_session(client, h, articles=good_apixaban_articles())
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    assert r.json()["ready_for_summary"] is True


@pytest.mark.asyncio
async def test_block_reason_no_usable_evidence(client):
    token, _ = await signup(client, "pkt9@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = [
        {**a, "abstract": None} for a in good_apixaban_articles()
    ]
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p = r.json()
    assert p["block_reason"] in ("no_usable_evidence", "no_selected_articles")


# ---------------------------------------------------------------------------
# Strict-user_included semantics (Phase 3 follow-up from tester L1).
# `user_included` is THE source of truth for the summarizer; an article where
# included_by_default=true AND user_included=false (user explicitly opted out)
# MUST NOT appear in the packet.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_evidence_packet_uses_user_included_strictly(client):
    """user_included True on 2 articles, False on 3 → packet count == 2."""
    token, _ = await signup(client, "strict1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = [
        {**good_apixaban_articles()[0], "title": f"Study {i}",
         "pmid": f"100{i}", "doi": f"10.1234/x{i}",
         "abstract": f"abstract {i}", "user_included": (i < 2),
         "included_by_default": True}
        for i in range(5)
    ]
    sid, _, article_ids = await make_packet_session(client, h, articles=arts)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    assert r.status_code == 200
    p = r.json()
    assert p["selected_articles_count"] == 2, p
    assert p["ready_for_summary"] is True
    assert p["block_reason"] in (None, "")


@pytest.mark.asyncio
async def test_prepare_evidence_packet_zero_when_all_user_included_false(client):
    """Every user_included=False (even though included_by_default=True) →
    packet count 0, ready_for_summary=False, block_reason=no_selected_articles.
    """
    token, _ = await signup(client, "strict2@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = [
        {**good_apixaban_articles()[0], "title": f"Study {i}",
         "pmid": f"200{i}", "doi": f"10.1234/y{i}",
         "abstract": f"abstract {i}", "user_included": False,
         "included_by_default": True}
        for i in range(3)
    ]
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p = r.json()
    assert p["selected_articles_count"] == 0
    assert p["ready_for_summary"] is False
    assert p["block_reason"] == "no_selected_articles"


@pytest.mark.asyncio
async def test_packet_updates_after_patch_user_included(client):
    """PATCH user_included=False on an article → packet count drops by one."""
    token, _ = await signup(client, "strict3@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = good_apixaban_articles()
    sid, _, article_ids = await make_packet_session(client, h, articles=arts)
    # All 3 selected initially.
    r = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    assert r.json()["selected_articles_count"] == 3
    # Toggle first article off via the session-articles PATCH.
    pr = await client.patch(
        f"/api/search-sessions/{sid}/articles/{article_ids[0]}",
        headers=h, json={"user_included": False},
    )
    assert pr.status_code == 200, pr.text
    r2 = await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
    p2 = r2.json()
    assert p2["selected_articles_count"] == 2
    assert p2["ready_for_summary"] is True
