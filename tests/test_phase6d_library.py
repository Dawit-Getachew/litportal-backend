"""Phase 6D — User library + folder workspace integration tests.

Uses the existing async-client pattern in ``conftest.py`` (httpx
AsyncClient against the FastAPI app with an isolated test DB).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from core.db import Collections, get_db
from models.common import new_id, utcnow


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
async def _signup_and_login(client, password: str = "P6dTesting1234!"):
    email = f"clin_{uuid.uuid4().hex[:8]}@p6d-test.com"
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": password, "name": "Clin P6D"},
    )
    assert r.status_code in (200, 201), r.text
    token = r.json()["access_token"]
    user = r.json()["user"]
    return token, user


async def _admin_login(client):
    r = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    if r.status_code != 200:
        pytest.skip("Admin seed credentials not present.")
    return r.json()["access_token"]


async def _create_article(canonical_prefix: str = "phase6d_test") -> str:
    """Insert one test article doc; return its id."""
    db = get_db()
    art_id = new_id()
    pmid = uuid.uuid4().hex[:7]  # synthetic, never live
    await db[Collections.ARTICLES].insert_one({
        "id": art_id,
        "canonical_key": f"pmid:{canonical_prefix}_{pmid}",
        "title": f"Phase 6D sample paper {pmid}",
        "abstract": "Some abstract text.",
        "journal": "J Test Med",
        "publication_year": 2024,
        "authors_json": [{"fullName": "Smith J"}],
        "doi": None, "doi_normalized": None,
        "pmid": pmid, "pmcid": None,
        "source_ids_json": {}, "publication_types_json": [],
        "is_guideline": False, "is_preprint": False,
        "is_retracted": False, "has_expression_of_concern": False,
        "full_text_status": "abstract_only",
        "source_urls_json": [],
        "created_at": utcnow(), "updated_at": utcnow(),
    })
    return art_id


# ============================================================
# Tests A — PHI guard on every library text field
# ============================================================
@pytest.mark.asyncio
async def test_phi_in_folder_name_blocked(client):
    token, _ = await _signup_and_login(client)
    r = await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Patient MRN 12345678", "description": ""},
    )
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["detail"] == "library_text_phi_blocked"
    assert body["context_type"] == "folder_name"
    assert "mrn_pattern_detected" in body["reason_codes"]
    # NEVER echo input.
    assert "12345678" not in str(body)


@pytest.mark.asyncio
async def test_phi_in_folder_description_blocked(client):
    token, _ = await _signup_and_login(client)
    r = await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "OK folder",
            "description": "Discussed case with Mr. Smith — DOB 03/14/1979",
        },
    )
    assert r.status_code == 422
    body = r.json()
    assert body["context_type"] == "folder_description"
    assert "exact_date_detected" in body["reason_codes"]
    assert "Smith" not in str(body)
    assert "03/14/1979" not in str(body)


@pytest.mark.asyncio
async def test_clean_folder_name_passes(client):
    token, _ = await _signup_and_login(client)
    r = await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "SGLT2 inhibitors in HFpEF",
            "description": "Cohort of recent trials, no PHI.",
        },
    )
    assert r.status_code == 201, r.text


@pytest.mark.asyncio
async def test_phi_in_library_tag_blocked(client):
    token, _ = await _signup_and_login(client)
    art_id = await _create_article("tag_phi")
    r = await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {token}"},
        json={"article_id": art_id, "tags": ["okay", "patient@example.com"]},
    )
    assert r.status_code == 422, r.text
    assert r.json()["context_type"] == "library_tag"


@pytest.mark.asyncio
async def test_long_tag_blocked_invalid_shape(client):
    token, _ = await _signup_and_login(client)
    art_id = await _create_article("tag_long")
    r = await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "article_id": art_id,
            "tags": ["this-tag-is-way-too-long-and-has-too-many-characters-please"],
        },
    )
    assert r.status_code == 422
    assert "invalid_tag_shape" in r.json()["reason_codes"]


@pytest.mark.asyncio
async def test_phi_in_library_item_note_blocked(client):
    token, _ = await _signup_and_login(client)
    art_id = await _create_article("note_phi")
    r = await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {token}"},
        json={"article_id": art_id, "note": "Email jane@x.com about this paper"},
    )
    assert r.status_code == 422
    assert r.json()["context_type"] == "library_item_note"
    assert "email_detected" in r.json()["reason_codes"]
    assert "jane@x.com" not in str(r.json())


# ============================================================
# Tests B — User-scoping
# ============================================================
@pytest.mark.asyncio
async def test_admin_cannot_read_user_library_items(client):
    """Phase 6D — admin sees ONLY their own (empty) library."""
    ct, clin_user = await _signup_and_login(client)
    art_id = await _create_article("admin_scope")
    save = await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {ct}"},
        json={"article_id": art_id},
    )
    assert save.status_code == 201, save.text
    clin_item_id = save.json()["library_item"]["id"]
    admin_token = await _admin_login(client)
    r = await client.get(
        "/api/library/items",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200
    assert all(it["user_id"] != clin_user["id"] for it in r.json()["items"])
    r2 = await client.get(
        f"/api/library/items/{clin_item_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r2.status_code == 404


@pytest.mark.asyncio
async def test_another_clinician_cannot_read_first_clinicians_library(client):
    ct, _ = await _signup_and_login(client)
    art_id = await _create_article("other_clin")
    save = await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {ct}"},
        json={"article_id": art_id},
    )
    item_id = save.json()["library_item"]["id"]
    bt, _ = await _signup_and_login(client)
    r = await client.get(
        f"/api/library/items/{item_id}",
        headers={"Authorization": f"Bearer {bt}"},
    )
    assert r.status_code == 404


# ============================================================
# Tests D — Library metrics admin-only + excludes folder names/notes
# ============================================================
@pytest.mark.asyncio
async def test_clinician_cannot_call_library_metrics(client):
    token, _ = await _signup_and_login(client)
    r = await client.get(
        "/api/admin/library-metrics",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_library_metrics_excludes_folder_names_and_notes(client):
    ct, _ = await _signup_and_login(client)
    art_id = await _create_article("metrics_leak")
    FOLDER_NAME = "FixtureFolderForLeakCheck_zzz"
    NOTE = "FixtureNoteForLeakCheck_zzz"
    f = await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {ct}"},
        json={"name": FOLDER_NAME, "description": "x"},
    )
    assert f.status_code == 201, f.text
    s = await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {ct}"},
        json={"article_id": art_id, "note": NOTE},
    )
    assert s.status_code == 201, s.text
    admin_token = await _admin_login(client)
    r = await client.get(
        "/api/admin/library-metrics",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200
    body = r.json()
    blob = str(body)
    assert FOLDER_NAME not in blob
    assert NOTE not in blob
    assert isinstance(body["library_items_total"], int)
    assert isinstance(body["folders_total"], int)


# ============================================================
# Tests E — Folder + batch save
# ============================================================
@pytest.mark.asyncio
async def test_batch_save_creates_library_items_and_folder_articles(client):
    token, _ = await _signup_and_login(client)
    ids = [await _create_article(f"batch_{i}") for i in range(3)]
    folder = (await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Batch folder"},
    )).json()
    r = await client.post(
        "/api/library/items/batch",
        headers={"Authorization": f"Bearer {token}"},
        json={"article_ids": ids, "folder_id": folder["id"]},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["n_articles"] == 3
    assert body["n_library_created"] == 3
    assert body["n_added_to_folder"] == 3
    rows = (await client.get(
        f"/api/folders/{folder['id']}/articles",
        headers={"Authorization": f"Bearer {token}"},
    )).json()
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_batch_save_skips_excluded_articles_from_session(client):
    """Phase 6D.1 — articles whose ``search_session_articles`` row has
    ``exclusion_reason != None`` MUST be skipped by the batch endpoint
    (server-side guard). Retracted articles MUST also be skipped.
    """
    token, user = await _signup_and_login(client)
    db = get_db()
    # Three articles: clean, excluded-in-session, retracted.
    aid_clean = await _create_article("batch_clean")
    aid_excluded = await _create_article("batch_excluded")
    aid_retracted = await _create_article("batch_retracted")
    await db[Collections.ARTICLES].update_one(
        {"id": aid_retracted}, {"$set": {"is_retracted": True}},
    )
    # Create a session and a search_session_articles row with
    # exclusion_reason for aid_excluded.
    sid = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": user["id"],
        "status": "awaiting_selection", "deleted_at": None,
        "original_question": "q", "sanitized_question": "q",
        "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    await db[Collections.SEARCH_SESSION_ARTICLES].insert_many([
        {
            "id": uuid.uuid4().hex, "search_session_id": sid,
            "article_id": aid_clean, "exclusion_reason": None,
            "user_included": True, "included_by_default": True,
            "created_at": now, "updated_at": now,
        },
        {
            "id": uuid.uuid4().hex, "search_session_id": sid,
            "article_id": aid_excluded,
            "exclusion_reason": "non_article_record",
            "user_included": False, "included_by_default": False,
            "created_at": now, "updated_at": now,
        },
        {
            "id": uuid.uuid4().hex, "search_session_id": sid,
            "article_id": aid_retracted, "exclusion_reason": None,
            "user_included": True, "included_by_default": True,
            "created_at": now, "updated_at": now,
        },
    ])

    r = await client.post(
        "/api/library/items/batch",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "article_ids": [aid_clean, aid_excluded, aid_retracted],
            "saved_from_search_session_id": sid,
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["n_articles"] == 3
    assert body["n_articles_attempted"] == 1, body
    assert body["n_library_created"] == 1, body
    assert body["n_skipped"] == 2, body
    skipped_reasons = {
        s["article_id"]: s["reason"] for s in body["skipped"]
    }
    assert skipped_reasons.get(aid_excluded) == "session_excluded"
    assert skipped_reasons.get(aid_retracted) == "retracted"
    # The clean article landed in the library.
    items = (await client.get(
        "/api/library/items",
        headers={"Authorization": f"Bearer {token}"},
    )).json()["items"]
    library_article_ids = {it["article_id"] for it in items}
    assert aid_clean in library_article_ids
    assert aid_excluded not in library_article_ids
    assert aid_retracted not in library_article_ids


# ============================================================
# Tests G — unique (user_id, article_id) for ACTIVE items
# ============================================================
@pytest.mark.asyncio
async def test_resaving_same_article_is_idempotent(client):
    token, _ = await _signup_and_login(client)
    art_id = await _create_article("idem")
    r1 = await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {token}"},
        json={"article_id": art_id, "note": "first save"},
    )
    assert r1.status_code == 201
    assert r1.json()["created"] is True
    r2 = await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {token}"},
        json={"article_id": art_id, "note": "second save"},
    )
    assert r2.status_code == 201
    assert r2.json()["created"] is False
    assert r1.json()["library_item"]["id"] == r2.json()["library_item"]["id"]


# ============================================================
# Tests F — passive relevance event, ranker never reads it
# ============================================================
@pytest.mark.asyncio
async def test_save_writes_relevance_event_recorded(client):
    token, _ = await _signup_and_login(client)
    art_id = await _create_article("relevance")
    db = get_db()
    before = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "relevance_event_recorded"},
    )
    await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "article_id": art_id,
            "saved_from_search_session_id": "session_x",
            "saved_from_rank": 3,
            "saved_from_question_type": "treatment_benefit",
            "saved_from_specialty": "cardiology",
        },
    )
    after = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "relevance_event_recorded"},
    )
    assert after == before + 1


def test_ranker_does_not_read_library_event_types():
    """Phase 6D — Ranker MUST NOT read library/relevance events."""
    import services.evidence_ranker as er
    import services.picott_ranker as pr
    import services.search_planning as sp
    for mod in (er, pr, sp):
        src = open(mod.__file__).read()
        assert "relevance_event_recorded" not in src
        assert "library_item_saved" not in src
        assert "library_batch_saved" not in src
        assert "user_library_items" not in src


# ============================================================
# Test J — Task 10 stub returns 501
# ============================================================
@pytest.mark.asyncio
async def test_save_summary_returns_501(client):
    token, _ = await _signup_and_login(client)
    folder = (await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "stubtest"},
    )).json()
    r = await client.post(
        "/api/library/summaries",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "folder_id": folder["id"],
            "summary_id": "fake",
            "title": "trial title",
        },
    )
    assert r.status_code == 501
    assert r.json()["detail"] == "folder_summaries_coming_soon"


# ============================================================
# Test C — Session expiration does NOT delete library items
# ============================================================
@pytest.mark.asyncio
async def test_session_expiration_does_not_delete_user_library_items(client):
    token, user = await _signup_and_login(client)
    art_id = await _create_article("expire_test")
    db = get_db()
    sid = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": user["id"],
        "original_question": "test q",
        "sanitized_question": "test q",
        "status": "complete", "deleted_at": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now - timedelta(days=1),  # ALREADY expired
    })
    r = await client.post(
        "/api/library/items",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "article_id": art_id,
            "saved_from_search_session_id": sid,
            "saved_from_rank": 1,
        },
    )
    assert r.status_code == 201
    lib_id = r.json()["library_item"]["id"]
    folder = (await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Preserve-test folder"},
    )).json()
    await client.post(
        f"/api/folders/{folder['id']}/articles",
        headers={"Authorization": f"Bearer {token}"},
        json={"article_id": art_id},
    )
    # Run retention cleanup.
    from services.retention_cleanup import retention_cleanup
    cleanup_result = await retention_cleanup.cleanup_expired_sessions()
    assert cleanup_result.get("deleted_count", 0) >= 1
    # Library item still exists.
    item = await db[Collections.USER_LIBRARY_ITEMS].find_one(
        {"id": lib_id}, {"_id": 0},
    )
    assert item is not None
    # Folder still exists.
    f = await db[Collections.FOLDERS].find_one(
        {"id": folder["id"]}, {"_id": 0},
    )
    assert f is not None
    # Folder articles still exist.
    fa = await db[Collections.FOLDER_ARTICLES].count_documents(
        {"folder_id": folder["id"]},
    )
    assert fa >= 1


# ============================================================
# Test H — Saved evidence packet only when verified
# ============================================================
@pytest.mark.asyncio
async def test_packet_requires_verified_session(client):
    token, user = await _signup_and_login(client)
    db = get_db()
    folder = (await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Packet folder"},
    )).json()
    sid = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": sid, "user_id": user["id"],
        "status": "searching",
        "original_question": "q", "sanitized_question": "q",
        "deleted_at": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    r = await client.post(
        "/api/library/evidence-packets",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "folder_id": folder["id"],
            "source_search_session_id": sid,
            "title": "Should fail - unverified",
        },
    )
    assert r.status_code == 409
    assert r.json()["detail"] == "source_session_not_verified"
    await db[Collections.SEARCH_SESSIONS].update_one(
        {"id": sid}, {"$set": {"status": "complete"}},
    )
    r2 = await client.post(
        "/api/library/evidence-packets",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "folder_id": folder["id"],
            "source_search_session_id": sid,
            "title": "Now verified",
        },
    )
    assert r2.status_code == 201, r2.text
    assert r2.json()["title"] == "Now verified"


@pytest.mark.asyncio
async def test_packet_title_phi_blocked(client):
    token, _ = await _signup_and_login(client)
    folder = (await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "phi test folder"},
    )).json()
    r = await client.post(
        "/api/library/evidence-packets",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "folder_id": folder["id"],
            "source_search_session_id": uuid.uuid4().hex,
            "title": "Patient SSN 123-45-6789",
        },
    )
    assert r.status_code == 422
    assert r.json()["context_type"] == "evidence_packet_title"
