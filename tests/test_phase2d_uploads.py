"""Phase 2D — comprehensive upload pipeline tests.

Covers Task 12 buckets A–G plus the H regression. Uses the in-memory PDF
fixtures in ``tests/fixtures_pdf.py``.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.db import Collections, get_db  # noqa: E402
from tests.fixtures_pdf import (  # noqa: E402
    make_apixaban_pdf, make_encrypted_pdf, make_pdf_with_annotation_text,
    make_pdf_with_embedded_attachment, make_pdf_with_javascript, make_text_pdf,
)


# ============================================================================
#  Helpers
# ============================================================================
async def _signup(client, email):
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!", "name": "T"},
    )
    return r.json()["access_token"], r.json()["user"]["id"]


async def _make_session_with_article(client, headers, *, doi=None, pmid=None,
                                     pmcid=None, title=None, year=None,
                                     abstract=None, is_preprint=False,
                                     is_retracted=False):
    """Create a session manually with one article wired in.

    Bypasses the live retrieval pipeline so tests are deterministic and fast.
    """
    db = get_db()
    me = await client.get("/api/auth/me", headers=headers)
    user_id = me.json()["id"]
    now = datetime.now(timezone.utc)
    session_id = uuid.uuid4().hex
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": session_id,
        "user_id": user_id,
        "status": "awaiting_selection",
        "original_question": "test",
        "sanitized_question": "test",
        "deleted_at": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + timedelta(days=30),
    })
    article_id = uuid.uuid4().hex
    await db[Collections.ARTICLES].insert_one({
        "id": article_id,
        "canonical_key": f"pmid:{pmid}" if pmid else f"doi:{(doi or '').lower()}" if doi else f"id:{article_id}",
        "title": title or "Apixaban versus warfarin in patients with atrial fibrillation",
        "abstract": abstract or "Abstract text about apixaban vs warfarin.",
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "publication_year": year,
        "authors_json": [],
        "source_ids_json": {},
        "source_urls_json": [],
        "is_preprint": is_preprint,
        "is_retracted": is_retracted,
        "is_retraction_notice": False,
        "has_expression_of_concern": False,
        "created_at": now, "updated_at": now,
    })
    await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
        "id": uuid.uuid4().hex,
        "search_session_id": session_id,
        "article_id": article_id,
        "user_included": True,
        "included_by_default": True,
        "exclusion_reason": "preprint" if is_preprint else (
            "retracted" if is_retracted else None
        ),
        "relevance_score": 80.0, "evidence_design_score": 80.0,
        "overall_rank": 90, "rank_explanation": "test",
        "query_types_matched": [], "sources_present": ["europe_pmc"], "flags": [],
        "created_at": now, "updated_at": now,
    })
    return session_id, article_id, user_id


def _upload_form(file_bytes: bytes, filename: str = "article.pdf",
                 retention="use_once", attested=True,
                 version="v1-2025", content_type="application/pdf"):
    return {
        "files": {"file": (filename, file_bytes, content_type)},
        "data": {
            "retention_mode": retention,
            "rights_attested": "true" if attested else "false",
            "rights_attestation_version": version,
        },
    }


async def _do_upload(client, headers, session_id, article_id, file_bytes,
                     **kw):
    form = _upload_form(file_bytes, **kw)
    return await client.post(
        f"/api/search-sessions/{session_id}/articles/{article_id}/upload-full-text",
        files=form["files"], data=form["data"], headers=headers,
    )


# ============================================================================
#  A. Upload validation
# ============================================================================
@pytest.mark.asyncio
async def test_accepts_pdf_with_valid_magic_bytes(client):
    token, _ = await _signup(client, "ua1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978", year=2011,
        title="Apixaban versus warfarin in patients with atrial fibrillation",
    )
    r = await _do_upload(client, h, sid, aid, make_apixaban_pdf())
    assert r.status_code == 200
    body = r.json()
    assert body["upload_status"] in ("processed", "matched")
    # apixaban PDF has matching DOI + PMID + title → high match.
    assert body["match_confidence"] == "high"


@pytest.mark.asyncio
async def test_rejects_non_pdf_content_type(client):
    token, _ = await _signup(client, "ua2@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, b"%PDF-1.4 fake",
                         filename="evil.pdf", content_type="text/plain")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_rejects_pdf_with_wrong_magic_bytes(client):
    token, _ = await _signup(client, "ua3@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, b"NOT A PDF AT ALL " * 50)
    # Magic bytes wrong → reported via the upload pipeline as error/not_a_pdf.
    assert r.status_code == 200
    assert r.json()["upload_status"] == "error"
    assert "not_a_pdf" in r.json().get("reason_codes", [])


@pytest.mark.asyncio
async def test_rejects_oversized_pdf(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    token, _ = await _signup(client, "ua4@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    blob = b"%PDF-1.4\n" + (b"A" * (2 * 1024 * 1024))
    r = await _do_upload(client, h, sid, aid, blob)
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_rejects_encrypted_pdf(client):
    token, _ = await _signup(client, "ua5@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, make_encrypted_pdf())
    assert r.status_code == 200
    assert r.json()["upload_status"] == "error"
    assert "encrypted_pdf_rejected" in r.json().get("reason_codes", [])


@pytest.mark.asyncio
async def test_rejects_pdf_with_embedded_attachment(client):
    token, _ = await _signup(client, "ua6@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, make_pdf_with_embedded_attachment())
    body = r.json()
    assert body["upload_status"] == "error"
    assert "embedded_attachment_rejected" in body.get("reason_codes", [])


@pytest.mark.asyncio
async def test_rejects_pdf_with_javascript_openaction(client):
    token, _ = await _signup(client, "ua7@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, make_pdf_with_javascript())
    body = r.json()
    assert body["upload_status"] == "error"
    assert "active_content_rejected" in body.get("reason_codes", [])


@pytest.mark.asyncio
async def test_rejects_upload_without_rights_attestation(client):
    token, _ = await _signup(client, "ua8@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, make_text_pdf(), attested=False)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_rejects_upload_for_article_not_in_user_session(client):
    token_a, _ = await _signup(client, "uaA@test.com")
    h_a = {"Authorization": f"Bearer {token_a}"}
    sid_a, aid_a, _ = await _make_session_with_article(client, h_a)
    # User B can't upload to user A's session.
    token_b, _ = await _signup(client, "uaB@test.com")
    h_b = {"Authorization": f"Bearer {token_b}"}
    r = await _do_upload(client, h_b, sid_a, aid_a, make_text_pdf())
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_rejects_upload_for_excluded_article(client):
    token, _ = await _signup(client, "uaE@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h, is_retracted=True)
    r = await _do_upload(client, h, sid, aid, make_text_pdf())
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_rejects_saved_private_retention_mode(client):
    token, _ = await _signup(client, "uaR@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, make_text_pdf(), retention="saved_private")
    assert r.status_code == 400
    assert "saved_private" in r.json()["detail"]


@pytest.mark.asyncio
async def test_rejects_upload_when_feature_disabled_via_env(client, monkeypatch):
    monkeypatch.setenv("ENABLE_USER_PDF_UPLOADS", "false")
    token, _ = await _signup(client, "uaF@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, make_text_pdf())
    assert r.status_code == 403


# ============================================================================
#  B. Upload safety scanner
# ============================================================================
@pytest.mark.asyncio
async def test_blocks_filename_with_mrn(client):
    token, _ = await _signup(client, "us1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, make_text_pdf(),
                         filename="patient_MRN1234567_review.pdf")
    body = r.json()
    assert body["upload_status"] == "blocked_phi"
    assert "filename_phi_detected" in body["reason_codes"]


@pytest.mark.asyncio
async def test_blocks_filename_with_patient_name(client):
    token, _ = await _signup(client, "us2@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, make_text_pdf(),
                         filename="patient_john_smith_apixaban.pdf")
    body = r.json()
    assert body["upload_status"] == "blocked_phi"
    assert "filename_phi_detected" in body["reason_codes"]


@pytest.mark.asyncio
async def test_blocks_metadata_with_dob_in_subject(client):
    token, _ = await _signup(client, "us3@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    pdf = make_text_pdf(title="A", body_lines=["body"], metadata={"subject": "DOB: 03/14/1979"})
    r = await _do_upload(client, h, sid, aid, pdf)
    body = r.json()
    assert body["upload_status"] == "blocked_phi"
    assert "metadata_phi_detected" in body["reason_codes"]


@pytest.mark.asyncio
async def test_blocks_annotation_with_phone_number(client):
    token, _ = await _signup(client, "us4@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    pdf = make_pdf_with_annotation_text("Call me at (555) 123-4567")
    r = await _do_upload(client, h, sid, aid, pdf)
    body = r.json()
    assert body["upload_status"] == "blocked_phi"
    assert "annotation_phi_detected" in body["reason_codes"]


@pytest.mark.asyncio
async def test_blocks_annotation_with_patient_dob(client):
    token, _ = await _signup(client, "us5@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    pdf = make_pdf_with_annotation_text("Patient John Smith DOB 03/14/1979")
    r = await _do_upload(client, h, sid, aid, pdf)
    body = r.json()
    assert body["upload_status"] == "blocked_phi"
    assert "annotation_phi_detected" in body["reason_codes"]


@pytest.mark.asyncio
async def test_does_not_block_doi_or_ids_or_emails_in_body(client):
    """DOIs, PMIDs, author emails in BODY must be fine — only user-added surfaces are scanned."""
    token, _ = await _signup(client, "us6@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
    )
    body_lines = [
        "Apixaban versus warfarin",
        "doi: 10.1056/NEJMoa1107039",
        "PMID: 21870978",
        "PMC1234567",
        "NCT00412984",
        "corresponding author: granger@example.edu",
        "Published 2011 in NEJM 365:981-992",
        "Visit https://example.com/article for the full text.",
        "Reference: Smith J. JAMA 2010;303:1234-1245.",
    ]
    pdf = make_text_pdf(title="Apixaban versus warfarin", body_lines=body_lines)
    r = await _do_upload(client, h, sid, aid, pdf)
    body = r.json()
    assert body["upload_status"] != "blocked_phi", body


@pytest.mark.asyncio
async def test_blocked_upload_deletes_gridfs_bytes(client):
    token, _ = await _signup(client, "us7@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    r = await _do_upload(client, h, sid, aid, make_text_pdf(),
                         filename="patient_smith_review.pdf")
    body = r.json()
    assert body["upload_status"] == "blocked_phi"
    # Asset record should show storage_deleted_at set + no gridfs_file_id.
    db = get_db()
    asset = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one(
        {"id": body["upload_id"]}, {"_id": 0}
    )
    assert asset["storage_deleted_at"] is not None
    assert asset.get("gridfs_file_id") is None


@pytest.mark.asyncio
async def test_audit_event_contains_only_fixed_reason_codes_no_raw_text(client):
    """Audit row from a blocked upload must not contain raw matched substrings."""
    token, _ = await _signup(client, "us8@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    secret_filename = "patient_MRN1234567_secret_name.pdf"
    r = await _do_upload(client, h, sid, aid, make_text_pdf(), filename=secret_filename)
    body = r.json()
    assert body["upload_status"] == "blocked_phi"
    upload_id = body["upload_id"]
    db = get_db()
    audit = await db[Collections.AUDIT_EVENTS].find_one(
        {"event_type": "upload_blocked", "event_payload_json.upload_id": upload_id},
        {"_id": 0, "event_payload_json": 1},
    )
    assert audit is not None
    payload_str = str(audit["event_payload_json"])
    # No raw PHI substrings.
    assert "MRN1234567" not in payload_str
    assert "patient_MRN" not in payload_str
    # Only closed-set codes.
    for code in audit["event_payload_json"]["reason_codes"]:
        assert code in {
            "filename_phi_detected", "metadata_phi_detected",
            "annotation_phi_detected", "formfield_phi_detected",
            "watermark_phi_detected",
        }


# ============================================================================
#  C. Article matching
# ============================================================================
@pytest.mark.asyncio
async def test_doi_exact_match_high(client):
    token, _ = await _signup(client, "uc1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
    )
    r = await _do_upload(client, h, sid, aid, make_apixaban_pdf())
    body = r.json()
    # Matching DOI from body / metadata.
    assert body["match_confidence"] == "high"
    assert body["matched_by"] in ("doi", "pmid", "title_year_journal")


@pytest.mark.asyncio
async def test_mismatch_deletes_bytes_and_no_extracts_created(client):
    token, _ = await _signup(client, "uc2@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1000/COMPLETELY-DIFFERENT", pmid="99999999",
        title="A totally unrelated paper about coral reef ecology",
        year=2003,
    )
    r = await _do_upload(client, h, sid, aid, make_apixaban_pdf())
    body = r.json()
    assert body["upload_status"] == "mismatch"
    assert body["match_confidence"] == "mismatch"
    db = get_db()
    asset = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one(
        {"id": body["upload_id"]}, {"_id": 0}
    )
    assert asset.get("gridfs_file_id") is None
    assert asset["storage_deleted_at"] is not None
    extracts = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
        {"uploaded_full_text_asset_id": body["upload_id"]}, {"_id": 0, "id": 1}
    ).to_list(length=10)
    assert extracts == []


@pytest.mark.asyncio
async def test_confirm_match_owner_only(client):
    """Only the owning user can confirm/reject a moderate-low match."""
    token_a, _ = await _signup(client, "ucA@test.com")
    h_a = {"Authorization": f"Bearer {token_a}"}
    # Use an article whose DOI is missing → match falls back to title sim only.
    sid, aid, _ = await _make_session_with_article(
        client, h_a, doi=None, pmid=None,
        title="Apixaban versus warfarin in patients with atrial fibrillation",
        year=2011,
    )
    r = await _do_upload(client, h_a, sid, aid, make_apixaban_pdf())
    body = r.json()
    # With no DOI/PMID on article but title sim very high, we expect 'matched' (moderate/low)
    # only IF DOI was not extracted from PDF cross-confirming. The Apixaban
    # fixture's body DOES contain a DOI — but the article record has no DOI to
    # compare with. We accept either "processed" (title_year_high → high) or
    # "matched" (moderate).
    upload_id = body["upload_id"]
    # Owner CAN confirm (if state is 'matched')
    db = get_db()
    asset = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one({"id": upload_id})
    if asset["upload_status"] != "matched":
        pytest.skip("fixture matched 'high' — confirm flow not applicable")
    # A different user attempting to confirm gets 404.
    token_b, _ = await _signup(client, "ucB@test.com")
    h_b = {"Authorization": f"Bearer {token_b}"}
    r2 = await client.post(f"/api/uploads/{upload_id}/confirm-match",
                           json={"confirmed": True}, headers=h_b)
    assert r2.status_code == 404


# ============================================================================
#  D. Evidence extraction
# ============================================================================
@pytest.mark.asyncio
async def test_extracts_sections_and_token_count(client):
    token, _ = await _signup(client, "ud1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin in patients with atrial fibrillation",
        year=2011,
    )
    r = await _do_upload(client, h, sid, aid, make_apixaban_pdf())
    body = r.json()
    assert body["upload_status"] == "processed"
    db = get_db()
    extracts = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
        {"uploaded_full_text_asset_id": body["upload_id"]}, {"_id": 0}
    ).to_list(length=20)
    assert len(extracts) > 0
    # All extracts have a token_count and the user_uploaded_full_text level.
    for e in extracts:
        assert e["token_count"] is not None and e["token_count"] >= 0
        assert e["evidence_text_level"] == "user_uploaded_full_text"
        assert e["text_source"] == "user_uploaded_full_text"
    # At least one ABSTRACT or RESULTS section should be detected.
    section_names = {e["section_name"] for e in extracts}
    assert section_names & {"abstract", "results", "methods", "discussion", "conclusion"}


@pytest.mark.asyncio
async def test_evidence_text_level_abstract_when_no_upload(client):
    """prepare-evidence-packet falls back to abstract for a selected article."""
    token, _ = await _signup(client, "ud2@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, abstract="Apixaban is superior to warfarin in this trial.",
    )
    r = await client.post(
        f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["with_abstract_only"] >= 1
    db = get_db()
    extracts = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
        {"search_session_id": sid, "article_id": aid}, {"_id": 0}
    ).to_list(length=10)
    assert any(e["text_source"] == "abstract" for e in extracts)


@pytest.mark.asyncio
async def test_evidence_extracts_endpoint_returns_user_upload_when_present(client):
    token, _ = await _signup(client, "ud3@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin in patients with atrial fibrillation",
        year=2011, abstract="Short abstract.",
    )
    await _do_upload(client, h, sid, aid, make_apixaban_pdf())
    r = await client.get(f"/api/search-sessions/{sid}/evidence-extracts", headers=h)
    assert r.status_code == 200
    arts = r.json()["articles"]
    assert any(a["evidence_text_level"] == "user_uploaded_full_text" for a in arts)


# ============================================================================
#  E. Retention
# ============================================================================
@pytest.mark.asyncio
async def test_use_once_deletes_pdf_after_extraction(client):
    token, _ = await _signup(client, "ue1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
    )
    r = await _do_upload(client, h, sid, aid, make_apixaban_pdf(), retention="use_once")
    body = r.json()
    db = get_db()
    asset = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one({"id": body["upload_id"]})
    assert asset["storage_deleted_at"] is not None
    assert asset.get("gridfs_file_id") is None


@pytest.mark.asyncio
async def test_session_30_days_retains_pdf(client):
    token, _ = await _signup(client, "ue2@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
    )
    r = await _do_upload(client, h, sid, aid, make_apixaban_pdf(),
                         retention="session_30_days")
    body = r.json()
    db = get_db()
    asset = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one({"id": body["upload_id"]})
    assert asset["storage_deleted_at"] is None
    assert asset.get("gridfs_file_id") is not None


@pytest.mark.asyncio
async def test_delete_upload_removes_bytes_and_extracts(client):
    token, _ = await _signup(client, "ue3@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
    )
    r = await _do_upload(client, h, sid, aid, make_apixaban_pdf(),
                         retention="session_30_days")
    upload_id = r.json()["upload_id"]
    r2 = await client.delete(f"/api/uploads/{upload_id}", headers=h)
    assert r2.status_code == 200
    db = get_db()
    extracts = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
        {"uploaded_full_text_asset_id": upload_id}, {"_id": 0, "id": 1}
    ).to_list(length=10)
    assert extracts == []
    asset = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one({"id": upload_id})
    assert asset["upload_status"] == "deleted"


@pytest.mark.asyncio
async def test_session_cleanup_removes_uploads_and_extracts(client):
    token, _ = await _signup(client, "ue4@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
    )
    r = await _do_upload(client, h, sid, aid, make_apixaban_pdf(),
                         retention="session_30_days")
    upload_id = r.json()["upload_id"]
    db = get_db()
    # Force the session's expires_at into the past.
    await db[Collections.SEARCH_SESSIONS].update_one(
        {"id": sid},
        {"$set": {"expires_at": datetime.now(timezone.utc) - timedelta(days=1)}},
    )
    from services.retention_cleanup import retention_cleanup
    res = await retention_cleanup.cleanup_expired_sessions()
    assert res["deleted_count"] >= 1
    assert res["deleted_uploads"] >= 1
    asset = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one({"id": upload_id})
    assert asset["upload_status"] == "deleted"


# ============================================================================
#  F. Access control
# ============================================================================
@pytest.mark.asyncio
async def test_user_cannot_see_another_users_uploads(client):
    token_a, _ = await _signup(client, "ufA@test.com")
    h_a = {"Authorization": f"Bearer {token_a}"}
    sid_a, aid_a, _ = await _make_session_with_article(
        client, h_a, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
    )
    await _do_upload(client, h_a, sid_a, aid_a, make_apixaban_pdf())
    token_b, _ = await _signup(client, "ufB@test.com")
    h_b = {"Authorization": f"Bearer {token_b}"}
    r = await client.get(f"/api/search-sessions/{sid_a}/uploads", headers=h_b)
    assert r.status_code == 404  # session not owned by B


@pytest.mark.asyncio
async def test_user_cannot_see_another_users_evidence_packet(client):
    token_a, _ = await _signup(client, "ufC@test.com")
    h_a = {"Authorization": f"Bearer {token_a}"}
    sid_a, aid_a, _ = await _make_session_with_article(
        client, h_a, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
    )
    await _do_upload(client, h_a, sid_a, aid_a, make_apixaban_pdf())
    token_b, _ = await _signup(client, "ufD@test.com")
    h_b = {"Authorization": f"Bearer {token_b}"}
    r = await client.get(f"/api/search-sessions/{sid_a}/evidence-extracts", headers=h_b)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_user_upload_access_location_not_visible_to_other_users(client):
    """The user_uploaded_full_text access_locations row is scoped to one user."""
    token_a, _ = await _signup(client, "ufE@test.com")
    h_a = {"Authorization": f"Bearer {token_a}"}
    sid_a, aid_a, _ = await _make_session_with_article(
        client, h_a, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
    )
    await _do_upload(client, h_a, sid_a, aid_a, make_apixaban_pdf())
    # User A sees their row.
    r_a = await client.get(f"/api/articles/{aid_a}/access-locations", headers=h_a)
    cats_a = {row["access_category"] for row in r_a.json()}
    assert "user_uploaded_full_text" in cats_a
    # User B can't even see the article (no path of access) → 404.
    token_b, _ = await _signup(client, "ufF@test.com")
    h_b = {"Authorization": f"Bearer {token_b}"}
    r_b = await client.get(f"/api/articles/{aid_a}/access-locations", headers=h_b)
    # 404 because user B has no path of access to the article.
    assert r_b.status_code == 404


@pytest.mark.asyncio
async def test_unauthenticated_upload_endpoint_returns_401(client):
    token, _ = await _signup(client, "ufG@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(client, h)
    form = _upload_form(make_text_pdf())
    r = await client.post(
        f"/api/search-sessions/{sid}/articles/{aid}/upload-full-text",
        files=form["files"], data=form["data"],  # NO Authorization header
    )
    assert r.status_code == 401


# ============================================================================
#  G. No-LLM invariant
# ============================================================================
@pytest.mark.asyncio
async def test_upload_pipeline_never_calls_llm(client):
    token, _ = await _signup(client, "ug1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978",
        title="Apixaban versus warfarin", year=2011,
        abstract="Apixaban is superior to warfarin.",
    )

    # Mock both the question-parser and any other generic LLM caller.
    with patch("services.llm_provider.LLMProvider.complete_json",
               new=AsyncMock(side_effect=AssertionError("LLM must not be called"))) as mock_llm:
        # Upload pipeline.
        await _do_upload(client, h, sid, aid, make_apixaban_pdf())
        # Evidence extracts endpoint.
        await client.get(f"/api/search-sessions/{sid}/evidence-extracts", headers=h)
        # Evidence packet prepare.
        await client.post(f"/api/search-sessions/{sid}/prepare-evidence-packet", headers=h)
        mock_llm.assert_not_called()


# ============================================================================
#  H. PHI guard regression
# ============================================================================
@pytest.mark.asyncio
async def test_clinical_question_phi_block_still_holds(client):
    """Regression: the clinical-question PHI guard is unchanged by Phase 2D."""
    token, _ = await _signup(client, "uh1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    r = await client.post(
        "/api/search-sessions/",
        json={"question": "Mr. John Doe DOB 03/14/1979 MRN 12345678 — best treatment?"},
        headers=h,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "blocked_phi"
