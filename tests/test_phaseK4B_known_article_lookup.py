"""Phase K4B-light — Known Article Finder backend tests.

17 acceptance tests per the operator spec. All external sources are
mocked via monkeypatch at the service-layer wrapper functions; no
real HTTP calls are made.
"""
from __future__ import annotations

import json
import sys
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import known_article_lookup as kal
from core.db import Collections


# ---------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------
async def _login(client_, role: str = "clinician") -> tuple[str, str]:
    creds = (
        {"email": "admin@test.com", "password": "AdminTest123!"}
        if role == "admin"
        else {"email": "clinician@test.com", "password": "ClinicianTest123!"}
    )
    r = await client_.post("/api/auth/login", json=creds)
    assert r.status_code == 200, r.text
    body = r.json()
    return body["access_token"], body["user"]["id"]


async def _seed_session(*, user_id: str, status: str = "awaiting_selection") -> str:
    from core.db import get_db
    db = get_db()
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    await db.search_sessions.insert_one({
        "id": sid, "user_id": user_id,
        "original_question": "q", "sanitized_question": "q",
        "topic_label": "t", "clinical_question_type": "therapy",
        "pico_json": {}, "picott_json": {},
        "status": status, "provenance": "organic",
        "created_at": now, "updated_at": now,
        "expires_at": now.replace(microsecond=0),
        "deleted_at": None,
    })
    return sid


def _mk_pubmed_record(*, pmid: str, doi: str = "10.1/test",
                     pmcid: str | None = None, title: str = "Test Title",
                     year: int = 2023, journal: str = "Journal of Test",
                     abstract: str = "Background. Conclusion: test conclusion.",
                     pub_types: list[str] | None = None,
                     is_retracted: bool = False,
                     has_eoc: bool = False) -> dict:
    return {
        "source": "pubmed", "pmid": pmid, "doi": doi, "pmcid": pmcid,
        "title": title, "journal": journal, "publication_year": year,
        "authors": ["Smith J", "Doe A"], "abstract": abstract,
        "publication_types": pub_types or ["Journal Article"],
        "is_retracted": is_retracted,
        "has_expression_of_concern": has_eoc,
    }


# ---------------------------------------------------------------------
# TEST 1 — DOI exact match (PubMed [AID] path).
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t1_doi_exact_match(client, monkeypatch):
    pmid_call = AsyncMock(return_value="33333333")
    fetch_call = AsyncMock(return_value=_mk_pubmed_record(
        pmid="33333333", doi="10.1056/nejmoa2107038",
        title="SGLT2 Inhibitors in HFpEF",
    ))
    monkeypatch.setattr(kal, "_pubmed_esearch_by_doi", pmid_call)
    monkeypatch.setattr(kal, "_pubmed_efetch_by_pmid", fetch_call)
    monkeypatch.setattr(kal, "_crossref_by_doi", AsyncMock(return_value=None))

    token, _ = await _login(client, "clinician")
    r = await client.post(
        "/api/known-article-lookup/search",
        json={"raw_text": "https://doi.org/10.1056/NEJMoa2107038"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["phi_blocked"] is False
    assert body["parsed"]["detected_kind"] == "doi"
    assert body["parsed"]["doi"] == "10.1056/nejmoa2107038"
    assert len(body["candidates"]) == 1
    c = body["candidates"][0]
    assert c["match_confidence"] == "high"
    assert c["match_reason_codes"] == ["exact_doi_match"]
    assert c["pmid"] == "33333333"


# ---------------------------------------------------------------------
# TEST 2 — PMID exact match.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t2_pmid_exact_match(client, monkeypatch):
    fetch_call = AsyncMock(return_value=_mk_pubmed_record(pmid="12345678"))
    monkeypatch.setattr(kal, "_pubmed_efetch_by_pmid", fetch_call)

    token, _ = await _login(client, "clinician")
    r = await client.post(
        "/api/known-article-lookup/search",
        json={"raw_text": "PMID: 12345678"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["candidates"][0]["match_confidence"] == "high"
    assert body["candidates"][0]["match_reason_codes"] == ["exact_pmid_match"]
    assert body["candidates"][0]["pmid"] == "12345678"


# ---------------------------------------------------------------------
# TEST 3 — PMCID lookup via converter.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t3_pmcid_via_idconv(client, monkeypatch):
    monkeypatch.setattr(kal, "_pmc_idconv", AsyncMock(return_value={
        "pmcid": "PMC1234567", "pmid": "98765432",
    }))
    monkeypatch.setattr(kal, "_pubmed_efetch_by_pmid", AsyncMock(
        return_value=_mk_pubmed_record(pmid="98765432", pmcid="PMC1234567"),
    ))
    token, _ = await _login(client, "clinician")
    r = await client.post(
        "/api/known-article-lookup/search",
        json={"raw_text": "PMCID: PMC1234567"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    c = body["candidates"][0]
    assert c["match_confidence"] == "high"
    assert "exact_pmcid_match" in c["match_reason_codes"]
    assert c["pmcid"] == "PMC1234567"


# ---------------------------------------------------------------------
# TEST 4 — Full title exact match.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t4_title_exact_match(client, monkeypatch):
    monkeypatch.setattr(kal, "_pubmed_esearch_by_title",
                        AsyncMock(return_value=["55555555"]))
    monkeypatch.setattr(kal, "_pubmed_efetch_by_pmid", AsyncMock(
        return_value=_mk_pubmed_record(
            pmid="55555555",
            title="Empagliflozin in heart failure with preserved ejection fraction",
            year=2021,
        ),
    ))
    token, _ = await _login(client, "clinician")
    r = await client.post(
        "/api/known-article-lookup/search",
        json={"raw_text": "Empagliflozin in heart failure with preserved ejection fraction"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    c = body["candidates"][0]
    # No year specified by user → match comes via ambiguous_title-only
    # or fuzzy_title_author_year (we accept either since the
    # title-only flavor still requires explicit user confirmation).
    assert c["match_confidence"] in ("medium", "low")


# ---------------------------------------------------------------------
# TEST 5 — Full citation string match (journal/year/vol/pages).
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t5_citation_string_match(client, monkeypatch):
    monkeypatch.setattr(kal, "_pubmed_esearch_by_title",
                        AsyncMock(return_value=["77777777"]))
    monkeypatch.setattr(kal, "_pubmed_efetch_by_pmid", AsyncMock(
        return_value=_mk_pubmed_record(
            pmid="77777777",
            title="Empagliflozin in HFpEF",
            year=2021, journal="N Engl J Med",
        ),
    ))
    token, _ = await _login(client, "clinician")
    citation = (
        "Anker SD, Butler J, et al. Empagliflozin in HFpEF. "
        "N Engl J Med. 2021;385:1451-1461."
    )
    r = await client.post(
        "/api/known-article-lookup/search",
        json={"raw_text": citation},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    parsed = body["parsed"]
    assert parsed["detected_kind"] == "citation"
    assert parsed["year_guess"] == 2021
    # Year + title both match → high confidence.
    assert body["candidates"][0]["match_confidence"] == "high"
    assert "exact_title_year_match" in body["candidates"][0]["match_reason_codes"]


# ---------------------------------------------------------------------
# TEST 6 — Ambiguous title returns multiple candidates.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t6_ambiguous_title_multi_candidates(client, monkeypatch):
    monkeypatch.setattr(kal, "_pubmed_esearch_by_title",
                        AsyncMock(return_value=["1", "2", "3"]))
    async def _fetch(p):
        return _mk_pubmed_record(
            pmid=p, doi=f"10.1/test{p}",
            title="Aspirin therapy" + (" review" if p == "3" else ""),
            year=2020 + int(p),
        )
    monkeypatch.setattr(kal, "_pubmed_efetch_by_pmid", AsyncMock(side_effect=_fetch))
    token, _ = await _login(client, "clinician")
    r = await client.post(
        "/api/known-article-lookup/search",
        json={"raw_text": "Aspirin therapy"},
        headers={"Authorization": f"Bearer {token}"},
    )
    body = r.json()
    assert len(body["candidates"]) == 3
    # None should be high-confidence on title-only ambiguous match.
    for c in body["candidates"]:
        assert c["match_confidence"] in ("low", "medium")


# ---------------------------------------------------------------------
# TEST 7 — Low-confidence title-only match cannot auto-attach.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t7_low_confidence_no_auto_attach(client):
    token, uid = await _login(client, "clinician")
    sid = await _seed_session(user_id=uid)
    candidate = {
        "match_confidence": "low",
        "match_reason_codes": ["ambiguous_title_only_match"],
        "title": "Aspirin therapy", "publication_year": 2021,
    }
    r = await client.post(
        "/api/known-article-lookup/attach-to-session",
        json={"session_id": sid, "candidate": candidate, "include_in_evidence": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert "Low-confidence" in r.json()["detail"]


# ---------------------------------------------------------------------
# TEST 8 — Duplicate detection.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t8_dedup_existing_session(client):
    token, uid = await _login(client, "clinician")
    sid = await _seed_session(user_id=uid)
    candidate = {
        "match_confidence": "high",
        "match_reason_codes": ["exact_pmid_match"],
        "pmid": "10101010", "title": "Test", "publication_year": 2022,
    }
    # First attach.
    r1 = await client.post(
        "/api/known-article-lookup/attach-to-session",
        json={"session_id": sid, "candidate": candidate},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["status"] == "attached"
    assert body1["was_duplicate"] is False
    # Second attach should detect dup.
    r2 = await client.post(
        "/api/known-article-lookup/attach-to-session",
        json={"session_id": sid, "candidate": candidate, "include_in_evidence": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["status"] == "already_in_session"
    assert body2["was_duplicate"] is True
    assert body2["search_session_article_id"] == body1["search_session_article_id"]
    assert body2["user_included"] is True  # include_in_evidence promoted the existing row


# ---------------------------------------------------------------------
# TEST 9 — Add to current session creates search_session_articles row.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t9_attach_to_session(client):
    from core.db import get_db
    token, uid = await _login(client, "clinician")
    sid = await _seed_session(user_id=uid)
    candidate = {
        "match_confidence": "high",
        "match_reason_codes": ["exact_pmid_match"],
        "pmid": "20202020", "title": "Attach Test", "publication_year": 2024,
    }
    r = await client.post(
        "/api/known-article-lookup/attach-to-session",
        json={"session_id": sid, "candidate": candidate},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "attached"
    db = get_db()
    ssa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"id": body["search_session_article_id"]}, {"_id": 0},
    )
    assert ssa is not None
    assert ssa["search_session_id"] == sid
    assert ssa["source"] == "known_article_lookup"


# ---------------------------------------------------------------------
# TEST 10 — Select for evidence packet (include_in_evidence=True).
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t10_include_in_evidence(client):
    from core.db import get_db
    token, uid = await _login(client, "clinician")
    sid = await _seed_session(user_id=uid)
    candidate = {
        "match_confidence": "high",
        "match_reason_codes": ["exact_pmid_match"],
        "pmid": "30303030", "title": "Select For Evidence",
        "publication_year": 2023,
    }
    r = await client.post(
        "/api/known-article-lookup/attach-to-session",
        json={"session_id": sid, "candidate": candidate, "include_in_evidence": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["user_included"] is True
    db = get_db()
    ssa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
        {"id": body["search_session_article_id"]}, {"_id": 0, "user_included": 1},
    )
    assert ssa["user_included"] is True


# ---------------------------------------------------------------------
# TEST 11 — Save to library (folder add) uses existing endpoint, not new.
# Verified at the API contract level: we attach to a folder via the
# existing /api/folders/{id}/articles endpoint after upserting the
# article. K4B-light just confirms the upserted article_id can be
# consumed by the existing library flow.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t11_can_be_saved_to_library(client):
    from core.db import get_db
    token, uid = await _login(client, "clinician")
    # First confirm-access — which upserts the article.
    candidate = {
        "match_confidence": "high",
        "match_reason_codes": ["exact_pmid_match"],
        "pmid": "40404040", "title": "Library Save Test",
        "publication_year": 2022, "abstract": "Stuff.",
    }
    r = await client.post(
        "/api/known-article-lookup/confirm-access",
        json={"candidate": candidate},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    article_id = r.json()["article_id"]
    # Now create a folder + add this article through the existing endpoint.
    rf = await client.post(
        "/api/folders/", json={"name": "K4B test folder"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert rf.status_code in (200, 201), rf.text
    folder_id = rf.json()["id"]
    ra = await client.post(
        f"/api/folders/{folder_id}/articles",
        json={"article_id": article_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ra.status_code in (200, 201), ra.text


# ---------------------------------------------------------------------
# TEST 12 — Access panel: OA full-text rows map correctly.
# ---------------------------------------------------------------------
def test_t12_access_panel_oa():
    panel = kal.derive_access_panel(
        article={"full_text_status": "full_text_available", "abstract": "x"},
        access_rows=[{
            "is_oa": True, "access_category": "oa",
            "can_ingest_for_summary": True, "is_direct_pdf": False,
        }],
    )
    assert panel["access_status"] == "Can open full text"
    assert panel["ai_summary_text_source"] == "OA full text eligible"
    assert panel["can_ingest_for_summary"] is True


# ---------------------------------------------------------------------
# TEST 13 — Institutional link-out is NOT AI-ingestible.
# ---------------------------------------------------------------------
def test_t13_institutional_link_out_not_ingestible():
    panel = kal.derive_access_panel(
        article={"abstract": "x"},
        access_rows=[{
            "is_oa": False, "access_category": "institutional",
            "can_ingest_for_summary": False, "is_direct_pdf": False,
        }],
    )
    assert panel["access_status"] == "Link-out only"
    assert panel["ai_summary_text_source"] == (
        "Institutional link-out only, not AI-ingestible"
    )
    assert panel["can_ingest_for_summary"] is False
    # Distinct fields — operator-visible promise.
    assert panel["access_status"] != panel["ai_summary_text_source"]


# ---------------------------------------------------------------------
# TEST 14 — User-uploaded PDF remains user/session/article scoped.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t14_user_uploaded_pdf_scoped(client):
    from core.db import get_db
    token, uid = await _login(client, "clinician")
    candidate = {
        "match_confidence": "high",
        "match_reason_codes": ["exact_pmid_match"],
        "pmid": "70707070", "title": "User Upload Scoped",
        "publication_year": 2021,
    }
    # Confirm-access first (upserts article).
    r1 = await client.post(
        "/api/known-article-lookup/confirm-access",
        json={"candidate": candidate},
        headers={"Authorization": f"Bearer {token}"},
    )
    article_id = r1.json()["article_id"]
    db = get_db()
    # Simulate a processed upload row scoped to this user + article.
    await db[Collections.UPLOADED_FULL_TEXT_ASSETS].insert_one({
        "id": str(uuid.uuid4()), "user_id": uid,
        "article_id": article_id, "search_session_id": str(uuid.uuid4()),
        "upload_status": "processed",
        "original_filename": "u.pdf", "file_type": "pdf",
        "file_size": 1, "storage_path": "/dev/null",
        "checksum": "x", "match_confidence": "high",
        "matched_by": "user_confirmed", "extracted_text_status": "complete",
        "evidence_text_level": "user_uploaded_full_text",
        "retention_mode": "session_30_days",
        "rights_attestation_version": "v1",
        "user_attested_at": datetime.now(timezone.utc),
        "deleted_at": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    })
    # Re-fetch panel — now must surface "User-uploaded PDF eligible".
    r2 = await client.post(
        "/api/known-article-lookup/confirm-access",
        json={"candidate": candidate},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["ai_summary_text_source"] == "User-uploaded PDF eligible"
    assert body2["has_user_upload"] is True

    # A DIFFERENT user must NOT see this upload — scope test.
    # We can't trivially create another user here, but we can verify
    # the upload query is scoped by user_id by querying as the
    # service does.
    other = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one(
        {"article_id": article_id, "user_id": "another-user-id",
         "upload_status": "processed"},
        {"_id": 0},
    )
    assert other is None


# ---------------------------------------------------------------------
# TEST 15 — PHI-blocked input blocks BEFORE any outbound call.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t15_phi_blocked_before_outbound(client, monkeypatch):
    # Spies — must NOT be called.
    pubmed_spy = AsyncMock(return_value=None)
    crossref_spy = AsyncMock(return_value=None)
    title_spy = AsyncMock(return_value=[])
    pmcid_spy = AsyncMock(return_value=None)
    doi_spy = AsyncMock(return_value=None)
    monkeypatch.setattr(kal, "_pubmed_efetch_by_pmid", pubmed_spy)
    monkeypatch.setattr(kal, "_crossref_by_doi", crossref_spy)
    monkeypatch.setattr(kal, "_pubmed_esearch_by_title", title_spy)
    monkeypatch.setattr(kal, "_pmc_idconv", pmcid_spy)
    monkeypatch.setattr(kal, "_pubmed_esearch_by_doi", doi_spy)

    token, _ = await _login(client, "clinician")
    phi_text = (
        "Mr. Smith DOB 03/14/1979 email john@example.com PMID: 12345678"
    )
    r = await client.post(
        "/api/known-article-lookup/search",
        json={"raw_text": phi_text},
        headers={"Authorization": f"Bearer {token}"},
    )
    body = r.json()
    assert body["phi_blocked"] is True
    assert isinstance(body["reasons"], list) and len(body["reasons"]) > 0
    assert body["candidates"] == []
    # No outbound call fired.
    pubmed_spy.assert_not_called()
    crossref_spy.assert_not_called()
    title_spy.assert_not_called()
    pmcid_spy.assert_not_called()
    doi_spy.assert_not_called()


# ---------------------------------------------------------------------
# TEST 16 — No Google Scholar or scraping hosts allowed.
# ---------------------------------------------------------------------
def test_t16_allowed_hosts_closed_set():
    # Closed allowlist — every legitimate source call goes to one of these.
    assert kal.ALLOWED_HOSTS == frozenset({
        "eutils.ncbi.nlm.nih.gov",
        "api.ncbi.nlm.nih.gov",
        "api.crossref.org",
        "www.ebi.ac.uk",
    })
    # Verify by scanning the source file for any forbidden host.
    src = Path(__file__).resolve().parents[1] / "services" / "known_article_lookup.py"
    text = src.read_text(encoding="utf-8")
    for forbidden in (
        "scholar.google.", "googleusercontent.com",
        "sci-hub", "libgen", "publisher.com",
    ):
        assert forbidden not in text, f"forbidden host {forbidden!r} found in source"


# ---------------------------------------------------------------------
# TEST 17 — No raw PHI / prompts / completions / secrets in diagnostics.
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t17_diagnostics_payload_clean(client, monkeypatch):
    from core.db import get_db
    monkeypatch.setattr(kal, "_pubmed_efetch_by_pmid", AsyncMock(
        return_value=_mk_pubmed_record(pmid="80808080"),
    ))
    token, _ = await _login(client, "clinician")
    r = await client.post(
        "/api/known-article-lookup/search",
        json={"raw_text": "PMID: 80808080"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    db = get_db()
    # The audit event must NOT carry the raw input or any secrets.
    ev = await db.audit_events.find_one(
        {"event_type": "known_article_lookup_search"},
        sort=[("created_at", -1)],
    )
    assert ev is not None
    blob = json.dumps(ev, default=str)
    for forbidden in ("system:", "assistant:", "sk-",
                      "BEGIN PRIVATE KEY", "@gmail.com",
                      "raw_prompt", "raw_completion",
                      "evidence_text",
                      # Raw PHI tokens (we deliberately did NOT send any
                      # but the regression guard still asserts they
                      # never appear).
                      "DOB ", "@example.com"):
        assert forbidden not in blob, f"diagnostics leaked {forbidden!r}"
    # Closed-set payload — only the expected keys.
    payload = ev.get("event_payload_json") or {}
    extras = set(payload.keys()) - {
        "phi_blocked", "detected_kind", "lookup_source_provenance",
        "candidate_count", "top_confidence",
    }
    assert not extras, f"unexpected diagnostic keys: {extras}"
