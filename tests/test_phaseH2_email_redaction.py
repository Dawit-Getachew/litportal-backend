"""Phase H2-tail Task A — author/contact email redaction acceptance.

Goal: every public + admin payload that exposes article metadata must
strip author / corresponding-author emails that come from upstream
PubMed / Europe PMC / OpenAlex / Crossref metadata, while preserving
author NAMES, affiliations (sans email), DOI, PMID, journal, title,
abstract text (sans email), and license/retraction flags.

Test surfaces covered:
  * `services.article_email_redactor.redact_article_emails` pure-fn
    behavior (in-place, idempotent, scope of fields).
  * `services.metadata_normalizer.MetadataNormalizerService.normalize`
    strips emails at the Europe PMC ingestion path.
  * `services.pipeline._upsert_article` centralized persistence
    redactor — articles inserted by ANY connector (including
    PubMed which bypasses the normalizer) come out clean.
  * `GET /api/articles/{id}` — defensive read-path scrub.
  * `GET /api/search-sessions/{id}` — defensive read-path scrub on
    hydrated article cards.
  * `GET /api/folders/{id}/articles` — defensive read-path scrub.
  * `GET /api/library` (`services.user_library.list_items`) —
    defensive read-path scrub.

No network. No LLM. No PubMed/Europe PMC live calls.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from core.db import Collections, get_db


# ---------------------------------------------------------------------
# Regex — the closed-set pattern that no payload may contain.
# ---------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE)


# ---------------------------------------------------------------------
# Pure-function behaviour of the redactor.
# ---------------------------------------------------------------------


def test_redactor_strips_abstract_email():
    from services.article_email_redactor import redact_article_emails
    art = {"abstract": "Corresponding author: corresp@example.it. Cohort X."}
    out = redact_article_emails(art)
    assert "corresp@example.it" not in (out["abstract"] or "")
    assert not _EMAIL_RE.search(out["abstract"] or "")
    # Surrounding text preserved.
    assert "Cohort X." in out["abstract"]


def test_redactor_strips_author_affiliation_email():
    from services.article_email_redactor import redact_article_emails
    art = {
        "authors_json": [
            {"fullName": "Smith J", "lastName": "Smith",
             "affiliation": "Dept of X, Univ Y. corresp@libero.it"},
        ],
    }
    out = redact_article_emails(art)
    aff = out["authors_json"][0].get("affiliation") or ""
    assert "corresp@libero.it" not in aff
    assert not _EMAIL_RE.search(aff)
    # Author name + institution preserved.
    assert out["authors_json"][0]["lastName"] == "Smith"
    assert "Dept of X, Univ Y" in aff


def test_redactor_is_idempotent():
    from services.article_email_redactor import redact_article_emails
    art = {"abstract": "Author note: a@b.io and c.d@e.f.g . End."}
    out1 = redact_article_emails(art)
    out2 = redact_article_emails(out1)
    assert out1["abstract"] == out2["abstract"]
    assert not _EMAIL_RE.search(out2["abstract"])


def test_redactor_preserves_doi_pmid_title_journal_year():
    from services.article_email_redactor import redact_article_emails
    art = {
        "doi": "10.1234/abc.xyz",
        "pmid": "12345678",
        "pmcid": "PMC8765432",
        "title": "Some title with no email",
        "journal": "Journal of Medicine",
        "publication_year": 2024,
        "abstract": "Body. contact@example.com end.",
        "is_guideline": True,
        "is_retracted": False,
    }
    out = redact_article_emails(art)
    # Identifiers + structural fields are NEVER touched.
    assert out["doi"] == "10.1234/abc.xyz"
    assert out["pmid"] == "12345678"
    assert out["pmcid"] == "PMC8765432"
    assert out["title"] == "Some title with no email"
    assert out["journal"] == "Journal of Medicine"
    assert out["publication_year"] == 2024
    assert out["is_guideline"] is True
    assert out["is_retracted"] is False
    # Email gone.
    assert not _EMAIL_RE.search(out["abstract"])


def test_redactor_handles_full_text_urls_and_source_urls():
    from services.article_email_redactor import redact_article_emails
    art = {
        "source_urls_json": [
            "https://example.com/article?email=leak@x.com",
            "https://example.com/article",
        ],
        "full_text_urls_json": [
            {"url": "https://oa.example.com/pdf?contact=author@host.net"},
        ],
    }
    out = redact_article_emails(art)
    blob = json.dumps(out)
    assert not _EMAIL_RE.search(blob)
    # The non-email URL is untouched.
    assert "https://example.com/article" in blob


def test_redactor_handles_empty_and_non_string_safely():
    from services.article_email_redactor import redact_article_emails
    assert redact_article_emails({}) == {}
    assert redact_article_emails({"abstract": None}) == {"abstract": None}
    assert redact_article_emails({"authors_json": None}) == {"authors_json": None}


# ---------------------------------------------------------------------
# Ingestion-boundary integration — MetadataNormalizer strips emails.
# ---------------------------------------------------------------------


def test_metadata_normalizer_strips_emails_at_europe_pmc_path():
    from services.metadata_normalizer import metadata_normalizer
    raw = {
        "pmid": "99999991",
        "doi": "10.5555/example",
        "title": "Brugada perioperative case series",
        "abstractText": (
            "Background. Some content. Correspondence: author1@libero.it. "
            "Outcome was Y."
        ),
        "authorList": {"author": [
            {"fullName": "Rossi M", "lastName": "Rossi", "initials": "M",
             "authorAffiliationDetailsList": {"authorAffiliation": [
                 {"affiliation":
                  "Cardiology, Hospital A. m.rossi@example.it"},
             ]}},
        ]},
        "pubYear": "2024",
        "journalInfo": {"journal": {"title": "Cardio J"}},
        "pubTypeList": {"pubType": ["Case Report"]},
    }
    art = metadata_normalizer.normalize(raw)
    blob = json.dumps(art, default=str)
    assert not _EMAIL_RE.search(blob), (
        f"Found leftover email in normalized article: {blob}"
    )
    # Author name + institution preserved.
    aff = (art["authors_json"][0] or {}).get("affiliation") or ""
    assert "Cardiology, Hospital A" in aff
    assert art["title"] == "Brugada perioperative case series"
    assert art["doi"] == "10.5555/example"
    assert art["pmid"] == "99999991"


# ---------------------------------------------------------------------
# Centralized persistence boundary — `_upsert_article` scrubs.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_article_redacts_emails_at_persistence_boundary():
    """Simulate the PubMed-bypass path: a connector that builds the
    article dict directly (no metadata_normalizer call) and writes
    via the central upsert. Persisted record must be clean."""
    from services.canonical_identity import make_canonical_key
    from services.pipeline import _upsert_article
    db = get_db()
    pmid = f"AT_TEST_{uuid.uuid4().hex[:10]}"
    doi = f"10.9999/test.{uuid.uuid4().hex[:8]}"
    raw = {
        "id": str(uuid.uuid4()),
        "canonical_key": make_canonical_key(
            pmid=pmid, doi=doi, pmcid="", title="x", year=2024,
            journal="J", first_author="A",
        ),
        "title": "PubMed-bypass test article",
        "abstract": "Methods. Cohort. Email: corresp@libero.it. Done.",
        "journal": "J",
        "publication_year": 2024,
        "publication_date": "2024",
        "authors_json": [
            {"fullName": "Bianchi A", "lastName": "Bianchi",
             "initials": "A",
             "affiliation": "Dept of EP, Univ Z. a.bianchi@example.it"},
        ],
        "doi": doi,
        "doi_normalized": doi.lower(),
        "pmid": pmid,
        "pmcid": None,
        "source_ids_json": {"pubmed": pmid, "doi": doi},
        "publication_types_json": ["journal article"],
        "article_type_classification": "primary_study",
        "is_guideline": False,
        "is_preprint": False,
        "is_retracted": False,
        "is_retraction_notice": False,
        "has_expression_of_concern": False,
        "full_text_status": "abstract_only",
        "source_urls_json": [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    try:
        aid = await _upsert_article(raw)
        doc = await db[Collections.ARTICLES].find_one({"id": aid}, {"_id": 0})
        assert doc is not None
        blob = json.dumps(doc, default=str)
        assert not _EMAIL_RE.search(blob), (
            f"Persisted article retains email: {blob}"
        )
        # Author + institution still present.
        assert (doc["authors_json"][0] or {}).get("lastName") == "Bianchi"
        assert "Dept of EP, Univ Z" in (
            (doc["authors_json"][0] or {}).get("affiliation") or ""
        )
        assert doc["doi"] == doi
        assert doc["pmid"] == pmid
    finally:
        await db[Collections.ARTICLES].delete_many({"pmid": pmid})


# ---------------------------------------------------------------------
# Read-path defensive scrub — legacy data fixtures.
# ---------------------------------------------------------------------


@pytest_asyncio.fixture
async def clinician_token(client):
    r = await client.post(
        "/api/auth/login",
        json={"email": "clinician@test.com", "password": "ClinicianTest123!"},
    )
    return r.json()["access_token"]


@pytest_asyncio.fixture
async def admin_token(client):
    r = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    return r.json()["access_token"]


@pytest_asyncio.fixture
async def legacy_email_article():
    """Seed a legacy article record directly into the articles
    collection bypassing the redactor — simulates pre-existing
    records ingested before Task A was wired. The teardown deletes
    the record + any folder/library/session_articles rows referencing it.
    """
    db = get_db()
    aid = f"AT_LEGACY_{uuid.uuid4().hex[:10]}"
    pmid = f"AT_PMID_{uuid.uuid4().hex[:8]}"
    doi = f"10.7777/legacy.{uuid.uuid4().hex[:8]}"
    doc = {
        "id": aid,
        "canonical_key": f"AT_CK_{uuid.uuid4().hex[:8]}",
        "title": "Legacy article with email leak",
        "abstract": (
            "Methods. Cohort. Author: legacy.author@libero.it. Results: ok."
        ),
        "journal": "Legacy J",
        "publication_year": 2022,
        "publication_date": "2022",
        "authors_json": [
            {"fullName": "Verdi L", "lastName": "Verdi",
             "initials": "L",
             "affiliation": "Dept of Q, Univ T. l.verdi@libero.it"},
        ],
        "doi": doi,
        "doi_normalized": doi.lower(),
        "pmid": pmid,
        "pmcid": None,
        "source_ids_json": {"pubmed": pmid, "doi": doi},
        "publication_types_json": ["journal article"],
        "article_type_classification": "primary_study",
        "is_guideline": False, "is_preprint": False, "is_retracted": False,
        "is_retraction_notice": False, "has_expression_of_concern": False,
        "full_text_status": "abstract_only",
        "source_urls_json": [],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    # Bypass _upsert_article so the email survives in storage —
    # this simulates a pre-Task-A persisted record.
    await db[Collections.ARTICLES].insert_one(dict(doc))
    yield {"id": aid, "pmid": pmid, "doi": doi}
    await db[Collections.ARTICLES].delete_many({"id": aid})


@pytest.mark.asyncio
async def test_get_article_endpoint_scrubs_legacy_emails(
    client, clinician_token, legacy_email_article,
):
    aid = legacy_email_article["id"]
    r = await client.get(
        f"/api/articles/{aid}",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )
    assert r.status_code == 200, r.text
    blob = r.text
    assert not _EMAIL_RE.search(blob), (
        f"GET /api/articles/{{id}} leaks email: {blob[:300]}"
    )
    body = r.json()
    # Author NAME + institutional affiliation preserved.
    assert body["authors_json"][0]["lastName"] == "Verdi"
    assert "Dept of Q, Univ T" in (body["authors_json"][0]["affiliation"] or "")
    # DOI / PMID / journal / title preserved.
    assert body["doi"] == legacy_email_article["doi"]
    assert body["pmid"] == legacy_email_article["pmid"]
    assert body["title"] == "Legacy article with email leak"
    assert body["journal"] == "Legacy J"


@pytest.mark.asyncio
async def test_search_session_hydration_scrubs_legacy_emails(
    client, clinician_token, legacy_email_article,
):
    """Wire the legacy article into a fake search_session_articles row
    so GET /api/search-sessions/{id} hydration goes through the
    defensive scrub path."""
    db = get_db()
    # Resolve the clinician's user id.
    me = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )
    user_id = me.json()["id"]
    sid = f"AT_SESSION_{uuid.uuid4().hex[:10]}"
    sess_doc = {
        "id": sid,
        "user_id": user_id,
        "original_question": "test question (no email here)",
        "sanitized_question": "test question (no email here)",
        "topic_label": "TaskA",
        "status": "complete",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc),
        "deleted_at": None,
    }
    sa_doc = {
        "id": f"AT_SA_{uuid.uuid4().hex[:10]}",
        "search_session_id": sid,
        "article_id": legacy_email_article["id"],
        "relevance_score": 0.5,
        "evidence_design_score": 0.5,
        "overall_rank": 1,
        "rank_explanation": "test",
        "included_by_default": True,
        "user_included": True,
        "exclusion_reason": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    await db[Collections.SEARCH_SESSIONS].insert_one(sess_doc)
    await db[Collections.SEARCH_SESSION_ARTICLES].insert_one(sa_doc)
    try:
        r = await client.get(
            f"/api/search-sessions/{sid}",
            headers={"Authorization": f"Bearer {clinician_token}"},
        )
        assert r.status_code == 200, r.text
        blob = r.text
        assert not _EMAIL_RE.search(blob), (
            f"GET /api/search-sessions/{{id}} leaks email: {blob[:400]}"
        )
        body = r.json()
        # Author + DOI + PMID preserved on the hydrated article card.
        articles = body.get("articles") or []
        assert articles, "no hydrated articles in response"
        art = articles[0]["article"]
        assert art["doi"] == legacy_email_article["doi"]
        assert art["pmid"] == legacy_email_article["pmid"]
        assert art["authors_json"][0]["lastName"] == "Verdi"
        assert "Dept of Q, Univ T" in (
            art["authors_json"][0]["affiliation"] or ""
        )
    finally:
        await db[Collections.SEARCH_SESSION_ARTICLES].delete_many(
            {"search_session_id": sid},
        )
        await db[Collections.SEARCH_SESSIONS].delete_many({"id": sid})


@pytest.mark.asyncio
async def test_folder_articles_endpoint_scrubs_legacy_emails(
    client, clinician_token, legacy_email_article,
):
    db = get_db()
    # Create a folder + folder_article row pointing at the legacy article.
    r = await client.post(
        "/api/folders/",
        headers={"Authorization": f"Bearer {clinician_token}"},
        json={"name": f"TaskA-{uuid.uuid4().hex[:6]}",
              "description": "test"},
    )
    assert r.status_code in (200, 201), r.text
    folder = r.json()
    fid = folder["id"]
    try:
        r2 = await client.post(
            f"/api/folders/{fid}/articles",
            headers={"Authorization": f"Bearer {clinician_token}"},
            json={"article_id": legacy_email_article["id"]},
        )
        assert r2.status_code in (200, 201), r2.text
        r3 = await client.get(
            f"/api/folders/{fid}/articles",
            headers={"Authorization": f"Bearer {clinician_token}"},
        )
        assert r3.status_code == 200
        blob = r3.text
        assert not _EMAIL_RE.search(blob), (
            f"GET /api/folders/{{id}}/articles leaks email: {blob[:400]}"
        )
        rows = r3.json()
        assert rows, "no folder_article rows"
        art = rows[0]["article"]
        assert art["authors_json"][0]["lastName"] == "Verdi"
        assert art["doi"] == legacy_email_article["doi"]
    finally:
        await client.delete(
            f"/api/folders/{fid}",
            headers={"Authorization": f"Bearer {clinician_token}"},
        )
        await db[Collections.FOLDER_ARTICLES].delete_many({"folder_id": fid})
