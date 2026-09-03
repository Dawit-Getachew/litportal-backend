"""Phase 2C — Institutional link generators + best-access logic."""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlparse, unquote


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.full_text_access_broker import (  # noqa: E402
    build_openathens_row,
    build_openurl_row,
    build_semantic_scholar_institutional_row,
    compute_best_access,
    institutional_rows_for_user,
)


# ---------- OpenAthens ----------
def test_openathens_link_percent_encodes_target_doi():
    art = {"id": "a1", "doi": "10.1056/NEJMoa2107038"}
    row = build_openathens_row(art, customer_domain="mock-university")
    assert row is not None
    assert row["access_category"] == "institutional_openathens"
    # The DOI target must be percent-encoded inside the url param.
    parsed = urlparse(row["url"])
    qs = dict(parse_qsl(parsed.query))
    assert "url" in qs
    # The decoded url should match the DOI URL.
    assert unquote(qs["url"]) == "https://doi.org/10.1056/NEJMoa2107038"


def test_openathens_link_uses_pubmed_when_no_doi():
    art = {"id": "a2", "doi": None, "pmid": "12345"}
    row = build_openathens_row(art, customer_domain="mock-u")
    parsed = urlparse(row["url"])
    qs = dict(parse_qsl(parsed.query))
    assert "pubmed.ncbi.nlm.nih.gov/12345" in unquote(qs["url"])


def test_openathens_row_requires_login_true_can_ingest_false():
    art = {"id": "a3", "doi": "10.1/x"}
    row = build_openathens_row(art, customer_domain="mock-u")
    assert row["requires_login"] is True
    assert row["is_oa"] is False
    assert row["can_ingest_for_summary"] is False
    assert row["can_view_in_app"] is False
    assert row["evidence_text_level"] == "institutional_linkout"


def test_openathens_returns_none_without_target():
    art = {"id": "a4", "doi": None, "pmid": None, "source_urls_json": []}
    assert build_openathens_row(art, customer_domain="x") is None


# ---------- OpenURL ----------
def test_openurl_includes_doi_pmid_title_journal_year_authors():
    art = {
        "id": "b1",
        "doi": "10.1056/NEJMoa2107038",
        "pmid": "34449189",
        "title": "Empagliflozin in HFpEF",
        "journal": "NEJM",
        "publication_year": 2021,
        "authors_json": [{"lastName": "Anker"}],
    }
    row = build_openurl_row(art, resolver_base="https://library.example/openurl")
    qs = dict(parse_qsl(urlparse(row["url"]).query))
    assert qs["rft.atitle"] == "Empagliflozin in HFpEF"
    assert qs["rft.jtitle"] == "NEJM"
    assert qs["rft.date"] == "2021"
    assert qs["rft.aulast"] == "Anker"
    # Both rft_id entries are present (parse_qsl preserves them as one of dup).
    raw_query = urlparse(row["url"]).query
    assert "rft_id=info%3Adoi%2F10.1056%2FNEJMoa2107038" in raw_query
    assert "rft_id=info%3Apmid%2F34449189" in raw_query


def test_openurl_includes_url_ver_and_rft_val_fmt():
    art = {"id": "b2", "doi": "10.1/x", "title": "t", "publication_year": 2020}
    row = build_openurl_row(art, resolver_base="https://library.example/openurl")
    raw_query = urlparse(row["url"]).query
    assert "url_ver=Z39.88-2004" in raw_query
    assert "rft_val_fmt=info%3Aofi%2Ffmt%3Akev%3Amtx%3Ajournal" in raw_query


def test_openurl_row_requires_login_true_can_ingest_false():
    art = {"id": "b3", "doi": "10.1/x", "title": "t"}
    row = build_openurl_row(art, resolver_base="https://library.example/openurl")
    assert row["requires_login"] is True
    assert row["can_ingest_for_summary"] is False
    assert row["access_category"] == "institutional_openurl"


# ---------- Semantic Scholar institutional ----------
def test_semantic_scholar_institutional_link_when_enabled():
    art = {"id": "c1", "doi": "10.1/x", "title": "t"}
    row = build_semantic_scholar_institutional_row(art)
    assert row["access_category"] == "semantic_scholar_institutional_linkout"
    assert "DOI:10.1/x" in row["url"] or "10.1/x" in row["url"]
    assert row["requires_login"] is True


# ---------- institutional_rows_for_user ----------
def test_institutional_rows_for_user_respects_settings():
    art = {"id": "d1", "doi": "10.1/x", "title": "t"}
    # No settings → no rows
    assert institutional_rows_for_user(art, None) == []
    # OpenAthens only
    rows = institutional_rows_for_user(art, {
        "openathens_customer_domain": "mock-u",
        "openurl_resolver_base_url": None,
        "semantic_scholar_fallback_enabled": False,
    })
    assert len(rows) == 1
    assert rows[0]["access_category"] == "institutional_openathens"
    # All three
    rows = institutional_rows_for_user(art, {
        "openathens_customer_domain": "mock-u",
        "openurl_resolver_base_url": "https://library.example/openurl",
        "semantic_scholar_fallback_enabled": True,
    })
    cats = {r["access_category"] for r in rows}
    assert cats == {
        "institutional_openathens",
        "institutional_openurl",
        "semantic_scholar_institutional_linkout",
    }


# ============================================================================
# compute_best_access — priority levels
# ============================================================================
def _row(**overrides):
    base = {
        "source": "x", "access_category": "unknown", "url": None,
        "is_direct_pdf": False, "is_oa": False, "requires_login": False,
        "can_view_in_app": False, "can_ingest_for_summary": False,
        "license": None, "version": None, "evidence_text_level": "unknown",
    }
    base.update(overrides)
    return base


def test_best_access_priority_open_pdf():
    art = {"doi": "10.1/x"}
    locs = [
        _row(source="unpaywall", access_category="oa_pdf", url="https://u/oa.pdf",
             is_direct_pdf=True, is_oa=True, can_view_in_app=True),
        _row(source="pmc", access_category="pmc_oa", url="https://pmc/foo",
             is_direct_pdf=True, is_oa=True, can_view_in_app=True),
    ]
    out = compute_best_access(art, locs)
    # PMC wins over unpaywall in PDF priority.
    assert out["button_label"] == "Open PDF"
    assert "pmc" in out["target_url"]


def test_best_access_priority_open_full_text():
    art = {"doi": "10.1/x"}
    locs = [
        _row(source="pmc", access_category="pmc_oa", url="https://pmc/foo",
             is_oa=True, can_view_in_app=True),
    ]
    out = compute_best_access(art, locs)
    assert out["button_label"] == "Open Full Text"


def test_best_access_priority_free_full_text():
    art = {"doi": None}
    locs = [
        _row(source="unpaywall", access_category="free_full_text_link",
             url="https://repo.example/x", is_oa=True),
    ]
    out = compute_best_access(art, locs)
    assert out["button_label"] == "Free Full Text"


def test_best_access_priority_open_via_institution():
    art = {"doi": "10.1/x"}
    locs = [
        _row(source="openathens", access_category="institutional_openathens",
             url="https://go.openathens.net/redirector/mock-u?url=https%3A%2F%2Fdoi.org%2F10.1%2Fx",
             requires_login=True),
    ]
    out = compute_best_access(art, locs)
    assert out["button_label"] == "Open via Institution"
    assert "ClinEvidence does not store institutional credentials" in out["helper_text"]


def test_best_access_priority_get_full_text_with_doi_only():
    art = {"doi": "10.1/x"}
    locs = []
    out = compute_best_access(art, locs)
    assert out["button_label"] == "Get Full Text"
    assert out["target_url"] == "https://doi.org/10.1/x"


def test_best_access_priority_semantic_scholar_institutional():
    art = {"doi": None}
    locs = [
        _row(source="semantic_scholar",
             access_category="semantic_scholar_institutional_linkout",
             url="https://www.semanticscholar.org/paper/x", requires_login=True),
    ]
    out = compute_best_access(art, locs)
    assert out["button_label"] == "Open in Semantic Scholar"


def test_best_access_priority_open_doi():
    art = {"doi": None}
    locs = [
        _row(source="crossref", access_category="publisher_doi_link",
             url="https://doi.org/10.1/x"),
    ]
    out = compute_best_access(art, locs)
    assert out["button_label"] == "Open DOI"


def test_best_access_abstract_only_disabled():
    art = {"doi": None}
    locs = [
        _row(source="article", access_category="abstract_only",
             url=None, evidence_text_level="abstract_only"),
    ]
    out = compute_best_access(art, locs)
    assert out["disabled"] is True
    assert "Settings" in out["helper_text"]


def test_best_access_unknown_disabled_when_nothing():
    art = {}
    out = compute_best_access(art, [])
    assert out["disabled"] is True
    assert out["button_label"] == "Access Unknown"
