"""Phase 4A — license gate + eligibility (Task 11 / A)."""
from __future__ import annotations

import os
import pytest

from services.oa_fetcher.license_gate import (
    determine_ingest_permission,
    is_article_eligible,
    normalize_license,
)


def _perm(license=None, source="europe_pmc_oa", access_category="europe_pmc_oa", metadata=None):
    return determine_ingest_permission(
        source=source, license=license,
        access_category=access_category, metadata=metadata,
    )


def test_cc_by_allowed():
    r = _perm(license="https://creativecommons.org/licenses/by/4.0/")
    assert r["can_ingest_for_summary"] is True
    assert r["license"] == "cc-by"
    assert r["license_category"] == "commercial_reuse_allowed"


def test_cc_0_allowed():
    r = _perm(license="CC0")
    assert r["can_ingest_for_summary"] is True
    assert r["license"] == "cc0"
    assert r["license_category"] == "commercial_reuse_allowed"


def test_cc_by_sa_allowed():
    r = _perm(license="cc-by-sa")
    assert r["can_ingest_for_summary"] is True
    assert r["license_category"] == "commercial_reuse_allowed"


def test_cc_by_nc_skipped_by_default(monkeypatch):
    monkeypatch.delenv("OA_ALLOW_NONCOMMERCIAL_LICENSES", raising=False)
    r = _perm(license="creativecommons.org/licenses/by-nc/4.0/")
    assert r["can_ingest_for_summary"] is False
    assert r["license"] == "cc-by-nc"
    assert r["license_category"] == "noncommercial_only"


def test_cc_by_nc_allowed_when_env_flag_true(monkeypatch):
    monkeypatch.setenv("OA_ALLOW_NONCOMMERCIAL_LICENSES", "true")
    r = _perm(license="by-nc-sa")
    assert r["can_ingest_for_summary"] is True
    assert r["license_category"] == "noncommercial_only"


def test_cc_by_nd_skipped_by_default(monkeypatch):
    monkeypatch.delenv("OA_ALLOW_NO_DERIVATIVES_LICENSES", raising=False)
    r = _perm(license="cc-by-nd")
    assert r["can_ingest_for_summary"] is False
    assert r["license_category"] == "restricted"


def test_cc_by_nc_nd_skipped_by_default(monkeypatch):
    monkeypatch.delenv("OA_ALLOW_NONCOMMERCIAL_LICENSES", raising=False)
    r = _perm(license="cc-by-nc-nd")
    assert r["can_ingest_for_summary"] is False
    assert r["license_category"] == "noncommercial_only"


def test_unknown_license_skipped():
    r = _perm(license="proprietary-foo")
    assert r["can_ingest_for_summary"] is False
    assert r["license_category"] == "unknown"
    assert r["license"] is None


def test_institutional_linkout_overridden_to_not_ingestible():
    r = _perm(license="cc-by", access_category="institutional_openathens", source="openathens")
    assert r["can_ingest_for_summary"] is False
    assert r["license_category"] == "restricted"


def test_publisher_doi_link_not_ingestible():
    r = _perm(license="cc-by", access_category="publisher_doi_link", source="crossref")
    assert r["can_ingest_for_summary"] is False


def test_user_uploaded_full_text_gate_does_not_apply():
    r = _perm(
        license="some-publisher-tos",
        access_category="user_uploaded_full_text",
        source="user_uploaded_full_text",
        metadata={"existing_can_ingest": True},
    )
    assert r["can_ingest_for_summary"] is True
    assert r["license"] == "user_attested"


def test_preprint_excluded_by_default(monkeypatch):
    monkeypatch.delenv("OA_ALLOW_PREPRINTS", raising=False)
    ok, reason = is_article_eligible({
        "is_preprint": True, "pmcid": "PMC123",
    })
    assert ok is False
    assert reason == "article_is_preprint"


def test_preprint_allowed_when_env_flag_true(monkeypatch):
    monkeypatch.setenv("OA_ALLOW_PREPRINTS", "true")
    ok, reason = is_article_eligible({"is_preprint": True, "pmcid": "PMC123"})
    assert ok is True


def test_retracted_never_fetched():
    ok, reason = is_article_eligible({"is_retracted": True, "pmcid": "PMC9"})
    assert ok is False
    assert reason == "article_retracted"


def test_retraction_notice_never_fetched():
    ok, reason = is_article_eligible({"is_retraction_notice": True, "pmcid": "PMC9"})
    assert ok is False
    assert reason == "article_retraction_notice"


def test_eligible_when_has_pmcid():
    ok, reason = is_article_eligible({"pmcid": "PMC9", "is_retracted": False})
    assert ok is True


def test_ineligible_when_no_oa_signal():
    ok, reason = is_article_eligible({"pmcid": "", "source_ids_json": {}})
    assert ok is False
    assert reason == "no_oa_signal"


def test_eligible_via_europe_pmc_inEPMC_flag():
    ok, _ = is_article_eligible({
        "pmcid": "", "source_ids_json": {"inEPMC": "Y"},
    })
    assert ok is True


def test_normalize_license_handles_urls_and_versions():
    assert normalize_license("https://creativecommons.org/licenses/by/4.0/") == "cc-by"
    assert normalize_license("BY-NC-4.0") == "cc-by-nc"
    assert normalize_license("cc-by-sa-3.0") == "cc-by-sa"
    assert normalize_license("not-a-license") is None
    assert normalize_license(None) is None
