"""Phase 6D — canonical_identity unit tests."""
from __future__ import annotations

import pytest

from services.canonical_identity import (
    make_canonical_key,
    normalize_doi,
    normalize_trial_id,
)


class TestNormalizeDoi:
    def test_strips_https_doi_org_prefix(self):
        assert (
            normalize_doi("https://doi.org/10.1056/NEJMOA2207533")
            == "10.1056/nejmoa2207533"
        )

    def test_strips_https_dx_doi_org_prefix(self):
        assert (
            normalize_doi("https://dx.doi.org/10.1136/BMJ.M158")
            == "10.1136/bmj.m158"
        )

    def test_strips_doi_colon_prefix(self):
        assert normalize_doi("doi:10.1001/jama.2020.1234") == "10.1001/jama.2020.1234"

    def test_case_insensitive_prefix(self):
        assert (
            normalize_doi("DOI: 10.1001/JAMA.2020.1234")
            == "10.1001/jama.2020.1234"
        )

    def test_returns_empty_for_garbage(self):
        assert normalize_doi("n/a") == ""
        assert normalize_doi("") == ""
        assert normalize_doi(None) == ""
        assert normalize_doi("not-a-doi") == ""

    def test_requires_10_dot_prefix(self):
        # Doesn't look like a DOI — return empty.
        assert normalize_doi("12.3456/foo") == ""


class TestNormalizeTrialId:
    def test_uppercase_nct(self):
        assert normalize_trial_id("nct01234567") == "NCT01234567"

    def test_strips_whitespace(self):
        assert normalize_trial_id("  NCT01234567  ") == "NCT01234567"

    def test_rejects_non_nct(self):
        assert normalize_trial_id("EUDRACT2020-001234") == ""
        assert normalize_trial_id("ISRCTN12345") == ""

    def test_rejects_short_nct(self):
        assert normalize_trial_id("NCT123") == ""

    def test_rejects_empty(self):
        assert normalize_trial_id("") == ""
        assert normalize_trial_id(None) == ""


class TestMakeCanonicalKey:
    def test_pmid_wins(self):
        assert make_canonical_key(
            pmid="12345", doi="10.1/foo", pmcid="PMC999",
            trial_id="NCT01234567",
        ) == "pmid:12345"

    def test_doi_when_no_pmid(self):
        assert make_canonical_key(
            doi="https://doi.org/10.1056/NEJMOA2207533",
            pmcid="PMC999",
        ) == "doi:10.1056/nejmoa2207533"

    def test_pmcid_when_no_pmid_no_doi(self):
        assert make_canonical_key(pmcid="PMC12345") == "pmcid:PMC12345"

    def test_pmcid_adds_prefix_if_missing(self):
        assert make_canonical_key(pmcid="12345") == "pmcid:PMC12345"

    def test_trial_id_rung(self):
        """Phase 6D — NCT rung between PMCID and title-fallback."""
        assert make_canonical_key(
            trial_id="NCT01234567",
            title="some title", year=2024,
            journal="Lancet", first_author="Smith",
        ) == "trial:NCT01234567"

    def test_title_fallback(self):
        ck = make_canonical_key(
            title="A great paper", year=2024,
            journal="NEJM", first_author="Smith",
        )
        assert ck.startswith("title_year_journal_firstauthor:")
        assert "a_great_paper" in ck
        assert "2024" in ck
        assert "nejm" in ck
        assert "smith" in ck

    def test_pmid_strips_non_digit(self):
        assert make_canonical_key(pmid="PMID:12345") == "pmid:12345"


class TestIdentifierLayeringWithDoiPrefixVariants:
    @pytest.mark.parametrize(
        "raw_doi",
        [
            "10.1056/NEJMOA2207533",
            "https://doi.org/10.1056/NEJMOA2207533",
            "https://dx.doi.org/10.1056/NEJMOA2207533",
            "doi:10.1056/NEJMOA2207533",
            "  10.1056/nejmoa2207533  ",
        ],
    )
    def test_all_doi_variants_normalize_identically(self, raw_doi):
        assert normalize_doi(raw_doi) == "10.1056/nejmoa2207533"
        assert (
            make_canonical_key(doi=raw_doi)
            == "doi:10.1056/nejmoa2207533"
        )
