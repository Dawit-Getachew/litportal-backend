"""SCOPE AUDIT — B8/B15 full-text-import decision matrix + B11 canonical identity.

Pure-logic, no app boot / DB / network. Exhaustively exercises:
  * FullTextImportService._top_status across the full (oa_status × can_ingest ×
    access_category × abstract) space (B8/B15 status contract).
  * _confidence_ok gating + _build_lookup_text identifier precedence (B15).
  * make_canonical_key priority + normalize_doi normalization (B11).
"""
from __future__ import annotations

import itertools

import pytest

import services.full_text_import as fti
from services.canonical_identity import make_canonical_key, normalize_doi

# ════════════════════════════════════════════════════════════════════
# GROUP D — _top_status full matrix
# ════════════════════════════════════════════════════════════════════
OA_STATUSES = ["fetched", "cache_hit", "skipped_license", "skipped_ineligible", "failed", "disabled"]
CATS = ["oa_pdf", "oa_html", "free_full_text_link", "institutional_openathens",
        "publisher_doi_link", "abstract_only", "unknown", None]
_LINKOUT_CATS = {"oa_pdf", "oa_html", "free_full_text_link",
                 "institutional_openathens", "publisher_doi_link"}

_D_CASES = list(itertools.product(OA_STATUSES, [True, False], CATS, [True, False]))


def _ref_status(ingested, cat, disabled, abstract):
    if ingested:
        return "full_text_ingested"
    if (not disabled) and cat not in (None, "abstract_only", "unknown"):
        return "full_text_link_out"
    if abstract:
        return "abstract_only"
    return "metadata_only"


@pytest.mark.parametrize("oa_status,can_ingest,cat,abstract", _D_CASES)
def test_b8_top_status_matrix(oa_status, can_ingest, cat, abstract):
    ingested = oa_status in ("fetched", "cache_hit") and can_ingest
    disabled = cat not in _LINKOUT_CATS  # compute_best_access disables abstract/unknown/none
    best = {"access_category": cat, "disabled": disabled}
    status = fti.FullTextImportService._top_status(
        ingested=ingested, best_access=best, abstract_present=abstract,
    )
    assert status == _ref_status(ingested, cat, disabled, abstract)
    assert status in {"full_text_ingested", "full_text_link_out", "abstract_only", "metadata_only"}


@pytest.mark.parametrize("oa_status,can_ingest,cat,abstract", _D_CASES)
def test_b8_ingested_requires_oa_success_and_license(oa_status, can_ingest, cat, abstract):
    """full_text_ingested ONLY when OA fetch succeeded AND license permits ingest."""
    ingested = oa_status in ("fetched", "cache_hit") and can_ingest
    disabled = cat not in _LINKOUT_CATS
    status = fti.FullTextImportService._top_status(
        ingested=ingested, best_access={"access_category": cat, "disabled": disabled},
        abstract_present=abstract,
    )
    if status == "full_text_ingested":
        assert ingested is True


# ════════════════════════════════════════════════════════════════════
# GROUP D2 — confidence gating
# ════════════════════════════════════════════════════════════════════
_CONF_CASES = list(itertools.product(
    ["high", "medium", "low", None, "garbage"], ["high", "medium", "low"],
))


@pytest.mark.parametrize("conf,minimum", _CONF_CASES)
def test_b15_confidence_ok(conf, minimum):
    rank = {"high": 0, "medium": 1, "low": 2}
    expected = rank.get(conf or "low", 3) <= rank.get(minimum, 1)
    assert fti._confidence_ok(conf, minimum) is expected


# ════════════════════════════════════════════════════════════════════
# GROUP D3 — identifier precedence for the resolver lookup string
# ════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("pmid,doi,pmcid,title,expected", [
    ("123456", "10.1/x", "PMC9", "A title", "PMID: 123456"),
    (None, "10.1/x", "PMC9", "A title", "10.1/x"),
    (None, None, "PMC9", "A title", "PMC9"),
    (None, None, None, "A title", "A title"),
    (None, None, None, None, None),
    ("  123  ", None, None, None, "PMID: 123"),
    ("", "10.2/y", None, None, "10.2/y"),
    ("", "", "PMC1", None, "PMC1"),
    ("", "", "", "T", "T"),
    ("", "", "", "", None),
])
def test_b15_build_lookup_text(pmid, doi, pmcid, title, expected):
    assert fti._build_lookup_text(pmid=pmid, doi=doi, pmcid=pmcid, title=title) == expected


@pytest.mark.parametrize("pmid", ["1", "39111111", "  77  "])
def test_b15_bare_pmid_is_labelled_for_phi_stripper(pmid):
    # A bare PMID must be emitted labelled so the resolver's PHI stripper removes
    # it before screening (this was the live-caught bug).
    out = fti._build_lookup_text(pmid=pmid, doi=None, pmcid=None, title=None)
    assert out.startswith("PMID: ")
    assert pmid.strip() in out


# ════════════════════════════════════════════════════════════════════
# GROUP J — B11 make_canonical_key priority
# ════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("pmid", ["123", "39111111", "7"])
def test_canonical_key_pmid_wins(pmid):
    k = make_canonical_key(pmid=pmid, doi="10.1056/x", pmcid="PMC1", title="T", year=2020,
                           journal="NEJM", first_author="Smith")
    assert k == f"pmid:{pmid}"


@pytest.mark.parametrize("doi,norm", [
    ("10.1056/NEJMoa2034577", "10.1056/nejmoa2034577"),
    ("https://doi.org/10.1001/jama.2020.1", "10.1001/jama.2020.1"),
    ("doi:10.1136/bmj.m1", "10.1136/bmj.m1"),
])
def test_canonical_key_doi_when_no_pmid(doi, norm):
    k = make_canonical_key(pmid=None, doi=doi, pmcid="PMC1", title="T", year=2020,
                           journal="NEJM", first_author="Smith")
    assert k == f"doi:{norm}"


@pytest.mark.parametrize("pmcid,expected", [
    ("PMC123", "PMC123"), ("pmc456", "PMC456"), ("PMC7092803", "PMC7092803"),
])
def test_canonical_key_pmcid_when_no_pmid_doi(pmcid, expected):
    k = make_canonical_key(pmid=None, doi=None, pmcid=pmcid, title="T", year=2020,
                           journal="NEJM", first_author="Smith")
    assert k == f"pmcid:{expected}"


@pytest.mark.parametrize("title,year,journal,author", [
    ("Empagliflozin in HFpEF", 2024, "NEJM", "Anker"),
    ("Sepsis bundles", 2019, "JAMA", "Rhodes"),
    ("A study", None, None, None),
])
def test_canonical_key_title_fallback(title, year, journal, author):
    k = make_canonical_key(pmid=None, doi=None, pmcid=None, title=title, year=year,
                           journal=journal, first_author=author)
    assert k.startswith("title_year_journal_firstauthor:")


@pytest.mark.parametrize("pmid,doi,pmcid", list(itertools.product(
    [None, "555"], [None, "10.1/zz"], [None, "PMC5"],
)))
def test_canonical_key_priority_order(pmid, doi, pmcid):
    k = make_canonical_key(pmid=pmid, doi=doi, pmcid=pmcid, title="T", year=2000,
                           journal="J", first_author="A")
    if pmid:
        assert k.startswith("pmid:")
    elif doi:
        assert k.startswith("doi:")
    elif pmcid:
        assert k.startswith("pmcid:")
    else:
        assert k.startswith("title_year_journal_firstauthor:")


# ════════════════════════════════════════════════════════════════════
# GROUP K — B11 normalize_doi
# ════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("raw,expected", [
    ("10.1056/NEJMoa2034577", "10.1056/nejmoa2034577"),
    ("https://doi.org/10.1056/NEJMoa2034577", "10.1056/nejmoa2034577"),
    ("https://dx.doi.org/10.1056/Foo", "10.1056/foo"),
    ("doi:10.1136/bmj.m1", "10.1136/bmj.m1"),
    ("  10.1001/JAMA.2020.1  ", "10.1001/jama.2020.1"),
    ("not-a-doi", ""),
    ("", ""),
    (None, ""),
    ("10x/broken", ""),
    ("https://example.com/foo", ""),
])
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected
