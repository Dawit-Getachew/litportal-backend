"""W1.3 — publication identity, study/report linking, dates, material hashing (pure)."""
from __future__ import annotations

import pytest

from clinical_engine.contracts.question import ClinicalQuestionMap, QuestionElement, material_hash
from clinical_engine.identity.dates import parse_publication_date
from clinical_engine.identity.resolver import (
    normalize_doi_strict,
    normalize_pmcid,
    normalize_pmid,
    report_role,
    resolve,
    titles_compatible,
)

TRIAL = "Hydrocortisone plus fludrocortisone for adults with septic shock"


def _rec(rid, source="pubmed", **kw):
    return {"record_id": rid, "source": source, **kw}


# ── identifiers ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("29490185", "29490185"), ("PMID: 29490185", "29490185"), (" 29490185 ", "29490185"),
    ("0", ""), ("abc", ""), ("1234567890", ""), (None, ""),
])
def test_normalize_pmid(raw, expected):
    assert normalize_pmid(raw) == expected


def test_normalize_doi_and_pmcid():
    assert normalize_doi_strict("https://doi.org/10.1056/NEJMoa1705716.") == "10.1056/nejmoa1705716"
    assert normalize_doi_strict("not a doi") == ""
    assert normalize_pmcid("pmc123") == "PMC123"
    assert normalize_pmcid("123") == "PMC123"
    assert normalize_pmcid("PMCX") == ""


def test_titles_compatible():
    assert titles_compatible(TRIAL, TRIAL.upper() + ".")
    assert titles_compatible(TRIAL + ": a randomized trial", TRIAL)
    assert not titles_compatible(TRIAL, "Vitamin D supplementation and fracture risk in older women")


# ── resolution ──────────────────────────────────────────────────────────────


def test_same_publication_from_two_sources_is_merged_with_authoritative_fields():
    report = resolve([
        _rec("a", "openalex", doi="10.1056/NEJMoa1705716", title=TRIAL.lower(), year=2018,
             abstract="x" * 5000),
        _rec("b", "pubmed", pmid="29490185", doi="https://doi.org/10.1056/NEJMoa1705716", title=TRIAL,
             year=2018, abstract="Official abstract."),
    ])
    assert report.merged_groups == 1
    assert len(report.publications) == 1
    pub = report.publications[0]
    assert pub.identity_status == "verified"
    assert pub.identifier("pmid") == "29490185"
    # Fields come from the most authoritative source, never from the longest value.
    assert pub.abstract == "Official abstract."
    assert pub.abstract_provenance.source == "pubmed"
    assert pub.title == TRIAL


def test_grafted_identifier_is_quarantined_not_merged():
    """A record carrying another paper's PMID must not produce a mixed record."""
    report = resolve([
        _rec("a", "pubmed", pmid="29490185", doi="10.1056/NEJMoa1705716", title=TRIAL, year=2018),
        _rec("b", "openalex", pmid="29490185", doi="10.1016/j.other.2011.01.001",
             title="Vitamin D supplementation and fracture risk in older women", year=2011),
    ])
    assert report.merged_groups == 0
    assert sorted(report.quarantined_record_ids) == ["a", "b"]
    assert len(report.publications) == 2
    assert all(p.identity_status == "quarantined" for p in report.publications)
    reasons = " ".join(c["reason"] for c in report.conflicts[0]["conflicts"])
    assert "disagree on doi" in reasons and "different titles" in reasons
    # Neither record borrowed the other's fields.
    titles = {p.merged_from[0]: p.title for p in report.publications}
    assert titles["a"] == TRIAL


def test_transitive_identifiers_form_one_cluster():
    report = resolve([
        _rec("a", "pubmed", pmid="111", title=TRIAL, year=2018),
        _rec("b", "europepmc", pmid="111", pmcid="PMC999", title=TRIAL, year=2018),
        _rec("c", "openalex", pmcid="PMC999", doi="10.1/abc", title=TRIAL, year=2018),
    ])
    assert len(report.publications) == 1
    assert report.publications[0].identifier("doi") == "10.1/abc"
    assert sorted(report.publications[0].merged_from) == ["a", "b", "c"]


def test_year_mismatch_blocks_merge():
    report = resolve([
        _rec("a", "pubmed", doi="10.1/x", title=TRIAL, year=2018),
        _rec("b", "openalex", doi="10.1/x", title=TRIAL, year=2012),
    ])
    assert report.quarantined_record_ids == ["a", "b"]


def test_invalid_identifiers_are_dropped_not_matched():
    report = resolve([
        _rec("a", "openalex", pmid="n/a", doi="none", title="Paper A"),
        _rec("b", "openalex", pmid="n/a", doi="none", title="Paper B"),
    ])
    assert len(report.publications) == 2  # junk ids never link two papers
    assert report.merged_groups == 0


def test_separate_reports_of_one_trial_stay_separate_and_are_linked():
    report = resolve([
        _rec("main", "pubmed", pmid="29490185", nct="NCT00625209", title=TRIAL, year=2018,
             publication_types=["Randomized Controlled Trial"]),
        _rec("sub", "pubmed", pmid="31234567", nct="nct00625209",
             title="Hydrocortisone plus fludrocortisone in septic shock: a post hoc subgroup analysis", year=2020),
    ])
    assert len(report.publications) == 2
    assert report.merged_groups == 0
    assert len(report.studies) == 1
    study = report.studies[0]
    assert study.registration_ids == ["NCT00625209"]
    roles = sorted(r.report_role for r in study.reports)
    assert roles == ["primary_result", "subgroup_analysis"]


def test_report_role_cues():
    pub = resolve([_rec("p", "pubmed", pmid="1", title="Study protocol for the X trial")]).publications[0]
    assert report_role(pub) == "protocol"


# ── dates ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,year,month,day,precision", [
    ("2019", 2019, None, None, "year"),
    ("2019 Mar", 2019, 3, None, "month"),
    ("2019 Mar 15", 2019, 3, 15, "day"),
    ("2019-03-15", 2019, 3, 15, "day"),
    ("2019-03", 2019, 3, None, "month"),
    ("2019 Spring", 2019, None, None, "season"),
    ("2019 Mar-Apr", 2019, None, None, "season"),
    ("Epub 2020 Jan 5", 2020, 1, 5, "day"),
    ("", None, None, None, "unknown"),
    ("in press", None, None, None, "unknown"),
])
def test_date_precision(raw, year, month, day, precision):
    d = parse_publication_date(raw)
    assert (d.year, d.month, d.day, d.precision) == (year, month, day, precision)


# ── material hash ───────────────────────────────────────────────────────────


def _map(**kw):
    base = dict(question_text="q", question_type="treatment", framework="PICO", elements=[
        QuestionElement(role="intervention", value="hydrocortisone", source="user_text"),
        QuestionElement(role="condition", value="septic shock", source="user_text"),
        QuestionElement(role="comparator", value="placebo", source="ai_inferred", active=False),
    ])
    base.update(kw)
    return ClinicalQuestionMap(**base)


def test_material_hash_ignores_inactive_suggestions_and_case():
    a = _map()
    b = _map(elements=[
        QuestionElement(role="condition", value="Septic  Shock", source="user_edited"),
        QuestionElement(role="intervention", value="Hydrocortisone", source="user_text"),
    ])
    assert material_hash(a) == material_hash(b)


def test_material_hash_changes_on_material_edit():
    a = _map()
    b = _map(elements=[*_map().elements, QuestionElement(role="age_group", value="adult", source="user_edited")])
    c = _map(question_type="prevention")
    assert len({material_hash(a), material_hash(b), material_hash(c)}) == 3
