"""Phase 3 — summarizer output validation + mechanical floor (Task 11 / C).

We unit-test the validator pipeline via the verifier mechanical floor directly,
since the API path always invokes the verifier after a draft is produced.
"""
from __future__ import annotations

import pytest

from services.summarization.evidence_spans import EvidenceSpan
from services.summarization.schemas import (
    ClaimMapEntry, DraftSummary, SelectedStudyEntry,
)
from services.summarization.verifier import MechanicalVerificationFloor


def _span(article_id, idx=0, level="abstract", section="abstract", text="evidence text"):
    return EvidenceSpan(
        evidence_span_id=f"span:{article_id}:{idx}",
        article_id=article_id, search_session_id="sid",
        section_name=section, text_source=level, evidence_text_level=level,
        start_char=0, end_char=len(text), span_text=text,
        citation_label=f"PMID:1{article_id[:6]}",
        pmid=f"1{article_id[:6]}", doi=f"10.1234/{article_id[:6]}",
        pmcid=None, title="t", journal="j", publication_year=2024,
    )


def _study(article_id, *, evidence_text_level="abstract", excluded=False):
    return {
        "article_id": article_id,
        "citation_label": f"PMID:1{article_id[:6]}",
        "pmid": f"1{article_id[:6]}",
        "doi": f"10.1234/{article_id[:6]}",
        "pmcid": None,
        "title": "t", "journal": "j", "publication_year": 2024,
        "study_type": "rct",
        "evidence_text_level": evidence_text_level,
        "_is_excluded": excluded,
    }


def _draft(claims, certainty="low", selected=None):
    return DraftSummary(
        clinical_bottom_line="bl",
        evidence_summary="es",
        applicability="ap",
        certainty=certainty,
        certainty_rationale="cr",
        selected_studies=selected or [],
        claim_map=claims,
        conflicting_evidence="",
        important_caveats="",
        what_was_not_answered="",
    )


def _study_entry(article_id, span_id, *, pmid=None, doi=None, level="abstract"):
    return SelectedStudyEntry(
        article_id=article_id,
        citation_label=f"PMID:1{article_id[:6]}",
        pmid=pmid if pmid is not None else f"1{article_id[:6]}",
        doi=doi if doi is not None else f"10.1234/{article_id[:6]}",
        pmcid=None,
        title="t", journal="j", publication_year=2024, study_type="rct",
        evidence_text_level=level,
        key_finding="finding", limitations="lim",
        supporting_evidence_span_ids=[span_id],
    )


def test_valid_structured_json_accepted():
    a1, a2 = "aaaaaaaaaa", "bbbbbbbbbb"
    spans = [_span(a1), _span(a2)]
    studies = [_study(a1), _study(a2)]
    draft = _draft(
        [ClaimMapEntry(
            claim_id="c1", claim_text="x",
            supporting_evidence_span_ids=[f"span:{a1}:0"],
            supporting_article_ids=[a1],
        )],
        selected=[_study_entry(a1, f"span:{a1}:0")],
    )
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, vs = floor.check(draft)
    assert passed, vs


def test_missing_claim_map_rejected():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1)]
    studies = [_study(a1)]
    draft = _draft([], certainty="low")
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, vs = floor.check(draft)
    assert not passed
    assert any("claim_map is empty" in v for v in vs)


def test_insufficient_with_empty_claim_map_allowed():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1)]
    studies = [_study(a1)]
    draft = _draft([], certainty="insufficient")
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, _ = floor.check(draft)
    assert passed


def test_claim_without_supporting_span_ids_rejected():
    # We have to bypass Pydantic min_length to construct an invalid model.
    with pytest.raises(Exception):
        _draft([ClaimMapEntry(claim_id="c1", claim_text="x", supporting_evidence_span_ids=[])])


def test_fabricated_pmid_rejected():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1)]
    studies = [_study(a1)]
    bad_pmid = "9999999"  # different from study's pmid
    draft = _draft(
        [ClaimMapEntry(claim_id="c1", claim_text="x",
                       supporting_evidence_span_ids=[f"span:{a1}:0"],
                       supporting_article_ids=[a1])],
        selected=[_study_entry(a1, f"span:{a1}:0", pmid=bad_pmid)],
    )
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, vs = floor.check(draft)
    assert not passed
    assert any("PMID mismatch" in v for v in vs)


def test_fabricated_doi_rejected():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1)]
    studies = [_study(a1)]
    draft = _draft(
        [ClaimMapEntry(claim_id="c1", claim_text="x",
                       supporting_evidence_span_ids=[f"span:{a1}:0"],
                       supporting_article_ids=[a1])],
        selected=[_study_entry(a1, f"span:{a1}:0", doi="10.9999/fake")],
    )
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, vs = floor.check(draft)
    assert not passed
    assert any("DOI mismatch" in v for v in vs)


def test_cited_unselected_article_rejected():
    a1, a2 = "aaaaaaaaaa", "bbbbbbbbbb"
    spans = [_span(a1)]
    # Only a1 is in the selected set; the draft cites a2.
    studies = [_study(a1)]
    draft = _draft(
        [ClaimMapEntry(claim_id="c1", claim_text="x",
                       supporting_evidence_span_ids=[f"span:{a2}:0"],
                       supporting_article_ids=[a2])],
    )
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, vs = floor.check(draft)
    assert not passed
    assert any("unknown evidence_span_id" in v or "unselected" in v for v in vs)


def test_cited_excluded_article_rejected():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1)]
    studies = [_study(a1, excluded=True)]
    draft = _draft(
        [ClaimMapEntry(claim_id="c1", claim_text="x",
                       supporting_evidence_span_ids=[f"span:{a1}:0"],
                       supporting_article_ids=[a1])],
    )
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, vs = floor.check(draft)
    assert not passed
    assert any("excluded/retracted" in v for v in vs)


def test_abstract_only_evidence_labeled_correctly():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1, level="abstract")]
    studies = [_study(a1, evidence_text_level="abstract")]
    draft = _draft(
        [ClaimMapEntry(claim_id="c1", claim_text="x",
                       supporting_evidence_span_ids=[f"span:{a1}:0"],
                       supporting_article_ids=[a1])],
        selected=[_study_entry(a1, f"span:{a1}:0", level="abstract")],
        certainty="low",
    )
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, _ = floor.check(draft)
    assert passed


def test_user_uploaded_full_text_evidence_labeled_correctly():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1, level="user_uploaded_full_text")]
    studies = [_study(a1, evidence_text_level="user_uploaded_full_text")]
    draft = _draft(
        [ClaimMapEntry(claim_id="c1", claim_text="x",
                       supporting_evidence_span_ids=[f"span:{a1}:0"],
                       supporting_article_ids=[a1])],
        selected=[_study_entry(a1, f"span:{a1}:0", level="user_uploaded_full_text")],
        certainty="moderate",
    )
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, _ = floor.check(draft)
    assert passed
