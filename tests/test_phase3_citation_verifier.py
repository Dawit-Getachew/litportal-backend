"""Phase 3 — citation verifier semantic + mechanical floor (Task 11 / D)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from services.summarization.evidence_spans import EvidenceSpan
from services.summarization.schemas import (
    ClaimMapEntry, DraftSummary, SelectedStudyEntry, VerifierResult,
)
from services.summarization.verifier import (
    CitationVerifierService, MechanicalVerificationFloor,
)


def _span(article_id, idx=0, level="abstract"):
    return EvidenceSpan(
        evidence_span_id=f"span:{article_id}:{idx}",
        article_id=article_id, search_session_id="sid",
        section_name="abstract", text_source=level, evidence_text_level=level,
        start_char=0, end_char=10, span_text="ev",
        citation_label=f"PMID:1{article_id[:6]}",
        pmid=f"1{article_id[:6]}", doi=f"10.1234/{article_id[:6]}",
        pmcid=None, title="t", journal="j", publication_year=2024,
    )


def _study(article_id, level="abstract", excluded=False):
    return {
        "article_id": article_id, "citation_label": f"PMID:1{article_id[:6]}",
        "pmid": f"1{article_id[:6]}", "doi": f"10.1234/{article_id[:6]}",
        "pmcid": None, "title": "t", "journal": "j", "publication_year": 2024,
        "study_type": "rct", "evidence_text_level": level, "_is_excluded": excluded,
    }


def _study_entry(article_id, span_id, *, pmid=None, doi=None, level="abstract"):
    return SelectedStudyEntry(
        article_id=article_id,
        citation_label=f"PMID:1{article_id[:6]}",
        pmid=pmid if pmid is not None else f"1{article_id[:6]}",
        doi=doi if doi is not None else f"10.1234/{article_id[:6]}",
        pmcid=None, title="t", journal="j", publication_year=2024, study_type="rct",
        evidence_text_level=level,
        key_finding="finding", limitations="lim",
        supporting_evidence_span_ids=[span_id],
    )


def _draft(claims, *, certainty="low", selected=None):
    return DraftSummary(
        clinical_bottom_line="bl", evidence_summary="es", applicability="ap",
        certainty=certainty, certainty_rationale="cr",
        selected_studies=selected or [], claim_map=claims,
        conflicting_evidence="", important_caveats="", what_was_not_answered="",
    )


def test_supported_claim_passes():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1)]
    draft = _draft([
        ClaimMapEntry(claim_id="c1", claim_text="x",
                      supporting_evidence_span_ids=[f"span:{a1}:0"],
                      supporting_article_ids=[a1]),
    ], selected=[_study_entry(a1, f"span:{a1}:0")])
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=[_study(a1)])
    passed, vs = floor.check(draft)
    assert passed, vs


def test_unsupported_claim_fails_via_llm():
    """Even if the mechanical floor passes, the LLM verifier can reject claims."""
    pytest.skip("Covered by API integration tests where the verifier LLM is mocked.")


def test_citation_to_nonexistent_span_fails():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1)]
    draft = _draft([
        ClaimMapEntry(claim_id="c1", claim_text="x",
                      supporting_evidence_span_ids=[f"span:{a1}:99"],
                      supporting_article_ids=[a1]),
    ])
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=[_study(a1)])
    passed, vs = floor.check(draft)
    assert not passed
    assert any("unknown evidence_span_id" in v for v in vs)


def test_citation_to_unselected_article_fails():
    a1, a2 = "aaaaaaaaaa", "bbbbbbbbbb"
    spans = [_span(a1), _span(a2)]
    draft = _draft([
        ClaimMapEntry(claim_id="c1", claim_text="x",
                      supporting_evidence_span_ids=[f"span:{a2}:0"],
                      supporting_article_ids=[a2]),
    ])
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=[_study(a1)])
    passed, vs = floor.check(draft)
    assert not passed


def test_citation_to_excluded_article_fails():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1)]
    draft = _draft([
        ClaimMapEntry(claim_id="c1", claim_text="x",
                      supporting_evidence_span_ids=[f"span:{a1}:0"],
                      supporting_article_ids=[a1]),
    ])
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=[_study(a1, excluded=True)])
    passed, vs = floor.check(draft)
    assert not passed


def test_overstated_certainty_failed_or_requires_revision():
    """Abstract-only evidence + certainty=high MUST be rejected."""
    a1 = "aaaaaaaaaa"
    spans = [_span(a1, level="abstract")]
    studies = [_study(a1, level="abstract")]
    draft = _draft(
        [ClaimMapEntry(claim_id="c1", claim_text="x",
                       supporting_evidence_span_ids=[f"span:{a1}:0"],
                       supporting_article_ids=[a1])],
        selected=[_study_entry(a1, f"span:{a1}:0")],
        certainty="high",
    )
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, vs = floor.check(draft)
    assert not passed
    assert any("Downgrade certainty" in v for v in vs)


@pytest.mark.asyncio
async def test_mechanical_floor_overrides_lenient_llm_verdict():
    """If the LLM verifier says 'passed' but the floor finds violations, fail."""
    a1 = "aaaaaaaaaa"
    spans = [_span(a1, level="abstract")]
    studies = [_study(a1, level="abstract")]
    bad_draft = _draft(
        [ClaimMapEntry(claim_id="c1", claim_text="x",
                       supporting_evidence_span_ids=[f"span:{a1}:0"],
                       supporting_article_ids=[a1])],
        selected=[_study_entry(a1, f"span:{a1}:0")],
        certainty="high",  # over-states certainty given only abstract spans
    )
    from services.llm_provider import llm_provider
    saved = (llm_provider.provider, llm_provider._emergent_key)
    llm_provider.provider = "emergent"
    llm_provider._emergent_key = "test"
    try:
        # The LLM verifier reports passed; floor must override.
        llm_provider.complete_json = AsyncMock(return_value={  # type: ignore[assignment]
            "verification_status": "passed",
            "final_answer_safe_to_show": True,
            "unsupported_claims": [], "incorrect_citations": [],
            "required_revisions": [], "verifier_notes": "ok",
            "verified_summary_json": None,
        })
        svc = CitationVerifierService()
        result = await svc.verify(draft=bad_draft, spans=spans, selected_studies=studies)
    finally:
        llm_provider.provider, llm_provider._emergent_key = saved
    assert isinstance(result, VerifierResult)
    assert result.verification_status == "failed"
    assert any("Downgrade certainty" in r for r in result.required_revisions)


def test_pmid_mismatch_with_article_record_fails():
    a1 = "aaaaaaaaaa"
    spans = [_span(a1)]
    studies = [_study(a1)]
    draft = _draft(
        [ClaimMapEntry(claim_id="c1", claim_text="x",
                       supporting_evidence_span_ids=[f"span:{a1}:0"],
                       supporting_article_ids=[a1])],
        selected=[_study_entry(a1, f"span:{a1}:0", pmid="9999")],
    )
    floor = MechanicalVerificationFloor(spans=spans, selected_studies=studies)
    passed, vs = floor.check(draft)
    assert not passed
    assert any("PMID mismatch" in v for v in vs)
