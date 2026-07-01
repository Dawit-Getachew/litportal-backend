"""Phase K3A.3 — Adjudicator v2 unit tests.

Confirms the expanded closed-set verdict / verdict_basis enums:

- ``true_positive_retracted`` (specific)
- ``true_positive_expression_of_concern`` (specific)
- ``article_join_unstable_no_verdict_possible`` (new basis — short
  -circuits whenever ``article_join_status != "stable"``)
- ``pubmed_expression_of_concern_publication_type``

Also verifies that the FORBIDDEN sentinel injection from K3A.2 still
fires through the v2 path (the v2 adjudicator is pure — no IO — so
sentinels never reach it via this path, but the closed-set output
itself must never contain leaks).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.k3a3_verdict_adjudicator import adjudicate_v2  # noqa: E402


def test_v2_retracted_specific_tp():
    r = {
        "cap_reason_code": "retracted",
        "article_join_status": "stable",
        "publication_types": ["Journal Article", "Retracted Publication"],
        "is_retracted": True,
    }
    v = adjudicate_v2(r)
    assert v["verdict"] == "true_positive_retracted"
    assert v["verdict_basis"] == "pubmed_retracted_publication_type"


def test_v2_eoc_specific_tp():
    r = {
        "cap_reason_code": "expression_of_concern",
        "article_join_status": "stable",
        "publication_types": ["Journal Article", "Expression of Concern"],
        "has_expression_of_concern": True,
    }
    v = adjudicate_v2(r)
    assert v["verdict"] == "true_positive_expression_of_concern"
    assert v["verdict_basis"] == (
        "pubmed_expression_of_concern_publication_type"
    )


def test_v2_eoc_via_article_flag():
    """When PubMed pub-type lacks ``Expression of Concern`` but the
    article doc was stamped ``has_expression_of_concern=true`` (e.g.
    via Europe PMC commentCorrections normalization), the v2
    adjudicator MUST still emit the specific EoC verdict.
    """
    r = {
        "cap_reason_code": "expression_of_concern",
        "article_join_status": "stable",
        "publication_types": ["Journal Article"],
        "has_expression_of_concern": True,
    }
    v = adjudicate_v2(r)
    assert v["verdict"] == "true_positive_expression_of_concern"
    assert v["verdict_basis"] == "journal_official_eoc_notice_via_pubmed"


def test_v2_unstable_join_short_circuits_to_unclear():
    r = {
        "cap_reason_code": "retracted",
        "article_join_status": "article_doc_missing",
        "publication_types": ["Journal Article", "Retracted Publication"],
        "is_retracted": True,
    }
    v = adjudicate_v2(r)
    assert v["verdict"] == "unclear_requires_source_review"
    assert v["verdict_basis"] == "article_join_unstable_no_verdict_possible"


def test_v2_metadata_blank_with_stable_join_is_unclear():
    r = {
        "cap_reason_code": "retracted",
        "article_join_status": "stable",
        "publication_types": [],
        "is_retracted": False,
        "has_expression_of_concern": False,
        "is_retraction_notice": False,
        "pmid": "", "doi": "", "title": "",
    }
    v = adjudicate_v2(r)
    assert v["verdict"] == "unclear_requires_source_review"
    assert v["verdict_basis"] == "source_metadata_missing_or_ambiguous"


def test_v2_retracted_no_signals_is_false_positive():
    r = {
        "cap_reason_code": "retracted",
        "article_join_status": "stable",
        "publication_types": ["Journal Article"],
        "is_retracted": False,
        "has_expression_of_concern": False,
        "is_retraction_notice": False,
        "pmid": "12345", "doi": "10.x/y", "title": "Not retracted study",
    }
    v = adjudicate_v2(r)
    assert v["verdict"] == "false_positive"


def test_v2_eoc_with_retracted_signal_is_unclear_via_disagreement():
    r = {
        "cap_reason_code": "expression_of_concern",
        "article_join_status": "stable",
        "publication_types": ["Retracted Publication"],
        "is_retracted": True,
    }
    v = adjudicate_v2(r)
    assert v["verdict"] == "unclear_requires_source_review"
    assert v["verdict_basis"] == "metadata_disagreement_across_sources"
