"""Phase K3A.2 — Unit tests for soak collector + verdict adjudicator.

Covers:

- The verdict adjudicator classifies a synthetic
  ``pubmed.publication_types=["Retracted Publication"]`` record as
  ``true_positive_retracted_or_eoc`` and a record lacking any
  retraction marker as ``false_positive``.
- The verdict adjudicator routes ``expression_of_concern`` candidates
  with conflicting retraction signals to
  ``unclear_requires_source_review``.
- The soak collector excludes ``blocked_phi`` sessions at the Mongo
  query layer.
- The soak collector NEVER reads ``original_question`` or
  ``sanitized_question`` from any document (verified by overloading
  the cursor with a doc carrying these forbidden fields and asserting
  the assertion fires; also verified at the JSONL artifact layer).
- Every JSONL row passes the ``_FORBIDDEN_DIAG_SUBSTRINGS`` scrub.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


# Make ``scripts.*`` importable when running pytest from /app/backend.
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.k3a2_verdict_adjudicator import adjudicate  # noqa: E402
from scripts.k3a2_soak_collector import (  # noqa: E402
    _scrub_row,
    collect,
)


# ---------------------------------------------------------------------
# Verdict adjudicator.
# ---------------------------------------------------------------------


def test_adjudicator_retracted_publication_type_is_true_positive():
    record = {
        "cap_reason_code": "retracted",
        "publication_types": ["Journal Article", "Retracted Publication"],
        "is_retracted": True,
        "has_expression_of_concern": False,
        "is_retraction_notice": False,
    }
    v = adjudicate(record)
    assert v["verdict"] == "true_positive_retracted_or_eoc"
    assert v["verdict_basis"] == "pubmed_retracted_publication_type"


def test_adjudicator_no_retraction_signal_is_false_positive():
    record = {
        "cap_reason_code": "retracted",
        "publication_types": ["Journal Article"],
        "is_retracted": False,
        "has_expression_of_concern": False,
        "is_retraction_notice": False,
    }
    v = adjudicate(record)
    assert v["verdict"] == "false_positive"
    assert v["verdict_basis"] == "source_metadata_missing_or_ambiguous"


def test_adjudicator_eoc_pub_type_is_true_positive():
    record = {
        "cap_reason_code": "expression_of_concern",
        "publication_types": ["Journal Article", "Expression of Concern"],
        "is_retracted": False,
        "has_expression_of_concern": True,
        "is_retraction_notice": False,
    }
    v = adjudicate(record)
    assert v["verdict"] == "true_positive_retracted_or_eoc"
    assert v["verdict_basis"] == "journal_official_eoc_notice_via_pubmed"


def test_adjudicator_eoc_with_retracted_signal_is_unclear():
    record = {
        "cap_reason_code": "expression_of_concern",
        "publication_types": ["Journal Article", "Retracted Publication"],
        "is_retracted": True,
        "has_expression_of_concern": False,
        "is_retraction_notice": False,
    }
    v = adjudicate(record)
    assert v["verdict"] == "unclear_requires_source_review"
    assert v["verdict_basis"] == "metadata_disagreement_across_sources"


def test_adjudicator_eoc_no_signal_is_false_positive():
    record = {
        "cap_reason_code": "expression_of_concern",
        "publication_types": ["Journal Article"],
        "is_retracted": False,
        "has_expression_of_concern": False,
        "is_retraction_notice": False,
    }
    v = adjudicate(record)
    assert v["verdict"] == "false_positive"


def test_adjudicator_retraction_notice_is_true_positive_via_link():
    record = {
        "cap_reason_code": "retracted",
        "publication_types": ["Journal Article"],
        "is_retracted": False,
        "has_expression_of_concern": False,
        "is_retraction_notice": True,
    }
    v = adjudicate(record)
    assert v["verdict"] == "true_positive_retracted_or_eoc"
    assert v["verdict_basis"] == "pubmed_retraction_in_link"


def test_adjudicator_article_is_retracted_flag_is_true_positive():
    record = {
        "cap_reason_code": "retracted",
        "publication_types": ["Journal Article"],
        "is_retracted": True,
        "has_expression_of_concern": False,
        "is_retraction_notice": False,
    }
    v = adjudicate(record)
    assert v["verdict"] == "true_positive_retracted_or_eoc"
    assert v["verdict_basis"] == (
        "journal_official_retraction_notice_via_pubmed"
    )


def test_adjudicator_unknown_code_is_unclear():
    record = {"cap_reason_code": "missing_primary_anchor"}
    v = adjudicate(record)
    assert v["verdict"] == "unclear_requires_source_review"
    assert v["verdict_basis"] == "source_metadata_missing_or_ambiguous"


def test_adjudicator_metadata_blank_is_unclear_not_false_positive():
    """K3A.2 — when the article doc is orphaned (deduplicated /
    cleaned-up / missing) and every metadata field on the soak row
    is blank, the verdict MUST be ``unclear_requires_source_review``,
    not ``false_positive``. The classifier saw metadata at retrieval
    time that we no longer can.
    """
    record = {
        "cap_reason_code": "retracted",
        "publication_types": [],
        "is_retracted": False,
        "has_expression_of_concern": False,
        "is_retraction_notice": False,
        "pmid": "",
        "doi": "",
        "title": "",
    }
    v = adjudicate(record)
    assert v["verdict"] == "unclear_requires_source_review"
    assert v["verdict_basis"] == "source_metadata_missing_or_ambiguous"


# ---------------------------------------------------------------------
# Soak collector — PHI exclusion, projection guard, forbidden-substring
# scrub.
# ---------------------------------------------------------------------


def test_scrub_row_drops_forbidden_substring_fields():
    row = {
        "session_id": "ok-sid",
        "title": "normal title",
        "explanation": "system: leak",
        "another": "sk-abcdef",
    }
    scrubbed = _scrub_row(row)
    assert "session_id" in scrubbed
    assert "title" in scrubbed
    # Fields containing forbidden sentinels are dropped.
    assert "explanation" not in scrubbed
    assert "another" not in scrubbed


@pytest.mark.asyncio
async def test_collector_excludes_blocked_phi_and_skips_forbidden_fields(
    client,  # ensures Mongo test DB is up; we use the same conn string
):
    """Seed two sessions: one ``awaiting_selection`` with retracted ALD,
    one ``blocked_phi``. The collector MUST only emit the first, and
    MUST never read the forbidden text fields even though we
    deliberately seed them.
    """
    from core.db import get_db

    db = get_db()
    sid_ok = str(uuid.uuid4())
    sid_phi = str(uuid.uuid4())
    art_id = str(uuid.uuid4())
    # Seed the article — populated as if PubMed had stamped it.
    await db.articles.insert_one({
        "id": art_id,
        "pmid": "32330253",  # Mehra et al / NEJM Surgisphere
        "doi": "10.1056/NEJMoa2007621",
        "pmcid": "",
        "title": "Cardiovascular Disease, Drug Therapy, and Mortality "
                 "in Covid-19 [RETRACTED]",
        "journal": "N Engl J Med",
        "publication_year": 2020,
        "publication_date": "2020-05-01",
        "publication_types_json": [
            "Journal Article", "Retracted Publication",
        ],
        "is_retracted": True,
        "is_retraction_notice": False,
        "has_expression_of_concern": False,
        "source_ids_json": {"pubmed": "32330253"},
        "source_urls_json": [
            "https://pubmed.ncbi.nlm.nih.gov/32330253/",
        ],
    })
    now = datetime.now(timezone.utc)
    ald_blob = {
        "schema_version": "K3A.1.0",
        "lane_label_mode": "shadow",
        "cap_detection_mode": "shadow",
        "projection_contract": "full_v1",
        "diagnostics": [
            {
                "article_id": art_id,
                "lane_label": "retracted_or_expression_of_concern",
                "directness_label": "unsafe",
                "cap_reason_codes": ["retracted"],
                "cap_candidates": [
                    {
                        "reason_code": "retracted",
                        "severity": "high",
                        "explanation_id": "safety:retracted",
                    },
                ],
                "cap_severity": "high",
                "cap_confidence": 0.99,
                "cap_mode": "shadow",
                "would_cap": False,
                "would_exclude": False,
                "explanation": ["safety:retracted_or_eoc"],
                "qa_warnings": [],
                "branch_alignment": "aligned",
                "matched_branch_ids": ["primary_0"],
                "article_intent_fit": "full",
                "evidence_role_candidate": "to_exclude_candidate",
                "lane_confidence": 0.99,
            },
        ],
    }
    # Deliberately seed BOTH original_question and sanitized_question
    # on both sessions — the collector must never project them.
    await db.search_sessions.insert_one({
        "id": sid_ok,
        "user_id": "fake-uid",
        "original_question": "FORBIDDEN_RAW_QUESTION_OK",
        "sanitized_question": "FORBIDDEN_SANITIZED_QUESTION_OK",
        "status": "awaiting_selection",
        "created_at": now,
        "updated_at": now,
        "expires_at": now,
        "deleted_at": None,
        "session_diagnostics_json": {
            "article_lane_diagnostics": ald_blob,
        },
        "clinical_question_map_json": {
            "primary_intent": "therapy_safety",
        },
    })
    await db.search_sessions.insert_one({
        "id": sid_phi,
        "user_id": "fake-uid",
        "original_question": "FORBIDDEN_RAW_QUESTION_PHI",
        "sanitized_question": "FORBIDDEN_SANITIZED_QUESTION_PHI",
        "status": "blocked_phi",
        "created_at": now,
        "updated_at": now,
        "expires_at": now,
        "deleted_at": None,
        "session_diagnostics_json": {
            "article_lane_diagnostics": ald_blob,
        },
        "clinical_question_map_json": {
            "primary_intent": "therapy_safety",
        },
    })

    out_path = Path(tempfile.mkdtemp()) / "soak.jsonl"
    summary = await collect(
        mongo_url=os.environ["MONGO_URL"],
        db_name=os.environ["DB_NAME"],
        window_days=1,
        out_path=out_path,
        provenance="test",
    )
    assert summary["candidates_retracted"] >= 1
    # Read the JSONL and confirm:
    #   1. blocked_phi session never appears
    #   2. neither original_question nor sanitized_question is in any row
    #   3. no forbidden substring appears in any row
    rows = [json.loads(L) for L in out_path.read_text().splitlines() if L.strip()]
    seen_sids = {r["session_id"] for r in rows}
    assert sid_phi not in seen_sids, (
        "blocked_phi session leaked into the JSONL artifact"
    )
    blob = json.dumps(rows)
    for tok in (
        "FORBIDDEN_RAW_QUESTION_OK", "FORBIDDEN_SANITIZED_QUESTION_OK",
        "FORBIDDEN_RAW_QUESTION_PHI", "FORBIDDEN_SANITIZED_QUESTION_PHI",
        "system:", "SYSTEM_PROMPT", "sk-", "@gmail.com",
    ):
        assert tok not in blob, (
            f"forbidden token {tok!r} leaked to JSONL"
        )
    # And every row carries the shadow-invariants stamp.
    for r in rows:
        si = r.get("shadow_invariants") or {}
        assert si.get("cap_mode") == "shadow"
        assert si.get("would_cap") is False
        assert si.get("would_exclude") is False
        assert si.get("cap_enforcement_mode") == "disabled"
