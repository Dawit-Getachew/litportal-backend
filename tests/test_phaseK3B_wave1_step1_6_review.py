"""Phase K3B Wave 1 Step 1.6 — Adjudicator + sentinel-scrub tests.

Covers the closed-set Wave-1 adjudicator (``adjudicate_wave1``) and the
deep leak-scrub on CSV / JSONL artifacts.

Key invariants:
- A WBE=true row WITHOUT structured retraction metadata returns
  ``false_positive`` (NOT silently treated as TP).
- A WBE=true row with ``article_join_status != stable`` returns the
  ``detector_defect_wbe_on_unstable_join`` defect (this should never
  fire post-Step-1.5, but the safety net must hold).
- An EoC-only WBE=true row returns the
  ``detector_defect_eoc_only_wbe`` defect.
- A WBE row with a non-Wave-1 reason code returns
  ``detector_defect_non_wave1_reason_code``.
- A WBE row missing ``retracted`` in WBE/cap codes returns the
  matching ``detector_defect_missing_retracted_reason_code`` defect.
- Sentinel substrings (``system:``, ``sk-``, ``@gmail.com``) are
  stripped from the CSV / JSONL outputs by the deep scrub.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.k3b_wave1_step1_6_verdict_adjudicator import (  # noqa: E402
    adjudicate_wave1,
)
from scripts.k3b_wave1_step1_soak_collector import (  # noqa: E402
    _deep_scrub,
)


# ---------------------------------------------------------------------
# Closed-set verdict assertions.
# ---------------------------------------------------------------------


def _stable_wbe(**overrides: Any) -> dict[str, Any]:
    """Build a stable-join WBE=true row with Wave-1-compliant codes."""
    base: dict[str, Any] = {
        "article_join_status": "stable",
        "would_have_been_excluded_reason_codes": ["retracted"],
        "cap_reason_codes": ["retracted"],
        "publication_types": ["Retracted Publication"],
        "is_retracted": True,
        "is_retraction_notice": False,
        "has_expression_of_concern": False,
        "pmid": "12345",
        "doi": "10.0/x.y",
        "title": "A retracted paper",
    }
    base.update(overrides)
    return base


def test_tp_when_pubtype_retracted_publication():
    """Pub-type ``Retracted Publication`` → TP via PubMed pub-type."""
    r = _stable_wbe()
    v = adjudicate_wave1(r)
    assert v["verdict"] == "true_positive_retracted"
    assert v["verdict_basis"] == "pubmed_retracted_publication_type"
    assert v["detector_defect"] is None


def test_tp_when_is_retracted_flag_only():
    """``is_retracted=True`` only → TP via journal notice basis."""
    r = _stable_wbe(publication_types=[], is_retracted=True)
    v = adjudicate_wave1(r)
    assert v["verdict"] == "true_positive_retracted"
    assert v["verdict_basis"] == (
        "journal_official_retraction_notice_via_pubmed"
    )
    assert v["detector_defect"] is None


def test_tp_when_is_retraction_notice_only():
    """``is_retraction_notice=True`` only → TP via PubMed link."""
    r = _stable_wbe(
        publication_types=[], is_retracted=False,
        is_retraction_notice=True,
    )
    v = adjudicate_wave1(r)
    assert v["verdict"] == "true_positive_retracted"
    assert v["verdict_basis"] == "pubmed_retraction_in_link"
    assert v["detector_defect"] is None


def test_fp_when_no_structured_retraction_signal():
    """**Required by Step 1.6 spec**: WBE=true with NO retraction
    metadata → ``false_positive``. Defense-in-depth: the adjudicator
    must NOT silently classify the row as TP just because the gate
    fired.
    """
    r = _stable_wbe(
        publication_types=[],
        is_retracted=False,
        is_retraction_notice=False,
        has_expression_of_concern=False,
    )
    v = adjudicate_wave1(r)
    assert v["verdict"] == "false_positive"
    assert v["verdict_basis"] == "source_metadata_missing_or_ambiguous"
    assert v["detector_defect"] is None


def test_unclear_when_eoc_only_signal_with_retracted_code():
    """Cap code says retracted but only EoC signal on the article →
    ``unclear_requires_source_review`` (no detector defect — this is
    a downstream metadata disagreement, not a gate bug).
    """
    r = _stable_wbe(
        publication_types=["Expression of Concern"],
        is_retracted=False,
        is_retraction_notice=False,
        has_expression_of_concern=True,
    )
    v = adjudicate_wave1(r)
    assert v["verdict"] == "unclear_requires_source_review"
    assert v["verdict_basis"] == (
        "metadata_disagreement_across_sources"
    )
    assert v["detector_defect"] is None


# ---------------------------------------------------------------------
# Detector-defect assertions.
# ---------------------------------------------------------------------


def test_defect_when_wbe_on_unstable_join():
    """**Required by Step 1.6 spec**: WBE=true on unstable join →
    ``detector_defect_wbe_on_unstable_join``. This should never
    happen post-Step-1.5 — the safety net must catch it.
    """
    r = _stable_wbe(
        article_join_status=(
            "article_removed_between_classification_and_review"
        ),
    )
    v = adjudicate_wave1(r)
    assert v["detector_defect"] == (
        "detector_defect_wbe_on_unstable_join"
    )
    assert v["verdict_basis"] == (
        "article_join_unstable_no_verdict_possible"
    )


def test_defect_when_eoc_only_wbe():
    """Wave-1 must not enforce EoC. EoC-only WBE → defect."""
    r = _stable_wbe(
        would_have_been_excluded_reason_codes=[
            "expression_of_concern",
        ],
        cap_reason_codes=["expression_of_concern"],
    )
    v = adjudicate_wave1(r)
    assert v["detector_defect"] == "detector_defect_eoc_only_wbe"


def test_defect_when_non_wave1_reason_code():
    """E.g. ``preprint_unverified`` alone → defect."""
    r = _stable_wbe(
        would_have_been_excluded_reason_codes=["preprint_unverified"],
        cap_reason_codes=["preprint_unverified"],
    )
    v = adjudicate_wave1(r)
    assert v["detector_defect"] == (
        "detector_defect_non_wave1_reason_code"
    )


def test_defect_when_wbe_missing_retracted_reason_code():
    """WBE codes are empty (no ``retracted``) → defect."""
    r = _stable_wbe(would_have_been_excluded_reason_codes=[])
    v = adjudicate_wave1(r)
    assert v["detector_defect"] == (
        "detector_defect_missing_retracted_reason_code"
    )


def test_defect_when_classifier_gate_disagree_on_retracted():
    """WBE says ``retracted`` but ``cap_reason_codes`` does not →
    classifier/gate disagreement → defect.
    """
    r = _stable_wbe(cap_reason_codes=["other_reason"])
    v = adjudicate_wave1(r)
    # The Wave-1 reason allow-list catches ``other_reason`` first
    # (since it's a non-Wave-1 reason path through cap_codes...
    # actually cap_codes are NOT checked against the allow-list —
    # only WBE codes are). So we expect the classifier/gate
    # disagreement defect to fire.
    assert v["detector_defect"] == (
        "detector_defect_missing_retracted_reason_code"
    )
    assert v["verdict_basis"] == (
        "metadata_disagreement_across_sources"
    )


# ---------------------------------------------------------------------
# Deep-scrub leak-guard assertions.
# ---------------------------------------------------------------------


_FORBIDDEN = ("system:", "sk-", "@gmail.com")


def test_deep_scrub_strips_nested_list_strings():
    """The deep scrub must descend into nested lists / dicts and
    drop any string that contains a forbidden substring.
    """
    payload = {
        "ok": "fine",
        "explanation": [
            "k3b:wave1", "system:leak", "sk-LEAKEDKEY",
            "user@gmail.com", "ok-string",
        ],
        "nested": {
            "deeper": ["safe", "system:nope", "fine"],
        },
    }
    scrubbed = _deep_scrub(payload)
    blob = json.dumps(scrubbed)
    for s in _FORBIDDEN:
        assert s not in blob
    assert "ok-string" in scrubbed["explanation"]
    assert "k3b:wave1" in scrubbed["explanation"]
    assert "fine" in scrubbed["nested"]["deeper"]


def test_deep_scrub_preserves_non_string_primitives():
    """Numbers, booleans, None pass through unchanged."""
    payload = {
        "n": 123, "b": True, "x": None, "lst": [1, 2, None, True],
    }
    out = _deep_scrub(payload)
    assert out == payload


def test_csv_writer_emits_no_forbidden_sentinels(tmp_path):
    """End-to-end: a row carrying forbidden sentinels in nested
    fields, written through the CSV writer pattern used by the
    review runner, must not produce a CSV containing any sentinel.
    """
    from scripts.k3b_wave1_step1_6_review_runner import CSV_HEADER

    bad_row = {
        "row_id": 1, "session_id": "s", "article_id": "a",
        "pmid": "1", "doi": "10.x", "title": "ok",
        "journal": "ok system:leak",  # poison
        "publication_date": "2020",
        "source_provenance": "test",
        "cap_reason_codes": json.dumps(["retracted"]),
        "would_have_been_excluded_reason_codes": json.dumps(
            ["retracted"],
        ),
        "article_join_status": "stable",
        "source_metadata_signal": "sk-LEAKEDKEY",  # poison
        "verdict": "true_positive_retracted",
        "verdict_basis": "pubmed_retracted_publication_type",
        "detector_defect": "",
    }
    scrubbed = _deep_scrub(bad_row)
    out = tmp_path / "review.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_HEADER))
        w.writeheader()
        w.writerow({k: scrubbed.get(k, "") for k in CSV_HEADER})
    blob = out.read_text(encoding="utf-8")
    for s in _FORBIDDEN:
        assert s not in blob


def test_jsonl_writer_emits_no_forbidden_sentinels(tmp_path):
    """End-to-end: a JSONL row carrying forbidden sentinels in nested
    fields, written through the deep-scrub + json.dumps pattern used
    by the soak collector, must not produce a JSONL containing any
    sentinel.
    """
    bad_row = {
        "session_id": "s", "article_id": "a",
        "enforcement_explanation": [
            "k3b:wave1", "system:leak",
            "user@gmail.com",
        ],
        "shadow_invariants": {
            "cap_mode": "shadow",
            "would_cap": False,
            "would_exclude": False,
            "cap_enforcement_mode": "shadow",
            "enforcement_candidate_type": "retracted_only",
        },
        "source_metadata_signal": "sk-FAKE12345",
    }
    out = tmp_path / "soak.jsonl"
    scrubbed = _deep_scrub(bad_row)
    out.write_text(
        json.dumps(scrubbed) + "\n",
        encoding="utf-8",
    )
    blob = out.read_text(encoding="utf-8")
    for s in _FORBIDDEN:
        assert s not in blob
