"""Phase 5D round-4 — Fix 4 (search_planning.py STOP filter).

Tester round-3 caught that the legacy ``search_planning.py`` planner (which
emits the families ``high_recall`` / ``high_precision`` /
``evidence_filtered`` / ``guideline_focused`` on each source) was bypassing
the STOP_PHRASES + STOP_TOKENS filter that round-3 wired into
``concept_block_query_builder.py``.

Leak examples the tester captured:
  - AMI session:        TITLE_ABS:"difference between", "difference between"[tiab],
                        TITLE_ABS:"between high"
  - tell-me-about-diabetes session: TITLE_ABS:"tell", TITLE_ABS:"about",
                                    "tell"[tiab], "about"[tiab]

This file exercises ``search_planning.plan()`` directly with synthetic
parser outputs (the shape ``clinical_question_parser.parse()`` produces)
so we don't depend on the LLM. Then asserts ZERO forbidden substrings
appear in any emitted query string across the 4 legacy families × 2
sources = 8 queries per case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.search_planning import search_planning  # noqa: E402


# Forbidden substrings — exact tokens the tester captured leaking on
# round 3, plus the broader set of single STOP_TOKENS we want to make
# sure never make it into a TITLE_ABS:"…" or "…"[tiab] expression.
FORBIDDEN_SUBSTRINGS = [
    "difference between",
    "comparison of",
    "comparing",
    "what is the dose of",
    "what is the role of",
    "tell me about",
    "tell me",
    'TITLE_ABS:"tell"',
    'TITLE_ABS:"about"',
    'TITLE_ABS:"me"',
    'TITLE_ABS:"the"',
    'TITLE_ABS:"is"',
    'TITLE_ABS:"a"',
    'TITLE_ABS:"an"',
    'TITLE_ABS:"of"',
    'TITLE_ABS:"in"',
    'TITLE_ABS:"between high"',
    '"tell"[tiab]',
    '"about"[tiab]',
    '"me"[tiab]',
    '"the"[tiab]',
    '"is"[tiab]',
    '"a"[tiab]',
    '"an"[tiab]',
    '"of"[tiab]',
    '"in"[tiab]',
    '"between high"[tiab]',
]


def _tell_me_about_diabetes_parsed() -> dict:
    """Synthetic legacy ParsedPICO output that mimics what
    ``clinical_question_parser.parse()`` returns for the question
    ``"tell me about diabetes"`` — the LLM may hallucinate "tell",
    "about", "me" as key_concepts/synonyms."""
    return {
        "clinical_question_type": "other",
        "normalized_question": "tell me about diabetes",
        "population": "patients with diabetes",
        "intervention_or_exposure": "tell me about",
        "comparator": "",
        "outcomes": ["about"],
        "key_concepts": ["tell", "about", "me", "diabetes"],
        "synonyms": ["tell me", "about diabetes", "me"],
        "drug_names_and_classes": [],
    }


def _ami_difference_parsed() -> dict:
    """Synthetic legacy ParsedPICO output for the AMI question
    ``"What is the difference between liberal and restrictive transfusion
    strategy in acute MI?"``. The LLM may produce "difference between",
    "between high", arbitrary bigram synonyms."""
    return {
        "clinical_question_type": "treatment",
        "normalized_question": (
            "what is the difference between liberal and restrictive "
            "transfusion strategy in acute myocardial infarction"
        ),
        "population": "patients with acute myocardial infarction",
        "intervention_or_exposure": "difference between liberal and restrictive transfusion",
        "comparator": "comparison of strategies",
        "outcomes": ["mortality"],
        "key_concepts": [
            "difference between",
            "liberal transfusion",
            "restrictive transfusion",
            "acute myocardial infarction",
            "anemia",
        ],
        "synonyms": [
            "between high", "high hemoglobin", "comparison of",
            "difference between liberal", "anemia in MI",
        ],
        "drug_names_and_classes": [],
    }


def _collect_legacy_emissions(parsed: dict) -> list:
    """Run the legacy planner on both sources and return the emitted
    PlannedQuery list."""
    return search_planning.plan(parsed, sources=["pubmed", "europe_pmc"])


def _assert_no_forbidden(plans, *, question_label: str):
    failures: list[str] = []
    for p in plans:
        qs_lower = p.query_string.lower()
        for sub in FORBIDDEN_SUBSTRINGS:
            sub_lower = sub.lower()
            if sub_lower in qs_lower:
                failures.append(
                    f"[{p.source}/{p.query_type}] forbidden={sub!r} "
                    f"query={p.query_string[:160]!r}"
                )
    assert not failures, (
        f"Found {len(failures)} STOP_PHRASE/STOP_TOKEN leaks in legacy "
        f"planner output for {question_label!r}:\n  " +
        "\n  ".join(failures[:12])
    )


def test_legacy_planner_no_leaks_tell_me_about_diabetes():
    """Round-4 contract: the legacy ``search_planning.plan()`` MUST NOT
    emit any STOP_TOKEN as a TITLE_ABS:"…" or "…"[tiab] term."""
    parsed = _tell_me_about_diabetes_parsed()
    plans = _collect_legacy_emissions(parsed)
    # Sanity: planner must still emit queries (the "diabetes" anchor
    # survives the filter).
    assert plans, "legacy planner emitted zero queries — fix went too far"
    assert any("diabetes" in p.query_string for p in plans), (
        "diabetes anchor should survive the filter"
    )
    _assert_no_forbidden(plans, question_label="tell me about diabetes")


def test_legacy_planner_no_leaks_ami_difference_question():
    """Round-4 contract: the AMI difference-between question MUST NOT
    leak ``TITLE_ABS:"difference between"`` etc."""
    parsed = _ami_difference_parsed()
    plans = _collect_legacy_emissions(parsed)
    assert plans, "legacy planner emitted zero queries on AMI input"
    # Sanity: real anchors survive.
    assert any(
        "myocardial infarction" in p.query_string.lower()
        or "transfusion" in p.query_string.lower()
        or "anemia" in p.query_string.lower()
        for p in plans
    ), "AMI anchors should survive the filter"
    _assert_no_forbidden(plans, question_label="AMI difference question")


def test_legacy_planner_all_four_families_per_source_clean():
    """Both probe sessions, ALL 4 legacy families × 2 sources, ZERO leaks."""
    for label, parsed in [
        ("tell me about diabetes", _tell_me_about_diabetes_parsed()),
        ("AMI difference question", _ami_difference_parsed()),
    ]:
        plans = _collect_legacy_emissions(parsed)
        # Verify we have queries from all 4 families on both sources
        # (some may be skipped if the filter empties a block — that's OK
        # as long as we don't emit ``()`` or leak STOP_TOKENS).
        for source in ("pubmed", "europe_pmc"):
            for fam in ("high_recall", "high_precision",
                         "evidence_filtered", "guideline_focused"):
                src_fam_plans = [
                    p for p in plans
                    if p.source == source and p.query_type == fam
                ]
                for p in src_fam_plans:
                    # Empty Boolean blocks (``()``) must not be emitted.
                    assert "()" not in p.query_string, (
                        f"empty Boolean block leaked: "
                        f"[{source}/{fam}] {p.query_string!r}"
                    )
        _assert_no_forbidden(plans, question_label=label)


def test_planner_never_emits_quoted_single_stop_tokens():
    """Strong invariant: no matter what shape the parsed dict has, the
    planner must never emit a quoted single-stopword term."""
    junk = {
        "clinical_question_type": "other",
        "normalized_question": "what is the best of the best",
        "population": "the patients",
        "intervention_or_exposure": "the",
        "comparator": "a",
        "outcomes": ["the", "a", "an"],
        "key_concepts": ["the", "a", "an", "is", "of", "in", "on"],
        "synonyms": ["the", "a", "an", "to", "from", "with"],
        "drug_names_and_classes": [],
    }
    plans = _collect_legacy_emissions(junk)
    for p in plans:
        # No quoted single-stop-token may appear.
        for tok in ("the", "a", "an", "is", "of", "in", "on", "to", "from", "with"):
            assert f'TITLE_ABS:"{tok}"' not in p.query_string, (
                f"single stop-token leaked into {p.source}/{p.query_type}: "
                f"{p.query_string!r}"
            )
            assert f'"{tok}"[tiab]' not in p.query_string, (
                f"single stop-token leaked into {p.source}/{p.query_type}: "
                f"{p.query_string!r}"
            )
