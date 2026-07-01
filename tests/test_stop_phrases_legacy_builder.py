"""Phase 5D — STOP_PHRASES filter must apply to ALL outgoing query strings.

Both the new ``BooleanQueryCompilerService`` (Phase 5D) AND the legacy
``concept_block_query_builder`` (Phase 5A) must strip prose noise like
``"difference between"``, ``"comparison of"``, ``"comparing"``, and
``"what is the dose of"`` before emitting any query.

Pre-fix, the legacy builder bypassed the STOP_PHRASES filter — so a
``TITLE_ABS:"what is the dose of milrinone"`` expression could leak into
Europe PMC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.concept_block_query_builder import concept_block_query_builder  # noqa: E402
from services.query_compiler import boolean_query_compiler  # noqa: E402


# Stop-phrase substrings that must NOT appear anywhere in an emitted query
# regardless of which builder produced it.
_FORBIDDEN_SUBSTRINGS = [
    "difference between",
    "comparison of",
    "comparing",
    "what is the dose of",
    "what is the role of",
    "what is the evidence for",
    "what is the best",
    "what is the optimal",
    "in patients that are",
    "in patients that have",
    "in patients who are",
    "in patients who have",
]


def _picott_with_leaks() -> dict:
    """A PICOTT whose concept terms intentionally contain stop-phrase
    fragments. The builders MUST strip them out before emission."""
    return {
        "type_of_question": "therapy",
        "must_have_concepts": [
            # "difference between" must be stripped from the concept term.
            {
                "concept": "difference between liberal and restrictive transfusion",
                "role": "intervention",
                "synonyms": [
                    "comparison of liberal vs restrictive",
                    "comparing transfusion strategies",
                ],
            },
            {
                "concept": "what is the dose of milrinone",
                "role": "intervention",
                "synonyms": ["what is the optimal milrinone dose"],
            },
            {"concept": "acute myocardial infarction", "role": "condition"},
        ],
        "ranking_concepts": [
            {"concept": "comparing strategies", "role": "context"},
            {"concept": "what is the role of erythropoietin", "role": "context"},
        ],
        "pico": {
            "population": ["in patients that are anemic"],
            "outcomes_or_target_diagnosis": [
                "what is the evidence for mortality benefit",
            ],
        },
    }


def _assert_no_forbidden_in(queries, builder_name: str):
    for q in queries:
        for s in _FORBIDDEN_SUBSTRINGS:
            assert s not in q.query_string.lower(), (
                f"{builder_name} leaked '{s}' into "
                f"[{q.source}/{q.query_family}]: {q.query_string!r}"
            )


def test_legacy_builder_strips_stop_phrases_pubmed_and_europe_pmc():
    """Phase 5A legacy builder must strip STOP_PHRASES across both sources."""
    picott = _picott_with_leaks()
    queries = concept_block_query_builder.build(
        picott, sources=["pubmed", "europe_pmc"],
    )
    assert queries, "legacy builder must emit at least one query for this PICOTT"
    _assert_no_forbidden_in(queries, "concept_block_query_builder")


def test_phase5d_compiler_strips_stop_phrases_pubmed_and_europe_pmc():
    """Phase 5D BooleanQueryCompiler must strip STOP_PHRASES (regression)."""
    picott = _picott_with_leaks()
    queries = boolean_query_compiler.compile(
        picott, sources=["pubmed", "europe_pmc"],
    )
    assert queries, "phase5d compiler must emit at least one query"
    _assert_no_forbidden_in(queries, "boolean_query_compiler")


def test_both_builders_strip_all_listed_stop_phrases():
    """End-to-end: across ALL 9 families and 2 sources, no STOP_PHRASE leaks."""
    picott = _picott_with_leaks()
    legacy = concept_block_query_builder.build(
        picott, sources=["pubmed", "europe_pmc"],
    )
    phase5d = boolean_query_compiler.compile(
        picott, sources=["pubmed", "europe_pmc"],
    )
    all_queries = list(legacy) + list(phase5d)
    _assert_no_forbidden_in(all_queries, "combined-builders")


def test_question_with_difference_between_emits_zero_leaks_after_parse():
    """Realistic scenario: a "difference between X and Y" question must
    not leak the phrase via any builder (verifies the integration path
    where the LLM/parser hands a real PICOTT to the builders)."""
    # Synthetic PICOTT (the parser-output shape) — no LLM dependency.
    picott = {
        "type_of_question": "therapy",
        "must_have_concepts": [
            {"concept": "transfusion strategy", "role": "intervention",
             "synonyms": ["liberal transfusion", "restrictive transfusion"]},
            {"concept": "acute myocardial infarction", "role": "condition"},
        ],
        "ranking_concepts": [],
        "pico": {},
    }
    legacy = concept_block_query_builder.build(
        picott, sources=["pubmed", "europe_pmc"],
    )
    phase5d = boolean_query_compiler.compile(
        picott, sources=["pubmed", "europe_pmc"],
    )
    # Sanity: both produce queries (the must-haves are clean).
    assert legacy, "legacy must emit queries from clean PICOTT"
    # No leaks (vacuously true since clean PICOTT — but also covers
    # regression where filter accidentally degraded the clean text).
    _assert_no_forbidden_in(legacy + list(phase5d), "clean-PICOTT")
    # And the queries should still contain the actual anchor concepts.
    pubmed_q_strs = " ".join(q.query_string for q in legacy if q.source == "pubmed")
    assert "transfusion strategy" in pubmed_q_strs or \
        "liberal transfusion" in pubmed_q_strs
