"""Recovery of must-have concepts whose role has no concept block.

BUG: ``concept_block_query_builder._build_blocks`` merged must-have
concepts into six blocks via an if/elif chain with **no else arm**. Any
must-have whose ``role`` was not one of {intervention, condition,
outcome, comparator, population, context} was silently dropped from
every query.

``ConceptRole`` (models/picott.py) also contains ``study_design``,
``setting``, ``time_horizon`` and ``other`` — and ``other`` is the
common case, assigned whenever the parser cannot confidently role-tag a
required concept.

Live example, production session 5786f432 ("What is the prognosis of
idiopathic pulmonary fibrosis?"): three of four must-haves were
role=other and never reached PubMed.

    Idiopathic Pulmonary Fibrosis          role=other   -> DROPPED
    Pulmonary Fibrosis                     role=other   -> DROPPED
    Prognosis                              role=other   -> DROPPED
    prognosis of idiopathic pulmonary...   role=condition -> kept

leaving a query of exactly one fused literal phrase, which matches
almost nothing.

THE INVARIANT THAT MAKES THE FIX SAFE: blocks are AND-ed together and
empty blocks are skipped by the emitters (``if terms:``). So recovered
terms must be OR-ed into an EXISTING non-empty block. Giving them a
block of their own would add a conjunct and REDUCE recall — the
opposite of the intent. Several tests below pin exactly that.
"""
from __future__ import annotations

import pytest

from services.concept_block_query_builder import (
    _build_blocks,
    _unmapped_merge_target,
    concept_block_query_builder,
)


def _conjuncts(blocks):
    """Non-empty blocks == AND groups in the emitted query."""
    return [k for k, v in blocks.items() if v]


@pytest.fixture(autouse=True)
def _recovery_on(monkeypatch):
    monkeypatch.setenv("PICOTT_RECOVER_UNMAPPED_ROLES", "true")


def _pfo_picott():
    """The exact must-have set observed in production session 5786f432."""
    return {
        "type_of_question": "prognosis",
        "question_scope": "foreground",
        "pico": {},
        "must_have_concepts": [
            {"concept": "Idiopathic Pulmonary Fibrosis", "role": "other",
             "synonyms": ["IPF"]},
            {"concept": "Pulmonary Fibrosis", "role": "other", "synonyms": []},
            {"concept": "Prognosis", "role": "other",
             "synonyms": ["survival", "mortality"]},
            {"concept": "prognosis of idiopathic pulmonary fibrosis",
             "role": "condition", "synonyms": []},
        ],
    }


# ---------------------------------------------------------------------
# The reported defect
# ---------------------------------------------------------------------
def test_role_other_concepts_are_no_longer_dropped():
    blocks = _build_blocks(_pfo_picott())
    joined = " | ".join(blocks["condition"])
    for concept in ("Idiopathic Pulmonary Fibrosis", "Pulmonary Fibrosis",
                    "Prognosis"):
        assert concept in joined, f"{concept!r} still dropped"


def test_recovered_synonyms_come_along():
    blocks = _build_blocks(_pfo_picott())
    assert "IPF" in blocks["condition"]
    assert "survival" in blocks["condition"]


def test_kill_switch_restores_the_old_drop_behaviour(monkeypatch):
    monkeypatch.setenv("PICOTT_RECOVER_UNMAPPED_ROLES", "false")
    blocks = _build_blocks(_pfo_picott())
    assert blocks["condition"] == ["prognosis of idiopathic pulmonary fibrosis"]


@pytest.mark.parametrize("off", ["false", "0", "no", "off", "FALSE", "Off"])
def test_kill_switch_accepts_the_usual_falsey_spellings(monkeypatch, off):
    monkeypatch.setenv("PICOTT_RECOVER_UNMAPPED_ROLES", off)
    assert len(_build_blocks(_pfo_picott())["condition"]) == 1


# ---------------------------------------------------------------------
# The safety invariant — recovery must never add an AND conjunct
# ---------------------------------------------------------------------
_SHAPES = [
    ("condition only", [
        {"concept": "sepsis", "role": "condition"},
        {"concept": "lactate", "role": "other"}]),
    ("intervention only", [
        {"concept": "heparin", "role": "intervention"},
        {"concept": "clot burden", "role": "other"}]),
    ("population only", [
        {"concept": "older adults", "role": "population"},
        {"concept": "frailty", "role": "other"}]),
    ("outcome only", [
        {"concept": "mortality", "role": "outcome"},
        {"concept": "ICU stay", "role": "other"}]),
    ("all other", [
        {"concept": "amyloidosis", "role": "other"},
        {"concept": "cardiac", "role": "other"}]),
    ("mixed", [
        {"concept": "AF", "role": "condition"},
        {"concept": "apixaban", "role": "intervention"},
        {"concept": "stroke", "role": "outcome"},
        {"concept": "anticoagulation", "role": "other"}]),
]


@pytest.mark.parametrize("label,must", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_recovery_never_narrows_an_already_usable_query(label, must, monkeypatch):
    """The precise invariant.

    Where at least one block was already populated, recovery must NOT
    add a conjunct — that would AND a new required group onto a working
    query and lose results.

    The single exception is the degenerate case where NO block was
    populated: the emitters skip empty blocks, so zero conjuncts means
    zero queries emitted. Going 0 -> 1 there converts "no query at all"
    into "one OR group", which is a strict improvement, not a
    narrowing. Asserted explicitly in
    ``test_all_other_roles_produce_a_single_block_not_several``.
    """
    picott = {"type_of_question": "therapy", "pico": {}, "must_have_concepts": must}
    monkeypatch.setenv("PICOTT_RECOVER_UNMAPPED_ROLES", "false")
    before = _conjuncts(_build_blocks(picott))
    monkeypatch.setenv("PICOTT_RECOVER_UNMAPPED_ROLES", "true")
    after = _conjuncts(_build_blocks(picott))
    if not before:
        assert len(after) <= 1, (
            f"{label}: degenerate case must yield at most ONE group, got {after}"
        )
        return
    assert len(after) <= len(before), (
        f"{label}: recovery added a conjunct ({before} -> {after}); "
        "that narrows the query instead of widening it"
    )


@pytest.mark.parametrize("label,must", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_recovery_never_removes_a_previously_present_term(label, must, monkeypatch):
    picott = {"type_of_question": "therapy", "pico": {}, "must_have_concepts": must}
    monkeypatch.setenv("PICOTT_RECOVER_UNMAPPED_ROLES", "false")
    before = _build_blocks(picott)
    monkeypatch.setenv("PICOTT_RECOVER_UNMAPPED_ROLES", "true")
    after = _build_blocks(picott)
    for role, terms in before.items():
        assert set(terms) <= set(after[role]), f"{label}/{role} lost a term"


def test_all_other_roles_produce_a_single_block_not_several():
    """When nothing else is populated the recovered terms become the ONE
    group — no conjunct is created, and the query goes from unusable to
    usable.
    """
    picott = {"type_of_question": "other", "pico": {}, "must_have_concepts": [
        {"concept": "cardiac amyloidosis", "role": "other"},
        {"concept": "transthyretin", "role": "other"},
    ]}
    blocks = _build_blocks(picott)
    assert _conjuncts(blocks) == ["condition"]
    assert len(blocks["condition"]) == 2


# ---------------------------------------------------------------------
# Roles that must STAY out of topical blocks
# ---------------------------------------------------------------------
@pytest.mark.parametrize("role", ["study_design", "setting", "time_horizon"])
def test_non_topical_roles_are_still_excluded(role):
    """``study_design`` is already encoded by the publication-type
    filter; ``setting`` / ``time_horizon`` OR-ed into a disease block
    would match any ICU or 30-day paper. Recovering these would cost
    precision, so they stay out.
    """
    picott = {"type_of_question": "therapy", "pico": {}, "must_have_concepts": [
        {"concept": "sepsis", "role": "condition"},
        {"concept": "SHOULD-NOT-APPEAR", "role": role},
    ]}
    blocks = _build_blocks(picott)
    assert "SHOULD-NOT-APPEAR" not in " ".join(
        t for terms in blocks.values() for t in terms
    )


# ---------------------------------------------------------------------
# Merge-target selection
# ---------------------------------------------------------------------
def test_merge_target_prefers_condition_then_intervention():
    assert _unmapped_merge_target({"condition": ["x"], "intervention": ["y"]}) == "condition"
    assert _unmapped_merge_target({"condition": [], "intervention": ["y"]}) == "intervention"
    assert _unmapped_merge_target(
        {"condition": [], "intervention": [], "population": ["p"]}) == "population"
    assert _unmapped_merge_target({}) == "condition"


# ---------------------------------------------------------------------
# End-to-end through the real emitter
# ---------------------------------------------------------------------
def test_emitted_pubmed_query_widens_for_the_live_pfo_case(monkeypatch):
    picott = _pfo_picott()

    def high_precision(flag):
        monkeypatch.setenv("PICOTT_RECOVER_UNMAPPED_ROLES", flag)
        qs = concept_block_query_builder.build(picott, sources=["pubmed"])
        hp = [q for q in qs if q.query_family == "high_precision"]
        return hp[0].query_string if hp else ""

    before, after = high_precision("false"), high_precision("true")
    assert before and after
    # Same number of AND groups...
    assert before.count(" AND ") == after.count(" AND ")
    # ...but strictly more OR alternatives.
    assert after.count(" OR ") > before.count(" OR ")
    assert "Idiopathic Pulmonary Fibrosis" in after
    assert "Idiopathic Pulmonary Fibrosis" not in before


def test_concept_blocks_keys_are_unchanged_for_the_frontend():
    """The frontend renders ``concept_blocks`` from the query record.
    Recovery must only add TERMS to existing keys — never add, rename or
    remove a key — or the transparency panel breaks.
    """
    expected = {"population", "intervention", "comparator", "outcome",
                "condition", "context"}
    assert set(_build_blocks(_pfo_picott())) == expected
    qs = concept_block_query_builder.build(_pfo_picott(), sources=["pubmed"])
    for q in qs:
        if q.concept_blocks:
            assert set(q.concept_blocks) <= expected
