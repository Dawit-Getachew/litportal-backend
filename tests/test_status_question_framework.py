"""``GET /api/search-sessions/{id}/status`` — question-framework surfacing.

The ClinicalQuestionMap has chosen a framework (PECO / PIRD / CoCoPop /
PFO / CHARMS / ...) on every search since Phase K1, but ``/status``
returned only the legacy ``pico_json`` under ``parsed``, which has no
notion of one. A PECO question and a PICOTT question were therefore
indistinguishable to the client, with no way to show that the backend
had in fact routed the question.

These tests exercise the pure projection logic the endpoint performs.
The endpoint itself needs Mongo + auth, so it is covered by the
Mongo-backed router suites; here we pin the response CONTRACT, which is
the part that can silently break a connected frontend.
"""
from __future__ import annotations

import pytest

from services.interpreted_question_display import (
    NOT_SPECIFIED,
    framework_display_fields,
)

# Mirror of the projection inside ``session_status``. Kept in step by
# ``test_projection_matches_the_router_implementation`` below.
_FW_KEYS = (
    "question_framework", "question_framework_raw",
    "clinical_intent", "clinical_intent_raw",
)


def _project(doc):
    fw = framework_display_fields(doc)
    picott = doc.get("picott_json") or {}
    parsed_out = doc.get("pico_json") or None
    if isinstance(parsed_out, dict) and parsed_out:
        parsed_out = {
            **{k: fw[k] for k in _FW_KEYS},
            "type_of_question": picott.get("type_of_question"),
            **{k: v for k, v in parsed_out.items() if k not in _FW_KEYS},
        }
    return {
        "question_framework": {
            "framework": fw["question_framework_raw"],
            "framework_label": fw["question_framework"],
            "frameworks": fw["question_frameworks"],
            "intent": fw["clinical_intent_raw"],
            "intent_label": fw["clinical_intent"],
            "secondary_intents": fw["secondary_clinical_intents"],
            "type_of_question": picott.get("type_of_question"),
            "question_subtype": picott.get("question_subtype"),
        },
        "parsed": parsed_out,
    }


def _doc(framework="PECO", intent="risk_etiology_harm", *, pico=True,
         cqm=True, toq="etiology_harm", subtype=None):
    d = {
        "picott_json": {"type_of_question": toq, "question_subtype": subtype},
    }
    if pico:
        d["pico_json"] = {
            "normalized_question": "does biomass smoke exposure increase ALRI",
            "clinical_question_type": "harm",
            "population": "children under 5",
            "intervention_or_exposure": "household biomass smoke",
            "comparator": "clean fuel households",
            "outcomes": ["incidence of ALRI"],
        }
    if cqm:
        d["clinical_question_map_json"] = {
            "primary_intent": intent,
            "secondary_intents": [],
            "frameworks": [framework],
        }
    return d


# ---------------------------------------------------------------------
# The new top-level block
# ---------------------------------------------------------------------
def test_status_exposes_the_framework_at_top_level():
    out = _project(_doc())
    qf = out["question_framework"]
    assert qf["framework"] == "PECO"
    assert qf["framework_label"] == "PECO"
    assert qf["frameworks"] == ["PECO"]
    assert qf["intent"] == "risk_etiology_harm"
    assert qf["intent_label"] == "Risk etiology harm"


def test_status_block_distinguishes_framework_from_legacy_type():
    """The framework and the legacy ``type_of_question`` are different
    things — the latter is what drives the PubMed publication-type
    filter. Both are surfaced so they cannot be confused.
    """
    qf = _project(_doc(framework="CoCoPop",
                       intent="epidemiology_prevalence_incidence",
                       toq="epidemiology"))["question_framework"]
    assert qf["framework"] == "CoCoPop"
    assert qf["type_of_question"] == "epidemiology"


def test_status_block_is_always_present_even_for_legacy_sessions():
    """A client must be able to render the badge unconditionally."""
    qf = _project(_doc(cqm=False))["question_framework"]
    assert qf["framework"] is None
    assert qf["framework_label"] == NOT_SPECIFIED
    assert qf["frameworks"] == []
    assert qf["intent"] is None


@pytest.mark.parametrize("framework,label", [
    ("PECO", "PECO"), ("PIRD", "PIRD"), ("CoCoPop", "CoCoPop"),
    ("PFO", "PFO"), ("CHARMS", "CHARMS"), ("PICOTT", "PICOTT"),
    ("PICOTS", "PICOTS"),
    ("concept_mechanism", "Concept mechanism"),
    ("guideline_stabilization", "Guideline stabilization"),
    ("problem_context_differential", "Problem context differential"),
])
def test_all_routable_frameworks_render_correctly(framework, label):
    qf = _project(_doc(framework=framework))["question_framework"]
    assert qf["framework"] == framework
    assert qf["framework_label"] == label


def test_acronym_casing_is_never_mangled():
    """``CoCoPop`` must not become ``Cocopop`` or ``COCOPOP``."""
    qf = _project(_doc(framework="CoCoPop"))["question_framework"]
    assert qf["framework_label"] == "CoCoPop"


# ---------------------------------------------------------------------
# ``parsed`` — additive, framework keys first
# ---------------------------------------------------------------------
def test_framework_keys_come_first_in_parsed():
    """Requested placement: visible at the top of the payload rather
    than appended after a dozen legacy keys.
    """
    parsed = _project(_doc())["parsed"]
    assert list(parsed)[:5] == [
        "question_framework", "question_framework_raw",
        "clinical_intent", "clinical_intent_raw", "type_of_question",
    ]


def test_parsed_keeps_every_legacy_key():
    """CONTRACT: the frontend already consumes ``parsed``. Nothing may
    be dropped, renamed or retyped.
    """
    doc = _doc()
    legacy = doc["pico_json"]
    parsed = _project(doc)["parsed"]
    for k, v in legacy.items():
        assert k in parsed, f"legacy key {k!r} disappeared from parsed"
        assert parsed[k] == v, f"legacy key {k!r} changed value"


def test_parsed_stays_none_when_there_is_no_pico_json():
    """Flipping ``None`` -> ``{}`` would break clients doing
    ``if (!parsed)``.
    """
    assert _project(_doc(pico=False))["parsed"] is None


def test_parsed_enrichment_does_not_mutate_the_source_document():
    """The projection must not write back into the Mongo doc."""
    doc = _doc()
    before = dict(doc["pico_json"])
    _project(doc)
    assert doc["pico_json"] == before
    assert "question_framework" not in doc["pico_json"]


def test_parsed_framework_agrees_with_the_top_level_block():
    """Two surfaces, one source — they must never drift."""
    out = _project(_doc(framework="PIRD", intent="diagnosis_accuracy"))
    assert out["parsed"]["question_framework_raw"] == \
        out["question_framework"]["framework"]
    assert out["parsed"]["clinical_intent_raw"] == \
        out["question_framework"]["intent"]


def test_legacy_key_collision_is_resolved_in_favour_of_the_framework():
    """Defensive: if a legacy pico_json ever carried one of the new key
    names, the framework value wins and the key is not duplicated.
    """
    doc = _doc()
    doc["pico_json"]["question_framework"] = "STALE"
    parsed = _project(doc)["parsed"]
    assert parsed["question_framework"] == "PECO"
    assert list(parsed).count("question_framework") == 1


# ---------------------------------------------------------------------
# Keep the test's projection honest against the real endpoint
# ---------------------------------------------------------------------
def test_projection_matches_the_router_implementation():
    """Guards against this file drifting from ``session_status``.

    Asserts the endpoint still builds the block from the shared helper
    and still emits both surfaces.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "routers" / "search_sessions.py").read_text(encoding="utf-8")
    assert "framework_display_fields(doc)" in src
    assert '"question_framework": {' in src
    assert '"framework_label": _fw["question_framework"]' in src
    assert '"parsed": parsed_out,' in src
    # ...and that the legacy keys are still merged in rather than replaced.
    assert "if k not in _fw_keys" in src
