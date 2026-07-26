"""Framework routing / surfacing quick-wins.

Covers the five changes that connect the (already-correct) Phase K1
framework decision to things the clinician and the search actually see:

  QW3 — ``k4_evidence_packet_builder`` read ``cqm["framework"]``
        (singular). The persisted ClinicalQuestionMap only has
        ``frameworks`` (a list) under ``extra="forbid"``, so every
        framework signal in the evidence packet was silently ``None``.
  QW9 — ``ACTIVE_FRAMEWORKS`` declared 14 frameworks; the router can
        emit 10. Split into ROUTED / DECLARED_NOT_ROUTED and enforce
        reachability (those live in test_phaseK1_clinical_question_map).
  QW7 — a prediction-model (CHARMS) question got PubMed's generic
        ``prognosis`` Clinical Queries filter; NCBI's fifth category,
        Clinical Prediction Guides, is the correct one.
  QW6 — ``CQM_TYPE_RECONCILIATION_MODE`` bridges the CQM intent onto
        the legacy ``type_of_question`` that actually reaches the query
        producers. Default-OFF, shadow, then active.
  QW2 — the clinician interpretation card rendered a fixed PICO grid
        for all 14 frameworks; it now carries the framework + intent,
        with a display schema version so cached cards get rebuilt.

No LLM calls. No network. No Mongo. Pure unit-level assertions on
closed-set behavior.
"""
from __future__ import annotations

import pytest

from services.clinical_question_map import (
    ACTIVE_FRAMEWORKS,
    DECLARED_NOT_ROUTED,
    ROUTED_FRAMEWORKS,
    _intent_to_legacy_type_of_question,
)
from models.picott import TYPE_OF_QUESTION_VALUES
from services.study_type_router import study_type_router


# =====================================================================
# QW3 — CQM framework/anchor readers
# =====================================================================
def _cqm(**over):
    base = {
        "primary_intent": "prognosis_natural_history",
        "frameworks": ["PFO"],
        "framework_neutral_slots": [
            {
                "canonical_concept": "idiopathic pulmonary fibrosis",
                "required_for_search": True,
            },
            {"canonical_concept": "mortality", "required_for_search": False},
        ],
    }
    base.update(over)
    return base


def test_qw3_primary_framework_reads_the_list_not_a_scalar_key():
    from services.k4_evidence_packet_builder import _cqm_primary_framework

    assert _cqm_primary_framework(_cqm()) == "PFO"
    # The old code path — a scalar "framework" key — does not exist on
    # the persisted model, which is exactly why it always read None.
    assert _cqm().get("framework") is None


def test_qw3_primary_framework_handles_empty_and_malformed():
    from services.k4_evidence_packet_builder import _cqm_primary_framework

    assert _cqm_primary_framework({}) is None
    assert _cqm_primary_framework({"frameworks": []}) is None
    assert _cqm_primary_framework({"frameworks": ["", "  "]}) is None
    assert _cqm_primary_framework({"frameworks": "PFO"}) is None  # not a list
    assert _cqm_primary_framework({"frameworks": [None, "PIRD"]}) == "PIRD"


def test_qw3_anchor_concepts_split_required_vs_all():
    from services.k4_evidence_packet_builder import _cqm_anchor_concepts

    all_anchors = _cqm_anchor_concepts(_cqm(), required_only=False)
    req_anchors = _cqm_anchor_concepts(_cqm(), required_only=True)
    assert all_anchors == ["idiopathic pulmonary fibrosis", "mortality"]
    assert req_anchors == ["idiopathic pulmonary fibrosis"]


def test_qw3_anchor_concepts_dedupe_and_tolerate_junk():
    from services.k4_evidence_packet_builder import _cqm_anchor_concepts

    cqm = _cqm(framework_neutral_slots=[
        {"canonical_concept": "sepsis", "required_for_search": True},
        {"canonical_concept": "sepsis", "required_for_search": True},
        {"canonical_concept": "", "required_for_search": True},
        "not-a-dict",
    ])
    assert _cqm_anchor_concepts(cqm, required_only=True) == ["sepsis"]


# =====================================================================
# QW9 — routed vs declared-not-routed frameworks
# =====================================================================
def test_qw9_routed_sets_partition_active_frameworks():
    assert ROUTED_FRAMEWORKS.isdisjoint(DECLARED_NOT_ROUTED)
    assert ROUTED_FRAMEWORKS | DECLARED_NOT_ROUTED == ACTIVE_FRAMEWORKS


def test_qw9_every_routed_framework_has_a_producing_intent():
    from services.pipeline import _k1_framework_for_intent
    from services.clinical_question_map import ACTIVE_INTENTS, DEFERRED_INTENTS

    reachable = {
        _k1_framework_for_intent(i) for i in (ACTIVE_INTENTS | DEFERRED_INTENTS)
    }
    assert ROUTED_FRAMEWORKS - reachable == set()


# =====================================================================
# QW7 — Clinical Queries category for prediction models
# =====================================================================
def test_qw7_prediction_model_gets_clinical_prediction_guides():
    from services.query_compiler import clinical_queries_category

    assert clinical_queries_category({
        "type_of_question": "prognosis",
        "question_subtype": "prediction_model",
    }) == "clinical_prediction_guides"


def test_qw7_plain_prognosis_is_unchanged():
    from services.query_compiler import clinical_queries_category

    assert clinical_queries_category(
        {"type_of_question": "prognosis"}
    ) == "prognosis"
    assert clinical_queries_category({
        "type_of_question": "prognosis",
        "question_subtype": "prognostic_factor",
    }) == "prognosis"


@pytest.mark.parametrize("qtype", ["guideline", "background", "cost", "other"])
def test_qw7_non_design_question_types_still_get_no_cq_filter(qtype):
    """PubMed Clinical Queries only has study-design categories.

    Forcing one onto a guideline/background question would filter out
    the narrative reviews and recommendation documents that answer it,
    so ``None`` is the correct result, not a gap.
    """
    from services.query_compiler import clinical_queries_category

    assert clinical_queries_category({"type_of_question": qtype}) is None


def test_qw7_new_category_has_both_scopes_and_is_wired():
    from services.query_compiler import CLINICAL_QUERIES_FILTERS

    cpg = CLINICAL_QUERIES_FILTERS["clinical_prediction_guides"]
    assert set(cpg) == {"broad", "narrow"}
    for scope in ("broad", "narrow"):
        # Balanced parens — the filter is appended raw into a PubMed query.
        assert cpg[scope].count("(") == cpg[scope].count(")")
        assert cpg[scope].startswith("(") and cpg[scope].endswith(")")


# =====================================================================
# QW6 — CQM -> legacy type reconciliation
# =====================================================================
def _picott(**over):
    base = {
        "type_of_question": "background",
        "question_subtype": None,
        "question_scope": "background",
        "ideal_study_types": ["narrative_review"],
    }
    base.update(over)
    return base


def _cqm_epi():
    """The canonical failure case: a prevalence question that the legacy
    parser typed as ``background`` and therefore filtered as a review.
    """
    return {
        "primary_intent": "epidemiology_prevalence_incidence",
        "frameworks": ["CoCoPop"],
        "clarification_required": False,
    }


def test_qw6_default_mode_is_disabled_and_is_a_no_op(monkeypatch):
    from services.pipeline import (
        _apply_cqm_type_reconciliation,
        _cqm_type_reconciliation_mode,
    )

    monkeypatch.delenv("CQM_TYPE_RECONCILIATION_MODE", raising=False)
    assert _cqm_type_reconciliation_mode() == "disabled"

    payload = _picott()
    before = dict(payload)
    diag = _apply_cqm_type_reconciliation(
        picott_payload=payload, cqm_json=_cqm_epi(),
    )
    assert payload == before
    assert diag["applied"] is False
    assert diag["skipped_reason"] == "mode_disabled"


@pytest.mark.parametrize("bad", ["", "ACTIVE!", "on", "true", "enabled"])
def test_qw6_unknown_mode_values_fail_safe_to_disabled(monkeypatch, bad):
    from services.pipeline import _cqm_type_reconciliation_mode

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", bad)
    assert _cqm_type_reconciliation_mode() == "disabled"


def test_qw6_shadow_computes_the_diff_but_changes_nothing(monkeypatch):
    from services.pipeline import _apply_cqm_type_reconciliation

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "shadow")
    payload = _picott()
    before = dict(payload)
    diag = _apply_cqm_type_reconciliation(
        picott_payload=payload, cqm_json=_cqm_epi(),
    )
    assert payload == before, "shadow mode must not mutate the payload"
    assert diag["applied"] is False
    assert diag["skipped_reason"] == "shadow_mode"
    assert diag["baseline_type_of_question"] == "background"
    assert diag["proposed_type_of_question"] == "epidemiology"
    assert diag["would_change"] is True


def test_qw6_active_applies_the_reconciled_shape(monkeypatch):
    from services.pipeline import _apply_cqm_type_reconciliation

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "active")
    payload = _picott()
    diag = _apply_cqm_type_reconciliation(
        picott_payload=payload, cqm_json=_cqm_epi(),
    )
    assert diag["applied"] is True
    assert payload["type_of_question"] == "epidemiology"
    assert payload["question_scope"] == "hybrid"
    # ideal_study_types must follow the reconciled pathway, not the
    # stale narrative-review list.
    pathway = study_type_router.pathway("epidemiology")
    assert payload["ideal_study_types"] == (
        list(pathway.preferred_study_types) + list(pathway.acceptable_study_types)
    )
    assert "narrative_review" not in payload["ideal_study_types"][:1]


def test_qw6_active_reaches_the_real_pubmed_pub_type_filter(monkeypatch):
    """End-to-end point of the change: the pub-type filter the query
    builder sends PubMed must change for a prevalence question.
    """
    from services.pipeline import _apply_cqm_type_reconciliation

    baseline_types = set(study_type_router.pathway("background").pubmed_pub_types)
    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "active")
    payload = _picott()
    _apply_cqm_type_reconciliation(
        picott_payload=payload, cqm_json=_cqm_epi(),
    )
    reconciled_types = set(
        study_type_router.pathway(payload["type_of_question"]).pubmed_pub_types
    )
    assert reconciled_types != baseline_types
    # The specific client-visible defect: a prevalence question was
    # being filtered as a review.
    assert "Review" in baseline_types
    assert "Review" not in reconciled_types


def test_qw6_active_sets_prediction_model_subtype_feeding_qw7(monkeypatch):
    """QW6 + QW7 compose: a CHARMS question must end up on the Clinical
    Prediction Guides filter, which only happens if QW6 writes the
    subtype QW7 keys off.
    """
    from services.pipeline import _apply_cqm_type_reconciliation
    from services.query_compiler import clinical_queries_category

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "active")
    payload = _picott(type_of_question="other", question_subtype=None)
    _apply_cqm_type_reconciliation(
        picott_payload=payload,
        cqm_json={
            "primary_intent": "prediction_model",
            "frameworks": ["CHARMS"],
            "clarification_required": False,
        },
    )
    assert payload["question_subtype"] == "prediction_model"
    assert clinical_queries_category(payload) == "clinical_prediction_guides"


def test_qw6_invalid_map_never_steers_retrieval(monkeypatch):
    """``_k1_clarification_required_placeholder`` sets
    ``clinical_validity=False`` and forces
    ``primary_intent="background_overview"`` / ``frameworks=["PICOTT"]``.
    That map is a "could not interpret" marker and must be ignored.
    """
    from services.pipeline import _apply_cqm_type_reconciliation

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "active")
    payload = _picott()
    before = dict(payload)
    diag = _apply_cqm_type_reconciliation(
        picott_payload=payload,
        cqm_json={
            "primary_intent": "background_overview",
            "frameworks": ["PICOTT"],
            "clinical_validity": False,
            "clarification_required": True,
        },
    )
    assert payload == before
    assert diag["applied"] is False
    assert diag["skipped_reason"] == "clinical_validity_false"


def test_qw6_clarification_required_alone_does_NOT_block(monkeypatch):
    """Regression guard for a bug in this feature's first cut.

    On the searching path ``clarification_required`` is set whenever the
    anchor extractor found no *primary anchor*, while the intent and
    framework still come from the deterministic signal router and are
    genuine. Gating on it disabled reconciliation for a large share of
    real questions — including the canonical prevalence case this
    feature exists to fix, which really does carry
    ``clarification_required=True`` end to end.
    """
    from services.pipeline import _apply_cqm_type_reconciliation

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "active")
    payload = _picott()
    diag = _apply_cqm_type_reconciliation(
        picott_payload=payload,
        cqm_json={
            "primary_intent": "epidemiology_prevalence_incidence",
            "frameworks": ["CoCoPop"],
            "clinical_validity": True,
            "clarification_required": True,
        },
    )
    assert diag["applied"] is True
    assert payload["type_of_question"] == "epidemiology"


def test_qw6_deferred_intents_are_skipped_explicitly(monkeypatch):
    from services.pipeline import _apply_cqm_type_reconciliation
    from services.clinical_question_map import DEFERRED_INTENTS

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "active")
    for intent in sorted(DEFERRED_INTENTS):
        payload = _picott()
        before = dict(payload)
        diag = _apply_cqm_type_reconciliation(
            picott_payload=payload,
            cqm_json={
                "primary_intent": intent,
                "frameworks": ["PICOTT"],
                "clinical_validity": True,
            },
        )
        assert payload == before, intent
        assert diag["skipped_reason"] == "deferred_intent", intent


@pytest.mark.parametrize("cqm_json", [None, {}, "not-a-dict", []])
def test_qw6_missing_or_malformed_cqm_is_a_no_op(monkeypatch, cqm_json):
    from services.pipeline import _apply_cqm_type_reconciliation

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "active")
    payload = _picott()
    before = dict(payload)
    diag = _apply_cqm_type_reconciliation(
        picott_payload=payload, cqm_json=cqm_json,
    )
    assert payload == before
    assert diag["applied"] is False
    assert diag["skipped_reason"] == "no_clinical_question_map"


def test_qw6_never_raises_on_a_hostile_payload(monkeypatch):
    from services.pipeline import _apply_cqm_type_reconciliation

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "active")

    class Hostile(dict):
        def __setitem__(self, k, v):
            raise RuntimeError("boom")

    payload = Hostile(type_of_question="other")
    diag = _apply_cqm_type_reconciliation(
        picott_payload=payload, cqm_json=_cqm_epi(),
    )
    assert diag["applied"] is False
    assert diag["skipped_reason"] == "exception"


def test_qw6_never_blanks_an_existing_subtype(monkeypatch):
    """Intents with no canonical subtype return None — that means 'no
    opinion', not 'clear what the parser found'.
    """
    from services.pipeline import _apply_cqm_type_reconciliation

    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", "active")
    payload = _picott(question_subtype="electrolyte_replacement")
    _apply_cqm_type_reconciliation(
        picott_payload=payload,
        cqm_json={
            "primary_intent": "therapy",
            "frameworks": ["PICOTT"],
            "clarification_required": False,
        },
    )
    assert payload["question_subtype"] == "electrolyte_replacement"


def test_qw6_every_intent_maps_into_the_downstream_closed_sets():
    """The reconciler writes ``type_of_question`` straight into the
    payload the ranker and query builders read. Every value the mapper
    can emit must be a valid enum member AND a real pathway key, or
    activation silently degrades to the ``other`` pathway.
    """
    from services.clinical_question_map import ACTIVE_INTENTS, DEFERRED_INTENTS

    for intent in sorted(ACTIVE_INTENTS | DEFERRED_INTENTS):
        toq = _intent_to_legacy_type_of_question(intent)
        assert toq in TYPE_OF_QUESTION_VALUES, (intent, toq)
        assert study_type_router.pathway(toq).type_of_question == toq, (
            f"{intent} -> {toq} falls through to the 'other' pathway"
        )


# =====================================================================
# QW6 — end-to-end through the REAL chain (no mocks, no LLM, no Mongo)
# =====================================================================
async def _real_chain(question: str, mode: str, monkeypatch):
    """parser -> signal router -> K1 CQM -> reconciliation -> query build.

    Uses the deterministic parser path (LLM_PROVIDER=none), which is the
    weakest case — if the framework survives here it survives with the
    LLM on. Unit-level asserts alone would not have caught the
    ``clarification_required`` over-gate, because the canonical
    prevalence question really does set that flag end to end.
    """
    from services.clinical_question_parser import clinical_question_parser
    from services.clinical_signal_router import clinical_signal_router
    from services import pipeline as P
    from services.concept_block_query_builder import concept_block_query_builder

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("CQM_TYPE_RECONCILIATION_MODE", mode)

    parsed = await clinical_question_parser.parse(question)
    picott = dict(parsed.get("picott_json") or {})
    signal = clinical_signal_router.analyze(
        parsed.get("normalized_question") or question
    ).model_dump(mode="json")
    cqm = P._k1_build_clinical_question_map(
        normalized_question=parsed.get("normalized_question") or question,
        original_question=question,
        picott_payload=picott,
        anchors_out=parsed.get("primary_clinical_anchors_json") or {},
        resolver_diag={},
        signal_summary=signal,
        clarification_only=False,
    )
    diag = P._apply_cqm_type_reconciliation(
        picott_payload=picott, cqm_json=cqm,
    )
    queries = concept_block_query_builder.build(picott, sources=["pubmed"])
    return cqm, picott, diag, queries


def _pub_types(queries):
    for q in queries:
        if q.query_family == "study_type_filtered":
            return q.query_string
    return ""


@pytest.mark.asyncio
async def test_qw6_e2e_prevalence_question_routes_to_cocopop(monkeypatch):
    cqm, _picott_out, _diag, _q = await _real_chain(
        "What is the prevalence of atrial fibrillation in older adults?",
        "disabled",
        monkeypatch,
    )
    assert cqm["frameworks"] == ["CoCoPop"]
    assert cqm["primary_intent"] == "epidemiology_prevalence_incidence"


@pytest.mark.asyncio
async def test_qw6_e2e_disabled_keeps_the_review_filter(monkeypatch):
    """Documents the defect: a prevalence question filtered as a Review."""
    _cqm, picott, diag, queries = await _real_chain(
        "What is the prevalence of atrial fibrillation in older adults?",
        "disabled",
        monkeypatch,
    )
    assert diag["applied"] is False
    assert picott["type_of_question"] == "background"
    assert '"Review"[Publication Type]' in _pub_types(queries)


@pytest.mark.asyncio
async def test_qw6_e2e_active_switches_to_the_epidemiology_filter(monkeypatch):
    """The fix, measured on the string that actually goes to PubMed."""
    _cqm, picott, diag, queries = await _real_chain(
        "What is the prevalence of atrial fibrillation in older adults?",
        "active",
        monkeypatch,
    )
    assert diag["applied"] is True, diag["skipped_reason"]
    assert picott["type_of_question"] == "epidemiology"
    pubs = _pub_types(queries)
    assert '"Cross-Sectional Studies"[Publication Type]' in pubs
    assert '"Cohort Studies"[Publication Type]' in pubs
    assert '"Review"[Publication Type]' not in pubs


@pytest.mark.asyncio
async def test_qw6_e2e_shadow_leaves_the_query_byte_identical(monkeypatch):
    _c1, _p1, _d1, q_disabled = await _real_chain(
        "What is the prevalence of atrial fibrillation in older adults?",
        "disabled", monkeypatch,
    )
    _c2, _p2, d_shadow, q_shadow = await _real_chain(
        "What is the prevalence of atrial fibrillation in older adults?",
        "shadow", monkeypatch,
    )
    assert d_shadow["would_change"] is True
    assert d_shadow["applied"] is False
    assert [q.query_string for q in q_shadow] == [
        q.query_string for q in q_disabled
    ]


@pytest.mark.asyncio
async def test_qw6_e2e_diagnostic_question_is_not_disturbed(monkeypatch):
    """Reconciliation must be a no-op where the legacy type was already
    right — otherwise activation is a regression risk, not a fix.
    """
    _c1, _p1, _d1, q_disabled = await _real_chain(
        "What is the diagnostic accuracy of D-dimer for pulmonary embolism?",
        "disabled", monkeypatch,
    )
    cqm, picott, diag, q_active = await _real_chain(
        "What is the diagnostic accuracy of D-dimer for pulmonary embolism?",
        "active", monkeypatch,
    )
    assert cqm["frameworks"] == ["PIRD"]
    assert picott["type_of_question"] == "diagnosis"
    assert diag["applied"] is True
    assert [q.query_string for q in q_active] == [
        q.query_string for q in q_disabled
    ]


# =====================================================================
# QW5 — the query categorizer's refinement must reach picott_json
# =====================================================================
@pytest.mark.asyncio
async def test_qw5_rebuilding_picott_propagates_the_refined_type(monkeypatch):
    """``AI_QUERY_CATEGORIZER_ENABLED`` was inert for routing.

    ``maybe_categorize`` rewrites ``parsed["clinical_question_type"]``,
    but ``parse()`` had already derived ``picott_json`` from the OLD
    value — and every downstream consumer reads ``picott_json``. This
    asserts the rebuild the pipeline now performs actually moves the
    type, i.e. that the flag is no longer a no-op.
    """
    from services.clinical_question_parser import clinical_question_parser
    from services.picott_extractor import picott_extractor

    monkeypatch.setenv("LLM_PROVIDER", "none")
    question = "Is drug X useful for condition Y?"
    parsed = await clinical_question_parser.parse(question)

    stale = (parsed.get("picott_json") or {}).get("type_of_question")
    assert stale == "other", "fixture must start from the weak type"

    # What agents/query_categorizer.py:99 does when it applies.
    parsed["clinical_question_type"] = "treatment"

    # Without the rebuild the payload is unchanged — the inert case.
    assert (parsed.get("picott_json") or {}).get("type_of_question") == "other"

    rebuilt = picott_extractor.build(question, parsed)
    assert rebuilt["type_of_question"] == "therapy"
    assert rebuilt["type_of_question"] != stale


# =====================================================================
# QW2 — framework on the clinician interpretation card
# =====================================================================
def _session_doc(**over):
    doc = {
        "picott_json": {
            "type_of_question": "prognosis",
            "question_scope": "foreground",
            "pico": {"population": ["idiopathic pulmonary fibrosis"]},
            "must_have_concepts": [
                {"concept": "idiopathic pulmonary fibrosis", "role": "condition"},
            ],
        },
        "clinical_question_map_json": {
            "primary_intent": "prognosis_natural_history",
            "secondary_intents": ["prognostic_factor"],
            "frameworks": ["PFO"],
        },
    }
    doc.update(over)
    return doc


def test_qw2_card_carries_framework_and_intent():
    from services.interpreted_question_display import (
        DISPLAY_SCHEMA_VERSION,
        build_interpreted_question_display,
    )

    out = build_interpreted_question_display(_session_doc())
    assert out["question_framework"] == "PFO"
    assert out["question_framework_raw"] == "PFO"
    assert out["question_frameworks"] == ["PFO"]
    assert out["clinical_intent"] == "Prognosis natural history"
    assert out["clinical_intent_raw"] == "prognosis_natural_history"
    assert out["secondary_clinical_intents"] == ["Prognostic factor"]
    assert out["display_schema_version"] == DISPLAY_SCHEMA_VERSION


def test_qw2_acronym_frameworks_keep_their_canonical_casing():
    from services.interpreted_question_display import _humanize_framework

    for acronym in ["PICOTT", "PIRD", "PFO", "PECO", "CHARMS", "CoCoPop", "PICOTS"]:
        assert _humanize_framework(acronym) == acronym


def test_qw2_snake_case_frameworks_are_humanized():
    from services.interpreted_question_display import _humanize_framework

    assert _humanize_framework("concept_mechanism") == "Concept mechanism"
    assert _humanize_framework("guideline_stabilization") == "Guideline stabilization"
    assert (
        _humanize_framework("problem_context_differential")
        == "Problem context differential"
    )


def test_qw2_missing_cqm_degrades_to_not_specified():
    from services.interpreted_question_display import (
        NOT_SPECIFIED,
        build_interpreted_question_display,
    )

    out = build_interpreted_question_display(
        _session_doc(clinical_question_map_json=None)
    )
    assert out["question_framework"] == NOT_SPECIFIED
    assert out["question_framework_raw"] is None
    assert out["question_frameworks"] == []
    assert out["clinical_intent"] == NOT_SPECIFIED


def test_qw2_legacy_only_session_still_gets_the_framework():
    """The CQM is built independently of PICOTT, so the legacy fallback
    render path must carry it too.
    """
    from services.interpreted_question_display import (
        DISPLAY_SCHEMA_VERSION,
        build_interpreted_question_display,
    )

    out = build_interpreted_question_display({
        "picott_json": {},
        "pico_json": {"population": "older adults"},
        "clinical_question_type": "other",
        "clinical_question_map_json": {
            "primary_intent": "epidemiology_prevalence_incidence",
            "frameworks": ["CoCoPop"],
        },
    })
    assert out["has_picott"] is False
    assert out["question_framework"] == "CoCoPop"
    assert out["display_schema_version"] == DISPLAY_SCHEMA_VERSION


def test_qw2_different_frameworks_produce_different_cards():
    """The whole point of the change: a prevalence question and a
    diagnostic-accuracy question must no longer look identical.
    """
    from services.interpreted_question_display import (
        build_interpreted_question_display,
    )

    cocopop = build_interpreted_question_display(_session_doc(
        clinical_question_map_json={
            "primary_intent": "epidemiology_prevalence_incidence",
            "frameworks": ["CoCoPop"],
        },
    ))
    pird = build_interpreted_question_display(_session_doc(
        clinical_question_map_json={
            "primary_intent": "diagnosis_accuracy",
            "frameworks": ["PIRD"],
        },
    ))
    assert cocopop["question_framework"] != pird["question_framework"]
    assert cocopop["clinical_intent"] != pird["clinical_intent"]


def test_qw2_display_schema_version_is_an_int_above_the_implicit_baseline():
    from services.interpreted_question_display import DISPLAY_SCHEMA_VERSION

    assert isinstance(DISPLAY_SCHEMA_VERSION, int)
    assert DISPLAY_SCHEMA_VERSION >= 2
