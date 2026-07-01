"""Phase 6G.1 — background clinical entity anchoring + disease-definition
query repair tests.

Covers:
  * "What is acne vulgaris?" no longer triggers clarification.
  * Pack-resolved, acronym-resolved, brand-resolved, and conservative
    fallback anchors all produce the expected primary anchor.
  * The 5 generic-only questions in the Phase 6G.1 spec still fire the
    fast-path (``is_generic_only`` returns True) → 0 LLM + 0 source.
  * Background-definition picott_overrides flow through the anchor
    service into the persisted picott_json AFTER step 4 ontology
    repair has already produced an anchor.
"""
from __future__ import annotations

import pytest

from services.primary_clinical_anchor import (
    _extract_definition_subject,
    _resolve_background_anchor,
    _subject_is_generic_only,
    primary_clinical_anchor_service as svc,
)
from services.picott_term_mapping_service import (
    build_picott_term_mapping,
)


# ---------------------------------------------------------------------
# Section A — background-definition anchor resolution (pure unit)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "question, expected_subject",
    [
        ("What is acne vulgaris?", "acne vulgaris"),
        ("What is psoriasis?", "psoriasis"),
        ("What is septic shock?", "septic shock"),
        ("What is D-dimer?", "D-dimer"),
        ("What is Seroquel?", "Seroquel"),
        ("What is varicella?", "varicella"),
        ("What is ARDS?", "ARDS"),
        ("What is atrial fibrillation?", "atrial fibrillation"),
        ("Define HIV", "HIV"),
        ("Tell me about psoriasis", "psoriasis"),
        ("Overview of acne vulgaris", "acne vulgaris"),
        ("Explain atrial fibrillation", "atrial fibrillation"),
    ],
)
def test_extract_definition_subject(question, expected_subject):
    assert _extract_definition_subject(question) == expected_subject


@pytest.mark.parametrize(
    "subject, expected_canonical, expected_entity_type, min_confidence",
    [
        ("acne vulgaris", "acne vulgaris", "disease_or_problem", 0.85),
        ("psoriasis", "psoriasis", "disease_or_problem", 0.85),
        ("ARDS", "acute respiratory distress syndrome",
         "disease_or_problem", 0.85),
        ("HIV", "human immunodeficiency virus", "disease_or_problem", 0.85),
        ("Seroquel", "quetiapine", "drug_or_drug_class", 0.85),
        # Conservative noun-phrase fallback.
        ("atrial fibrillation", "atrial fibrillation",
         "disease_or_problem", 0.55),
        ("septic shock", "septic shock", "disease_or_problem", 0.55),
        ("varicella", "varicella", "disease_or_problem", 0.55),
        ("D-dimer", "d-dimer", "disease_or_problem", 0.55),
    ],
)
def test_resolve_background_anchor_success(
    subject, expected_canonical, expected_entity_type, min_confidence,
):
    res = _resolve_background_anchor(subject)
    assert res is not None, f"resolver failed for {subject!r}"
    anchor, overrides = res
    assert anchor.canonical_concept == expected_canonical
    assert anchor.entity_type == expected_entity_type
    assert anchor.confidence >= min_confidence
    assert anchor.required_for_search is True
    # Background overrides MUST always tag the question scope.
    assert overrides.get("type_of_question") == "background"
    assert overrides.get("question_scope") == "background"


def test_resolve_background_anchor_rejects_generic_subjects():
    for s in (
        "treatment", "first-line treatment", "second-line treatment",
        "side effects", "the dose", "benefit", "the benefit",
        "diagnostic accuracy", "diagnosis", "the recommended age",
    ):
        assert _resolve_background_anchor(s) is None, (
            f"resolver MUST refuse generic subject {s!r}"
        )


# ---------------------------------------------------------------------
# Section B — generic-only guard preservation (P0 regression)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What are the side effects?",
        "What is first-line treatment?",
        "What is the dose?",
        "What is the recommended age?",
        "What is diagnostic accuracy?",
        "What is treatment?",
        "What is the benefit?",
        "What are side effects?",
    ],
)
def test_generic_only_guard_still_fires(question):
    """The Phase 6F.1 fast-path MUST still short-circuit these
    questions so the pipeline emits ``awaiting_clinical_subject``
    with 0 LLM + 0 source. Adding the background-definition repair
    must NEVER weaken this."""
    assert svc.is_generic_only(question) is True, (
        f"generic-only guard regressed for {question!r}"
    )


@pytest.mark.parametrize(
    "question",
    [
        "What is acne vulgaris?",
        "What is psoriasis?",
        "What is septic shock?",
        "What is D-dimer?",
        "What is Seroquel?",
        "What is ARDS?",
        "What is varicella?",
        "What is atrial fibrillation?",
    ],
)
def test_background_definition_NOT_generic_only(question):
    """The questions WE WANT to repair must NOT trip the
    is_generic_only fast-path (otherwise we'd never reach the
    background anchor resolver)."""
    assert svc.is_generic_only(question) is False, (
        f"is_generic_only should not fire for {question!r}"
    )


def test_subject_is_generic_only_helper():
    for s in (
        "treatment", "first-line treatment", "the dose", "side effects",
        "the benefit",
    ):
        assert _subject_is_generic_only(s) is True
    for s in ("acne vulgaris", "psoriasis", "atrial fibrillation",
              "D-dimer"):
        assert _subject_is_generic_only(s) is False


# ---------------------------------------------------------------------
# Section C — anchor service integration (derive() with empty
#             must_have_concepts triggers the background repair).
# ---------------------------------------------------------------------


def test_derive_background_repair_creates_anchor():
    out = svc.derive(
        picott={"must_have_concepts": []},
        ontology_mappings=[],
        normalized_question="What is acne vulgaris?",
    )
    pas = out["primary_anchors"]
    assert pas, "background repair should have produced an anchor"
    assert pas[0]["canonical_concept"] == "acne vulgaris"
    assert pas[0]["entity_type"] == "disease_or_problem"
    assert out["anchor_failure_reason"] is None
    assert out["picott_overrides"]["question_subtype"] == "disease_overview"
    assert out["picott_overrides"]["specialty"] == "dermatology"


def test_derive_step5b_overrides_when_anchor_already_present():
    """Even when must_have_concepts already supplies the anchor (step
    1 path), the background-definition pattern must still propagate
    the specialty + subtype overrides so the legacy "primary_care"
    leak is corrected."""
    out = svc.derive(
        picott={
            "must_have_concepts": [{
                "concept": "acne vulgaris", "role": "condition",
                "synonyms": ["acne"],
                "required_for_high_relevance": True,
            }],
            "specialty": "primary_care",
            "question_subtype": "",
        },
        ontology_mappings=[],
        normalized_question="What is acne vulgaris?",
    )
    assert out["picott_overrides"].get("specialty") == "dermatology"
    assert out["picott_overrides"].get("question_subtype") == "disease_overview"


def test_derive_generic_only_NOT_routed_through_background_repair():
    """When the question is generic-only (e.g. "What is treatment?")
    AND must_have_concepts is empty, the background repair MUST NOT
    fabricate an anchor. The pipeline-level ``is_generic_only`` would
    short-circuit anyway, but the resolver itself must also refuse."""
    out = svc.derive(
        picott={"must_have_concepts": []},
        ontology_mappings=[],
        normalized_question="What is treatment?",
    )
    assert out["primary_anchors"] == []
    assert out["anchor_failure_reason"] == "no_primary_clinical_entity"


# ---------------------------------------------------------------------
# Section D — Phase 6G term-mapping projection support for
#             disease_overview subtype.
# ---------------------------------------------------------------------


def test_term_mapping_disease_overview_subtype():
    session_doc = {
        "picott_json": {
            "type_of_question": "background",
            "question_subtype": "disease_overview",
            "specialty": "dermatology",
            "question_scope": "background",
            "must_have_concepts": [
                {"concept": "acne vulgaris", "role": "condition",
                 "synonyms": ["acne"],
                 "required_for_high_relevance": True},
            ],
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [{
                "text": "acne vulgaris",
                "canonical_concept": "acne vulgaris",
                "entity_type": "disease_or_problem",
                "role": "population",
                "synonyms": ["acne"],
                "confidence": 0.99,
                "required_for_search": True,
            }],
            "generic_modifiers": [],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
    }
    m = build_picott_term_mapping(session_doc, queries=[])
    assert m.completeness.has_primary_anchor is True
    assert m.completeness.has_required_role_mapping is True
    assert m.completeness.missing_required_roles == []
    assert m.completeness.safe_to_search is True
    assert m.completeness.clarification_needed is False
    pop = [e.canonical_concept for e in m.fields["population_problem"]]
    assert "acne vulgaris" in pop
    # Synthetic "definition / disease overview" entry under
    # clinical_action.
    ca = [e.canonical_concept for e in m.fields["clinical_action"]]
    assert any("definition" in c.lower() or "overview" in c.lower()
               for c in ca)
    # Empty comparator / time / setting MUST NOT break completeness.
    for r in ("comparator_reference_standard", "time_horizon", "setting"):
        assert m.fields[r] == []


def test_term_mapping_required_roles_for_disease_overview():
    """The required-roles table extension MUST list only
    population_problem for disease_overview. Other roles are
    informative but not required for completeness."""
    from services.picott_term_mapping_service import (
        _REQUIRED_ROLES_BY_SUBTYPE,
    )
    assert _REQUIRED_ROLES_BY_SUBTYPE["disease_overview"] == [
        "population_problem",
    ]
    assert _REQUIRED_ROLES_BY_SUBTYPE["definition"] == ["population_problem"]
    assert _REQUIRED_ROLES_BY_SUBTYPE["drug_overview"] == [
        "intervention_exposure_index_test",
    ]


# ---------------------------------------------------------------------
# Section E — dermatology / acne pack content checks
# ---------------------------------------------------------------------


def test_dermatology_acne_pack_loaded():
    from services.specialty_packs import all_packs
    packs = all_packs()
    assert "dermatology_acne" in packs
    pack = packs["dermatology_acne"]
    assert pack.get("specialty") == "dermatology"
    groups = pack.get("concept_groups") or {}
    # Disease group must exist with the required entries.
    assert "acne_disease" in groups
    concepts = {
        (c.get("concept") or "").lower()
        for c in (groups["acne_disease"].get("concepts") or [])
    }
    for required in (
        "acne vulgaris", "acne", "inflammatory acne",
        "comedonal acne", "nodulocystic acne",
    ):
        assert required in concepts, f"missing {required!r} in acne pack"
    # Related-but-distinct group must exist with rosacea + chloracne.
    rb = groups["acne_related_but_distinct"]
    rb_concepts = {
        (c.get("concept") or "").lower()
        for c in (rb.get("concepts") or [])
    }
    assert "rosacea" in rb_concepts
    assert "chloracne" in rb_concepts


def test_acne_pack_rosacea_is_capped_via_pack_metadata():
    """Rosacea / chloracne must declare the
    ``related_but_distinct_from = acne vulgaris`` + ``related_but_distinct_cap``
    fields so the downstream ranker can apply the moderate
    related-but-distinct cap (≤50). The cap itself is enforced by the
    ranker; this test only enforces the pack metadata contract."""
    from services.specialty_packs import all_packs
    rb = (all_packs()["dermatology_acne"]["concept_groups"]
          ["acne_related_but_distinct"]["concepts"])
    by_concept = {c["concept"].lower(): c for c in rb}
    for concept in ("rosacea", "chloracne", "hidradenitis suppurativa"):
        c = by_concept[concept]
        assert c.get("related_but_distinct_from") == "acne vulgaris"
        assert c.get("related_but_distinct_cap") == 50
