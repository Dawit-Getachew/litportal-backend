"""Phase J — Bounded AI Clinical Context Interpreter — generalized
question-understanding rescue path. Verifies the additive J1/J2/J3/J5
changes in:

  * services/spelling_normalizer.py
      - typo pair "repalcement" → "replacement" (+ related)
  * services/primary_clinical_anchor.py
      - _MODIFIER_PREFIX_PATTERNS closed-set: "risk of",
        "complications of", "side effects of", "benefits and risks
        of", "complications after", etc.
      - _CLINICAL_SETTING_ONLY_TOKENS closed-set: ICU / intensive
        care / ED / etc., counted in ``_subject_is_generic_only``
        as insufficient by themselves
  * services/clinical_context_resolver.py
      - additive entity_type tokens: complication_or_adverse_event,
        clinical_setting (J3)
      - additive inferred=False default on ResolverPrimaryAnchor (J3)
      - SYSTEM_PROMPT expanded with device-complication and
        therapy+setting guidance

The tests deliberately verify GENERALIZED logic — no exact-string
matches on the two failing queries from the brief.
"""
from __future__ import annotations

import pytest

from services.clinical_context_resolver import (
    ResolverPrimaryAnchor,
    REJECTION_REASONS,
    SYSTEM_PROMPT,
    validate_output,
    _RESOLVER_ENTITY_TYPES,
)
from services.primary_clinical_anchor import (
    primary_clinical_anchor_service as anchor_service,
    _MODIFIER_PREFIX_PATTERNS,
    _CLINICAL_SETTING_ONLY_TOKENS,
)
from services.spelling_normalizer import spelling_normalizer


# ---------------------------------------------------------------------
# J5 — typo handling.
# ---------------------------------------------------------------------


def test_typo_repalcement_normalizes_to_replacement():
    """The brief's mandatory J5 test case."""
    r = spelling_normalizer.normalize(
        "What is the benefits and risks of magnesium repalcement in patients in the ICU?",
    )
    assert "replacement" in r.normalized
    assert "repalcement" not in r.normalized
    corrections = {c["original"].lower() for c in r.corrections}
    assert "repalcement" in corrections


@pytest.mark.parametrize("typo,expected", [
    ("repalcement", "replacement"),
    ("replacment", "replacement"),
    ("repalce", "replace"),
    ("supplemnt", "supplement"),
    ("pacmaker", "pacemaker"),
    ("defribillator", "defibrillator"),
])
def test_typo_pairs_normalize(typo: str, expected: str):
    r = spelling_normalizer.normalize(f"sentence with {typo} in it")
    assert expected in r.normalized
    assert typo not in r.normalized


def test_typo_preserves_doi_pmid_clinical_identifiers():
    """Typo normalization is conservative — clinical identifiers must
    pass through unchanged."""
    r = spelling_normalizer.normalize(
        "Trial nct12345 published doi 10.1056/NEJMoa1234567 about replacement.",
    )
    assert "10.1056/NEJMoa1234567" in r.normalized
    assert "nct12345" in r.normalized


# ---------------------------------------------------------------------
# J1 — anchor over-extraction guard (modifier-prefix patterns).
# ---------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "What is the risk of pacemaker migrations in patients who have a pacemaker?",
    "What is the risk of ICD lead dislodgement?",
    "What is the risk of central line infection?",
    "What are the complications of tracheostomy?",
    "What are the complications after maze procedure?",
    "What are the side effects of long-term steroid use?",
    "What are the benefits and risks of potassium replacement in ICU patients?",
    "What is the safety of rivaroxaban in elderly?",
])
def test_modifier_prefix_yields_no_primary_clinical_entity(question: str):
    """Questions whose definition-subject begins with a modifier
    prefix (risk of / complications of / side effects of / ...) must
    fall through to ``anchor_failure_reason=no_primary_clinical_entity``
    so the bounded LLM resolver is given a chance. Generalized — no
    exact-string matching."""
    out = anchor_service.derive(
        picott={}, ontology_mappings=[], normalized_question=question,
    )
    assert out["anchor_failure_reason"] == "no_primary_clinical_entity", question
    # Must NOT auto-extract a low-quality "modifier + subject" anchor.
    assert not out["primary_anchors"], (question, out["primary_anchors"])


def test_modifier_prefix_constant_is_closed_set_strings():
    assert isinstance(_MODIFIER_PREFIX_PATTERNS, tuple)
    assert _MODIFIER_PREFIX_PATTERNS
    for p in _MODIFIER_PREFIX_PATTERNS:
        assert isinstance(p, str)
        assert p.endswith(" "), f"prefix {p!r} must end with space"


# ---------------------------------------------------------------------
# J2 — clinical_setting-alone insufficient.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "What is the treatment in ICU?",
    "What is the management in the intensive care unit?",
    "What is care in the emergency department?",
])
def test_clinical_setting_alone_does_not_satisfy_subject(question: str):
    """Setting / population alone is insufficient — the J2 rule (3)
    says clinical_setting cannot satisfy the clinical-subject
    requirement. These queries must fall through to
    ``anchor_failure_reason=no_primary_clinical_entity`` so the
    resolver is given a chance to extract the missing intervention
    / problem (or, if the resolver also can't, clarification
    fires)."""
    out = anchor_service.derive(
        picott={}, ontology_mappings=[], normalized_question=question,
    )
    assert out["anchor_failure_reason"] == "no_primary_clinical_entity", question


def test_clinical_setting_token_closed_set_includes_icu():
    assert "icu" in _CLINICAL_SETTING_ONLY_TOKENS
    assert "intensive" in _CLINICAL_SETTING_ONLY_TOKENS
    assert "critically" in _CLINICAL_SETTING_ONLY_TOKENS


# ---------------------------------------------------------------------
# J1 — generic-only guard still fires.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "What is the dose?",
    "What are the recommendations?",
    "What are the side effects?",
    "What is the treatment?",
])
def test_generic_only_fast_path_still_fires(question: str):
    assert anchor_service.is_generic_only(question) is True


# ---------------------------------------------------------------------
# J3 — entity_type closed-set expansion is registered.
# ---------------------------------------------------------------------


def test_complication_or_adverse_event_in_resolver_entity_types():
    assert "complication_or_adverse_event" in _RESOLVER_ENTITY_TYPES


def test_clinical_setting_in_resolver_entity_types():
    assert "clinical_setting" in _RESOLVER_ENTITY_TYPES


def test_resolver_primary_anchor_accepts_new_entity_types():
    """ResolverPrimaryAnchor.entity_type Literal[] enum must allow
    the two additive J3 tokens via Pydantic validation."""
    a = ResolverPrimaryAnchor(
        text="pacemaker migration",
        canonical_concept="pacemaker migration",
        entity_type="complication_or_adverse_event",
        role="outcomes_target_diagnosis",
        required_for_search=True,
        missing_relevance_cap=50,
    )
    assert a.entity_type == "complication_or_adverse_event"
    a2 = ResolverPrimaryAnchor(
        text="ICU",
        canonical_concept="intensive care unit",
        entity_type="clinical_setting",
        role="setting",
        required_for_search=True,
        missing_relevance_cap=50,
    )
    assert a2.entity_type == "clinical_setting"


def test_resolver_primary_anchor_inferred_field_defaults_false():
    a = ResolverPrimaryAnchor(
        text="hypomagnesemia",
        canonical_concept="hypomagnesemia",
        entity_type="disease_or_problem",
        role="population_problem",
        required_for_search=False,
        missing_relevance_cap=50,
    )
    assert a.inferred is False


def test_resolver_primary_anchor_inferred_field_accepts_true():
    a = ResolverPrimaryAnchor(
        text="hypomagnesemia",
        canonical_concept="hypomagnesemia",
        entity_type="disease_or_problem",
        role="population_problem",
        required_for_search=False,
        missing_relevance_cap=50,
        inferred=True,
    )
    assert a.inferred is True


def test_validate_output_accepts_dual_anchor_device_complication():
    """Round-trip: a closed-set device-complication-risk resolver
    output validates through the scrubber + Pydantic + closed-set
    enum checks."""
    raw = {
        "is_valid_clinical_question": True,
        "question_scope": "foreground",
        "type_of_question": "device_complication_risk",
        "question_subtype": None,
        "specialty": ["cardiology"],
        "picott": {
            "population_problem": ["pacemaker", "cardiac implantable electronic device"],
            "intervention_exposure_index_test": [],
            "comparator_reference_standard": [],
            "outcomes_target_diagnosis": ["pacemaker migration", "lead displacement"],
            "time_horizon": [],
            "setting": [],
            "clinical_action": [],
        },
        "primary_anchors": [
            {
                "text": "pacemaker",
                "canonical_concept": "cardiac pacemaker",
                "entity_type": "device_or_implant",
                "role": "population_problem",
                "required_for_search": True,
                "missing_relevance_cap": 30,
            },
            {
                "text": "pacemaker migration",
                "canonical_concept": "pacemaker migration",
                "entity_type": "complication_or_adverse_event",
                "role": "outcomes_target_diagnosis",
                "required_for_search": True,
                "missing_relevance_cap": 40,
            },
        ],
        "generic_modifiers": [],
        "must_have_concepts": [],
        "ranking_concepts": [],
        "negative_concepts": [],
        "clarification_needed": False,
        "clarification_message": None,
        "refinement_note": None,
        "confidence": 0.85,
        "rationale": "Device complication risk question.",
    }
    out, rej = validate_output(raw)
    assert rej is None, rej
    assert out is not None
    assert len(out.primary_anchors) == 2
    by_type = {a.entity_type for a in out.primary_anchors}
    assert "device_or_implant" in by_type
    assert "complication_or_adverse_event" in by_type


def test_validate_output_accepts_therapy_in_setting_with_inferred():
    """Therapy+setting shape with an inferred problem anchor (J3
    feature)."""
    raw = {
        "is_valid_clinical_question": True,
        "question_scope": "foreground",
        "type_of_question": "electrolyte_replacement",
        "question_subtype": None,
        "specialty": ["critical care"],
        "picott": {
            "population_problem": ["ICU patients", "critically ill"],
            "intervention_exposure_index_test": ["magnesium replacement", "magnesium sulfate"],
            "comparator_reference_standard": [],
            "outcomes_target_diagnosis": [],
            "time_horizon": [],
            "setting": ["intensive care unit"],
            "clinical_action": [],
        },
        "primary_anchors": [
            {
                "text": "magnesium replacement",
                "canonical_concept": "magnesium replacement",
                "entity_type": "management_intervention",
                "role": "intervention_exposure_index_test",
                "required_for_search": True,
                "missing_relevance_cap": 30,
            },
            {
                "text": "ICU",
                "canonical_concept": "intensive care unit",
                "entity_type": "clinical_setting",
                "role": "setting",
                "required_for_search": True,
                "missing_relevance_cap": 50,
            },
            {
                "text": "hypomagnesemia",
                "canonical_concept": "hypomagnesemia",
                "entity_type": "disease_or_problem",
                "role": "population_problem",
                "required_for_search": False,
                "missing_relevance_cap": 50,
                "inferred": True,
            },
        ],
        "generic_modifiers": [],
        "must_have_concepts": [],
        "ranking_concepts": [],
        "negative_concepts": [],
        "clarification_needed": False,
        "clarification_message": None,
        "refinement_note": None,
        "confidence": 0.82,
        "rationale": "Electrolyte replacement in ICU.",
    }
    out, rej = validate_output(raw)
    assert rej is None, rej
    assert out is not None
    inferred = [a for a in out.primary_anchors if a.inferred]
    assert len(inferred) == 1
    assert inferred[0].canonical_concept == "hypomagnesemia"


# ---------------------------------------------------------------------
# SYSTEM_PROMPT expansion sanity.
# ---------------------------------------------------------------------


def test_system_prompt_mentions_device_complication_guidance():
    assert "DEVICE-COMPLICATION" in SYSTEM_PROMPT or "device_complication_risk" in SYSTEM_PROMPT


def test_system_prompt_mentions_therapy_in_setting_guidance():
    assert "THERAPY+SETTING" in SYSTEM_PROMPT or "electrolyte_replacement" in SYSTEM_PROMPT


def test_system_prompt_mentions_clinical_setting_alone_insufficient():
    s = SYSTEM_PROMPT
    assert "clinical_setting" in s
    assert "INSUFFICIENT" in s or "insufficient" in s


def test_system_prompt_mentions_inferred_flag():
    assert "inferred" in SYSTEM_PROMPT


# ---------------------------------------------------------------------
# Regression — existing deterministic paths preserved.
# ---------------------------------------------------------------------


def test_regression_rituximab_asct_stays_deterministic():
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question=(
            "recommended dose of rituximab after autologous stem cell transplant"
        ),
    )
    assert out["anchor_failure_reason"] is None
    canonicals = {a["canonical_concept"].lower() for a in out["primary_anchors"]}
    assert "autologous stem cell transplant" in canonicals
    assert "rituximab" in canonicals


def test_regression_brugada_perioperative_stays_deterministic():
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question=(
            "perioperative management of patients with Brugada syndrome"
        ),
    )
    assert out["anchor_failure_reason"] is None
    canonicals = {a["canonical_concept"].lower() for a in out["primary_anchors"]}
    assert "brugada syndrome" in canonicals


def test_regression_psoriasis_disease_overview_stays_deterministic():
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question="What is psoriasis?",
    )
    assert out["anchor_failure_reason"] is None
    canonicals = {a["canonical_concept"].lower() for a in out["primary_anchors"]}
    assert "psoriasis" in canonicals


def test_regression_acne_disease_overview_stays_deterministic():
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question="What is acne vulgaris?",
    )
    assert out["anchor_failure_reason"] is None


def test_regression_brugada_drug_avoidance_stays_deterministic():
    """The cardiology pack regression test E from the brief — Brugada
    drug-avoidance MUST remain on the deterministic path."""
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question=(
            "medications to avoid in Brugada syndrome"
        ),
    )
    assert out["anchor_failure_reason"] is None
    canonicals = {a["canonical_concept"].lower() for a in out["primary_anchors"]}
    assert "brugada syndrome" in canonicals


@pytest.mark.parametrize("passive_phrasing", [
    "What medications should be avoided in Brugada syndrome?",
    "What drugs should be avoided in Brugada syndrome?",
    "What medications should be used in Brugada syndrome?",
    "What medicines should be considered in patients with Brugada syndrome?",
    "What anesthetics should be avoided in Brugada syndrome?",
    "What medications should be avoided in long QT syndrome?",
    "What drugs should be avoided in CPVT?",
])
def test_passive_voice_medication_avoidance_routes_deterministic(passive_phrasing: str):
    """Phase J — passive-voice medication-avoidance phrasings for
    inherited cardiac arrhythmia syndromes must hit the deterministic
    post-procedure router (subdomain=perioperative_medication_management)
    and yield dual anchors (condition + management). Generalized — no
    exact-string matching on a single failing query."""
    from services.post_procedure_router import post_procedure_router
    pp = post_procedure_router.detect(passive_phrasing)
    assert pp["matched"] is True, passive_phrasing
    assert pp["subdomain"] == "perioperative_medication_management"
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question=passive_phrasing,
    )
    assert out["anchor_failure_reason"] is None, passive_phrasing
    canonicals = {a["canonical_concept"].lower() for a in out["primary_anchors"]}
    # Either Brugada syndrome, long QT syndrome, or CPVT must appear.
    assert any(c in canonicals for c in (
        "brugada syndrome",
        "long qt syndrome",
        "catecholaminergic polymorphic ventricular tachycardia",
    )), (passive_phrasing, canonicals)
    assert "perioperative medication management" in canonicals


def test_passive_voice_pattern_does_not_match_non_arrhythmia_conditions():
    """Over-matching guard: the passive-voice medication-avoidance
    pattern is gated by ``_looks_like_inherited_arrhythmia`` so it
    must NOT fire on conditions outside the closed-set allowlist."""
    from services.post_procedure_router import post_procedure_router
    pp = post_procedure_router.detect(
        "What medications should be avoided in pregnancy?",
    )
    assert pp["matched"] is False


# ---------------------------------------------------------------------
# Closed-set rejection enum stays unchanged.
# ---------------------------------------------------------------------


def test_rejection_reasons_closed_set_unchanged():
    expected = frozenset({
        "schema_invalid",
        "missing_primary_anchor_for_foreground",
        "confidence_below_threshold",
        "contains_pmid_doi_pattern",
        "contains_article_reference",
        "contains_answer_text",
        "contains_phi_pattern",
        "references_unapproved_source",
        "enum_violation",
        "empty_output",
    })
    assert REJECTION_REASONS == expected
