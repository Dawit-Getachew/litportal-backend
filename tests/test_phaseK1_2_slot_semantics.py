"""Phase K1.2 — narrow slot semantics patch.

Two extractor fixes:

  Fix 1 (Age/population qualifier extraction): when a deterministic
  parser / concept pack folds a demographic qualifier ("in adults",
  "in older adults", "in children", "in pregnancy", ...) into the
  disease phrase, the K1.2 normalizer splits it back out into a
  separate population/age_group/pregnancy slot AND strips it from the
  disease slot's ``original_phrase``.

  Fix 2 (Generic modifier mis-extraction): tokens such as "benefits",
  "risks", "outcomes", "complications" (when used as a generic
  modifier), "recommendations", "management", "treatment" MUST NEVER
  appear as ``slot_role=population`` slots. They are filtered out of
  required / optional / display buckets.

  Overcorrection guard: specific complications/outcomes ("infection",
  "mortality", "bleeding", "stroke", "lead dislodgement", "pacemaker
  migration") are NOT in the deny-list and MUST still extract as
  legitimate outcome / complication slots.

All tests are pure / sync. They drive the closed-set
``_k1_build_clinical_question_map`` directly with closed-set anchor
payloads so no live LLM is required in CI. Live preview re-verification
is captured separately in the K1.2 memory report.
"""
from __future__ import annotations

import pytest

from services.clinical_question_map import (
    DEMOGRAPHIC_QUALIFIER_LEXICON,
    GENERIC_MODIFIER_DENY_LIST,
    is_generic_modifier_token,
    normalize_slot_semantics,
    split_demographic_qualifier_from_disease_phrase,
)


# ---------------------------------------------------------------------
# Helpers — closed-set anchor / slot payload builders.
# ---------------------------------------------------------------------


def _anchor(
    *,
    canonical: str,
    text: str | None = None,
    entity_type: str = "disease_or_problem",
    required: bool = True,
    confidence: float = 0.9,
    source: str = "deterministic_concept_pack",
    inferred: bool = False,
    synonyms: list[str] | None = None,
) -> dict:
    return {
        "canonical_concept": canonical,
        "entity_type": entity_type,
        "required_for_search": required,
        "confidence": confidence,
        "source": source,
        "inferred": inferred,
        "synonyms": list(synonyms or []),
        "text": text if text is not None else canonical,
    }


def _build_cqm(
    question: str,
    anchors: list[dict],
    *,
    generic_modifiers: list[str] | None = None,
    type_of_question: str = "background",
    question_subtype: str | None = None,
    must_have_concepts: list[dict] | None = None,
    resolver_called: bool = False,
    resolver_accepted: bool = False,
) -> dict:
    from services.clinical_signal_router import clinical_signal_router
    from services.pipeline import _k1_build_clinical_question_map

    if must_have_concepts is None:
        must_have_concepts = [
            {
                "concept": a["canonical_concept"],
                "role": "condition",
                "synonyms": list(a.get("synonyms") or []),
                "required_for_high_relevance": bool(a.get("required_for_search")),
            }
            for a in anchors
        ]
    return _k1_build_clinical_question_map(
        normalized_question=question.lower(),
        original_question=question,
        picott_payload={
            "type_of_question": type_of_question,
            "question_subtype": question_subtype,
            "must_have_concepts": must_have_concepts,
            "pico": {},
            "question_scope": (
                "background" if type_of_question == "background"
                else "foreground"
            ),
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": anchors,
            "generic_modifiers": list(generic_modifiers or []),
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
        resolver_diag={
            "called": resolver_called,
            "accepted": resolver_accepted,
            "resolver_confidence": 0.85 if resolver_accepted else None,
        },
        signal_summary=clinical_signal_router.analyze(
            question
        ).model_dump(mode="json"),
        clarification_only=False,
    )


def _slots(cqm: dict) -> list[dict]:
    return list(cqm.get("framework_neutral_slots") or [])


def _has(
    slots: list[dict],
    *,
    role: str | set[str] | None = None,
    role_in: set[str] | None = None,
    phrase_contains: str | None = None,
    canonical_contains: str | None = None,
    not_phrase_contains: str | None = None,
) -> bool:
    """Generalised matcher — closed-set fields only."""
    target_roles: set[str] | None = None
    if isinstance(role, str):
        target_roles = {role}
    elif isinstance(role, set):
        target_roles = role
    elif role_in:
        target_roles = role_in
    for s in slots:
        if target_roles and s.get("slot_role") not in target_roles:
            continue
        op = (s.get("original_phrase") or "").lower()
        cc = (s.get("canonical_concept") or "").lower()
        if phrase_contains and phrase_contains.lower() not in op:
            continue
        if canonical_contains and canonical_contains.lower() not in cc:
            continue
        if not_phrase_contains and not_phrase_contains.lower() in op:
            continue
        return True
    return False


# ---------------------------------------------------------------------
# Closed-set invariants — module-level constants.
# ---------------------------------------------------------------------


def test_deny_list_contains_required_generic_modifiers() -> None:
    for tok in [
        "benefits", "risks", "side effects", "adverse effects",
        "outcomes", "complications", "recommendations",
        "management", "treatment",
    ]:
        assert tok in GENERIC_MODIFIER_DENY_LIST, tok


def test_deny_list_does_not_contain_specific_clinical_entities() -> None:
    # Overcorrection guard — these MUST NOT be in the deny-list.
    for tok in [
        "infection", "mortality", "bleeding", "stroke",
        "lead dislodgement", "pacemaker migration",
        "myocardial infarction", "death",
    ]:
        assert tok not in GENERIC_MODIFIER_DENY_LIST, (
            f"specific clinical entity {tok!r} must NOT be in deny-list"
        )


def test_demographic_lexicon_covers_required_set() -> None:
    canonicals = {canon for (_p, canon, _r) in DEMOGRAPHIC_QUALIFIER_LEXICON}
    for required in {
        "adult", "older adults", "children", "pediatric",
        "adolescents", "infants", "neonates", "pregnancy",
    }:
        assert required in canonicals, required


# ---------------------------------------------------------------------
# Pure helper tests.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "benefits", "Benefits", "BENEFITS",
    "risks", "side effects", "adverse effects",
    "outcomes", "complications", "recommendations",
    "management", "treatment",
])
def test_is_generic_modifier_token_positive(text: str) -> None:
    assert is_generic_modifier_token(text) is True


@pytest.mark.parametrize("text", [
    "infection", "mortality", "stroke", "bleeding",
    "lead dislodgement", "pacemaker migration",
    "atrial fibrillation", "magnesium replacement",
    "ICU", "pediatric",
])
def test_is_generic_modifier_token_negative(text: str) -> None:
    assert is_generic_modifier_token(text) is False


@pytest.mark.parametrize("phrase,expected_stripped,expected_canon,expected_role", [
    ("prevalence of atrial fibrillation in adults",
     "prevalence of atrial fibrillation", "adult", "population"),
    ("atrial fibrillation in older adults",
     "atrial fibrillation", "older adults", "age_group"),
    ("asthma in children",
     "asthma", "children", "age_group"),
    ("anticoagulation in pregnancy",
     "anticoagulation", "pregnancy", "pregnancy"),
    ("UTI in pediatric patients",
     "UTI", "pediatric", "age_group"),
    ("dosing among elderly",
     "dosing", "older adults", "age_group"),
])
def test_split_demographic_qualifier(
    phrase: str,
    expected_stripped: str,
    expected_canon: str,
    expected_role: str,
) -> None:
    stripped, qual = split_demographic_qualifier_from_disease_phrase(phrase)
    assert qual is not None, f"no qualifier extracted from {phrase!r}"
    canon, role, _orig = qual
    assert canon == expected_canon
    assert role == expected_role
    assert stripped.lower() == expected_stripped.lower()


def test_split_demographic_qualifier_no_match_returns_none() -> None:
    stripped, qual = split_demographic_qualifier_from_disease_phrase(
        "atrial fibrillation"
    )
    assert qual is None
    assert stripped == "atrial fibrillation"


def test_normalize_drops_generic_modifier_population_slot() -> None:
    required = [
        {
            "original_phrase": "benefits",
            "canonical_concept": "benefits",
            "entity_type": "population_group",
            "slot_role": "population",
            "required_for_search": True,
            "slot_search_policy": "required_for_retrieval",
            "inferred": False, "confidence": 0.7,
            "synonyms": [], "search_terms": [],
            "source": "deterministic_parser",
        },
    ]
    new_req, _new_opt, _new_disp = normalize_slot_semantics(
        required_slots=required, optional_slots=[], display_only_slots=[],
    )
    assert new_req == []


def test_normalize_preserves_specific_outcome_slot() -> None:
    # ``mortality`` with slot_role=outcome MUST NEVER be dropped — it's
    # a legitimate clinical outcome anchor, not a generic modifier.
    optional = [
        {
            "original_phrase": "mortality",
            "canonical_concept": "mortality",
            "entity_type": "biomarker_or_lab",
            "slot_role": "outcome",
            "required_for_search": False,
            "slot_search_policy": "optional_for_retrieval",
            "inferred": False, "confidence": 0.85,
            "synonyms": [], "search_terms": [],
            "source": "bounded_resolver_fallback",
        },
    ]
    _new_req, new_opt, _new_disp = normalize_slot_semantics(
        required_slots=[], optional_slots=optional, display_only_slots=[],
    )
    assert len(new_opt) == 1
    assert new_opt[0]["canonical_concept"] == "mortality"
    assert new_opt[0]["slot_role"] == "outcome"


# ---------------------------------------------------------------------
# Case A — Epidemiology age qualifier ("AF in adults").
# ---------------------------------------------------------------------


def test_case_A_af_prevalence_in_adults_splits_demographic() -> None:
    cqm = _build_cqm(
        question="What is the prevalence of atrial fibrillation in adults?",
        anchors=[
            _anchor(
                canonical="prevalence of atrial fibrillation in adults",
                text="prevalence of atrial fibrillation in adults",
                entity_type="disease_or_problem",
                required=True,
                source="deterministic_parser",
            ),
            _anchor(
                canonical="atrial fibrillation",
                text="atrial fibrillation",
                entity_type="disease_or_problem",
                required=True,
                source="deterministic_concept_pack",
            ),
        ],
    )
    slots = _slots(cqm)
    # 1. primary_intent stays epidemiology.
    assert cqm["primary_intent"] == "epidemiology_prevalence_incidence"
    assert "CoCoPop" in cqm["frameworks"]
    # 2. At least 2 slots total.
    assert len(slots) >= 2
    # 3. Disease slot present, atrial fibrillation in original_phrase,
    #    NO "in adults" remnant.
    assert _has(
        slots,
        role_in={"disease_or_condition", "clinical_problem"},
        phrase_contains="atrial fibrillation",
        not_phrase_contains="in adults",
    )
    # 4. Population/age_group slot present with "adult" content.
    assert _has(
        slots,
        role_in={"population", "age_group"},
        canonical_contains="adult",
    )


# ---------------------------------------------------------------------
# Case B — Older adults.
# ---------------------------------------------------------------------


def test_case_B_af_prevalence_in_older_adults_splits_demographic() -> None:
    cqm = _build_cqm(
        question="What is the prevalence of atrial fibrillation in older adults?",
        anchors=[
            _anchor(
                canonical="atrial fibrillation in older adults",
                text="atrial fibrillation in older adults",
                entity_type="disease_or_problem",
                required=True,
                source="deterministic_parser",
            ),
        ],
    )
    slots = _slots(cqm)
    assert _has(
        slots,
        role_in={"disease_or_condition", "clinical_problem"},
        phrase_contains="atrial fibrillation",
        not_phrase_contains="older adults",
    )
    assert _has(
        slots,
        role_in={"age_group", "population"},
        canonical_contains="older adults",
    )


# ---------------------------------------------------------------------
# Case C — Pediatric / children.
# ---------------------------------------------------------------------


def test_case_C_asthma_in_children_splits_demographic() -> None:
    cqm = _build_cqm(
        question="What is the prevalence of asthma in children?",
        anchors=[
            _anchor(
                canonical="asthma in children",
                text="asthma in children",
                entity_type="disease_or_problem",
                required=True,
                source="deterministic_parser",
            ),
        ],
    )
    slots = _slots(cqm)
    assert _has(
        slots,
        role_in={"disease_or_condition", "clinical_problem"},
        phrase_contains="asthma",
        not_phrase_contains="in children",
    )
    assert _has(
        slots,
        role_in={"age_group", "population"},
        canonical_contains="children",
    )


# ---------------------------------------------------------------------
# Case D — Magnesium ICU generic modifier filtering.
# ---------------------------------------------------------------------


def test_case_D_magnesium_icu_does_not_emit_benefits_as_population() -> None:
    """Even if a downstream parser hands back ``benefits`` as a
    ``slot_role=population`` anchor, the K1.2 normalizer MUST drop it.
    The intervention + ICU setting anchors remain untouched.
    """
    cqm = _build_cqm(
        question="What are the benefits and risks of magnesium replacement in ICU patients?",
        type_of_question="therapy",
        question_subtype="drug_safety",
        anchors=[
            _anchor(
                canonical="magnesium replacement",
                text="magnesium replacement",
                entity_type="management_intervention",
                required=True,
                source="bounded_resolver_fallback",
            ),
            _anchor(
                canonical="intensive care unit",
                text="ICU",
                entity_type="clinical_setting",
                required=True,
                source="bounded_resolver_fallback",
            ),
            # Defensive payload — the downstream parser SOMETIMES
            # emits ``benefits`` as a population anchor (verified live
            # in tester session eed5ee6e-…). The K1.2 fix MUST drop
            # this.
            _anchor(
                canonical="benefits",
                text="benefits",
                entity_type="population_group",
                required=False,
                source="deterministic_parser",
            ),
        ],
        resolver_called=True,
        resolver_accepted=True,
    )
    slots = _slots(cqm)
    # 1. No slot has BOTH slot_role=population AND original_phrase or
    #    canonical containing "benefits".
    for s in slots:
        op = (s.get("original_phrase") or "").lower()
        cc = (s.get("canonical_concept") or "").lower()
        if s.get("slot_role") == "population":
            assert "benefits" not in op
            assert "benefits" not in cc
            assert "risks" not in op
            assert "risks" not in cc
    # 2. Intervention slot present.
    assert _has(
        slots,
        role_in={"intervention", "drug_or_drug_class"},
        phrase_contains="magnesium",
    )
    # 3. ICU setting slot present.
    assert _has(
        slots,
        role_in={"clinical_setting", "population"},
        phrase_contains="icu",
    ) or _has(
        slots,
        role_in={"clinical_setting"},
        canonical_contains="intensive care",
    )
    # 4. If benefits/risks appear at all (e.g. as display_only), NONE
    #    are ``required_for_retrieval``.
    for s in slots:
        op = (s.get("original_phrase") or "").lower()
        cc = (s.get("canonical_concept") or "").lower()
        if op in {"benefits", "risks"} or cc in {"benefits", "risks"}:
            assert s.get("slot_search_policy") != "required_for_retrieval"
            assert s.get("required_for_search") is False


# ---------------------------------------------------------------------
# Case E — True outcome must still work (mortality).
# ---------------------------------------------------------------------


def test_case_E_troponin_sepsis_mortality_outcome_preserved() -> None:
    cqm = _build_cqm(
        question="Does troponin elevation predict mortality in sepsis?",
        type_of_question="prognosis",
        question_subtype="prognostic_factor",
        anchors=[
            _anchor(
                canonical="sepsis",
                text="sepsis",
                entity_type="disease_or_problem",
                required=True,
                source="deterministic_concept_pack",
            ),
            _anchor(
                canonical="troponin elevation",
                text="troponin elevation",
                entity_type="biomarker_or_lab",
                required=True,
                source="deterministic_parser",
            ),
            _anchor(
                canonical="mortality",
                text="mortality",
                entity_type="biomarker_or_lab",
                required=True,
                source="deterministic_parser",
            ),
        ],
    )
    slots = _slots(cqm)
    # sepsis still a clinical problem / disease.
    assert _has(
        slots,
        role_in={"disease_or_condition", "clinical_problem"},
        phrase_contains="sepsis",
    )
    # mortality still present — NOT dropped by the deny-list.
    assert _has(slots, phrase_contains="mortality")
    # mortality should NOT be classified as population.
    for s in slots:
        if (s.get("canonical_concept") or "").lower() == "mortality":
            assert s.get("slot_role") != "population"


# ---------------------------------------------------------------------
# Case F — True complication must still work (pacemaker infection).
# ---------------------------------------------------------------------


def test_case_F_pacemaker_infection_complication_preserved() -> None:
    cqm = _build_cqm(
        question="What is the risk of infection after pacemaker implantation?",
        type_of_question="etiology_harm",
        question_subtype="device_complication_risk",
        anchors=[
            _anchor(
                canonical="pacemaker",
                text="pacemaker",
                entity_type="device_or_implant",
                required=True,
                source="deterministic_concept_pack",
            ),
            _anchor(
                canonical="infection",
                text="infection",
                entity_type="complication_or_adverse_event",
                required=True,
                source="deterministic_parser",
            ),
        ],
    )
    slots = _slots(cqm)
    # pacemaker still device/procedure.
    assert _has(
        slots,
        role_in={"device_or_implant", "procedure_or_surgery"},
        phrase_contains="pacemaker",
    )
    # infection remains complication_or_adverse_event / outcome.
    assert _has(
        slots,
        role_in={"complication_or_adverse_event", "outcome"},
        phrase_contains="infection",
    )
    # "risk" itself is NOT population (even if some parser stage
    # mistagged it). We've added no "risk"-anchor here on purpose;
    # this guard fires if a future regression sneaks one in.
    for s in slots:
        if (s.get("original_phrase") or "").lower() == "risk":
            assert s.get("slot_role") != "population"


# ---------------------------------------------------------------------
# Case G — Safety regression: generic-only queries clarify.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "What is the dose?",
    "What are the recommendations?",
    "What are the side effects?",
])
def test_case_G_generic_only_queries_clarify(question: str) -> None:
    # When the question has no clinical entity, the pipeline path
    # returns clarification_required. We exercise the closed-set
    # builder directly with no primary anchors.
    from services.clinical_signal_router import clinical_signal_router
    from services.pipeline import _k1_build_clinical_question_map

    cqm = _k1_build_clinical_question_map(
        normalized_question=question.lower(),
        original_question=question,
        picott_payload={
            "type_of_question": "other",
            "question_subtype": None,
            "must_have_concepts": [],
            "pico": {},
            "question_scope": "foreground",
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": [],
            "generic_modifiers": [],
            "anchor_repair_applied": False,
            "anchor_failure_reason": "no_primary_clinical_entity",
        },
        resolver_diag={
            "called": False, "accepted": False, "resolver_confidence": None,
        },
        signal_summary=clinical_signal_router.analyze(
            question
        ).model_dump(mode="json"),
        clarification_only=True,
    )
    assert cqm["clarification_required"] is True
    assert cqm["interpretation_provenance"] == "clarification_required"
    assert cqm["branches"] == []


# ---------------------------------------------------------------------
# Closed-set leak guard for the new normalizer.
# ---------------------------------------------------------------------


def test_normalizer_preserves_closed_set_field_shape() -> None:
    """After normalization, every slot in every bucket MUST have the
    closed-set required keys + closed-set ``slot_search_policy`` and
    ``source`` values.
    """
    required = [
        {
            "original_phrase": "atrial fibrillation in adults",
            "canonical_concept": "atrial fibrillation",
            "entity_type": "disease_or_problem",
            "slot_role": "disease_or_condition",
            "required_for_search": True,
            "slot_search_policy": "required_for_retrieval",
            "inferred": False, "confidence": 0.9,
            "synonyms": [], "search_terms": [],
            "source": "deterministic_concept_pack",
        },
    ]
    new_req, new_opt, new_disp = normalize_slot_semantics(
        required_slots=required, optional_slots=[], display_only_slots=[],
    )
    closed_set_policies = {
        "required_for_retrieval", "optional_for_retrieval",
        "ranking_only", "display_only", "clarification_needed",
    }
    closed_set_sources = {
        "deterministic_parser", "deterministic_concept_pack",
        "bounded_resolver_fallback", "manual", "unknown",
    }
    for bucket in (new_req, new_opt, new_disp):
        for s in bucket:
            assert s["slot_search_policy"] in closed_set_policies
            assert s["source"] in closed_set_sources
            assert isinstance(s["synonyms"], list)
            assert isinstance(s["search_terms"], list)
