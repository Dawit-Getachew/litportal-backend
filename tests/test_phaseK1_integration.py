"""Phase K1.1 — Integration fixes regression tests.

Validates the three bug-fix surfaces introduced after the K1 tester
verification report:

  Bug 1 — deterministic concept pack short-circuit no longer produces
          empty `framework_neutral_slots` for non-therapy intents.
          `generic_modifiers` are bridged into the CQM as display-only
          slots; signal-router lexical signals can override a picott
          fallback intent (background_overview / therapy) to a more
          specific closed-set intent (epidemiology, pathophysiology,
          stabilization, ...).

  Bug 2 — multi-intent conjunction regex now fires on "and what
          diagnoses..." / "and how should I ..." / ", and what is Y".

  Bug 3 — resolver SYSTEM_PROMPT carries explicit disambiguation rules
          and prompt_version bumped to "j-tail-v2-K1.1". The 9
          mandatory safety clauses, Confidence calibration block, and
          K1 multi-intent / framework block all remain intact.

These tests run WITHOUT live LLM (no resolver fixture required) by
calling `_k1_build_clinical_question_map` directly with the closed-set
inputs the pipeline would produce after a deterministic / resolver
path. Live preview verification of the persisted JSON is captured
separately in `memory/phaseK1_integration_fix_<utc-stamp>.md`.
"""
from __future__ import annotations

import pytest

from services.clinical_question_map import (
    ACTIVE_FRAMEWORKS,
    ACTIVE_INTENTS,
    PROMPT_VERSION_K1,
)
from services.clinical_signal_router import clinical_signal_router


# ---------------------------------------------------------------------
# Bug 2 — multi-intent conjunction regex regression tests.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "How should I manage syncope in an older adult, and what diagnoses should I not miss?",
    "How should I manage syncope and what diagnoses should I not miss?",
    "What is the treatment for sepsis, and what are the don't miss diagnoses?",
    "Initial management of acute chest pain and red flags to consider?",
    "Workup for syncope and differential diagnosis in elderly patients?",
])
def test_bug2_multi_intent_conjunction_with_dont_miss_phrase(
    question: str,
) -> None:
    """The router MUST emit ``possible_multi_intent=True`` for any
    sentence that joins a management / treatment / workup phrase to a
    "what diagnoses (should I) not miss" / "don't miss" / "differential" /
    "red flags" phrase.
    """
    summary = clinical_signal_router.analyze(question)
    assert summary.possible_multi_intent is True, (
        f"multi-intent NOT detected on {question!r}; "
        f"signals={summary.detected_signals}, "
        f"intents={summary.possible_intents}"
    )
    # The differential_diagnosis token MUST be in the detected intents.
    assert "differential_diagnosis" in summary.possible_intents


def test_bug2_differential_diagnosis_phrase_without_apostrophe() -> None:
    """The previous regex required ``don't`` / ``cannot`` with strict
    punctuation. The fix loosens this so casual phrasings still fire.
    """
    summary = clinical_signal_router.analyze(
        "what diagnoses should I not miss?",
    )
    assert "differential_diagnosis" in summary.possible_intents


# ---------------------------------------------------------------------
# Bug 1 — signal-router intent override + generic-modifier slot bridge.
# ---------------------------------------------------------------------


def test_bug1_epidemiology_intent_override_when_picott_falls_back() -> None:
    """When picott_extractor produces ``type_of_question="background"``
    (the "What is X?" fallback for non-therapy disease questions) but
    the signal router detects an explicit epidemiology lexical signal,
    the CQM MUST promote ``primary_intent`` to
    ``epidemiology_prevalence_incidence`` and ``frameworks`` MUST include
    ``CoCoPop``.
    """
    from services.pipeline import _k1_build_clinical_question_map

    cqm = _k1_build_clinical_question_map(
        normalized_question="what is the prevalence of atrial fibrillation in adults",
        original_question="What is the prevalence of atrial fibrillation in adults?",
        picott_payload={
            "type_of_question": "background",
            "question_subtype": "disease_overview",
            "must_have_concepts": [
                {"concept": "atrial fibrillation", "role": "condition",
                 "synonyms": [], "required_for_high_relevance": True},
            ],
            "pico": {},
            "question_scope": "background",
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": [
                {
                    "canonical_concept": "atrial fibrillation",
                    "entity_type": "disease_or_problem",
                    "required_for_search": True,
                    "confidence": 0.95,
                    "source": "ontology_pack",
                    "inferred": False,
                    "synonyms": [],
                    "text": "atrial fibrillation",
                },
            ],
            "generic_modifiers": ["adults"],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
        resolver_diag={
            "called": False,
            "accepted": False,
            "resolver_confidence": None,
        },
        signal_summary=clinical_signal_router.analyze(
            "what is the prevalence of atrial fibrillation in adults"
        ).model_dump(mode="json"),
        clarification_only=False,
    )
    assert cqm["primary_intent"] == "epidemiology_prevalence_incidence"
    assert "CoCoPop" in cqm["frameworks"]
    # Generic modifier "adults" lands in framework_neutral_slots as
    # display_only.
    slot_concepts = {
        s["canonical_concept"].lower()
        for s in cqm["framework_neutral_slots"]
    }
    assert "atrial fibrillation" in slot_concepts
    assert "adults" in slot_concepts
    # Verify the "adults" slot is display_only.
    adults_slot = next(
        s for s in cqm["framework_neutral_slots"]
        if s["canonical_concept"].lower() == "adults"
    )
    assert adults_slot["slot_search_policy"] == "display_only"
    assert adults_slot["slot_role"] in {"population", "age_group"}


def test_bug1_pathophysiology_intent_override() -> None:
    from services.pipeline import _k1_build_clinical_question_map

    cqm = _k1_build_clinical_question_map(
        normalized_question="pathophysiology of antiphospholipid syndrome",
        original_question="What is the pathophysiology of antiphospholipid syndrome?",
        picott_payload={
            "type_of_question": "background",
            "question_subtype": "disease_overview",
            "must_have_concepts": [
                {"concept": "antiphospholipid syndrome", "role": "condition",
                 "synonyms": [], "required_for_high_relevance": True},
            ],
            "pico": {},
            "question_scope": "background",
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": [
                {
                    "canonical_concept": "antiphospholipid syndrome",
                    "entity_type": "disease_or_problem",
                    "required_for_search": True,
                    "confidence": 0.95,
                    "source": "ontology_pack",
                    "inferred": False,
                    "synonyms": [],
                    "text": "antiphospholipid syndrome",
                },
            ],
            "generic_modifiers": [],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
        resolver_diag={
            "called": False, "accepted": False, "resolver_confidence": None,
        },
        signal_summary=clinical_signal_router.analyze(
            "pathophysiology of antiphospholipid syndrome"
        ).model_dump(mode="json"),
        clarification_only=False,
    )
    assert cqm["primary_intent"] == "pathophysiology_mechanism"
    assert "concept_mechanism" in cqm["frameworks"]


def test_bug1_icu_setting_bridges_into_display_only_slot() -> None:
    """The generic modifier "ICU patients" MUST appear as a
    display_only slot with ``slot_role=clinical_setting``.
    """
    from services.pipeline import _k1_build_clinical_question_map

    cqm = _k1_build_clinical_question_map(
        normalized_question="benefits and risks of magnesium replacement in icu patients",
        original_question="What are the benefits and risks of magnesium replacement in ICU patients?",
        picott_payload={
            "type_of_question": "therapy",
            "question_subtype": "drug_safety",
            "must_have_concepts": [
                {"concept": "magnesium replacement",
                 "role": "intervention",
                 "synonyms": [], "required_for_high_relevance": True},
            ],
            "pico": {},
            "question_scope": "foreground",
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": [
                {
                    "canonical_concept": "magnesium replacement",
                    "entity_type": "drug_or_drug_class",
                    "required_for_search": True,
                    "confidence": 0.9,
                    "source": "ontology_pack",
                    "inferred": False,
                    "synonyms": [],
                    "text": "magnesium replacement",
                },
            ],
            "generic_modifiers": ["icu patients"],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
        resolver_diag={
            "called": False, "accepted": False, "resolver_confidence": None,
        },
        signal_summary=clinical_signal_router.analyze(
            "benefits and risks of magnesium replacement in ICU patients"
        ).model_dump(mode="json"),
        clarification_only=False,
    )
    # Generic modifier landed as a display_only slot with the
    # closed-set ``clinical_setting`` role.
    icu_slot = next(
        s for s in cqm["framework_neutral_slots"]
        if s["canonical_concept"].lower() == "icu patients"
    )
    assert icu_slot["slot_search_policy"] == "display_only"
    assert icu_slot["slot_role"] == "clinical_setting"
    # The therapy_safety override OR the original therapy intent are
    # both acceptable for the benefits/risks question — but the intent
    # MUST NOT be background_overview.
    assert cqm["primary_intent"] != "background_overview"
    assert cqm["primary_intent"] in {
        "therapy", "therapy_safety", "drug_safety",
        "drug_complication", "complication_risk",
    }


def test_bug1_multi_intent_secondary_intent_emitted() -> None:
    """When the signal router detects multi-intent + management +
    differential_diagnosis tokens, the CQM emits a secondary branch
    with intent=differential_diagnosis OR diagnostic_workup.
    """
    from services.pipeline import _k1_build_clinical_question_map

    question = (
        "How should I manage syncope in an older adult, and what "
        "diagnoses should I not miss?"
    )
    cqm = _k1_build_clinical_question_map(
        normalized_question=question.lower(),
        original_question=question,
        picott_payload={
            "type_of_question": "therapy",
            "question_subtype": None,
            "must_have_concepts": [
                {"concept": "syncope", "role": "condition",
                 "synonyms": [], "required_for_high_relevance": True},
            ],
            "pico": {},
            "question_scope": "foreground",
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": [
                {
                    "canonical_concept": "syncope",
                    "entity_type": "disease_or_problem",
                    "required_for_search": True,
                    "confidence": 0.9,
                    "source": "ontology_pack",
                    "inferred": False,
                    "synonyms": [],
                    "text": "syncope",
                },
            ],
            "generic_modifiers": ["older adults"],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
        resolver_diag={
            "called": True, "accepted": True, "resolver_confidence": 0.85,
        },
        signal_summary=clinical_signal_router.analyze(
            question
        ).model_dump(mode="json"),
        clarification_only=False,
    )
    # Secondary intents populated.
    assert len(cqm["secondary_intents"]) >= 1
    # differential_diagnosis OR diagnostic_workup MUST be in the
    # secondary intent set (the signal router promotes both).
    secondary_ok = any(
        si in {"differential_diagnosis", "diagnostic_workup"}
        for si in cqm["secondary_intents"]
    )
    assert secondary_ok, f"secondary_intents={cqm['secondary_intents']}"
    # At least 2 branches MUST be emitted (primary + secondary).
    assert len(cqm["branches"]) >= 2
    # Schema-cap invariant: never more than 3 branches.
    assert len(cqm["branches"]) <= 3
    # The "older adults" generic modifier landed as display_only.
    older_slot = next(
        s for s in cqm["framework_neutral_slots"]
        if s["canonical_concept"].lower() == "older adults"
    )
    assert older_slot["slot_search_policy"] == "display_only"
    assert older_slot["slot_role"] == "age_group"


# ---------------------------------------------------------------------
# Bug 3 — resolver SYSTEM_PROMPT carries disambiguation rules +
# prompt_version bumped to "j-tail-v2-K1.1".
# ---------------------------------------------------------------------


def test_bug3_resolver_system_prompt_carries_k11_disambiguation_rules() -> None:
    from services.clinical_context_resolver import SYSTEM_PROMPT

    # 9 mandatory safety clauses (verbatim — verifying the 9 short
    # sentinel phrases each safety clause is built around).
    safety_clauses = [
        "You are not answering the medical question.",
        "You are only extracting structured clinical search interpretation.",
        "Return JSON only.",
        "Do not cite articles.",
        "Do not recommend treatment.",
        "Do not reference Google Scholar",
        "Do not invent MeSH terms.",
        "If the question lacks a clinical subject",
        "clarification_needed=true",
    ]
    for clause in safety_clauses:
        assert clause in SYSTEM_PROMPT, (
            f"safety clause sentinel missing: {clause!r}"
        )

    # Confidence calibration block is intact.
    assert "Confidence guidance:" in SYSTEM_PROMPT
    assert "Reserve confidence < 0.5" in SYSTEM_PROMPT

    # K1 multi-intent / framework block (additive K1.G) intact.
    assert "Multi-intent and framework routing" in SYSTEM_PROMPT
    assert "slot_search_policy guidance" in SYSTEM_PROMPT
    assert "Reserved frameworks (SPIDER / SPICE / ECLIPSE)" in SYSTEM_PROMPT

    # K1.1 additive disambiguation rules — explicit sentinels.
    assert "Intent disambiguation rules (Phase K1.1" in SYSTEM_PROMPT
    assert "Therapy should NOT be the default fallback" in SYSTEM_PROMPT
    assert "treat it as MULTI-INTENT" in SYSTEM_PROMPT
    assert "benefits and risks" in SYSTEM_PROMPT
    assert "Epidemiology (prevalence / incidence" in SYSTEM_PROMPT
    assert "Pathophysiology / mechanism" in SYSTEM_PROMPT


def test_bug3_prompt_version_bumped_to_K11() -> None:
    assert PROMPT_VERSION_K1 == "j-tail-v2-K1.1"


# ---------------------------------------------------------------------
# Closed-set guards verifying nothing leaked outside the schema.
# ---------------------------------------------------------------------


def test_signal_router_intents_remain_in_active_set() -> None:
    """The expanded multi-intent / differential regex MUST not emit
    any intent that is outside the closed-set ``ACTIVE_INTENTS``.
    """
    samples = [
        "How should I manage syncope and what diagnoses should I not miss?",
        "What are the don't miss diagnoses for chest pain?",
        "Workup for AKI and differential diagnosis",
        "Risk of stroke after MI and complications to monitor",
    ]
    for q in samples:
        summary = clinical_signal_router.analyze(q)
        for intent in summary.possible_intents:
            assert intent in ACTIVE_INTENTS, (
                f"signal router emitted out-of-set intent {intent!r} on {q!r}"
            )


def test_frameworks_remain_in_active_set_after_secondary_branch_emission() -> None:
    from services.pipeline import _k1_build_clinical_question_map

    question = (
        "How should I manage syncope in an older adult, and what "
        "diagnoses should I not miss?"
    )
    cqm = _k1_build_clinical_question_map(
        normalized_question=question.lower(),
        original_question=question,
        picott_payload={
            "type_of_question": "therapy",
            "question_subtype": None,
            "must_have_concepts": [
                {"concept": "syncope", "role": "condition",
                 "synonyms": [], "required_for_high_relevance": True},
            ],
            "pico": {},
            "question_scope": "foreground",
            "specialty": "",
        },
        anchors_out={
            "primary_anchors": [
                {
                    "canonical_concept": "syncope",
                    "entity_type": "disease_or_problem",
                    "required_for_search": True,
                    "confidence": 0.9,
                    "source": "ontology_pack",
                    "inferred": False,
                    "synonyms": [],
                    "text": "syncope",
                },
            ],
            "generic_modifiers": ["older adults"],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
        resolver_diag={
            "called": True, "accepted": True, "resolver_confidence": 0.85,
        },
        signal_summary=clinical_signal_router.analyze(
            question
        ).model_dump(mode="json"),
        clarification_only=False,
    )
    for fw in cqm["frameworks"]:
        assert fw in ACTIVE_FRAMEWORKS, (
            f"framework {fw!r} not in ACTIVE_FRAMEWORKS"
        )


# ---------------------------------------------------------------------
# Phase K1.1 follow-up — public GET response surfaces
# ``framework_neutral_slots`` (not just the distribution counter).
#
# Strategy: inject a minimal session document into the test DB with a
# closed-set ``clinical_question_map_json`` payload, then hit
# ``GET /api/search-sessions/{id}`` via the in-process AsyncClient and
# assert on the JSON. This exercises the projection in
# ``routers/search_sessions.py`` directly without any LLM dependency.
# ---------------------------------------------------------------------

import asyncio
import json
import re as _re_pii_guard
import uuid as _k1_uuid
from datetime import datetime, timedelta, timezone

import pytest


async def _k1_seed_session_with_cqm(
    client_,
    *,
    question: str,
    cqm_payload: dict,
) -> tuple[str, str]:
    """Create a real authed session via the test client, then patch the
    backing document with the supplied ``clinical_question_map_json``.
    Returns ``(session_id, bearer_token)``.
    """
    from core.db import get_db
    # Use the seeded clinician for auth (real token; real DB).
    login = await client_.post(
        "/api/auth/login",
        json={"email": "clinician@test.com", "password": "ClinicianTest123!"},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    user_id = login.json()["user"]["id"]
    # Insert a minimal session doc directly — bypasses the pipeline so
    # we deterministically assert on the projection layer.
    db = get_db()
    sid = str(_k1_uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc = {
        "id": sid,
        "user_id": user_id,
        "original_question": question,
        "sanitized_question": question,
        "topic_label": question[:80],
        "clinical_question_type": "background",
        "pico_json": {},
        "picott_json": {},
        "status": "awaiting_selection",
        "created_at": now,
        "updated_at": now,
        "expires_at": now + timedelta(days=30),
        "deleted_at": None,
        "session_diagnostics_json": {
            "phi_guard_blocked": False,
            "phi_guard_reasons": [],
        },
        "clinical_question_map_json": cqm_payload,
    }
    await db.search_sessions.insert_one(doc)
    return sid, token


@pytest.mark.asyncio
async def test_public_get_surfaces_framework_neutral_slots(client) -> None:
    """Phase K1.1 follow-up — the AF prevalence shape is the canonical
    example. Submitting a session with a populated
    ``framework_neutral_slots`` MUST surface the list (not just the
    distribution counter) on the public clinician GET response.
    """
    from services.clinical_question_map import (
        ClinicalQuestionMap,
        SCHEMA_VERSION,
    )
    # Build a closed-set valid ClinicalQuestionMap and persist it as
    # the session's clinical_question_map_json.
    cqm_model = ClinicalQuestionMap(
        original_question="What is the prevalence of atrial fibrillation in adults?",
        normalized_question="what is the prevalence of atrial fibrillation in adults",
        interpretation_provenance="deterministic_concept_pack",
        clinical_validity=True,
        clarification_required=False,
        clarification_reason=None,
        primary_intent="epidemiology_prevalence_incidence",
        secondary_intents=[],
        frameworks=["CoCoPop"],
        branches=[],
        framework_neutral_slots=[
            {
                "original_phrase": "atrial fibrillation",
                "canonical_concept": "atrial fibrillation",
                "entity_type": "disease_or_problem",
                "slot_role": "disease_or_condition",
                "required_for_search": True,
                "slot_search_policy": "required_for_retrieval",
                "inferred": False,
                "confidence": 0.95,
                "synonyms": [],
                "search_terms": [],
                "source": "deterministic_concept_pack",
            },
            {
                "original_phrase": "adults",
                "canonical_concept": "adults",
                "entity_type": "population_group",
                "slot_role": "population",
                "required_for_search": False,
                "slot_search_policy": "display_only",
                "inferred": False,
                "confidence": 0.7,
                "synonyms": [],
                "search_terms": [],
                "source": "deterministic_parser",
            },
        ],
        inferred_concepts=[],
        candidate_keywords=["atrial fibrillation"],
        candidate_MeSH_terms_unvalidated=[],
        confidence=0.95,
        schema_version=SCHEMA_VERSION,
        prompt_version=None,
    )
    sid, token = await _k1_seed_session_with_cqm(
        client,
        question="What is the prevalence of atrial fibrillation in adults?",
        cqm_payload=cqm_model.model_dump(mode="json"),
    )
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    cqm = (body.get("session_diagnostics_json") or {}).get(
        "clinical_question_map"
    ) or {}
    # 1. framework_neutral_slots is present AND non-empty on the
    #    public GET response (not just the distribution counter).
    slots = cqm.get("framework_neutral_slots")
    assert isinstance(slots, list) and len(slots) >= 2, (
        f"framework_neutral_slots missing or empty on public GET: "
        f"cqm={json.dumps(cqm, indent=2)}"
    )
    # 2. Each slot has the closed-set shape.
    required_keys = {
        "original_phrase", "canonical_concept", "entity_type",
        "slot_role", "slot_search_policy", "required_for_search",
        "inferred", "confidence", "synonyms", "search_terms", "source",
    }
    closed_set_policies = {
        "required_for_retrieval", "optional_for_retrieval",
        "ranking_only", "display_only", "clarification_needed",
    }
    closed_set_sources = {
        "deterministic_parser", "deterministic_concept_pack",
        "bounded_resolver_fallback", "manual", "unknown",
    }
    for s in slots:
        assert isinstance(s, dict)
        assert required_keys.issubset(set(s.keys())), (
            f"slot missing keys: have={set(s.keys())} expected_subset={required_keys}"
        )
        assert s["slot_search_policy"] in closed_set_policies
        assert s["source"] in closed_set_sources
        assert isinstance(s["confidence"], (int, float))
        assert isinstance(s["synonyms"], list)
        assert isinstance(s["search_terms"], list)
    # 3. The distribution counter still agrees with the slot list.
    dist = cqm.get("slot_search_policy_distribution") or {}
    assert dist.get("required_for_retrieval", 0) >= 1
    assert dist.get("display_only", 0) >= 1
    # 4. Closed-set slot_role coverage — at least one slot is a
    #    population / age_group / clinical_setting / disease_or_condition.
    target_roles = {
        "disease_or_condition", "population", "age_group", "clinical_setting",
    }
    assert any(s["slot_role"] in target_roles for s in slots), (
        f"no slot in target roles {target_roles}; got "
        f"{[s['slot_role'] for s in slots]}"
    )
    # 5. PHI / prompt / completion / secret no-leak grep on the full
    #    response body. The response is closed-set, but we positively
    #    assert that none of the high-risk substrings appear.
    blob = json.dumps(body)
    for forbidden in [
        # Resolver internals — must never appear in a public GET.
        "SYSTEM_PROMPT", "Confidence guidance",
        "Reserve confidence < 0.5",
        # Email / SSN patterns.
        "@gmail.com", "@yahoo.com",
        # JWT / secrets.
        "JWT_ACCESS_SECRET", "ACCESS_SECRET", "JWT_REFRESH_SECRET",
        # Mongo internals.
        "ObjectId(", "BSON",
    ]:
        assert forbidden not in blob, (
            f"forbidden substring {forbidden!r} leaked into public GET"
        )
    # SSN-style pattern guard (defensive — the test payload has none).
    assert not _re_pii_guard.search(r"\b\d{3}-\d{2}-\d{4}\b", blob)


@pytest.mark.asyncio
async def test_public_get_framework_neutral_slots_handles_missing_cqm(
    client,
) -> None:
    """Defensive: when ``clinical_question_map_json`` is absent from
    the session document, the public GET still returns 200 and does
    NOT surface a phantom ``clinical_question_map`` block.
    """
    from core.db import get_db
    login = await client.post(
        "/api/auth/login",
        json={"email": "clinician@test.com", "password": "ClinicianTest123!"},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    user_id = login.json()["user"]["id"]
    db = get_db()
    sid = str(_k1_uuid.uuid4())
    now = datetime.now(timezone.utc)
    await db.search_sessions.insert_one({
        "id": sid,
        "user_id": user_id,
        "original_question": "test",
        "sanitized_question": "test",
        "topic_label": "test",
        "clinical_question_type": "background",
        "pico_json": {}, "picott_json": {},
        "status": "draft",
        "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
        "deleted_at": None,
        "session_diagnostics_json": {
            "phi_guard_blocked": False, "phi_guard_reasons": [],
        },
        # NO clinical_question_map_json key.
    })
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    diag = (r.json().get("session_diagnostics_json") or {})
    assert "clinical_question_map" not in diag, (
        f"phantom clinical_question_map surfaced on session without "
        f"persisted clinical_question_map_json: {diag}"
    )
