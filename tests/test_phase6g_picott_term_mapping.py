"""Phase 6G — PICOTT term mapping projection + completeness guard.

These tests exercise the pure projection (no LLM, no network) and the
read-time GET surface on /api/search-sessions/{id}.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

import pytest

from core.db import Collections, get_db
from models.picott_term_mapping import PicottTermMapping
from services.picott_term_mapping_service import (
    build_picott_term_mapping,
    derive_display_from_mapping,
    is_foreground_subtype,
    needs_display_self_heal,
)


# ---------------------------------------------------------------------
# Section A — pure projection unit tests
# ---------------------------------------------------------------------


def _seroquel_doc() -> dict:
    """Mirror of the persisted shape for a Seroquel adverse-effects
    question (Q1)."""
    return {
        "id": "test_seroquel",
        "picott_json": {
            "type_of_question": "etiology_harm",
            "question_subtype": "drug_safety",
            "specialty": "psychiatry",
            "question_scope": "foreground",
            "must_have_concepts": [
                {"concept": "quetiapine", "role": "intervention",
                 "synonyms": ["seroquel"],
                 "required_for_high_relevance": True},
                {"concept": "adverse effects", "role": "outcome",
                 "synonyms": ["side effects"],
                 "required_for_high_relevance": True},
            ],
            "concept_translations": [
                {"original_phrase": "Seroquel",
                 "canonical_concept": "quetiapine",
                 "confidence": 0.95},
            ],
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [
                {"text": "Seroquel", "canonical_concept": "quetiapine",
                 "entity_type": "drug_or_drug_class", "role": "intervention",
                 "synonyms": ["quetiapine"], "confidence": 0.9,
                 "required_for_search": True,
                 "missing_relevance_cap": 20},
            ],
            "generic_modifiers": ["side effects"],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
        "role_repairs_json": [],
        "ontology_lifts_json": [],
        "multi_intent_json": {},
    }


def _psoriasis_doc() -> dict:
    return {
        "id": "test_psoriasis",
        "picott_json": {
            "type_of_question": "therapy",
            "question_subtype": "treatment_sequencing",
            "specialty": "dermatology",
            "question_scope": "foreground",
            "must_have_concepts": [
                {"concept": "psoriasis", "role": "condition",
                 "synonyms": ["psoriatic disease"],
                 "required_for_high_relevance": True},
                {"concept": "first-line treatment", "role": "context",
                 "synonyms": [],
                 "required_for_high_relevance": False},
            ],
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [
                {"text": "psoriasis", "canonical_concept": "psoriasis",
                 "entity_type": "disease_or_problem", "role": "population",
                 "synonyms": ["psoriatic disease"], "confidence": 0.92,
                 "required_for_search": True,
                 "missing_relevance_cap": 20},
            ],
            "generic_modifiers": ["first-line treatment",
                                  "second-line treatment"],
            "anchor_repair_applied": False,
            "anchor_failure_reason": None,
        },
    }


def _generic_only_doc() -> dict:
    """Q6 — 'What are the side effects?'"""
    return {
        "id": "test_generic_only",
        "picott_json": {
            "type_of_question": "other",
            "question_subtype": None,
            "specialty": "",
            "question_scope": "foreground",
            "must_have_concepts": [],
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [],
            "generic_modifiers": ["side effects"],
            "anchor_repair_applied": True,
            "anchor_failure_reason": "no_primary_clinical_entity",
        },
    }


class TestProjection:
    def test_seroquel_maps_to_intervention(self):
        m = build_picott_term_mapping(_seroquel_doc(), queries=[])
        roles_with = [r for r, v in m.fields.items() if v]
        assert "intervention_exposure_index_test" in roles_with
        canonicals = [
            e.canonical_concept for e in
            m.fields["intervention_exposure_index_test"]
        ]
        assert "quetiapine" in canonicals

    def test_seroquel_outcome_present(self):
        m = build_picott_term_mapping(_seroquel_doc(), queries=[])
        assert m.fields["outcomes_target_diagnosis"]
        names = [e.canonical_concept
                 for e in m.fields["outcomes_target_diagnosis"]]
        assert any("adverse" in n.lower() or "side" in n.lower()
                   for n in names)

    def test_psoriasis_maps_to_population_problem(self):
        m = build_picott_term_mapping(_psoriasis_doc(), queries=[])
        canonicals = [e.canonical_concept
                      for e in m.fields["population_problem"]]
        assert "psoriasis" in canonicals

    def test_first_line_treatment_routes_to_clinical_action(self):
        m = build_picott_term_mapping(_psoriasis_doc(), queries=[])
        canonicals = [
            e.canonical_concept for e in m.fields["clinical_action"]
        ]
        assert any("first-line" in c.lower() for c in canonicals)

    def test_side_effects_routes_to_outcome(self):
        m = build_picott_term_mapping(_generic_only_doc(), queries=[])
        # No primary anchor — generic modifier routed lexically.
        canonicals = [
            e.canonical_concept for e in m.fields["outcomes_target_diagnosis"]
        ]
        assert "side effects" in canonicals

    def test_completeness_safe_to_search_when_required_roles_present(self):
        m = build_picott_term_mapping(_seroquel_doc(), queries=[])
        assert m.completeness.has_primary_anchor is True
        assert m.completeness.has_required_role_mapping is True
        assert m.completeness.safe_to_search is True
        assert m.completeness.clarification_needed is False
        assert m.completeness.clarification_message is None
        assert m.completeness.missing_required_roles == []

    def test_completeness_clarification_for_generic_only(self):
        m = build_picott_term_mapping(_generic_only_doc(), queries=[])
        assert m.completeness.has_primary_anchor is False
        assert m.completeness.safe_to_search is False
        assert m.completeness.clarification_needed is True
        assert m.completeness.clarification_message is not None
        assert "clinical subject" in m.completeness.clarification_message

    def test_comparator_time_setting_may_remain_empty(self):
        m = build_picott_term_mapping(_seroquel_doc(), queries=[])
        # Empty optional roles MUST NOT flip safe_to_search.
        for role in ("comparator_reference_standard", "time_horizon",
                     "setting"):
            assert m.fields[role] == []
        assert m.completeness.safe_to_search is True

    def test_extra_keys_rejected(self):
        m = build_picott_term_mapping(_seroquel_doc(), queries=[])
        bad = m.model_dump()
        bad["evil_extra"] = "leaked"
        with pytest.raises(Exception):
            PicottTermMapping(**bad)


class TestForbiddenContent:
    """Closed-set safety guards: PHI/URL-with-query/prompt/key bleed."""

    def _payload(self) -> dict:
        m = build_picott_term_mapping(_seroquel_doc(), queries=[])
        return m.model_dump(mode="json")

    def test_no_phi_email_pattern(self):
        blob = str(self._payload())
        # Allow @ in word boundaries only outside email-shaped patterns.
        # We assert no actual email shape leaks in.
        assert not re.search(r"[\w.\-]+@[\w.\-]+\.\w{2,}", blob)

    def test_no_phone_number(self):
        blob = str(self._payload())
        assert not re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", blob)

    def test_no_ssn_pattern(self):
        blob = str(self._payload())
        assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", blob)

    def test_no_prompt_or_completion_keys(self):
        p = self._payload()
        flat_keys: set[str] = set()

        def collect(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    flat_keys.add(k)
                    collect(v, f"{path}.{k}")
            elif isinstance(o, list):
                for item in o:
                    collect(item, path)
        collect(p)
        bad = {"prompt", "completion", "secret", "api_key",
               "access_token", "openai_api_key"}
        assert flat_keys.isdisjoint(bad), \
            f"forbidden key leaked: {flat_keys & bad}"

    def test_no_full_url_with_query_string(self):
        blob = str(self._payload())
        assert not re.search(r"https?://\S+\?\S+", blob)


class TestDisplaySelfHeal:
    """Task 5 — legacy "Background / primary_care" labels self-heal
    when the new mapping says foreground."""

    def test_foreground_subtype_detection(self):
        assert is_foreground_subtype("drug_safety") is True
        assert is_foreground_subtype("treatment_sequencing") is True
        assert is_foreground_subtype("background") is False
        assert is_foreground_subtype(None) is False

    def test_needs_heal_for_blank_picott_with_primary_care_label(self):
        m = build_picott_term_mapping(_seroquel_doc(), queries=[])
        legacy_display = {
            "type_of_question": "Background",
            "specialty": "primary_care",
            "population": ["Not specified"],
            "intervention_or_exposure_or_index_test": ["Not specified"],
            "outcomes_or_target_diagnosis": ["Not specified"],
        }
        assert needs_display_self_heal(legacy_display, m) is True
        healed = derive_display_from_mapping(legacy_display, m)
        # Forbidden labels replaced.
        assert healed["specialty"] != "primary_care"
        assert healed["type_of_question"] != "Background"
        # Blank fields filled.
        assert healed["intervention_or_exposure_or_index_test"] != [
            "Not specified",
        ]

    def test_does_not_overwrite_original_question(self):
        m = build_picott_term_mapping(_seroquel_doc(), queries=[])
        legacy = {
            "original_question": "What are the side effects of Seroquel?",
            "sanitized_question": "What are the side effects of Seroquel?",
            "type_of_question": "Background",
            "specialty": "primary_care",
            "population": ["Not specified"],
            "intervention_or_exposure_or_index_test": ["Not specified"],
        }
        healed = derive_display_from_mapping(legacy, m)
        # The display itself doesn't carry original_question normally,
        # but the helper must not strip keys it didn't touch.
        assert healed.get("original_question") == \
            "What are the side effects of Seroquel?"
        assert healed.get("sanitized_question") == \
            "What are the side effects of Seroquel?"


# ---------------------------------------------------------------------
# Section B — GET endpoint integration
# ---------------------------------------------------------------------


async def _login(client, email: str, password: str) -> str:
    r = await client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


async def _seed_session_with_anchors(user_id: str) -> str:
    """Insert a completed session shape so the GET endpoint can be
    exercised without running the full pipeline. Uses the same shape
    as the real pipeline persists.
    """
    db = get_db()
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc = _seroquel_doc()
    doc.update({
        "id": sid,
        "user_id": user_id,
        "status": "awaiting_selection",
        "original_question": "What are the side effects of Seroquel?",
        "sanitized_question": "What are the side effects of Seroquel?",
        "pico_json": {},
        "created_at": now,
        "updated_at": now,
        "expires_at": now,
    })
    await db[Collections.SEARCH_SESSIONS].insert_one(doc)
    return sid


@pytest.mark.asyncio
async def test_get_session_surfaces_picott_term_mapping(client):
    token = await _login(client, "clinician@test.com", "ClinicianTest123!")
    db = get_db()
    user = await db[Collections.USERS].find_one({"email": "clinician@test.com"})
    sid = await _seed_session_with_anchors(user["id"])

    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    m = body.get("picott_term_mapping_json")
    assert isinstance(m, dict) and m, f"missing/empty mapping: {m!r}"
    # Required-by-spec top-level keys.
    for k in ("fields", "completeness", "primary_clinical_anchors",
              "generic_modifiers"):
        assert k in m, f"missing top-level key {k}"
    assert m["completeness"]["safe_to_search"] is True
    assert m["completeness"]["clarification_needed"] is False
    # Cleanup.
    await db[Collections.SEARCH_SESSIONS].delete_one({"id": sid})


@pytest.mark.asyncio
async def test_generic_only_session_returns_clarification(client):
    token = await _login(client, "clinician@test.com", "ClinicianTest123!")
    db = get_db()
    user = await db[Collections.USERS].find_one({"email": "clinician@test.com"})
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc = _generic_only_doc()
    doc.update({
        "id": sid,
        "user_id": user["id"],
        "status": "awaiting_clinical_subject",
        "clarification_message_text": (
            "I could not confidently identify the clinical subject of the "
            "question. Please specify the drug, condition, test, vaccine, "
            "or procedure."
        ),
        "original_question": "What are the side effects?",
        "sanitized_question": "What are the side effects?",
        "pico_json": {},
        "created_at": now,
        "updated_at": now,
        "expires_at": now,
    })
    await db[Collections.SEARCH_SESSIONS].insert_one(doc)

    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    m = body.get("picott_term_mapping_json")
    assert isinstance(m, dict) and m
    assert m["completeness"]["safe_to_search"] is False
    assert m["completeness"]["clarification_needed"] is True
    assert m["completeness"]["clarification_message"], \
        "clarification_message should be non-empty for generic-only branch"
    # No retrieval should have run — source_api_call_count == 0.
    sdg = body.get("session_diagnostics_json") or {}
    assert int(sdg.get("source_api_call_count") or 0) == 0
    # No LLM call. llm_calls_count may be None for legacy seed; treat None as 0.
    assert int(sdg.get("llm_calls_count") or 0) == 0
    await db[Collections.SEARCH_SESSIONS].delete_one({"id": sid})
