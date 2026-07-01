"""Phase 5A Task 14 retest — admin diagnostics payload contract fixes.

Fix B1: ``normalized_question`` is exposed at the top level so admins see
         what was actually parsed/queried (not just the sanitized input).
Fix B2: ``sources_enabled_at_session_time`` lets admins interpret a
         missing source under ``queries_by_family``. Also verifies that
         when both PubMed and Europe PMC are enabled the diagnostics
         endpoint surfaces both sources' queries.
Fix B3: ``retrieval_evaluator`` is a populated flat block — never ``{}``.
         All 5 spec-required keys are present with predictable types.

These tests use the existing seeded ``admin@test.com`` account.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from core.db import Collections, get_db
from tests._phase3_helpers import login, signup


# ---------------------------------------------------------------------------
# Fix B1 — normalized_question is exposed on diagnostics.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRetestB1NormalizedQuestion:
    async def test_normalized_question_field_present_and_populated(self, client):
        """The diagnostics endpoint MUST expose ``normalized_question`` so
        admins can see the post-PICOTT-spelling-normalization form."""
        db = get_db()
        # Seed a session directly with picott_json.normalized_question
        # populated. We do not need to run the live pipeline.
        sid = uuid.uuid4().hex
        # Create an owning user (we only need diagnostics readback as
        # admin, so the owner can be any seeded user).
        t, uid = await signup(client, "retest-b1@test.com")
        now = datetime.now(timezone.utc)
        await db[Collections.SEARCH_SESSIONS].insert_one({
            "id": sid,
            "user_id": uid,
            "original_question": (
                "What is the dose of milrinone for patients with "
                "acute subarchnoid hemorrhage?"
            ),
            "sanitized_question": (
                "What is the dose of milrinone for patients with "
                "acute subarchnoid hemorrhage?"
            ),
            "topic_label": "milrinone for subarachnoid hemorrhage",
            "status": "awaiting_selection",
            "clinical_question_type": "pharmacology",
            "pico_json": {"clinical_question_type": "pharmacology"},
            "picott_json": {
                "type_of_question": "dosing_protocol",
                "question_scope": "foreground",
                "specialty": "neurology",
                "normalized_question": (
                    "What is the dose of milrinone for patients with "
                    "acute subarachnoid hemorrhage?"
                ),
                "spelling_corrections": [{
                    "original": "subarchnoid", "normalized": "subarachnoid",
                    "reason": "curated_typo", "confidence": 0.99,
                }],
                "must_have_concepts": [
                    {"concept": "milrinone", "role": "intervention",
                     "synonyms": [], "missing_relevance_cap": 30},
                    {"concept": "subarachnoid hemorrhage", "role": "condition",
                     "synonyms": ["SAH"], "missing_relevance_cap": 50},
                ],
                "ideal_study_types": ["rct", "pharmacokinetic_study"],
            },
            "sources_searched": ["europe_pmc", "pubmed"],
            "evaluator_json": {
                "recommended_next_action": "continue",
                "reason": "Initial retrieval produced sufficient ranked evidence.",
                "problems": [],
                "picott_diagnostics": {
                    "anchor_coverage_top10": {
                        "coverage_pct": 100.0, "all_anchors_covered": True,
                        "missing_anchors_in_topk": [],
                    },
                    "anchor_coverage_top20": {
                        "coverage_pct": 100.0, "all_anchors_covered": True,
                        "missing_anchors_in_topk": [],
                    },
                    "rescue_query_needed": False,
                    "rescue_must_haves": [],
                    "broad_recall_only_dominates_topk": False,
                },
            },
            "created_at": now, "updated_at": now,
            "expires_at": now,
        })

        adm = await login(client, "admin@test.com", "AdminTest123!")
        h_adm = {"Authorization": f"Bearer {adm}"}
        d = await client.get(
            f"/api/admin/search-sessions/{sid}/diagnostics", headers=h_adm,
        )
        assert d.status_code == 200, d.text
        body = d.json()
        # B1 — normalized_question is present and non-empty.
        assert "normalized_question" in body
        assert isinstance(body["normalized_question"], str)
        assert body["normalized_question"].strip(), "normalized_question is empty"
        # Sanitized question is preserved alongside (audit transparency).
        assert "sanitized_question" in body

    async def test_normalized_question_falls_back_to_sanitized(self, client):
        """When the session has no PICOTT-normalized question (e.g. a
        legacy row), the endpoint MUST still return a string — falling
        back to ``sanitized_question`` — never null."""
        db = get_db()
        sid = uuid.uuid4().hex
        t, uid = await signup(client, "retest-b1b@test.com")
        now = datetime.now(timezone.utc)
        await db[Collections.SEARCH_SESSIONS].insert_one({
            "id": sid, "user_id": uid,
            "original_question": "what is the evidence for thing?",
            "sanitized_question": "what is the evidence for thing?",
            "topic_label": "x", "status": "awaiting_selection",
            "pico_json": {}, "picott_json": {},  # no normalized_question
            "sources_searched": ["europe_pmc"],
            "evaluator_json": None,
            "created_at": now, "updated_at": now, "expires_at": now,
        })

        adm = await login(client, "admin@test.com", "AdminTest123!")
        h_adm = {"Authorization": f"Bearer {adm}"}
        d = await client.get(
            f"/api/admin/search-sessions/{sid}/diagnostics", headers=h_adm,
        )
        assert d.status_code == 200, d.text
        body = d.json()
        assert body.get("normalized_question") == "what is the evidence for thing?"


# ---------------------------------------------------------------------------
# Fix B2 — sources_enabled_at_session_time + both pubmed + europe_pmc.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRetestB2SourcesEnabledAtSessionTime:
    async def test_sources_enabled_at_session_time_field_present(self, client):
        """When PubMed was disabled at session-creation time, the endpoint
        MUST expose ``sources_enabled_at_session_time`` so admins
        understand why pubmed queries are absent."""
        db = get_db()
        sid = uuid.uuid4().hex
        t, uid = await signup(client, "retest-b2@test.com")
        now = datetime.now(timezone.utc)
        await db[Collections.SEARCH_SESSIONS].insert_one({
            "id": sid, "user_id": uid,
            "original_question": "any q",
            "sanitized_question": "any q",
            "topic_label": "x", "status": "awaiting_selection",
            "pico_json": {}, "picott_json": {},
            "sources_searched": ["europe_pmc"],  # pubmed NOT enabled here
            "evaluator_json": None,
            "created_at": now, "updated_at": now, "expires_at": now,
        })
        # Seed a single europe_pmc query so queries_by_family is non-empty.
        await db[Collections.SEARCH_QUERIES].insert_one({
            "id": uuid.uuid4().hex,
            "search_session_id": sid,
            "source": "europe_pmc",
            "query_type": "high_recall",
            "query_family": "high_recall",
            "query_string": 'TITLE_ABS:"x"',
            "rationale": "",
            "created_at": now,
        })

        adm = await login(client, "admin@test.com", "AdminTest123!")
        h_adm = {"Authorization": f"Bearer {adm}"}
        d = await client.get(
            f"/api/admin/search-sessions/{sid}/diagnostics", headers=h_adm,
        )
        assert d.status_code == 200, d.text
        body = d.json()
        assert body.get("sources_enabled_at_session_time") == ["europe_pmc"]
        # And queries_by_family contains only europe_pmc — explainable.
        sources_seen = {q["source"] for q in body["queries_by_family"]}
        assert sources_seen == {"europe_pmc"}

    async def test_both_pubmed_and_europe_pmc_queries_surfaced(self, client):
        """When the pipeline ran with BOTH sources enabled, the
        diagnostics endpoint MUST surface queries from both sources.
        This protects against an accidental filter regression."""
        db = get_db()
        sid = uuid.uuid4().hex
        t, uid = await signup(client, "retest-b2b@test.com")
        now = datetime.now(timezone.utc)
        await db[Collections.SEARCH_SESSIONS].insert_one({
            "id": sid, "user_id": uid,
            "original_question": "any q",
            "sanitized_question": "any q",
            "topic_label": "x", "status": "awaiting_selection",
            "pico_json": {}, "picott_json": {},
            "sources_searched": ["europe_pmc", "pubmed"],
            "evaluator_json": None,
            "created_at": now, "updated_at": now, "expires_at": now,
        })
        # Seed queries for both sources.
        for src, qtype in (
            ("europe_pmc", "high_recall"),
            ("europe_pmc", "cb_high_precision"),
            ("pubmed", "high_recall"),
            ("pubmed", "cb_broad_recall"),
        ):
            await db[Collections.SEARCH_QUERIES].insert_one({
                "id": uuid.uuid4().hex,
                "search_session_id": sid,
                "source": src,
                "query_type": qtype,
                "query_family": qtype.removeprefix("cb_") if qtype.startswith("cb_") else qtype,
                "query_string": f'TITLE_ABS:"x" /* {src} */',
                "rationale": "",
                "created_at": now,
            })

        adm = await login(client, "admin@test.com", "AdminTest123!")
        h_adm = {"Authorization": f"Bearer {adm}"}
        d = await client.get(
            f"/api/admin/search-sessions/{sid}/diagnostics", headers=h_adm,
        )
        assert d.status_code == 200, d.text
        body = d.json()
        sources_seen = {q["source"] for q in body["queries_by_family"]}
        assert "europe_pmc" in sources_seen
        assert "pubmed" in sources_seen, (
            f"PubMed queries dropped by diagnostics filter; saw {sources_seen}"
        )
        # sources_enabled_at_session_time should agree.
        assert set(body.get("sources_enabled_at_session_time") or []) >= {
            "europe_pmc", "pubmed",
        }


# ---------------------------------------------------------------------------
# Fix B3 — retrieval_evaluator block is populated with explicit fields.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRetestB3RetrievalEvaluatorPopulated:
    async def test_retrieval_evaluator_has_all_required_fields(self, client):
        """The ``retrieval_evaluator`` block MUST contain the 5 spec-
        required keys with predictable types — never an empty dict."""
        db = get_db()
        sid = uuid.uuid4().hex
        t, uid = await signup(client, "retest-b3@test.com")
        now = datetime.now(timezone.utc)
        # Build a session that DID run the evaluator (typical
        # awaiting_selection state).
        await db[Collections.SEARCH_SESSIONS].insert_one({
            "id": sid, "user_id": uid,
            "original_question": (
                "What is the dose of milrinone for patients with "
                "acute subarachnoid hemorrhage?"
            ),
            "sanitized_question": (
                "What is the dose of milrinone for patients with "
                "acute subarachnoid hemorrhage?"
            ),
            "topic_label": "milrinone for subarachnoid hemorrhage",
            "status": "awaiting_selection",
            "pico_json": {"clinical_question_type": "pharmacology"},
            "picott_json": {
                "type_of_question": "dosing_protocol",
                "specialty": "neurology",
                "normalized_question": (
                    "What is the dose of milrinone for patients with "
                    "acute subarachnoid hemorrhage?"
                ),
                "must_have_concepts": [
                    {"concept": "milrinone", "role": "intervention",
                     "synonyms": [], "missing_relevance_cap": 30},
                    {"concept": "subarachnoid hemorrhage", "role": "condition",
                     "synonyms": ["SAH"], "missing_relevance_cap": 50},
                ],
                "ideal_study_types": ["rct", "pharmacokinetic_study"],
            },
            "sources_searched": ["europe_pmc", "pubmed"],
            "evaluator_json": {
                "recommended_next_action": "trigger_rescue_query",
                "reason": (
                    "Top-10 results miss at least one mandatory anchor "
                    "concept (milrinone). Running a rescue query."
                ),
                "problems": ["top10_missing_must_have_anchors"],
                "picott_diagnostics": {
                    "anchor_coverage_top10": {
                        "k": 10, "coverage_pct": 50.0,
                        "all_anchors_covered": False,
                        "missing_anchors_in_topk": ["milrinone"],
                    },
                    "anchor_coverage_top20": {
                        "k": 20, "coverage_pct": 100.0,
                        "all_anchors_covered": True,
                        "missing_anchors_in_topk": [],
                    },
                    "rescue_query_needed": True,
                    "rescue_must_haves": ["milrinone"],
                    "broad_recall_only_dominates_topk": False,
                },
            },
            "created_at": now, "updated_at": now, "expires_at": now,
        })

        adm = await login(client, "admin@test.com", "AdminTest123!")
        h_adm = {"Authorization": f"Bearer {adm}"}
        d = await client.get(
            f"/api/admin/search-sessions/{sid}/diagnostics", headers=h_adm,
        )
        assert d.status_code == 200, d.text
        body = d.json()
        re = body.get("retrieval_evaluator")
        assert isinstance(re, dict) and re, (
            "retrieval_evaluator must be a non-empty dict"
        )
        # All 5 spec-required keys with predictable types.
        assert isinstance(re["anchor_coverage_top_10"], (int, float))
        assert isinstance(re["anchor_coverage_top_20"], (int, float))
        assert isinstance(re["recommended_next_action"], str)
        assert isinstance(re["broad_recall_only_domination"], bool)
        # Preferred study-type coverage may be a number (when ideal types
        # were declared) or None (when not). Either is acceptable — the
        # key MUST be present.
        assert "preferred_study_type_coverage_top_10" in re
        # Concrete expectations against the seeded fixture above.
        assert re["anchor_coverage_top_10"] == pytest.approx(50.0)
        assert re["anchor_coverage_top_20"] == pytest.approx(100.0)
        assert re["recommended_next_action"] == "trigger_rescue_query"
        assert re["broad_recall_only_domination"] is False
        assert re["rescue_query_needed"] is True
        assert "milrinone" in (re.get("rescue_must_haves") or [])

    async def test_retrieval_evaluator_no_picott_returns_null_anchor_cov(self, client):
        """When the session has no must-have anchors, anchor coverage
        is honestly reported as ``None`` (renders 'No data') — never
        a fake 100.0."""
        db = get_db()
        sid = uuid.uuid4().hex
        t, uid = await signup(client, "retest-b3b@test.com")
        now = datetime.now(timezone.utc)
        await db[Collections.SEARCH_SESSIONS].insert_one({
            "id": sid, "user_id": uid,
            "original_question": "background q",
            "sanitized_question": "background q",
            "topic_label": "x", "status": "awaiting_selection",
            "pico_json": {}, "picott_json": {},  # NO must_have_concepts
            "sources_searched": ["europe_pmc"],
            "evaluator_json": {
                "recommended_next_action": "continue",
                "reason": "ok",
                "problems": [],
                "picott_diagnostics": {},
            },
            "created_at": now, "updated_at": now, "expires_at": now,
        })

        adm = await login(client, "admin@test.com", "AdminTest123!")
        h_adm = {"Authorization": f"Bearer {adm}"}
        d = await client.get(
            f"/api/admin/search-sessions/{sid}/diagnostics", headers=h_adm,
        )
        assert d.status_code == 200, d.text
        re = d.json()["retrieval_evaluator"]
        assert re["anchor_coverage_top_10"] is None
        assert re["anchor_coverage_top_20"] is None
        # preferred_study_type_coverage_top_10 is null when no ideal types.
        assert re["preferred_study_type_coverage_top_10"] is None
        # Recommended_next_action is still a string ("continue").
        assert re["recommended_next_action"] == "continue"
        assert re["broad_recall_only_domination"] is False
