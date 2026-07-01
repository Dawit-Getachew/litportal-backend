"""Phase K3A — router projection tests (clinician + admin).

Seeds a search_session document directly with a closed-set
``session_diagnostics_json.article_lane_diagnostics`` payload, then
exercises both:

  1. ``GET /api/search-sessions/{id}``       — clinician (minimal projection)
  2. ``GET /api/admin/search-sessions/{id}/diagnostics`` — admin (full)

Asserts on the closed-set shape, the K3A shadow-mode invariants
(``cap_mode='shadow'``, ``would_cap=False``, ``would_exclude=False``),
the leak-guard (forbidden substrings cause the entire block to be
suppressed), and the admin vs clinician verbosity contract.
"""
from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timedelta, timezone

import pytest


async def _seed_session_with_lane_diag(
    client_,
    *,
    article_lane_diagnostics: dict,
    user_role: str = "clinician",
) -> tuple[str, str]:
    """Create a real authed session via the test client, then patch the
    backing document with the supplied lane-diagnostics payload. Returns
    ``(session_id, bearer_token)``.
    """
    from core.db import get_db

    if user_role == "admin":
        creds = {"email": "admin@test.com", "password": "AdminTest123!"}
    else:
        creds = {
            "email": "clinician@test.com", "password": "ClinicianTest123!",
        }
    login = await client_.post("/api/auth/login", json=creds)
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    user_id = login.json()["user"]["id"]

    db = get_db()
    sid = str(_uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc = {
        "id": sid,
        "user_id": user_id,
        "original_question": "What is the evidence for X?",
        "sanitized_question": "What is the evidence for X?",
        "topic_label": "X",
        "clinical_question_type": "therapy",
        "pico_json": {},
        "picott_json": {"must_have_concepts": []},
        "status": "awaiting_selection",
        "created_at": now,
        "updated_at": now,
        "expires_at": now + timedelta(days=30),
        "deleted_at": None,
        "session_diagnostics_json": {
            "article_lane_diagnostics": article_lane_diagnostics,
        },
    }
    await db.search_sessions.insert_one(doc)
    return sid, token


def _sample_batch() -> dict:
    return {
        "schema_version": "K3A.1.0",
        "lane_label_mode": "shadow",
        "cap_detection_mode": "shadow",
        "diagnostics": [
            {
                "article_id": "art_rct_1",
                "search_session_article_id": None,
                "lane_label": "therapy_rct",
                "lane_confidence": 0.9,
                "branch_alignment": "aligned",
                "matched_branch_ids": ["primary_0"],
                "article_intent_fit": "full",
                "directness_label": "direct",
                "cap_candidates": [],
                "cap_reason_codes": [],
                "cap_severity": "low",
                "cap_confidence": 0.0,
                "cap_mode": "shadow",
                "would_cap": False,
                "would_exclude": False,
                "explanation": ["therapy:rct"],
                "evidence_role_candidate": "primary_evidence",
                "qa_warnings": [],
            },
            {
                "article_id": "art_retracted_1",
                "search_session_article_id": None,
                "lane_label": "retracted_or_expression_of_concern",
                "lane_confidence": 0.99,
                "branch_alignment": "aligned",
                "matched_branch_ids": ["primary_0"],
                "article_intent_fit": "full",
                "directness_label": "unsafe",
                "cap_candidates": [
                    {
                        "reason_code": "retracted",
                        "severity": "high",
                        "explanation_id": "safety:retracted",
                    },
                ],
                "cap_reason_codes": ["retracted"],
                "cap_severity": "high",
                "cap_confidence": 0.85,
                "cap_mode": "shadow",
                "would_cap": False,
                "would_exclude": False,
                "explanation": ["safety:retracted_or_eoc"],
                "evidence_role_candidate": "to_exclude_candidate",
                "qa_warnings": [],
            },
        ],
    }


# ---------------------------------------------------------------------
# Clinician projection.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clinician_get_surfaces_article_lane_diagnostics(client) -> None:
    """Phase K3A.1 — the clinician GET surfaces the closed-set
    ``article_lane_diagnostics`` block under the ``trimmed_v1``
    contract. The raw per-rule trace + ``cap_candidates`` array are
    NOT included; ``has_cap_candidates`` summarizes them as a bool.
    """
    sid, token = await _seed_session_with_lane_diag(
        client, article_lane_diagnostics=_sample_batch(),
    )
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    block = (body.get("session_diagnostics_json") or {}).get(
        "article_lane_diagnostics"
    )
    assert isinstance(block, dict), body
    assert block["schema_version"] == "K3A.1.0"
    assert block["lane_label_mode"] == "shadow"
    assert block["cap_detection_mode"] == "shadow"
    assert block.get("projection_contract") == "trimmed_v1.3"
    diags = block.get("diagnostics") or []
    assert len(diags) == 2
    by_id = {d["article_id"]: d for d in diags}
    rct = by_id["art_rct_1"]
    retr = by_id["art_retracted_1"]
    # Shadow-mode invariants on every diagnostic.
    for d in (rct, retr):
        assert d["cap_mode"] == "shadow"
        assert d["would_cap"] is False
        assert d["would_exclude"] is False
        # Phase K3A.1 — required closed-set fields on the trimmed
        # clinician projection.
        for key in (
            "lane_label", "directness_label", "has_cap_candidates",
            "cap_reason_codes", "explanation", "qa_warnings",
        ):
            assert key in d, f"missing {key} in {d}"
        # Phase K3A.1 — the raw per-rule trace, raw cap_candidates
        # array, branch_alignment, matched_branch_ids,
        # article_intent_fit, evidence_role_candidate, lane_confidence,
        # cap_severity, and cap_confidence must NOT leak to clinician.
        for key in (
            "cap_candidates", "branch_alignment", "matched_branch_ids",
            "article_intent_fit", "evidence_role_candidate",
            "lane_confidence", "cap_severity", "cap_confidence",
        ):
            assert key not in d, f"clinician must not surface {key}: {d}"
        # ``explanation`` is now a joined string, ``has_cap_candidates``
        # is a bool.
        assert isinstance(d["explanation"], str)
        assert isinstance(d["has_cap_candidates"], bool)
    assert rct["lane_label"] == "therapy_rct"
    assert retr["lane_label"] == "retracted_or_expression_of_concern"
    # Cap reason codes round-trip closed-set strings (kept on clinician).
    assert "retracted" in retr["cap_reason_codes"]
    # The retracted article carried a cap_candidate in the seed; the
    # clinician sees the summarized boolean.
    assert retr["has_cap_candidates"] is True
    assert rct["has_cap_candidates"] is False


@pytest.mark.asyncio
async def test_clinician_projection_drops_forbidden_substring_block(client) -> None:
    """If a forbidden substring (LLM prompt header, JWT secret name) ever
    appears anywhere in the persisted diagnostic blob, the WHOLE block
    is suppressed in the clinician projection. K3A leak-guard contract.
    """
    poisoned = _sample_batch()
    # Inject a forbidden substring into the explanation list — this
    # would never happen via the classifier itself (rule ids are
    # closed-set), but the projection guard MUST still hold.
    poisoned["diagnostics"][0]["explanation"] = [
        "therapy:rct",
        "JWT_ACCESS_SECRET=leak",
    ]
    sid, token = await _seed_session_with_lane_diag(
        client, article_lane_diagnostics=poisoned,
    )
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    block = (body.get("session_diagnostics_json") or {}).get(
        "article_lane_diagnostics"
    )
    # Entire block suppressed.
    assert block is None or block == {}, body


# ---------------------------------------------------------------------
# Admin projection.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_diagnostics_surfaces_full_lane_block(client) -> None:
    sid, token = await _seed_session_with_lane_diag(
        client, article_lane_diagnostics=_sample_batch(), user_role="admin",
    )
    r = await client.get(
        f"/api/admin/search-sessions/{sid}/diagnostics",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    block = body.get("article_lane_diagnostics")
    assert isinstance(block, dict), body
    assert block["schema_version"] == "K3A.1.0"
    assert block["lane_label_mode"] == "shadow"
    assert block["cap_detection_mode"] == "shadow"
    diags = block.get("diagnostics") or []
    assert len(diags) == 2
    for d in diags:
        # Shadow-mode invariants hold on the admin projection too.
        assert d["cap_mode"] == "shadow"
        assert d["would_cap"] is False
        assert d["would_exclude"] is False
        # Explanation list is a list of closed-set rule ids.
        assert isinstance(d["explanation"], list)
        for rid in d["explanation"]:
            assert isinstance(rid, str)


@pytest.mark.asyncio
async def test_admin_diagnostics_drops_forbidden_block(client) -> None:
    poisoned = _sample_batch()
    poisoned["diagnostics"][1]["qa_warnings"] = [
        "low_lane_confidence", "SYSTEM_PROMPT_leak",
    ]
    sid, token = await _seed_session_with_lane_diag(
        client, article_lane_diagnostics=poisoned, user_role="admin",
    )
    r = await client.get(
        f"/api/admin/search-sessions/{sid}/diagnostics",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("article_lane_diagnostics") is None, body



# ---------------------------------------------------------------------
# K2 branch_query_plan — admin projection contract (regression fix).
# ---------------------------------------------------------------------


async def _seed_session_with_bqp(
    client_,
    *,
    branch_query_plan: dict,
    user_role: str = "admin",
) -> tuple[str, str]:
    """Seed a session document with a closed-set
    ``clinical_question_map_json.branch_query_plan`` payload (sibling
    key on the CQM, mirroring what ``services/pipeline.py`` persists).
    """
    from core.db import get_db

    if user_role == "admin":
        creds = {"email": "admin@test.com", "password": "AdminTest123!"}
    else:
        creds = {
            "email": "clinician@test.com", "password": "ClinicianTest123!",
        }
    login = await client_.post("/api/auth/login", json=creds)
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    user_id = login.json()["user"]["id"]

    db = get_db()
    sid = str(_uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc = {
        "id": sid,
        "user_id": user_id,
        "original_question": "magnesium replacement icu",
        "sanitized_question": "magnesium replacement icu",
        "topic_label": "magnesium",
        "clinical_question_type": "therapy",
        "pico_json": {},
        "picott_json": {"must_have_concepts": []},
        "status": "awaiting_selection",
        "created_at": now,
        "updated_at": now,
        "expires_at": now + timedelta(days=30),
        "deleted_at": None,
        "clinical_question_map_json": {
            "branches": [],
            "branch_query_plan": branch_query_plan,
        },
    }
    await db.search_sessions.insert_one(doc)
    return sid, token


@pytest.mark.asyncio
async def test_admin_diagnostics_surfaces_branch_query_plan(client) -> None:
    """Regression: ``branch_query_plan`` MUST be reachable via the
    admin diagnostics endpoint with shadow invariants force-stamped.
    """
    bqp_payload = {
        "schema_version": "K2.1.0",
        "mode": "shadow",
        "active_for_retrieval": False,
        "generated_at": "2026-06-04T00:00:00+00:00",
        "branches": [
            {
                "branch_id": "primary_0",
                "intent": "therapy_safety",
                "framework": "PICOTT",
                "branch_priority": 1,
                "branch_confidence": 0.8,
                "required_concepts_used": [],
                "optional_concepts_used": [],
                "query_families": [
                    {
                        "family_name": "broad_recall",
                        "source": "pubmed",
                        "qa_checks_passed": True,
                        "query_string": "magnesium[Title/Abstract]",
                        "notes": [],
                    },
                ],
                "active_for_retrieval": False,
                "mode": "shadow",
            },
        ],
    }
    sid, token = await _seed_session_with_bqp(
        client, branch_query_plan=bqp_payload, user_role="admin",
    )
    r = await client.get(
        f"/api/admin/search-sessions/{sid}/diagnostics",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    bqp = body.get("branch_query_plan")
    assert isinstance(bqp, dict), body
    # Shadow invariants force-stamped at projection layer.
    assert bqp["mode"] == "shadow"
    assert bqp["active_for_retrieval"] is False
    # Branches surface with the expected closed-set shape.
    branches = bqp.get("branches") or []
    assert len(branches) == 1
    assert branches[0]["intent"] == "therapy_safety"
    assert branches[0]["branch_id"] == "primary_0"
    assert len(branches[0].get("query_families") or []) == 1


@pytest.mark.asyncio
async def test_admin_diagnostics_drops_bqp_on_forbidden_substring(
    client,
) -> None:
    """If a forbidden substring leaks into the persisted BQP, the
    admin projection MUST suppress the whole block (defense-in-depth).
    """
    poisoned = {
        "schema_version": "K2.1.0",
        "mode": "shadow",
        "active_for_retrieval": False,
        "generated_at": "2026-06-04T00:00:00+00:00",
        "branches": [
            {
                "branch_id": "primary_0",
                "intent": "therapy",
                "framework": "PICOTT",
                "query_families": [
                    {
                        # JWT secret leak — must trigger suppression.
                        "query_string": "JWT_ACCESS_SECRET=foo",
                        "family_name": "broad_recall",
                        "source": "pubmed",
                        "qa_checks_passed": False,
                    },
                ],
            },
        ],
    }
    sid, token = await _seed_session_with_bqp(
        client, branch_query_plan=poisoned, user_role="admin",
    )
    r = await client.get(
        f"/api/admin/search-sessions/{sid}/diagnostics",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("branch_query_plan") is None, body
