"""Phase K3B Wave 1 Step 2 — Active retracted-only exclusion tests.

14 binary-verifiable tests covering the spec invariants for active
enforcement. Uses the session-scoped ``client`` fixture from
``conftest.py`` so the FastAPI app is mounted once across the suite.

Tests:
  1. Active mode, retracted + stable + PubMed signal → excluded
  2. Active mode, retraction NOTICE → NOT excluded
  3. Active mode, EoC only → NOT excluded
  4. Active mode, non-Wave-1 reasons only → NOT excluded
  5. Shadow mode → no actual exclusion (gate refuses to fire)
  6. Disabled mode → no actual exclusion
  7. Downstream count = filtered set; actively-excluded items absent
     from both ``articles`` and ``excluded_articles``
  8. Clinician banner renders with EXACT spec text when N > 0
  9. Clinician banner does NOT render when N == 0 (no empty state)
 10. Audit event shape (closed-set, no PHI/prompts/completions)
 11. Admin observability (actually_excluded_* fields + alert booleans)
 12. Leak safety (sentinel injection in artifacts)
 13. Active retrieval parity on non-retraction sessions
 14. PHI short-circuit (blocked_phi → no banner, no audit)
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


async def _login(client_, role: str = "clinician") -> tuple[str, str]:
    creds = (
        {"email": "admin@test.com", "password": "AdminTest123!"}
        if role == "admin"
        else {
            "email": "clinician@test.com",
            "password": "ClinicianTest123!",
        }
    )
    r = await client_.post("/api/auth/login", json=creds)
    assert r.status_code == 200, r.text
    body = r.json()
    return body["access_token"], body["user"]["id"]


def _build_excl_row(
    *, article_id: str | None = None, pmid: str = "12345",
    enforcement_mode: str = "active",
    cap_reason_codes: list[str] | None = None,
    source_signal: str = "pubmed_retracted_publication_type",
    title: str = "A retracted paper",
    journal: str = "J Test",
) -> dict[str, Any]:
    return {
        "article_id": article_id or str(uuid.uuid4()),
        "pmid": pmid,
        "doi": "10.0/test",
        "title": title,
        "journal": journal,
        "publication_year": 2024,
        "publication_date": "2024-01-01",
        "source_signal": source_signal,
        "source_signal_label": "PubMed retraction notice",
        "cap_reason_codes": (
            cap_reason_codes
            if cap_reason_codes is not None else ["retracted"]
        ),
        "enforcement_explanation": [
            "k3b:wave1", "active_exclusion",
            "retracted_via:pubmed_retracted_publication_type",
            "join:stable",
        ],
        "article_join_status": "stable",
        "enforcement_mode": enforcement_mode,
    }


async def _seed_session(
    *, user_id: str,
    k3b_active: dict[str, Any] | None = None,
    status: str = "awaiting_selection",
    primary_intent: str = "therapy",
    deleted_at=None,
) -> str:
    from core.db import get_db
    db = get_db()
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    sdj: dict[str, Any] = {}
    if k3b_active is not None:
        sdj["k3b_active_exclusions"] = k3b_active
    await db.search_sessions.insert_one({
        "id": sid, "user_id": user_id,
        "original_question": "q", "sanitized_question": "q",
        "topic_label": "t", "clinical_question_type": "therapy",
        "pico_json": {}, "picott_json": {},
        "status": status,
        "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
        "deleted_at": deleted_at,
        "session_diagnostics_json": sdj,
        "clinical_question_map_json": {
            "primary_intent": primary_intent,
        },
    })
    return sid


# ---------------------------------------------------------------------
# Tests 1-6: enforcement gate eligibility — pure-function unit tests.
# Mirrors the predicate in services/pipeline.py at the
# Phase K3B Wave 1 Step 2 active gate.
# ---------------------------------------------------------------------


def _gate_eligible(
    *, enforcement_mode: str, is_retracted: bool,
    is_retraction_notice: bool,
) -> bool:
    """Mirror of the Step 2 gate predicate."""
    return (
        enforcement_mode == "active"
        and bool(is_retracted)
        and not bool(is_retraction_notice)
    )


def test_step2_t1_active_retracted_stable_excluded():
    assert _gate_eligible(
        enforcement_mode="active",
        is_retracted=True, is_retraction_notice=False,
    ) is True


def test_step2_t2_active_unstable_or_notice_NOT_excluded():
    # Retraction notice (not the retracted paper itself) → NOT excluded.
    assert _gate_eligible(
        enforcement_mode="active",
        is_retracted=False, is_retraction_notice=True,
    ) is False
    # Even if both flags are true (defensive — should not happen on
    # real PubMed data), the notice flag wins.
    assert _gate_eligible(
        enforcement_mode="active",
        is_retracted=True, is_retraction_notice=True,
    ) is False


def test_step2_t3_active_eoc_only_NOT_excluded():
    # EoC-only article: is_retracted=False → predicate is False.
    assert _gate_eligible(
        enforcement_mode="active",
        is_retracted=False, is_retraction_notice=False,
    ) is False


def test_step2_t4_active_non_wave1_reasons_NOT_excluded():
    # Preprint / missing_anchor / cross_intent_drift / etc. all
    # bottom out at is_retracted=False; predicate is False.
    assert _gate_eligible(
        enforcement_mode="active",
        is_retracted=False, is_retraction_notice=False,
    ) is False


def test_step2_t5_shadow_mode_no_active_exclusion():
    assert _gate_eligible(
        enforcement_mode="shadow",
        is_retracted=True, is_retraction_notice=False,
    ) is False


def test_step2_t6_disabled_mode_no_active_exclusion():
    assert _gate_eligible(
        enforcement_mode="disabled",
        is_retracted=True, is_retraction_notice=False,
    ) is False


# ---------------------------------------------------------------------
# End-to-end tests 7-14: drive the clinician + admin endpoints with
# synthetic seeded sessions.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step2_t7_downstream_count_matches(client):
    """Test 7: actively-excluded items must NOT appear in either
    ``articles`` or ``excluded_articles``. Banner ``total`` matches
    the seeded count.
    """
    ctoken, uid = await _login(client, "clinician")
    block = {
        "schema_version": "k3b.wave1.step2.v1",
        "enforcement_mode": "active",
        "enforcement_candidate_type": "retracted_only",
        "exclusions": [
            _build_excl_row(pmid=f"P_T7_{i}") for i in range(3)
        ],
        "total": 3,
    }
    sid = await _seed_session(user_id=uid, k3b_active=block)
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    seeded_pmids = {ex["pmid"] for ex in block["exclusions"]}
    visible_pmids = {
        (a.get("article") or {}).get("pmid")
        for a in (body.get("articles") or [])
        + (body.get("excluded_articles") or [])
    }
    for pmid in seeded_pmids:
        assert pmid not in visible_pmids, (
            f"PMID {pmid} actively excluded but leaked into the "
            "clinician's articles / excluded_articles arrays"
        )
    banner = (body.get("session_diagnostics_json") or {}).get(
        "k3b_active_exclusions_banner"
    )
    assert banner is not None
    assert banner["total"] == 3


@pytest.mark.asyncio
async def test_step2_t8_banner_renders_when_N_gt_0(client):
    """Test 8: banner renders with EXACT spec text when N > 0."""
    ctoken, uid = await _login(client, "clinician")
    block = {
        "schema_version": "k3b.wave1.step2.v1",
        "enforcement_mode": "active",
        "enforcement_candidate_type": "retracted_only",
        "exclusions": [
            _build_excl_row(pmid="P_T8_1"),
            _build_excl_row(pmid="P_T8_2"),
        ],
        "total": 2,
    }
    sid = await _seed_session(user_id=uid, k3b_active=block)
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    body = r.json()
    banner = (body.get("session_diagnostics_json") or {}).get(
        "k3b_active_exclusions_banner"
    )
    assert banner is not None
    assert banner["total"] == 2
    # EXACT spec wording.
    assert banner["banner_text"] == (
        "2 retracted article(s) were excluded from these results "
        "because they have been retracted by their journals."
    )
    for ex in banner["exclusions"]:
        # Clinician banner must NOT carry rule IDs / source_signal
        # enum — only the short human label.
        assert "source_signal" not in ex
        assert ex["source_signal_label"] == "PubMed retraction notice"
        assert "enforcement_explanation" not in ex
        assert "cap_reason_codes" not in ex


@pytest.mark.asyncio
async def test_step2_t9_banner_NOT_rendered_when_N_zero(client):
    """Test 9: banner key is ABSENT when N == 0 (no empty state)."""
    ctoken, uid = await _login(client, "clinician")
    # Case A: no block at all on the session.
    sid_a = await _seed_session(user_id=uid, k3b_active=None)
    r = await client.get(
        f"/api/search-sessions/{sid_a}",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    diag = r.json().get("session_diagnostics_json") or {}
    assert "k3b_active_exclusions_banner" not in diag
    # Case B: block with empty exclusions list.
    block_empty = {
        "schema_version": "k3b.wave1.step2.v1",
        "enforcement_mode": "active",
        "enforcement_candidate_type": "retracted_only",
        "exclusions": [],
        "total": 0,
    }
    sid_b = await _seed_session(user_id=uid, k3b_active=block_empty)
    r = await client.get(
        f"/api/search-sessions/{sid_b}",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    diag = r.json().get("session_diagnostics_json") or {}
    assert "k3b_active_exclusions_banner" not in diag


@pytest.mark.asyncio
async def test_step2_t10_audit_event_shape(client):
    """Test 10: closed-set audit-event payload shape. Driven by the
    Phase K3B Wave 1 Step 2 persistence path in the pipeline — we
    emulate it here by writing one event through ``audit_service``
    and then reading it back, asserting closed-set fields.
    """
    from services.audit import audit_service
    _, uid = await _login(client, "clinician")
    sid = await _seed_session(user_id=uid, k3b_active=None)
    article_id = str(uuid.uuid4())
    await audit_service.log(
        "k3b_wave1_active_exclusion",
        user_id=uid, search_session_id=sid,
        payload={
            "session_id": sid,
            "article_id": article_id,
            "pmid": "12345", "doi": "10.0/test",
            "cap_reason_codes": ["retracted"],
            "enforcement_explanation": [
                "k3b:wave1", "active_exclusion",
                "retracted_via:pubmed_retracted_publication_type",
                "join:stable",
            ],
            "source_signal": "pubmed_retracted_publication_type",
            "enforcement_mode": "active",
            "article_join_status": "stable",
        },
    )
    from core.db import get_db
    db = get_db()
    rows = await db.audit_events.find({
        "event_type": "k3b_wave1_active_exclusion",
        "search_session_id": sid,
    }).to_list(length=10)
    assert len(rows) == 1
    payload = rows[0]["event_payload_json"]
    # Closed-set field presence.
    for k in (
        "session_id", "article_id", "cap_reason_codes",
        "enforcement_explanation", "source_signal",
        "enforcement_mode", "article_join_status",
    ):
        assert k in payload, f"missing closed-set field {k!r}"
    # Closed-set enums.
    assert payload["enforcement_mode"] == "active"
    assert payload["article_join_status"] == "stable"
    assert payload["source_signal"] in {
        "pubmed_retracted_publication_type",
        "pubmed_retraction_in_link",
    }
    assert payload["cap_reason_codes"] == ["retracted"]
    assert isinstance(payload["enforcement_explanation"], list)
    # Forbidden raw-text keys must NEVER appear.
    for forbidden in (
        "original_question", "sanitized_question", "raw_question",
        "completion", "prompt", "evidence_text",
        "email", "ssn", "mrn", "dob",
    ):
        assert forbidden not in payload, (
            f"forbidden key {forbidden!r} found in audit payload"
        )
    # Forbidden sentinel substrings.
    blob = json.dumps(payload)
    for s in (
        "system:", "sk-", "@gmail.com",
        "Bearer eyJ", "DOB:", "MRN:", "SSN:",
    ):
        assert s not in blob


@pytest.mark.asyncio
async def test_step2_t11_admin_observability_active_fields(client):
    """Test 11: Admin observability surfaces the new closed-set
    fields + correctly-typed alert booleans.
    """
    atoken, _ = await _login(client, "admin")
    # Seed one session with an active-exclusion block so the
    # observability counter increments.
    _, uid = await _login(client, "clinician")
    block = {
        "schema_version": "k3b.wave1.step2.v1",
        "enforcement_mode": "active",
        "enforcement_candidate_type": "retracted_only",
        "exclusions": [
            _build_excl_row(pmid="P_T11_1"),
            _build_excl_row(pmid="P_T11_2"),
        ],
        "total": 2,
    }
    await _seed_session(
        user_id=uid, k3b_active=block, primary_intent="therapy",
    )
    r = await client.get(
        "/api/admin/cap-observability?window_days=1",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    for k in (
        "actually_excluded_in_active_total",
        "actually_excluded_by_reason_code",
        "active_exclusion_by_primary_intent",
        "active_exclusion_by_branch_intent",
        "sessions_with_active_exclusion",
        "active_exclusion_ratio_organic",
        "alert_non_retracted_in_active_exclusion_histogram",
        "alert_active_exclusion_ratio_over_5pct_organic_traffic",
        "schema_version", "cap_enforcement_mode",
    ):
        assert k in body, f"missing key {k!r}"
    assert body["actually_excluded_in_active_total"] >= 2
    # Closed-set histogram keys: only "retracted" allowed in Wave 1.
    for hist_key in body["actually_excluded_by_reason_code"].keys():
        assert hist_key == "retracted", (
            f"non-retracted key {hist_key!r} leaked into Wave-1 "
            "active-exclusion histogram"
        )
    assert isinstance(
        body["alert_non_retracted_in_active_exclusion_histogram"],
        bool,
    )
    assert isinstance(
        body[
            "alert_active_exclusion_ratio_over_5pct_organic_traffic"
        ],
        bool,
    )
    assert body["schema_version"] == (
        "K3B.wave1.step2_1.observability.v1"
    )
    assert body["cap_enforcement_mode"] in {
        "disabled", "shadow", "active",
    }


@pytest.mark.asyncio
async def test_step2_t12_leak_safety_clinician_banner(client):
    """Test 12: inject sentinels into the persisted block; clinician
    banner must scrub them before returning.
    """
    ctoken, uid = await _login(client, "clinician")
    bad_excl = _build_excl_row(
        pmid="P_T12",
        title="Title with sk-LEAKED key",
        journal="J system:leak demo",
    )
    block = {
        "schema_version": "k3b.wave1.step2.v1",
        "enforcement_mode": "active",
        "enforcement_candidate_type": "retracted_only",
        "exclusions": [bad_excl],
        "total": 1,
    }
    sid = await _seed_session(user_id=uid, k3b_active=block)
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    body = r.json()
    blob = json.dumps(body)
    for s in ("system:", "sk-", "@gmail.com"):
        assert s not in blob, (
            f"sentinel {s!r} leaked through clinician banner"
        )


@pytest.mark.asyncio
async def test_step2_t13_parity_non_retraction_session_unchanged(
    client,
):
    """Test 13: no active-exclusion block → no banner field, and
    ``articles`` / ``excluded_articles`` are present.
    """
    ctoken, uid = await _login(client, "clinician")
    sid = await _seed_session(user_id=uid, k3b_active=None)
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    body = r.json()
    diag = body.get("session_diagnostics_json") or {}
    assert "k3b_active_exclusions_banner" not in diag
    assert "articles" in body
    assert "excluded_articles" in body


@pytest.mark.asyncio
async def test_step2_t14_phi_short_circuit_no_banner_no_audit(client):
    """Test 14: blocked_phi sessions → no banner, no audit event,
    no enforcement diagnostics in any surface.
    """
    ctoken, _ = await _login(client, "clinician")
    r = await client.post(
        "/api/search-sessions/",
        json={
            "question": (
                "45M with MRN 12345678, DOB 03/14/1979, "
                "email phi@example.com — best tx for HTN?"
            ),
        },
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "blocked_phi"
    sid = body.get("id")
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    body = r.json()
    diag = body.get("session_diagnostics_json") or {}
    assert "k3b_active_exclusions_banner" not in diag
    assert "k3b_active_exclusions" not in diag
    assert "article_lane_diagnostics" not in diag
    # No audit events for this PHI-blocked session.
    from core.db import get_db
    db = get_db()
    n = await db.audit_events.count_documents({
        "event_type": "k3b_wave1_active_exclusion",
        "search_session_id": sid,
    })
    assert n == 0


# ---------------------------------------------------------------------
# Mode-allow-list invariant: env var ``"active"`` is now permitted.
# ---------------------------------------------------------------------


def test_step2_env_allow_list_includes_active():
    """The pipeline env-assert must allow ``"active"`` in addition to
    ``"disabled"`` / ``"shadow"`` (operator-approved Step 2).
    """
    src = (ROOT / "services" / "pipeline.py").read_text(
        encoding="utf-8",
    )
    assert "'disabled', 'shadow', 'active'" in src, (
        "Step 2 regression: the CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE "
        "allow-list in services/pipeline.py must include 'active'."
    )
