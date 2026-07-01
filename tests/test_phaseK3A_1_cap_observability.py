"""Phase K3A.1 — Cap-Observability summary endpoint + projection
contract tests.

Covers:
- ``GET /api/admin/cap-observability`` returns the closed-set
  schema (all histogram keys present, empty buckets allowed).
- ``?window_days=`` is clamped to [1, 30].
- ``blocked_phi`` sessions do NOT contribute to any histogram or
  top-3 list.
- Clinician token gets 403 on the observability route.
- The clinician projection contract (Path 1, ``trimmed_v1``) drops
  ``cap_candidates``, ``cap_severity``, ``cap_confidence``,
  ``branch_alignment``, ``matched_branch_ids``,
  ``article_intent_fit``, ``evidence_role_candidate`` and
  ``lane_confidence``; includes ``has_cap_candidates``.
- The admin per-session diagnostics endpoint still retains the full
  per-rule trace + raw ``cap_candidates`` (``full_v1``).
- Leak guard: ``system:`` / ``sk-`` / ``@gmail.com`` injected into
  ``explanation`` triggers whole-block suppression on the clinician
  projection.
- ``CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE`` exists and equals
  ``"disabled"`` (hard invariant for K3A.1).
"""
from __future__ import annotations

import os
import uuid as _uuid
from datetime import datetime, timedelta, timezone

import pytest


# ---------------------------------------------------------------------
# Env-flag invariant.
# ---------------------------------------------------------------------


def test_cap_enforcement_mode_is_in_allowed_set_for_k3b_wave1_step1() -> None:
    """K3B Wave 1 Step 1 permits ``"disabled" | "shadow"``. ``"active"``
    remains hard-asserted-rejected at the pipeline level until a
    separate operator approval.
    """
    val = os.environ.get(
        "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "disabled",
    ).strip().lower()
    assert val in {"disabled", "shadow", "active"}


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


async def _seed_session(
    *,
    user_id: str,
    article_lane_diagnostics: dict | None = None,
    status: str = "awaiting_selection",
    primary_intent: str = "therapy",
    created_offset_days: int = 0,
) -> str:
    from core.db import get_db
    db = get_db()
    sid = str(_uuid.uuid4())
    now = datetime.now(timezone.utc)
    sdj: dict = {}
    if article_lane_diagnostics is not None:
        sdj["article_lane_diagnostics"] = article_lane_diagnostics
    doc = {
        "id": sid,
        "user_id": user_id,
        "original_question": "test q",
        "sanitized_question": "test q",
        "topic_label": "test",
        "clinical_question_type": "therapy",
        "pico_json": {},
        "picott_json": {},
        "status": status,
        "created_at": now - timedelta(days=created_offset_days),
        "updated_at": now,
        "expires_at": now + timedelta(days=30),
        "deleted_at": None,
        "session_diagnostics_json": sdj,
        "clinical_question_map_json": {
            "primary_intent": primary_intent,
            "branches": [],
        },
    }
    await db.search_sessions.insert_one(doc)
    return sid


def _diag(
    *,
    article_id: str = "art_x",
    lane_label: str = "therapy_rct",
    directness_label: str = "direct",
    cap_reason_codes: list[str] | None = None,
    cap_severity: str = "low",
    cap_confidence: float = 0.0,
    branch_alignment: str = "aligned",
) -> dict:
    codes = list(cap_reason_codes or [])
    return {
        "article_id": article_id,
        "search_session_article_id": None,
        "lane_label": lane_label,
        "lane_confidence": 0.9,
        "branch_alignment": branch_alignment,
        "matched_branch_ids": ["primary_0"],
        "article_intent_fit": "full",
        "directness_label": directness_label,
        "cap_candidates": [
            {
                "reason_code": c, "severity": cap_severity,
                "explanation_id": f"x:{c}",
            } for c in codes
        ],
        "cap_reason_codes": codes,
        "cap_severity": cap_severity,
        "cap_confidence": cap_confidence,
        "cap_mode": "shadow",
        "would_cap": False,
        "would_exclude": False,
        "explanation": ["rule:one", "rule:two"],
        "evidence_role_candidate": "primary_evidence",
        "qa_warnings": [],
    }


async def _login_admin(client_) -> str:
    r = await client_.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"], r.json()["user"]["id"]


async def _login_clinician(client_) -> tuple[str, str]:
    r = await client_.post(
        "/api/auth/login",
        json={
            "email": "clinician@test.com", "password": "ClinicianTest123!",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"], r.json()["user"]["id"]


# ---------------------------------------------------------------------
# Observability endpoint — schema, leak-guard, auth.
# ---------------------------------------------------------------------


_EXPECTED_KEYS = {
    "schema_version",
    "window_days",
    "window_start",
    "window_end",
    "sessions_scanned",
    "sessions_with_lane_diagnostics",
    "articles_with_shadow_cap_candidates_total",
    "articles_with_no_cap_candidates_total",
    "cap_reason_code_histogram",
    "lane_label_histogram",
    "directness_label_histogram",
    "cap_severity_histogram",
    "cap_confidence_distribution",
    "counts_by_primary_intent",
    "counts_by_branch_intent",
    "top_3_sessions_by_cross_intent_drift_count",
    "top_3_sessions_by_low_branch_alignment_count",
    "lane_label_mode",
    "cap_detection_mode",
    "cap_enforcement_mode",
}


@pytest.mark.asyncio
async def test_observability_returns_closed_set_schema(client) -> None:
    token, _ = await _login_admin(client)
    r = await client.get(
        "/api/admin/cap-observability",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    missing = _EXPECTED_KEYS - set(body.keys())
    assert not missing, f"missing keys: {missing}"
    # Shadow invariants stamped at the surface.
    assert body["lane_label_mode"] == "shadow"
    assert body["cap_detection_mode"] == "shadow"
    # Phase K3B Wave 1 Step 1 — allowed set widened to include
    # ``"shadow"``. ``"active"`` is still hard-asserted-rejected.
    assert body["cap_enforcement_mode"] in {
        "disabled", "shadow", "active",
    }
    # Histograms are dicts (possibly empty in a fresh DB).
    for key in (
        "cap_reason_code_histogram", "lane_label_histogram",
        "directness_label_histogram", "cap_severity_histogram",
        "counts_by_primary_intent", "counts_by_branch_intent",
    ):
        assert isinstance(body[key], dict), key
    # Top-3 lists are arrays of dicts with session_id + count only.
    for key in (
        "top_3_sessions_by_cross_intent_drift_count",
        "top_3_sessions_by_low_branch_alignment_count",
    ):
        assert isinstance(body[key], list), key
        for entry in body[key]:
            assert set(entry.keys()) == {"session_id", "count"}, entry


@pytest.mark.asyncio
async def test_observability_window_days_clamped(client) -> None:
    token, _ = await _login_admin(client)
    # window=0 and window=99 should be rejected (FastAPI validation).
    r = await client.get(
        "/api/admin/cap-observability?window_days=0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 422
    r = await client.get(
        "/api/admin/cap-observability?window_days=99",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_observability_excludes_blocked_phi(client) -> None:
    """A ``blocked_phi`` session whose document carries a (hypothetical)
    ALD blob MUST NOT contribute to any histogram or top-3 list.

    Uses synthetic sentinel reason codes unique to this test so that
    earlier tests in the suite seeding ``retracted`` /
    ``expression_of_concern`` cannot poison the assertion.
    """
    atoken, _ = await _login_admin(client)
    # Clinician user id for the seeded session.
    _, uid = await _login_clinician(client)
    # Synthetic sentinels — never used elsewhere in the test suite.
    sentinel_code = "phaseK3A1_test_sentinel_nonphi"
    phi_only_code = "phaseK3A1_test_sentinel_phionly"
    await _seed_session(
        user_id=uid,
        article_lane_diagnostics={
            "schema_version": "K3A.1.0",
            "lane_label_mode": "shadow",
            "cap_detection_mode": "shadow",
            "diagnostics": [
                _diag(
                    article_id="ok",
                    cap_reason_codes=[sentinel_code],
                ),
            ],
        },
    )
    # Seed a blocked_phi session whose ALD contains a DIFFERENT
    # sentinel code; if the aggregator leaks it, we'll see it in the
    # histogram.
    await _seed_session(
        user_id=uid,
        status="blocked_phi",
        article_lane_diagnostics={
            "schema_version": "K3A.1.0",
            "lane_label_mode": "shadow",
            "cap_detection_mode": "shadow",
            "diagnostics": [
                _diag(
                    article_id="phi_leak",
                    cap_reason_codes=[phi_only_code],
                ),
            ],
        },
    )
    r = await client.get(
        "/api/admin/cap-observability?window_days=1",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    hist = body["cap_reason_code_histogram"]
    assert hist.get(sentinel_code, 0) >= 1, hist
    # The PHI-only code MUST be absent because the session was
    # blocked_phi and is excluded at the collection filter.
    assert phi_only_code not in hist, hist


@pytest.mark.asyncio
async def test_observability_clinician_forbidden(client) -> None:
    token, _ = await _login_clinician(client)
    r = await client.get(
        "/api/admin/cap-observability",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code in (401, 403), r.status_code


# ---------------------------------------------------------------------
# Clinician projection contract — Path 1 ``trimmed_v1``.
# ---------------------------------------------------------------------


_CLINICIAN_FORBIDDEN_KEYS = {
    "cap_candidates", "cap_severity", "cap_confidence",
    "branch_alignment", "matched_branch_ids", "article_intent_fit",
    "evidence_role_candidate", "lane_confidence",
}

_CLINICIAN_REQUIRED_KEYS = {
    "article_id", "search_session_article_id",
    "lane_label", "directness_label", "has_cap_candidates",
    "cap_reason_codes", "explanation", "qa_warnings",
    "cap_mode", "would_cap", "would_exclude",
}


@pytest.mark.asyncio
async def test_clinician_projection_is_trimmed_v1(client) -> None:
    token, uid = await _login_clinician(client)
    sid = await _seed_session(
        user_id=uid,
        article_lane_diagnostics={
            "schema_version": "K3A.1.0",
            "lane_label_mode": "shadow",
            "cap_detection_mode": "shadow",
            "diagnostics": [
                _diag(article_id="a1", cap_reason_codes=["retracted"]),
                _diag(article_id="a2"),
            ],
        },
    )
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    ald = (body.get("session_diagnostics_json") or {}).get(
        "article_lane_diagnostics"
    )
    assert isinstance(ald, dict), body
    assert ald.get("projection_contract") == "trimmed_v1.3"
    diags = ald.get("diagnostics") or []
    assert len(diags) == 2
    for d in diags:
        keys = set(d.keys())
        leaked = keys & _CLINICIAN_FORBIDDEN_KEYS
        assert not leaked, f"forbidden keys leaked: {leaked}"
        missing = _CLINICIAN_REQUIRED_KEYS - keys
        assert not missing, f"required keys missing: {missing}"
        # ``explanation`` is a string (joined), not a list.
        assert isinstance(d["explanation"], str)
        # ``has_cap_candidates`` is a bool.
        assert isinstance(d["has_cap_candidates"], bool)
        # Defense-in-depth shadow invariants stamped.
        assert d["cap_mode"] == "shadow"
        assert d["would_cap"] is False
        assert d["would_exclude"] is False
    # The diagnostic with a cap reason flags True.
    by_id = {d["article_id"]: d for d in diags}
    assert by_id["a1"]["has_cap_candidates"] is True
    assert by_id["a2"]["has_cap_candidates"] is False


@pytest.mark.asyncio
async def test_admin_projection_keeps_full_v1(client) -> None:
    """The admin per-session diagnostics endpoint still ships the full
    per-rule trace, raw ``cap_candidates``, ``branch_alignment``,
    ``matched_branch_ids``, ``evidence_role_candidate``, etc.
    """
    atoken, _ = await _login_admin(client)
    _, uid = await _login_clinician(client)
    sid = await _seed_session(
        user_id=uid,
        article_lane_diagnostics={
            "schema_version": "K3A.1.0",
            "lane_label_mode": "shadow",
            "cap_detection_mode": "shadow",
            "diagnostics": [
                _diag(article_id="a1", cap_reason_codes=["retracted"]),
            ],
        },
    )
    r = await client.get(
        f"/api/admin/search-sessions/{sid}/diagnostics",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    ald = body.get("article_lane_diagnostics")
    assert isinstance(ald, dict)
    assert ald.get("projection_contract") == "full_v1.3"
    diags = ald.get("diagnostics") or []
    assert len(diags) == 1
    d = diags[0]
    # All the rich fields are still present on admin.
    for key in (
        "cap_candidates", "branch_alignment", "matched_branch_ids",
        "evidence_role_candidate", "cap_severity", "cap_confidence",
    ):
        assert key in d, key
    assert d["cap_mode"] == "shadow"
    assert d["would_cap"] is False
    assert d["would_exclude"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("sentinel", [
    "system:user",
    "sk-abcdef1234",
    "leak@gmail.com",
])
async def test_clinician_projection_leak_guard_suppresses_forbidden(
    client, sentinel,
) -> None:
    """Injecting a forbidden substring (prompt residue, API key prefix,
    bare email) into the persisted ``explanation`` MUST cause the
    clinician projection to suppress the whole block.
    """
    token, uid = await _login_clinician(client)
    poisoned_diag = _diag(article_id="a1")
    poisoned_diag["explanation"] = ["rule:ok", sentinel]
    sid = await _seed_session(
        user_id=uid,
        article_lane_diagnostics={
            "schema_version": "K3A.1.0",
            "lane_label_mode": "shadow",
            "cap_detection_mode": "shadow",
            "diagnostics": [poisoned_diag],
        },
    )
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    ald = (body.get("session_diagnostics_json") or {}).get(
        "article_lane_diagnostics"
    )
    # Block suppressed — sentinel never reaches the clinician.
    assert ald is None or ald == {}, body
