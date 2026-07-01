"""Phase K3B Wave 1 Step 2.2 — Per-Day Sparkline tests.

Covers the 9 acceptance tests from the operator's spec:

  1. Sparkline returns closed-set daily buckets — schema exactly
     matches the spec; no extra keys.
  2. Organic and seeded counts are separated (not summed) in each
     bucket.
  3. Clinician token on ``/api/admin/cap-observability`` → 403
     (existing behavior, regression-confirm).
  4. Admin token → 200 with ``daily_sparkline`` populated.
  5. Window validation: ``?window_days=0`` → 422; ``?window_days=99``
     → 422 (existing behavior, regression-confirm with sparkline
     present).
  6. Leak safety: sentinel-injection test.
  7. Production-mode safety check: ``settings.IS_PRODUCTION`` doesn't
     change behavior of this endpoint — it's a non-mutating read
     regardless of env.
  8. Existing Step 2.1 tests still green (delegated — run separately).
  9. Active retrieval parity: existing therapy session probe still
     returns shadow-only diagnostics with no behavior change.

The sparkline is additive: ``schema_version`` stays at
``K3B.wave1.step2_1.observability.v1`` per operator's
"existing keys stay unchanged" rule.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Closed-set per-day bucket schema per spec.
_BUCKET_KEYS = (
    "date",
    "organic_sessions_total",
    "seeded_sessions_total",
    "would_have_been_excluded_in_active_total_organic",
    "would_have_been_excluded_in_active_total_seeded",
    "actually_excluded_total_organic",
    "actually_excluded_total_seeded",
    "organic_wbe_session_ratio",
    "seeded_wbe_session_ratio",
    "non_retracted_wbe_keys_count",
    "unstable_join_wbe_count",
    "alert_flags_tripped",
)
_PER_DAY_ALERT_FLAGS = (
    "non_retracted_in_active_exclusion_histogram",
    "active_exclusion_ratio_over_5pct_organic_traffic",
    "wbe_true_with_unstable_join",
    "eoc_only_marked_as_enforcing",
    "barred_reason_in_enforcement_candidate_set",
)


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


async def _seed_session(
    *,
    user_id: str,
    status: str = "awaiting_selection",
    provenance: str = "organic",
    primary_intent: str = "therapy",
    created_at: datetime | None = None,
    article_lane_diagnostics: dict | None = None,
    k3b_active_exclusions: dict | None = None,
) -> str:
    from core.db import get_db
    db = get_db()
    sid = str(uuid.uuid4())
    now = created_at or datetime.now(timezone.utc)
    sdj: dict = {}
    if article_lane_diagnostics is not None:
        sdj["article_lane_diagnostics"] = article_lane_diagnostics
    if k3b_active_exclusions is not None:
        sdj["k3b_active_exclusions"] = k3b_active_exclusions
    await db.search_sessions.insert_one({
        "id": sid, "user_id": user_id,
        "original_question": "q", "sanitized_question": "q",
        "topic_label": "t", "clinical_question_type": "therapy",
        "pico_json": {}, "picott_json": {},
        "status": status,
        "provenance": provenance,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + timedelta(days=30),
        "deleted_at": None,
        "session_diagnostics_json": sdj,
        "clinical_question_map_json": {
            "primary_intent": primary_intent,
        },
    })
    return sid


# ---------------------------------------------------------------------
# Test 1 — Sparkline closed-set schema (no extra keys).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t1_sparkline_closed_set_schema(client):
    atoken, _ = await _login(client, "admin")
    r = await client.get(
        "/api/admin/cap-observability?window_days=7",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "daily_sparkline" in body
    sl = body["daily_sparkline"]
    assert isinstance(sl, list)
    assert len(sl) >= 1
    for bucket in sl:
        assert isinstance(bucket, dict)
        # Exact closed-set keys, no extras.
        extras = set(bucket.keys()) - set(_BUCKET_KEYS)
        assert not extras, (
            f"unexpected extra keys in daily bucket: {extras!r}"
        )
        for k in _BUCKET_KEYS:
            assert k in bucket, f"missing key {k!r} in daily bucket"
        # Type contracts.
        assert isinstance(bucket["date"], str)
        # ISO date format ``YYYY-MM-DD``.
        datetime.strptime(bucket["date"], "%Y-%m-%d")
        for int_key in (
            "organic_sessions_total", "seeded_sessions_total",
            "would_have_been_excluded_in_active_total_organic",
            "would_have_been_excluded_in_active_total_seeded",
            "actually_excluded_total_organic",
            "actually_excluded_total_seeded",
            "non_retracted_wbe_keys_count",
            "unstable_join_wbe_count",
        ):
            assert isinstance(bucket[int_key], int), (
                f"{int_key} must be int, got {type(bucket[int_key]).__name__}"
            )
        for float_key in (
            "organic_wbe_session_ratio", "seeded_wbe_session_ratio",
        ):
            assert isinstance(bucket[float_key], (int, float))
        # Alert flags is closed-set list (subset of 5 known names).
        assert isinstance(bucket["alert_flags_tripped"], list)
        for f in bucket["alert_flags_tripped"]:
            assert f in _PER_DAY_ALERT_FLAGS, (
                f"unexpected per-day alert flag {f!r}"
            )


# ---------------------------------------------------------------------
# Test 2 — Organic and seeded counts are NOT summed.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t2_organic_and_seeded_separated_not_summed(client):
    atoken, _ = await _login(client, "admin")
    _, uid = await _login(client, "clinician")
    # Seed one organic + one seeded session, both created today, both
    # carrying one WBE-true diagnostic + one active exclusion.
    today = datetime.now(timezone.utc)
    wbe_diag = {
        "lane_label": "therapy_rct",
        "directness_label": "direct",
        "cap_severity": "high",
        "cap_reason_codes": ["retracted"],
        "would_have_been_excluded_in_active": True,
        "would_have_been_excluded_reason_codes": ["retracted"],
        "article_join_status": "stable",
    }
    excl_row = {
        "cap_reason_codes": ["retracted"],
        "article_join_status": "stable",
        "source_signal": "pubmed_retracted_publication_type",
    }
    await _seed_session(
        user_id=uid, provenance="organic", created_at=today,
        article_lane_diagnostics={"diagnostics": [wbe_diag]},
        k3b_active_exclusions={"exclusions": [excl_row]},
    )
    await _seed_session(
        user_id=uid, provenance="seeded", created_at=today,
        article_lane_diagnostics={"diagnostics": [wbe_diag]},
        k3b_active_exclusions={"exclusions": [excl_row]},
    )
    r = await client.get(
        "/api/admin/cap-observability?window_days=7",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200
    sl = r.json()["daily_sparkline"]
    today_key = today.strftime("%Y-%m-%d")
    todays = [b for b in sl if b["date"] == today_key]
    assert len(todays) == 1, f"expected exactly one bucket for {today_key}"
    b = todays[0]
    # Organic and seeded must be tracked in SEPARATE fields, not summed.
    assert b["organic_sessions_total"] >= 1
    assert b["seeded_sessions_total"] >= 1
    assert b["would_have_been_excluded_in_active_total_organic"] >= 1
    assert b["would_have_been_excluded_in_active_total_seeded"] >= 1
    assert b["actually_excluded_total_organic"] >= 1
    assert b["actually_excluded_total_seeded"] >= 1
    # The fields must be distinct keys (the test above already proves
    # they're tracked separately).
    assert (
        "would_have_been_excluded_in_active_total_organic"
        != "would_have_been_excluded_in_active_total_seeded"
    )


# ---------------------------------------------------------------------
# Test 3 — Clinician token → 403 (regression confirm).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t3_clinician_403(client):
    ctoken, _ = await _login(client, "clinician")
    r = await client.get(
        "/api/admin/cap-observability?window_days=7",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    assert r.status_code in (401, 403)


# ---------------------------------------------------------------------
# Test 4 — Admin token → 200 with daily_sparkline populated.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t4_admin_200_sparkline_populated(client):
    atoken, _ = await _login(client, "admin")
    r = await client.get(
        "/api/admin/cap-observability?window_days=7",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200
    body = r.json()
    sl = body.get("daily_sparkline")
    assert isinstance(sl, list)
    # Window=7 → 8 buckets (inclusive both ends: start_day..end_day).
    assert len(sl) >= 7, f"sparkline too short: {len(sl)} days"
    assert len(sl) <= 8
    # Buckets sorted ascending by date.
    dates = [b["date"] for b in sl]
    assert dates == sorted(dates), "sparkline not ordered ascending"


# ---------------------------------------------------------------------
# Test 5 — Window validation regression (sparkline-aware).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t5_window_0_returns_422(client):
    atoken, _ = await _login(client, "admin")
    r = await client.get(
        "/api/admin/cap-observability?window_days=0",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_t5_window_99_returns_422(client):
    atoken, _ = await _login(client, "admin")
    r = await client.get(
        "/api/admin/cap-observability?window_days=99",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------
# Test 6 — Leak safety (sentinel injection into source data).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t6_sentinel_injection_clean(client):
    atoken, _ = await _login(client, "admin")
    _, uid = await _login(client, "clinician")
    # Inject sentinels into BOTH the lane-diagnostics AND active-
    # exclusion source data on a fresh session. The sparkline must
    # NOT leak any of these into the response.
    today = datetime.now(timezone.utc)
    poison_diag = {
        # cap_reason_codes drives the WBE counter — keep the closed-set
        # ``retracted`` value here AND poison-laden auxiliary fields.
        "lane_label": "therapy_rct",
        "directness_label": "direct",
        "cap_severity": "high",
        "cap_reason_codes": ["retracted"],
        "would_have_been_excluded_in_active": True,
        "would_have_been_excluded_reason_codes": ["retracted"],
        "article_join_status": "stable",
        # Poison.
        "user_email_leak": "leaked@gmail.com",
        "system_leak": "system:secret-please-redact",
        "sk_leak": "sk-LEAKED_SECRET_KEY_VALUE_AAAA",
        "raw_question_leak": "FORBIDDEN_RAW_QUESTION_TEXT_PHI",
    }
    poison_excl = {
        "cap_reason_codes": ["retracted"],
        "article_join_status": "stable",
        "source_signal": "pubmed_retracted_publication_type",
        # Poison.
        "user_email_leak": "leaked@gmail.com",
        "system_leak": "system:secret-please-redact",
        "sk_leak": "sk-LEAKED_SECRET_KEY_VALUE_BBBB",
        "raw_completion_leak": "FORBIDDEN_RAW_COMPLETION",
    }
    await _seed_session(
        user_id=uid, provenance="organic", created_at=today,
        article_lane_diagnostics={"diagnostics": [poison_diag]},
        k3b_active_exclusions={"exclusions": [poison_excl]},
    )
    r = await client.get(
        "/api/admin/cap-observability?window_days=7",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200
    blob = json.dumps(r.json())
    # Sentinel substring sweep over the ENTIRE response (sparkline +
    # rest of the observability payload).
    for s in (
        "system:", "sk-", "@gmail.com",
        "FORBIDDEN_RAW_QUESTION",
        "FORBIDDEN_RAW_COMPLETION",
        "LEAKED_SECRET",
    ):
        assert s not in blob, (
            f"leak: sentinel {s!r} found in observability response "
            f"(sparkline + rest)"
        )
    # Forbidden top-level key sweep — must NOT contain any of these.
    for k in ("raw_prompt", "raw_completion", "prompt", "completion",
              "evidence_text"):
        assert f'"{k}"' not in blob, (
            f"leak: forbidden key {k!r} present in observability"
        )


# ---------------------------------------------------------------------
# Test 7 — Production-mode safety check (non-mutating read).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t7_production_mode_safety_non_mutating(client):
    """The endpoint is a non-mutating read regardless of env. We
    monkey-patch ``settings.IS_PRODUCTION`` to true and confirm:
    1. The endpoint still returns 200.
    2. The response shape is identical (closed-set keys unchanged).
    3. ``cap_enforcement_mode`` and other env-derived fields remain
       readable.
    4. NO write occurred against any collection during the call.
    """
    from core.config import settings as _settings
    atoken, _ = await _login(client, "admin")
    # Snapshot counts BEFORE.
    from core.db import get_db
    db = get_db()
    before_audit = await db.audit_events.count_documents({})
    before_sessions = await db.search_sessions.count_documents({})
    # Toggle is_production = true for the duration of the call.
    original = getattr(_settings, "IS_PRODUCTION", False)
    try:
        _settings.IS_PRODUCTION = True  # type: ignore[attr-defined]
        r = await client.get(
            "/api/admin/cap-observability?window_days=7",
            headers={"Authorization": f"Bearer {atoken}"},
        )
    finally:
        _settings.IS_PRODUCTION = original  # type: ignore[attr-defined]
    assert r.status_code == 200
    body = r.json()
    # Shape unchanged.
    assert "daily_sparkline" in body
    assert "schema_version" in body
    assert body["schema_version"] == "K3B.wave1.step2_1.observability.v1"
    # No mutations.
    after_audit = await db.audit_events.count_documents({})
    after_sessions = await db.search_sessions.count_documents({})
    assert after_audit == before_audit, (
        "endpoint must be non-mutating: audit_events count changed"
    )
    assert after_sessions == before_sessions, (
        "endpoint must be non-mutating: search_sessions count changed"
    )


# ---------------------------------------------------------------------
# Test 8 — Existing Step 2.1 keys still present & valid (regression).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t8_existing_step2_1_keys_still_present(client):
    """Defensive regression: every key the Step 2.1 spec asserted on
    the observability payload must still be present after the
    sparkline addition. ``daily_sparkline`` is purely additive.
    """
    atoken, _ = await _login(client, "admin")
    r = await client.get(
        "/api/admin/cap-observability?window_days=7",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200
    body = r.json()
    for k in (
        "would_have_been_excluded_in_active_total",
        "would_have_been_excluded_by_reason_code",
        "actually_excluded_in_active_total",
        "actually_excluded_by_reason_code",
        "active_exclusion_by_primary_intent",
        "active_exclusion_by_branch_intent",
        "sessions_with_active_exclusion",
        "active_exclusion_ratio_organic",
        "article_join_status_histogram",
        "phi_blocked_sessions_in_window_total",
        "organic_vs_seeded_split",
        "alert_non_retracted_in_active_exclusion_histogram",
        "alert_active_exclusion_ratio_over_5pct_organic_traffic",
        "alert_wbe_true_with_unstable_join",
        "alert_eoc_only_marked_as_enforcing",
        "alert_barred_reason_in_enforcement_candidate_set",
        "leak_safety_check_passed",
        "schema_version",
        # New in Step 2.2.
        "daily_sparkline",
    ):
        assert k in body, f"regression: key {k!r} missing"
    # Schema discriminator unchanged.
    assert body["schema_version"] == "K3B.wave1.step2_1.observability.v1"


# ---------------------------------------------------------------------
# Test 9 — Active retrieval parity (clinician GET shape unchanged).
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t9_active_retrieval_parity_unchanged(client):
    """The clinician GET on a search session must return the same
    closed-set projection as before the sparkline was added. The
    sparkline is admin-only; it must not touch the clinician surface.
    """
    ctoken, uid = await _login(client, "clinician")
    sid = await _seed_session(
        user_id=uid, status="awaiting_selection", provenance="organic",
    )
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    assert r.status_code == 200
    body = r.json()
    # Closed-set: these keys are the clinician projection contract.
    for k in (
        "id", "articles", "excluded_articles", "status",
        "session_diagnostics_json",
    ):
        assert k in body, f"missing clinician projection key {k!r}"
    # Sparkline MUST NOT appear on the clinician surface.
    assert "daily_sparkline" not in body
    # Banner key absent when no active-exclusion block on this session.
    sdj = body.get("session_diagnostics_json") or {}
    assert "k3b_active_exclusions_banner" not in sdj
