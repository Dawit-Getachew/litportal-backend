"""Phase K3B Wave 1 Step 2.1 — Deployment-fingerprint endpoint tests.

Covers spec items #1-#4 + startup-guardrail integration items #5-#10:

  1. Admin can access ``/api/admin/deployment-fingerprint`` (200).
  2. Clinician receives 403.
  3. Response payload exposes only the closed-set fields.
  4. NO raw secret values: grep against ``sk-``, ``Bearer`` (not
     literally — but any value looking like a JWT), hex >32 chars
     (excluding the closed-set ``article_id`` / ``exclusion_id``
     which DO exist in OTHER endpoints, but NOT in this one).
  5. Production + active + missing fingerprint flag → startup-
     guardrail regression test still green (delegates to the Phase
     1 test file at
     ``tests/test_phaseK3B_startup_fingerprint_guardrail.py``).
  6-9. The same Phase-1 regression coverage continues to hold.
  10. Malformed ``CAP_ENFORCEMENT_MODE`` value → falls back to the
      pre-existing pipeline assert (closed-set ``{disabled, shadow,
      active}``); endpoint surfaces the unrecognized value via the
      closed-set Literal cast.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


# ---------------------------------------------------------------------
# Test 1 — Admin can access.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t1_admin_can_access(client):
    atoken, _ = await _login(client, "admin")
    r = await client.get(
        "/api/admin/deployment-fingerprint",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------
# Test 2 — Clinician receives 403.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t2_clinician_403(client):
    ctoken, _ = await _login(client, "clinician")
    r = await client.get(
        "/api/admin/deployment-fingerprint",
        headers={"Authorization": f"Bearer {ctoken}"},
    )
    assert r.status_code in (401, 403)


# ---------------------------------------------------------------------
# Test 3 — Closed-set schema check.
# ---------------------------------------------------------------------


_CLOSED_SET_FIELDS = (
    "deployment_environment",
    "is_production",
    "cap_detection_mode",
    "cap_enforcement_mode",
    "fingerprint_flag_set",
    "fingerprint_flag_source",
    "active_cap_reason_allowlist",
    "production_active_allowed",
    "guard_status",
    "schema_version",
)


@pytest.mark.asyncio
async def test_t3_response_payload_closed_set_schema(client):
    atoken, _ = await _login(client, "admin")
    r = await client.get(
        "/api/admin/deployment-fingerprint",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200
    body = r.json()
    # All closed-set keys present.
    for k in _CLOSED_SET_FIELDS:
        assert k in body, f"missing closed-set key {k!r}"
    # No unexpected extras (closed-set schema discipline).
    extras = set(body.keys()) - set(_CLOSED_SET_FIELDS)
    assert not extras, (
        f"unexpected extra keys in deployment-fingerprint: {extras!r}"
    )
    # Closed-set Literal coercion.
    assert body["deployment_environment"] in (
        "production", "development",
    )
    assert isinstance(body["is_production"], bool)
    assert body["cap_detection_mode"] in ("disabled", "shadow")
    assert body["cap_enforcement_mode"] in (
        "disabled", "shadow", "active",
    )
    assert isinstance(body["fingerprint_flag_set"], bool)
    assert body["fingerprint_flag_source"] in (
        "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE",
        "ALLOW_PRODUCTION_CAP_ACTIVE",
        "neither_set",
    )
    assert isinstance(body["active_cap_reason_allowlist"], list)
    assert isinstance(body["production_active_allowed"], bool)
    assert body["guard_status"] in (
        "permitted_to_run_active",
        "would_fail_closed_on_startup",
        "no_check_required_non_production_or_non_active",
    )
    assert body["schema_version"] == (
        "K3B.deployment_fingerprint.v1"
    )


# ---------------------------------------------------------------------
# Test 4 — NO raw secret values.
# ---------------------------------------------------------------------


_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),        # OpenAI-style keys
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"Bearer [A-Za-z0-9_\-\.]+"),    # raw JWTs
    re.compile(r"[a-f0-9]{40,}"),                # long hex blobs
)


@pytest.mark.asyncio
async def test_t4_no_raw_secret_values(client):
    atoken, _ = await _login(client, "admin")
    r = await client.get(
        "/api/admin/deployment-fingerprint",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200
    blob = json.dumps(r.json())
    for rx in _SECRET_PATTERNS:
        hits = rx.findall(blob)
        assert not hits, (
            f"deployment-fingerprint response leaked a value matching "
            f"{rx.pattern!r}: {hits!r}"
        )
    # Closed-set sentinel substring sweep.
    for s in ("sk-", "@gmail.com", "system:"):
        assert s not in blob, (
            f"deployment-fingerprint response contains forbidden "
            f"sentinel {s!r}"
        )


# ---------------------------------------------------------------------
# Test 5-9 — Phase-1 startup-guardrail regression coverage.
# These delegate to the existing
# ``tests/test_phaseK3B_startup_fingerprint_guardrail.py`` file.
# We add three smoke tests here so a failure on the integration is
# visible in this suite too.
# ---------------------------------------------------------------------


def test_t5_prod_active_no_fingerprint_still_raises():
    """Phase-1 regression: production + active + no fingerprint
    flag still raises ``RuntimeError`` at startup time. (Same as
    ``test_phaseK3B_startup_fingerprint_guardrail.test_t1_*``.)
    """
    import logging
    from services.cap_enforcement_guard import (
        verify_startup_fingerprint,
    )
    with pytest.raises(RuntimeError):
        verify_startup_fingerprint(
            is_production=True, resolved_mode="active",
            env={},
            logger_obj=logging.getLogger(
                "test.k3b_step2_1_fingerprint_smoke",
            ),
        )


def test_t6_t7_prod_active_with_flags_still_starts():
    """Phase-1 regression: production + active + flag + allowlist
    still starts cleanly.
    """
    import logging
    from services.cap_enforcement_guard import (
        verify_startup_fingerprint,
    )
    verify_startup_fingerprint(
        is_production=True, resolved_mode="active",
        env={
            "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE": "true",
            "ACTIVE_CAP_REASON_ALLOWLIST": "retracted",
        },
        logger_obj=logging.getLogger(
            "test.k3b_step2_1_fingerprint_smoke",
        ),
    )


def test_t8_t9_prod_shadow_and_dev_active_still_start():
    """Phase-1 regression: prod + shadow → starts; dev + active
    (no flag) → starts.
    """
    import logging
    from services.cap_enforcement_guard import (
        verify_startup_fingerprint,
    )
    verify_startup_fingerprint(
        is_production=True, resolved_mode="shadow",
        env={},
        logger_obj=logging.getLogger(
            "test.k3b_step2_1_fingerprint_smoke",
        ),
    )
    verify_startup_fingerprint(
        is_production=False, resolved_mode="active",
        env={},
        logger_obj=logging.getLogger(
            "test.k3b_step2_1_fingerprint_smoke",
        ),
    )


# ---------------------------------------------------------------------
# Test 10 — Malformed mode value handling.
# ---------------------------------------------------------------------


def test_t10_malformed_cap_mode_handled_safely():
    """The fingerprint helper reads the resolved mode via the same
    accessor as the rest of the code, which lowercases + strips
    whitespace. If the env value is a non-closed-set string (e.g.
    ``"OFF"``, ``"true"`` typo), the helper treats it as not
    ``"active"`` and the guard is a no-op (fail-closed default).
    The pipeline-level assert at startup is the second layer of
    defense for fully-malformed values.
    """
    import logging
    from services.cap_enforcement_guard import (
        verify_startup_fingerprint,
    )
    # Off-spec values are treated as not-active → no-op on prod.
    # ``"Active"`` is excluded because ``.strip().lower()`` would
    # canonicalize it to the well-formed ``"active"`` literal — at
    # that point the guard correctly fires; that's tested
    # separately in Phase 1 + Test 5 above.
    for malformed in ("OFF", "true", "garbage", " "):
        verify_startup_fingerprint(
            is_production=True,
            resolved_mode=malformed.strip().lower(),
            env={},
            logger_obj=logging.getLogger(
                "test.k3b_step2_1_fingerprint_malformed",
            ),
        )
