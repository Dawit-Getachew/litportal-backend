"""SCOPE AUDIT — auth chain (B9) + PHI/governance smoke, through the live app.

Uses the session-scoped ``client`` fixture (ASGI over Atlas) + seeded clinician.
Covers the native-token validator behavior and the non-negotiable PHI preflight.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from core.config import settings


async def _token(client) -> str:
    r = await client.post("/api/auth/login", json={
        "email": "clinician@test.com", "password": "ClinicianTest123!"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


async def test_valid_token_authenticates(client):
    tok = await _token(client)
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["email"] == "clinician@test.com"


async def test_no_auth_rejected(client):
    r = await client.get("/api/auth/me")
    assert r.status_code in (401, 403)


@pytest.mark.parametrize("bad", ["not.a.jwt", "Bearerless", "", "a.b.c"])
async def test_garbage_token_rejected(client, bad):
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {bad}"})
    assert r.status_code == 401


async def test_expired_native_token_rejected(client):
    now = datetime.now(timezone.utc)
    expired = jwt.encode(
        {"sub": "x", "email": "x@test.com", "role": "user", "type": "access",
         "iat": int((now - timedelta(hours=2)).timestamp()),
         "exp": int((now - timedelta(hours=1)).timestamp())},
        settings.JWT_ACCESS_SECRET, algorithm=settings.JWT_ALGORITHM)
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401


async def test_wrong_type_token_rejected(client):
    now = datetime.now(timezone.utc)
    refreshish = jwt.encode(
        {"sub": "x", "type": "refresh",
         "iat": int(now.timestamp()), "exp": int((now + timedelta(hours=1)).timestamp())},
        settings.JWT_ACCESS_SECRET, algorithm=settings.JWT_ALGORITHM)
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {refreshish}"})
    assert r.status_code == 401


async def test_wrong_signature_token_rejected(client):
    now = datetime.now(timezone.utc)
    forged = jwt.encode(
        {"sub": "x", "email": "x@test.com", "role": "admin", "type": "access",
         "iat": int(now.timestamp()), "exp": int((now + timedelta(hours=1)).timestamp())},
        "totally-wrong-secret", algorithm="HS256")
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


# ── Governance: PHI preflight must block before any pipeline work ──
@pytest.mark.parametrize("phi_question", [
    "Patient John Smith MRN 12345678 with SSN 123-45-6789 dob 01/01/1950 has sepsis",
    "my patient jane.doe@hospital.org phone 555-123-4567 needs anticoagulation advice",
])
async def test_phi_query_blocked_before_pipeline(client, phi_question):
    tok = await _token(client)
    r = await client.post("/api/search-sessions/", json={"question": phi_question},
                          headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "blocked_phi"
    # The raw PHI question must NOT be echoed back.
    assert "123-45-6789" not in str(body)


async def test_clean_question_creates_session(client):
    tok = await _token(client)
    r = await client.post(
        "/api/search-sessions/",
        json={"question": "In adults with hypertension, do ACE inhibitors reduce stroke risk?"},
        headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") not in ("blocked_phi", None)
    assert body.get("id") or body.get("session_id")
