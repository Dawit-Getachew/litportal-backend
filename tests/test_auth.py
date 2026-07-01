"""Auth happy-path: signup → /me, login, admin gate, refresh cookie."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_signup_login_me_flow(client):
    email = "newclinician@test.com"
    password = "Newpass1234!"
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": password, "name": "New Clinician"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    token = body["access_token"]
    assert body["user"]["email"] == email
    assert body["user"]["role"] == "user"
    assert body["user"]["first_use_acknowledged"] is False
    # Refresh cookie should be set
    assert "ces_refresh_token" in r.headers.get("set-cookie", "").lower() or any(
        "ces_refresh_token" in c.name for c in r.cookies.jar
    )

    r2 = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200
    assert r2.json()["email"] == email

    r3 = await client.post(
        "/api/auth/login", json={"email": email, "password": password}
    )
    assert r3.status_code == 200
    token2 = r3.json()["access_token"]
    r4 = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token2}"})
    assert r4.status_code == 200


@pytest.mark.asyncio
async def test_signup_duplicate_email(client):
    email = "dup@test.com"
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!", "name": "Dup"},
    )
    assert r.status_code == 201
    r2 = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!", "name": "Dup"},
    )
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_weak_password_rejected(client):
    # Too short
    r = await client.post(
        "/api/auth/signup",
        json={"email": "short@test.com", "password": "Short1!", "name": "X"},
    )
    assert r.status_code == 422
    # Missing special char
    r2 = await client.post(
        "/api/auth/signup",
        json={"email": "nospec@test.com", "password": "NoSpecial1234", "name": "X"},
    )
    assert r2.status_code == 422
    # Missing number
    r3 = await client.post(
        "/api/auth/signup",
        json={"email": "nonum@test.com", "password": "NoNumberHere!", "name": "X"},
    )
    assert r3.status_code == 422


@pytest.mark.asyncio
async def test_me_without_token_401(client):
    r = await client.get("/api/auth/me")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_admin_route_forbidden_for_user(client):
    r = await client.post(
        "/api/auth/signup",
        json={"email": "user1@test.com", "password": "Pass1234567!", "name": "U1"},
    )
    token = r.json()["access_token"]
    r2 = await client.get(
        "/api/admin/dashboard",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r2.status_code == 403


@pytest.mark.asyncio
async def test_admin_route_works_for_seeded_admin(client):
    r = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    r2 = await client.get(
        "/api/admin/dashboard",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert "source_configs" in body
    assert body["source_configs"]["total"] >= 16


@pytest.mark.asyncio
async def test_refresh_endpoint_returns_new_access_token(client):
    # Sign up to get the refresh cookie set on the client
    email = "refresh@test.com"
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Refresh12345!", "name": "Refresh"},
    )
    assert r.status_code == 201
    first_token = r.json()["access_token"]

    # The httpx AsyncClient should have stored the refresh cookie. Use refresh.
    r2 = await client.post("/api/auth/refresh")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert "access_token" in body
    assert body["user"]["email"] == email
    # Different identities — at minimum the iat differs; just confirm we got a token.
    assert isinstance(body["access_token"], str) and len(body["access_token"]) > 20
    # First token should still be valid (until natural expiry)
    r3 = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {first_token}"})
    assert r3.status_code == 200


@pytest.mark.asyncio
async def test_refresh_without_cookie_401(client):
    # New client w/o cookies via header-only request: explicitly clear cookies.
    client.cookies.clear()
    r = await client.post("/api/auth/refresh")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_seeded_clinician_login(client):
    r = await client.post(
        "/api/auth/login",
        json={"email": "clinician@test.com", "password": "ClinicianTest123!"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["user"]["role"] == "user"
    assert r.json()["user"]["first_use_acknowledged"] is False
