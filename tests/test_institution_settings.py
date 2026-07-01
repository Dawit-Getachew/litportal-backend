"""Phase 2C — institution settings endpoint."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


async def _signup(client, email):
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!", "name": "T"},
    )
    return r.json()["access_token"], r.json()["user"]["id"]


@pytest.mark.asyncio
async def test_user_can_save_openathens_domain(client):
    token, _ = await _signup(client, "is1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    r = await client.put(
        "/api/user/institution-settings",
        json={"openathens_customer_domain": "mock-university"},
        headers=h,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["openathens_customer_domain"] == "mock-university"
    # GET returns the same value.
    r2 = await client.get("/api/user/institution-settings", headers=h)
    assert r2.status_code == 200
    assert r2.json()["openathens_customer_domain"] == "mock-university"


@pytest.mark.asyncio
async def test_user_can_save_openurl_resolver(client):
    token, _ = await _signup(client, "is2@test.com")
    h = {"Authorization": f"Bearer {token}"}
    r = await client.put(
        "/api/user/institution-settings",
        json={"openurl_resolver_base_url": "https://library.example/openurl"},
        headers=h,
    )
    assert r.status_code == 200
    assert r.json()["openurl_resolver_base_url"] == "https://library.example/openurl"


@pytest.mark.asyncio
async def test_invalid_resolver_rejects(client):
    token, _ = await _signup(client, "is3@test.com")
    h = {"Authorization": f"Bearer {token}"}
    # No scheme
    r = await client.put(
        "/api/user/institution-settings",
        json={"openurl_resolver_base_url": "library.example/openurl"},
        headers=h,
    )
    assert r.status_code == 422
    # Has query string
    r = await client.put(
        "/api/user/institution-settings",
        json={"openurl_resolver_base_url": "https://library.example/openurl?foo=1"},
        headers=h,
    )
    assert r.status_code == 422
    # Plain text
    r = await client.put(
        "/api/user/institution-settings",
        json={"openurl_resolver_base_url": "not a url at all"},
        headers=h,
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_openathens_domain_rejects_protocol_prefix(client):
    token, _ = await _signup(client, "is4@test.com")
    h = {"Authorization": f"Bearer {token}"}
    r = await client.put(
        "/api/user/institution-settings",
        json={"openathens_customer_domain": "https://mock.university"},
        headers=h,
    )
    assert r.status_code == 422
    # Spaces also invalid
    r = await client.put(
        "/api/user/institution-settings",
        json={"openathens_customer_domain": "mock university"},
        headers=h,
    )
    assert r.status_code == 422
    # @ invalid
    r = await client.put(
        "/api/user/institution-settings",
        json={"openathens_customer_domain": "user@mock-u"},
        headers=h,
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_credential_like_fields_rejected(client):
    token, _ = await _signup(client, "is5@test.com")
    h = {"Authorization": f"Bearer {token}"}
    # Field NAME is suspect.
    r = await client.put(
        "/api/user/institution-settings",
        json={"password": "secret123", "openathens_customer_domain": "mock-u"},
        headers=h,
    )
    assert r.status_code == 400
    assert "credentials" in r.json()["detail"].lower()
    # Field VALUE contains "bearer XYZ"
    r = await client.put(
        "/api/user/institution-settings",
        json={"openathens_customer_domain": "Bearer abc123token"},
        headers=h,
    )
    assert r.status_code == 400
    # Field name "username"
    r = await client.put(
        "/api/user/institution-settings",
        json={"username": "alice"},
        headers=h,
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_user_cannot_read_other_users_settings(client):
    # User A saves settings.
    token_a, _ = await _signup(client, "isA@test.com")
    h_a = {"Authorization": f"Bearer {token_a}"}
    await client.put(
        "/api/user/institution-settings",
        json={"openathens_customer_domain": "alice-u"},
        headers=h_a,
    )
    # User B gets DEFAULT (empty) settings; not A's.
    token_b, _ = await _signup(client, "isB@test.com")
    h_b = {"Authorization": f"Bearer {token_b}"}
    r = await client.get("/api/user/institution-settings", headers=h_b)
    assert r.status_code == 200
    assert r.json()["openathens_customer_domain"] is None


@pytest.mark.asyncio
async def test_clear_resets_to_defaults(client):
    token, _ = await _signup(client, "is6@test.com")
    h = {"Authorization": f"Bearer {token}"}
    await client.put(
        "/api/user/institution-settings",
        json={"openathens_customer_domain": "mock-u"},
        headers=h,
    )
    r = await client.delete("/api/user/institution-settings", headers=h)
    assert r.status_code == 200
    assert r.json()["openathens_customer_domain"] is None
    r2 = await client.get("/api/user/institution-settings", headers=h)
    assert r2.json()["openathens_customer_domain"] is None
