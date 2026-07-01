"""Phase 5F.2 — cross-user / admin access control for
``GET /api/search-sessions/{id}``.

- owner GETs own session → 200
- different clinician GETs another user's session → 404
- admin GETs any user's session → 200
- unauthenticated GET → 401
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_owner_other_clinician_admin_session_visibility(client):
    """Three users:
      - alice: clinician, owns the session
      - bob:   clinician, must NOT see alice's session
      - admin: admin role, MUST see alice's session
    """
    # 1. Sign up alice + bob.
    alice = await client.post(
        "/api/auth/signup",
        json={"email": "alice5f2@test.com", "password": "Pass1234567!", "name": "Alice"},
    )
    alice_tok = alice.json()["access_token"]
    alice_h = {"Authorization": f"Bearer {alice_tok}"}

    bob = await client.post(
        "/api/auth/signup",
        json={"email": "bob5f2@test.com", "password": "Pass1234567!", "name": "Bob"},
    )
    bob_tok = bob.json()["access_token"]
    bob_h = {"Authorization": f"Bearer {bob_tok}"}

    # 2. Promote a third user to admin role directly via DB. (Signing
    #    up with role=admin is not exposed by the public API.)
    admin = await client.post(
        "/api/auth/signup",
        json={"email": "admin5f2@test.com", "password": "Pass1234567!", "name": "Admin Tester"},
    )
    admin_tok = admin.json()["access_token"]
    admin_h = {"Authorization": f"Bearer {admin_tok}"}
    # Bump the role via the DB (mirrors what the seed flow does).
    from core.db import Collections, get_db
    db = get_db()
    await db[Collections.USERS].update_one(
        {"email": "admin5f2@test.com"}, {"$set": {"role": "admin"}},
    )

    # Re-login to pick up the new role in the JWT.
    relogin = await client.post(
        "/api/auth/login",
        json={"email": "admin5f2@test.com", "password": "Pass1234567!"},
    )
    admin_tok = relogin.json()["access_token"]
    admin_h = {"Authorization": f"Bearer {admin_tok}"}

    # 3. Alice creates a session. (Mock the pipeline so we don't make
    #    LLM calls in tests.)
    with patch("routers.search_sessions.schedule_pipeline"):
        r = await client.post(
            "/api/search-sessions/",
            json={"question": "What is the evidence for SGLT2 inhibitors in HFpEF?"},
            headers=alice_h,
        )
        assert r.status_code == 200, r.text
        sid = r.json()["id"]

    # 4. Owner can GET own session → 200.
    r = await client.get(f"/api/search-sessions/{sid}", headers=alice_h)
    assert r.status_code == 200, r.text
    assert r.json()["id"] == sid

    # 5. Other clinician GETs the session → 404 (no info leak; the
    #    response body must NOT echo the original question).
    r = await client.get(f"/api/search-sessions/{sid}", headers=bob_h)
    assert r.status_code == 404, r.text
    body = r.text
    assert "SGLT2" not in body
    assert "HFpEF" not in body

    # 6. Admin GETs the session → 200.
    r = await client.get(f"/api/search-sessions/{sid}", headers=admin_h)
    assert r.status_code == 200, r.text
    assert r.json()["id"] == sid

    # 7. Unauthenticated → 401 (or 403 — accept either; matches the
    #    project's existing auth-dependency behaviour).
    r = await client.get(f"/api/search-sessions/{sid}")
    assert r.status_code in (401, 403), r.text


@pytest.mark.asyncio
async def test_cross_user_404_not_distinguishable_from_missing(client):
    """A bogus session id and a real-but-not-owned session id MUST
    both return 404 with the same body shape — so the cross-user
    caller can't deduce whether the session exists."""
    alice = await client.post(
        "/api/auth/signup",
        json={"email": "alice5f2b@test.com", "password": "Pass1234567!", "name": "Alice B"},
    )
    alice_tok = alice.json()["access_token"]
    alice_h = {"Authorization": f"Bearer {alice_tok}"}
    bob = await client.post(
        "/api/auth/signup",
        json={"email": "bob5f2b@test.com", "password": "Pass1234567!", "name": "Bob B"},
    )
    bob_h = {"Authorization": f"Bearer {bob.json()['access_token']}"}

    with patch("routers.search_sessions.schedule_pipeline"):
        r = await client.post(
            "/api/search-sessions/",
            json={"question": "What about ACE inhibitors after MI?"},
            headers=alice_h,
        )
        sid = r.json()["id"]

    bogus_id = "00000000-0000-0000-0000-000000000000"
    a = await client.get(f"/api/search-sessions/{sid}", headers=bob_h)
    b = await client.get(f"/api/search-sessions/{bogus_id}", headers=bob_h)
    assert a.status_code == 404 and b.status_code == 404
    # Identical bodies (no info leak).
    assert a.json()["detail"] == b.json()["detail"]
