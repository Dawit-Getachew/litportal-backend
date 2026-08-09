"""Search session: PHI-blocked + clean (pipeline kicked off) paths."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_blocked_phi_path_does_not_store_question(client):
    r = await client.post(
        "/api/auth/signup",
        json={"email": "phitest@test.com", "password": "Pass1234567!", "name": "PHI Tester"},
    )
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # The pipeline MUST NOT be scheduled when PHI is blocked.
    with patch("routers.search_sessions.schedule_pipeline") as sched:
        bad = "45M with MRN 12345678, DOB 03/14/1979, email john@x.com — best tx for HTN?"
        r2 = await client.post("/api/search-sessions/", json={"question": bad}, headers=headers)
        assert r2.status_code == 200, r2.text
        sched.assert_not_called()
    body = r2.json()
    assert body["status"] == "blocked_phi"
    assert isinstance(body["reason_codes"], list) and len(body["reason_codes"]) >= 1
    assert bad not in body["user_message"]

    sid = body["session_id"]
    r3 = await client.get(f"/api/search-sessions/{sid}", headers=headers)
    assert r3.status_code == 200
    session = r3.json()
    assert session["status"] == "blocked_phi"
    assert session["original_question"] is None
    assert session["sanitized_question"] is None


@pytest.mark.asyncio
async def test_clean_question_kicks_off_pipeline(client):
    r = await client.post(
        "/api/auth/signup",
        json={"email": "cleanq@test.com", "password": "Pass1234567!", "name": "Clean Q"},
    )
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with patch("routers.search_sessions.schedule_pipeline") as sched:
        r2 = await client.post(
            "/api/search-sessions/",
            json={"question": "What is the evidence for SGLT2 inhibitors in HFpEF?"},
            headers=headers,
        )
        assert r2.status_code == 200, r2.text
        sched.assert_called_once()
    body = r2.json()
    assert body["status"] == "searching"
    assert body["original_question"] == "What is the evidence for SGLT2 inhibitors in HFpEF?"
    expires_at = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00"))
    delta = expires_at - datetime.now(timezone.utc)
    assert timedelta(days=29) < delta < timedelta(days=31)


@pytest.mark.asyncio
async def test_list_sessions_grouped(client):
    r = await client.post(
        "/api/auth/signup",
        json={"email": "listq@test.com", "password": "Pass1234567!", "name": "List Q"},
    )
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    with patch("routers.search_sessions.schedule_pipeline"):
        await client.post(
            "/api/search-sessions/",
            json={"question": "Best evidence for early metformin in T2DM?"},
            headers=headers,
        )
    r2 = await client.get("/api/search-sessions/?group=recent", headers=headers)
    assert r2.status_code == 200
    body = r2.json()
    assert set(body.keys()) == {"today", "previous_7_days", "previous_30_days"}
    assert len(body["today"]) >= 1
