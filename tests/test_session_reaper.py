"""Stale search-session reaper (R1): stuck transient sessions must be flipped
to error; live runs and terminal sessions must never be touched."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.session_reaper import (
    REAPED_ERROR_MESSAGE,
    TRANSIENT_STATUSES,
    reap_stale_sessions_once,
    session_reaper_loop,
)

NOW = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


class FakeSessions:
    def __init__(self, docs):
        self.docs = docs

    async def update_many(self, query, update):
        cutoff = None
        for clause in query["$or"]:
            if "$lt" in clause.get("updated_at", {}):
                cutoff = clause["updated_at"]["$lt"]
        modified = 0
        for d in self.docs:
            if d["status"] not in query["status"]["$in"]:
                continue
            ts = d.get("updated_at") or d.get("created_at")
            if ts is not None and ts < cutoff:
                d.update(update["$set"])
                modified += 1
        return SimpleNamespace(modified_count=modified)


class FakeDB:
    def __init__(self, docs):
        self.sessions = FakeSessions(docs)

    def __getitem__(self, name):
        assert name == "search_sessions"
        return self.sessions


def doc(status, age_minutes, **over):
    base = {
        "id": f"ses-{status}-{age_minutes}",
        "status": status,
        "updated_at": NOW - timedelta(minutes=age_minutes),
    }
    base.update(over)
    return base


async def test_stuck_ranking_session_is_reaped():
    """The observed prod failure: sessions stuck in 'ranking' for weeks."""
    stuck = doc("ranking", age_minutes=60 * 24 * 20)
    db = FakeDB([stuck])
    assert await reap_stale_sessions_once(db, now=NOW) == 1
    assert stuck["status"] == "error"
    assert stuck["error_message"] == REAPED_ERROR_MESSAGE
    assert stuck["reaped_at"] == NOW


@pytest.mark.parametrize("status", TRANSIENT_STATUSES)
async def test_every_transient_status_is_covered(status):
    stuck = doc(status, age_minutes=120)
    assert await reap_stale_sessions_once(FakeDB([stuck]), now=NOW) == 1
    assert stuck["status"] == "error"


async def test_live_run_is_not_reaped():
    """90-minute threshold: a 33-minute worst-case run must never be touched."""
    live = doc("ranking", age_minutes=35)
    assert await reap_stale_sessions_once(FakeDB([live]), now=NOW) == 0
    assert live["status"] == "ranking"


@pytest.mark.parametrize(
    "status", ["awaiting_selection", "complete", "error", "blocked_phi"]
)
async def test_terminal_sessions_are_never_touched(status):
    terminal = doc(status, age_minutes=60 * 24 * 30)
    assert await reap_stale_sessions_once(FakeDB([terminal]), now=NOW) == 0
    assert terminal["status"] == status


async def test_missing_updated_at_falls_back_to_created_at():
    old = {
        "id": "ses-old",
        "status": "searching",
        "created_at": NOW - timedelta(days=3),
    }
    assert await reap_stale_sessions_once(FakeDB([old]), now=NOW) == 1
    assert old["status"] == "error"


async def test_loop_respects_disable_flag(monkeypatch):
    monkeypatch.setenv("SESSION_REAPER_ENABLED", "0")
    stop = asyncio.Event()
    await asyncio.wait_for(session_reaper_loop(stop), timeout=1.0)
