"""Tests for the clinician-search ECS worker entrypoint.

No AWS, no database: db and pipeline are injected. The two load-bearing
behaviors: a finished-with-error session makes the task FAIL (run_pipeline
never raises, so this is where ECS retries come from), and a redelivered
already-terminal session is ACKed without re-running a ~30-minute pipeline.
"""

import json

import pytest

from ecs.clinician_search.entrypoint import execute, extract_bodies
from jobs.contracts import PermanentInputError

SID, UID = "ses_456", "usr_123"


def envelope(**data_over):
    data = {"search_session_id": SID, "user_id": UID}
    data.update(data_over)
    return {
        "schema_version": "1.0",
        "job": "litportal.clinician_search_pipeline",
        "correlation_id": "corr-1",
        "idempotency_key": f"search_pipeline#{SID}",
        "enqueued_at": "2026-08-13T08:10:00.000Z",
        "producer": "litportal-backend",
        "data": data,
    }


class FakeSessions:
    def __init__(self, doc):
        self.doc = doc

    async def find_one(self, query, projection=None):
        if self.doc and self.doc.get("id") == query.get("id"):
            return dict(self.doc)
        return None


class FakeDB:
    def __init__(self, doc):
        self.sessions = FakeSessions(doc)

    def __getitem__(self, name):
        assert name == "search_sessions"
        return self.sessions


def session(**over):
    base = {
        "id": SID,
        "user_id": UID,
        "status": "searching",
        "sanitized_question": "empagliflozin for hfpef?",
    }
    base.update(over)
    return base


def pipeline_recorder(final_status="awaiting_selection", db=None):
    calls = []

    async def _pipeline(session_id, user_id, sanitized_question):
        calls.append((session_id, user_id, sanitized_question))
        if db is not None:
            db.sessions.doc["status"] = final_status

    return _pipeline, calls


# ── Pipe event shapes ────────────────────────────────────────────────


def test_extract_bodies_accepts_bare_envelope():
    assert extract_bodies(envelope()) == [envelope()]


def test_extract_bodies_accepts_pipes_sqs_record_list():
    records = [{"messageId": "m1", "body": json.dumps(envelope())}]
    assert extract_bodies(records) == [envelope()]


def test_extract_bodies_accepts_json_string():
    assert extract_bodies(json.dumps([envelope()])) == [envelope()]


def test_extract_bodies_rejects_garbage():
    with pytest.raises(PermanentInputError):
        extract_bodies(42)


# ── execution ────────────────────────────────────────────────────────


async def test_runs_pipeline_with_session_question_and_succeeds():
    db = FakeDB(session())
    pipeline, calls = pipeline_recorder("awaiting_selection", db)
    result = await execute(envelope(), db=db, pipeline=pipeline)
    assert result.status == "awaiting_selection"
    assert calls == [(SID, UID, "empagliflozin for hfpef?")]


async def test_error_status_fails_the_task():
    """run_pipeline never raises — the worker must convert status=error into a
    failure, or ECS/alarms never see it."""
    db = FakeDB(session())
    pipeline, _ = pipeline_recorder("error", db)
    db.sessions.doc["error_message"] = "pipeline error: boom"
    with pytest.raises(RuntimeError, match="status='error'"):
        await execute(envelope(), db=db, pipeline=pipeline)


async def test_already_terminal_session_is_not_rerun():
    """Redelivery must not re-run a 30-minute pipeline (and prod showed
    DuplicateKeyError crashes when the same session ran twice)."""
    db = FakeDB(session(status="awaiting_selection"))
    pipeline, calls = pipeline_recorder(db=db)
    result = await execute(envelope(), db=db, pipeline=pipeline)
    assert result.status == "awaiting_selection"
    assert calls == []


async def test_missing_session_is_permanent():
    with pytest.raises(PermanentInputError, match="does not exist"):
        await execute(envelope(), db=FakeDB(None), pipeline=pipeline_recorder()[0])


async def test_wrong_owner_is_permanent():
    db = FakeDB(session(user_id="someone-else"))
    with pytest.raises(PermanentInputError, match="own"):
        await execute(envelope(), db=db, pipeline=pipeline_recorder()[0])


@pytest.mark.parametrize("missing", ["search_session_id", "user_id"])
async def test_missing_required_field_is_permanent(missing):
    body = envelope()
    body["data"].pop(missing)
    with pytest.raises(PermanentInputError):
        await execute(body)
