"""Contract + handler tests for the summary-generation Lambda.

Runs with no AWS and no database: the job's business logic is injected, so
these assert the packaging contract — envelope validation, dedup behavior,
and (most importantly) that failures PROPAGATE so SQS can redeliver and the
DLQ works.
"""

import json
from pathlib import Path

import pytest

from jobs.contracts import PermanentInputError, SummaryGenerationMessage
from jobs.summary_generation import run
from src.main import handler

EVENTS = Path(__file__).resolve().parent.parent / "events"


# ── helpers ──────────────────────────────────────────────────────────


def envelope(**data_over):
    data = {
        "job_id": "job-1",
        "user_id": "usr_123",
        "session_id": "ses_456",
        "selected_article_ids": ["art_1"],
        "acknowledge_abstract_only_limitations": False,
    }
    data.update(data_over)
    return {
        "schema_version": "1.0",
        "job": "litportal.summary_generation",
        "correlation_id": "corr-1",
        "idempotency_key": "summary_generation#usr_123#ses_456",
        "enqueued_at": "2026-08-09T12:34:56.789Z",
        "producer": "litportal-backend",
        "data": data,
    }


def make_sqs_event(*bodies):
    return {
        "Records": [
            {
                "messageId": f"msg-{i}",
                "receiptHandle": f"handle-{i}",
                "body": json.dumps(b) if isinstance(b, dict) else b,
                "attributes": {"ApproximateReceiveCount": "1"},
                "messageAttributes": {},
                "md5OfBody": "test",
                "eventSource": "aws:sqs",
                "eventSourceARN": "arn:aws:sqs:us-east-1:1:scienthesis-summary-generation",
                "awsRegion": "us-east-1",
            }
            for i, b in enumerate(bodies)
        ]
    }


class FakeState:
    """Stand-in for the Redis dedup store."""

    def __init__(self, already_held: bool = False):
        self.already_held = already_held
        self.claims: list[str] = []
        self.releases: list[str] = []

    async def claim(self, key: str, ttl_seconds: int) -> bool:
        self.claims.append(key)
        return not self.already_held

    async def release(self, key: str) -> None:
        self.releases.append(key)


def recorder():
    calls = []

    async def _runner(**kwargs):
        calls.append(kwargs)

    return _runner, calls


# ── the shipped sample event ─────────────────────────────────────────


def test_sample_event_matches_the_contract():
    """events/sqs_event.json must stay a valid v1.0 message."""
    event = json.loads((EVENTS / "sqs_event.json").read_text(encoding="utf-8"))
    body = json.loads(event["Records"][0]["body"])
    env, msg = SummaryGenerationMessage.parse(body)
    assert env.job == "litportal.summary_generation"
    assert msg.user_id and msg.session_id and msg.job_id
    assert env.idempotency_key == SummaryGenerationMessage.dedup_key(
        msg.user_id, msg.session_id
    )


# ── happy paths ──────────────────────────────────────────────────────


async def test_valid_message_runs_the_job():
    runner, calls = recorder()
    result = await run(envelope(), runner=runner)
    assert result.status == "succeeded"
    assert calls[0]["session_id"] == "ses_456"
    assert calls[0]["selected_article_ids"] == ["art_1"]


async def test_duplicate_claim_is_suppressed_without_running():
    runner, calls = recorder()
    state = FakeState(already_held=True)
    result = await run(envelope(), state=state, runner=runner)
    assert result.status == "skipped_duplicate"
    assert calls == [], "a duplicate must not re-run the LLM job"


async def test_claim_uses_the_proposal_key_shape():
    runner, _ = recorder()
    state = FakeState()
    await run(envelope(), state=state, runner=runner)
    assert state.claims == ["summary_generation#usr_123#ses_456"]


def test_handler_processes_multiple_records(monkeypatch):
    seen = []
    monkeypatch.setattr("src.main.run", _fake_run(seen))
    assert handler(make_sqs_event(envelope(), envelope(job_id="job-2")), None) == {
        "statusCode": 200
    }
    assert len(seen) == 2


def test_handler_accepts_empty_batch():
    assert handler({"Records": []}, None) == {"statusCode": 200}


# ── failure propagation (what makes the DLQ work) ────────────────────


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        handler(make_sqs_event("not-json"), None)


async def test_transient_failure_propagates_and_releases_the_claim():
    state = FakeState()

    async def boom(**_kwargs):
        raise RuntimeError("LLM provider timed out")

    with pytest.raises(RuntimeError):
        await run(envelope(), state=state, runner=boom)
    assert state.releases == ["summary_generation#usr_123#ses_456"], (
        "a failed run must release its dedup claim, or the retry is blocked "
        "by our own guard"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": "2.0"},
        {"job": "litportal.something_else"},
        {"correlation_id": ""},
    ],
)
async def test_bad_envelope_is_permanent_input_error(mutation):
    body = envelope()
    body.update(mutation)
    with pytest.raises(PermanentInputError):
        await run(body)


@pytest.mark.parametrize("missing", ["job_id", "user_id", "session_id"])
async def test_missing_required_payload_field_is_permanent(missing):
    body = envelope()
    body["data"].pop(missing)
    with pytest.raises(PermanentInputError):
        await run(body)


async def test_malformed_selected_article_ids_is_permanent():
    with pytest.raises(PermanentInputError):
        await run(envelope(selected_article_ids=["ok", ""]))


def _fake_run(sink):
    from jobs.summary_generation import RunResult

    async def _run(body):
        sink.append(body)
        return RunResult(status="succeeded", job_id="job-x")

    return _run
