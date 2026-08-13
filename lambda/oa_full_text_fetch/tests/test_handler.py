"""Contract + handler tests for the OA full-text-fetch Lambda (A2).

No AWS, no database, no network: the fetcher is injected. The load-bearing
assertion: the service "never raises" by contract and the inline caller
swallows — so the WORKER must convert a failed status into an exception, or
the DLQ stays permanently empty.
"""

import json
from pathlib import Path

import pytest

from jobs.contracts import OAFullTextFetchMessage, PermanentInputError
from jobs.oa_full_text_fetch import run
from src.main import handler

EVENTS = Path(__file__).resolve().parent.parent / "events"


def envelope(**data_over):
    data = {
        "article_id": "art_789",
        "session_id": "ses_456",
        "user_id": "usr_123",
        "force_refresh": False,
    }
    data.update(data_over)
    return {
        "schema_version": "1.0",
        "job": "litportal.oa_full_text_fetch",
        "correlation_id": "corr-1",
        "idempotency_key": "oa_fetch#art_789#ses_456",
        "enqueued_at": "2026-08-13T08:05:00.000Z",
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
                "eventSourceARN": (
                    "arn:aws:sqs:us-east-1:1:scienthesis-oa-full-text-fetch"
                ),
                "awsRegion": "us-east-1",
            }
            for i, b in enumerate(bodies)
        ]
    }


class FakeFetcher:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def fetch_for_article(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


# ── the shipped sample event ─────────────────────────────────────────


def test_sample_event_matches_the_contract():
    event = json.loads((EVENTS / "sqs_event.json").read_text(encoding="utf-8"))
    body = json.loads(event["Records"][0]["body"])
    env, msg = OAFullTextFetchMessage.parse(body)
    assert env.job == "litportal.oa_full_text_fetch"
    assert env.idempotency_key == OAFullTextFetchMessage.dedup_key(
        msg.article_id, msg.session_id
    )


# ── legitimate terminal outcomes ─────────────────────────────────────


@pytest.mark.parametrize(
    "oa_status", ["fetched", "cache_hit", "skipped_license", "skipped_ineligible"]
)
async def test_ok_statuses_succeed(oa_status):
    fetcher = FakeFetcher({"status": oa_status, "oa_sections_created": 3})
    result = await run(envelope(), fetcher=fetcher)
    assert result.status == "succeeded"
    assert result.oa_status == oa_status
    call = fetcher.calls[0]
    assert call["article_id"] == "art_789"
    assert call["session_id"] == "ses_456"
    assert call["user_id"] == "usr_123"


async def test_force_refresh_is_forwarded():
    fetcher = FakeFetcher({"status": "fetched"})
    await run(envelope(force_refresh=True), fetcher=fetcher)
    assert fetcher.calls[0]["force_refresh"] is True


# ── failure propagation (the whole point of this worker) ─────────────


async def test_failed_status_raises_so_the_dlq_works():
    fetcher = FakeFetcher(
        {"status": "failed", "oa_fetch_error_code": "upstream_timeout"}
    )
    with pytest.raises(RuntimeError, match="upstream_timeout"):
        await run(envelope(), fetcher=fetcher)


async def test_unknown_status_is_treated_as_failure():
    fetcher = FakeFetcher({"status": "mystery"})
    with pytest.raises(RuntimeError):
        await run(envelope(), fetcher=fetcher)


async def test_fetcher_exception_propagates():
    class Boom(FakeFetcher):
        async def fetch_for_article(self, **kwargs):
            raise TimeoutError("europepmc timed out")

    with pytest.raises(TimeoutError):
        await run(envelope(), fetcher=Boom(None))


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        handler(make_sqs_event("not-json"), None)


def test_handler_accepts_empty_batch():
    assert handler({"Records": []}, None) == {"statusCode": 200}


@pytest.mark.parametrize("missing", ["article_id", "session_id", "user_id"])
async def test_missing_required_field_is_permanent(missing):
    body = envelope()
    body["data"].pop(missing)
    with pytest.raises(PermanentInputError):
        await run(body)
