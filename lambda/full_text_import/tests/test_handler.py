"""Contract + handler tests for the full-text-import Lambda.

No AWS, no database, no network: the broker and the OA fetcher are injected.
The headline assertion is that the worker brokers exactly ONE article — the
inline path brokered every article in the caller's session (up to 500), which
is the difference between ~26s and ~11,800s per message.
"""

import json
from pathlib import Path

import pytest

from jobs.contracts import PermanentInputError
from jobs.full_text_import import FullTextImportMessage, run
from src.main import handler

EVENTS = Path(__file__).resolve().parent.parent / "events"


# ── helpers ──────────────────────────────────────────────────────────


def envelope(**data_over):
    data = {
        "user_id": "usr_123",
        "session_id": "ses_import_456",
        "article_id": "art_789",
        "canonical_key": "pmid:39123456",
        "force_refresh": False,
        "save": False,
    }
    data.update(data_over)
    return {
        "schema_version": "1.0",
        "job": "litportal.full_text_import.ingest",
        "correlation_id": "corr-1",
        "idempotency_key": "full_text_import#usr_123#pmid:39123456",
        "enqueued_at": "2026-08-09T13:15:00.000Z",
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
                "eventSourceARN": ("arn:aws:sqs:us-east-1:1:scienthesis-full-text-import"),
                "awsRegion": "us-east-1",
            }
            for i, b in enumerate(bodies)
        ]
    }


class FakeBroker:
    def __init__(self, new_rows=2):
        self.calls = []
        self.new_rows = new_rows

    async def broker_for_articles(self, article_ids, *, session_id, client=None):
        self.calls.append({"article_ids": list(article_ids), "session_id": session_id})
        return {"new_access_locations": self.new_rows}

    async def broker_for_session(self, session_id, *, client=None):  # pragma: no cover
        raise AssertionError("the worker must never broker a whole session")


class FakeOAFetcher:
    def __init__(self, result=None):
        self.result = result if result is not None else {"status": "fetched"}
        self.calls = []

    async def fetch_for_article(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


# ── the shipped sample event ─────────────────────────────────────────


def test_sample_event_matches_the_contract():
    event = json.loads((EVENTS / "sqs_event.json").read_text(encoding="utf-8"))
    body = json.loads(event["Records"][0]["body"])
    env, msg = FullTextImportMessage.parse(body)
    assert env.job == "litportal.full_text_import.ingest"
    assert env.idempotency_key == FullTextImportMessage.dedup_key(
        msg.user_id, msg.canonical_key, msg.article_id
    )


# ── happy path ───────────────────────────────────────────────────────


async def test_brokers_exactly_one_article():
    """The blocking prerequisite from the contract."""
    broker, fetcher = FakeBroker(), FakeOAFetcher()
    result = await run(envelope(), broker=broker, oa_fetcher=fetcher)
    assert result.status == "succeeded"
    assert broker.calls == [
        {"article_ids": ["art_789"], "session_id": "ses_import_456"}
    ]
    assert result.access_locations == 2


async def test_oa_fetch_receives_the_message_scope():
    broker, fetcher = FakeBroker(), FakeOAFetcher()
    await run(envelope(force_refresh=True), broker=broker, oa_fetcher=fetcher)
    call = fetcher.calls[0]
    assert call["article_id"] == "art_789"
    assert call["session_id"] == "ses_import_456"
    assert call["user_id"] == "usr_123"
    assert call["force_refresh"] is True


async def test_skipped_oa_is_still_a_success():
    """License-gated skips are a normal outcome, not a failure."""
    broker = FakeBroker()
    fetcher = FakeOAFetcher({"status": "skipped_license"})
    result = await run(envelope(), broker=broker, oa_fetcher=fetcher)
    assert result.status == "succeeded"
    assert result.oa_status == "skipped_license"


# ── failure propagation ──────────────────────────────────────────────


async def test_failed_oa_fetch_raises_so_the_dlq_works():
    """fetch_for_article never raises by contract, and the inline caller
    swallows — so a bare worker would ACK failures forever."""
    broker = FakeBroker()
    fetcher = FakeOAFetcher({"status": "failed", "error_code": "upstream_timeout"})
    with pytest.raises(RuntimeError, match="upstream_timeout"):
        await run(envelope(), broker=broker, oa_fetcher=fetcher)


async def test_broker_exception_propagates():
    class Boom(FakeBroker):
        async def broker_for_articles(self, *args, **kwargs):
            raise TimeoutError("unpaywall timed out")

    with pytest.raises(TimeoutError):
        await run(envelope(), broker=Boom(), oa_fetcher=FakeOAFetcher())


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        handler(make_sqs_event("not-json"), None)


def test_handler_accepts_empty_batch():
    assert handler({"Records": []}, None) == {"statusCode": 200}


@pytest.mark.parametrize(
    "mutation", [{"schema_version": "3.0"}, {"job": "litportal.other"}]
)
async def test_bad_envelope_is_permanent(mutation):
    body = envelope()
    body.update(mutation)
    with pytest.raises(PermanentInputError):
        await run(body)


@pytest.mark.parametrize("missing", ["user_id", "session_id", "article_id"])
async def test_missing_required_field_is_permanent(missing):
    body = envelope()
    body["data"].pop(missing)
    with pytest.raises(PermanentInputError):
        await run(body)
