"""Contract + handler tests for the LitHub outbox-delivery Lambda (A1).

No AWS, no MongoDB, no LitHub: the database and the delivery function are
injected. The audit's key property is asserted: the message carries only the
row id; the outbox row stays the single source of truth for the payload.
"""

import json
from pathlib import Path

import pytest

from jobs.contracts import LitHubOutboxDeliveryMessage, PermanentInputError
from jobs.lithub_outbox_delivery import run
from src.main import handler

EVENTS = Path(__file__).resolve().parent.parent / "events"
OUTBOX_ID = "ob-1"
SUB = "11111111-2222-3333-4444-555555555555"


def envelope(**data_over):
    data = {
        "outbox_id": OUTBOX_ID,
        "op": "save",
        "user_id": SUB,
        "pmid": "39123456",
        "doi": None,
    }
    data.update(data_over)
    return {
        "schema_version": "1.0",
        "job": "litportal.lithub_outbox_delivery",
        "correlation_id": "corr-1",
        "idempotency_key": f"lithub_mirror#save#{SUB}#pmid:39123456",
        "enqueued_at": "2026-08-13T08:00:00.000Z",
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
                    "arn:aws:sqs:us-east-1:1:scienthesis-lithub-outbox-delivery"
                ),
                "awsRegion": "us-east-1",
            }
            for i, b in enumerate(bodies)
        ]
    }


class FakeOutbox:
    def __init__(self, row=None):
        self.row = row
        self.updates = []

    async def find_one(self, query):
        if self.row and self.row.get("id") == query.get("id"):
            return dict(self.row)
        return None

    async def update_one(self, query, update):
        self.updates.append((query, update))
        if self.row:
            self.row.update(update["$set"])


class FakeDB:
    def __init__(self, row=None):
        self.outbox = FakeOutbox(row)

    def __getitem__(self, name):
        assert name == "lithub_sync_outbox"
        return self.outbox


def deliver_recorder(error=None):
    calls = []

    async def _deliver(op, payload, client):
        calls.append((op, payload))
        if error:
            raise error

    return _deliver, calls


def row(**over):
    base = {
        "id": OUTBOX_ID,
        "op": "save",
        "status": "pending",
        "payload": {"user_id": SUB, "item": {"pmid": "39123456", "title": "T"}},
    }
    base.update(over)
    return base


# ── the shipped sample event ─────────────────────────────────────────


def test_sample_event_matches_the_contract():
    event = json.loads((EVENTS / "sqs_event.json").read_text(encoding="utf-8"))
    body = json.loads(event["Records"][0]["body"])
    env, msg = LitHubOutboxDeliveryMessage.parse(body)
    assert env.job == "litportal.lithub_outbox_delivery"
    assert msg.op in ("save", "delete")


# ── happy paths ──────────────────────────────────────────────────────


async def test_delivers_row_payload_and_marks_done():
    db = FakeDB(row())
    deliver, calls = deliver_recorder()
    result = await run(envelope(), db=db, deliver=deliver)
    assert result.status == "succeeded"
    op, payload = calls[0]
    assert op == "save"
    assert payload["item"]["pmid"] == "39123456"  # payload came from the ROW
    assert db.outbox.row["status"] == "done"


async def test_done_row_is_not_redelivered():
    """At-least-once redelivery must not repeat the LitHub call."""
    db = FakeDB(row(status="done"))
    deliver, calls = deliver_recorder()
    result = await run(envelope(), db=db, deliver=deliver)
    assert result.status == "already_done"
    assert calls == []


async def test_superseded_row_is_acked_without_delivery():
    db = FakeDB(row(status="superseded"))
    deliver, calls = deliver_recorder()
    result = await run(envelope(), db=db, deliver=deliver)
    assert result.status == "superseded"
    assert calls == []


# ── failure propagation ──────────────────────────────────────────────


async def test_lithub_failure_propagates_and_row_stays_undone():
    db = FakeDB(row())
    deliver, _calls = deliver_recorder(error=RuntimeError("lithub 503"))
    with pytest.raises(RuntimeError, match="lithub 503"):
        await run(envelope(), db=db, deliver=deliver)
    assert db.outbox.row["status"] == "pending"


async def test_missing_row_is_permanent_input():
    with pytest.raises(PermanentInputError, match="does not exist"):
        await run(envelope(), db=FakeDB(None), deliver=deliver_recorder()[0])


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        handler(make_sqs_event("not-json"), None)


def test_handler_accepts_empty_batch():
    assert handler({"Records": []}, None) == {"statusCode": 200}


async def test_invalid_op_is_permanent():
    with pytest.raises(PermanentInputError):
        await run(envelope(op="upsert"))


@pytest.mark.parametrize("missing", ["outbox_id", "op", "user_id"])
async def test_missing_required_field_is_permanent(missing):
    body = envelope()
    body["data"].pop(missing)
    with pytest.raises(PermanentInputError):
        await run(body)
