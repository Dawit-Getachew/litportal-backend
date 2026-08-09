import json
import pytest
from src.main import handler


# ── helpers ───────────────────────────────────────────────────────────────────

def make_sqs_event(*bodies):
    """Build a minimal SQS event with one record per body dict/str."""
    return {
        "Records": [
            {
                "messageId": f"msg-{i}",
                "receiptHandle": f"handle-{i}",
                "body": json.dumps(body) if isinstance(body, dict) else body,
                "attributes": {
                    "ApproximateReceiveCount": "1",
                    "SentTimestamp": "1700000000000",
                    "SenderId": "AIDATEST",
                    "ApproximateFirstReceiveTimestamp": "1700000000001",
                },
                "messageAttributes": {},
                "md5OfBody": "test",
                "eventSource": "aws:sqs",
                "eventSourceARN": "arn:aws:sqs:us-east-1:123456789012:test-queue",
                "awsRegion": "us-east-1",
            }
            for i, body in enumerate(bodies)
        ]
    }


class MockContext:
    function_name = "boilerplate"
    aws_request_id = "test-request-id"
    memory_limit_in_mb = 128


# ── tests ─────────────────────────────────────────────────────────────────────

def test_single_record_returns_200():
    event = make_sqs_event({"message": "Hello World!"})
    result = handler(event, MockContext())
    assert result["statusCode"] == 200


def test_multiple_records_returns_200():
    event = make_sqs_event({"message": "first"}, {"message": "second"})
    result = handler(event, MockContext())
    assert result["statusCode"] == 200


def test_empty_batch_returns_200():
    result = handler({"Records": []}, MockContext())
    assert result["statusCode"] == 200


def test_invalid_json_body_raises():
    """Handler must raise so SQS retries and routes to DLQ on exhaustion."""
    event = make_sqs_event("not valid json {{{")
    with pytest.raises(Exception):
        handler(event, MockContext())
