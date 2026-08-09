"""Message envelope + payload validation for SOW1 queue jobs.

Implements the common envelope from ``SOW1_EVENT_CONTRACTS.md`` v1.0. Kept
dependency-free (stdlib only) so a Lambda package can vendor just this module
without pulling the whole application in.

Failure classes map onto retry behavior:

* ``PermanentInputError`` — the message can never succeed (unknown major
  version, missing/!typed required field). The worker marks the job failed and
  re-raises so the message reaches the DLQ on the first pass, tagged, instead
  of burning the whole ``maxReceiveCount`` budget.
* anything else propagating out of ``run(...)`` is treated as transient by
  SQS and retried until redrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Envelope major version this build implements.
SUPPORTED_MAJOR = 1


class ContractError(Exception):
    """Base class for contract violations."""


class PermanentInputError(ContractError):
    """The message is structurally unusable — never retry it."""


def _require(data: dict[str, Any], key: str, expected: type | tuple[type, ...]):
    if key not in data:
        raise PermanentInputError(f"missing required field: {key!r}")
    value = data[key]
    if value is None or not isinstance(value, expected):
        raise PermanentInputError(
            f"field {key!r} must be {expected} — got {type(value).__name__}"
        )
    if isinstance(value, str) and not value.strip():
        raise PermanentInputError(f"field {key!r} must not be empty")
    return value


def _optional(data: dict[str, Any], key: str, expected: type | tuple[type, ...], default):
    if key not in data or data[key] is None:
        return default
    value = data[key]
    if not isinstance(value, expected):
        raise PermanentInputError(
            f"field {key!r} must be {expected} — got {type(value).__name__}"
        )
    return value


@dataclass(frozen=True)
class Envelope:
    """The common envelope every SOW1 queue message carries."""

    schema_version: str
    job: str
    correlation_id: str
    idempotency_key: str
    enqueued_at: str
    producer: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def major(self) -> int:
        return int(self.schema_version.split(".", 1)[0])


def parse_envelope(body: dict[str, Any], *, expected_job: str | None = None) -> Envelope:
    """Validate the envelope. Raises ``PermanentInputError`` — never retryable."""
    if not isinstance(body, dict):
        raise PermanentInputError(f"message body must be a JSON object, got {type(body).__name__}")

    version = _require(body, "schema_version", str)
    try:
        major = int(version.split(".", 1)[0])
    except ValueError as exc:
        raise PermanentInputError(f"malformed schema_version: {version!r}") from exc
    if major != SUPPORTED_MAJOR:
        raise PermanentInputError(
            f"unsupported schema_version {version!r} — this build implements "
            f"major {SUPPORTED_MAJOR}. Deploy a worker that understands it "
            f"before switching the producer (see the version-bump procedure)."
        )

    job = _require(body, "job", str)
    if expected_job and job != expected_job:
        raise PermanentInputError(
            f"message routed to the wrong worker: job={job!r}, expected {expected_job!r}"
        )

    return Envelope(
        schema_version=version,
        job=job,
        correlation_id=_require(body, "correlation_id", str),
        idempotency_key=_require(body, "idempotency_key", str),
        enqueued_at=_require(body, "enqueued_at", str),
        producer=_require(body, "producer", str),
        data=_optional(body, "data", dict, {}),
    )


@dataclass(frozen=True)
class SummaryGenerationMessage:
    """``litportal.summary_generation`` payload — contracts doc §3.1."""

    JOB = "litportal.summary_generation"

    job_id: str
    user_id: str
    session_id: str
    selected_article_ids: list[str] | None = None
    acknowledge_abstract_only_limitations: bool = False

    @classmethod
    def parse(cls, body: dict[str, Any]) -> tuple[Envelope, SummaryGenerationMessage]:
        envelope = parse_envelope(body, expected_job=cls.JOB)
        data = envelope.data
        selected = _optional(data, "selected_article_ids", list, None)
        if selected is not None:
            bad = [a for a in selected if not isinstance(a, str) or not a.strip()]
            if bad:
                raise PermanentInputError(
                    "selected_article_ids must contain non-empty strings"
                )
        return envelope, cls(
            job_id=_require(data, "job_id", str),
            user_id=_require(data, "user_id", str),
            session_id=_require(data, "session_id", str),
            selected_article_ids=selected,
            acknowledge_abstract_only_limitations=_optional(
                data, "acknowledge_abstract_only_limitations", bool, False
            ),
        )

    @staticmethod
    def dedup_key(user_id: str, session_id: str) -> str:
        """Redis dedup key — the shape the proposal specifies."""
        return f"summary_generation#{user_id}#{session_id}"
