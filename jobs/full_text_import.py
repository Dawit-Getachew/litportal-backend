"""AWS-neutral entry point for ``litportal.full_text_import.ingest`` (Phase 3).

The producer half (resolve + attach + enrich) stays synchronous in the API —
it returns candidates the clinician picks from, so it cannot be queued. This
worker runs the expensive half: access brokering plus license-gated OA
full-text ingestion for exactly ONE article.

BatchSize MUST be 1: two messages at the ~439s worst case already exceed the
900s ceiling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from jobs.contracts import (
    Envelope,
    PermanentInputError,
    _optional,
    _require,
    parse_envelope,
)

logger = logging.getLogger("litportal.jobs.full_text_import")


@dataclass(frozen=True)
class FullTextImportMessage:
    """``litportal.full_text_import.ingest`` payload — contracts doc §3.5."""

    JOB = "litportal.full_text_import.ingest"

    user_id: str
    session_id: str
    article_id: str
    canonical_key: str | None = None
    force_refresh: bool = False
    save: bool = False

    @classmethod
    def parse(cls, body: dict[str, Any]) -> tuple[Envelope, FullTextImportMessage]:
        envelope = parse_envelope(body, expected_job=cls.JOB)
        data = envelope.data
        return envelope, cls(
            user_id=_require(data, "user_id", str),
            session_id=_require(data, "session_id", str),
            article_id=_require(data, "article_id", str),
            canonical_key=_optional(data, "canonical_key", str, None),
            force_refresh=_optional(data, "force_refresh", bool, False),
            save=_optional(data, "save", bool, False),
        )

    @staticmethod
    def dedup_key(user_id: str, canonical_key: str | None, article_id: str) -> str:
        return f"full_text_import#{user_id}#{canonical_key or article_id}"


@dataclass
class RunResult:
    status: str  # succeeded
    article_id: str
    access_locations: int = 0
    oa_status: str | None = None


async def run(
    body: dict[str, Any],
    *,
    broker: Any = None,
    oa_fetcher: Any = None,
) -> RunResult:
    """Broker access and ingest OA full text for one article.

    Transient failures raise so SQS retries. Unlike the inline path, neither
    stage's exception is swallowed here — swallowing them would ACK a message
    whose work failed and leave the DLQ permanently empty.
    """
    envelope, message = FullTextImportMessage.parse(body)
    log_ctx = {
        "correlation_id": envelope.correlation_id,
        "article_id": message.article_id,
        "session_id": message.session_id,
    }
    logger.info("full_text_import_started", extra=log_ctx)

    if broker is None:
        broker = _default_broker()
    if oa_fetcher is None:
        oa_fetcher = _default_oa_fetcher()

    # Exactly this article — never the whole session (see broker_for_articles).
    broker_summary = await broker.broker_for_articles(
        [message.article_id], session_id=message.session_id
    )

    oa_result = await oa_fetcher.fetch_for_article(
        article_id=message.article_id,
        session_id=message.session_id,
        user_id=message.user_id,
        force_refresh=message.force_refresh,
    )
    oa_status = (oa_result or {}).get("status")
    if oa_status == "failed":
        # fetch_for_article never raises by contract; a worker must surface the
        # failure or the DLQ never sees it.
        raise RuntimeError(
            f"OA fetch failed for article {message.article_id}: "
            f"{(oa_result or {}).get('error_code') or 'unknown'}"
        )

    logger.info("full_text_import_succeeded", extra=log_ctx)
    return RunResult(
        status="succeeded",
        article_id=message.article_id,
        access_locations=int(
            (broker_summary or {}).get("new_access_locations", 0) or 0
        ),
        oa_status=oa_status,
    )


def _default_broker() -> Any:
    from services.full_text_access_broker import full_text_access_broker

    return full_text_access_broker


def _default_oa_fetcher() -> Any:
    from services.oa_fetcher import oa_fetcher

    return oa_fetcher


__all__ = ["FullTextImportMessage", "PermanentInputError", "RunResult", "run"]
