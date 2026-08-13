"""AWS-neutral entry point for ``litportal.oa_full_text_fetch`` (A2).

Per-article OA full-text ingestion. Inline today this runs inside the search
and summary flows and is only safe while ``OA_MAX_FULL_TEXT_FETCHES_PER_SEARCH``
stays ≤ 24 — the value an operator naturally reaches for (200) produces a
105-minute run. One message per article caps the work at ~185s regardless of
any tuning value, which is why the amendment promotes it to its own queue.

The service's own contract is "fetch_for_article never raises" and the inline
caller swallows — so a bare worker would ACK every failure and the DLQ would
stay permanently empty. This module re-raises ``failed`` results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from jobs.contracts import OAFullTextFetchMessage

logger = logging.getLogger("litportal.jobs.oa_full_text_fetch")

#: OA statuses that are legitimate terminal outcomes, not failures.
_OK_STATUSES = {"fetched", "cache_hit", "skipped_license", "skipped_ineligible"}


@dataclass
class RunResult:
    status: str  # succeeded
    article_id: str
    oa_status: str | None = None
    sections_created: int = 0


async def run(
    body: dict[str, Any],
    *,
    fetcher: Any = None,
) -> RunResult:
    """Fetch OA full text for one article. Raises on failure so SQS retries."""
    envelope, message = OAFullTextFetchMessage.parse(body)
    log_ctx = {
        "correlation_id": envelope.correlation_id,
        "article_id": message.article_id,
        "session_id": message.session_id,
    }
    logger.info("oa_full_text_fetch_started", extra=log_ctx)

    if fetcher is None:
        fetcher = _default_fetcher()

    result = await fetcher.fetch_for_article(
        article_id=message.article_id,
        session_id=message.session_id,
        user_id=message.user_id,
        force_refresh=message.force_refresh,
    )
    oa_status = (result or {}).get("status")

    if oa_status not in _OK_STATUSES:
        # "failed" (or anything unknown) must surface — the service never
        # raises by contract, so the worker is the only place retries and the
        # DLQ can come from.
        raise RuntimeError(
            f"OA fetch did not succeed for article {message.article_id}: "
            f"status={oa_status!r} error={(result or {}).get('oa_fetch_error_code') or (result or {}).get('error_code')!r}"
        )

    logger.info("oa_full_text_fetch_succeeded", extra=log_ctx)
    return RunResult(
        status="succeeded",
        article_id=message.article_id,
        oa_status=oa_status,
        sections_created=int((result or {}).get("oa_sections_created", 0) or 0),
    )


def _default_fetcher() -> Any:
    from services.oa_fetcher import oa_fetcher

    return oa_fetcher
