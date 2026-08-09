"""AWS-neutral entry point for ``litportal.summary_generation`` (Phase 1).

This is the seam between the queue and the existing business logic. The
migration moves WHERE the code runs, not what it does: ``run`` validates the
contract, applies the Redis dedup guard, and then calls the SAME
``_run_generate`` coroutine the in-process job runner calls today.

Nothing here imports boto3 or Lambda types, so it can be driven from a test,
a local script, the FastAPI process, or a Lambda handler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from jobs.contracts import SummaryGenerationMessage

logger = logging.getLogger("litportal.jobs.summary_generation")


class JobStateStore(Protocol):
    """Minimal job-state surface the worker needs (Redis in AWS, a dict in tests)."""

    async def claim(self, key: str, ttl_seconds: int) -> bool:
        """Atomic SET NX EX. True if this worker won the claim."""

    async def release(self, key: str) -> None: ...


@dataclass
class RunResult:
    status: str  # succeeded | skipped_duplicate | failed
    job_id: str
    detail: str = ""


#: TTL for the dedup claim = 6x the recommended function timeout (900s), so a
#: crashed worker's key expires and the message can legitimately be retried.
CLAIM_TTL_SECONDS = 5400


async def run(
    body: dict[str, Any],
    *,
    state: JobStateStore | None = None,
    runner: Any | None = None,
) -> RunResult:
    """Execute one summary-generation message.

    ``state`` is the Redis-backed dedup store (optional so the job can be run
    locally without Redis). ``runner`` is injectable purely for tests; in
    production it defaults to the application's existing generate coroutine.

    Raises on transient failures so SQS retries and the DLQ works. Contract
    violations raise ``PermanentInputError`` (never retryable).
    """
    envelope, message = SummaryGenerationMessage.parse(body)
    log_ctx = {
        "correlation_id": envelope.correlation_id,
        "job_id": message.job_id,
        "session_id": message.session_id,
    }
    logger.info("summary_generation_started", extra=log_ctx)

    claim_key = SummaryGenerationMessage.dedup_key(
        message.user_id, message.session_id
    )
    if state is not None:
        won = await state.claim(claim_key, CLAIM_TTL_SECONDS)
        if not won:
            # Another worker already owns this (user, session). Returning
            # normally ACKs the duplicate instead of re-running an LLM job.
            logger.info("summary_generation_duplicate_suppressed", extra=log_ctx)
            return RunResult(
                status="skipped_duplicate",
                job_id=message.job_id,
                detail=f"claim held: {claim_key}",
            )

    try:
        if runner is None:
            runner = _default_runner()
        await runner(
            job_id=message.job_id,
            user_id=message.user_id,
            session_id=message.session_id,
            selected_article_ids=message.selected_article_ids,
            acknowledge_abstract_only_limitations=(
                message.acknowledge_abstract_only_limitations
            ),
        )
    except Exception:
        # Release the claim so a retry is not blocked by our own guard, then
        # let the exception propagate — the handler must NOT swallow it.
        if state is not None:
            await state.release(claim_key)
        logger.exception("summary_generation_failed", extra=log_ctx)
        raise

    logger.info("summary_generation_succeeded", extra=log_ctx)
    return RunResult(status="succeeded", job_id=message.job_id)


def _default_runner():
    """Resolve the in-app coroutine lazily so importing this module (e.g. in a
    Lambda that only validates contracts) does not drag in the whole app."""
    from routers.summary import run_summary_job_for_worker

    return run_summary_job_for_worker
