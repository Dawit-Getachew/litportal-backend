"""ECS Fargate worker for ``litportal.clinician_search_pipeline`` (job 2).

Runtime shape per the approved proposal: SQS → EventBridge Pipe → ECS RunTask.
The Pipe hands the SQS batch to the task; this entrypoint accepts it either as
the ``PIPE_EVENT`` environment variable or as ``argv[1]`` (both JSON), in any
of the shapes Pipes produce: a bare envelope object, a list of envelopes, or a
list of SQS record objects whose ``body`` is the envelope JSON.

Two contract rules this worker enforces:

* ``sanitized_question`` rides on the session document, never in the message
  (contracts §3.6) — the worker reads it here.
* ``run_pipeline`` is documented "Never raises" and records failures on the
  session instead. Under a swallow-everything worker, retries would never fire
  and the DLQ would stay empty — so this entrypoint reads the FINAL session
  status and exits non-zero on ``error``, which is what makes ECS retries and
  alarms work. This is the "gate strict behavior to the worker path" answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

from jobs.contracts import ClinicianSearchMessage, PermanentInputError

logger = logging.getLogger("litportal.ecs.clinician_search")

#: Session statuses that are legitimate terminal outcomes for the worker.
_OK_TERMINAL = {"awaiting_selection", "complete", "completed", "awaiting_clinical_subject"}


def extract_bodies(event_input: Any) -> list[dict[str, Any]]:
    """Normalize every Pipe delivery shape into a list of envelope dicts."""
    if isinstance(event_input, str):
        event_input = json.loads(event_input)
    if isinstance(event_input, dict):
        candidates = [event_input]
    elif isinstance(event_input, list):
        candidates = event_input
    else:
        raise PermanentInputError(
            f"unsupported event shape: {type(event_input).__name__}"
        )
    bodies: list[dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, dict) and "body" in item and "job" not in item:
            raw = item["body"]
            bodies.append(json.loads(raw) if isinstance(raw, str) else raw)
        else:
            bodies.append(item)
    return bodies


@dataclass
class RunResult:
    status: str  # final session status
    search_session_id: str


async def execute(
    body: dict[str, Any],
    *,
    db: Any = None,
    pipeline: Any = None,
) -> RunResult:
    """Run the full pipeline for one session and surface its final status."""
    envelope, message = ClinicianSearchMessage.parse(body)
    log_ctx = {
        "correlation_id": envelope.correlation_id,
        "search_session_id": message.search_session_id,
    }
    logger.info("clinician_search_started", extra=log_ctx)

    if db is None:
        db = _default_db()

    session = await db["search_sessions"].find_one(
        {"id": message.search_session_id}
    )
    if session is None:
        raise PermanentInputError(
            f"search session {message.search_session_id} does not exist"
        )
    if str(session.get("user_id")) != message.user_id:
        raise PermanentInputError(
            "message user_id does not own this session"
        )
    if session.get("status") in _OK_TERMINAL:
        # Redelivery of an already-finished session: ACK, do not re-run a
        # 30-minute pipeline (no unique index protects against duplicates —
        # observed as DuplicateKeyError crashes in prod on Jul 13).
        logger.info("clinician_search_already_terminal", extra=log_ctx)
        return RunResult(
            status=str(session.get("status")),
            search_session_id=message.search_session_id,
        )

    if pipeline is None:
        pipeline = _default_pipeline()
    await pipeline(
        message.search_session_id,
        message.user_id,
        str(session.get("sanitized_question") or ""),
    )

    final = await db["search_sessions"].find_one(
        {"id": message.search_session_id}, {"_id": 0, "status": 1, "error_message": 1}
    )
    final_status = str((final or {}).get("status") or "unknown")
    if final_status == "error" or final_status == "unknown":
        raise RuntimeError(
            f"pipeline finished with status={final_status!r}: "
            f"{str((final or {}).get('error_message'))[:200]}"
        )
    logger.info("clinician_search_finished status=%s", final_status, extra=log_ctx)
    return RunResult(status=final_status, search_session_id=message.search_session_id)


def _default_db() -> Any:
    from core.db import get_db

    return get_db()


def _default_pipeline() -> Any:
    from services.pipeline import run_pipeline

    return run_pipeline


def main() -> int:
    raw = os.environ.get("PIPE_EVENT") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not raw:
        print("no event: set PIPE_EVENT or pass the JSON as argv[1]", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO)
    bodies = extract_bodies(raw)
    for body in bodies:
        result = asyncio.run(execute(body))
        print(f"session {result.search_session_id} -> {result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
