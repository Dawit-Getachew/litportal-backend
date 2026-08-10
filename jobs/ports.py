"""Store-facing ports for SOW1 job workers.

Protocols only — no Motor, no boto3, no FastAPI. A worker that depends on
these can be unit-tested with plain dicts and can have its datastore swapped
without touching job logic. See ``REPOSITORY_BOUNDARIES_AND_ADAPTERS.md`` for
the adoption plan and the DocumentDB compatibility risks these concentrate.

Stage 2 of that plan: implementations live beside existing code and wrap the
current calls verbatim; no existing call site is forced to change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class JobStateStore(Protocol):
    """Job lifecycle + dedup claim.

    Backed by Redis in AWS (proposal), by the Mongo ``jobs`` collection today,
    and by a dict in tests. ``claim`` is an atomic SET-NX-EX: True means this
    worker won and must do the work.
    """

    async def claim(self, key: str, ttl_seconds: int) -> bool: ...

    async def release(self, key: str) -> None: ...

    async def mark_running(self, job_id: str, *, stage: str, percent: int) -> None: ...

    async def succeed(self, job_id: str, *, result_ref: dict[str, Any]) -> None: ...

    async def fail(
        self, job_id: str, *, error_code: str, message_safe: str
    ) -> None: ...

    async def get(self, job_id: str) -> dict[str, Any] | None: ...


class SessionRepository(Protocol):
    """Search sessions — the unit of work for the search/summary/import jobs."""

    async def get_owned(
        self, session_id: str, user_id: str
    ) -> dict[str, Any] | None: ...

    async def set_status(self, session_id: str, status: str) -> None: ...

    async def selected_article_ids(self, session_id: str) -> list[str]: ...

    async def set_user_included(
        self, session_id: str, article_ids: list[str]
    ) -> None: ...


class ArticleRepository(Protocol):
    """Canonical article records and their derived text extracts."""

    async def get_many(self, article_ids: list[str]) -> list[dict[str, Any]]: ...

    async def get_by_pmid(self, pmid: str) -> dict[str, Any] | None: ...

    async def upsert_summary(
        self, article_id: str, *, summary: str, status: str
    ) -> None: ...

    async def extracts_for(
        self, article_id: str, *, user_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    async def put_extract(
        self, article_id: str, extract: dict[str, Any]
    ) -> None: ...


class OutboxRepository(Protocol):
    """The LitHub mirror outbox.

    Both apps already implement this shape; the port exists so the delivery
    worker can move to SQS (register §1a, job A1) without rewriting callers.
    """

    async def enqueue(
        self, op: str, user_id: str, payload: dict[str, Any]
    ) -> None: ...

    async def claim_due(self) -> dict[str, Any] | None: ...

    async def mark_done(self, row_id: str) -> None: ...

    async def mark_retry(
        self, row_id: str, *, error: str, next_attempt_at: datetime
    ) -> None: ...

    async def mark_dead(self, row_id: str, *, error: str) -> None: ...
