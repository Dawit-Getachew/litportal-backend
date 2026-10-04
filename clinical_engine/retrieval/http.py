"""Outbound HTTP for evidence sources: one rate limiter per host for the whole
process, bounded retries that honour Retry-After, and failures reported as
failures — never as "no results".

NCBI allows 3 requests/s per IP without an API key and 10 with one, counted
across every E-utilities call (MeSH lookups, counts and searches share it).
"""
from __future__ import annotations

import asyncio
import email.utils
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

# Minimum spacing between requests per host (seconds).
_INTERVALS = {
    "ncbi_key": 0.11,
    "ncbi": 0.35,
    "europepmc": 0.1,
    "unpaywall": 0.1,
    "pmc_cloud": 0.05,
    "icite": 0.1,
}
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class SourceError(Exception):
    """A source call that failed after retries. ``code`` is closed-set:
    timeout | http_<status> | transport_error | invalid_response."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail[:200]


@dataclass
class _Slot:
    lock: asyncio.Lock
    next_at: float = 0.0


_slots: dict[tuple[int, str], _Slot] = {}

# Tests route every source call through a mock transport (never the network).
TEST_TRANSPORT: Optional[httpx.AsyncBaseTransport] = None


def transport_or_default(transport: Optional[httpx.AsyncBaseTransport]) -> Optional[httpx.AsyncBaseTransport]:
    return transport if transport is not None else TEST_TRANSPORT


def ncbi_host() -> str:
    return "ncbi_key" if os.environ.get("NCBI_API_KEY", "").strip() else "ncbi"


def ncbi_params(**extra: Any) -> dict[str, Any]:
    p: dict[str, Any] = {"tool": os.environ.get("NCBI_TOOL", "LitPortal").strip() or "LitPortal"}
    if os.environ.get("NCBI_EMAIL", "").strip():
        p["email"] = os.environ["NCBI_EMAIL"].strip()
    if os.environ.get("NCBI_API_KEY", "").strip():
        p["api_key"] = os.environ["NCBI_API_KEY"].strip()
    p.update(extra)
    return p


async def wait_turn(host: str) -> None:
    """Space requests to *host*. One slot per event loop (locks are loop-bound)."""
    key = (id(asyncio.get_running_loop()), host)
    slot = _slots.get(key)
    if slot is None:
        for k in [k for k in _slots if k[0] != key[0]]:
            _slots.pop(k, None)
        slot = _slots[key] = _Slot(asyncio.Lock())
    async with slot.lock:
        wait = slot.next_at - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        slot.next_at = time.monotonic() + _INTERVALS.get(host, 0.2)


def _retry_after(resp: httpx.Response) -> Optional[float]:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        when = email.utils.parsedate_to_datetime(raw)
        if when is None:
            return None
        return max(0.0, when.timestamp() - time.time())


def _scrub(text: str) -> str:
    key = os.environ.get("NCBI_API_KEY", "").strip()
    return text.replace(key, "<redacted>") if key else text


async def request(
    client: httpx.AsyncClient, method: str, url: str, *, host: str,
    params: Optional[dict[str, Any]] = None, data: Optional[dict[str, Any]] = None,
    max_attempts: int = 4, max_retry_wait_s: float = 8.0,
) -> httpx.Response:
    """Send with the host's rate limit; retry 429/5xx/timeouts with backoff
    (Retry-After wins when present). Raises :class:`SourceError` when exhausted."""
    last: Optional[SourceError] = None
    for attempt in range(max_attempts):
        await wait_turn(host)
        try:
            resp = await client.request(method, url, params=params, data=data)
        except httpx.TimeoutException:
            last = SourceError("timeout")
        except httpx.HTTPError as exc:
            last = SourceError("transport_error", _scrub(type(exc).__name__))
        else:
            if resp.status_code < 400:
                return resp
            last = SourceError(f"http_{resp.status_code}", _scrub(resp.text[:200]))
            if resp.status_code not in _RETRY_STATUSES:
                raise last
            hinted = _retry_after(resp)
            if hinted is not None and attempt < max_attempts - 1:
                await asyncio.sleep(min(max_retry_wait_s, hinted))
                continue
        if attempt < max_attempts - 1:
            await asyncio.sleep(min(max_retry_wait_s, 0.5 * 2 ** attempt + random.uniform(0, 0.25)))
    raise last or SourceError("transport_error")
