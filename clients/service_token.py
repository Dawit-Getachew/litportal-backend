"""Mint HS256 service tokens for outbound internal-API calls.

LitPortal identifies itself as ``iss="litportal"`` (already an allowed issuer
on both Identity and LitHub). Tokens are short-lived and cached per audience.
Ported from litbridge-backend/src/clients/service_token.py.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import jwt

from core.config import get_settings

logger = logging.getLogger("litportal.clients.service_token")

_ALGORITHM = "HS256"
_DEFAULT_EXPIRES_SECONDS = 300
_SAFETY_MARGIN_SECONDS = 30
_SERVICE_NAME = "litportal"


class _TokenCache:
    def __init__(self) -> None:
        self._tokens: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, audience: str) -> str | None:
        with self._lock:
            entry = self._tokens.get(audience)
            if entry is None:
                return None
            token, expires_at = entry
            if time.time() >= expires_at - _SAFETY_MARGIN_SECONDS:
                self._tokens.pop(audience, None)
                return None
            return token

    def put(self, audience: str, token: str, expires_at: float) -> None:
        with self._lock:
            self._tokens[audience] = (token, expires_at)


_cache = _TokenCache()


def mint_service_token(audience: str, *, expires_seconds: int = _DEFAULT_EXPIRES_SECONDS) -> str:
    """Return a fresh (or cached) HS256 service token for ``audience``."""
    cached = _cache.get(audience)
    if cached is not None:
        return cached

    settings = get_settings()
    secret = getattr(settings, "SERVICE_TOKEN_SECRET", "")
    if not secret:
        raise RuntimeError(
            "SERVICE_TOKEN_SECRET is not configured; LitPortal cannot mint service tokens.",
        )

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=expires_seconds)
    payload = {
        "iss": _SERVICE_NAME,
        "aud": audience,
        "type": "service",
        "iat": now,
        "exp": expires_at,
    }
    token = jwt.encode(payload, secret, algorithm=_ALGORITHM)
    _cache.put(audience, token, expires_at.timestamp())
    return token


def reset_cache_for_tests() -> None:
    _cache._tokens.clear()  # noqa: SLF001
