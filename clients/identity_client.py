"""HTTP client + JWKS validator for the Scienthesis Identity Service.

Ported from litbridge-backend/src/clients/identity_client.py and adapted to
this backend (stdlib logging, core.config settings). The per-request token
path is pure local crypto: download Identity's JWKS once, cache it, and verify
the RS256 signature/issuer/audience in-process (no HTTP call per request).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import UUID

import httpx
import jwt
from jwt import algorithms

from core.config import Settings, get_settings
from clients.service_token import mint_service_token

logger = logging.getLogger("litportal.clients.identity")

# When a JWKS fetch fails and we hold no usable keys, don't re-hit the endpoint
# (serialized behind the 8s timeout) on every auth request for this long.
_NEG_CACHE_SECONDS = 5.0


class _JWKSCache:
    """Async-safe in-process JWKS cache."""

    def __init__(self) -> None:
        self._keys_by_kid: dict[str, Any] = {}
        self._fetched_at: float = 0.0
        self._failed_at: float = 0.0
        self._lock = asyncio.Lock()

    def _fresh(self, ttl: int) -> bool:
        return bool(self._keys_by_kid) and (time.time() - self._fetched_at) < ttl

    async def fetch(self, http_client: httpx.AsyncClient, settings: Settings) -> dict[str, Any]:
        if self._fresh(settings.JWKS_CACHE_TTL_SECONDS):
            return self._keys_by_kid
        async with self._lock:
            if self._fresh(settings.JWKS_CACHE_TTL_SECONDS):
                return self._keys_by_kid
            # Negative cache: a recent failure with no usable keys → back off
            # instead of serializing every caller behind a fresh 8s fetch.
            if not self._keys_by_kid and (time.time() - self._failed_at) < _NEG_CACHE_SECONDS:
                return self._keys_by_kid
            url = settings.IDENTITY_JWKS_URL or (
                f"{settings.IDENTITY_BASE_URL.rstrip('/')}/.well-known/jwks.json"
            )
            try:
                resp = await http_client.get(url, timeout=8.0)
                resp.raise_for_status()
                body = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                self._failed_at = time.time()
                logger.error("identity_jwks_fetch_failed url=%s error=%s", url, exc)
                return self._keys_by_kid
            self._keys_by_kid = {
                k["kid"]: k for k in body.get("keys", []) if "kid" in k
            }
            self._fetched_at = time.time()
            return self._keys_by_kid

    def clear(self) -> None:
        # Drop cached keys but KEEP _failed_at so the negative cache still
        # throttles the forced refetch path during an outage.
        self._keys_by_kid = {}
        self._fetched_at = 0.0

    def reset(self) -> None:
        self._keys_by_kid = {}
        self._fetched_at = 0.0
        self._failed_at = 0.0


_jwks_cache = _JWKSCache()


def reset_jwks_cache_for_tests() -> None:
    _jwks_cache.reset()


async def validate_identity_access_token(
    token: str,
    *,
    http_client: httpx.AsyncClient,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    """Verify an Identity-issued RS256 token. Returns ``None`` when the token
    is not Identity-shaped (caller can fall back). Raises ``jwt.InvalidTokenError``
    for tokens that ARE Identity-shaped but invalid (expired, bad signature)."""
    settings = settings or get_settings()
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError:
        return None
    if header.get("alg") != "RS256":
        return None
    kid = header.get("kid")
    if not kid:
        return None

    keys = await _jwks_cache.fetch(http_client, settings)
    key_dict = keys.get(kid)
    if key_dict is None:
        _jwks_cache.clear()
        keys = await _jwks_cache.fetch(http_client, settings)
        key_dict = keys.get(kid)
        if key_dict is None:
            if not keys:
                # No keys at all → JWKS unreachable/empty. This token IS
                # Identity-shaped (RS256 + kid), so fail loud rather than
                # silently downgrade it to "not Identity" (a misleading 401).
                raise IdentityUpstreamError(
                    "Identity JWKS is unreachable or empty; cannot verify token.",
                )
            # Keys present but this kid is unknown → not an Identity token we
            # can verify (foreign/forged kid) → let the caller fall through.
            return None

    public_key = algorithms.RSAAlgorithm.from_jwk(key_dict)
    payload = jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        audience=settings.IDENTITY_JWT_AUDIENCE,
        issuer=settings.IDENTITY_JWT_ISSUER,
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("Identity token type is not 'access'.")
    return payload


# ── HTTP client ────────────────────────────────────────────────────


class IdentityClientError(Exception):
    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Identity {status_code}: {detail}")


class IdentityUpstreamError(Exception):
    """Identity unreachable or 5xx."""


class IdentityClient:
    """Async client for Identity internal calls (optional profile lookups)."""

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings | None = None) -> None:
        self._client = http_client
        self._settings = settings or get_settings()
        # Tolerate an unset base at construction; raise at call time instead so
        # dependency wiring never crashes an un-configured deploy.
        self._base = (self._settings.IDENTITY_BASE_URL or "").rstrip("/")

    def _require_base(self) -> str:
        if not self._base:
            raise IdentityUpstreamError("IDENTITY_BASE_URL is not configured.")
        return self._base

    # ── User auth flows (delegated; issue Identity RS256 tokens) ────

    async def signup(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/api/v1/auth/signup", json=body)

    async def login(self, email: str, password: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/api/v1/auth/login", json={"email": email, "password": password},
        )

    async def request_otp(self, email: str) -> dict[str, Any]:
        return await self._request("POST", "/api/v1/auth/request-otp", json={"email": email})

    async def verify_otp(self, email: str, code: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/api/v1/auth/verify-otp", json={"email": email, "code": code},
        )

    async def verify_email_code(self, email: str, code: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/api/v1/auth/verify-code", json={"email": email, "code": code},
        )

    async def resend_verification(self, email: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/api/v1/auth/resend-verification", json={"email": email},
        )

    async def refresh(self, refresh_token: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/api/v1/auth/refresh", json={"refresh_token": refresh_token},
        )

    async def logout(self, refresh_token: str, access_token: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/auth/logout",
            json={"refresh_token": refresh_token},
            headers={"Authorization": f"Bearer {access_token}"},
        )

    # ── Internal (service-token) calls ─────────────────────────────

    async def internal_get_user(self, user_id: UUID | str) -> dict[str, Any] | None:
        try:
            return await self._internal_get(f"/api/v1/internal/users/{user_id}")
        except IdentityClientError as exc:
            if exc.status_code == 404:
                return None
            raise

    # ── helpers ────────────────────────────────────────────────────

    async def _internal_get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        token = mint_service_token("scienthesis-identity")
        return await self._request(
            "GET", path, params=params, headers={"X-Service-Token": token},
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._require_base()}{path}"
        try:
            resp = await self._client.request(
                method, url, json=json, params=params, headers=headers,
            )
        except httpx.HTTPError as exc:
            logger.error("identity_request_transport_failed path=%s error=%s", path, exc)
            raise IdentityUpstreamError(
                f"Could not reach Identity at {url}: {exc}",
            ) from exc

        if resp.status_code >= 500:
            raise IdentityUpstreamError(
                f"Identity returned {resp.status_code}: {resp.text[:200]}",
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise IdentityClientError(status_code=resp.status_code, detail=detail)
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()
