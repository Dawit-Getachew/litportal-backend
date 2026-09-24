"""JWT + password-hashing helpers, FastAPI deps, and login rate-limiter."""
from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import logging

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo.errors import DuplicateKeyError

from .config import settings
from .db import Collections, get_db
from models.common import new_id, utcnow

logger = logging.getLogger("litportal.security")

bearer_scheme = HTTPBearer(auto_error=False)

# Projection used everywhere we return a user dict — never leak the hash.
_USER_PROJECTION = {"hashed_password": 0, "_id": 0}


# ---------- password hashing ----------
# New hashes are Argon2id (OWASP low-memory profile: 19 MiB, t=2, p=1 — far
# cheaper per login than bcrypt cost 12 on a small task); existing bcrypt hashes
# still verify. Imports are lazy: the summary-generation worker imports this
# module but never hashes. Request handlers must use the *_async variants so
# CPU-bound hashing never blocks the event loop.
# Full-format check first: bcrypt 4.1.x panics (BaseException) on malformed hashes.
_BCRYPT_HASH_RE = re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$")
_BCRYPT_MAX_PASSWORD_BYTES = 72  # bcrypt only reads the first 72 bytes
_argon2_hasher = None
_hash_executor: ThreadPoolExecutor | None = None
_hash_lock = threading.Lock()


def _argon2():
    global _argon2_hasher
    if _argon2_hasher is None:
        from argon2 import PasswordHasher

        _argon2_hasher = PasswordHasher(
            time_cost=int(os.environ.get("PASSWORD_ARGON2_TIME_COST", "2")),
            memory_cost=int(os.environ.get("PASSWORD_ARGON2_MEMORY_KIB", "19456")),
            parallelism=int(os.environ.get("PASSWORD_ARGON2_PARALLELISM", "1")),
        )
    return _argon2_hasher


def _hash_pool() -> ThreadPoolExecutor:
    global _hash_executor
    if _hash_executor is None:
        with _hash_lock:
            if _hash_executor is None:
                _hash_executor = ThreadPoolExecutor(
                    max_workers=max(1, int(os.environ.get("PASSWORD_HASH_MAX_CONCURRENCY", "2"))),
                    thread_name_prefix="password-hash",
                )
    return _hash_executor


def hash_password(password: str) -> str:
    return _argon2().hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Argon2id or legacy bcrypt. Never raises (shadow users store an empty hash)."""
    if not isinstance(plain, str) or not isinstance(hashed, str):
        return False
    try:
        if hashed.startswith("$argon2"):
            from argon2.exceptions import InvalidHashError, VerificationError

            try:
                return _argon2().verify(hashed, plain)
            except (VerificationError, InvalidHashError):
                return False
        if _BCRYPT_HASH_RE.match(hashed):
            import bcrypt

            secret = plain.encode("utf-8")[:_BCRYPT_MAX_PASSWORD_BYTES]
            return bcrypt.checkpw(secret, hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
    return False


async def hash_password_async(password: str) -> str:
    return await asyncio.get_running_loop().run_in_executor(_hash_pool(), hash_password, password)


async def verify_password_async(plain: str, hashed: str) -> bool:
    return await asyncio.get_running_loop().run_in_executor(_hash_pool(), verify_password, plain, hashed)


# ---------- token mgmt ----------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: str, email: str, role: str) -> tuple[str, int]:
    """Return (token, expires_in_seconds)."""
    now = _now()
    ttl = timedelta(minutes=settings.JWT_ACCESS_TTL_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
        "type": "access",
    }
    token = jwt.encode(payload, settings.JWT_ACCESS_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token, int(ttl.total_seconds())


def create_refresh_token(user_id: str) -> tuple[str, int]:
    """Return (token, max_age_seconds)."""
    now = _now()
    ttl = timedelta(days=settings.JWT_REFRESH_TTL_DAYS)
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
        "type": "refresh",
    }
    token = jwt.encode(payload, settings.JWT_REFRESH_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token, int(ttl.total_seconds())


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.JWT_ACCESS_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")
    return payload


def decode_refresh_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.JWT_REFRESH_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Refresh token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid refresh token") from exc
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")
    return payload


# ---------- FastAPI deps ----------
async def _load_native_user(raw_token: str) -> dict[str, Any]:
    """Original Portal-Engine path: decode native HS256 token + load user.

    Raises HTTPException(401) exactly like before. Used as-is when Identity is
    disabled, and as the first validator in the chain when Identity is enabled.
    """
    payload = decode_access_token(raw_token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    db = get_db()
    user = await db[Collections.USERS].find_one({"id": user_id}, _USER_PROJECTION)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def _upsert_identity_user(sub: str, email: str | None) -> dict[str, Any]:
    """Provision (or link) a local Mongo shadow user for an Identity ``sub``.

    Resolution: (1) existing doc with id == sub → stamp identity_id if missing;
    (2) existing native doc with the same email → link by stamping identity_id
    (keeps its local id); (3) otherwise create a new shadow doc with id == sub.
    Returns the user dict (never includes the password hash).
    """
    db = get_db()
    users = db[Collections.USERS]
    now = utcnow()

    existing = await users.find_one({"id": sub}, _USER_PROJECTION)
    if existing:
        if not existing.get("identity_id"):
            await users.update_one(
                {"id": sub}, {"$set": {"identity_id": sub, "updated_at": now}},
            )
            existing["identity_id"] = sub
            existing["updated_at"] = now
        return existing

    if email:
        by_email = await users.find_one({"email": email}, _USER_PROJECTION)
        if by_email:
            await users.update_one(
                {"id": by_email["id"]},
                {"$set": {"identity_id": sub, "updated_at": now}},
            )
            by_email["identity_id"] = sub
            by_email["updated_at"] = now
            return by_email

    doc = {
        "id": sub,
        "identity_id": sub,
        "email": email or f"unknown+{sub}@scienthesis.local",
        "name": (email.split("@")[0] if email else "LitPortal User"),
        "hashed_password": "",
        "role": "user",
        "first_use_acknowledged": False,
        "created_at": now,
        "updated_at": now,
    }
    try:
        await users.insert_one(dict(doc))
    except DuplicateKeyError:
        # Race or email-unique collision — re-read the winning row.
        winner = await users.find_one({"id": sub}, _USER_PROJECTION)
        if winner is None and email:
            winner = await users.find_one({"email": email}, _USER_PROJECTION)
        if winner is not None:
            return winner
    doc.pop("_id", None)
    return doc


async def _resolve_identity_user(
    raw_token: str, http_client: Any, settings_obj: Any,
) -> dict[str, Any] | None:
    """Validate an Identity RS256 token (via JWKS) and return the shadow user.

    Returns ``None`` when the token is not Identity-shaped (so the caller can
    fall through). Raises HTTPException(401) for an Identity-shaped-but-invalid
    token — a valid-but-failed Identity token is never silently downgraded.
    """
    from clients.identity_client import (
        IdentityUpstreamError,
        validate_identity_access_token,
    )

    try:
        payload = await validate_identity_access_token(
            raw_token, http_client=http_client, settings=settings_obj,
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid Identity access token: {exc}") from exc
    except IdentityUpstreamError as exc:
        # An Identity-shaped token we could not verify because the JWKS was
        # unreachable — fail loud (503) instead of a misleading 401.
        raise HTTPException(
            status_code=503, detail="Identity service unavailable; cannot verify token.",
        ) from exc
    if payload is None:
        return None
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Identity token missing 'sub' claim")
    return await _upsert_identity_user(str(sub), payload.get("email"))


async def _upsert_litpulse_user(litpulse_user_id: str, email: str) -> dict[str, Any]:
    """Provision/link a local shadow user for a LitPulse legacy token.

    Keyed by email (the unique index). Stores ``litpulse_user_id`` for
    reference. These users have NO identity_id, so they are not mirrored to
    LitHub until they next authenticate via central Identity.
    """
    db = get_db()
    users = db[Collections.USERS]
    now = utcnow()
    existing = await users.find_one({"email": email}, _USER_PROJECTION)
    if existing:
        if existing.get("litpulse_user_id") != litpulse_user_id:
            await users.update_one(
                {"id": existing["id"]},
                {"$set": {"litpulse_user_id": litpulse_user_id, "updated_at": now}},
            )
            existing["litpulse_user_id"] = litpulse_user_id
            existing["updated_at"] = now
        return existing
    doc = {
        "id": new_id(),
        "litpulse_user_id": litpulse_user_id,
        "email": email,
        "name": email.split("@")[0],
        "hashed_password": "",
        "role": "user",
        "first_use_acknowledged": False,
        "created_at": now,
        "updated_at": now,
    }
    try:
        await users.insert_one(dict(doc))
    except DuplicateKeyError:
        winner = await users.find_one({"email": email}, _USER_PROJECTION)
        if winner is not None:
            return winner
    doc.pop("_id", None)
    return doc


async def _resolve_litpulse_user(raw_token: str) -> dict[str, Any] | None:
    """Validate a LitPulse-issued HS256 legacy token; return the shadow user or
    None when the token isn't a LitPulse token (so the caller can fall through)."""
    if not settings.LITPULSE_JWT_ENABLED or not settings.LITPULSE_JWT_SECRET_KEY:
        return None
    try:
        payload = jwt.decode(
            raw_token, settings.LITPULSE_JWT_SECRET_KEY, algorithms=["HS256"],
        )
    except jwt.InvalidTokenError:
        return None
    if payload.get("type") != "access":
        return None
    litpulse_user_id = payload.get("user_id")
    email = payload.get("email")
    if not litpulse_user_id or not email:
        return None
    return await _upsert_litpulse_user(str(litpulse_user_id), str(email).lower().strip())


async def get_current_user(
    request: Request = None,  # noqa: RUF013 — FastAPI injects Request; default lets tests omit it
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    raw_token = credentials.credentials

    # Fast path — no integration flags → behave EXACTLY like the standalone
    # Emergent backend (byte-identical).
    if not settings.LITPORTAL_USE_IDENTITY and not settings.LITPULSE_JWT_ENABLED:
        return await _load_native_user(raw_token)

    # Validator chain: native (HS256) → Identity (RS256/JWKS) → LitPulse legacy
    # (HS256). Each token format is unambiguous by algorithm/signature.
    try:
        return await _load_native_user(raw_token)
    except HTTPException:
        pass  # not a valid native token — try the next validator.

    if settings.LITPORTAL_USE_IDENTITY:
        http_client = getattr(getattr(request, "app", None), "state", None)
        http_client = getattr(http_client, "http_client", None) if http_client else None
        if http_client is not None:
            identity_user = await _resolve_identity_user(raw_token, http_client, settings)
            if identity_user is not None:
                return identity_user

    if settings.LITPULSE_JWT_ENABLED:
        litpulse_user = await _resolve_litpulse_user(raw_token)
        if litpulse_user is not None:
            return litpulse_user

    raise HTTPException(status_code=401, detail="Invalid or expired access token")


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return current_user


# ---------- in-memory login rate limiter ----------
# TODO(Phase 2+): swap for Redis-backed sliding-window limiter so state survives
# restarts and is shared across worker processes.
_LOGIN_ATTEMPTS: dict[str, list[float]] = defaultdict(list)


def _rate_key(email: str, request: Request | None) -> str:
    ip = ""
    if request is not None and request.client is not None:
        ip = request.client.host or ""
    return f"{email.lower()}|{ip}"


def check_login_rate_limit(email: str, request: Request | None = None) -> None:
    """Raise 429 if the key has exceeded the failure budget in the current window."""
    key = _rate_key(email, request)
    now = time.monotonic()
    window = settings.LOGIN_RATE_LIMIT_WINDOW_MIN * 60
    # purge old
    _LOGIN_ATTEMPTS[key] = [t for t in _LOGIN_ATTEMPTS[key] if now - t < window]
    if len(_LOGIN_ATTEMPTS[key]) >= settings.LOGIN_RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please wait and try again.",
        )


def record_login_failure(email: str, request: Request | None = None) -> None:
    _LOGIN_ATTEMPTS[_rate_key(email, request)].append(time.monotonic())


def reset_login_attempts(email: str, request: Request | None = None) -> None:
    _LOGIN_ATTEMPTS.pop(_rate_key(email, request), None)
