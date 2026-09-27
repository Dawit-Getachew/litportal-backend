"""Auth router: signup, login, OTP, refresh (cookie), logout, /me, ack-first-use.

Two modes, selected by ``LITPORTAL_USE_IDENTITY``:

* OFF (default) — native HS256 tokens minted locally (original Emergent
  behavior; byte-identical).
* ON — every auth flow is DELEGATED to the central Scienthesis Identity
  Service, which mints RS256 tokens whose audience covers litpulse + litportal
  + lithub. The SAME token then works across all Scienthesis apps, and the
  local Mongo ``users`` row is a shadow keyed by the Identity ``sub``.

The response shape (``TokenResponse`` + refresh cookie) is identical in both
modes, so the frontend is unaffected.
"""
from __future__ import annotations

from datetime import datetime, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials

from clients.identity_client import (
    IdentityClient,
    IdentityClientError,
    IdentityUpstreamError,
)
from core.config import settings
from core.db import Collections, get_db
from core.security import (
    bearer_scheme,
    check_login_rate_limit,
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_current_user,
    hash_password_async,
    record_login_failure,
    reset_login_attempts,
    verify_password_async,
)
from core.security import _upsert_identity_user  # provisioning helper (shared)
from models.users import (
    OTPRequest,
    OTPVerify,
    TokenResponse,
    UserDoc,
    UserLogin,
    UserPublic,
    UserSignup,
)
from services.audit import audit_service

router = APIRouter(prefix="/auth", tags=["auth"])


def _to_public(u: dict) -> UserPublic:
    return UserPublic(
        id=u["id"],
        email=u["email"],
        name=u["name"],
        role=u["role"],
        first_use_acknowledged=u.get("first_use_acknowledged", False),
        created_at=u["created_at"],
        updated_at=u["updated_at"],
    )


def _set_refresh_cookie(response: Response, token: str, max_age_seconds: int) -> None:
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=token,
        max_age=max_age_seconds,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        path=settings.REFRESH_COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        path=settings.REFRESH_COOKIE_PATH,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
    )


def _token_response(user: dict, response: Response) -> TokenResponse:
    """Native (local HS256) token response."""
    access_token, expires_in = create_access_token(user["id"], user["email"], user["role"])
    refresh_token, max_age = create_refresh_token(user["id"])
    _set_refresh_cookie(response, refresh_token, max_age)
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        user=_to_public(user),
    )


# ── Central Identity delegation ─────────────────────────────────────

def _identity_enabled() -> bool:
    return bool(settings.LITPORTAL_USE_IDENTITY and settings.IDENTITY_BASE_URL)


def _make_identity_client(request: Request) -> IdentityClient:
    http_client = getattr(request.app.state, "http_client", None)
    return IdentityClient(http_client=http_client, settings=settings)


async def _delegated_token_response(data: dict, response: Response) -> TokenResponse:
    """Turn an Identity token pair into LitPortal's TokenResponse + refresh cookie.

    Sets the Identity-issued (opaque) refresh token as the cookie, provisions
    the local shadow user from the access token's claims, and returns the
    Identity RS256 access token verbatim.
    """
    access = data.get("access_token")
    if not access:
        raise HTTPException(status_code=502, detail="Identity returned no access token.")
    refresh = data.get("refresh_token")
    expires_in = int(data.get("expires_in") or settings.JWT_ACCESS_TTL_MINUTES * 60)
    if refresh:
        _set_refresh_cookie(response, refresh, settings.JWT_REFRESH_TTL_DAYS * 86400)
    # Decode unverified ONLY to read sub/email for provisioning — the token was
    # just minted by Identity over a trusted server-to-server channel; full
    # signature verification happens on subsequent requests via JWKS.
    try:
        claims = jwt.decode(access, options={"verify_signature": False})
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=502, detail="Identity token is malformed.") from exc
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status_code=502, detail="Identity token missing 'sub'.")
    user = await _upsert_identity_user(str(sub), claims.get("email"))
    return TokenResponse(
        access_token=access,
        token_type=data.get("token_type", "bearer"),
        expires_in=expires_in,
        user=_to_public(user),
    )


def _raise_identity_http(exc: Exception, *, client_default: int) -> None:
    if isinstance(exc, IdentityUpstreamError):
        raise HTTPException(status_code=503, detail="Identity service unavailable.") from exc
    if isinstance(exc, IdentityClientError):
        raise HTTPException(
            status_code=exc.status_code or client_default, detail=exc.detail,
        ) from exc
    raise exc


# ── Endpoints ───────────────────────────────────────────────────────

@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: UserSignup, request: Request, response: Response):
    email = body.email.lower().strip()
    if _identity_enabled():
        client = _make_identity_client(request)
        try:
            data = await client.signup({
                "email": email,
                "password": body.password,
                "signup_method": "password",
                "full_name": body.name.strip(),
            })
        except (IdentityUpstreamError, IdentityClientError) as exc:
            _raise_identity_http(exc, client_default=400)
        await audit_service.log("user_signup", payload={"email": email, "via": "identity"})
        return await _delegated_token_response(data, response)

    # Native path (unchanged).
    db = get_db()
    existing = await db[Collections.USERS].find_one({"email": email}, {"_id": 0, "id": 1})
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    user = UserDoc(
        email=email,
        name=body.name.strip(),
        hashed_password=await hash_password_async(body.password),
        role="user",
        first_use_acknowledged=False,
    )
    doc = user.model_dump()
    for k in ("created_at", "updated_at"):
        doc[k] = doc[k].astimezone(timezone.utc)
    await db[Collections.USERS].insert_one(doc)
    await audit_service.log("user_signup", user_id=user.id, payload={"email": email})
    return _token_response(doc, response)


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin, request: Request, response: Response):
    email = body.email.lower().strip()
    if _identity_enabled():
        client = _make_identity_client(request)
        try:
            data = await client.login(email, body.password)
        except (IdentityUpstreamError, IdentityClientError) as exc:
            _raise_identity_http(exc, client_default=401)
        await audit_service.log("user_login", payload={"email": email, "via": "identity"})
        return await _delegated_token_response(data, response)

    # Native path (unchanged).
    db = get_db()
    check_login_rate_limit(email, request)
    user = await db[Collections.USERS].find_one({"email": email}, {"_id": 0})
    if not user or not await verify_password_async(body.password, user.get("hashed_password", "")):
        record_login_failure(email, request)
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    reset_login_attempts(email, request)
    await audit_service.log("user_login", user_id=user["id"], payload={"email": email})
    return _token_response(user, response)


@router.post("/request-otp", status_code=200)
async def request_otp(body: OTPRequest, request: Request):
    """Passwordless sign-in — delegated to central Identity (no native OTP)."""
    if not _identity_enabled():
        raise HTTPException(status_code=503, detail="OTP sign-in requires the Identity Service.")
    client = _make_identity_client(request)
    try:
        await client.request_otp(body.email.lower().strip())
    except (IdentityUpstreamError, IdentityClientError) as exc:
        _raise_identity_http(exc, client_default=400)
    return {"message": "Verification code sent. Check your email."}


@router.post("/verify-otp", response_model=TokenResponse)
async def verify_otp(body: OTPVerify, request: Request, response: Response):
    if not _identity_enabled():
        raise HTTPException(status_code=503, detail="OTP sign-in requires the Identity Service.")
    client = _make_identity_client(request)
    try:
        data = await client.verify_otp(body.email.lower().strip(), body.code)
    except (IdentityUpstreamError, IdentityClientError) as exc:
        _raise_identity_http(exc, client_default=400)
    await audit_service.log("user_login", payload={"email": body.email, "via": "identity-otp"})
    return await _delegated_token_response(data, response)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response):
    token = request.cookies.get(settings.REFRESH_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="No refresh token provided.")

    if _identity_enabled():
        client = _make_identity_client(request)
        try:
            data = await client.refresh(token)
        except IdentityUpstreamError as exc:
            raise HTTPException(status_code=503, detail="Identity service unavailable.") from exc
        except IdentityClientError as exc:
            _clear_refresh_cookie(response)
            raise HTTPException(status_code=exc.status_code or 401, detail=exc.detail) from exc
        return await _delegated_token_response(data, response)

    # Native path (unchanged).
    payload = decode_refresh_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token payload.")
    db = get_db()
    user = await db[Collections.USERS].find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return _token_response(user, response)


@router.post("/logout", status_code=200)
async def logout(
    request: Request,
    response: Response,
    current_user: dict = Depends(get_current_user),
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    if _identity_enabled():
        token = request.cookies.get(settings.REFRESH_COOKIE_NAME)
        if token:
            client = _make_identity_client(request)
            try:  # best-effort — a transient Identity outage must not block logout
                await client.logout(token, creds.credentials if creds else "")
            except Exception:  # noqa: BLE001
                pass
    _clear_refresh_cookie(response)
    await audit_service.log("user_logout", user_id=current_user["id"])
    return {"status": "ok"}


@router.get("/me", response_model=UserPublic)
async def me(current_user: dict = Depends(get_current_user)):
    return _to_public(current_user)


@router.post("/acknowledge-first-use", response_model=UserPublic)
async def acknowledge_first_use(current_user: dict = Depends(get_current_user)):
    db = get_db()
    now = datetime.now(timezone.utc)
    await db[Collections.USERS].update_one(
        {"id": current_user["id"]},
        {"$set": {"first_use_acknowledged": True, "updated_at": now}},
    )
    await audit_service.log("first_use_acknowledged", user_id=current_user["id"])
    updated = await db[Collections.USERS].find_one({"id": current_user["id"]}, {"_id": 0})
    return _to_public(updated)
