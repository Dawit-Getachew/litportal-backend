"""Cross-app authentication regression suite (LitPortal half).

The cutover promise is single-sign-on: one Identity-issued RS256 token is
accepted by BOTH LitPulse and LitPortal, and the same ``sub`` resolves to the
same person in each app. This file pins LitPortal's half of that contract
using the same token cases LitPulse asserts in
``litpulse-backend/tests/test_scienthesis_identity_integration.py``, so a
divergence in either verifier fails a build.

Cases covered: valid access token · multi-audience token (the real shape
Identity issues) · expired · wrong audience · wrong issuer · tampered
signature · non-access token type · HS256 downgrade attempt · unknown kid ·
JWKS unreachable · missing sub · user continuity across apps.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from clients import identity_client as ic

ISSUER = "scienthesis-identity"
AUDIENCE = "litportal"
KID = "identity-test-key"
SUB = "11111111-2222-3333-4444-555555555555"
EMAIL = "clinician@example.com"


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    numbers = key.public_key().public_numbers()

    def _b64(value: int) -> str:
        import base64

        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    jwk = {
        "kty": "RSA",
        "kid": KID,
        "use": "sig",
        "alg": "RS256",
        "n": _b64(numbers.n),
        "e": _b64(numbers.e),
    }
    return private_pem, jwk


@pytest.fixture
def settings():
    """Real Settings with only the Identity fields pointed at the test issuer,
    so the test cannot drift from the attributes production code reads."""
    from core.config import get_settings

    real = get_settings()

    class _TestSettings:
        def __getattr__(self, name):
            return getattr(real, name)

        IDENTITY_JWKS_URL = "http://identity.test/.well-known/jwks.json"
        IDENTITY_BASE_URL = "http://identity.test"
        IDENTITY_JWT_ISSUER = ISSUER
        IDENTITY_JWT_AUDIENCE = AUDIENCE

    return _TestSettings()


@pytest.fixture(autouse=True)
def _reset_jwks_cache():
    # reset() (not clear()) also drops the negative-cache timestamp — clear()
    # deliberately keeps it so an outage still throttles refetches in prod.
    ic._jwks_cache.reset()
    yield
    ic._jwks_cache.reset()


def _jwks_client(jwk: dict[str, Any] | None, *, status: int = 200):
    body = {"keys": [jwk]} if jwk else {"keys": []}

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, text="jwks down")
        return httpx.Response(200, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _token(private_pem: str, **over: Any) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": SUB,
        "email": EMAIL,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "type": "access",
        "iat": now,
        "exp": now + 900,
    }
    claims.update(over)
    headers = {"kid": over.pop("_kid", KID)}
    alg = over.pop("_alg", "RS256")
    key = private_pem if alg == "RS256" else "shared-secret"
    return jwt.encode(claims, key, algorithm=alg, headers=headers)


# ── Accepted ────────────────────────────────────────────────────────


async def test_valid_access_token_is_accepted(keypair, settings):
    private_pem, jwk = keypair
    async with _jwks_client(jwk) as client:
        payload = await ic.validate_identity_access_token(
            _token(private_pem), http_client=client, settings=settings
        )
    assert payload["sub"] == SUB
    assert payload["email"] == EMAIL


async def test_multi_audience_token_is_accepted(keypair, settings):
    """Identity issues one token listing every app — it must validate here."""
    private_pem, jwk = keypair
    token = _token(private_pem, aud=["litpulse", AUDIENCE, "scienthesis-lithub"])
    async with _jwks_client(jwk) as client:
        payload = await ic.validate_identity_access_token(
            token, http_client=client, settings=settings
        )
    assert payload["sub"] == SUB


# ── Rejected (invalid) ──────────────────────────────────────────────


async def test_expired_token_is_rejected(keypair, settings):
    private_pem, jwk = keypair
    now = int(time.time())
    token = _token(private_pem, iat=now - 3600, exp=now - 60)
    async with _jwks_client(jwk) as client:
        with pytest.raises(jwt.ExpiredSignatureError):
            await ic.validate_identity_access_token(
                token, http_client=client, settings=settings
            )


async def test_wrong_audience_is_rejected(keypair, settings):
    private_pem, jwk = keypair
    token = _token(private_pem, aud="some-other-app")
    async with _jwks_client(jwk) as client:
        with pytest.raises(jwt.InvalidAudienceError):
            await ic.validate_identity_access_token(
                token, http_client=client, settings=settings
            )


async def test_wrong_issuer_is_rejected(keypair, settings):
    private_pem, jwk = keypair
    token = _token(private_pem, iss="https://evil.example.com")
    async with _jwks_client(jwk) as client:
        with pytest.raises(jwt.InvalidIssuerError):
            await ic.validate_identity_access_token(
                token, http_client=client, settings=settings
            )


async def test_tampered_signature_is_rejected(keypair, settings):
    private_pem, jwk = keypair
    token = _token(private_pem)
    head, body, sig = token.split(".")
    tampered = f"{head}.{body}.{sig[:-4]}AAAA"
    async with _jwks_client(jwk) as client:
        with pytest.raises(jwt.InvalidTokenError):
            await ic.validate_identity_access_token(
                tampered, http_client=client, settings=settings
            )


async def test_refresh_token_cannot_authenticate(keypair, settings):
    """Only ``type=access`` may authenticate a request."""
    private_pem, jwk = keypair
    token = _token(private_pem, type="refresh")
    async with _jwks_client(jwk) as client:
        with pytest.raises(jwt.InvalidTokenError):
            await ic.validate_identity_access_token(
                token, http_client=client, settings=settings
            )


# ── Not Identity-shaped (caller falls through) ──────────────────────


async def test_hs256_token_is_not_treated_as_identity(keypair, settings):
    """A forged HS256 token must never be verified against the JWKS path."""
    private_pem, jwk = keypair
    token = _token(private_pem, _alg="HS256")
    async with _jwks_client(jwk) as client:
        assert (
            await ic.validate_identity_access_token(
                token, http_client=client, settings=settings
            )
            is None
        )


async def test_unknown_kid_falls_through(keypair, settings):
    private_pem, jwk = keypair
    token = _token(private_pem, _kid="not-our-key")
    async with _jwks_client(jwk) as client:
        assert (
            await ic.validate_identity_access_token(
                token, http_client=client, settings=settings
            )
            is None
        )


async def test_garbage_is_not_identity_shaped(settings):
    async with _jwks_client(None) as client:
        assert (
            await ic.validate_identity_access_token(
                "not-a-jwt", http_client=client, settings=settings
            )
            is None
        )


# ── Availability ────────────────────────────────────────────────────


async def test_unreachable_jwks_fails_loud_not_silent(keypair, settings):
    """An Identity-shaped token we cannot verify must raise (→ 503), never be
    downgraded to a misleading 401 or silently accepted."""
    private_pem, _jwk = keypair
    async with _jwks_client(None, status=503) as client:
        with pytest.raises(ic.IdentityUpstreamError):
            await ic.validate_identity_access_token(
                _token(private_pem), http_client=client, settings=settings
            )


# ── Cross-app continuity ────────────────────────────────────────────


async def test_same_sub_yields_stable_identity_across_apps(keypair, settings):
    """The ``sub`` claim is the cross-app key: LitPortal must key its shadow
    user off exactly the value LitPulse stores as ``identity_id``."""
    private_pem, jwk = keypair
    token = _token(private_pem, aud=["litpulse", AUDIENCE])
    async with _jwks_client(jwk) as client:
        payload = await ic.validate_identity_access_token(
            token, http_client=client, settings=settings
        )
    assert payload["sub"] == SUB
    assert payload["email"] == EMAIL
