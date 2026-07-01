"""Unit tests for the Scienthesis integration clients.

These exercise the framework-agnostic crypto/HTTP paths in isolation — no Mongo
and no live Identity/LitHub services required:
  - service-token minting (HS256, iss=litportal)
  - Identity RS256 token validation via a mocked JWKS endpoint
  - the raise-vs-None contract that drives the additive auth chain

NOTE: this backend's ``Settings`` reads env into CLASS attributes at import
time, so tests patch ``config.Settings`` attributes directly (changing
``os.environ`` after import has no effect). pytest.ini sets
``asyncio_mode = auto`` so async tests need no marker.
"""
from __future__ import annotations

import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

import core.config as config
from clients import identity_client, service_token
from core.config import get_settings


# ── helpers ─────────────────────────────────────────────────────────
@pytest.fixture()
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture()
def settings(monkeypatch):
    """Settings with an absolute JWKS URL so httpx.MockTransport can dispatch."""
    monkeypatch.setattr(
        config.Settings, "IDENTITY_JWKS_URL",
        "http://identity.test/.well-known/jwks.json",
    )
    identity_client.reset_jwks_cache_for_tests()
    return get_settings()


def _jwks_for(key, kid: str) -> dict:
    pub = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    pub.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": [pub]}


def _private_pem(key) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _mock_client(jwks: dict) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=jwks)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _mint_user_token(key, kid, settings_obj, *, aud=None, iss=None, exp_delta=300):
    now = int(time.time())
    return jwt.encode(
        {
            "sub": "11111111-1111-1111-1111-111111111111",
            "email": "user@example.org",
            "type": "access",
            "iss": iss or settings_obj.IDENTITY_JWT_ISSUER,
            "aud": aud or settings_obj.IDENTITY_JWT_AUDIENCE,
            "iat": now,
            "exp": now + exp_delta,
        },
        _private_pem(key), algorithm="RS256", headers={"kid": kid},
    )


# ── service token ───────────────────────────────────────────────────
def test_mint_service_token_shape(monkeypatch):
    monkeypatch.setattr(config.Settings, "SERVICE_TOKEN_SECRET", "unit-test-secret")
    service_token.reset_cache_for_tests()
    tok = service_token.mint_service_token("scienthesis-lithub")
    claims = jwt.decode(
        tok, "unit-test-secret", algorithms=["HS256"], audience="scienthesis-lithub",
    )
    assert claims["iss"] == "litportal"
    assert claims["aud"] == "scienthesis-lithub"
    assert claims["type"] == "service"


def test_mint_service_token_requires_secret(monkeypatch):
    monkeypatch.setattr(config.Settings, "SERVICE_TOKEN_SECRET", "")
    service_token.reset_cache_for_tests()
    with pytest.raises(RuntimeError):
        service_token.mint_service_token("scienthesis-identity")


# ── identity validator ──────────────────────────────────────────────
async def test_validator_accepts_valid_rs256(rsa_key, settings):
    kid = "unit-kid-valid"
    token = _mint_user_token(rsa_key, kid, settings)
    async with _mock_client(_jwks_for(rsa_key, kid)) as c:
        payload = await identity_client.validate_identity_access_token(
            token, http_client=c, settings=settings,
        )
    assert payload is not None
    assert payload["sub"] == "11111111-1111-1111-1111-111111111111"
    assert payload["email"] == "user@example.org"


async def test_validator_returns_none_for_hs256(settings):
    """An HS256 token is 'not Identity-shaped' → None (caller falls back)."""
    token = jwt.encode({"sub": "x", "type": "access"}, "hs-secret", algorithm="HS256")
    async with _mock_client({"keys": []}) as c:
        payload = await identity_client.validate_identity_access_token(
            token, http_client=c, settings=settings,
        )
    assert payload is None


async def test_validator_raises_on_expired(rsa_key, settings):
    """An Identity-shaped-but-expired token must RAISE (never silently fall back)."""
    kid = "unit-kid-expired"
    token = _mint_user_token(rsa_key, kid, settings, exp_delta=-300)
    async with _mock_client(_jwks_for(rsa_key, kid)) as c:
        with pytest.raises(jwt.InvalidTokenError):
            await identity_client.validate_identity_access_token(
                token, http_client=c, settings=settings,
            )


async def test_validator_rejects_wrong_audience(rsa_key, settings):
    """A valid RS256 token for a DIFFERENT audience must raise."""
    kid = "unit-kid-aud"
    token = _mint_user_token(rsa_key, kid, settings, aud="some-other-app")
    async with _mock_client(_jwks_for(rsa_key, kid)) as c:
        with pytest.raises(jwt.InvalidTokenError):
            await identity_client.validate_identity_access_token(
                token, http_client=c, settings=settings,
            )


async def test_validator_raises_when_jwks_unreachable(rsa_key, settings):
    """An Identity-shaped token whose JWKS can't be fetched must fail loud."""
    kid = "unit-kid-down"
    token = _mint_user_token(rsa_key, kid, settings)

    def down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"e": 1})

    async with httpx.AsyncClient(transport=httpx.MockTransport(down)) as c:
        with pytest.raises(identity_client.IdentityUpstreamError):
            await identity_client.validate_identity_access_token(
                token, http_client=c, settings=settings,
            )


# ── identity auth-flow delegation (the token-issuance handoff) ───────
@pytest.fixture()
def identity_base(monkeypatch):
    monkeypatch.setattr(config.Settings, "IDENTITY_BASE_URL", "http://identity.test")
    return get_settings()


def _capture_client():
    """A mock Identity that records the last request and returns a token pair."""
    cap: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        cap["path"] = request.url.path
        cap["headers"] = {k.lower(): v for k, v in request.headers.items()}
        try:
            cap["body"] = _json.loads(request.content.decode() or "{}")
        except Exception:  # noqa: BLE001
            cap["body"] = {}
        return httpx.Response(200, json={
            "access_token": "a.b.c", "refresh_token": "r",
            "token_type": "bearer", "expires_in": 1800,
        })

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), cap


async def test_identity_client_login_delegates(identity_base):
    from clients.identity_client import IdentityClient

    client_http, cap = _capture_client()
    async with client_http as c:
        ic = IdentityClient(http_client=c, settings=identity_base)
        data = await ic.login("u@x.io", "pw")
    assert cap["path"] == "/api/v1/auth/login"
    assert cap["body"] == {"email": "u@x.io", "password": "pw"}
    assert data["access_token"] == "a.b.c"


async def test_identity_client_refresh_and_logout(identity_base):
    from clients.identity_client import IdentityClient

    client_http, cap = _capture_client()
    async with client_http as c:
        ic = IdentityClient(http_client=c, settings=identity_base)
        await ic.refresh("rtok")
        assert cap["path"] == "/api/v1/auth/refresh"
        assert cap["body"] == {"refresh_token": "rtok"}
        await ic.logout("rtok", "atok")
        assert cap["path"] == "/api/v1/auth/logout"
        assert cap["headers"].get("authorization") == "Bearer atok"


async def test_identity_client_signup_and_otp(identity_base):
    from clients.identity_client import IdentityClient

    client_http, cap = _capture_client()
    async with client_http as c:
        ic = IdentityClient(http_client=c, settings=identity_base)
        await ic.signup({"email": "u@x.io", "password": "pw", "signup_method": "password"})
        assert cap["path"] == "/api/v1/auth/signup"
        await ic.verify_otp("u@x.io", "123456")
        assert cap["path"] == "/api/v1/auth/verify-otp"
        assert cap["body"] == {"email": "u@x.io", "code": "123456"}
