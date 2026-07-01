"""SCOPE AUDIT — Identity + LitHub integration CONTRACTS (B9/B11/B12).

The central Identity/LitHub services are separate apps not booted here, so this
proves the litportal SIDE of the contract with real crypto + mocked transports:

  * Identity RS256/JWKS token validation (`validate_identity_access_token`):
    accepts a correctly-issued token; rejects expired / wrong-aud / wrong-iss /
    wrong-type / wrong-alg / unknown-kid / missing-kid; fails LOUD when JWKS is
    empty/unreachable (no silent downgrade). — B9 shared auth.
  * Service-token minting (`mint_service_token`): correct iss/aud/type, caching,
    fail-without-secret. — internal-API auth for Identity + LitHub.
  * LitHub client request shaping (`LitHubClient`): correct method/path/headers/
    body for save/membership/delete; upstream + client error mapping. — B12 save.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from clients import identity_client as idc
from clients.identity_client import (
    IdentityUpstreamError,
    validate_identity_access_token,
)
from clients.lithub_client import (
    LitHubClient,
    LitHubClientError,
    LitHubUpstreamError,
)
from clients import service_token as st
from core.config import get_settings

AUD = "litportal"
ISS = "scienthesis-identity"


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


def _jwk(public_key, kid: str) -> dict:
    d = json.loads(RSAAlgorithm.to_jwk(public_key))
    d["kid"] = kid
    d.setdefault("alg", "RS256")
    d.setdefault("use", "sig")
    return d


def _identity_settings():
    return SimpleNamespace(
        JWKS_CACHE_TTL_SECONDS=300,
        IDENTITY_JWKS_URL="https://identity.test/.well-known/jwks.json",
        IDENTITY_BASE_URL="https://identity.test",
        IDENTITY_JWT_AUDIENCE=AUD,
        IDENTITY_JWT_ISSUER=ISS,
    )


def _jwks_client(jwks_keys: list[dict]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"keys": jwks_keys})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _token(private_key, *, kid="kid-1", alg="RS256", aud=AUD, iss=ISS,
           typ="access", exp_delta=timedelta(minutes=10), sub="user-sub-123",
           email="doc@test.com"):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub, "email": email, "aud": aud, "iss": iss, "type": typ,
        "iat": int(now.timestamp()), "exp": int((now + exp_delta).timestamp()),
    }
    headers = {"kid": kid} if kid else {}
    if alg == "HS256":
        return jwt.encode(payload, "shared-secret", algorithm="HS256", headers=headers)
    return jwt.encode(payload, private_key, algorithm="RS256", headers=headers)


@pytest.fixture(autouse=True)
def _reset():
    idc.reset_jwks_cache_for_tests()
    st.reset_cache_for_tests()
    yield
    idc.reset_jwks_cache_for_tests()
    st.reset_cache_for_tests()


# ════════════════════════════════════════════════════════════════════
# Identity RS256 / JWKS validation (B9)
# ════════════════════════════════════════════════════════════════════
async def test_identity_valid_token_accepted():
    key, pub = _keypair()
    async with _jwks_client([_jwk(pub, "kid-1")]) as c:
        payload = await validate_identity_access_token(
            _token(key), http_client=c, settings=_identity_settings())
    assert payload is not None
    assert payload["sub"] == "user-sub-123"
    assert payload["type"] == "access"


async def test_identity_expired_token_rejected():
    key, pub = _keypair()
    async with _jwks_client([_jwk(pub, "kid-1")]) as c:
        with pytest.raises(jwt.InvalidTokenError):
            await validate_identity_access_token(
                _token(key, exp_delta=timedelta(minutes=-5)),
                http_client=c, settings=_identity_settings())


async def test_identity_wrong_audience_rejected():
    key, pub = _keypair()
    async with _jwks_client([_jwk(pub, "kid-1")]) as c:
        with pytest.raises(jwt.InvalidTokenError):
            await validate_identity_access_token(
                _token(key, aud="some-other-app"),
                http_client=c, settings=_identity_settings())


async def test_identity_wrong_issuer_rejected():
    key, pub = _keypair()
    async with _jwks_client([_jwk(pub, "kid-1")]) as c:
        with pytest.raises(jwt.InvalidTokenError):
            await validate_identity_access_token(
                _token(key, iss="evil-issuer"),
                http_client=c, settings=_identity_settings())


async def test_identity_non_access_type_rejected():
    key, pub = _keypair()
    async with _jwks_client([_jwk(pub, "kid-1")]) as c:
        with pytest.raises(jwt.InvalidTokenError):
            await validate_identity_access_token(
                _token(key, typ="refresh"),
                http_client=c, settings=_identity_settings())


async def test_identity_hs256_token_not_identity_shaped():
    key, pub = _keypair()
    async with _jwks_client([_jwk(pub, "kid-1")]) as c:
        out = await validate_identity_access_token(
            _token(key, alg="HS256"), http_client=c, settings=_identity_settings())
    assert out is None  # caller falls through to other validators


async def test_identity_missing_kid_not_identity_shaped():
    key, pub = _keypair()
    async with _jwks_client([_jwk(pub, "kid-1")]) as c:
        out = await validate_identity_access_token(
            _token(key, kid=None), http_client=c, settings=_identity_settings())
    assert out is None


async def test_identity_unknown_kid_falls_through():
    key, pub = _keypair()
    # JWKS serves kid-A; token signed claiming kid-ZZZ.
    async with _jwks_client([_jwk(pub, "kid-A")]) as c:
        out = await validate_identity_access_token(
            _token(key, kid="kid-ZZZ"), http_client=c, settings=_identity_settings())
    assert out is None


async def test_identity_empty_jwks_fails_loud():
    key, _pub = _keypair()
    async with _jwks_client([]) as c:  # JWKS unreachable/empty
        with pytest.raises(IdentityUpstreamError):
            await validate_identity_access_token(
                _token(key), http_client=c, settings=_identity_settings())


async def test_identity_garbage_token_not_identity_shaped():
    key, pub = _keypair()
    async with _jwks_client([_jwk(pub, "kid-1")]) as c:
        out = await validate_identity_access_token(
            "not.a.jwt", http_client=c, settings=_identity_settings())
    assert out is None


# ════════════════════════════════════════════════════════════════════
# Service-token minting (internal API auth)
# ════════════════════════════════════════════════════════════════════
def test_service_token_shape(monkeypatch):
    monkeypatch.setattr(get_settings(), "SERVICE_TOKEN_SECRET", "test-secret")
    st.reset_cache_for_tests()
    tok = st.mint_service_token("scienthesis-lithub")
    payload = jwt.decode(tok, "test-secret", algorithms=["HS256"], audience="scienthesis-lithub")
    assert payload["iss"] == "litportal"
    assert payload["aud"] == "scienthesis-lithub"
    assert payload["type"] == "service"
    assert payload["exp"] > payload["iat"]


def test_service_token_audience_isolation(monkeypatch):
    monkeypatch.setattr(get_settings(), "SERVICE_TOKEN_SECRET", "test-secret")
    st.reset_cache_for_tests()
    a = st.mint_service_token("scienthesis-identity")
    b = st.mint_service_token("scienthesis-lithub")
    assert a != b
    # decode each against its own audience
    assert jwt.decode(a, "test-secret", algorithms=["HS256"], audience="scienthesis-identity")["aud"] == "scienthesis-identity"
    assert jwt.decode(b, "test-secret", algorithms=["HS256"], audience="scienthesis-lithub")["aud"] == "scienthesis-lithub"


def test_service_token_cached(monkeypatch):
    monkeypatch.setattr(get_settings(), "SERVICE_TOKEN_SECRET", "test-secret")
    st.reset_cache_for_tests()
    assert st.mint_service_token("scienthesis-lithub") == st.mint_service_token("scienthesis-lithub")


def test_service_token_requires_secret(monkeypatch):
    monkeypatch.setattr(get_settings(), "SERVICE_TOKEN_SECRET", "")
    st.reset_cache_for_tests()
    with pytest.raises(RuntimeError):
        st.mint_service_token("scienthesis-lithub")


def test_service_token_signature_must_match(monkeypatch):
    monkeypatch.setattr(get_settings(), "SERVICE_TOKEN_SECRET", "real-secret")
    st.reset_cache_for_tests()
    tok = st.mint_service_token("scienthesis-lithub")
    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(tok, "WRONG-secret", algorithms=["HS256"], audience="scienthesis-lithub")


# ════════════════════════════════════════════════════════════════════
# LitHub client request shaping (B12 save-to-LitHub)
# ════════════════════════════════════════════════════════════════════
def _lithub_client(monkeypatch, handler, base="https://lithub.test"):
    monkeypatch.setattr(get_settings(), "SERVICE_TOKEN_SECRET", "test-secret")
    st.reset_cache_for_tests()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LitHubClient(http, settings=SimpleNamespace(LITHUB_BASE_URL=base)), http


async def test_lithub_save_paper_contract(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["svc"] = request.headers.get("X-Service-Token")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"paper_id": "P-123", "library_entry_id": "L-1"})

    client, http = _lithub_client(monkeypatch, handler)
    try:
        out = await client.internal_save_paper("sub-1", {"pmid": "39", "title": "X"})
    finally:
        await http.aclose()
    assert out["paper_id"] == "P-123"
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/api/v1/internal/library/save")
    assert seen["body"] == {"user_id": "sub-1", "item": {"pmid": "39", "title": "X"}}
    # service token present + correct audience/issuer
    decoded = jwt.decode(seen["svc"], "test-secret", algorithms=["HS256"], audience="scienthesis-lithub")
    assert decoded["iss"] == "litportal" and decoded["aud"] == "scienthesis-lithub"


async def test_lithub_membership_contract(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"in_library": True})

    client, http = _lithub_client(monkeypatch, handler)
    try:
        out = await client.internal_membership("sub-9", pmid="39111111")
    finally:
        await http.aclose()
    assert out["in_library"] is True
    assert seen["method"] == "GET"
    assert "/api/v1/internal/library/membership" in seen["url"]
    assert seen["params"]["user_id"] == "sub-9"
    assert seen["params"]["pmid"] == "39111111"


async def test_lithub_delete_contract(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        return httpx.Response(204)

    client, http = _lithub_client(monkeypatch, handler)
    try:
        out = await client.internal_delete_library("sub-1", doi="10.1/x")
    finally:
        await http.aclose()
    assert out == {}          # 204 → empty dict
    assert seen["method"] == "DELETE"


async def test_lithub_5xx_raises_upstream(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")
    client, http = _lithub_client(monkeypatch, handler)
    try:
        with pytest.raises(LitHubUpstreamError):
            await client.internal_save_paper("sub-1", {"pmid": "1"})
    finally:
        await http.aclose()


async def test_lithub_4xx_raises_client_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "no such user"})
    client, http = _lithub_client(monkeypatch, handler)
    try:
        with pytest.raises(LitHubClientError):
            await client.internal_membership("sub-1", pmid="1")
    finally:
        await http.aclose()


async def test_lithub_unconfigured_base_raises(monkeypatch):
    monkeypatch.setattr(get_settings(), "SERVICE_TOKEN_SECRET", "test-secret")
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    client = LitHubClient(http, settings=SimpleNamespace(LITHUB_BASE_URL=""))
    try:
        with pytest.raises(LitHubUpstreamError):
            await client.internal_save_paper("sub-1", {"pmid": "1"})
    finally:
        await http.aclose()
