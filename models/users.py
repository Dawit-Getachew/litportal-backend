"""User-related Pydantic schemas. Strict password policy enforced at boundary."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from .common import new_id, utcnow


# Min 12 chars, ≥1 letter, ≥1 digit, ≥1 special.
_SPECIAL_CHARS = r"!@#$%^&*()_+\-=\[\]{};:'\",.<>?/\\|"
_PASSWORD_RE = re.compile(
    rf"^(?=.*[A-Za-z])(?=.*\d)(?=.*[{_SPECIAL_CHARS}]).{{12,}}$"
)


def _validate_password(value: str) -> str:
    if not isinstance(value, str) or not _PASSWORD_RE.match(value):
        raise ValueError(
            "Password must be at least 12 characters and include at least one letter, "
            "one number, and one special character "
            "(!@#$%^&*()_+-=[]{};:'\",.<>?/\\|)."
        )
    return value


# ----- DB document -----
class UserDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    email: EmailStr
    name: str
    # Optional: identity-provisioned shadow users carry no local password.
    hashed_password: str = ""
    role: Literal["user", "admin"] = "user"
    # Scienthesis Identity subject (UUID). Set when the user was provisioned
    # via the central Identity service; equals ``id`` for identity-first users.
    # Stays None for native (OTP/password) accounts until they are linked.
    identity_id: Optional[str] = None
    # LitPulse legacy user id — set only for users provisioned via the optional
    # LitPulse legacy bridge during cutover.
    litpulse_user_id: Optional[str] = None
    first_use_acknowledged: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ----- API request/response -----
class UserSignup(BaseModel):
    email: EmailStr
    password: str
    name: str = Field(min_length=1, max_length=120)

    @field_validator("password")
    @classmethod
    def _pw(cls, v: str) -> str:
        return _validate_password(v)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    name: str
    role: Literal["user", "admin"]
    first_use_acknowledged: bool
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int  # seconds
    user: UserPublic


class UserProfileUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)


# ----- OTP (passwordless sign-in via central Identity) -----
class OTPRequest(BaseModel):
    email: EmailStr


class OTPVerify(BaseModel):
    email: EmailStr
    code: str = Field(min_length=4, max_length=12)


# ----- Institution settings -----
InstitutionMethod = Literal["openathens", "openurl", "semantic_scholar", "none"]


class InstitutionSettingsDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    method: InstitutionMethod = "none"
    openathens_customer_domain: Optional[str] = None
    openurl_resolver_base_url: Optional[str] = None
    semantic_scholar_fallback_enabled: bool = False
    is_default: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class InstitutionSettingsUpsert(BaseModel):
    """Phase 2C: hard validation. NEVER accept credentials of any kind.

    The Settings page must reject anything that smells like a username,
    password, secret, token, API key, or auth header. ClinEvidence is
    link-out-only — institutional credentials are NEVER stored or accepted.
    """

    method: InstitutionMethod = "none"
    openathens_customer_domain: Optional[str] = None
    openurl_resolver_base_url: Optional[str] = None
    semantic_scholar_fallback_enabled: bool = False
    is_default: bool = True

    @field_validator("openathens_customer_domain")
    @classmethod
    def _openathens_domain(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        v = v.strip()
        if not (1 <= len(v) <= 100):
            raise ValueError("openathens_customer_domain must be 1–100 characters.")
        if v.startswith("http://") or v.startswith("https://"):
            raise ValueError(
                "openathens_customer_domain is just the customer subdomain, "
                "not a URL — do not include http:// or https://."
            )
        if not re.match(r"^[a-z0-9][a-z0-9.\-]{0,99}$", v):
            raise ValueError(
                "openathens_customer_domain may only contain lowercase letters, "
                "digits, hyphens, and periods (no @, slashes, spaces, or query strings)."
            )
        return v

    @field_validator("openurl_resolver_base_url")
    @classmethod
    def _openurl_resolver(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        v = v.strip()
        if len(v) > 500:
            raise ValueError("openurl_resolver_base_url must be ≤500 characters.")
        if "?" in v or "#" in v:
            raise ValueError(
                "openurl_resolver_base_url must not contain query parameters or "
                "fragments — the resolver builds query params itself."
            )
        from urllib.parse import urlparse
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("openurl_resolver_base_url must start with https:// (or http://localhost).")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1"):
            raise ValueError("openurl_resolver_base_url must use https:// (http allowed only for localhost).")
        if not parsed.netloc:
            raise ValueError("openurl_resolver_base_url is missing a hostname.")
        return v
