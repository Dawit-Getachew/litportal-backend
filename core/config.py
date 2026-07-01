"""Centralized environment & configuration loading for ClinEvidence Search backend.

All secrets MUST come from environment variables. Loaded once at import time.
"""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get_or_generate(env_name: str) -> str:
    val = os.environ.get(env_name, "").strip()
    if not val:
        val = secrets.token_hex(32)
        os.environ[env_name] = val
    return val


def _bool(env_name: str, default: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(env_name: str, default: int) -> int:
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _float(env_name: str, default: float) -> float:
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


class Settings:
    # Database
    MONGO_URL: str = os.environ["MONGO_URL"]
    DB_NAME: str = os.environ["DB_NAME"]

    # CORS / runtime env
    CORS_ORIGINS: list[str] = os.environ.get("CORS_ORIGINS", "*").split(",")
    APP_ENV: str = os.environ.get("APP_ENV", "development").lower()
    IS_PRODUCTION: bool = APP_ENV == "production"

    # JWT (separate secrets for access + refresh)
    JWT_ACCESS_SECRET: str = _get_or_generate("JWT_ACCESS_SECRET")
    JWT_REFRESH_SECRET: str = _get_or_generate("JWT_REFRESH_SECRET")
    JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TTL_MINUTES: int = int(os.environ.get("JWT_ACCESS_TTL_MINUTES", "60"))
    JWT_REFRESH_TTL_DAYS: int = int(os.environ.get("JWT_REFRESH_TTL_DAYS", "7"))

    # Cookie attributes (refresh token).
    # NOTE: In production deployments, set COOKIE_SECURE=true (HTTPS required).
    COOKIE_SECURE: bool = _bool("COOKIE_SECURE", False)
    COOKIE_SAMESITE: str = os.environ.get("COOKIE_SAMESITE", "lax").lower()
    REFRESH_COOKIE_NAME: str = "ces_refresh_token"
    REFRESH_COOKIE_PATH: str = "/api/auth"

    # Login brute-force protection
    LOGIN_RATE_LIMIT_MAX: int = int(os.environ.get("LOGIN_RATE_LIMIT_MAX", "5"))
    LOGIN_RATE_LIMIT_WINDOW_MIN: int = int(os.environ.get("LOGIN_RATE_LIMIT_WINDOW_MIN", "15"))

    # Uploads
    MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "25"))
    MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

    # Seed accounts (DEV-ONLY — gated by SEED_TEST_USERS + APP_ENV)
    SEED_TEST_USERS: bool = _bool("SEED_TEST_USERS", True)
    SEED_CLINICIAN_EMAIL: str = os.environ.get("SEED_CLINICIAN_EMAIL", "clinician@test.com")
    SEED_CLINICIAN_PASSWORD: str = os.environ.get("SEED_CLINICIAN_PASSWORD", "ClinicianTest123!")
    SEED_ADMIN_EMAIL: str = os.environ.get("SEED_ADMIN_EMAIL", "admin@test.com")
    SEED_ADMIN_PASSWORD: str = os.environ.get("SEED_ADMIN_PASSWORD", "AdminTest123!")

    # Retention
    SESSION_TTL_DAYS: int = 30

    # PHI tuning
    PHI_MIN_DIGIT_RUN: int = 7

    # Phase 5D — clinical_queries family lives on PubMed only. If PubMed is
    # disabled in the DB AND this env flag is true, the startup logger
    # emits a warning so admins know clinical_queries can't fire.
    PHASE5D_CLINICAL_QUERIES_REQUIRED: bool = _bool(
        "PHASE5D_CLINICAL_QUERIES_REQUIRED", False,
    )
    # Phase 5D — for dev/preview envs ONLY (gated by APP_ENV != production).
    # When set to true, the source_configs seeder will FORCE-RE-ENABLE the
    # PubMed source on startup even if an admin previously disabled it.
    # Default false. Never auto-enables in production.
    DEV_FORCE_ENABLE_PUBMED: bool = _bool("DEV_FORCE_ENABLE_PUBMED", False)

    # Placeholder integration keys (Phase 1 — documented, all empty)
    # PUBMED_API_KEY — used by future PubMed E-utilities integration
    # EUROPE_PMC_API_KEY — used by future Europe PMC REST API integration
    # OPENALEX_EMAIL — polite-pool identifier for OpenAlex
    # UNPAYWALL_EMAIL — polite-pool identifier for Unpaywall
    # SEMANTIC_SCHOLAR_API_KEY — used by future Semantic Scholar API
    # CROSSREF_EMAIL — polite-pool identifier for Crossref
    PUBMED_API_KEY: str = os.environ.get("PUBMED_API_KEY", "")
    EUROPE_PMC_API_KEY: str = os.environ.get("EUROPE_PMC_API_KEY", "")
    OPENALEX_EMAIL: str = os.environ.get("OPENALEX_EMAIL", "")

    # Phase M3.1 — preview/dev-only admin A/B retrieval benchmark.
    # Default OFF in every env. Production fail-closed guard is in
    # ``services/semantic_repair_active_guard.verify_m3_1_benchmark_startup``.
    M3_1_BENCHMARK_ENABLED: bool = (
        (os.environ.get("M3_1_BENCHMARK_ENABLED", "false") or "").strip().lower() == "true"
    )
    UNPAYWALL_EMAIL: str = os.environ.get("UNPAYWALL_EMAIL", "")
    SEMANTIC_SCHOLAR_API_KEY: str = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    CROSSREF_EMAIL: str = os.environ.get("CROSSREF_EMAIL", "")

    # -------------------------------------------------------------------
    # Scienthesis platform integration (Identity + LitHub).
    #
    # Both master flags default OFF: with them unset the backend behaves
    # exactly like the standalone Emergent backend (native HS256 auth +
    # local Mongo library). See INTEGRATION_PLAN.md.
    # -------------------------------------------------------------------
    # Master feature flags.
    LITPORTAL_USE_IDENTITY: bool = _bool("LITPORTAL_USE_IDENTITY", False)
    LITPORTAL_DUAL_WRITE_LITHUB: bool = _bool("LITPORTAL_DUAL_WRITE_LITHUB", False)

    # Identity service (scienthesis-identity). Tokens are RS256, verified
    # offline against the JWKS. IDENTITY_JWT_AUDIENCE is this app's own
    # audience and must be one of Identity's JWT_AUDIENCES.
    IDENTITY_BASE_URL: str = os.environ.get("IDENTITY_BASE_URL", "")
    IDENTITY_JWKS_URL: str = os.environ.get("IDENTITY_JWKS_URL", "")
    IDENTITY_JWT_ISSUER: str = os.environ.get("IDENTITY_JWT_ISSUER", "scienthesis-identity")
    IDENTITY_JWT_AUDIENCE: str = os.environ.get("IDENTITY_JWT_AUDIENCE", "litportal")
    JWKS_CACHE_TTL_SECONDS: int = int(os.environ.get("JWKS_CACHE_TTL_SECONDS", "300"))

    # LitHub central library (scienthesis-lithub).
    LITHUB_BASE_URL: str = os.environ.get("LITHUB_BASE_URL", "")

    # Shared HS256 secret signing the X-Service-Token header on internal
    # (/api/v1/internal/*) calls to Identity and LitHub. MUST be identical
    # across all Scienthesis services.
    SERVICE_TOKEN_SECRET: str = os.environ.get("SERVICE_TOKEN_SECRET", "")

    # Optional LitPulse legacy bridge (cutover only). When enabled,
    # get_current_user ALSO accepts HS256 tokens minted by the LitPulse backend
    # (signed with the shared LITPULSE_JWT_SECRET_KEY) so LitPulse users keep
    # working before LitPulse fully migrates to central Identity. NOTE: legacy
    # tokens carry LitPulse's own user id (not the Identity sub), so such users
    # are NOT mirrored to LitHub until they next authenticate via Identity.
    LITPULSE_JWT_ENABLED: bool = _bool("LITPULSE_JWT_ENABLED", False)
    LITPULSE_JWT_SECRET_KEY: str = os.environ.get("LITPULSE_JWT_SECRET_KEY", "")

    # B10 — LitPulse module mount. When enabled, every LitPortal router is ALSO
    # mounted under LITPORTAL_MODULE_PREFIX (default /api/litportal) so LitPortal
    # works both standalone (/api/*) and as a LitPulse module, without changing
    # the existing routes.
    LITPORTAL_MODULE_MOUNT_ENABLED: bool = _bool("LITPORTAL_MODULE_MOUNT_ENABLED", True)
    LITPORTAL_MODULE_PREFIX: str = os.environ.get("LITPORTAL_MODULE_PREFIX", "/api/litportal")

    # -------------------------------------------------------------------
    # AI agents (LitPortal 2.5-week scope). Three advisory sidecars; ALL
    # default OFF. They never call sources, select/exclude/reorder
    # deterministic results, or generate citations — they enrich the
    # parse, store understanding diagnostics, and surface advisory
    # candidates on a separate shelf. See agents/ package.
    # -------------------------------------------------------------------
    # Agent 1 — pre-parser query categorizer/enricher.
    AI_QUERY_CATEGORIZER_ENABLED: bool = _bool("AI_QUERY_CATEGORIZER_ENABLED", False)
    # Only override a weak deterministic type ("other") when AI is at least this confident.
    AI_QUERY_CATEGORIZER_MIN_CONFIDENCE: float = _float("AI_QUERY_CATEGORIZER_MIN_CONFIDENCE", 0.75)

    # Agent 2 — question understanding / ContextAugment (B3). Diagnostics only.
    AI_CONTEXT_AUGMENT_ENABLED: bool = _bool("AI_CONTEXT_AUGMENT_ENABLED", False)
    AI_CONTEXT_AUGMENT_PREVIEW_ENABLED: bool = _bool("AI_CONTEXT_AUGMENT_PREVIEW_ENABLED", False)

    # Agent 3 — evidence surfacing + ranker optimization (B4 / B4a).
    AI_EVIDENCE_SURFACING_ENABLED: bool = _bool("AI_EVIDENCE_SURFACING_ENABLED", False)
    AI_EVIDENCE_SURFACING_PREVIEW_VISIBLE: bool = _bool("AI_EVIDENCE_SURFACING_PREVIEW_VISIBLE", False)
    AI_EVIDENCE_SURFACING_USER_VISIBLE_ALLOWED: bool = _bool("AI_EVIDENCE_SURFACING_USER_VISIBLE_ALLOWED", False)
    AI_RANKER_OPTIMIZATION_ENABLED: bool = _bool("AI_RANKER_OPTIMIZATION_ENABLED", False)
    AI_RANKER_PARALLEL_SEARCH_ENABLED: bool = _bool("AI_RANKER_PARALLEL_SEARCH_ENABLED", False)
    MAX_AI_SURFACING_CANDIDATES: int = _int("MAX_AI_SURFACING_CANDIDATES", 30)
    MAX_RANKER_PARALLEL_AI_CANDIDATES: int = _int("MAX_RANKER_PARALLEL_AI_CANDIDATES", 20)
    MIN_AI_SURFACING_CONFIDENCE: float = _float("MIN_AI_SURFACING_CONFIDENCE", 0.65)
    MIN_RANKER_OPTIMIZATION_CONFIDENCE: float = _float("MIN_RANKER_OPTIMIZATION_CONFIDENCE", 0.65)

    # -------------------------------------------------------------------
    # B15 — single-paper full-text finder/import. Powers the LitPulse
    # "Full text import" button: given a PMID / DOI / title for ONE paper,
    # resolve it, run the OA-first access broker + OA full-text fetcher,
    # and return a ready single-article session the clinician can summarize
    # ("evidence chat") or save. Orchestrates existing services only; the
    # access broker (UNPAYWALL_EMAIL / CROSSREF_MAILTO) and OA fetcher
    # (OA_FETCHER_ENABLED + OA_ALLOW_* license flags) keep their own gates.
    # Default ON; flip off to hide the endpoint entirely.
    # -------------------------------------------------------------------
    FULL_TEXT_IMPORT_ENABLED: bool = _bool("FULL_TEXT_IMPORT_ENABLED", True)
    # Lowest match confidence accepted without an explicit user confirmation
    # (title-only lookups often score "low"; the caller may set
    # accept_low_confidence=true to import the top candidate anyway).
    FULL_TEXT_IMPORT_MIN_CONFIDENCE: str = os.environ.get(
        "FULL_TEXT_IMPORT_MIN_CONFIDENCE", "medium",
    ).strip().lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
