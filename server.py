"""FastAPI app: ClinEvidence Search backend (Phase 1).

Mounts all /api routers, ensures Mongo indexes, seeds source_configs and
test accounts on startup. NO LLM, NO external calls. PHI screen is
deterministic & runs before any session is persisted.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load env BEFORE any other import that may consume os.environ.
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from starlette.middleware.cors import CORSMiddleware  # noqa: E402

from core.config import settings  # noqa: E402
from core.db import Collections, close_client, ensure_indexes, get_db  # noqa: E402
from core.security import hash_password  # noqa: E402
from models.source_configs import SourceConfigDoc  # noqa: E402
from models.guideline_sources import GuidelineSourceDoc  # noqa: E402
from models.users import UserDoc  # noqa: E402
from routers import (  # noqa: E402
    access_locations,
    admin,
    ai_surfaced,
    articles,
    audit,
    auth,
    benchmark_gold as benchmark_gold_router,
    cleanup,
    evidence_packets as evidence_packets_router,
    folders,
    full_text_import as full_text_import_router,
    handoff_downloads as handoff_downloads_router,
    institution_settings,
    jobs as jobs_router,
    known_article_lookup as known_article_lookup_router,
    library as library_router,
    litarchive,
    oa_fetcher as oa_fetcher_router,
    research_collections,
    search_sessions,
    source_configs,
    summary,
    uploads,
    users,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("clinevidence")


# ---- startup seeders ----
# Per-source seed config: (source_name, requires_api_key, enabled_default, status_message).
# Phase 2B: pubmed/europe_pmc/openalex/pubmed_guidelines are enabled by default
# so multi-source retrieval works out of the box. Guideline portals remain
# disabled until connectors are implemented.
# Phase 2C: dynamic status messages for link-out / OA-rescue sources (unpaywall,
# semantic_scholar_oa, crossref, openathens, openurl_resolver,
# semantic_scholar_institutional).
APPROVED_SOURCES: list[tuple[str, bool, bool, str]] = [
    ("pubmed", True, True, "__PUBMED_DYNAMIC__"),
    ("europe_pmc", False, True, "Open-access REST search; no key required."),
    ("openalex", False, True, "Enrichment-only metadata source."),
    ("pubmed_guidelines", False, True, "Served via PubMed guideline_focused queries."),
    ("uspstf", False, False, "Connector scaffolded; API/access not configured."),
    ("nice", False, False, "Connector scaffolded; API/access not configured."),
    ("who", False, False, "Connector scaffolded; API/access not configured."),
    ("cdc", False, False, "Connector scaffolded; API/access not configured."),
    ("specialty_society", True, False, "Connector scaffolded; API/access not configured."),
    ("pmc_oa", False, True, "PMC OA full-text/PDF lookup (via PMCID; no extra API call)."),
    ("europe_pmc_oa", False, True, "Europe PMC OA full-text lookup (uses Europe PMC metadata)."),
    ("unpaywall", False, False, "__UNPAYWALL_DYNAMIC__"),
    ("semantic_scholar_oa", False, True, "OA rescue + free full-text lookup. Optional API key for higher rate limits."),
    ("semantic_scholar_institutional", False, True, "Per-user institutional fallback toggle in Settings."),
    ("crossref", False, False, "__CROSSREF_DYNAMIC__"),
    ("openathens", False, True, "Link-out only; per-user customer domain in Settings."),
    ("openurl_resolver", False, True, "Link-out only; per-user resolver base URL in Settings."),
    ("libkey_getftr", True, False, "Connector scaffolded; integration deferred."),
    # Phase 4A OA fetcher.
    ("europe_pmc_oa_fetcher", True, False, "Europe PMC fullTextXML retrieval. License-gated."),
    ("pmc_oa_fetcher", True, False, "PMC OA Web Service + BioC retrieval. License-gated."),
    ("oa_license_gate", True, False, "CC0/BY/BY-SA allowed; noncommercial and unknown skipped by default."),
]

# Legacy status markers (Phase 1 / Phase 2A): a row whose status_message
# contains ANY of these is considered "never touched by an admin" and the
# seeder is allowed to flip enabled=true on the Phase 2B sources. An admin's
# deliberate disable (status_message will not contain these markers after a
# manual PATCH) is always preserved.
#
# Includes auto-generated status messages produced by the retrieval service
# (e.g. "retrieved N records across queries" / "all queries failed") — those
# are system-written, not admin-written, so the seeder may still migrate them.
_LEGACY_STATUS_MARKERS = (
    "Phase 1",
    "seeded; integration disabled",
    "retrieved ",          # set by literature_retrieval on success
    "all queries failed",  # set by literature_retrieval on full failure
    "Connector scaffolded; integration deferred",  # previous Phase 2B seed
)


def _pubmed_status_message() -> str:
    """Reflect NCBI key state in the seeded status_message."""
    has_key = bool((os.environ.get("NCBI_API_KEY") or os.environ.get("PUBMED_API_KEY") or "").strip())
    if has_key:
        return "Using NCBI API key (10 req/s)."
    return "No NCBI API key configured; using public rate limits (3 req/s)."


def _unpaywall_status_message() -> str:
    if (os.environ.get("UNPAYWALL_EMAIL") or "").strip():
        return "Unpaywall: license-aware OA lookup."
    return "Unpaywall not configured (set UNPAYWALL_EMAIL)."


def _crossref_status_message() -> str:
    if (os.environ.get("CROSSREF_MAILTO") or "").strip():
        return "Crossref: license + DOI link discovery."
    return "Crossref not configured (set CROSSREF_MAILTO)."


def _resolve_status_message(name: str, raw: str) -> str:
    if raw == "__PUBMED_DYNAMIC__":
        return _pubmed_status_message()
    if raw == "__UNPAYWALL_DYNAMIC__":
        return _unpaywall_status_message()
    if raw == "__CROSSREF_DYNAMIC__":
        return _crossref_status_message()
    return raw


# Phase K5.0 — Guideline Source Registry seed. Closed-set. Deterministic.
# ai_ingestion_allowed is FALSE for every entry. Only ``pubmed_guidelines``
# is ``live`` (served via the existing PubMed/NCBI E-utilities path, K5.1);
# the official-source connectors remain scaffolded/deferred (K5.2+).
GUIDELINE_SOURCE_REGISTRY: list[dict] = [
    {
        "source_name": "pubmed_guidelines",
        "organization": "NCBI / PubMed",
        "source_type": "pubmed_guideline_lane",
        "enabled": True,
        "requires_api_key": False,
        "connector_status": "live",
        "retrieval_allowed": True,
        "link_out_allowed": True,
        "display_snippet_allowed": "yes",
        "ai_ingestion_allowed": False,
        "license_notes": (
            "PubMed-indexed practice guidelines, consensus statements, and "
            "guideline-like reviews via NCBI E-utilities. Metadata + link-out. "
            "Not official-society-source connector coverage."
        ),
        "last_checked": "2026-06-08",
    },
    {
        "source_name": "uspstf",
        "organization": "USPSTF",
        "source_type": "official_guideline_api",
        "enabled": False,
        "requires_api_key": True,
        "connector_status": "scaffolded",
        "retrieval_allowed": False,
        "link_out_allowed": True,
        "display_snippet_allowed": "verify",
        "ai_ingestion_allowed": False,
        "license_notes": (
            "USPSTF Prevention TaskForce API. Verify API terms, attribution, "
            "and AI-ingestion limits before activation. Deferred (K5.2)."
        ),
        "last_checked": "2026-06-08",
    },
    {
        "source_name": "cdc_hhs",
        "organization": "CDC / HHS",
        "source_type": "syndication_api",
        "enabled": False,
        "requires_api_key": False,
        "connector_status": "scaffolded",
        "retrieval_allowed": False,
        "link_out_allowed": True,
        "display_snippet_allowed": "verify",
        "ai_ingestion_allowed": False,
        "license_notes": (
            "CDC/HHS syndication or API content. Metadata/link-out first; "
            "ingestion only if terms allow. Deferred (K5.2)."
        ),
        "last_checked": "2026-06-08",
    },
    {
        "source_name": "nice",
        "organization": "NICE",
        "source_type": "syndication_api",
        "enabled": False,
        "requires_api_key": True,
        "connector_status": "scaffolded",
        "retrieval_allowed": False,
        "link_out_allowed": True,
        "display_snippet_allowed": "verify",
        "ai_ingestion_allowed": False,
        "license_notes": (
            "NICE Syndication API. Requires accepted application / API key / "
            "license; ingestion only if license permits. Deferred (K5.2)."
        ),
        "last_checked": "2026-06-08",
    },
    {
        "source_name": "who",
        "organization": "WHO",
        "source_type": "source_registry",
        "enabled": False,
        "requires_api_key": False,
        "connector_status": "scaffolded",
        "retrieval_allowed": False,
        "link_out_allowed": True,
        "display_snippet_allowed": "verify",
        "ai_ingestion_allowed": False,
        "license_notes": (
            "WHO guideline / SMART metadata. Source registry first; ingestion "
            "only with explicit rights. Deferred (K5.2)."
        ),
        "last_checked": "2026-06-08",
    },
    {
        "source_name": "specialty_society",
        "organization": "Specialty societies (AHA-ACC, ESC, IDSA, ATS, SCCM, ACEP, etc.)",
        "source_type": "source_registry",
        "enabled": False,
        "requires_api_key": True,
        "connector_status": "scaffolded",
        "retrieval_allowed": False,
        "link_out_allowed": True,
        "display_snippet_allowed": "verify",
        "ai_ingestion_allowed": False,
        "license_notes": (
            "Specialty-society guideline sources. Start as source registry / "
            "link-out; no ingestion unless permission / license. Deferred (K5.2)."
        ),
        "last_checked": "2026-06-08",
    },
    {
        "source_name": "myhealthfinder_hhs",
        "organization": "HHS / ODPHP (MyHealthfinder)",
        "source_type": "official_guideline_api",
        "enabled": True,
        "requires_api_key": False,
        "connector_status": "live",
        "retrieval_allowed": True,
        "link_out_allowed": True,
        "display_snippet_allowed": "yes",
        "ai_ingestion_allowed": False,
        "license_notes": (
            "HHS/ODPHP MyHealthfinder API v3 (odphp.health.gov). Public-domain "
            "US government work; NO API key. Evidence-based prevention/screening "
            "recommendation topics. Metadata + link-out ONLY; NO full-text / NO "
            "AI ingestion. K5.2A first no-key official connector."
        ),
        "last_checked": "2026-06-09",
    },
    {
        "source_name": "cochrane_pubmed_indexed",
        "organization": "Cochrane (PubMed-indexed CDSR)",
        "source_type": "systematic_review_bibliographic_lane",
        "enabled": True,
        "requires_api_key": False,
        "connector_status": "live_pubmed_indexed",
        "retrieval_allowed": True,
        "link_out_allowed": True,
        "display_snippet_allowed": "verify",
        "ai_ingestion_allowed": False,
        "license_notes": (
            "PubMed-indexed Cochrane systematic reviews (journal 'Cochrane "
            "Database Syst Rev') via NCBI E-utilities. Bibliographic metadata "
            "+ link-out ONLY. NOT the Cochrane Library / Wiley full-text API; "
            "NO scraping; NO full-text ingestion; NO AI ingestion. K5.2C-0 "
            "preview lane."
        ),
        "last_checked": "2026-06-11",
    },
]


async def _seed_guideline_sources() -> None:
    """Idempotent seeder for the K5.0 guideline source registry.

    Insert-if-missing only. NEVER overwrites an existing row (so an admin's
    deliberate edits are preserved). ``ai_ingestion_allowed`` is False for
    every seeded entry.
    """
    db = get_db()
    coll = db[Collections.GUIDELINE_SOURCES]
    inserted = 0
    for entry in GUIDELINE_SOURCE_REGISTRY:
        existing = await coll.find_one(
            {"source_name": entry["source_name"]}, {"_id": 0, "id": 1},
        )
        if existing is not None:
            continue
        doc = GuidelineSourceDoc(**entry).model_dump()
        # Defensive: the model default already forces this False, but never
        # allow a seed entry to enable AI ingestion.
        doc["ai_ingestion_allowed"] = False
        for k in ("created_at", "updated_at"):
            doc[k] = doc[k].astimezone(timezone.utc)
        await coll.insert_one(doc)
        inserted += 1
    logger.info(
        "guideline_sources seeded (%d sources, inserted=%d)",
        len(GUIDELINE_SOURCE_REGISTRY), inserted,
    )


async def _seed_source_configs() -> None:
    """Idempotent seeder for source_configs.

    Rules:
    - Missing row → insert with the configured (enabled_default, status_message).
    - Existing row whose status_message still contains a legacy marker
      (Phase 1 / "seeded; integration disabled") → ONE-SHOT migration to the
      new defaults: set enabled=enabled_default and status_message=current.
      This brings the older seed forward to Phase 2B defaults exactly once.
    - Existing row WITHOUT a legacy marker → leave alone. An admin's
      deliberate enable/disable is never overwritten.
    """
    db = get_db()
    coll = db[Collections.SOURCE_CONFIGS]
    now = datetime.now(timezone.utc)
    migrated = 0
    inserted = 0
    for name, requires_api_key, enabled_default, status_message in APPROVED_SOURCES:
        status_message = _resolve_status_message(name, status_message)
        existing = await coll.find_one(
            {"source_name": name},
            {"_id": 0, "id": 1, "enabled": 1, "status_message": 1},
        )
        if existing is None:
            doc = SourceConfigDoc(
                source_name=name,
                enabled=enabled_default,
                requires_api_key=requires_api_key,
                status_message=status_message,
            ).model_dump()
            for k in ("created_at", "updated_at"):
                doc[k] = doc[k].astimezone(timezone.utc)
            await coll.insert_one(doc)
            inserted += 1
            continue
        existing_msg = existing.get("status_message") or ""
        if any(marker in existing_msg for marker in _LEGACY_STATUS_MARKERS):
            # One-shot migration: never overwrites an admin-set status_message.
            await coll.update_one(
                {"source_name": name},
                {"$set": {
                    "enabled": enabled_default,
                    "status_message": status_message,
                    "requires_api_key": requires_api_key,
                    "updated_at": now,
                }},
            )
            migrated += 1
    logger.info(
        "source_configs seeded (%d sources, inserted=%d, migrated=%d)",
        len(APPROVED_SOURCES), inserted, migrated,
    )
    # Phase 5D — dev/preview override: force-re-enable PubMed if an admin
    # previously disabled it. Honored ONLY when APP_ENV != production so
    # production deployments still require an explicit admin opt-in.
    # Closed-set log keys for the tester to verify on every startup:
    #   phase5d_dev_force_enable_pubmed_hook_attempted
    #   phase5d_dev_force_enable_pubmed_hook_succeeded
    #   phase5d_dev_force_enable_pubmed_hook_failed_reason=<closed_set>
    if settings.DEV_FORCE_ENABLE_PUBMED:
        logger.info(
            "phase5d_dev_force_enable_pubmed_hook_attempted "
            "app_env=%s is_production=%s",
            settings.APP_ENV, settings.IS_PRODUCTION,
        )
        if settings.IS_PRODUCTION:
            logger.warning(
                "phase5d_dev_force_enable_pubmed_hook_failed_reason="
                "is_production_env",
            )
        else:
            try:
                result = await coll.update_one(
                    {"source_name": "pubmed"},
                    {
                        "$set": {
                            "enabled": True,
                            "updated_at": now,
                        }
                    },
                )
                # Verify the row state AFTER the write.
                verify = await coll.find_one(
                    {"source_name": "pubmed"},
                    {"_id": 0, "enabled": 1},
                )
                enabled_after = bool(verify and verify.get("enabled"))
                if result.matched_count == 0:
                    logger.warning(
                        "phase5d_dev_force_enable_pubmed_hook_failed_reason="
                        "pubmed_row_missing",
                    )
                elif not enabled_after:
                    logger.warning(
                        "phase5d_dev_force_enable_pubmed_hook_failed_reason="
                        "post_write_verify_still_disabled",
                    )
                else:
                    logger.warning(
                        "phase5d_dev_force_enable_pubmed_hook_succeeded "
                        "matched=%d modified=%d enabled_after=true",
                        result.matched_count, result.modified_count,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "phase5d_dev_force_enable_pubmed_hook_failed_reason="
                    "write_exception exc=%s", type(exc).__name__,
                )
    # Phase 5D — startup warning when PubMed is disabled but clinical_queries
    # are required. Logs (does NOT fail) — saves admins from chasing
    # "why didn't clinical_queries emit?" later. Closed-set log key.
    if settings.PHASE5D_CLINICAL_QUERIES_REQUIRED:
        pm = await coll.find_one(
            {"source_name": "pubmed"}, {"_id": 0, "enabled": 1},
        )
        if pm is not None and not pm.get("enabled", False):
            logger.warning(
                "phase5d_pubmed_disabled_clinical_queries_blocked: "
                "PubMed source is disabled but "
                "PHASE5D_CLINICAL_QUERIES_REQUIRED=true. The "
                "clinical_queries family lives on PubMed only and will "
                "not emit any queries until PubMed is enabled via the "
                "admin UI or DEV_FORCE_ENABLE_PUBMED=true (non-prod only)."
            )


async def _seed_user_if_missing(
    email: str,
    password: str,
    name: str,
    role: str,
    first_use_acknowledged: bool,
) -> None:
    # DEV-ONLY seed accounts. Do not expose in production UI.
    db = get_db()
    existing = await db[Collections.USERS].find_one({"email": email}, {"_id": 0, "id": 1})
    if existing:
        return
    user = UserDoc(
        email=email,
        name=name,
        hashed_password=hash_password(password),
        role=role,  # type: ignore[arg-type]
        first_use_acknowledged=first_use_acknowledged,
    )
    doc = user.model_dump()
    for k in ("created_at", "updated_at"):
        doc[k] = doc[k].astimezone(timezone.utc)
    await db[Collections.USERS].insert_one(doc)
    logger.info("seeded user: %s (role=%s)", email, role)


async def _ensure_all_collections_exist() -> None:
    """Touch every collection so they exist (helpful for ops + tests)."""
    db = get_db()
    for name in Collections.ALL:
        # `count_documents` is cheap and ensures the collection becomes visible.
        await db[name].count_documents({})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase K3B Wave 1 Step 2.0.1 — fail-closed production-active
    # guardrail. Runs BEFORE the try/except wrapping the rest of
    # startup so any RuntimeError propagates and crashes startup.
    # The guard is a no-op on dev / staging and on production when
    # mode is anything other than ``"active"``. See
    # ``services/cap_enforcement_guard.verify_startup_fingerprint``.
    from services.cap_enforcement_guard import (
        verify_startup_fingerprint,
    )
    verify_startup_fingerprint()
    # Phase M3-preflight Step 1 — fail-closed production-active
    # guardrail for the future M3 active semantic-repair phase. The
    # default config (enabled=false, kill_switch=true) makes this a
    # no-op; on production env it ONLY raises if an operator has set
    # an unsafe flag combination. See
    # ``services/semantic_repair_active_guard``.
    from services.semantic_repair_active_guard import (
        verify_semantic_repair_active_startup,
        verify_m3_1_benchmark_startup,
    )
    verify_semantic_repair_active_startup()
    # Phase M3.1 — preview/dev-only A/B retrieval benchmark guard.
    verify_m3_1_benchmark_startup()
    # Phase M4.1 — preview/dev-only active M4 promotion guard. Default
    # config (enabled=false, kill_switch=true) is a no-op; on production
    # it ONLY raises if an operator set an unsafe flag combination.
    from services.m4_active_guard import verify_m4_active_startup
    verify_m4_active_startup()

    # Scienthesis integration — created AFTER the fail-closed guards (which do
    # no I/O) so a guard that aborts startup can never leak an open HTTP client
    # or a dangling task. Shared async HTTP client (Identity JWKS verification)
    # + the LitHub dual-write outbox sweeper (a no-op unless
    # LITPORTAL_DUAL_WRITE_LITHUB is enabled). Torn down after yield.
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    from services.lithub_sync import outbox_sweeper_loop
    app.state.lithub_sweeper_stop = asyncio.Event()
    app.state.lithub_sweeper_task = asyncio.create_task(
        outbox_sweeper_loop(app.state.lithub_sweeper_stop),
    )
    try:
        # Phase 6F.2 closeout — Atlas-resilient index bootstrap. The
        # ``ensure_indexes`` function is itself idempotent + tolerant
        # to the closed-set ``OperationFailure`` codes that occur
        # under concurrent worker boots, but we wrap the call here so
        # that ANY unexpected failure logs but never prevents the
        # FastAPI app from starting (HTTP must come up even if
        # indexes had a transient hiccup; PHI guard + safety paths do
        # not depend on indexes — only on collection existence).
        try:
            await ensure_indexes()
        except Exception:  # noqa: BLE001
            logger.exception(
                "ensure_indexes failed during startup — app will continue. "
                "Indexes may need manual reconciliation; see core/db.py "
                "for the closed-set of tolerated OperationFailure codes."
            )
        await _seed_source_configs()
        # Phase K5.0 — seed the guideline source registry (idempotent).
        await _seed_guideline_sources()
        # Seed test users only in non-production AND when SEED_TEST_USERS=true.
        if settings.SEED_TEST_USERS and not settings.IS_PRODUCTION:
            await _seed_user_if_missing(
                settings.SEED_CLINICIAN_EMAIL,
                settings.SEED_CLINICIAN_PASSWORD,
                "Test Clinician",
                "user",
                first_use_acknowledged=False,
            )
            await _seed_user_if_missing(
                settings.SEED_ADMIN_EMAIL,
                settings.SEED_ADMIN_PASSWORD,
                "Test Admin",
                "admin",
                first_use_acknowledged=True,
            )
        else:
            logger.info(
                "test-user seeding skipped (APP_ENV=%s, SEED_TEST_USERS=%s)",
                settings.APP_ENV,
                settings.SEED_TEST_USERS,
            )
        await _ensure_all_collections_exist()
        # Phase R1A — seed the 4 closed-set gold-comparator
        # benchmark cases (Part 5). Idempotent; safe to run on
        # every startup. NO PHI, NO LLM. NO clinician-visible
        # effect.
        try:
            from services.r1a_gold_comparator_cases import (
                seed_r1a_gold_comparator_cases,
            )
            await seed_r1a_gold_comparator_cases()
        except Exception:  # noqa: BLE001
            logger.exception(
                "phaseR1A gold_comparator_cases seed failed (non-fatal)",
            )
        logger.info("ClinEvidence Search startup complete")
        # R1D-hybrid Step 1 — env observability for the operator
        # re-smoke. Truthy-only for the secret; literal for the
        # non-secret polite-pool mailto.
        _pubmed_set = bool(
            (os.environ.get("PUBMED_API_KEY") or os.environ.get("NCBI_API_KEY") or "").strip()
        )
        logger.info(
            "r1d_step1_env_check PUBMED_API_KEY: %s | CROSSREF_MAILTO: %s",
            "set" if _pubmed_set else "unset",
            (os.environ.get("CROSSREF_MAILTO") or "").strip() or "unset",
        )
    except Exception:
        logger.exception("startup failed")
    yield
    # ---- shutdown ----
    try:
        app.state.lithub_sweeper_stop.set()
        await asyncio.wait_for(app.state.lithub_sweeper_task, timeout=5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        app.state.lithub_sweeper_task.cancel()
    except Exception:  # noqa: BLE001
        logger.exception("lithub sweeper shutdown failed (non-fatal)")
    try:
        await app.state.http_client.aclose()
    except Exception:  # noqa: BLE001
        logger.exception("http client close failed (non-fatal)")
    await close_client()


app = FastAPI(
    title="ClinEvidence Search API",
    description=(
        "Clinician-facing evidence search & synthesis backend (Phase 1). "
        "NOT a diagnostic or treatment-recommendation tool. NO PHI permitted."
    ),
    version="0.1.0",
    lifespan=lifespan,
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

def _resolve_cors_origins() -> list[str]:
    """Env-driven, credentials-safe CORS origins.

    Uses comma-separated ``CORS_ORIGINS`` from the environment when it names
    explicit origins; drops any "*" (invalid with allow_credentials=True);
    falls back to the operator-approved explicit allowlist. Never returns
    a bare "*" and never uses a broad subdomain regex.
    """
    operator_allowlist = [
        # Explicit, credentials-safe origins only (allow_credentials=True forbids "*").
        "https://med-evidence-qa.preview.emergentagent.com",       # active preview (frontend/.env)
        "https://clinician-answers.emergent.host",                 # CANONICAL PRODUCTION (current)
        "https://clinician-answers.preview.emergentagent.com",     # preview for current app
        "https://clinevidence-core.emergent.host",                 # legacy/stale production (kept until reattached)
        "https://clinevidence-core.preview.emergentagent.com",     # legacy preview
    ]
    raw = (os.environ.get("CORS_ORIGINS") or "").strip()
    env_origins = [
        o.strip() for o in raw.split(",")
        if o.strip() and o.strip() != "*"
    ]
    # De-dupe while preserving order; env origins first, then allowlist.
    seen: set[str] = set()
    out: list[str] = []
    for o in env_origins + operator_allowlist:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    # Phase Release Checkpoint — env-driven, credentials-safe CORS.
    # Reads ``CORS_ORIGINS`` (comma-separated) from the environment so the
    # deploy pipeline can inject the production origin. Because
    # ``allow_credentials=True`` forbids "*", any "*" entry is dropped and
    # we fall back to the explicit operator-approved allowlist (canonical
    # preview + production + the currently-active preview origin). We never
    # emit a bare "*" and never use a broad subdomain regex.
    allow_origins=_resolve_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- meta endpoints ----
@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/")
async def root() -> dict:
    return {
        "service": "ClinEvidence Search",
        "phase": 1,
        "status": "ok",
        "disclaimer": (
            "This API is for clinician evidence review. It does not provide diagnostic "
            "or treatment recommendations and is not a substitute for clinical judgment. "
            "Do not submit protected health information (PHI)."
        ),
    }


# ---- routers ----
_API_ROUTERS = [
    auth.router,
    users.router,
    search_sessions.router,
    folders.router,
    articles.router,
    access_locations.router,
    institution_settings.router,
    uploads.router,
    summary.router,
    oa_fetcher_router.router,
    jobs_router.router,
    admin.router,
    audit.router,
    source_configs.router,
    cleanup.router,
    library_router.router,
    research_collections.router,
    litarchive.router,
    ai_surfaced.router,
    benchmark_gold_router.router,
    known_article_lookup_router.router,
    full_text_import_router.router,
    evidence_packets_router.clinician_router,
    evidence_packets_router.admin_router,
    handoff_downloads_router.router,
]
for r in _API_ROUTERS:
    app.include_router(r, prefix="/api")

# B10 — LitPulse module mount: also expose every router under
# LITPORTAL_MODULE_PREFIX (e.g. /api/litportal/*) so LitPortal runs both
# standalone and as a module inside the LitPulse platform.
_module_prefix = (settings.LITPORTAL_MODULE_PREFIX or "").rstrip("/")
if (
    settings.LITPORTAL_MODULE_MOUNT_ENABLED
    and _module_prefix
    and _module_prefix != "/api"
):
    for r in _API_ROUTERS:
        app.include_router(r, prefix=_module_prefix)

    @app.get(f"{_module_prefix}/health")
    async def module_health() -> dict:
        return {"status": "ok", "module": "litportal"}
