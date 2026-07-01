"""Mongo (Motor) async client + collection helpers.

Phase 6F.2 closeout — Atlas-resilient ``ensure_indexes``:

* Every ``create_index`` / ``drop_index`` call is wrapped in
  ``_safe_create_index`` / ``_safe_drop_index`` which tolerate the
  closed-set of ``OperationFailure`` codes that occur under
  concurrent worker boots against MongoDB Atlas
  (``IndexBuildAborted=276``, ``IndexNotFound=27``,
  ``NamespaceNotFound=26``, ``CannotCreateIndex=68``,
  ``IndexOptionsConflict=85``, ``IndexKeySpecsConflict=86``).
* A per-collection ``index_information()`` cache lets us
  **short-circuit** the drop+create cycle whenever an index with the
  same name already carries the desired keys + options — eliminating
  the churn that triggers Atlas concurrent-build conflicts.
* A module-level ``_INDEXES_ENSURED`` guard makes the function a
  no-op on second invocation in the same process (protects against
  re-entrant lifespan / health-probe restarts).
* Code 276 invocations retry up to ``MAX_INDEX_BOOTSTRAP_ATTEMPTS``
  times with exponential backoff (0.5s, 1.5s, 3.5s); other tolerated
  codes do NOT retry.
* Unexpected ``OperationFailure`` codes (e.g. 13 Unauthorized) are
  re-raised so misconfiguration stays visible. ``AutoReconnect`` /
  ``NetworkTimeout`` are logged + continued (transient).
* The application boot path NEVER crashes for an index-only problem:
  the lifespan hook in ``server.py`` ALSO catches and logs.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import AutoReconnect, NetworkTimeout, OperationFailure

from .config import settings

logger = logging.getLogger("clinevidence.db")

_client: AsyncIOMotorClient | None = None

# ---------------------------------------------------------------------------
# Atlas-resilient ensure_indexes — closed-set tolerated codes + idempotency
# ---------------------------------------------------------------------------

#: Closed-set OperationFailure codes that ``ensure_indexes`` tolerates.
#: Any code outside this set is re-raised so misconfiguration stays visible.
#:
#: * 26  NamespaceNotFound       — collection doesn't exist yet (drop noop)
#: * 27  IndexNotFound           — index already absent (drop noop)
#: * 68  CannotCreateIndex       — log warning + continue
#: * 85  IndexOptionsConflict    — existing index has same key but different
#:                                 options → caller handles drop+recreate
#: * 86  IndexKeySpecsConflict   — name collision with different keys → caller
#:                                 handles drop+recreate
#: * 276 IndexBuildAborted       — concurrent build aborted → log + continue
_TOLERATED_OP_FAILURE_CODES: frozenset[int] = frozenset({26, 27, 68, 85, 86, 276})

#: Retry budget for the specific case where a code-276 race fires. Other
#: tolerated codes are NOT retried.
MAX_INDEX_BOOTSTRAP_ATTEMPTS: int = 3
_INDEX_BUILD_BACKOFF_S: tuple[float, ...] = (0.5, 1.5, 3.5)

#: Per-process idempotency guard. Reset only by ``_reset_indexes_ensured_for_test``.
_INDEXES_ENSURED: bool = False


def _is_pytest_context() -> bool:
    """Phase 6D — detect pytest so we can relax unique index enforcement.

    Test fixtures intentionally re-insert canonical (pmid, pmcid, doi) and
    would fail with ``DuplicateKeyError`` if the unique indexes were active.
    Production / dev runs (where ``PYTEST_CURRENT_TEST`` is not set) keep
    the unique indexes for data integrity.
    """
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or "test" in (
        settings.DB_NAME or ""
    )


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGO_URL)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[settings.DB_NAME]


async def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


# Collection name constants — single source of truth.
class Collections:
    USERS = "users"
    SEARCH_SESSIONS = "search_sessions"
    SEARCH_QUERIES = "search_queries"
    ARTICLES = "articles"
    SEARCH_SESSION_ARTICLES = "search_session_articles"
    SUMMARIES = "summaries"
    JOBS = "jobs"
    FOLDERS = "folders"
    FOLDER_ARTICLES = "folder_articles"
    SOURCE_CONFIGS = "source_configs"
    AUDIT_EVENTS = "audit_events"
    USER_INSTITUTION_SETTINGS = "user_institution_settings"
    ARTICLE_ACCESS_LOCATIONS = "article_access_locations"
    UPLOADED_FULL_TEXT_ASSETS = "uploaded_full_text_assets"
    ARTICLE_TEXT_EXTRACTS = "article_text_extracts"
    # Phase 5C — persisted benchmark runs.
    BENCHMARK_RUNS = "benchmark_runs"
    # Phase 6D — canonical user library + saved-packet collections.
    USER_LIBRARY_ITEMS = "user_library_items"
    SAVED_EVIDENCE_PACKETS = "saved_evidence_packets"
    FOLDER_SUMMARIES = "folder_summaries"  # SCAFFOLDED for Task 10
    # Phase 6E — benchmark gold-set curation.
    BENCHMARK_GOLD_CANDIDATES = "benchmark_gold_candidates"
    BENCHMARK_CASE_OVERRIDES = "benchmark_case_overrides"
    # Phase M3.1 — preview-only admin A/B benchmark runs (TTL 30d).
    M3_1_BENCHMARK_RUNS = "m3_1_benchmark_runs"
    # Phase M3.2 — preview-only active semantic repair audit sidecar
    # (TTL 30d). Records every active-pipeline decision (eligible /
    # fallback / kill / K4A excluded) so an operator can adjudicate
    # adoption without flipping the production flag.
    M3_2_SESSION_DECISIONS = "m3_2_session_decisions"
    # Phase M3.2 — runtime feature-flag collection. Currently owns the
    # Mongo-backed kill-switch for M3.2 active semantic repair. Reads
    # are OR-combined with the env var (either source engaged = kill).
    FEATURE_FLAGS = "feature_flags"
    # Phase M4 — preview/dev-only Full-Question LLM ClinicalQuestionMap
    # interpreter shadow sidecar (TTL 30d). One row per pipeline run
    # that TRIGGERED the M4 interpreter. Records baseline CQM, bounded
    # LLM candidate, deterministic validation result, and the would-be
    # compiler projection. NEVER clinician-visible. No prompt/completion.
    M4_SHADOW_DECISIONS = "m4_shadow_decisions"
    # Phase R1A — admin-only shadow reranking sidecar (TTL 30d).
    # One row per benchmark replay OR per live session whose
    # detected shape is ``pathophysiology_mechanism`` or
    # ``procedure_complication_risk``. Stores side-by-side
    # current-active vs shadow-simulated top-N. Never clinician-
    # visible. Leak-scanned.
    R1A_SHADOW_DECISIONS = "r1a_shadow_decisions"
    # Phase R1A — admin-only curated gold-comparator benchmark
    # cases. Closed-set schema. Append-only. No PHI. Used by the
    # replay endpoint to seed the shadow simulator with a known
    # gold article target.
    R1A_GOLD_COMPARATOR_CASES = "r1a_gold_comparator_cases"
    # Phase R1B — admin-only shadow simulator for retrieval
    # recall + directness calibration. TTL 30d on ``expires_at``.
    # One row per ``POST /api/admin/r1b/replay/{case_id}`` invocation.
    # NEVER clinician-visible. Leak-scanned.
    R1B_SHADOW_DECISIONS = "r1b_shadow_decisions"
    # Phase R1C — admin-only shadow simulator for gold-recall +
    # anchor-specific retrieval calibration. TTL 30d on ``expires_at``.
    # One row per ``POST /api/admin/r1c/replay/{case_id}`` invocation.
    # NEVER clinician-visible. Leak-scanned.
    R1C_SHADOW_DECISIONS = "r1c_shadow_decisions"
    # Phase K4 — clinician-visible branch-aware evidence packets.
    # One packet per ``search_session_id`` (unique). Computed at
    # session terminal status (``awaiting_selection`` / ``complete``)
    # OR lazily on first GET. No TTL — packet lives the life of the
    # session. Leak-scanned at write time (no abstracts, no prompts,
    # no completions, no credentials, no PHI).
    EVIDENCE_PACKETS = "evidence_packets"

    # Phase K5.0 — official guideline source registry + permission model.
    # Deterministic, no-network. Holds the Northstar v3 §12.1 permission
    # flags (retrieval / link-out / display-snippet / ai-ingestion) per
    # source. ``ai_ingestion_allowed`` defaults FALSE for every source.
    GUIDELINE_SOURCES = "guideline_sources"

    # Scienthesis integration — reliable dual-write outbox for mirroring
    # library saves/deletes to the central LitHub service. Drained by a
    # background sweeper (see services/lithub_sync.py). Only written when
    # LITPORTAL_DUAL_WRITE_LITHUB is enabled.
    LITHUB_SYNC_OUTBOX = "lithub_sync_outbox"

    # AI agents (LitPortal 2.5-week scope) — advisory sidecar outputs.
    # Session-level diagnostics from the query categorizer + question
    # understanding agents (one row per agent run, keyed by session).
    AI_AGENT_DECISIONS = "ai_agent_decisions"
    # Per-article advisory candidates from the evidence-surfacing agent
    # (the "AI-surfaced" shelf; never merged into deterministic results).
    AI_SURFACED_ARTICLES = "ai_surfaced_articles"

    # Research collections — user-owned hierarchical collections of saved
    # papers (mirrored to LitHub when dual-write is enabled). See
    # models/research_collections.py + services/research_collection_service.py.
    RESEARCH_COLLECTIONS = "research_collections"
    RESEARCH_COLLECTION_ITEMS = "research_collection_items"

    ALL = (
        USERS,
        SEARCH_SESSIONS,
        SEARCH_QUERIES,
        ARTICLES,
        SEARCH_SESSION_ARTICLES,
        SUMMARIES,
        JOBS,
        FOLDERS,
        FOLDER_ARTICLES,
        SOURCE_CONFIGS,
        AUDIT_EVENTS,
        USER_INSTITUTION_SETTINGS,
        ARTICLE_ACCESS_LOCATIONS,
        UPLOADED_FULL_TEXT_ASSETS,
        ARTICLE_TEXT_EXTRACTS,
        USER_LIBRARY_ITEMS,
        SAVED_EVIDENCE_PACKETS,
        FOLDER_SUMMARIES,
        BENCHMARK_GOLD_CANDIDATES,
        BENCHMARK_CASE_OVERRIDES,
        EVIDENCE_PACKETS,
        GUIDELINE_SOURCES,
        LITHUB_SYNC_OUTBOX,
        AI_AGENT_DECISIONS,
        AI_SURFACED_ARTICLES,
        RESEARCH_COLLECTIONS,
        RESEARCH_COLLECTION_ITEMS,
    )


# ---------------------------------------------------------------------------
# Index spec helpers — used for short-circuit comparison and for inferring
# the default name MongoDB would assign when no ``name=`` kwarg is given.
# ---------------------------------------------------------------------------


def _normalize_keys(keys: Any) -> list[tuple[str, int]]:
    """Turn a Mongo ``keys`` spec into the ``[(field, direction), ...]`` form
    that ``index_information()`` returns."""
    if isinstance(keys, str):
        return [(keys, 1)]
    out: list[tuple[str, int]] = []
    for k in keys:
        if isinstance(k, tuple) or isinstance(k, list):
            out.append((str(k[0]), int(k[1])))
        else:
            out.append((str(k), 1))
    return out


def _infer_index_name(keys: Any) -> str:
    """Mirror MongoDB's default index-name algorithm
    (``field_1_field_-1...``)."""
    parts = [f"{f}_{d}" for f, d in _normalize_keys(keys)]
    return "_".join(parts)


def _index_spec_matches(existing: dict, keys: Any, opts: dict) -> bool:
    """Closed-set spec comparison — keys + ``unique`` + ``sparse`` +
    ``partialFilterExpression`` + ``expireAfterSeconds``."""
    desired_keys = _normalize_keys(keys)
    existing_keys_raw = existing.get("key") or []
    existing_keys: list[tuple[str, int]] = []
    for item in existing_keys_raw:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            existing_keys.append((str(item[0]), int(item[1])))
    if existing_keys != desired_keys:
        return False
    for opt_key in ("unique", "sparse"):
        if bool(existing.get(opt_key, False)) != bool(opts.get(opt_key, False)):
            return False
    if existing.get("partialFilterExpression") != opts.get("partialFilterExpression"):
        return False
    if existing.get("expireAfterSeconds") != opts.get("expireAfterSeconds"):
        return False
    return True


async def _get_index_info(coll, cache: dict[str, dict]) -> dict:
    """Cached ``index_information()`` per collection. Tolerates
    NamespaceNotFound (collection doesn't exist yet → empty dict)."""
    coll_name = getattr(coll, "name", id(coll))
    if coll_name in cache:
        return cache[coll_name]
    try:
        info = await coll.index_information()
    except OperationFailure as exc:
        if getattr(exc, "code", None) == 26:
            logger.debug(
                "index_information(%s) NamespaceNotFound — treating as empty",
                coll_name,
            )
            info = {}
        else:
            raise
    except (AutoReconnect, NetworkTimeout) as exc:
        logger.warning(
            "index_information(%s) transient %s — treating as empty",
            coll_name, type(exc).__name__,
        )
        info = {}
    cache[coll_name] = info
    return info


async def _safe_drop_index(coll, name: str, *, cache: dict[str, dict]) -> None:
    """Drop an index, tolerating the closed-set OperationFailure codes.

    Codes 26 / 27 → debug + return (drop is idempotent for absent state).
    Codes 68 / 85 / 86 / 276 → warning + return.
    Other ``OperationFailure`` codes → log error and re-raise (caller's
    fault).
    """
    coll_name = getattr(coll, "name", "<unknown>")
    try:
        await coll.drop_index(name)
        # Invalidate cache so subsequent reads of index_information see the drop.
        cache.pop(coll_name, None)
    except OperationFailure as exc:
        code = getattr(exc, "code", None)
        if code in (26, 27):
            logger.debug(
                "drop_index(%s.%s) noop code=%s",
                coll_name, name, code,
            )
            cache.pop(coll_name, None)
            return
        if code in _TOLERATED_OP_FAILURE_CODES:
            logger.warning(
                "drop_index(%s.%s) tolerated code=%s errmsg=%s",
                coll_name, name, code, str(exc),
            )
            cache.pop(coll_name, None)
            return
        logger.error(
            "drop_index(%s.%s) unexpected OperationFailure code=%s errmsg=%s",
            coll_name, name, code, str(exc),
        )
        raise
    except (AutoReconnect, NetworkTimeout) as exc:
        logger.warning(
            "drop_index(%s.%s) transient %s — continuing",
            coll_name, name, type(exc).__name__,
        )


async def _safe_create_index(
    coll,
    keys: Any,
    *,
    cache: dict[str, dict],
    name: str | None = None,
    **opts: Any,
) -> bool:
    """Create an index with spec-match short-circuit + Atlas-tolerant
    retry. Returns True if a new index was created, False otherwise.
    """
    coll_name = getattr(coll, "name", "<unknown>")
    target_name = name or _infer_index_name(keys)

    # Spec-match short-circuit.
    info = await _get_index_info(coll, cache)
    existing = info.get(target_name)
    if existing is not None and _index_spec_matches(existing, keys, opts):
        logger.debug(
            "create_index(%s.%s) short-circuit — spec matches existing",
            coll_name, target_name,
        )
        return False

    # If a same-named index exists with a different spec, drop first so
    # the create can succeed cleanly.
    if existing is not None:
        logger.info(
            "create_index(%s.%s) spec differs — drop+recreate",
            coll_name, target_name,
        )
        await _safe_drop_index(coll, target_name, cache=cache)

    kwargs = dict(opts)
    if name is not None:
        kwargs["name"] = name

    last_exc: Exception | None = None
    for attempt in range(MAX_INDEX_BOOTSTRAP_ATTEMPTS):
        try:
            await coll.create_index(keys, **kwargs)
            cache.pop(coll_name, None)
            return True
        except OperationFailure as exc:
            last_exc = exc
            code = getattr(exc, "code", None)
            if code in (85, 86):
                # Race: another worker created a conflicting version
                # between our drop and create. One drop+recreate cycle.
                logger.warning(
                    "create_index(%s.%s) conflict code=%s — drop+retry",
                    coll_name, target_name, code,
                )
                await _safe_drop_index(coll, target_name, cache=cache)
                continue
            if code == 276 and attempt < MAX_INDEX_BOOTSTRAP_ATTEMPTS - 1:
                # IndexBuildAborted — concurrent build aborted, retry
                # with exponential backoff. Other tolerated codes do
                # NOT retry.
                delay = _INDEX_BUILD_BACKOFF_S[
                    min(attempt, len(_INDEX_BUILD_BACKOFF_S) - 1)
                ]
                logger.warning(
                    "create_index(%s.%s) IndexBuildAborted code=276 — "
                    "retry attempt=%d after %.1fs",
                    coll_name, target_name, attempt + 1, delay,
                )
                await asyncio.sleep(delay)
                continue
            if code in _TOLERATED_OP_FAILURE_CODES:
                logger.warning(
                    "create_index(%s.%s) tolerated code=%s errmsg=%s",
                    coll_name, target_name, code, str(exc),
                )
                return False
            logger.error(
                "create_index(%s.%s) unexpected OperationFailure code=%s errmsg=%s",
                coll_name, target_name, code, str(exc),
            )
            raise
        except (AutoReconnect, NetworkTimeout) as exc:
            last_exc = exc
            logger.warning(
                "create_index(%s.%s) transient %s — continuing",
                coll_name, target_name, type(exc).__name__,
            )
            return False
    # Exhausted retries.
    code = getattr(last_exc, "code", None)
    logger.warning(
        "create_index(%s.%s) exhausted %d attempts last_code=%s — continuing",
        coll_name, target_name, MAX_INDEX_BOOTSTRAP_ATTEMPTS, code,
    )
    return False


# ---------------------------------------------------------------------------
# ensure_indexes — Atlas-resilient, idempotent, never crashes startup
# ---------------------------------------------------------------------------


def _reset_indexes_ensured_for_test() -> None:
    """Reset the per-process idempotency guard. Test-only."""
    global _INDEXES_ENSURED
    _INDEXES_ENSURED = False


async def ensure_indexes(db: AsyncIOMotorDatabase | None = None) -> None:
    """Create indexes on hot lookups. Idempotent + Atlas-tolerant.

    Safe to call multiple times in the same process — the second call
    is a no-op (debug log). Safe under concurrent worker boots: the
    spec-match short-circuit avoids drop+create churn, and the closed-
    set of tolerated ``OperationFailure`` codes prevents the
    deployment from crashing on the index-build race that Atlas
    surfaces as ``IndexBuildAborted`` (code 276).
    """
    global _INDEXES_ENSURED
    if _INDEXES_ENSURED:
        logger.debug("ensure_indexes — idempotency guard, already ensured this process")
        return
    db = db if db is not None else get_db()
    # Per-collection index_information cache. Invalidated on
    # drop / create.
    cache: dict[str, dict] = {}

    async def _ci(coll_name: str, keys: Any, **opts: Any) -> bool:
        return await _safe_create_index(db[coll_name], keys, cache=cache, **opts)

    async def _di(coll_name: str, idx_name: str) -> None:
        await _safe_drop_index(db[coll_name], idx_name, cache=cache)

    await _ci(Collections.USERS, "email", unique=True)
    # Scienthesis integration — sparse lookup on the Identity subject. Sparse
    # (not unique) so native accounts with no identity_id are excluded and a
    # transient null never collides.
    await _ci(Collections.USERS, "identity_id", sparse=True)
    # Scienthesis integration — outbox sweeper drains by (status, next_attempt_at).
    await _ci(
        Collections.LITHUB_SYNC_OUTBOX, [("status", 1), ("next_attempt_at", 1)],
    )
    # AI agents — advisory decision sidecars (30-day TTL; append-only).
    await _ci(Collections.AI_AGENT_DECISIONS, "search_session_id")
    await _ci(Collections.AI_AGENT_DECISIONS, [("search_session_id", 1), ("agent", 1)])
    await _ci(Collections.AI_AGENT_DECISIONS, "created_at", expireAfterSeconds=30 * 24 * 3600)
    await _ci(Collections.AI_SURFACED_ARTICLES, "search_session_id")
    await _ci(Collections.AI_SURFACED_ARTICLES, [("search_session_id", 1), ("article_id", 1)])
    await _ci(Collections.AI_SURFACED_ARTICLES, "created_at", expireAfterSeconds=30 * 24 * 3600)
    # Research collections (persistent — no TTL).
    await _ci(Collections.RESEARCH_COLLECTIONS, "user_id")
    await _ci(Collections.RESEARCH_COLLECTIONS, [("user_id", 1), ("parent_id", 1)])
    await _ci(Collections.RESEARCH_COLLECTION_ITEMS, "collection_id")
    await _ci(Collections.RESEARCH_COLLECTION_ITEMS, "user_id")
    await _ci(
        Collections.RESEARCH_COLLECTION_ITEMS,
        [("collection_id", 1), ("article_id", 1)],
        unique=True, name="research_collection_item_unique",
    )
    # Phase 6D — canonical_key/pmid/doi/pmcid uniqueness is set up later
    # in this function with proper names and partial filters. Skip the
    # legacy ``doi``/``pmid`` sparse indexes here to avoid name
    # collisions with the existing ``pmid_1``/``doi_1`` indexes that
    # had different sparse settings across deploys.
    await _ci(Collections.SEARCH_SESSIONS, "user_id")
    await _ci(Collections.SEARCH_SESSIONS, "expires_at")
    await _ci(Collections.FOLDERS, "user_id")
    await _ci(Collections.FOLDER_ARTICLES, "folder_id")
    await _ci(Collections.SOURCE_CONFIGS, "source_name", unique=True)
    await _ci(Collections.GUIDELINE_SOURCES, "source_name", unique=True)
    await _ci(Collections.AUDIT_EVENTS, "created_at")
    # Phase 2D indexes.
    await _ci(
        Collections.UPLOADED_FULL_TEXT_ASSETS,
        [("user_id", 1), ("search_session_id", 1), ("article_id", 1)],
    )
    await _ci(Collections.UPLOADED_FULL_TEXT_ASSETS, "deleted_at", sparse=True)
    await _ci(Collections.UPLOADED_FULL_TEXT_ASSETS, "expires_at", sparse=True)
    await _ci(
        Collections.ARTICLE_TEXT_EXTRACTS,
        [("search_session_id", 1), ("article_id", 1)],
    )
    await _ci(Collections.ARTICLE_TEXT_EXTRACTS, "expires_at", sparse=True)
    # Phase 3 — latest-summary-per-session lookup.
    await _ci(
        Collections.SUMMARIES, [("search_session_id", 1), ("created_at", -1)],
    )
    await _ci(Collections.SUMMARIES, "expires_at", sparse=True)
    # Phase 4B — durable jobs.
    await _ci(Collections.JOBS, [("user_id", 1), ("created_at", -1)])
    await _ci(Collections.JOBS, [("search_session_id", 1), ("created_at", -1)])
    await _ci(Collections.JOBS, [("status", 1), ("updated_at", 1)])

    # Phase 6D — canonical article identity. Uniqueness uses partial filter
    # expressions because the legacy collection has many docs with explicit
    # ``null`` values on pmid/doi/pmcid. ``canonical_key`` is unique
    # outright because every article carries one. SKIPPED under pytest
    # because the test fixtures intentionally re-insert the same canonical
    # identifiers (production-only data-integrity rule).
    if not _is_pytest_context():
        await _ci(
            Collections.ARTICLES, "canonical_key",
            unique=True, name="canonical_key_unique",
        )
        await _ci(
            Collections.ARTICLES, "pmid",
            unique=True, name="pmid_unique",
            partialFilterExpression={"pmid": {"$type": "string"}},
        )
        await _ci(
            Collections.ARTICLES, "doi_normalized",
            unique=True, name="doi_normalized_unique",
            partialFilterExpression={"doi_normalized": {"$type": "string"}},
        )
        await _ci(
            Collections.ARTICLES, "pmcid",
            unique=True, name="pmcid_unique",
            partialFilterExpression={"pmcid": {"$type": "string"}},
        )
    else:
        # Pytest mode — drop any conflicting unique-or-sparse index first,
        # then create non-unique. Idempotent across runs.
        for idx_name in (
            "canonical_key_unique", "pmid_unique",
            "doi_normalized_unique", "pmcid_unique",
            "pmid_1", "doi_1", "pmcid_1",
            "canonical_key_1", "doi_normalized_1",
        ):
            await _di(Collections.ARTICLES, idx_name)
        await _ci(Collections.ARTICLES, "canonical_key")
        await _ci(Collections.ARTICLES, "pmid")
        await _ci(Collections.ARTICLES, "doi_normalized")
        await _ci(Collections.ARTICLES, "pmcid")

    # Phase 6D — user_library_items.
    await _ci(
        Collections.USER_LIBRARY_ITEMS, [("user_id", 1), ("saved_at", -1)],
    )
    await _ci(Collections.USER_LIBRARY_ITEMS, "article_id")
    await _ci(Collections.USER_LIBRARY_ITEMS, "canonical_key")
    # Uniqueness on (user_id, article_id) only for ACTIVE items so the
    # same article can be re-saved after being archived.
    await _ci(
        Collections.USER_LIBRARY_ITEMS,
        [("user_id", 1), ("article_id", 1)],
        unique=True, name="user_library_active_unique",
        partialFilterExpression={"status": "active"},
    )
    # Phase 6D — saved_evidence_packets + folder_summaries.
    await _ci(
        Collections.SAVED_EVIDENCE_PACKETS,
        [("user_id", 1), ("folder_id", 1), ("created_at", -1)],
    )
    await _ci(
        Collections.FOLDER_SUMMARIES,
        [("user_id", 1), ("folder_id", 1), ("created_at", -1)],
    )
    # Phase 6E — benchmark gold curation.
    await _ci(
        Collections.BENCHMARK_GOLD_CANDIDATES,
        [("benchmark_case_id", 1), ("status", 1)],
    )
    await _ci(
        Collections.BENCHMARK_GOLD_CANDIDATES,
        [("benchmark_case_id", 1), ("identifier_type", 1),
         ("identifier_value", 1)],
        name="benchmark_gold_natural_key",
    )
    await _ci(Collections.BENCHMARK_GOLD_CANDIDATES, "created_at")
    await _ci(
        Collections.BENCHMARK_CASE_OVERRIDES,
        "benchmark_case_id", unique=True,
    )

    # Phase M3.1 — preview/dev-only admin A/B benchmark runs. TTL = 30d.
    await _ci(Collections.M3_1_BENCHMARK_RUNS, "expires_at",
              expireAfterSeconds=0)
    await _ci(Collections.M3_1_BENCHMARK_RUNS, "triggered_by_admin_user_id")
    await _ci(Collections.M3_1_BENCHMARK_RUNS, "search_session_id")
    await _ci(Collections.M3_1_BENCHMARK_RUNS, [("created_at", -1)])

    # Phase M3.2 — preview-only active semantic repair decision
    # sidecar. TTL on ``created_at`` = 30 days. One row per active-
    # pipeline decision (the same session may be re-searched and
    # produce a new sidecar row; sidecar is append-only).
    await _ci(Collections.M3_2_SESSION_DECISIONS, "created_at",
              expireAfterSeconds=30 * 24 * 3600)
    await _ci(Collections.M3_2_SESSION_DECISIONS, "search_session_id")
    await _ci(Collections.M3_2_SESSION_DECISIONS, "user_id")
    await _ci(Collections.M3_2_SESSION_DECISIONS, [("created_at", -1)])

    # Phase M3.2 — feature flag uniqueness. ``key`` is the closed-set
    # flag name (currently only ``m3_2_active_repair_kill_switch``).
    await _ci(Collections.FEATURE_FLAGS, "key", unique=True,
              name="feature_flag_key_unique")

    # Phase M4 — Full-Question interpreter shadow sidecar. TTL on
    # ``created_at`` = 30 days. Append-only; one row per triggered run.
    await _ci(Collections.M4_SHADOW_DECISIONS, "created_at",
              expireAfterSeconds=30 * 24 * 3600)
    await _ci(Collections.M4_SHADOW_DECISIONS, "search_session_id")
    await _ci(Collections.M4_SHADOW_DECISIONS, "user_id")
    await _ci(Collections.M4_SHADOW_DECISIONS, [("created_at", -1)])

    # Phase R1A — shadow reranking sidecar. TTL on ``expires_at``
    # = 30 days (set at write time). Secondary lookup by session
    # OR benchmark case OR shape.
    await _ci(Collections.R1A_SHADOW_DECISIONS, "expires_at",
              expireAfterSeconds=0)
    await _ci(Collections.R1A_SHADOW_DECISIONS, "search_session_id")
    await _ci(Collections.R1A_SHADOW_DECISIONS, "benchmark_case_id")
    await _ci(Collections.R1A_SHADOW_DECISIONS, "shape")
    await _ci(Collections.R1A_SHADOW_DECISIONS, [("created_at", -1)])

    # Phase R1A — gold-comparator cases. Append-only, no TTL.
    # Unique by case_id.
    await _ci(Collections.R1A_GOLD_COMPARATOR_CASES, "case_id",
              unique=True, name="r1a_gold_case_id_unique")
    await _ci(Collections.R1A_GOLD_COMPARATOR_CASES, "shape")

    # Phase R1B — retrieval-recall + directness shadow sidecar.
    # TTL on ``expires_at`` = 30 days (set at write time).
    # Secondary lookups by case OR session OR shape.
    await _ci(Collections.R1B_SHADOW_DECISIONS, "expires_at",
              expireAfterSeconds=0)
    await _ci(Collections.R1B_SHADOW_DECISIONS, "search_session_id")
    await _ci(Collections.R1B_SHADOW_DECISIONS, "benchmark_case_id")
    await _ci(Collections.R1B_SHADOW_DECISIONS, "shape")
    await _ci(Collections.R1B_SHADOW_DECISIONS, [("created_at", -1)])

    # Phase R1C — gold-recall + anchor-specific shadow sidecar.
    # TTL on ``expires_at`` = 30 days.
    await _ci(Collections.R1C_SHADOW_DECISIONS, "expires_at",
              expireAfterSeconds=0)
    await _ci(Collections.R1C_SHADOW_DECISIONS, "search_session_id")
    await _ci(Collections.R1C_SHADOW_DECISIONS, "benchmark_case_id")
    await _ci(Collections.R1C_SHADOW_DECISIONS, "shape")
    await _ci(Collections.R1C_SHADOW_DECISIONS, [("created_at", -1)])

    # Phase K4 — clinician-visible branch-aware evidence packets.
    # Unique per ``search_session_id`` (one packet per session).
    # No TTL — packets live the life of the session. Leak-scanned.
    await _ci(Collections.EVIDENCE_PACKETS, "search_session_id",
              unique=True, name="k4_evidence_packet_session_unique")
    await _ci(Collections.EVIDENCE_PACKETS, [("created_at", -1)])

    _INDEXES_ENSURED = True
