# LitPortal Backend → Scienthesis Identity + LitHub Integration Plan

**Status:** in progress · **Author:** platform team · **Date:** 2026-06-27

This document is the deep plan for wiring the extracted `litportal-backend`
(Emergent-built "ClinEvidence Search API", FastAPI + MongoDB) into the central
Scienthesis platform — replacing the role currently played by `litbridge-backend`.
It follows **Design A** from `SCIENTHESIS_INTEGRATION.md` (the only design with
standalone central identity + library *services*), and mirrors the proven
integration already shipped in `litbridge-backend`.

---

## 0. Origin & guiding principles

- **Source:** `LitPortal-On-Emergent-05.26.26/backend` → extracted verbatim to
  this folder (415 files, no code changes during extraction).
- **Stack reality:** FastAPI + **MongoDB/Motor** + Pydantic v2 + PyJWT (HS256) +
  bcrypt. This differs from the Postgres/SQLAlchemy stack in
  `litbridge-backend`/`identity`/`lithub`, so we **port the pattern, not the
  code** for anything DB-bound. The three platform *client* modules are
  framework-agnostic (`httpx` + `pyjwt`) and port near-verbatim.
- **Principles (non-negotiable, from the platform docs):**
  1. **Default-OFF, reversible.** Both integration master flags
     (`LITPORTAL_USE_IDENTITY`, `LITPORTAL_DUAL_WRITE_LITHUB`) default `false`.
     With them off, the backend behaves *exactly* like the standalone Emergent
     backend. Rollback = flag flip + redeploy, no data migration.
  2. **Additive, not destructive.** We do not delete native auth or the local
     Mongo library. We add an Identity validation path and a LitHub mirror
     alongside them. Existing routes, response shapes, and tests stay green.
  3. **Identity `sub` is the cross-app join key.** Every LitHub call is keyed by
     the user's Identity `sub` (UUID). Saves only mirror to LitHub for
     Identity-authenticated users (where the `sub` is known).
  4. **Match conventions where cheap, document divergence where not.** We keep
     the Emergent top-level module layout (`server:app`, run from repo root) —
     restructuring 415 files into `src/` is out of scope and risky. We add a
     `clients/` package mirroring litbridge's `src/clients/`.

---

## 1. Target topology (Design A)

| Service | Role | DB | Port |
|---|---|---|---|
| `scienthesis-identity` | Source of truth for users; RS256 JWT + JWKS | Postgres | 8100 |
| `scienthesis-lithub` | Central library (papers + per-user entries + folders) | Postgres | 8200 |
| **`litportal-backend`** (this) | LitPortal BFF / Portal Engine (search, ranking, summaries) | **MongoDB** | 8000 |
| `litpulse-backend` | LitPulse BFF | MongoDB | 8001 |

Auth: user-facing = RS256 JWT validated **offline via Identity's JWKS**;
service-to-service = shared-secret **HS256 `X-Service-Token`** on `/internal/*`.

---

## 2. The two integration seams in this backend

The 9-agent analysis found the entire app funnels through exactly two seams:

### Seam A — Authentication (one chokepoint)
- `core/security.py::get_current_user` (the ONLY current-user dependency, ~50
  call sites + `require_admin`). Returns a **dict** with `id`, `email`, `role`,
  `name`, `first_use_acknowledged`, `created_at`, `updated_at`.
- Today: decodes a native HS256 JWT (`JWT_ACCESS_SECRET`) and loads the user
  from Mongo `users` by `id`.

### Seam B — Library save/list (service layer)
- `services/user_library.py` (the `user_library_items` flat "my saved papers"
  list) — `save_item`, `list_items`, `import_by_identifier`/`_fetch_and_persist`,
  `batch_save`, `get/update/delete_item`. Routers only call this service.
- Folder articles (`routers/folders.py`) and saved evidence packets
  (`routers/library.py`) are inline-in-router and **out of scope for this phase**
  (seam identified below for a follow-up).
- Library items reference an article by the **internal `article_id` UUID**;
  external ids (DOI/PMID/PMCID) live on the local `articles` doc and are joined
  in at read time. LitHub is keyed by **pmid/doi** → we extract those from the
  local `articles` doc when mirroring.

---

## 3. New code (ported from litbridge, adapted to this backend)

A new top-level `clients/` package:

- `clients/service_token.py` — `mint_service_token(audience)`. HS256, `iss="litportal"`,
  signed with `SERVICE_TOKEN_SECRET`, 300s TTL, per-audience cache. **Near-verbatim**
  copy (swap `src.core.config.get_settings` → this backend's `core.config.get_settings`;
  read secret off the `settings` object).
- `clients/identity_client.py` — `validate_identity_access_token(token, *, http_client, settings)`
  (JWKS cache + RS256 verify, enforces `iss`/`aud`/`type=access`) and an
  `IdentityClient` HTTP class for optional internal user lookups. **Near-verbatim.**
- `clients/lithub_client.py` — `LitHubClient` with the internal (service-token)
  methods `internal_save_paper`, `internal_list_library`, `internal_membership`,
  `internal_delete_library`, `internal_papers_bulk`, `internal_bulk_import`.
  **Near-verbatim.**

Adaptation notes:
- These reference a `Settings` object via attribute access. The Emergent
  `Settings` is a plain class (not pydantic). We add the needed attributes (§4)
  so `getattr(settings, "X")` works. `mint_service_token` already uses
  `getattr(settings, "SERVICE_TOKEN_SECRET", "")`.
- The clients need a shared `httpx.AsyncClient`. The Emergent backend has none →
  we create one in the FastAPI `lifespan` and store it on `app.state.http_client`
  (§5), exactly like litbridge.

---

## 4. Config additions (`core/config.py`)

Add to `Settings` (all read via `os.environ.get`, typed, defaulted):

```python
# --- Scienthesis platform integration (default OFF = no-op) ---
LITPORTAL_USE_IDENTITY: bool        = _bool("LITPORTAL_USE_IDENTITY", False)
LITPORTAL_DUAL_WRITE_LITHUB: bool   = _bool("LITPORTAL_DUAL_WRITE_LITHUB", False)
IDENTITY_BASE_URL: str              = os.environ.get("IDENTITY_BASE_URL", "")
IDENTITY_JWKS_URL: str              = os.environ.get("IDENTITY_JWKS_URL", "")
IDENTITY_JWT_ISSUER: str            = os.environ.get("IDENTITY_JWT_ISSUER", "scienthesis-identity")
IDENTITY_JWT_AUDIENCE: str          = os.environ.get("IDENTITY_JWT_AUDIENCE", "litportal")
JWKS_CACHE_TTL_SECONDS: int         = int(os.environ.get("JWKS_CACHE_TTL_SECONDS", "300"))
LITHUB_BASE_URL: str                = os.environ.get("LITHUB_BASE_URL", "")
SERVICE_TOKEN_SECRET: str           = os.environ.get("SERVICE_TOKEN_SECRET", "")
```

All already documented in `.env.example`.

---

## 5. Shared HTTP client (`server.py` lifespan)

- On startup: `app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))`.
- On shutdown: `await app.state.http_client.aclose()`.
- Wrapped so it never blocks the existing seeders/index bootstrap.

---

## 6. Seam A implementation — Identity auth (delegation + validation)

**Two halves, both gated by `LITPORTAL_USE_IDENTITY` (default OFF):**

**(a) Endpoint delegation (token issuance handoff).** `routers/auth.py`
signup/login/request-otp/verify-otp/refresh/logout DELEGATE to the Identity
Service via `IdentityClient` and return Identity's **RS256** token pair (set as
the same `ces_refresh_token` cookie + `TokenResponse`). This is what makes a
user who signs in *on LitPortal* receive a token whose audience covers
`litpulse,litportal,lithub` — so the **same login works across all apps**. The
local Mongo shadow user is provisioned from the token's `sub`/`email`. With the
flag OFF, these endpoints keep minting native HS256 tokens (byte-identical).

**(b) Token validation (below) — accepting Identity tokens on every request.**



Modify `core/security.py::get_current_user` to a small validator chain, **keeping
the existing dict return shape and 401 semantics**:

1. **Native HS256** (unchanged) — try `decode_access_token` + Mongo lookup first
   (cheapest, current behavior).
2. **Identity RS256** (new, only when `settings.LITPORTAL_USE_IDENTITY` and a
   `request` with `app.state.http_client` is available):
   - `payload = await validate_identity_access_token(raw, http_client=..., settings=...)`
   - `None` → not Identity-shaped → fall through; raises on Identity-shaped-but-invalid → 401.
   - On success: **upsert a Mongo `users` shadow doc** keyed by the Identity
     `sub`, with `id == sub` (so `current_user["id"]` is the cross-app `sub`),
     `identity_id == sub`, `email`, `role` (default `"user"`; preserved if the
     doc already exists), `name`, `first_use_acknowledged`, timestamps.
   - Return the same dict shape `get_current_user` already returns.
3. If all fail → 401 (unchanged).

To reach `app.state.http_client`, add an optional `request: Request` param to
`get_current_user` (FastAPI injects it; the ~50 `Depends(get_current_user)` call
sites are unaffected). When called without a request (unit tests), the Identity
path is skipped — same trick litbridge uses.

New helper `core/security.py::_upsert_identity_user(db, sub, email, ...)` (Mongo
upsert on `users` keyed by `id == sub`). Add a partial-safe path: if a native
user already exists with the same email, link it by stamping `identity_id` (do
not duplicate). Provisioning is lazy (first authenticated request).

`models/users.py::UserDoc` gains an optional `identity_id: str | None = None`
(extra fields are already `ignore`d, but we declare it for clarity + future
indexing). Optional Mongo index `users.identity_id` (sparse) added in
`core/db.py::ensure_indexes` (idempotent, non-fatal like the rest).

`require_admin` is **unchanged** (reads `current_user["role"]`).

### Auth correctness invariants
- Native and Identity tokens are unambiguous by algorithm (HS256 vs RS256) and
  signature, so the chain never mis-routes.
- A valid-but-failed Identity token raises 401 (never silently downgraded).
- With `LITPORTAL_USE_IDENTITY=false`, behavior is byte-identical to today.

---

## 7. Seam B implementation — LitHub library mirror (additive)

Inject an optional `LitHubClient` into `UserLibraryService` and gate on
`settings.LITHUB_BASE_URL and settings.LITPORTAL_DUAL_WRITE_LITHUB`.

- **save (`save_item`)**: after the local save commits, **best-effort mirror** to
  LitHub via `internal_save_paper(user_id=sub, item={pmid,doi,title,journal,
  abstract,authors,year,ai_summary,source:"litportal-library", ...})`, where
  `sub = current_user.identity_id` (skip mirror if no `sub` / no pmid&doi). The
  pmid/doi/title are read from the local `articles` doc (already loaded during
  save). Mirror failures never fail the user's save (logged; see reliability).
- **list (`list_items`)**: when enabled, optionally **enrich/merge** the local
  list with LitHub's `internal_list_library(user_id=sub)` so papers saved from
  LitPulse also appear — mapped into the existing `{items, total, limit, skip}`
  shape. Best-effort: any LitHub error → fall back to local-only (never error
  the page).
- **delete (`delete_item`)**: mirror an idempotent `internal_delete_library`
  keyed by pmid/doi when enabled.

**Reliability (Mongo-native outbox):** mirror writes are enqueued to a new Mongo
collection `lithub_sync_outbox` and drained by a 60s background sweeper task
(started in lifespan, guarded so it never crashes the loop) with capped
exponential backoff and a `dead` terminal state — the Mongo analogue of
litbridge's `lithub_sync_outbox` table + sweeper. The synchronous path attempts
the mirror inline first (fast path); on failure it enqueues.

User id for LitHub = the Identity `sub` (`current_user["identity_id"]`). This is
the only key that makes a LitPortal save visible to a LitPulse read.

### Out of scope this phase (seam identified for follow-up)
- Folder articles (`routers/folders.py`) and saved evidence packets
  (`routers/library.py`) — currently inline-in-router. Recommended follow-up:
  extract a `FolderService` mirroring `UserLibraryService`, then apply the same
  gated mirror. Documented, not done now.

---

## 8. Runnability & stability

- **Dockerfile** (new): python:3.12-slim, install `requirements.txt`, copy app,
  `CMD uvicorn server:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}` (run from
  repo root so the top-level imports resolve).
- **docker-compose.yml** (new, dev): litportal-backend + a local MongoDB, env
  from `.env`.
- **`run.sh` / documented run command** for non-Docker local dev.
- **Fix hardcoded path:** `routers/handoff_downloads.py` `_MEMORY_DIR =
  Path("/app/memory")` → env-overridable `HANDOFF_MEMORY_DIR` (default keeps
  `/app/memory` for prod parity; settable for local).
- **Health:** `GET /api/health` already exists. Smoke test: boot with a throwaway
  Mongo + flags off, assert `/api/health` 200 and `/api/openapi.json` loads.
- **Tests:** the ported clients ship with focused unit tests
  (`tests/test_scienthesis_integration.py` analogue) covering: RS256 validator
  accepts a valid multi-aud token, returns `None` for HS256/unknown-kid, raises
  on expired; service-token mint shape; LitHub client payload/headers; and the
  additive `get_current_user` chain (native still works; identity path upserts).

---

## 9. Deploy constraints (carried from platform docs)

- `CORS_ORIGINS`: explicit origins in prod, **no wildcard**.
- Base URLs (`IDENTITY_BASE_URL`/`IDENTITY_JWKS_URL`/`LITHUB_BASE_URL`) must be
  set when flags are on, or auth/mirror fail.
- `SERVICE_TOKEN_SECRET` byte-identical across all four services; rotate jointly.
- `JWT_ACCESS_SECRET`/`JWT_REFRESH_SECRET` explicitly set in prod.
- Identity service needs its RS256 keypair generated once before first run.

---

## 10. Decisions taken (call out for review)

1. **Layout kept flat** (`server:app` from repo root), not restructured to `src/`.
2. **Additive default-OFF auth** (native + Identity), not a hard cutover.
3. **Library phase 1 = `user_library_items` only** (the canonical save/list);
   folders + evidence packets deferred with the seam documented.
4. **Shadow user `id == Identity sub`** for Identity-provisioned users, so all
   user-scoped Mongo data and LitHub calls share the cross-app key.

If any of these should differ (e.g. hard auth cutover, or mirror folders too),
the default-OFF design means changing course is localized and low-risk.
