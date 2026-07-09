# Setup

How to run the Portal Engine locally, in Docker, and what every environment variable does. The app is a FastAPI service backed by MongoDB. There is no SQL migration step; Mongo indexes are created at startup.

## Prerequisites

- Python 3.12
- [`uv`](https://github.com/astral-sh/uv) for dependency install (the Docker image and the pinned `requirements.txt` both assume it)
- A reachable MongoDB 7 instance (local, or Atlas for production)
- Optional for integration work: running scienthesis-identity (:8100) and scienthesis-lithub (:8200), and a shared `SERVICE_TOKEN_SECRET`
- Optional for real summaries: an API key for one of OpenAI, OpenRouter, or Anthropic

## Local run

Run from the repo root. `server.py` imports `core`, `routers`, `services`, `clients`, and `models` as top-level modules, so the working directory has to be the repo root and the app is `server:app`.

```bash
cp .env.example .env
# Edit .env: set MONGO_URL and DB_NAME. Everything else has a working default for dev.
uv sync
uvicorn server:app --host 0.0.0.0 --port 8000
```

Minimum viable `.env` for standalone dev:

```
MONGO_URL=mongodb://localhost:27017
DB_NAME=litportal
APP_ENV=development
JWT_ACCESS_SECRET=dev-access-secret-change-me
JWT_REFRESH_SECRET=dev-refresh-secret-change-me
```

With `APP_ENV=development` and `SEED_TEST_USERS=true`, startup seeds a clinician (`clinician@test.com` / `ClinicianTest123!`) and an admin (`admin@test.com` / `AdminTest123!`). Seeding never runs when `APP_ENV=production`.

Interactive API docs are at `http://localhost:8000/api/docs`; the OpenAPI JSON is at `/api/openapi.json`.

## Docker

The image is multi-stage, installs with `uv`, and runs as a non-root `app` user. The entry command expands `HOST` and `PORT` from the environment.

```bash
docker build -t litportal-backend .
docker run --rm -p 8000:8000 \
  -e MONGO_URL="mongodb://host.docker.internal:27017" \
  -e DB_NAME="litportal" \
  -e APP_ENV="development" \
  -e JWT_ACCESS_SECRET="dev-access-secret-change-me" \
  -e JWT_REFRESH_SECRET="dev-refresh-secret-change-me" \
  litportal-backend
```

`docker-compose.yml` brings up the app together with a MongoDB container:

```bash
docker compose up --build
```

Compose sets `MONGO_URL=mongodb://litportal-mongo:27017`, `DB_NAME=litportal`, and stable dev JWT secrets. The Identity and LitHub wiring is commented out in the compose file; uncomment the flags and base URLs to test the integration path against sibling services on the host.

## Database and migrations

There is no Alembic or migration command. On startup the lifespan handler:

1. Calls `ensure_indexes()` (idempotent; tolerates the closed set of `OperationFailure` codes seen under concurrent worker boots). An index failure logs but does not stop the app.
2. Seeds `source_configs` (PubMed, Europe PMC, OpenAlex, and the guideline lanes) and the `guideline_sources` registry.
3. Touches every collection so it exists.
4. Seeds dev test users only when `SEED_TEST_USERS` is true and `APP_ENV` is not `production`.

Point `MONGO_URL` at a fresh database and the app provisions itself. Do not use a `DB_NAME` containing `test` outside CI: the test-mode branch relaxes the unique article indexes.

## Environment variables

`core/config.py` reads these at import. `MONGO_URL` and `DB_NAME` use `os.environ[...]` with no default, so a missing value raises `KeyError` before the app object is built. Everything below is grouped by concern. Full annotated surface lives in `.env.example`.

### Core / runtime

| Variable | Required | Default | Description |
|---|---|---|---|
| `MONGO_URL` | yes | none (`KeyError`) | Mongo connection string |
| `DB_NAME` | yes | none (`KeyError`) | Mongo database name; avoid `test` outside CI |
| `APP_ENV` | no | `development` | `IS_PRODUCTION = (APP_ENV == "production")`; drives fail-closed startup guards |
| `CORS_ORIGINS` | no | `*` | Comma-separated origins; a bare `*` is stripped and the code falls back to a hardcoded allowlist |
| `HOST` | no | `0.0.0.0` | Uvicorn / Docker bind host |
| `PORT` | no | `8000` | Uvicorn / Docker bind port |

### Native authentication

Used when the Identity integration is off (the default). The secrets auto-generate per process if unset, which invalidates every issued token on restart and across workers. Set them explicitly in any environment that outlives a single process.

| Variable | Required | Default | Description |
|---|---|---|---|
| `JWT_ACCESS_SECRET` | prod | auto-gen | HS256 access-token secret |
| `JWT_REFRESH_SECRET` | prod | auto-gen | HS256 refresh-token secret |
| `JWT_ALGORITHM` | no | `HS256` | Native token algorithm |
| `JWT_ACCESS_TTL_MINUTES` | no | `60` | Access-token lifetime |
| `JWT_REFRESH_TTL_DAYS` | no | `7` | Refresh-token lifetime |
| `COOKIE_SECURE` | no | `false` | Set `true` behind HTTPS in production |
| `COOKIE_SAMESITE` | no | `lax` | SameSite for the `ces_refresh_token` cookie |
| `LOGIN_RATE_LIMIT_MAX` | no | `5` | Failed-login budget per email+IP (in-memory, per worker) |
| `LOGIN_RATE_LIMIT_WINDOW_MIN` | no | `15` | Rate-limit window in minutes |

### Scienthesis integration

Both master flags default false, so the app runs standalone with native auth and no LitHub mirroring. Turning either on requires the matching base URL and the shared service-token secret.

| Variable | Required | Default | Description |
|---|---|---|---|
| `LITPORTAL_USE_IDENTITY` | no | `false` | Master flag: delegate auth to Identity and accept RS256 tokens verified against JWKS |
| `LITPORTAL_DUAL_WRITE_LITHUB` | no | `false` | Master flag: mirror library saves and deletes into LitHub via the outbox |
| `IDENTITY_BASE_URL` | when USE_IDENTITY | `""` | Identity base URL (:8100) |
| `IDENTITY_JWKS_URL` | no | `{IDENTITY_BASE_URL}/.well-known/jwks.json` | Explicit JWKS endpoint |
| `IDENTITY_JWT_ISSUER` | no | `scienthesis-identity` | Expected `iss` on Identity tokens |
| `IDENTITY_JWT_AUDIENCE` | no | `litportal` | Expected `aud`; must be in Identity's `JWT_AUDIENCES` |
| `JWKS_CACHE_TTL_SECONDS` | no | `300` | JWKS cache lifetime |
| `LITHUB_BASE_URL` | for dual-write | `""` | LitHub base URL (:8200); the outbox is a no-op without it |
| `SERVICE_TOKEN_SECRET` | integration | none | Shared HS256 secret for `X-Service-Token` (`iss=litportal`); byte-identical across all four services; minting raises `RuntimeError` if unset |
| `LITPULSE_JWT_ENABLED` | no | `false` | Also accept legacy LitPulse HS256 tokens |
| `LITPULSE_JWT_SECRET_KEY` | no | none | Legacy LitPulse token secret; must equal LitPulse's |
| `LITPORTAL_MODULE_MOUNT_ENABLED` | no | `true` | Also mount every router under the module prefix |
| `LITPORTAL_MODULE_PREFIX` | no | `/api/litportal` | Module-mount prefix |

### LLM provider

Summary and verification only. With no provider configured the pipeline still runs retrieval, ranking, brokering, and selection; the summary stage returns a not-ready state.

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | no | `openai` (example) | One of `none`, `openai`, `openrouter`, `anthropic`, `emergent` |
| `OPENAI_API_KEY` | for openai | none | OpenAI key |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Default OpenAI model |
| `OPENROUTER_API_KEY` | for openrouter | none | OpenRouter key |
| `OPENROUTER_MODEL` | no | `openai/gpt-4o-mini` | OpenRouter model id |
| `ANTHROPIC_API_KEY` | for anthropic | none | Anthropic key |
| `EMERGENT_LLM_KEY` | for emergent | none | Emergent gateway key (on-platform package only) |
| `SUMMARY_MODEL` / `VERIFIER_MODEL` | no | none | Per-role model overrides |
| `LLM_TIMEOUT_SECONDS` | no | `60` | LLM request timeout |

### Retrieval sources

| Variable | Required | Default | Description |
|---|---|---|---|
| `NCBI_API_KEY` / `PUBMED_API_KEY` | no | none | NCBI key (both names honored); raises PubMed from 3 to 10 req/s |
| `NCBI_EMAIL` / `NCBI_TOOL` | no | `""` / `ClinEvidenceSearch` | NCBI polite-pool identifiers |
| `PUBMED_MAX_RESULTS_PER_QUERY` / `PUBMED_MAX_CONCURRENT_QUERIES` | no | `50` / `5` | PubMed fan-out caps |
| `OPENALEX_MAILTO` | no | none | OpenAlex polite pool; the connector reads THIS name |
| `OPENALEX_EMAIL` | no | none | Declared but NOT read by the connector; set `OPENALEX_MAILTO` |
| `OPENALEX_API_KEY` / `OPENALEX_MAX_ENRICHMENTS_PER_SEARCH` | no | `""` / `50` | OpenAlex key and enrichment cap |
| `EUROPEPMC_PAGE_SIZE` | no | `25` | Europe PMC page size (`EUROPE_PMC_API_KEY` is declared but not read) |
| `ICITE_MAILTO` / `ICITE_CACHE_TTL_DAYS` | no | `OPENALEX_MAILTO` / `30` | NIH iCite polite pool and cache TTL |

### Access broker and OA full-text fetch

| Variable | Required | Default | Description |
|---|---|---|---|
| `UNPAYWALL_EMAIL` | no | `Info@scienthesis.ai` (example) | Presence enables Unpaywall OA lookup |
| `CROSSREF_MAILTO` | no | none | Read by the access broker and r1c |
| `CROSSREF_EMAIL` | no | none | Read only by `known_article_lookup`; set both to the same address |
| `SEMANTIC_SCHOLAR_API_KEY` / `SS_MAX_LOOKUPS_PER_SEARCH` | no | `""` / `50` | Semantic Scholar broker and cap |
| `ACCESS_BROKER_MAX_CONCURRENCY` | no | `8` | Concurrent broker lookups |
| `OA_FETCHER_ENABLED` | no | `true` | Master switch for the OA full-text stage |
| `OA_ALLOW_NONCOMMERCIAL` / `OA_ALLOW_NO_DERIVATIVES` / `OA_ALLOW_PREPRINTS_LICENSES` | no | `false` | License gate; default admits only CC0 / BY / BY-SA |
| `OA_MAX_FULL_TEXT_FETCHES_PER_SEARCH` / `OA_MAX_CONCURRENT_FETCHES` / `MAX_EXTRACT_CHARS_PER_ARTICLE` | no | `10` / `6` / `8000` | OA fetch caps |

### Uploads

| Variable | Required | Default | Description |
|---|---|---|---|
| `ENABLE_USER_PDF_UPLOADS` | no | `true` | Clinician PDF upload toggle |
| `MAX_UPLOAD_MB` / `MAX_PDF_PAGES` | no | `25` / `100` | Upload size and page caps |
| `PDF_UPLOAD_ALLOW_WITHOUT_AV` | no | `true` | Set `false` in production so uploads require a passing malware scan |
| `PDF_MALWARE_SCAN_ENABLED` / `PDF_MALWARE_SCAN_MODE` | no | `false` / `disabled` | AV scan (`disabled`, `clamav`, or `external`) plus `CLAMAV_HOST/PORT` or `EXTERNAL_MALWARE_SCAN_URL/API_KEY` |
| `FULL_TEXT_IMPORT_ENABLED` | no | `true` | Known-article import endpoint; `false` returns 404 |

### Jobs and concurrency

| Variable | Required | Default | Description |
|---|---|---|---|
| `JOB_RUNNER_MODE` | no | `in_process` | Async job runner mode |
| `MAX_CONCURRENT_SUMMARY_JOBS` / `MAX_CONCURRENT_OA_FETCH_JOBS` | no | `2` / `3` | Job concurrency caps |
| `JOB_STALE_AFTER_MINUTES` | no | `15` | Stale-job sweep threshold |
| `SUMMARY_JOB_TIMEOUT_SECONDS` / `MAX_SELECTED_ARTICLES_FOR_SUMMARY` / `MAX_SUMMARY_REVISIONS` | no | `120` / `10` / `1` | Summarization runner caps |

### Dev seeding

| Variable | Required | Default | Description |
|---|---|---|---|
| `SEED_TEST_USERS` | no | `true` (code) | Seed test accounts; only when set and `APP_ENV` is not `production` |
| `SEED_CLINICIAN_EMAIL` / `SEED_CLINICIAN_PASSWORD` | no | `clinician@test.com` / `ClinicianTest123!` | Dev clinician account |
| `SEED_ADMIN_EMAIL` / `SEED_ADMIN_PASSWORD` | no | `admin@test.com` / `AdminTest123!` | Dev admin account |

The advisory AI sidecars and the phased shadow stages (K2 to K5, M3, M4, R1A to R1C) add roughly 40 more flags, almost all default off or in `shadow` mode. They do not affect the live retrieval-to-summary path. See `.env.example` and ARCHITECTURE.md for the ones that carry production guards.

## Health check

```bash
curl -fsS http://localhost:8000/api/health
# {"status":"ok"}
```

Under module mount, `GET /api/litportal/health` returns `{"status":"ok","module":"litportal"}`. The Docker `HEALTHCHECK` curls `/api/health` every 30s.

## Running tests

```bash
uv run pytest
```

`pytest.ini` sets `testpaths=tests`, `asyncio_mode=auto`, and `--dist=loadgroup` so the live-API regression files and the phase4a pipeline integration file stay pinned to one xdist worker (they share Mongo state and flake under parallel startup). Test mode is detected by `PYTEST_CURRENT_TEST` or a `DB_NAME` containing `test`, which relaxes the unique article indexes. `scripts/` holds benchmark seed corpora and QA tooling; it is not part of the test run or the runtime.

## Common failure modes

- **`KeyError` at import.** `MONGO_URL` or `DB_NAME` is unset. Both are read with no default.
- **`ImportError` / `ModuleNotFoundError` on boot.** The app was started from the wrong directory or as `backend.server:app`. Run from the repo root as `server:app`.
- **Tokens stop working after a restart or across workers.** `JWT_ACCESS_SECRET` / `JWT_REFRESH_SECRET` are unset and auto-generated per process. Set them explicitly in staging and production.
- **CORS rejects the browser origin.** `CORS_ORIGINS` is `*` (or unset). The bare `*` is stripped because `allow_credentials=True` forbids it, and the code falls back to a hardcoded Emergent allowlist. List explicit origins, and edit `_resolve_cors_origins` in `server.py` if the allowlist is stale.
- **`RuntimeError` when a service token is minted.** `SERVICE_TOKEN_SECRET` is unset while an integration flag is on. Set the same secret across all four services.
- **Import error with `LLM_PROVIDER=emergent` off-platform.** The Emergent gateway package is commented out in `requirements.txt` and only resolvable on the Emergent platform. Use `openai`, `openrouter`, or `anthropic` elsewhere.
- **Boot crashes on a production guard.** With `APP_ENV=production`, the startup guards for the cap-enforcement, M3, and M4 stages fail closed if an operator set an unsafe active flag combination without the fingerprint gates. Leave those stages at their defaults unless you have the gate values.
- **Uploads accepted without scanning in production.** `PDF_UPLOAD_ALLOW_WITHOUT_AV` defaults `true`. Set it `false` and configure `PDF_MALWARE_SCAN_MODE` before exposing uploads.
- **PubMed or Crossref polite-pool ignored.** Naming split: set `OPENALEX_MAILTO` (not `OPENALEX_EMAIL`) and set both `CROSSREF_MAILTO` and `CROSSREF_EMAIL`. PubMed accepts either `NCBI_API_KEY` or `PUBMED_API_KEY`.
