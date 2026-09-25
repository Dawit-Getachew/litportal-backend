# W1.1 — Asset qualification, isolation and Day-2 checkpoint record

Status as of 2026-09-25. Every line below was checked against the code or the live repos on that
date; items that need the owner are listed at the end.

## 1. Source commits (fetched 2026-09-25 from `Scienthesis-Loka`)

| Repo | Branch → commit | Notes |
|---|---|---|
| litportal-backend | `main` → `9bc6720` (2026-09-21) | Week-1 work branches from here. `feat/aws` and `dev` were deleted on the org; everything on them is contained in `main`. Other org branches: `test-dev` `3f63f70`, `gists/aws-services-examples` `84dc59b`, `gists/python-lambda-boilerplate` `f8621fa` (Loka reference material). `personal/main` `e9ceaaa` is an older, divergent line. |
| litpulse-backend | `main` → `747fb7b` | Org also has `test` `7dd1493`. |
| scienthesis-identity | `main` → `4bfd4a9` | Includes `d0bc597` (FTRAPP, "async password verification", PR #2). `fix/multi-replica-migration-race` `49f6b1a` carries **one commit not in `main`** (Postgres advisory lock for concurrent replica migrations). |
| scienthesis-lithub | `main` → `2380e3a` | Same unmerged advisory-lock commit on `fix/multi-replica-migration-race` `8769ab1`. |
| litportal-FE | `master` → `9ce8401` | litpulse-FE embeds it in an iframe (`/litportal`), so the review step reaches LitPulse without a litpulse-FE change. |
| litpulse-FE | `master` → `fb78cf4` | Not modified. |
| scienthesis-cdk | `main` → `50b519a` | Read only (sizing, env, Lambda/ECS definitions). |

## 2. Active flags that matter here (code default → AWS task env)

| Flag | Default | AWS (CDK `stack.ts`) | Effect |
|---|---|---|---|
| `CLINICAL_ENGINE_ENABLED` | off | not set → **off** | Engine routes 404, nothing starts. |
| `LITPORTAL_USE_IDENTITY` | off | **on** | Login/signup delegated to Identity. |
| `APP_ENV` | `development` | **not set** | See §7 (seed accounts). |
| `SEED_TEST_USERS` | **true** | not set → **true** | See §7. |
| `AI_*` agents | off | not set → off | Legacy advisory agents. |
| `LLM_PROVIDER`, `OPENROUTER_MODEL` | — | from SSM | Shared with the engine's model gateway. |

## 3. Shared-service boundaries and data ownership

| Store / service | Owner | Engine access |
|---|---|---|
| Identity (Postgres) | scienthesis-identity | Only through the existing auth dependency (`core.security.get_current_user`). No change. |
| LitHub (Postgres) | scienthesis-lithub | None in Week 1 (explicit save arrives with W3.5). |
| LitPortal Mongo — legacy collections (`search_sessions`, `search_session_articles`, `articles`, …) | legacy pipeline | **Read-only**, except that an *approved* question starts a search through the existing `create_session` code path (same PHI guard, audit and TTL as `POST /api/search-sessions/`). |
| LitPortal Mongo — `engine_*` collections | clinical engine | Owned: questions, versions, runs, tasks, traces, evidence sets, rate counters. |

In AWS the LitPortal database is **Amazon DocumentDB**, not Atlas. The engine store only uses
single-document conditional updates, `find_one_and_update` and ordinary indexes (incl. one TTL
index) — no multi-document transactions, `$expr` or server-side JS.

## 4. Isolation checks (W1.1: "startup, credentials, jobs and telemetry cannot reach live resources")

- **Package boundary**: `clinical_engine/` is imported only by `server.py`. The five Lambda images
  and the ECS image copy `jobs/ core/ models/ clients/ services/ routers/ agents/` — never the engine.
  Their import graphs were traced: no worker reaches the engine.
- **Startup**: when the flag is off, `lifecycle.start()` returns immediately; when on, it creates the
  engine indexes and a recovery loop — no external calls.
- **Credentials / live data in development**: the repo's local `.env` points at the production
  Atlas database. All Week-1 test and E2E runs used explicit overrides
  (`week1-e2e/litportal-test.env`, `litportal-e2e.env`: local Docker Mongo, local Identity, empty
  keys except the interpretation model); `load_dotenv` never overrides exported values.
- **Jobs**: no SQS producer, schedule or worker was added. Existing Lambda/ECS packages and their
  tests are unchanged (see the handoff report for their test runs).
- **Telemetry**: the engine writes its own traces to `engine_traces` only; no external telemetry.

## 5. Reuse / adapt / replace register (condensed; verified 2026-09-25)

| Asset | Verdict | Why |
|---|---|---|
| `services/phi_guard.py` | **Reuse as-is** (+ engine adds ages > 89) | Deterministic, 27 tests. Misses "94-year-old" (HIPAA identifier) — added in the engine, shared guard untouched. |
| `services/spelling_normalizer.py` | Reuse | Deterministic. |
| `services/canonical_identity.py` (`normalize_doi`, `normalize_trial_id`) | **Reused** | Pure, 19 tests. Engine adds PMID/PMCID validation and trailing-punctuation stripping. |
| `services/source_permission_resolver.py`, `services/downstream_eligibility.py` | **Reused** (fail-closed use) | The known fail-open is in a caller that skips the gate when sources are empty; the engine always calls it. |
| `summarization/verifier.py` mechanical floor | Reuse later (W3.4) | Deterministic citation checks. |
| `services/clinical_question_map.py` | Adapt (pattern only) | Good schema style, but built from PICOTT; no versions, provenance or "age not stated". New `contracts/question.py` replaces it for the new path. |
| `services/m4_full_question_interpreter.py` | Adapt (pattern only) | Strict schema + deterministic validator + scrub → the engine's model-reading verifier follows this pattern. |
| `services/query_compiler/`, `concept_block_query_builder.py` emitters | Adapt in W2.1 | Keep validator/filters; feed the new map. Silently caps anchors at 4 and truncates at 1,500 chars today. |
| `services/llm_provider.py` | **Not reused** by the engine | Returns no token usage and hides failure causes → fails W1.2 budget criteria. Engine has `gateway.py` on the same env; the legacy wrapper is untouched (the summary Lambda depends on it). |
| `services/jobs.py` | **Not reused** as orchestrator | Status-only cancel (a late `succeed()` overwrites `canceled`), non-atomic dedup, no resume, `sweep_stale_jobs` never called. |
| `services/clinical_question_parser.py`, `picott_extractor.py`, `interpretation_canonical.py`, `question_scope_classifier.py` | **Retire from new path** | "impact of X on Y" → prognosis; conditions appended to outcomes; "what is the impact…" → background; 8 independent question derivations. |
| `services/deduplication.py`, `pipeline._upsert_article`, OpenAlex title matcher | **Retire from new path** | Merge on any single identifier and keep the longer abstract → mixed records (reproduced). Replaced by `identity/resolver.py`. |
| `services/search_benchmark*.py`, `benchmark_runs.py` | Retire as evidence | Persisted recall/MRR always 0.0 (key mismatch), "live" runs use fixtures, placeholder PMIDs are real 1970s papers. W1.5 set replaces them. |
| `services/session_reaper.py` | Reuse (interim) | Legacy sessions only. |
| `/api/workflow/*` (frontend GuidedWorkflowWizard) | Not present | The wizard is dead code; the backend never had these routes. |

## 6. One runtime and storage design (decision)

- **Runtime**: one in-process asyncio orchestrator inside the LitPortal API service, with Mongo as
  the source of truth and task leases for multi-replica safety. No new service, queue or
  infrastructure in Week 1. The task handlers are plain async functions over a persisted run, so a
  later SQS/Fargate worker can drive the same state machine (`runner.drive(run_id)`) without
  changing the contracts.
- **Storage**: the existing LitPortal database (`engine_*` collections).

## 7. Seed-account risk — VERIFIED, remediation NOT performed (needs owner approval)

- `SEED_TEST_USERS` defaults to **true** (`core/config.py`); seeding runs when `APP_ENV` is not
  `production` (`server.py` lifespan). `APP_ENV` defaults to `development`.
- The AWS CDK sets neither variable for litportal-backend → **by the code, every AWS start seeds**
  `clinician@test.com` / `ClinicianTest123!` (role user) and **`admin@test.com` / `AdminTest123!`
  (role admin)** unless they already exist.
- With `LITPORTAL_USE_IDENTITY=true`, `core/security.py` links an Identity token to an existing
  LitPortal user **by email**, and Identity signup does not require email verification before issuing
  tokens. Someone who registers `admin@test.com` at Identity could therefore be mapped onto the
  seeded **admin** row.
- **Not verified live** (no AWS/DocumentDB access from this machine): whether those rows exist in
  dev/prod right now.
- **Recommended** (owner decision): (1) set `APP_ENV=production` and `SEED_TEST_USERS=false` in the
  CDK task env for prod (and dev unless testers rely on the seeded logins); (2) check both
  databases for the two emails and delete or re-role them; (3) change the code default to off; (4) stop
  linking Identity users by unverified email (tracked separately as the account-takeover finding).

## 8. Per-run limits (defaults; all env-overridable)

Interpretation: ≤ 900 output tokens and 20 s per model call, ≤ 60 interpretations per user per hour
(then rules-only), ≤ 20 versions per question, ≤ 1,000 characters. Run: ≤ 20 tool calls,
≤ 20,000 LLM tokens, ≤ 30 min wall time, ≤ 3 attempts per task; delegated evidence set ≤ 8 records.

## 9. Owner decisions needed (Day-2 items the scope assigns to the owner)

1. **Reference set**: the client's "eight-work / nine-record" steroid reference list is not in the
   workspace — needed for W1.5 must-find labels (candidate identities are in the benchmark plan).
2. **Second steroid question**: only one is in the workspace; the ARDS timing/dose question is the
   best candidate — confirm or supply.
3. **Reviewer plan**: physician adjudicator + second reviewer for held-out cases.
4. **Held-out minimum and Precision@5 target** — confirm before any tuning.
5. **Staging Identity/LitHub access** for W3.5/W4.4 (real, not mocked adapters).
6. **Seed-account remediation** (§7).
7. **Capacity**: Week 1 foundation is complete; the scope's 20-day plan remains ambitious for one
   developer (earlier estimate ~95 dev-days across the combined register).
