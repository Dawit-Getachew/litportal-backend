# Clinical evidence engine — Week 1 foundation

New, isolated package for the LitPortal *Clinical Evidence and Answer Engine* scope
(W1.1–W1.5). It adds a **human-in-the-loop question gate** in front of evidence search,
a **durable orchestrator**, and the **typed contracts** the Week 2–4 work builds on.

- Package: `clinical_engine/` (top-level, next to `services/`). Nothing outside it imports it
  except `server.py` (router + lifespan hooks). The Lambda/ECS worker images copy only
  `jobs/ core/ models/ clients/ services/ routers/ agents/`, so they never ship or import it.
- API: `/api/v2/engine/*` (also `/api/litportal/v2/engine/*`). **Additive only** — no existing
  route, request or response shape changed.
- Default **on**. `CLINICAL_ENGINE_ENABLED=false` switches it off: every route except
  `/capabilities` then answers 404, nothing starts at boot, and the frontend falls back to the
  direct-search flow.

## The flow (two-step review)

```
clinician types question
      │
      ▼
POST /questions ──► PHI preflight ──► rules reading ─┐
                                    model reading ───┴─► verify vs. text ─► ClinicalQuestionMap v1
      │                                                   (quotes checked, age never inferred)
      ▼
STEP 1 — check the question: summary + the framework's fields (Clinical Question Map V2:
         PICO, PEO/PECO, diagnostic accuracy, workup, prognosis, PFO, prediction model,
         PICo/SPIDER, PCC, CoCoPop, PIPOH, operational map, SPICE/CIMO)
      ├── edit chips / change question type / retype ──► POST /questions/{id}/revise
      ├── decline ────────────────────────────────────► POST /questions/{id}/decline
      └── approve ────────────────────────────────────► POST /questions/{id}/approve-interpretation {version}
                                                          records the interpretation approval, maps each
                                                          concept to an exact MeSH heading (NCBI) and
                                                          classified wordings (model) — nothing is searched
      ▼
STEP 2 — check the search terms: per concept "must appear", MeSH heading, wordings
         (same meaning / more specific ON when safe; broader / related OFF), a switch to
         turn all alternative wordings off, publication years, animal-only filter,
         sources, record counts (GET …/search-terms/counts — counts only) and the exact
         PubMed query
      ├── edit terms ─────► POST /questions/{id}/search-terms/revise {version, base_terms_version, edits}
      ├── back to step 1 ─ (client-side; a real edit creates a new version, terms are rebuilt)
      ├── decline
      └── run ────────────► POST /questions/{id}/confirm {version, mode, search_terms_version}
                             records Approval bound to the terms version + hash (+ Delegation)
      ▼
run: plan_search → retrieve (session gets engine_concept_plan) → await_retrieval → integrity_check
     [→ delegated_selection]
```

**The approved terms are exactly what is searched.** `services/engine_concept_plan.py` (shipped
in the worker images; plain dicts, no engine import) makes the pipeline use the approved concepts
as its must-have / ranking concepts and anchors, builds the queries from them (one AND block per
"must appear" concept: MeSH OR title/abstract words; Europe PMC gets words only — its MESH field
misbehaves inside OR groups), and skips the parser's ontology lifts, AI expansion, semantic-repair
rewrites and the legacy fallback / thin-pool broadening. Sessions without a plan behave as before.

The one-step `confirm` (no `search_terms_version`) still works for API clients; it runs the
existing pipeline on the question text.

## API (all JSON, Bearer auth, owner-scoped)

| Method | Path | Purpose |
|---|---|---|
| GET | `/capabilities` | Feature detection (no auth). `{enabled, modes, question_types, age_groups, limits, …}` |
| POST | `/questions` `{text}` | Interpret. 201 → question view (`status`: `awaiting_confirmation` \| `needs_clarification` \| `blocked_phi`) |
| GET | `/questions` | Recent questions (summary only) |
| GET | `/questions/{id}` | Question view: interpretation, chips, conflicts, warnings, versions, approval, run |
| POST | `/questions/{id}/revise` `{base_version, text? \| edits?}` | Text re-interpretation or chip edits (`set`, `add`, `remove`, `accept`, `exclude`, `set_type`, `set_importance`). 409 `stale_version` on a stale base |
| POST | `/questions/{id}/approve-interpretation` `{version, use_synonyms?}` | Step 1 → 2: interpretation approval + search terms (MeSH + wordings). `use_synonyms: false` = step-1 "exact terms only" (wordings listed but off). Idempotent; a changed `use_synonyms` after going back is applied as a terms edit |
| POST | `/questions/{id}/search-terms/revise` `{version, base_terms_version, edits}` | Term edits: `set_use_synonyms`, `set_term_enabled`, `add_term`, `remove_term`, `set_mesh_enabled`, `set_required`, `set_publication_years`, `set_exclude_animal_only`. 409 `stale_terms` |
| GET | `/questions/{id}/search-terms/counts?version=&terms_version=` | PubMed / Europe PMC record counts for the terms (no records retrieved) |
| POST | `/questions/{id}/confirm` `{version, mode, search_terms_version?}` | `mode`: `review_papers` \| `answer_for_me`. In step 2 the terms version is required (409 `stale_terms`). Idempotent for the same version+mode; 409 on conflicts / stale version |
| POST | `/questions/{id}/feedback` `{helpful, comment?, run_id?}` | "Were these results helpful?" (PHI-screened comment); searched questions only |
| POST | `/questions/{id}/decline` `{reason}` | `wrong_interpretation` \| `new_question` \| `other` |
| GET | `/questions/{id}/trace` | Safe audit trace (no question text, prompts or evidence text) |
| GET | `/runs/{id}` | Run state, task plan, usage/budget, search plan, integrity report, links |
| POST | `/runs/{id}/pause` \| `/resume` \| `/cancel` | Run control |
| GET | `/runs/{id}/trace` | Run trace |
| GET | `/evidence-sets/{id}` | AuthorizedEvidenceSet (answer-for-me) |
| GET | `/sessions/{session_id}/report?live_counts=` | "How this search was built": approved question reading, MeSH + wordings, the boolean logic handed to the search, every executed query (exact strings + open-in-PubMed/Europe PMC links), papers found per search family / source, raw JSON. 404 for searches not started from the review. `live_counts=true` re-runs the executed main queries as counts |

Errors: `404 {code:not_found}`, `409 {code, message}` (state conflicts), `422 {code, message}`
(input, PHI in edits). Question view example (abridged):

```json
{"question_id": "…", "status": "awaiting_confirmation", "version": 1,
 "question_text": "Impact of corticosteroids on mortality in septic shock",
 "interpretation": {"summary": "Effect of corticosteroids on mortality in patients with septic shock.",
   "question_type": "treatment", "framework": "PICO", "age_group": "not_stated",
   "chips": [{"id": "question_type", "label": "Question type", "display": "Treatment effect", "state": "active", …},
             {"id": "…", "role": "intervention", "label": "Treatment", "display": "corticosteroids", "state": "active", "source": "user_text"},
             {"id": "add:age_group", "label": "Age group", "display": "Not specified", "state": "placeholder", "options": […]}],
   "conflicts": [], "warnings": ["No age group stated — results will not be limited to adults or children. …"],
   "method": "model"},
 "can_confirm": true, "allowed_actions": ["confirm_review_papers", "confirm_answer_for_me", "revise", "decline"],
 "approval": null, "delegation": null, "run": null}
```

## Interpretation (W1.4)

- `interpretation/rules.py` — deterministic reading (always available; CI and model outages).
  Conservative: only what the words support, with the supporting span; negation aware
  ("excluding children", "non-elderly"). Independent of the legacy parser.
- `interpretation/model_reading.py` — model output is untrusted: strict schema, closed type/role
  sets, `stated` elements must quote the question verbatim (else demoted to an inactive
  suggestion), age groups need age words, citation-like values dropped, generic effect words
  ("impact") never become outcomes.
- `interpretation/service.py` — PHI preflight (reuses `services/phi_guard`, plus ages > 89),
  merge (rules authoritative for age/setting/time), per-call token/timeout limits.
- `interpretation/consistency.py` — blocking conflicts: missing core element for the question
  type, intervention = comparator, condition as outcome (treatment/diagnosis/prognosis),
  contradictory or included-and-excluded ages.
- `interpretation/present.py` — chips, the one-line summary (built from active elements only;
  no age → "patients", never "adults"), and edit operations.
- **Material edits** are detected by `material_hash` (type, framework, active elements). A
  no-op edit returns the same version with `unchanged: true`.

## Orchestration (W1.2)

- State in Mongo (`engine_runs`, `engine_tasks`, `engine_traces`, …); one runtime (in-process
  asyncio driver); safe across replicas via **task leases** (`lease_owner`, `lease_expires_at`,
  heartbeat renewal). A dead process's lease expires and the recovery loop (startup + every
  `CLINICAL_ENGINE_RECOVERY_INTERVAL_S`) resumes the run from the task checkpoint.
- **Fenced writes**: a task result commits only if the writer still holds the lease, the task is
  still `running`, and the question version matches. Cancel/supersede flip open tasks to
  `cancelled`, so late results are rejected and traced (`stale_result_rejected`).
- **Tool gateway**: registered tools only; per-task allowlist; approval check for evidence
  sources; atomic per-run budgets (`max_tool_calls`, `max_llm_tokens`) + wall-time budget.
  Blocked calls and budget breaches never retry.
- **Retries** bounded (`CLINICAL_ENGINE_TASK_MAX_ATTEMPTS`, exponential backoff). Cooperative
  waits (`TaskYield`) consume no attempt.
- **Idempotency**: the legacy search is started once per run even across crashes (checkpointed
  intent + adoption of a session created just after the intent, never one owned by another run).

## Contracts (W1.3) — `contracts/`

`ClinicalQuestionMap`, `QuestionElement` (source: `user_text` \| `user_edited` \| `ai_inferred`,
span, confidence, importance, negated), `IntentConflict`; `Publication` (identifiers with
provenance, `identity_status`, `identity_conflicts`, date with precision), `Study` + `StudyReport`
(separate reports of one trial stay separate), `Result`, `Assessment` (framework + target kind +
`source_reported` \| `ai_assisted_provisional` \| `reviewed`; missing information is not a grade),
`AuthorizedEvidenceSet` (selection mode, delegation id, per-item permission basis and input
scope, `ineligible` disclosed separately, `manifest_hash`). Run/task/trace documents in `run.py`.

`identity/resolver.py` — identity-safe union: transitive clustering on same-kind identifiers,
merge only when consistent (one value per identifier kind, compatible titles, years within one),
otherwise **quarantine** every member; fields from the most authoritative source, never the
longest; NCT links reports into a Study without merging them.

## What is live vs. adapter vs. mock in Week 1

| Part | Week 1 state |
|---|---|
| Question interpretation, versions, approval, delegation | **Live** (model via `LLM_PROVIDER`, rules fallback) |
| Task plan (three evidence lanes, concept blocks, filters) | **Live**, recorded for inspection — lanes are **not executed** yet (W2) |
| Retrieval | **Adapter**: starts the existing LitPortal pipeline with the approved question (`legacy_pipeline`), or labelled fixtures (`mock`, demos/tests only) |
| Integrity check | Live on the retrieved set (cross-record identifier checks). Per-record verification against PubMed/Europe PMC is W2.2 |
| Delegated selection (answer for me) | Live, documented policy `delegated_top_ranked_v1` over the legacy results, AI-use permission checked per record (fail-closed) |
| Synthesis + independent verifier | Not in Week 1 (W3.4). The frontend reuses the existing grounded summary on the delegated set |

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `CLINICAL_ENGINE_ENABLED` | `true` | Master switch (`false` = off) |
| `CLINICAL_ENGINE_RETRIEVAL_ADAPTER` | `legacy_pipeline` | or `mock` (fixtures, never live evidence) |
| `CLINICAL_ENGINE_INTERPRETER_MODEL` | provider default | Model for interpretation |
| `CLINICAL_ENGINE_INTERPRETER_MAX_TOKENS` / `_TIMEOUT_S` | 900 / 20 | Per interpretation call |
| `CLINICAL_ENGINE_MAX_QUESTION_CHARS` | 1000 | |
| `CLINICAL_ENGINE_MAX_VERSIONS_PER_QUESTION` | 20 | |
| `CLINICAL_ENGINE_MAX_INTERPRETATIONS_PER_HOUR` | 60 | Per user; beyond it interpretation is rules-only |
| `CLINICAL_ENGINE_RUN_MAX_TOOL_CALLS` / `_LLM_TOKENS` / `_WALL_SECONDS` | 20 / 20000 / 1800 | Per-run budgets |
| `CLINICAL_ENGINE_TASK_MAX_ATTEMPTS` / `_RETRY_BACKOFF_BASE_S` | 3 / 2.0 | |
| `CLINICAL_ENGINE_LEASE_SECONDS` / `_POLL_INTERVAL_S` / `_RECOVERY_INTERVAL_S` | 60 / 3 / 30 | |
| `CLINICAL_ENGINE_DELEGATED_MAX_ITEMS` | 8 | Answer-for-me evidence set size |
| `CLINICAL_ENGINE_TERMS_MODEL` | GPT-4.1 on the configured provider | Model that classifies search terms (a small model mislabels siblings as synonyms); falls back to the interpreter model if the call fails |
| `CLINICAL_ENGINE_TERMS_MAX_TOKENS` / `_TIMEOUT_S` | 1400 / 25 | Per term-classification call |
| `CLINICAL_ENGINE_MESH_TIMEOUT_S` | 12 | NCBI MeSH lookups (uses `NCBI_API_KEY` / `NCBI_EMAIL` when set) |
| `CLINICAL_ENGINE_MAX_TERM_REVISIONS` | 60 | Term edits per question version |

Model provider settings are shared with the rest of the app (`LLM_PROVIDER`, `OPENAI_API_KEY` /
`OPENROUTER_API_KEY`, `OPENAI_MODEL` / `OPENROUTER_MODEL`). Collections are created with their
indexes at startup when enabled (`store.ensure_indexes`). Storage uses only operations Amazon
DocumentDB supports (no transactions, `$expr` or server-side JS).

## Tests

```bash
# local Mongo only — never the production .env values
set -a; . ../week1-e2e/litportal-test.env; set +a
.venv/bin/python -m pytest -o addopts="" tests/test_clinical_engine_*.py
```

`test_clinical_engine_contracts.py` (identity, dates, hashing), `…_interpretation.py`
(rules, model verification, fallbacks, PHI, conflicts, chips), `…_orchestrator.py`
(pause/resume, cancel + stale rejection, crash recovery without duplicate writes, blocked
unapproved/off-allowlist calls, budgets, bounded retries, supersession, delegation),
`…_api.py` (HTTP: auth, ownership, flag, PHI, versioning, idempotent confirm, no search before
approval), `…_benchmark.py` (frozen set discipline), `…_search_terms.py` (V2 frameworks, exact MeSH,
term classification guards, edits, plan → queries, the two-step flow over HTTP, the plan reaching
the search session, whole-blood → red-cell regression).

Benchmark baseline: `python -m clinical_engine.benchmark.run_interpretation_baseline [--model]`.
See `benchmark/BENCHMARK_PLAN.md` and `W1_REGISTER.md`.
