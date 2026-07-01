# LitPortal AI Agents

Three **distinct, advisory-only** LLM sidecars for the clinical search pipeline.
Each is its own module, feature-flagged (default OFF), fail-closed (any failure
is a no-op), and bound by the scope's governance: **agents never call sources,
select/exclude/reorder deterministic results, generate PMIDs/DOIs/citations, or
bypass permission checks.** They run only on PHI-screened input (the router runs
`phi_guard.screen()` before the pipeline starts).

| # | Agent | Module | Scope | Runs | Effect | Flag |
|---|-------|--------|-------|------|--------|------|
| 1 | **QueryCategorizer** | `agents/query_categorizer.py` | pre-parser enrich | after `clinical_question_parser.parse()` | refines a weak `clinical_question_type` (`"other"`) at high confidence; stashes advisory entities/terms | `AI_QUERY_CATEGORIZER_ENABLED` |
| 2 | **QuestionInterpreter** | `agents/question_interpreter.py` | understanding (B3) | after PICOTT repair, before compiler | stores interpretation diagnostics; **never** alters retrieval | `AI_CONTEXT_AUGMENT_ENABLED` |
| 3 | **EvidenceCurator** | `agents/evidence_curator.py` | surfacing + ranker opt (B4/B4a) | after deterministic rank+exclude | writes advisory candidates to a **separate** `ai_surfaced_articles` shelf; broader-topic mapping; parallel-search recommendation | `AI_EVIDENCE_SURFACING_ENABLED` |

## Why each exists
1. **QueryCategorizer** — the deterministic `_classify_type` is keyword-gated, so a
   drug/clinical question lacking trigger words falls to `"other"` and mis-routes.
   This agent corrects the *type* only when the deterministic answer was the
   catch-all `"other"` and the agent is confident (`AI_QUERY_CATEGORIZER_MIN_CONFIDENCE`).
   The deterministic parser still runs and still wins otherwise.
2. **QuestionInterpreter** — produces intent / broad-vs-specific / review-or-guideline
   preference / missing & fused PICO slots as **diagnostics** for reviewers. Pure
   storage; it does not touch the planner, compiler, ranker, or selection.
3. **EvidenceCurator** — reviews the deterministically *excluded* (soft) candidates
   (capped / `relevance_below_60` / `negative_concept_fired`) and surfaces the
   genuinely-relevant ones onto a separate shelf with advisory labels
   (direct / borderline / indirect). **Retracted / preprint / unsafe / non-article
   exclusions are never surfaced.** It references candidates by integer index only
   (cannot fabricate identifiers) and never mutates the deterministic lists.

## Shared infra
- `agents/base.py` — `run_json_agent(...)` (LLM call + strict Pydantic validation,
  fail-closed → `(model|None, reason_code)`) and `store_agent_decision(...)`.
- `agents/schemas.py` — closed-set, self-clamping output models.
- Storage: session-level advisories → `ai_agent_decisions` (30-day TTL); per-article
  surfacing → `ai_surfaced_articles` (30-day TTL). Closed-set audit events per run.

## Wiring (hooks in `services/pipeline.py::run_pipeline`)
- Agent 1: right after `clinical_question_parser.parse()`.
- Agent 2: right after the repaired-PICOTT replaces `picott_payload`.
- Agent 3: right after the ranked-article persistence loop.
All three hooks are wrapped in `try/except` (non-fatal) on top of each agent's own
fail-closed behavior.

## Not yet wired (follow-ups)
- **Router exposure** of the `ai_surfaced_articles` shelf in the `GET /api/search-sessions/{id}`
  response (frontend F4 "Preview Evidence Shelf"). The data is stored; the response
  field is the remaining step.
- **Parallel AI search execution** (B4a part 3): the curator *recommends* it via
  `parallel_ai_search_recommended`; actually running a supplemental search and merging
  candidates is gated behind `AI_RANKER_PARALLEL_SEARCH_ENABLED` and not yet implemented.
- **Source-permission gating (B1)**: when the granular `ai_*_allowed` flags land,
  filter agent inputs/outputs by per-source permission.
