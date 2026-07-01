# Deployment Environment Checklist (Emergent)

**Production target:** https://clinician-answers.emergent.host

## Golden rule — secrets never live in the repo
- `backend/.env` is **git-ignored** and must **never** be committed.
- Only `backend/.env.example` (keys-only template, **no secret values**) is
  committed, for documentation.
- **All production secrets and env values are set ONLY in the Emergent Deploy
  environment-variable interface** (shown after you click Deploy). That layer
  overrides any committed values at runtime.

## Set these in the Emergent Deploy env interface (production)
### Behavior / safe-stance flags
```
APP_ENV=production
CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE=disabled
SEED_TEST_USERS=false
DEV_FORCE_ENABLE_PUBMED=false
CORS_ORIGINS=https://clinician-answers.emergent.host
M4_ACTIVE_ENABLED=false
M4_ACTIVE_KILL_SWITCH=true
SEMANTIC_REPAIR_ACTIVE_ENABLED=false
SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH=true
M3_1_BENCHMARK_ENABLED=false
```
### Must remain UNSET or false (never true in production)
```
M4_ACTIVE_PROD_ALLOW_ACTIVE          (false)
SEMANTIC_REPAIR_PROD_ALLOW_ACTIVE    (false)
L1_PREFERENCE_PROD_UI_VISIBLE        (false)
L1_PREFERENCE_PROD_SOFT_BOOST_ALLOWED(false)
COCHRANE_PROD_BOOST_ALLOWED          (false)
```
### Secrets (generate NEW production values; do not reuse dev)
```
JWT_ACCESS_SECRET=<new random>
JWT_REFRESH_SECRET=<new random>
EMERGENT_LLM_KEY=<your universal key>
MONGO_URL=<production MongoDB connection string>
```
### Optional third-party keys (only if directly used)
```
NCBI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENALEX_* / TIGRIS_* ...
```

## Resulting production posture (enforced by code, keyed on APP_ENV=production)
- M4 hard-off; M3.2 (semantic repair active) dormant.
- R-series / R2A / ImpactRank shadow / admin-only; L1 controls hidden.
- Cochrane PubMed query family may retrieve + label PubMed-indexed Cochrane
  reviews; Cochrane **score boost is shadow-only** (not applied).
- `cochrane_pubmed_indexed` registry: `ai_ingestion_allowed=false`; metadata +
  link-out only; **no** Cochrane/Wiley API, **no** scraping, **no** full-text
  ingestion.
- No summary / citation-verifier / connector behavior change.
- Seed test accounts are **not** created in production.
