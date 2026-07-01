# Running litportal-backend

FastAPI + MongoDB (Motor) service. The ASGI app is **`server:app`** and must be
run **from this directory** (its modules import as top-level packages — `core`,
`routers`, `services`, …). All HTTP routes live under `/api`.

## Prerequisites
- Python 3.10+ (3.12 recommended) and a reachable MongoDB, **or** Docker.
- `MONGO_URL` and `DB_NAME` are **required** — the process raises at import if
  they are unset.

## 1. Configure
```bash
cp .env.example .env
# At minimum set MONGO_URL and DB_NAME. In production also set
# JWT_ACCESS_SECRET, JWT_REFRESH_SECRET, and an explicit CORS_ORIGINS.
```

## 2a. Run with Docker (brings up Mongo too)
```bash
docker compose up --build
# API on http://localhost:8000  (health: http://localhost:8000/api/health)
# Docs: http://localhost:8000/api/docs
```

## 2b. Run locally (uvicorn)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Point at a local/remote Mongo:
export MONGO_URL=mongodb://localhost:27017 DB_NAME=litportal   # PowerShell: $env:MONGO_URL=...
uvicorn server:app --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl -fsS http://localhost:8000/api/health      # -> {"status":"ok"}
```

## 3. Enable the Scienthesis integration (optional, default OFF)
With the flags below unset, the backend runs exactly like the standalone
Emergent backend (native auth + local Mongo library). To wire it into the
platform, set in `.env`:
```
LITPORTAL_USE_IDENTITY=true
LITPORTAL_DUAL_WRITE_LITHUB=true
IDENTITY_BASE_URL=http://localhost:8100
IDENTITY_JWKS_URL=http://localhost:8100/.well-known/jwks.json
IDENTITY_JWT_ISSUER=scienthesis-identity
IDENTITY_JWT_AUDIENCE=litportal
LITHUB_BASE_URL=http://localhost:8200
SERVICE_TOKEN_SECRET=<same secret used by identity + lithub>
```
- **Auth (handoff):** the `/api/auth/*` endpoints (signup, login, request-otp,
  verify-otp, refresh, logout) **delegate to Identity** and return Identity
  RS256 tokens — so a login on LitPortal also works on LitPulse. `get_current_user`
  validates Identity tokens offline via JWKS and lazy-provisions a Mongo shadow
  user keyed by the Identity `sub`.
- **Library (handoff):** saves/deletes mirror to LitHub (background outbox + 60s
  sweeper) keyed by the `sub`; listings best-effort merge the central library.

Optional cutover bridge: set `LITPULSE_JWT_ENABLED=true` + `LITPULSE_JWT_SECRET_KEY`
(= LitPulse's signing secret) so LitPulse-issued tokens are also accepted until
LitPulse fully moves to Identity.

See `INTEGRATION_PLAN.md` for the full design.

## 4. Tests
```bash
# Integration-client unit tests (no Mongo / no live services needed):
pytest tests/test_scienthesis_integration.py -q

# Full suite (needs a reachable Mongo; uses an isolated per-run test DB):
pytest -q
```
