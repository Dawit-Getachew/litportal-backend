"""Pytest fixtures: spin up an HTTPX async client against the FastAPI app
with an isolated test database. Each test session uses a unique DB name so
tests don't collide with the running dev backend.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# Ensure the backend dir is on sys.path so `core`, `routers`, ... import cleanly
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Set test DB env BEFORE importing the app so settings pick it up.
_TEST_DB_NAME = f"clinevidence_test_{uuid.uuid4().hex[:8]}"
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ["DB_NAME"] = _TEST_DB_NAME
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("SEED_TEST_USERS", "true")
os.environ.setdefault("JWT_ACCESS_SECRET", "test-access-secret-do-not-use-in-prod")
os.environ.setdefault("JWT_REFRESH_SECRET", "test-refresh-secret-do-not-use-in-prod")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("JWT_ACCESS_TTL_MINUTES", "60")
os.environ.setdefault("JWT_REFRESH_TTL_DAYS", "7")
# Make login rate limit non-blocking during tests.
os.environ.setdefault("LOGIN_RATE_LIMIT_MAX", "1000")
os.environ.setdefault("SEED_CLINICIAN_EMAIL", "clinician@test.com")
os.environ.setdefault("SEED_CLINICIAN_PASSWORD", "ClinicianTest123!")
os.environ.setdefault("SEED_ADMIN_EMAIL", "admin@test.com")
os.environ.setdefault("SEED_ADMIN_PASSWORD", "AdminTest123!")

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

# Importing the app triggers env loading.
from server import app  # noqa: E402
from core.db import get_client  # noqa: E402


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest_asyncio.fixture(scope="session")
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        # Trigger startup (lifespan) once.
        async with app.router.lifespan_context(app):
            yield c
    # Cleanup the test DB at end of session.
    cli = get_client()
    await cli.drop_database(_TEST_DB_NAME)
