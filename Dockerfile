# LitPortal Backend (FastAPI + MongoDB/Motor).
#
# Runs from the repo root: server.py imports its siblings as top-level modules
# (server:app, NOT backend.server:app), so the whole tree is copied to /app.
# Multi-stage build with uv for a small, reproducible runtime image that runs as
# a non-root user.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# build-essential covers any wheels that need compilation at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache -r requirements.txt


FROM python:3.12-slim AS runtime

# curl is used by the container HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app . /app/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/health" || exit 1

# Shell form so ${HOST}/${PORT} expand from the environment.
CMD ["sh", "-c", "exec uvicorn server:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
