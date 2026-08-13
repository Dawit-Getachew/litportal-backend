# ECS Fargate worker image for the clinician search pipeline (job 2).
# Build context is the REPO ROOT:
#   docker build --platform linux/amd64 -f ecs/clinician_search/Dockerfile -t litportal-clinician-search .
# Triggered by EventBridge Pipe (SQS -> RunTask); the Pipe passes the batch in
# the PIPE_EVENT container-override environment variable.
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

COPY jobs/ ./jobs/
COPY ecs/ ./ecs/
COPY core/ ./core/
COPY models/ ./models/
COPY clients/ ./clients/
COPY services/ ./services/
COPY routers/ ./routers/
COPY agents/ ./agents/

ENV PYTHONPATH=/app
CMD ["python", "-m", "ecs.clinician_search.entrypoint"]
