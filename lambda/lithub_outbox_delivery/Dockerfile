# Build context is the REPO ROOT (litportal-backend/), not this directory:
#   docker build -f lambda/lithub_outbox_delivery/Dockerfile -t litportal-summary-generation .
# The Makefile does this for you. --platform is pinned per Loka's boilerplate:
# on Apple Silicon an ARM image builds fine locally and fails on Lambda x86.
#
# The proposal's rule is that the worker runs the ORIGINAL business logic
# unchanged, so this image carries the application packages the job imports
# (routers/services/core/...) rather than a re-implementation. The handler and
# the AWS-neutral entry point in jobs/ are the only migration-specific code.
FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Application dependencies (same file the API image installs from).
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Handler + AWS-neutral job layer.
COPY lambda/lithub_outbox_delivery/src/ ./src/
COPY jobs/ ./jobs/

# Application packages the job's logic imports.
COPY core/ ./core/
COPY models/ ./models/
COPY clients/ ./clients/
COPY services/ ./services/
COPY routers/ ./routers/
COPY agents/ ./agents/

CMD ["src.main.handler"]
