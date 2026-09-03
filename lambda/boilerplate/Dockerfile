FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy only dependency files first for better Docker caching
COPY pyproject.toml uv.lock* ./

RUN uv export -o requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY src/ ./src/

# Command to run the Lambda handler
CMD ["src.main.handler"]
