# Lambda Boilerplate

A minimal Python 3.12 Lambda function designed for **SQS-triggered** workloads, with local development using Docker and the AWS Lambda Runtime Interface Client (RIC).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- `make`
- `python3` (for JSON pretty-printing output)
- [uv](https://docs.astral.sh/uv/) (for running unit tests locally)

## Project structure

```
boilerplate/
├── src/
│   └── main.py              # Lambda handler — edit this
├── events/
│   └── test_event.json      # Sample SQS event payload
├── tests/
│   └── test_handler.py      # Unit tests
├── Dockerfile               # Lambda container image (linux/amd64)
├── .dockerignore            # Excludes noise from the build context
├── pyproject.toml           # Python dependencies (managed by uv)
└── Makefile                 # Dev commands
```

## Local development workflow

### 1. Start the Lambda container

```bash
make run
```

Builds the Docker image and starts the container in the background on port 9000.

### 2. Send a test event

```bash
make send-event
```

Edit `events/test_event.json` to change the SQS payload. The response is pretty-printed JSON.

### 3. View logs

```bash
make logs
```

### 4. Stop the container

```bash
make stop
```

## All-in-one test

Builds, runs, invokes, and tears down in a single command — no manual steps needed:

```bash
make test-container
```

## Running unit tests

Unit tests run against the handler directly, no Docker required:

```bash
make test
```

## Adding dependencies

Add packages to `pyproject.toml` under `[project] dependencies`, or by using the command `uv add [package]` then rebuild:

```bash
make clean
make run
```

## Writing your handler

The entry point is `src/main.py`. Each SQS invocation delivers a batch of one or more records. The handler iterates over them and processes each message body (a JSON-encoded string).

```python
def handler(event, context):
    for record in event["Records"]:
        body = json.loads(record["body"])
        process_message(body)   # raises → Lambda retries → DLQ on exhaustion
    return {"statusCode": 200}
```

**Exceptions should propagate.** Do not catch and swallow errors — letting them raise is what triggers SQS retries and eventual DLQ routing. The `context` object provides Lambda runtime metadata (`context.function_name`, `context.aws_request_id`, `context.memory_limit_in_mb`, etc.).

> **Batch size > 1:** If you configure a batch size greater than 1, consider enabling [ReportBatchItemFailures](https://docs.aws.amazon.com/lambda/latest/dg/with-sqs.html#services-sqs-batchfailurereporting) to avoid re-processing already-succeeded records in a partially failed batch.

## Dead Letter Queue (DLQ)

When the handler raises an uncaught exception, SQS retries delivery up to the queue's `maxReceiveCount`. After that, the message is moved to the configured DLQ.

Checklist:
- The source SQS queue has a `RedrivePolicy` pointing to a DLQ
- The Lambda execution role has `sqs:ReceiveMessage`, `sqs:DeleteMessage`, and `sqs:GetQueueAttributes` on the source queue
- An alarm is set on the DLQ's `ApproximateNumberOfMessagesVisible` metric so failures are visible

## Lambda limits (worth knowing)

| Limit | Value |
|---|---|
| Max execution timeout | 15 minutes |
| Memory | 128 MB – 10,240 MB |
| `/tmp` ephemeral storage | 512 MB (configurable up to 10,240 MB) |
| Container image size | 10 GB |
| Concurrent executions (default) | 1,000 per region (soft limit) |

SQS-triggered Lambdas scale by adding concurrent pollers (up to 1,000 per standard queue by default). Factor this into downstream rate limits.

## Platform

The Dockerfile and Makefile target `linux/amd64`. On Apple Silicon this prevents the build producing an ARM image that won't run on Lambda x86. If you deploy to Lambda `arm64` (Graviton2), change `--platform linux/amd64` → `--platform linux/arm64` in both files.

## Deploying to AWS

The `Dockerfile` produces an image compatible with [Lambda container image deployments](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html). Deployment steps (ECR push, IAM roles, Lambda configuration) are covered separately.
