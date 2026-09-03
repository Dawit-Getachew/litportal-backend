# `litportal.summary_generation` — SOW1 Phase 1 Lambda

The pathfinder package for the SQS migration. Built from Loka's boilerplate
(PR #1 `gists/python-lambda-boilerplate`), which stays in `lambda/boilerplate/`
as the template for the remaining jobs.

| | |
|---|---|
| Queue | `scienthesis-summary-generation` → `settings.SUMMARY_GENERATION_QUEUE_URL` |
| Contract | `SOW1_EVENT_CONTRACTS.md` §3.1, `schema_version` 1.0 |
| Recommended timeout | 900 s (worst case ~360 s) · visibility 5400 s |
| BatchSize | 1 |
| Dedup key | `summary_generation#{user_id}#{session_id}` (Redis `SET NX`) |

## Layout

```
src/main.py      thin SQS handler — no try/except around processing
../../jobs/      AWS-neutral contracts + entry point (shared with the API)
events/          realistic SQS event that matches the contract
tests/           runs with no AWS, no database
```

The handler does three things: unwrap `Records`, `json.loads` the body, call
`jobs.summary_generation.run`. Everything else is the application's existing
code — the migration moves *where* the job runs, not what it does.

## Local development (no AWS account needed)

```bash
make test            # unit tests, no Docker
make test-container  # build, run, invoke, tear down
make run             # start the container on :9000
make send-event      # POST events/sqs_event.json to it
make logs
make stop
```

Build context is the **repo root** so the image can carry `jobs/` and the
application packages; the Makefile handles this.

## Why there is no try/except

Catching an exception and returning 200 tells Lambda the invocation succeeded,
so SQS deletes the message and the DLQ stays empty forever. Failures propagate
on purpose. The single exception is `PermanentInputError` — it is caught only
to log it, then **re-raised**, so an unusable message reaches the DLQ on the
first receive instead of after `maxReceiveCount` pointless retries.

`RunResult.status` describes *message handling*, not the business outcome: a
job that fails validation records that failure on its own job row and the
message is still ACKed (retrying would not help). Only transient faults raise.

## Verified

- `pytest tests/` → **15 passed** with no AWS and no database: sample event
  matches the contract, dedup suppression, claim-key shape, multi-record and
  empty batches, invalid JSON raises, transient failure propagates *and*
  releases its dedup claim, bad envelope/missing fields → `PermanentInputError`.
- Container build (`linux/amd64`) succeeds and the Lambda RIE invocation of
  `events/sqs_event.json` against a live MongoDB returns `{"statusCode": 200}`
  in ~2.5 s, with `summary_generation_started` → `summary_generation_succeeded`
  in the logs, exercising the real `routers.summary` code path.
  *Caveat:* that smoke run used a synthetic job id and session id, so the Mongo
  writes were no-ops. It proves packaging, imports, dependencies, contract
  handling and DB connectivity — **not** a full summary generation, which
  additionally needs seeded session data and LLM credentials.

## Not done yet (needs Loka)

Queue/DLQ creation, the event-source mapping, the Redis endpoint for the dedup
store (`JobStateStore` is injected — a no-op today), and IAM. The producer-side
`sqs.send_message` change is deliberately **not** wired: per the SOW that lands
only once this contract is approved in writing.
