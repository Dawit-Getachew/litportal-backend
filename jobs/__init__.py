"""AWS-neutral job contracts and entry points for the SOW1 queue migration.

Nothing in this package imports boto3 or knows it is running in Lambda. Each
job exposes:

* a message dataclass that parses and validates its contract
  (``SOW1_EVENT_CONTRACTS.md`` v1.0), and
* an async ``run(...)`` entry point that executes the job's work.

The Lambda packages under ``lambda/`` are thin adapters: they unwrap SQS
records, call ``parse``/``run`` from here, and let exceptions propagate so SQS
redelivers and the DLQ works. The same entry points can be called from the
FastAPI process, a test, or a local script — no AWS required.
"""

from jobs.contracts import (
    ContractError,
    Envelope,
    PermanentInputError,
    parse_envelope,
)

__all__ = [
    "ContractError",
    "Envelope",
    "PermanentInputError",
    "parse_envelope",
]
