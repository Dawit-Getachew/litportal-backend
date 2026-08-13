"""Lambda handler for ``litportal.oa_full_text_fetch`` (amendment A2).

Built from Loka's boilerplate (PR #1). The two rules that matter are kept:

* **No try/except around message processing.** Catching an error and returning
  200 makes Lambda mark the invocation successful, SQS never retries, and the
  message never reaches the DLQ. Exceptions propagate on purpose.
* The handler stays thin. All logic lives in ``jobs.summary_generation.run``,
  which is AWS-neutral and runs identically from a test or a local script.

``PermanentInputError`` is the one deliberate exception to "just raise": a
malformed message can never succeed, so re-raising it immediately (rather than
after ``maxReceiveCount`` retries) sends it to the DLQ tagged, on the first
pass. It is still raised — never swallowed.
"""

import asyncio
import json
import logging

from jobs.contracts import PermanentInputError
from jobs.oa_full_text_fetch import run

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    records = event.get("Records", [])
    logger.info("Processing %d SQS record(s)", len(records))

    for record in records:
        message_id = record["messageId"]
        body = json.loads(record["body"])  # raises → Lambda retries → DLQ
        logger.info("Processing message %s", message_id)
        process_message(body)

    return {"statusCode": 200}


def process_message(body: dict) -> None:
    try:
        result = asyncio.run(run(body))
    except PermanentInputError:
        # Unusable message: no retry can fix it. Log loudly and re-raise so it
        # lands in the DLQ on the first receive instead of after 5 attempts.
        logger.exception("permanent_input_error — routing straight to the DLQ")
        raise
    logger.info(
        "status=%s article_id=%s oa_status=%s sections=%d",
        result.status,
        result.article_id,
        result.oa_status,
        result.sections_created,
    )
