import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    records = event.get("Records", [])
    logger.info("Processing %d SQS record(s)", len(records))

    for record in records:
        message_id = record["messageId"]
        body = json.loads(record["body"])  # raises JSONDecodeError → Lambda retries → DLQ
        logger.info("Processing message %s", message_id)

        # TODO: implement your logic here
        process_message(body)

    return {"statusCode": 200}


def process_message(body: dict) -> None:
    # TODO: implement your message processing logic
    pass