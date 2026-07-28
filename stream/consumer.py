"""
stream/consumer.py - stand at the end of the Kafka belt and take events off.

For Phase 1 this is deliberately simple: consume each transaction, count it,
and print a heartbeat. No features yet - that is Phase 2. The only goal here
is to prove the belt runs end to end.

    uv run python -m stream.consumer

Run this in one terminal, then run the producer in another and watch the
count climb. Stop it any time with Ctrl+C.
"""
from __future__ import annotations

import json
import time

from confluent_kafka import Consumer

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"
GROUP_ID = "fds-consumer"


def build_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        # Where to start if this group has never read before: the very
        # beginning, so we see everything the producer already sent.
        "auto.offset.reset": "earliest",
    })


def read_label(headers) -> str:
    """The label rides as a Kafka header, not inside the event body."""
    if not headers:
        return ""
    for key, value in headers:
        if key == "isFraud":
            return value.decode() if value else ""
    return ""


def run() -> None:
    consumer = build_consumer()
    consumer.subscribe([TOPIC])
    print(f"Listening on '{TOPIC}'. Ctrl+C to stop.\n")

    seen = 0
    fraud = 0
    start = time.time()

    try:
        while True:
            msg = consumer.poll(1.0)          # wait up to 1s for a message
            if msg is None:                   # nothing arrived this second
                continue
            if msg.error():
                print(f"  ! consume error: {msg.error()}")
                continue

            event = json.loads(msg.value())
            label = read_label(msg.headers())
            if label == "1":
                fraud += 1
            seen += 1

            # Show the very first event in full, so you can eyeball it.
            if seen == 1:
                print("first event off the belt:")
                print(" ", event, "\n")

            if seen % 10000 == 0:
                rate = seen / (time.time() - start)
                pct = 100 * fraud / seen
                print(f"  seen {seen:,}  fraud {fraud:,} ({pct:.2f}%)  {rate:,.0f}/sec")

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start
        print(
            f"\nStopped. seen={seen:,}  fraud={fraud:,}  "
            f"in {elapsed:.1f}s  ({seen / elapsed if elapsed else 0:,.0f}/sec)"
        )
        consumer.close()             # tell Kafka we are leaving the group


if __name__ == "__main__":
    run()
