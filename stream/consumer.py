"""
stream/consumer.py - the live scoring path: take events off the belt, work out
what we know about the card, and leave the answer where a scorer can find it.

Per event:
    1. read this card's recent history from Redis
    2. compute the clues  (core/features.py - the SAME code the offline
       training build uses, which is what keeps train and serve in step)
    3. write the event into history, so it counts for the NEXT transaction

The scoring API does not read a cached feature row: features depend on the
transaction being scored (its amount, its device), so it recomputes them the
same way, from the same history. Caching them would serve last transaction's
answer - exactly the skew this project exists to prevent.

    uv run python -m stream.consumer

Run this in one terminal, then run the producer in another and watch it work.
Stop it any time with Ctrl+C.
"""
from __future__ import annotations

import json
import time

import redis
from confluent_kafka import Consumer

from core.features import FEATURE_NAMES, FeatureEngine
from core.schema import TransactionEvent
from core.store import RedisStore

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"
GROUP_ID = "fds-consumer"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0


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
    # decode_responses: hand us str, not bytes, so core/store.py can parse
    # members without sprinkling .decode() everywhere.
    client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
    )
    client.ping()               # fail loudly now rather than on the first event
    engine = FeatureEngine(RedisStore(client))

    consumer = build_consumer()
    consumer.subscribe([TOPIC])
    print(f"Listening on '{TOPIC}', feature state in Redis db{REDIS_DB}. Ctrl+C to stop.\n")

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

            event = TransactionEvent(**json.loads(msg.value()))
            label = read_label(msg.headers())

            # Kafka can hand us the same message twice after a restart. That is
            # harmless here: the sorted-set member is built from TransactionID,
            # so re-adding an event overwrites its own row instead of
            # duplicating it. The history stays correct either way.
            features = engine.process(event)

            if label == "1":
                fraud += 1
            seen += 1

            # Show the very first event in full, so you can eyeball it.
            if seen == 1:
                print(f"first event off the belt: card1={event.card1} "
                      f"amt={event.TransactionAmt}")
                for name in FEATURE_NAMES:
                    print(f"    {name:<22} {features[name]}")
                print()

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
