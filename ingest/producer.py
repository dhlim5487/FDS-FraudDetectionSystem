"""
ingest/producer.py - replay the dataset into Kafka in event-time order.

We read the historical CSV, sort by TransactionDT (the real order things
happened), and send each transaction to the 'transactions' topic. A speed
multiplier lets six months of history replay in about an hour.

This is exactly what real fraud teams do to backfill features and test new
models against past traffic. Replaying in event order is legitimate; the only
sin would be shuffling the rows or letting the future leak in.

    uv run python -m ingest.producer --limit 1000          # quick test
    uv run python -m ingest.producer --speed 1000          # full replay
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer

from core.schema import from_csv_row, EVENT_FIELDS, LABEL_FIELD

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"
TRANSACTION_CSV = "data/train_transaction.csv"
IDENTITY_CSV = "data/train_identity.csv"


def load_data(limit: int | None) -> pd.DataFrame:
    """Load transactions, attach device data, sort into event-time order."""
    # We only read the columns we actually keep, plus the label and the
    # TransactionID needed to join device data. Reading 18 columns instead of
    # 394 makes this load in seconds instead of eating all your memory.
    usecols = list(dict.fromkeys(EVENT_FIELDS + [LABEL_FIELD]))
    # DeviceType/DeviceInfo live in the other file, so drop them from this read.
    usecols = [c for c in usecols if c not in ("DeviceType", "DeviceInfo")]

    txn = pd.read_csv(TRANSACTION_CSV, usecols=usecols, nrows=limit)
    idn = pd.read_csv(IDENTITY_CSV, usecols=["TransactionID", "DeviceType", "DeviceInfo"])

    # LEFT join: keep every transaction. ~75% will have no device row, and
    # those columns stay empty (NaN) on purpose - missing device IS a signal.
    df = txn.merge(idn, on="TransactionID", how="left")

    # THE important line: replay in the order things really happened.
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    return df

"""Kafka calls this back for each message - only shout if it failed."""
def delivery_report(err, msg) -> None:
    if err is not None:
        print(f"  ! delivery failed: {err}")


def run(speed: float, limit: int | None) -> None:
    df = load_data(limit)
    print(f"Loaded {len(df):,} transactions. Replaying at {speed}x into '{TOPIC}'.")

    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

    sent = 0
    skipped = 0
    prev_dt: int | None = None
    wall_start = time.time()

    for row in df.to_dict("records"):
        # Pull the label out BEFORE building the event. The event that rides
        # the belt must never carry the answer. We ship the label on a side
        # channel (Kafka header) so the training pipeline can find it later,
        # but the scoring path will simply ignore it.
        label = row.get(LABEL_FIELD)

        event = from_csv_row(row)
        if event is None:            # no card1 -> unusable, count and skip
            skipped += 1
            continue

        # Pace the replay: if this event happened N seconds after the previous
        # one, wait N/speed seconds. That reproduces the real rhythm of traffic
        # - quiet nights, busy middays - just compressed in time.
        if prev_dt is not None and speed > 0:
            gap = (event.TransactionDT - prev_dt) / speed
            if gap > 0:
                time.sleep(min(gap, 2.0))   # cap so huge gaps don't stall us
        prev_dt = event.TransactionDT

        producer.produce(
            TOPIC,
            key=str(event.card1),                    # same card -> same partition
            value=json.dumps(event.to_dict()),
            headers={"isFraud": str(int(label)) if pd.notna(label) else ""},
            callback=delivery_report,
        )
        producer.poll(0)            # let Kafka flush delivery callbacks
        sent += 1

        if sent % 10000 == 0:
            rate = sent / (time.time() - wall_start)
            print(f"  sent {sent:,}  ({rate:,.0f}/sec)")

    producer.flush()                # wait for everything in-flight to land
    elapsed = time.time() - wall_start
    print(
        f"\nDone. sent={sent:,}  skipped={skipped:,}  "
        f"in {elapsed:.1f}s  ({sent / elapsed:,.0f}/sec average)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay transactions into Kafka.")
    parser.add_argument("--speed", type=float, default=1000.0,
                        help="time compression; 1000 = 1000x faster than real time")
    parser.add_argument("--limit", type=int, default=None,
                        help="only replay the first N rows (for quick tests)")
    args = parser.parse_args()
    run(speed=args.speed, limit=args.limit)


if __name__ == "__main__":
    main()
