"""
offline/build_training_set.py - turn the whole history into a table the model
can learn from.

Same events, same feature code as the live stream (core/features.py), just fed
in bulk from an in-memory store instead of one at a time from Redis. That is
the point: tests/test_parity.py proves this shortcut changes none of the
numbers, so the model trains on exactly what it will be scored on.

Events are replayed in TransactionDT order and each row's features are computed
from ONLY the rows before it - so the resulting table is safe to split by time.

    uv run python -m offline.build_training_set --limit 5000   # quick check
    uv run python -m offline.build_training_set                # the real thing

IMPORTANT: split this table by TIME, never randomly. A random split lets the
model peek at future transactions of the same card, and the score it reports
will be a lie.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from core.features import FEATURE_NAMES, FeatureEngine
from core.schema import LABEL_FIELD, from_csv_row
from core.store import InMemoryStore
from ingest.producer import load_data

OUTPUT = "data/training_set.parquet"


def build(limit: int | None) -> pd.DataFrame:
    df = load_data(limit)          # already sorted by TransactionDT
    print(f"Loaded {len(df):,} transactions in event-time order.")

    engine = FeatureEngine(InMemoryStore())
    rows = []
    skipped = 0
    start = time.time()

    for raw in df.to_dict("records"):
        event = from_csv_row(raw)
        if event is None:          # no card1 -> no grouping key -> unusable
            skipped += 1
            continue

        features = engine.process(event)

        # Keep the id and the timestamp alongside the features: the timestamp
        # is what makes an honest train/test split possible later.
        rows.append({
            "TransactionID": event.TransactionID,
            "TransactionDT": event.TransactionDT,
            **features,
            LABEL_FIELD: raw.get(LABEL_FIELD),
        })

        if len(rows) % 100000 == 0:
            rate = len(rows) / (time.time() - start)
            print(f"  built {len(rows):,}  ({rate:,.0f}/sec)")

    elapsed = time.time() - start
    print(f"Built {len(rows):,} rows, skipped {skipped:,}, in {elapsed:.1f}s "
          f"({len(rows) / elapsed:,.0f}/sec)")
    return pd.DataFrame(rows, columns=["TransactionID", "TransactionDT",
                                       *FEATURE_NAMES, LABEL_FIELD])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the training table.")
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N rows (for quick tests)")
    parser.add_argument("--out", default=OUTPUT, help="where to write the parquet")
    args = parser.parse_args()

    out = build(args.limit)
    out.to_parquet(args.out, index=False)

    fraud_rate = 100 * out[LABEL_FIELD].mean()
    span_days = (out["TransactionDT"].max() - out["TransactionDT"].min()) / 86400
    print(f"\nWrote {args.out}")
    print(f"  {len(out):,} rows x {len(out.columns)} columns")
    print(f"  fraud rate {fraud_rate:.2f}%   spanning {span_days:.0f} days")


if __name__ == "__main__":
    main()
