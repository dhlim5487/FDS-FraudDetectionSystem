"""
core/features.py - THE feature definitions. The heart of the project.

This code computes the "clues" about each card from its transaction history.
It is written ONCE and used by BOTH:

  * the live stream  (stream/consumer.py)  - one event at a time, as it arrives
  * the offline build (offline/build_training_set.py) - all history, in bulk

Because both paths call this same code, the clues cannot drift apart. That is
what prevents train-serve skew - the single most common bug in production ML.
tests/test_parity.py exists to prove the two paths really do agree.

WHERE the history is kept is a separate question, answered by core/store.py:
the live stream keeps it in Redis (survives a restart), the offline build keeps
it in memory (24x faster). Swapping the store must never change the numbers.

THE GOLDEN RULE (point-in-time correctness):
    When we score a transaction, we may only use information from BEFORE it.
    So we compute features from a card's PRIOR events, then add the current
    event to history AFTERWARDS. Never the other way around.
"""
from __future__ import annotations

import statistics

from core.schema import TransactionEvent
from core.store import InMemoryStore

# Window sizes, in seconds (TransactionDT is measured in seconds).
HOUR = 3600
DAY = 24 * HOUR
WEEK = 7 * DAY

# The clues, in a fixed order. Everything downstream relies on this order,
# so the model always sees the same columns in the same positions.
FEATURE_NAMES = [
    "txn_count_1h",         # how many prior txns on this card in the last hour
    "txn_count_24h",        # ... in the last 24 hours
    "amt_sum_1h",           # total spent in the last hour (prior txns)
    "amt_sum_24h",          # ... in the last 24 hours
    "amt_mean_1h",          # average txn size in the last hour
    "amt_mean_24h",         # ... in the last 24 hours
    "amt_zscore_7d",        # how weird is THIS amount vs this card's 7-day norm
    "secs_since_prev_txn",  # seconds since this card's previous txn (-1 if none)
    "distinct_addr1_24h",   # how many billing regions this card touched in 24h
    "is_new_device",        # 1 if this device was never seen on this card before
]


def compute_features(
    history: list[tuple[int, float, float | None]],
    devices: set[str],
    event: TransactionEvent,
) -> dict[str, float]:
    """
    Turn one card's past into the clues about one event. Pure: no I/O, no
    hidden state, same inputs always give the same numbers. That is what makes
    it safe to share between the live stream and the offline build.

    history: (dt, amount, addr1) rows for this card, every one of them strictly
             BEFORE this event. Rows older than WEEK are filtered out here, so
             the answer never depends on how aggressively the store pruned.
    devices: device strings seen on this card before this event.
    """
    t = event.TransactionDT
    amt = event.TransactionAmt

    # Slice the past into the windows we care about.
    in_1h = [row for row in history if row[0] >= t - HOUR]
    in_24h = [row for row in history if row[0] >= t - DAY]
    in_7d = [row for row in history if row[0] >= t - WEEK]

    amts_1h = [row[1] for row in in_1h]
    amts_24h = [row[1] for row in in_24h]
    amts_7d = [row[1] for row in in_7d]

    # z-score: is this amount unusual for THIS card's recent norm?
    # Needs at least 2 prior points and non-zero spread, else it's 0.
    if len(amts_7d) >= 2:
        mean_7d = statistics.fmean(amts_7d)
        std_7d = statistics.pstdev(amts_7d)
        zscore = (amt - mean_7d) / std_7d if std_7d > 0 else 0.0
    else:
        zscore = 0.0

    # seconds since the previous txn on this card (-1 if this is the first).
    # max() rather than in_7d[-1] so the answer cannot depend on row order -
    # the two stores are free to hand back the same rows in a different order.
    secs_since_prev = (t - max(row[0] for row in in_7d)) if in_7d else -1

    # distinct billing regions in the last 24h, INCLUDING this event's own
    # (its addr1 is known at scoring time, so using it is not leakage).
    addrs = {row[2] for row in in_24h if row[2] is not None}
    if event.addr1 is not None:
        addrs.add(event.addr1)

    # new device? known from the current event, compared to past devices.
    device = event.DeviceInfo
    is_new_device = 1 if (device is not None and device not in devices) else 0

    return {
        "txn_count_1h": float(len(in_1h)),
        "txn_count_24h": float(len(in_24h)),
        "amt_sum_1h": float(sum(amts_1h)),
        "amt_sum_24h": float(sum(amts_24h)),
        "amt_mean_1h": float(statistics.fmean(amts_1h)) if amts_1h else 0.0,
        "amt_mean_24h": float(statistics.fmean(amts_24h)) if amts_24h else 0.0,
        "amt_zscore_7d": float(zscore),
        "secs_since_prev_txn": float(secs_since_prev),
        "distinct_addr1_24h": float(len(addrs)),
        "is_new_device": float(is_new_device),
    }


class FeatureEngine:
    """
    Feed it events in event-time order; it returns each event's clues.

    The store decides where the per-card history lives. Default is in-memory
    (offline builds, tests); pass a RedisStore for the live stream.
    """

    def __init__(self, store=None) -> None:
        self.store = store if store is not None else InMemoryStore()

    def process(self, event: TransactionEvent) -> dict[str, float]:
        """Compute this event's features, THEN fold it into history."""
        history, devices = self.store.load(event.card1, event.TransactionDT, WEEK)
        features = compute_features(history, devices, event)

        # --- ONLY NOW do we remember this event. Order matters: computing
        #     first, then updating, is what enforces point-in-time correctness.
        self.store.append(event)
        return features


# --- self-test: run this file directly to watch clues build up -----------
#     uv run python -m core.features
if __name__ == "__main__":
    def fake(txn_id, dt, amt, addr1=100.0, device=None):
        return TransactionEvent(
            TransactionID=txn_id, TransactionDT=dt, TransactionAmt=amt, card1=777,
            ProductCD="W", card2=None, card3=None, card4=None, card5=None, card6=None,
            addr1=addr1, addr2=None, dist1=None, dist2=None,
            P_emaildomain=None, R_emaildomain=None,
            DeviceType=None, DeviceInfo=device,
        )

    # One card, five transactions marching forward in time.
    events = [
        fake(1, 1000, 50.0, addr1=100, device="phoneA"),   # first ever
        fake(2, 1000 + 60, 50.0, addr1=100, device="phoneA"),   # 1 min later
        fake(3, 1000 + 120, 55.0, addr1=200, device="phoneA"),  # new region
        fake(4, 1000 + 130, 900.0, addr1=200, device="phoneB"), # HUGE amt, new device
        fake(5, 1000 + 90000, 50.0, addr1=100, device="phoneA"),# >1 day later
    ]

    engine = FeatureEngine()
    for e in events:
        f = engine.process(e)
        print(f"txn {e.TransactionID}  amt={e.TransactionAmt:>6}  ->  "
              f"count_1h={f['txn_count_1h']:.0f}  "
              f"z={f['amt_zscore_7d']:+.2f}  "
              f"secs_since_prev={f['secs_since_prev_txn']:.0f}  "
              f"regions_24h={f['distinct_addr1_24h']:.0f}  "
              f"new_device={f['is_new_device']:.0f}")
