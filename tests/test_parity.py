"""
tests/test_parity.py - proof that the two measuring cups are the same size.

The whole project rests on one claim: the features the model TRAINS on and the
features it SCORES on are computed by the same code, so they cannot drift.
core/features.py holds that code; core/store.py gives it two different places
to keep the raw history (memory for the offline build, Redis for the live
stream). This file proves that swapping the store changes nothing.

If someone later "optimises" one path and forgets the other, these tests fail
before the model silently starts making bad decisions in production.

    uv run pytest tests/ -v

Uses Redis db 15 and flushes it, so the live data in db 0 is never touched.
"""
from __future__ import annotations

import pytest
import redis

from core.features import FEATURE_NAMES, DAY, HOUR, WEEK, FeatureEngine
from core.schema import TransactionEvent
from core.store import InMemoryStore, RedisStore


TEST_DB = 15


@pytest.fixture
def redis_client():
    """A clean, isolated Redis. Skips the test if Redis is not running."""
    client = redis.Redis(host="localhost", port=6379, db=TEST_DB,
                         decode_responses=True)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis is not running - start it with docker compose up -d")
    client.flushdb()
    yield client
    client.flushdb()


def evt(txn_id, dt, amt, card=100, addr1=1.0, device=None, id_31=None):
    return TransactionEvent(
        TransactionID=txn_id, TransactionDT=dt, TransactionAmt=amt, card1=card,
        ProductCD="W", card2=None, card3=None, card4=None, card5=None, card6=None,
        addr1=addr1, addr2=None, dist1=None, dist2=None,
        P_emaildomain=None, R_emaildomain=None,
        DeviceType=None, DeviceInfo=device,
        id_31=id_31
    )


def run_through(store, events):
    """Feed events to an engine in order, collect every feature row."""
    engine = FeatureEngine(store)
    return [engine.process(e) for e in events]


def assert_same(offline_rows, online_rows, events):
    """Compare row by row so a failure names the exact event and feature."""
    assert len(offline_rows) == len(online_rows)
    for event, offline, online in zip(events, offline_rows, online_rows):
        assert set(offline) == set(FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert offline[name] == online[name], (
                f"MISMATCH on TransactionID={event.TransactionID} "
                f"card1={event.card1} feature={name}: "
                f"offline(memory)={offline[name]!r} online(redis)={online[name]!r}"
            )


# --- the edge cases most likely to break parity --------------------------
EDGE_CASE_EVENTS = [
    # card 100: walks across every window boundary
    evt(1, 0, 10.0, device="A"),                        # first transaction: no history
    evt(2, 0, 20.0, device="A"),                        # TIE on TransactionDT
    evt(3, HOUR, 30.0, addr1=2.0, device=None),         # exactly on the 1h edge
    evt(4, HOUR + 1, 40.5, addr1=None, device="B", id_31="ie 11.0 for tablet"),   # addr1 missing, new device, id_31 present
    evt(5, DAY, 50.25, addr1=3.0, device="A"),          # exactly on the 24h edge
    evt(6, DAY + 1, 7.77, addr1=3.0, device="C"),
    evt(7, WEEK, 99.99, addr1=1.0, device="A"),         # 7d edge: prunes the start
    evt(8, WEEK + DAY, 1.0, addr1=None, device=None),   # long gap, nothing known

    # card 200: interleaved, to prove cards do not leak into each other
    evt(9, 5, 500.0, card=200, addr1=9.0, device="Z"),
    evt(10, 10, 500.0, card=200, addr1=9.0, device="Z"),  # zero spread -> z=0
    evt(11, 20, 500.0, card=200, addr1=9.0, device="Z"),

    # card 300: one lonely transaction
    evt(12, 12345, 3.5, card=300, addr1=None, device=None),
]


def test_edge_case_parity(redis_client):
    """Hand-picked events: window edges, ties, missing fields, multiple cards."""
    offline = run_through(InMemoryStore(), EDGE_CASE_EVENTS)
    online = run_through(RedisStore(redis_client), EDGE_CASE_EVENTS)
    assert_same(offline, online, EDGE_CASE_EVENTS)


def test_real_data_parity(redis_client):
    """
    The same check against real IEEE-CIS rows, in true event-time order.
    """
    pd = pytest.importorskip("pandas")
    from pathlib import Path

    if not Path("data/train_transaction.csv").exists():
        pytest.skip("IEEE-CIS data not present in data/")

    from core.schema import from_csv_row
    from ingest.producer import load_data

    df = load_data(limit=3000)
    events = [e for e in (from_csv_row(r) for r in df.to_dict("records"))
              if e is not None]
    assert len(events) > 1000, "expected a decent sample to test against"

    offline = run_through(InMemoryStore(), events)
    online = run_through(RedisStore(redis_client), events)
    assert_same(offline, online, events)

    """To prevent learning from data never showed in the presence datasets """
def test_no_future_leakage():
    """
    The golden rule, checked directly: a card's very first transaction must see
    an empty past, no matter how enormous it is. If features were computed
    AFTER folding the event into history, count_1h would be 1 here instead of 0
    and the model would learn from an answer it will never have in production.
    """
    engine = FeatureEngine(InMemoryStore())
    first = engine.process(evt(1, 1000, 9999.0, device="A"))

    assert first["txn_count_1h"] == 0.0
    assert first["txn_count_24h"] == 0.0
    assert first["amt_sum_24h"] == 0.0
    assert first["secs_since_prev_txn"] == -1.0
    assert first["amt_zscore_7d"] == 0.0     # nothing to compare against yet

    # the second transaction now sees exactly one prior event - and only one
    second = engine.process(evt(2, 1060, 10.0, device="A"))
    assert second["txn_count_1h"] == 1.0
    assert second["amt_sum_1h"] == 9999.0
    assert second["secs_since_prev_txn"] == 60.0