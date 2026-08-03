"""
Card's history is kept.

core/features.py answers "how do we compute the clues". This file answers
"where do we keep the raw material". Splitting them is what lets us have both:

  * InMemoryStore - offline builds and tests. ~61,000 events/sec.
  * RedisStore    - the live stream. ~2,500 events/sec, but the history
                    survives a consumer restart.

Both stores expose the same two methods, so core/features.py never knows or
cares which one it is talking to. The measuring cup stays the same size.

    load(card1, now, window) -> (history, devices)
        history: (dt, amount, addr1) rows for this card, all strictly BEFORE
                 the current event, no older than `window` seconds.
        devices: device strings ever seen on this card.

    append(event) -> None
        Remember this event so it counts as history for the NEXT one.

The two stores MUST behave identically. tests/test_parity.py proves they do.
"""
from __future__ import annotations

from collections import defaultdict, deque

from core.schema import TransactionEvent


class InMemoryStore:
    """Per-card history in plain Python objects. Dies with the process."""

    def __init__(self) -> None:
        # card1 -> deque of (dt, amount, addr1), kept in arrival order
        self._history: dict[int, deque] = defaultdict(deque)
        # card1 -> set of device strings ever seen (never pruned)
        self._devices: dict[int, set] = defaultdict(set)

    def load(self, card1: int, now: int, window: int):
        hist = self._history[card1]
        # Drop anything older than the window to bound memory.
        while hist and hist[0][0] < now - window:
            hist.popleft()
        return list(hist), self._devices[card1]

    def append(self, event: TransactionEvent) -> None:
        self._history[event.card1].append(
            (event.TransactionDT, event.TransactionAmt, event.addr1)
        )
        if event.DeviceInfo is not None:
            self._devices[event.card1].add(event.DeviceInfo)


# --- Redis encoding -------------------------------------------------------
# A Redis sorted set scores each member by TransactionDT, which gives us the
# time window for free. Members must be UNIQUE, so TransactionID leads the
# string - two identical-looking transactions would otherwise collapse into one.
#
# repr() is used for the floats because float(repr(x)) == x exactly in Python 3.
# Any sloppier formatting here would show up as a parity failure.

def _encode(event: TransactionEvent) -> str:
    addr = "" if event.addr1 is None else repr(event.addr1)
    return f"{event.TransactionID}|{event.TransactionDT}|{repr(event.TransactionAmt)}|{addr}"


def _decode(member: str) -> tuple[int, float, float | None]:
    _, dt, amt, addr = member.split("|")
    return (int(dt), float(amt), None if addr == "" else float(addr))


class RedisStore:
    """
    Per-card history in Redis. Survives a restart.

    Costs two network round trips per event - one to read, one to write. They
    cannot be merged: we must see the history before we can compute anything.
    Each round trip is pipelined, so it is 2 trips and not 5.
    """

    def __init__(self, client) -> None:
        self.r = client

    def load(self, card1: int, now: int, window: int):
        cutoff = now - window
        pipe = self.r.pipeline()
        # Prune first, then read - so the read never has to filter.
        # "(" means exclusive: drop scores strictly below the cutoff, matching
        # InMemoryStore's `hist[0][0] < now - window`.
        pipe.zremrangebyscore(f"hist:{card1}", "-inf", f"({cutoff}")
        pipe.zrangebyscore(f"hist:{card1}", cutoff, "+inf")
        pipe.smembers(f"dev:{card1}")
        _, rows, devices = pipe.execute()
        return [_decode(m) for m in rows], set(devices)

    def append(self, event: TransactionEvent) -> None:
        pipe = self.r.pipeline()
        pipe.zadd(f"hist:{event.card1}", {_encode(event): event.TransactionDT})
        if event.DeviceInfo is not None:
            pipe.sadd(f"dev:{event.card1}", event.DeviceInfo)
        pipe.execute()
