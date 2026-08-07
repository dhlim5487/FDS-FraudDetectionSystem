"""
Measure scoring latency.
    uv run uvicorn serving.app:app --port 8000     # in one terminal
    uv run python -m bench.latency                 # in another

Reports p50/p95/p99 rather than an average. An average of 5ms hides a system
that stalls for 3 seconds once every hundred calls; at a million transactions a
day that is ten thousand people waiting three seconds. The tail is the story.

Requests are built from cards that really have history in Redis - scoring a
card the system has never seen is unrealistically cheap, because there is
nothing to read and nothing to compute. Cards are rotated so we measure Redis
doing work rather than Redis serving the same hot key over and over.

/score only reads, so this can be run as many times as you like. Run it twice:
if p99 moves a lot between runs, the sample is too small to quote.

Standard library only - a load-testing dependency would be more setup than the
forty lines it replaces.
"""
from __future__ import annotations

import argparse
import http.client
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import redis

HOST, PORT = "localhost", 8000
WARMUP = 50          # discarded: the first calls pay one-off import/JIT costs


def pick_cards(limit: int = 200) -> list[tuple[int, int]]:
    """(card1, last seen TransactionDT) for cards that actually have history."""
    r = redis.Redis(decode_responses=True)
    cards = []
    for key in r.scan_iter(match="hist:*", count=1000):
        newest = r.zrange(key, -1, -1, withscores=True)
        if newest:
            cards.append((int(key.split(":")[1]), int(newest[0][1])))
        if len(cards) >= limit:
            break
    if not cards:
        raise SystemExit("Redis has no history - run stream.consumer first")
    return cards


def make_body(card: int, last_dt: int, i: int) -> str:
    return json.dumps({
        "TransactionID": 900000 + i,
        "TransactionDT": last_dt + 600,      # a new txn, 10 min after the last
        "TransactionAmt": 50.0 + (i % 200),
        "card1": card,
        "ProductCD": "W", "card2": None, "card3": None, "card4": None,
        "card5": None, "card6": None,
        "addr1": 204.0, "addr2": None, "dist1": None, "dist2": None,
        "P_emaildomain": None, "R_emaildomain": None,
        "DeviceType": None, "DeviceInfo": None,
    })


def run_batch(cards, count: int, offset: int = 0) -> list[float]:
    """Fire `count` requests down ONE reused connection; return latencies in ms."""
    conn = http.client.HTTPConnection(HOST, PORT, timeout=10)
    headers = {"content-type": "application/json"}
    latencies = []
    for i in range(count):
        card, last_dt = cards[(offset + i) % len(cards)]
        body = make_body(card, last_dt, offset + i)
        start = time.perf_counter()
        conn.request("POST", "/score", body=body, headers=headers)
        resp = conn.getresponse()
        resp.read()                      # must drain before the next request
        latencies.append((time.perf_counter() - start) * 1000)
        if resp.status != 200:
            raise SystemExit(f"server returned {resp.status}")
    conn.close()
    return latencies


def pct(sorted_ms: list[float], p: float) -> float:
    """The value below which p percent of calls fall."""
    return sorted_ms[min(int(len(sorted_ms) * p / 100), len(sorted_ms) - 1)]


def report(title: str, latencies: list[float], wall: float) -> None:
    s = sorted(latencies)
    print(f"\n{title}")
    print(f"  requests   {len(s):,} in {wall:.1f}s   ->  {len(s) / wall:,.0f} req/sec")
    print(f"  p50        {pct(s, 50):6.2f} ms")
    print(f"  p95        {pct(s, 95):6.2f} ms")
    print(f"  p99        {pct(s, 99):6.2f} ms")
    print(f"  max        {s[-1]:6.2f} ms")
    print(f"  mean       {statistics.fmean(s):6.2f} ms   <- the number that lies")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure scoring latency.")
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=10,
                        help="parallel connections for the throughput run")
    args = parser.parse_args()

    cards = pick_cards()
    print(f"{len(cards)} cards with history, {args.requests:,} requests each run")

    run_batch(cards, WARMUP)             # thrown away on purpose

    # 1. one connection, one at a time: latency with nothing in the way
    start = time.perf_counter()
    seq = run_batch(cards, args.requests)
    report("sequential (1 connection)", seq, time.perf_counter() - start)

    # 2. several at once: what the box can actually push
    per_worker = args.requests // args.workers
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_batch, cards, per_worker, w * per_worker)
                   for w in range(args.workers)]
        con = [ms for f in futures for ms in f.result()]
    report(f"concurrent ({args.workers} connections)", con,
           time.perf_counter() - start)


if __name__ == "__main__":
    main()
