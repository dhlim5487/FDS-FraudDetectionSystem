# FDS — Real-Time Fraud Detection System

[한국어](README.ko.md) | English

A personal project that replays the IEEE-CIS fraud dataset as a live payment stream, computes per-card features on the fly, and scores each transaction with a LightGBM model behind a FastAPI endpoint. The constraint I designed around is that offline training and online serving have to compute features the exact same way (train/serve parity). The loop runs end to end: scheduled retraining, validation, a promotion gate, and a running service that picks up the new model on its own.

## Architecture

```
train_transaction.csv ─┐
train_identity.csv     ├─▶ ingest/producer.py ─▶ Kafka ─▶ stream/consumer.py ─▶ Redis (per-card history)
                                                                                       │
                                                                                       ▼
                                                                        serving/app.py (FastAPI, /score)
                                                                                       ▲
                                                                   reloads when model.txt changes
                                                                        (no restart)   │
offline/build_training_set.py ─▶ training_set.parquet ─▶ offline/train.py ─▶ model_candidate.txt
                                                                                       │
                                                            offline/promote.py (7 gates) ┘
                                                              passes → atomic swap of model.txt

Airflow DAG (orchestration/dags/retrain.py) — Mondays 05:00, build → retrain → promote
   metadata DB: Postgres (fds-postgres)
```

Design decisions (see the docstrings in `core/schema.py`, `core/store.py`, `serving/app.py`):

- **The event only carries what a real payment terminal would actually send.** Of the 394 columns in the raw IEEE-CIS CSV, the pre-engineered Vesta columns (V1-V339, C1-C14, D1-D15, M1-M9) are dropped on purpose — `core/features.py` computes its own equivalents instead. What survives is the 23 fields a terminal can genuinely observe (16 transaction + 7 device/identity).
- **"How features are computed" (`core/features.py`) is split from "where history lives" (`core/store.py`).** That's what lets the offline build use `InMemoryStore` (~61,000 events/sec) and the live stream use `RedisStore` (~2,500 events/sec, survives a consumer restart) behind one identical interface.
- **`/score` is read-only.** Only `stream/consumer.py` writes history. If the scoring endpoint also wrote, every scored transaction would get recorded twice, and the next score would be computed against a past that never actually happened.
- **Features are recomputed at scoring time, never cached.** Signals like `amt_zscore_7d` and `is_new_device` depend on the transaction currently being scored — its amount, its device. Serving a pre-computed row would answer for the card's *last* transaction, not this one.
- **A promoted model reaches the service without a restart.** `promote.py` writes to a temp name and renames it over the target (atomic), and `serving/app.py` checks the file's mtime per request, reloading only when it moved. It has to be a rename rather than an overwrite so a reader can never observe a half-written model.

## Build phases

| Phase | What got built |
|---|---|
| 1 | Replay IEEE-CIS into Kafka in event-time order |
| 2 | Redis-backed feature state + train/serve parity tests |
| 3 | LightGBM model + FastAPI scoring service (10 features, AUC 0.7235 / PR-AUC 0.1140) |
| 4 | Latency benchmark — p99 5.2ms at ~360 req/sec |
| 5 | Airflow retraining DAG with a measured 7-check promotion gate |
| 6 | Feature-importance validation, then a schema expansion (15 features, **AUC 0.7513 / PR-AUC 0.1837**) |

### Phase 1 — Replaying the stream

`ingest/producer.py` sorts `train_transaction.csv` + `train_identity.csv` by event time and pushes them into Kafka (`--speed` controls the replay rate). The event schema was fixed here: of the 394 raw columns, the pre-engineered Vesta families (V1-V339, C1-C14, D1-D15, M1-M9) are dropped, leaving the 23 fields a terminal can genuinely observe (16 transaction + 7 device/identity).

### Phase 2 — Feature state and parity

Feature computation (`core/features.py`) is split from history storage (`core/store.py`), so the offline build runs on `InMemoryStore` (~61,000 events/sec) and the live stream on `RedisStore` (~2,500 events/sec) behind one interface. Only `stream/consumer.py` writes history. Whether the two paths really agree is pinned by `tests/test_parity.py`: 3,000 real transactions plus the time-window edge cases.

### Phase 3 — Training and serving

LightGBM (seed 42, 64 trees) trained on a time-based split: 472,432 rows over 140 days for training, 118,108 rows over 42 days for test, zero overlap. Ten features gave AUC 0.7235 / PR-AUC 0.1140. `/score` in `serving/app.py` is read-only and recomputes features per request rather than serving a cached row.

### Phase 4 — Measuring latency

`bench/latency` put a single worker at p99 5.2ms, ~360 req/sec. Going to 4 workers pushed p50 up to 44ms instead of down; the cause turned out to be a socket option rather than resource contention (issue 2 below).

### Phase 5 — Automated retraining

The Airflow DAG (`orchestration/dags/retrain.py`) runs `build → retrain → promote` every Monday at 05:00. Seven checks in `offline/promote.py` decide whether a candidate replaces production, and both the refusal and the approval branch were exercised. A manual trigger racing the schedule was fixed with `max_active_runs=1`.

### Phase 6 — Expanding the schema

Every column the schema had discarded up front (415 of them) was pulled back out of the raw CSV and measured for contribution, in three stages:

1. **Statistical screening** (`analysis/screen_columns.py`) — missing rate, near-constant check, univariate AUC / Cramér's V, ranking all 415 columns (`analysis/out/screen_results.csv`)
2. **Gain importance + SHAP + permutation importance** (`analysis/feature_contribution.py`) — does a candidate actually improve the model when stacked on top of the existing 10 features?
3. **Null importance** (`analysis/null_importance.py`) — shuffle the label, retrain 80 times to establish how important a feature looks by pure chance, then check whether the real importance clears that bar

Five columns cleared all three: `id_31`, `id_19`, `id_20`, `id_29`, `id_30`. Adding them moved AUC 0.7235 → 0.7513 and PR-AUC 0.1140 → 0.1837. The same run flagged three existing features sitting barely above the noise floor — `amt_sum_1h` (0.21), `txn_count_1h` (0.32), `amt_sum_24h` (0.41). They are still in the model, but their contribution is not established.

The retraining loop was closed after that: mtime-based reloading in `serving/app.py` and atomic file replacement in `offline/promote.py`, so a promoted model reaches the service without a restart, verified round-trip against a live server (see "How it was verified"). Airflow's metadata DB also moved from SQLite to the Postgres container that had been declared in `docker-compose.yml` and used by nothing — no code changes, three lines in `airflow.cfg`.

## Things that broke, and what fixed them

Issues actually hit during development, with the cause and the fix.

### 1. The per-card feature cache answered for the wrong transaction (Phase 3)
I tried caching computed features under `feat:<card1>` and reusing them at scoring time. The cache reflected the card's *previous* transaction, which doesn't match the transaction actually being scored — its amount and device are exactly what the features are supposed to react to. Fix: drop the cache, recompute from `core/features.py` on every scoring request.

### 2. `--workers 4` made latency dramatically worse, not better (Phase 4)
More concurrency was supposed to help; instead p50 jumped from 2.5ms to 44ms. Breaking one request into phases showed server-side work was unchanged — the missing 43ms sat between the response header and the body.

| Phase of the request | 1 worker | 4 workers |
|---|---|---|
| TTFB (to first byte) | 3.87 ms | 4.06 ms |
| body (rest of response) | 0.06 ms | **43.38 ms** |

Root cause: `TCP_NODELAY` wasn't set on the multiprocess socket path, so Nagle's algorithm (hold small packets, batch them) on one side and the client's delayed ACK (hold ACKs, batch them) on the other each waited for the other until the 40ms timer fired.

uvicorn multi-worker is standard practice on a real Linux server, so I don't generalize this. It's recorded as a fixed delay observed *in this environment (WSL2)*. (`OMP_NUM_THREADS=1` was ruled out — its effect was inside the machine's own ~30% run-to-run noise.)

### 3. The promotion gate was validating itself, not the candidate model (Phase 5)
To exercise the gate I deliberately trained a weakened model (`--warmup-days 150`) — and it came back at 0.7443, well *above* the healthy model's 0.7235. The cause: `promote.py` scored candidates on a standard split it built itself, rather than on the split the candidate had actually been trained on. The weak model's training window (day 150–182) sat entirely inside that "held-out" test window (day 140–182). The 0.7443 was the candidate being re-scored on data it had already seen.

Adding a "no train/test overlap" gate caught it: 99,622 of the 118,108 test rows had already been trained on (84%). It is binary rather than statistical, so there is no threshold to pick and no way for it to reject a healthy model.

Both branches were then verified end-to-end under Airflow: all 3 tasks green with the gate correctly refusing at a measured +0.0000 improvement, and — with the margin temporarily lowered — an actual `PROMOTED` file swap.

### 4. A manual Airflow trigger raced the schedule despite `catchup=False` (Phase 5)
`catchup=False` only stops Airflow from backfilling the *entire* history — it still fires the most recently missed slot the moment the DAG is unpaused. That's how a manual trigger ended up racing a scheduled run, both writing to the same candidate file and roughly doubling training time. Fix: `max_active_runs=1`.

### 5. The first analysis improved the model with data that can't exist in real time (Phase 6)
The first `feature_contribution.py` run threw in every V/C/D/M column as a candidate and pushed AUC from 0.7235 to **0.8806**. But those columns are Vesta's own after-the-fact engineering — a payment terminal cannot send them live. The design constraint written at the top of this README was being broken in the analysis step itself.

Fix: split the script into `--mode full` (reference upper bound) and `--mode realistic` (only what a terminal could send), and base the actual schema expansion solely on the realistic run (AUC 0.7578). The 0.8806 is kept only as "the ceiling this dataset allows".

### 6. The analysis script crashed twice (Phase 6)
- **First crash**: died while saving SHAP dependence plots (just before Stage 3 finished)
- **Second crash**: died immediately on entering permutation importance — sklearn's `permutation_importance` requires a `.fit` method, which a raw LightGBM `Booster` doesn't have. Fixed with a small wrapper class.

Because neither run had ever reached the final stage, I couldn't tell "hung" from "genuinely slow" when the third run passed 17 minutes. It was normal: `n_repeats=10 × 128 candidates = 1,280` full re-predictions over a 118k-row test set.

### 7. Adding the 5 `id_` features broke three places at once (Phase 6)
A schema change touches `core/schema.py` (the contract) → `core/features.py` (computation) → `ingest/producer.py` (input), and every file I forgot surfaced somewhere else:

- a `NameError` in `core/features.py`
- `ingest/producer.py` wasn't reading `train_identity.csv` at all — every `id_` field arrived as None
- mixed float/str dtypes out of `core/features.py` **crashed pyarrow in the final `to_parquet`, after all 590,540 rows had been built** (57 seconds wasted). Fixed by normalizing missing values to the string `"__missing__"`.

`tests/test_parity.py` broke twice on the same change: `ImportError: cannot import name 'id_31' from 'core.schema'`, then — after that fix — `NameError: name 'id_31' is not defined`, because the `evt()` helper's signature was never given the parameter.

### 8. `data/model.txt` was silently overwritten (Phase 6)
`offline/train.py` defaults its output to `data/model.txt`, so a single verification retrain replaced the production 10-feature model and its `model.meta.json`. The baseline I was comparing against no longer existed on disk. Candidate training now always uses `--out data/model_candidate.txt`.

### 9. `promote.py` was missing categorical handling, and the logic was duplicated anyway
`offline/promote.py` had no categorical handling for the 5 new `id_` fields and errored out; `offline/train.py` and `analysis/null_importance.py` each carried their own copy of similar logic. Consolidated all three into a shared `apply_categorical_dtype()` in `core/features.py`.

### 10. That consolidation missed one caller, and `/score` was dead
While consolidating in issue 9 I updated `train.py`, `promote.py` and `null_importance.py` — and **missed `serving/app.py`**. The model had been trained with the 5 `id_` fields as categoricals, but the scoring path handed them over as plain values, so the endpoint died on every request:

```
ValueError: train and valid dataset categorical_feature do not match.
```

Nothing caught it. All three parity tests kept passing — they only exercise feature *computation* — and the server hadn't been started once since the schema expansion. It surfaced on the very first request when uvicorn came up to verify the model-reload work above.

Now I grep for callers before consolidating anything into a shared helper, and on any day the schema changes I start uvicorn and put one request through `/score` rather than trusting pytest alone.

### Setup snags

Not deep problems, but they cost real time:

- **`ModuleNotFoundError: No module named 'core'`** — from running `python ingest/producer.py`. Package-relative imports need the module form: `python -m ingest.producer`.
- **WSL files owned by root** — files created via containers/sudo ended up root-owned, so VSCode-WSL failed to save with `EACCES: permission denied`. Fixed with `chown` plus running the container as a non-root user.
- **Docker Desktop WSL integration off** — `The command 'docker' could not be found in this WSL 2 distro` on the WSL side, and a failed connection to `npipe:////./pipe/dockerDesktopLinuxEngine` on the PowerShell side. Has to be enabled in Docker Desktop settings.
- **Container names and script paths** — `docker exec -it kafka ...` gives `No such container: kafka`; the real name is `fds-kafka`, and the CLI needs its full path with extension (`/opt/kafka/bin/kafka-console-consumer.sh`).
- **Airflow standalone failed on first launch** — `Exception in thread scheduler / dag-processor / triggerer / api-server`, all of them dying. Fixed by setting `AIRFLOW__CORE__DAGS_FOLDER` before restarting. Shutting it down needed `pkill -9` too; plain `pkill -f` and SIGTERM left port 8080 held.
- **In Airflow 3, `dags list` reads the DB, not the folder** — right after the move to Postgres the DAG list came back empty. Coming from Airflow 2 you assume a file in the folder is enough; it isn't. `airflow dags reserialize` parses the folder and writes the DAGs into the DB.
- **Turning off `load_examples` broke the CLI** — 113 example DAGs were already registered, and `dags list` failed outright with `DeserializationError: ... 'example_custom_weight'`, because disabling examples also removes the classes those DAGs reference. Marking them stale wasn't enough; the rows whose `fileloc` pointed inside Airflow's own `example_dags/` had to be deleted.

## Known limitations

- **The stream path and the training path aren't connected by data.** `offline/build_training_set.py` re-reads the original CSV rather than the events that went through Kafka. The parity tests prove both paths run the same feature code, but the cold path — persisting events to Parquet/Postgres and training from those — was never built.
- **Two data-hygiene thresholds in `offline/promote.py`** (training-row floor, fraud-rate band) are still marked in code as judgement calls with no measurements behind them. They need several real weekly runs before they can be derived from data.
- **The three lowest null-importance features** (`amt_sum_1h`, `txn_count_1h`, `amt_sum_24h`) barely clear the noise floor; whether dropping them costs anything is untested.
- **Redis runs with persistence off** (`--save "" --appendonly no`), so taking the container down loses per-card history. Fine for a demo environment where a replay refills it.

## Running it

### 1. Bring up infrastructure (Kafka / Redis / Postgres)
```bash
docker compose up -d
```

### 2. Build the training set and train a model
```bash
uv run python -m offline.build_training_set     # writes data/training_set.parquet
uv run python -m offline.train                  # writes data/model.txt, prints AUC/PR-AUC
```

### 3. Replay the stream and start the consumer
```bash
uv run python -m ingest.producer --speed 1000     # replays transactions at event-time speed into Kafka
uv run python -m stream.consumer                  # consumes from Kafka, updates Redis history + feat:<card1>
```

### 4. Scoring service
```bash
uv run uvicorn serving.app:app --port 8000
curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{...}'
```

### 5. Latency benchmark
```bash
uv run python -m bench.latency --requests 10000
```

### 6. Retrain and run the promotion gate (manual)
```bash
uv run python -m offline.build_training_set
uv run python -m offline.train --out data/model_candidate.txt
uv run python -m offline.promote --candidate data/model_candidate.txt
```
If the gate passes, `data/model.txt` is swapped atomically and a running scoring service picks up the new model on its next request.

### 7. The Airflow retraining DAG
```bash
airflow standalone                       # settings live in airflow.cfg, no exports needed
airflow dags unpause fds_retrain         # only when demoing
```
Runs `build → retrain → promote` every Monday at 05:00. Metadata is stored in the `fds-postgres` container.

### 8. Model validation (Phase 6)
```bash
uv run python -m analysis.screen_columns
uv run python -m analysis.feature_contribution --mode realistic   # --mode full is the reference ceiling
uv run python -m analysis.null_importance --runs 80
```

### 9. Tests
```bash
uv run pytest tests/ -v      # 3 parity + 3 model-reload (Redis required)
```

## Project layout

```
core/         event schema, feature computation, history stores (in-memory / Redis)
ingest/       IEEE-CIS CSV → Kafka replay
stream/       Kafka consumer (writes history, publishes latest features)
offline/      training set build, training, promotion gate
serving/      FastAPI scoring service (auto-reloads the model)
orchestration/ Airflow retraining DAG
analysis/     feature-importance validation (SHAP, permutation, null importance, etc.)
bench/        latency benchmark
tests/        train/serve parity + model reload tests
```

## How it was verified

- **train/serve parity** — 3,000 real transactions computed through both the offline (in-memory) and online (Redis) paths and compared to the last decimal. Boundary conditions are pinned as separate edge cases: exact 1h/24h/7d window edges, transactions sharing a timestamp, and isolation between cards.
- **No future leakage** — a test asserting no feature ever sees a transaction later than itself.
- **Model reload** — with uvicorn running and the request held identical, `promote.py`'s `install()` swapped the model underneath it. The score moved `0.0198 → 0.0197` with no restart, and returned to `0.0198` when the original was restored. The unit tests also pin the other half — that an *unchanged* file is not reloaded — since without it, re-reading 400KB on every request would pass too.
- **Promotion gate** — both branches exercised: refusal (at +0.0000 improvement) and approval (`PROMOTED`, with the margin temporarily lowered).

## Performance

Current production model (`data/model.meta.json` — 15 features, seed 42, 64 trees):

| Metric | Value | What it means |
|---|---|---|
| AUC | **0.7513** | 0.5 = meaningless |
| PR-AUC | **0.1837** | 5.3x over random guessing (0.0344) |
| recall@fpr0.001 | 0.0381 | detection rate when only 1 in 1,000 legit txns may be flagged |
| recall@fpr0.01 | 0.1442 | same, at 1 in 100 |
| prec@top0.5pct | 0.4644 | precision when blocking only the riskiest 0.5% |
| prec@top1pct | 0.3810 | same, at the top 1% |

- Split is time-based: train 472,432 rows over 140 days, test 118,108 rows over 42 days, zero overlap
- The Phase 3 10-feature model scored AUC 0.7235 / PR-AUC 0.1140; the Phase 6 schema expansion added +0.028 / +0.070
- Including every pre-engineered Vesta column reaches AUC 0.8806, but it isn't reproducible in real time, so it wasn't adopted (see issue 5 above)

Also:

- Latency: p99 5.2ms at ~360 req/sec (single worker)
- Feature store: InMemoryStore ~61,000 events/sec (offline) / RedisStore ~2,500 events/sec (live, survives restarts)

## Requirements

- Python >= 3.11 (managed with uv)
- Docker / Docker Compose (Kafka, Redis, Postgres)
- Core deps: `confluent-kafka`, `fastapi`, `lightgbm`, `pandas`, `pyarrow`, `redis`, `scikit-learn`, `uvicorn` (dev: `matplotlib`, `pytest`, `shap`)
- Airflow is installed separately from the project via `uv tool`, with the metadata-DB drivers alongside it:
  ```bash
  uv tool install apache-airflow --with psycopg2-binary --with asyncpg
  ```
  (`asyncpg` is needed because Airflow 3's API server uses async SQLAlchemy)
