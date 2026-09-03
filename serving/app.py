"""
serving/app.py - the counter. A terminal asks "is this transaction fraud?" and
gets a score back.

    uv run uvicorn serving.app:app --port 8000
    curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{...}'

Three things here are deliberate:

READ-ONLY. This endpoint never writes to history. stream/consumer.py owns that;
if the counter wrote too, every scored transaction would be recorded twice and
the next score would be computed from a past that never happened.

NO CACHED FEATURES. The clues depend on the transaction being scored - its
amount drives amt_zscore_7d, its device drives is_new_device - so they are
recomputed here from the same history, with the same core/features.py the
training build used. Serving a stored row would answer about the card's LAST
transaction instead of this one.

RELOADS ITSELF. offline/promote.py replaces data/model.txt when a candidate
clears the gate, and this process has to notice without being restarted -
otherwise "automated retraining" stops one step short of the thing being
automated. See current_model().
"""
from __future__ import annotations

import os

import lightgbm as lgb
import pandas as pd
import redis
from fastapi import FastAPI

from core.features import (FEATURE_NAMES, WEEK, apply_categorical_dtype,
                           compute_features)
from core.schema import TransactionEvent
from core.store import RedisStore

MODEL_FILE = "data/model.txt"

app = FastAPI(title="FDS scoring")

# Loaded once at import, not per request - the model is 400KB of trees and
# reading it takes far longer than scoring with it.
model = lgb.Booster(model_file=MODEL_FILE)
model_mtime = os.path.getmtime(MODEL_FILE)
store = RedisStore(redis.Redis(host="localhost", port=6379, db=0,
                               decode_responses=True))


def current_model() -> lgb.Booster:
    """Pick up a model promote.py installed, without restarting the server.

    One stat() per request against a 5ms score is free; the actual reload
    happens only on the weeks a candidate clears the gate. No lock: rebinding
    a global is atomic, so the worst a race costs is reading the file twice.
    promote.py renames the new file into place, so we never see a half-written
    one.
    """
    global model, model_mtime
    mtime = os.path.getmtime(MODEL_FILE)
    if mtime != model_mtime:
        model, model_mtime = lgb.Booster(model_file=MODEL_FILE), mtime
    return model


@app.post("/score")
def score(event: TransactionEvent):
    history, devices = store.load(event.card1, event.TransactionDT, WEEK)
    features = compute_features(history, devices, event)

    # A DataFrame with the training column order: the booster remembers the
    # names it was fitted on and will complain if they arrive differently. The
    # id_ columns must arrive as category dtype for the same reason - the
    # booster re-maps them through the categories it was fitted on.
    row = pd.DataFrame([[features[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
    row = apply_categorical_dtype(row)
    fraud_score = float(current_model().predict(row)[0])

    # features come back too - a score with no reason behind it
    # is not much use to whoever has to explain the decline to a customer.
    return {
        "TransactionID": event.TransactionID,
        "card1": event.card1,
        "fraud_score": round(fraud_score, 4),
        "prior_txns_7d": len(history),
        "features": features,
    }
