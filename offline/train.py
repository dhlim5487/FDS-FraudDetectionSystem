"""
    Teaches model

    Reads the table built by offline/build_training_set.py and fits LightGBM on it.

    THE ONE RULE HERE: split by TIME, never at random. 

        uv run python -m offline.train
        uv run python -m offline.train --train-frac 0.8 --warmup-days 0

    Accuracy is not reported on purpose: 96.5% of rows are legitimate, so a model
    that always says "fine" scores 96.5% while catching nothing. AUC and PR-AUC
    measure what we actually care about, ranking fraud above non-fraud.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                            roc_auc_score, roc_curve)

from core.features import FEATURE_NAMES, CATEGORICAL_FEATURES
from core.schema import LABEL_FIELD

TRAINING_SET = "data/training_set.parquet"
MODEL_OUT = "data/model.txt"
DAY = 86400


'''
    The first transactions of the dataset have empty history through no
    fault of their own - every card looks brand new. Dropping that period
     stops the model reading "0 prior transactions" as "safe".
'''

# Operating points to report. NOT chosen from any business requirement - we do
# not have one. They are here so we can see how much each one wobbles between
# identical training runs, and then only gate on the ones that hold still.
FPR_POINTS = (0.001, 0.01)          # recall when we allow this false-positive rate
RECALL_POINTS = (0.5, 0.8)          # precision when we insist on catching this much
TOPK_POINTS = (0.005, 0.01, 0.02)   # precision when we review this share of traffic

# Measured across 5 seeds on identical data (see README): the wobble you get
# with no real change at all. Any gate threshold has to clear it.
#   auc 0.30%   pr_auc 1.66%   recall@fpr0.01 3.66%   recall@fpr0.001 10.9%


def evaluate(y_true, scores) -> dict[str, float]:
    """
    Every number the promotion gate might care about, from one set of scores.

    AUC and PR-AUC average over every possible cut-off. The rest pin a single
    operating point, because in production only one cut-off is ever used - and
    a model can improve on average while getting worse exactly where we stand.
    """
    out = {
        "auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }

    # roc_curve gives fpr ascending, so np.interp can read tpr straight off it
    fpr, tpr, _ = roc_curve(y_true, scores)
    for point in FPR_POINTS:
        out[f"recall@fpr{point:g}"] = float(np.interp(point, fpr, tpr))

    # precision_recall_curve gives recall descending; reverse it for np.interp
    precision, recall, _ = precision_recall_curve(y_true, scores)
    for point in RECALL_POINTS:
        out[f"prec@recall{point:g}"] = float(
            np.interp(point, recall[::-1], precision[::-1]))

    # "if a reviewer works through the riskiest k% of traffic, how much of what
    # they open is really fraud" - the one line here a non-specialist can read.
    ranked = np.asarray(y_true)[np.argsort(-scores)]
    for point in TOPK_POINTS:
        cut = max(1, int(len(ranked) * point))
        out[f"prec@top{point * 100:g}pct"] = float(ranked[:cut].mean())

    return out


def split_by_time(df: pd.DataFrame, train_frac: float, warmup_days: int):
    if warmup_days:
        start = df["TransactionDT"].min()
        df = df[df["TransactionDT"] >= start + warmup_days * DAY]

    cutoff = df["TransactionDT"].quantile(train_frac)
    return df[df["TransactionDT"] <= cutoff], df[df["TransactionDT"] > cutoff]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud model.")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="fraction of the timeline used for training")
    parser.add_argument("--warmup-days", type=int, default=0,
                        help="drop this many days off the front (the cold-start rows)")
    parser.add_argument("--seed", type=int, default=42,
                        help="LightGBM seed; vary it to measure run-to-run wobble")
    parser.add_argument("--out", default=MODEL_OUT,
                        help="where to write the model; use a candidate path to "
                             "train without touching what is being served")
    args = parser.parse_args()

    df = pd.read_parquet(TRAINING_SET)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("__missing__").astype("category")
    train, test = split_by_time(df, args.train_frac, args.warmup_days)

    span = lambda d: (d.TransactionDT.max() - d.TransactionDT.min()) / DAY
    print(f"train {len(train):,} rows  {span(train):.0f} days  "
          f"fraud {100 * train[LABEL_FIELD].mean():.2f}%")
    print(f"test  {len(test):,} rows  {span(test):.0f} days  "
          f"fraud {100 * test[LABEL_FIELD].mean():.2f}%")
    print(f"no overlap: train ends {train.TransactionDT.max():,}, "
          f"test starts {test.TransactionDT.min():,}\n")

    # Defaults plus early stopping. No tuning yet - get an honest baseline first,
    # then find out whether tuning is even worth the trouble.
    model = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "verbosity": -1, "seed": args.seed},
        lgb.Dataset(train[FEATURE_NAMES], train[LABEL_FIELD], categorical_feature=CATEGORICAL_FEATURES),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(test[FEATURE_NAMES], test[LABEL_FIELD], categorical_feature=CATEGORICAL_FEATURES)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )

    scores = model.predict(test[FEATURE_NAMES])
    metrics = evaluate(test[LABEL_FIELD], scores)
    baseline = test[LABEL_FIELD].mean()   # PR-AUC of random guessing

    print(f"\nseed {args.seed}   trees used {model.best_iteration}")
    for name, value in metrics.items():
        print(f"  {name:<22} {value:.4f}")
    print(f"  (random guessing would score {baseline:.4f} on pr_auc)")

    # one grep-able line, so several runs can be lined up into a table
    flat = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"\nMETRICS seed={args.seed} trees={model.best_iteration} {flat}")

    print("\nwhich clue mattered:")
    gains = sorted(zip(FEATURE_NAMES, model.feature_importance("gain")),
                   key=lambda x: -x[1])
    total = sum(g for _, g in gains)
    for name, gain in gains:
        print(f"  {name:<22} {100 * gain / total:5.1f}%")

    out = Path(args.out)
    model.save_model(str(out))

    # Keep the test-set scores. Working out a NEW metric later then costs a
    # calculation instead of a retrain - which is what made adding
    # precision@top-k a 17-minute job the first time round.
    preds_file = out.with_suffix(".preds.parquet")
    pd.DataFrame({
        "TransactionID": test["TransactionID"].to_numpy(),
        "TransactionDT": test["TransactionDT"].to_numpy(),
        "score": scores,
        LABEL_FIELD: test[LABEL_FIELD].to_numpy(),
    }).to_parquet(preds_file, index=False)

    # What the promotion gate needs but cannot recompute from the model file:
    # the shape of the data this model was actually trained on.
    meta_file = out.with_suffix(".meta.json")
    meta_file.write_text(json.dumps({
        "seed": args.seed,
        "trees": model.best_iteration,
        "train_rows": len(train),
        "train_fraud_rate": float(train[LABEL_FIELD].mean()),
        # When the training window ends. The gate needs this to check that a
        # candidate was not fitted on rows it is about to be examined on.
        "train_min_dt": int(train["TransactionDT"].min()),
        "train_max_dt": int(train["TransactionDT"].max()),
        "test_rows": len(test),
        "metrics": metrics,
    }, indent=2))

    print(f"\nsaved {out}")
    print(f"      {preds_file}")
    print(f"      {meta_file}")


if __name__ == "__main__":
    main()
