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

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from core.features import FEATURE_NAMES
from core.schema import LABEL_FIELD

TRAINING_SET = "data/training_set.parquet"
MODEL_OUT = "data/model.txt"
DAY = 86400


'''
    The first transactions of the dataset have empty history through no
    fault of their own - every card looks brand new. Dropping that period
     stops the model reading "0 prior transactions" as "safe".
'''

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
    args = parser.parse_args()

    df = pd.read_parquet(TRAINING_SET)
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
         "verbosity": -1, "seed": 42},
        lgb.Dataset(train[FEATURE_NAMES], train[LABEL_FIELD]),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(test[FEATURE_NAMES], test[LABEL_FIELD])],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )

    scores = model.predict(test[FEATURE_NAMES])
    auc = roc_auc_score(test[LABEL_FIELD], scores)
    pr_auc = average_precision_score(test[LABEL_FIELD], scores)
    baseline = test[LABEL_FIELD].mean()   # PR-AUC of random guessing

    print(f"\ntrees used   {model.best_iteration}")
    print(f"AUC          {auc:.4f}   (0.5 = coin flip)")
    print(f"PR-AUC       {pr_auc:.4f}   (random guessing = {baseline:.4f}, "
          f"so {pr_auc / baseline:.1f}x better)")

    print("\nwhich clue mattered:")
    gains = sorted(zip(FEATURE_NAMES, model.feature_importance("gain")),
                   key=lambda x: -x[1])
    total = sum(g for _, g in gains)
    for name, gain in gains:
        print(f"  {name:<22} {100 * gain / total:5.1f}%")

    model.save_model(MODEL_OUT)
    print(f"\nsaved {MODEL_OUT}")


if __name__ == "__main__":
    main()
