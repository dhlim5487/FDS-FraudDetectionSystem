"""
Stage 5: null importance

Trains the SAME 15 features (10 existing + 5 id_ candidates) twice:
  1. once on the real isFraud labels          -> actual importance
  2. nb_runs times on SHUFFLED isFraud labels  -> the "by chance" baseline

A feature whose actual gain barely beats the shuffled-label baseline is
noise, not signal - regardless of how high it ranked in gain/SHAP/
permutation importance earlier.

No early stopping here: a shuffled-label run has nothing to validate
against (AUC stays ~0.5), so early stopping would cut it off after a
handful of rounds while the real-label run gets many more - making gain
values incomparable. A fixed tree count keeps both sides fair.

    uv run python -m analysis.null_importance --runs 80
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from core.features import FEATURE_NAMES, CATEGORICAL_FEATURES, apply_categorical_dtype
from core.schema import LABEL_FIELD
from offline.train import split_by_time

TRAINING_SET = "data/training_set.parquet"
OUT_CSV = "analysis/out/null_importance.csv"
NUM_BOOST_ROUND = 100


def get_feature_importances(train_df: pd.DataFrame, shuffle: bool, seed: int | None = None) -> pd.DataFrame:
    target = train_df[LABEL_FIELD].to_numpy()
    if shuffle:
        target = np.random.RandomState(seed).permutation(target)

    dataset = lgb.Dataset(train_df[FEATURE_NAMES], target,
                          categorical_feature=CATEGORICAL_FEATURES)
    # No early stopping, so real and shuffled runs get the same tree count
    # and their gain values stay comparable.
    model = lgb.train(
        {"objective": "binary", "learning_rate": 0.05, "verbosity": -1},
        dataset, num_boost_round=NUM_BOOST_ROUND,
    )

    return pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance_gain": model.feature_importance("gain"),
    })


def load_train_split() -> pd.DataFrame:
    df = pd.read_parquet(TRAINING_SET)
    df = apply_categorical_dtype(df)
    train, _ = split_by_time(df, train_frac=0.8, warmup_days=0)
    return train


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: null importance.")
    parser.add_argument("--runs", type=int, default=80,
                        help="number of shuffled-label runs to build the null distribution")
    parser.add_argument("--out", default=OUT_CSV, help="where to write the ranked CSV")
    args = parser.parse_args()

    train = load_train_split()
    print(f"train {len(train):,} rows   {len(FEATURE_NAMES)} features   "
          f"{NUM_BOOST_ROUND} trees per run (no early stopping)")

    actual_imp_df = get_feature_importances(train, shuffle=False)

    null_imp_df = pd.DataFrame()
    start = time.time()
    for i in range(args.runs):
        imp_df = get_feature_importances(train, shuffle=True, seed=i)
        imp_df["run"] = i + 1
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        if (i + 1) % 10 == 0 or (i + 1) == args.runs:
            spent = (time.time() - start) / 60
            print(f"  {i + 1:>3}/{args.runs}  ({spent:5.1f} min)")

    # score: how far actual gain sits above the 75th percentile of what
    # chance alone produces - low/negative means "indistinguishable from noise"
    scores = []
    for feat in FEATURE_NAMES:
        actual = actual_imp_df.loc[actual_imp_df["feature"] == feat, "importance_gain"].values[0]
        null_vals = null_imp_df.loc[null_imp_df["feature"] == feat, "importance_gain"].values
        null_p75 = np.percentile(null_vals, 75)
        score = np.log(1e-10 + actual / (1 + null_p75))
        scores.append({
            "feature": feat,
            "actual_gain": actual,
            "null_p75": null_p75,
            "null_max": null_vals.max(),
            "score": score,
        })

    result = pd.DataFrame(scores).sort_values("score", ascending=False).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(f"\n{result.to_string(index=False)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
