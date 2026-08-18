"""
Model-Based Verification Stage(2, 3, 4): 
Verify if the raw columns that survived Stage 1 screening
(analysis/screen_columns.py) actually help on top of the existing 10 engineered features.

Trains ONE LightGBM model on FEATURE_NAMES + candidates, then reads three
importance signals off it: gain importance, SHAP, permutation importance.
Reuses offline.train's split_by_time and evaluate so the
comparison uses the exact same time-split rule and metrics as the served
model's training run.

Two modes:
  --mode full        all 118 candidates, C/D/V/M included. Vesta's V/C/D/M
                      columns are pre-computed from a card's full history -
                      no live payment terminal could send them. This mode
                      is a THEORETICAL CEILING, kept as a reference
                      benchmark, not a proposal to change the schema.
  --mode realistic    only the id_ columns (device/session signals a real
                      checkout page could send in real time). This is the
                      one that actually informs whether core/schema.py
                      should grow.

    uv run python -m analysis.feature_contribution --mode full
    uv run python -m analysis.feature_contribution --mode realistic
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from core.features import FEATURE_NAMES
from core.schema import LABEL_FIELD
from offline.train import evaluate, split_by_time

TRAINING_SET = "data/training_set.parquet"
TRANSACTION_CSV = "data/train_transaction.csv"
IDENTITY_CSV = "data/train_identity.csv"
BASELINE_META = "data/model.meta.json"
OUT_DIR = "analysis/out"

# Picked by hand from analysis/out/screen_results.csv 
# (top numeric by AUC, top categorical by Cramer's V)
CANDIDATE_NUMERIC = [
    "C4", "C2", "C12", "C7", "D3", "C1", "D5", "C8", "C11", "V283", "C5",
    "V294", "V317", "C10", "V280", "V308", "V30", "V70", "V29", "V69",
    "V282", "D15", "V79", "V94", "V52", "V91", "V90", "V51", "C6", "V102",
    "V295", "V318", "V133", "D10", "V218", "V264", "V219", "V265", "D2",
    "V97", "V258", "V93", "V92", "V50", "V128", "V74", "V58", "V103",
    "V306", "V134", "V217", "V263", "V279", "D8", "V73", "V57", "V72",
    "V95", "V40", "V203", "V126", "V257", "V229", "V71", "V81", "V168",
    "V204", "V230", "V16", "V34", "V292", "V33", "V39", "V15", "V80",
    "V232", "V274", "V85", "V43", "V49", "V32", "V48", "V31", "C9",
    "V233", "V101", "V275", "V132", "V307", "V291", "V84", "V60", "V42",
]
CANDIDATE_CATEGORICAL = [
    "id_19", "id_20", "M4", "id_17", "id_13", "id_35", "id_31", "id_25",
    "id_21", "id_26", "id_15", "id_33", "id_22", "id_29", "id_14", "id_16",
    "id_36", "id_28", "M6", "id_38", "id_37", "id_32", "id_12", "id_30",
    "id_18",
]
# Realistic mode: only id_ columns are session/device signals a real
# checkout page could send. M4/M6 are Vesta match flags, same category as
# C/D/V - dropped here.
REALISTIC_CATEGORICAL = [c for c in CANDIDATE_CATEGORICAL if c.startswith("id_")]

MODES = {
    "full": (CANDIDATE_NUMERIC, CANDIDATE_CATEGORICAL),
    "realistic": ([], REALISTIC_CATEGORICAL),
}


def load_candidates(candidates: list[str]) -> pd.DataFrame:
    """Read only the chosen candidate columns from the raw CSVs."""
    txn_cols = ["TransactionID"] + [c for c in candidates if not c.startswith("id_")]
    idn_cols = ["TransactionID"] + [c for c in candidates if c.startswith("id_")]

    txn = pd.read_csv(TRANSACTION_CSV, usecols=txn_cols)
    idn = pd.read_csv(IDENTITY_CSV, usecols=idn_cols)
    return txn.merge(idn, on="TransactionID", how="left")


def build_dataset(candidates: list[str], categorical: list[str]) -> pd.DataFrame:
    """Existing training table (10 features) + the chosen raw candidates,
    joined on TransactionID. Left join off the existing table on purpose:
    it already dropped the cardless rows build_training_set.py can't use."""
    base = pd.read_parquet(TRAINING_SET)
    raw = load_candidates(candidates)
    df = base.merge(raw, on="TransactionID", how="left")

    # Missingness is a signal (see Stage 1) - keep it as its own category
    # rather than dropping it, same treatment as screen_columns.py.
    for col in categorical:
        df[col] = df[col].fillna("__missing__").astype("category")

    return df


def auc_scorer(model, X: pd.DataFrame, y: pd.Series) -> float:
    return roc_auc_score(y, model.predict(X))


class FittedBooster:
    """sklearn's permutation_importance requires estimator.fit to exist,
    even though it never needs to call it here - the booster is already
    trained. This just satisfies that check."""

    def __init__(self, booster: lgb.Booster) -> None:
        self.booster = booster

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.booster.predict(X)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2+3+4: score candidate columns.")
    parser.add_argument("--mode", choices=MODES, default="full",
                        help="full = all candidates incl. Vesta pre-engineered "
                             "V/C/D/M (theoretical ceiling, reference only); "
                             "realistic = id_ columns only (live-reproducible)")
    args = parser.parse_args()

    numeric, categorical = MODES[args.mode]
    candidates = numeric + categorical
    all_features = FEATURE_NAMES + candidates

    df = build_dataset(candidates, categorical)
    train, test = split_by_time(df, train_frac=0.8, warmup_days=0)
    print(f"[{args.mode}] train {len(train):,} rows   test {len(test):,} rows   "
          f"{len(all_features)} features ({len(FEATURE_NAMES)} existing + "
          f"{len(candidates)} candidates)")

    model = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "verbosity": -1, "seed": 42},
        lgb.Dataset(train[all_features], train[LABEL_FIELD],
                    categorical_feature=categorical),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(test[all_features], test[LABEL_FIELD],
                                 categorical_feature=categorical)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    scores = model.predict(test[all_features])
    metrics = evaluate(test[LABEL_FIELD], scores)
    print(f"\nwith candidates added:")
    for name, value in metrics.items():
        print(f"  {name:<22} {value:.4f}")

    try:
        import json
        baseline = json.loads(open(BASELINE_META).read())["metrics"]
        print(f"\nbaseline (existing 10 features only, from {BASELINE_META}):")
        for name, value in baseline.items():
            delta = metrics[name] - value
            print(f"  {name:<22} {value:.4f}   (delta {delta:+.4f})")
    except FileNotFoundError:
        print(f"\n(no {BASELINE_META} found - run offline.train first for a side-by-side)")

    # --- Stage 2: gain importance ------------------------------------
    gains = dict(zip(all_features, model.feature_importance("gain")))
    total_gain = sum(gains.values())

    # --- Stage 3: SHAP --------------------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test[all_features])
    if isinstance(shap_values, list):   # some shap/lgbm combos return [class0, class1]
        shap_values = shap_values[1]
    mean_abs_shap = dict(zip(all_features, np.abs(shap_values).mean(axis=0)))

    # shap.dependence_plot converts the whole features frame to one 2D
    # array; mixing numeric float columns with string category columns in
    # that array makes its internal np.unique() crash ("float vs str").
    # Codes-only copy sidesteps it - same SHAP values, just a plottable axis.
    plot_features = test[all_features].copy()
    for col in categorical:
        plot_features[col] = plot_features[col].cat.codes

    top_candidates_by_shap = sorted(
        candidates, key=lambda c: -mean_abs_shap[c])[:3]
    for col in top_candidates_by_shap:
        plt.figure()
        shap.dependence_plot(col, shap_values, plot_features,
                              interaction_index=None, show=False)
        plt.savefig(f"{OUT_DIR}/shap_dependence_{args.mode}_{col}.png", bbox_inches="tight")
        plt.close()

    # --- Stage 4: permutation importance ---------------------------------
    perm = permutation_importance(
        FittedBooster(model), test[all_features], test[LABEL_FIELD],
        scoring=auc_scorer, n_repeats=10, random_state=42)
    perm_importance = dict(zip(all_features, perm.importances_mean))

    # --- combined comparison table ---------------------------------------
    table = pd.DataFrame({
        "feature": all_features,
        "is_existing": [f in FEATURE_NAMES for f in all_features],
        "gain_pct": [100 * gains[f] / total_gain for f in all_features],
        "mean_abs_shap": [mean_abs_shap[f] for f in all_features],
        "perm_delta_auc": [perm_importance[f] for f in all_features],
    }).sort_values("perm_delta_auc", ascending=False).reset_index(drop=True)

    out_csv = f"{OUT_DIR}/feature_contribution_{args.mode}.csv"
    table.to_csv(out_csv, index=False)
    print(f"\ntop 20 by permutation importance (delta AUC when shuffled):")
    print(table.head(20).to_string(index=False))
    print(f"\nwrote {out_csv}")
    print(f"wrote {OUT_DIR}/shap_dependence_{args.mode}_{{{','.join(top_candidates_by_shap)}}}.png")


if __name__ == "__main__":
    main()
