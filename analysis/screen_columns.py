"""
Stage 1 statistical screening of the raw columns core/schema.py deliberately
excludes: C1-C14, D1-D15, M1-M9, V1-V339, id_01-id_38.

schema's 18-column live-event restriction on purpose - that
restriction is a live-pipeline decision, not a statistical one.

For each column: missing rate, near-constant flag, and univariate signal
vs isFraud - point-biserial correlation + single-feature AUC for numeric
columns, Cramer's V + per-category fraud-rate spread for categorical ones.

    uv run python -m analysis.screen_columns --limit 50000   # smoke test
    uv run python -m analysis.screen_columns                 # full run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.metrics import roc_auc_score

TRANSACTION_CSV = "data/train_transaction.csv"
IDENTITY_CSV = "data/train_identity.csv"
OUT_CSV = "analysis/out/screen_results.csv"

# id_01-id_11 are numeric, id_12-id_38 are categorical (per the competition's
# own data description) - M1-M9 are categorical (mostly T/F).
NUMERIC_COLS = (
    [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"V{i}" for i in range(1, 340)]
    + [f"id_{i:02d}" for i in range(1, 12)]
)
CATEGORICAL_COLS = (
    [f"M{i}" for i in range(1, 10)]
    + [f"id_{i:02d}" for i in range(12, 39)]
)


def load_raw(limit: int | None) -> pd.DataFrame:
    """Read only the excluded columns (+ the join key and the label)."""
    txn_cols = ["TransactionID", "isFraud"] + [
        c for c in NUMERIC_COLS + CATEGORICAL_COLS if not c.startswith("id_")
    ]
    idn_cols = ["TransactionID"] + [
        c for c in NUMERIC_COLS + CATEGORICAL_COLS if c.startswith("id_")
    ]

    txn = pd.read_csv(TRANSACTION_CSV, usecols=txn_cols, nrows=limit)
    idn = pd.read_csv(IDENTITY_CSV, usecols=idn_cols)
    return txn.merge(idn, on="TransactionID", how="left")


def score_numeric(col: pd.Series, y: pd.Series) -> dict[str, float]:
    valid = col.notna()
    if valid.sum() < 2 or col[valid].nunique() < 2:
        return {"corr": 0.0, "auc": 0.5}

    corr, _ = pointbiserialr(y[valid], col[valid])
    # Median-fill so every row still ranks - missingness itself can be a
    # signal, and is captured separately by missing_rate anyway.
    filled = col.fillna(col.median())
    auc = roc_auc_score(y, filled)
    auc = max(auc, 1 - auc)  # separation strength, direction doesn't matter here
    return {"corr": float(corr) if not np.isnan(corr) else 0.0, "auc": float(auc)}


def score_categorical(col: pd.Series, y: pd.Series) -> dict[str, float]:
    filled = col.fillna("__missing__").astype(str)
    if filled.nunique() < 2:
        return {"cramers_v": 0.0, "fraud_rate_spread": 0.0}

    table = pd.crosstab(filled, y)
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0

    rates = y.groupby(filled).mean()
    spread = float(rates.max() - rates.min())
    return {"cramers_v": cramers_v, "fraud_rate_spread": spread}


def screen(df: pd.DataFrame) -> pd.DataFrame:
    y = df["isFraud"]
    rows = []
    for col_name in NUMERIC_COLS + CATEGORICAL_COLS:
        col = df[col_name]
        missing_rate = float(col.isna().mean())
        non_null = col.dropna()
        near_constant = bool(
            len(non_null) > 0 and non_null.value_counts(normalize=True).iloc[0] > 0.99
        )

        row = {
            "column": col_name,
            "kind": "numeric" if col_name in NUMERIC_COLS else "categorical",
            "missing_rate": missing_rate,
            "flag_high_missing": missing_rate > 0.90,
            "flag_near_constant": near_constant,
        }
        if col_name in NUMERIC_COLS:
            row.update(score_numeric(col, y))
            row["rank_metric"] = row["auc"]
        else:
            row.update(score_categorical(col, y))
            row["rank_metric"] = row["cramers_v"]
        rows.append(row)

    return (pd.DataFrame(rows)
            .sort_values("rank_metric", ascending=False)
            .reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: screen excluded raw columns.")
    parser.add_argument("--limit", type=int, default=None,
                        help="only read the first N rows (for a quick smoke test)")
    parser.add_argument("--out", default=OUT_CSV, help="where to write the ranked CSV")
    args = parser.parse_args()

    df = load_raw(args.limit)
    total_cols = len(NUMERIC_COLS) + len(CATEGORICAL_COLS)
    print(f"Loaded {len(df):,} rows x {total_cols} candidate columns "
          f"({len(NUMERIC_COLS)} numeric, {len(CATEGORICAL_COLS)} categorical).")

    results = screen(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    n_high_missing = int(results["flag_high_missing"].sum())
    n_near_constant = int(results["flag_near_constant"].sum())
    print(f"flagged: {n_high_missing} high-missing (>90%), "
          f"{n_near_constant} near-constant (>99% one value)")

    # numeric ranks by AUC (0.5-1), categorical by Cramer's V (0-1) - not the
    # same scale, so look at each kind's own top rows, not just the merged sort.
    print("\ntop 15 numeric by AUC:")
    print(results[results["kind"] == "numeric"].head(15).to_string(index=False))
    print("\ntop 15 categorical by Cramer's V:")
    print(results[results["kind"] == "categorical"].head(15).to_string(index=False))

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
