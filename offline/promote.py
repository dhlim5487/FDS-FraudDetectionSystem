"""
offline/promote.py - the gate a retrained model has to pass before it serves.

    uv run python -m offline.promote --candidate data/model_candidate.txt

Run by hand this is barely worth it: you would look at the numbers yourself.
It earns its keep once Airflow retrains at 3am on a Sunday, because the most
common way an automated retraining pipeline hurts you is not failing loudly -
it is quietly shipping a worse model while everyone is asleep.

EVERY THRESHOLD BELOW IS EITHER MEASURED OR MARKED AS NOT MEASURED.

The wobble figures come from training five models on identical data with only
the seed changed, so any spread is noise by construction:

    metric              wobble      gate                headroom
    auc                  0.30%      +0.005 absolute     ~2x
    pr_auc               1.66%      no worse than -5%   ~3x
    recall@fpr0.01       3.66%      no worse than -10%  ~2.7x
    recall@fpr0.001     10.9%       EXCLUDED - only 54 frauds decide it

Refusing is not a failure. The gate doing its job and the gate having nothing
to do both end in exit 0; a non-zero exit means the gate itself broke.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from core.features import FEATURE_NAMES
from core.schema import LABEL_FIELD
from offline.train import TRAINING_SET, evaluate, split_by_time

PRODUCTION = "data/model.txt"

# --- measured thresholds (see the table above) ---------------------------
AUC_MARGIN = 0.005        # candidate must beat production by this much
PR_AUC_FLOOR = 0.95       # ...without pr_auc falling more than 5%
RECALL_FPR_FLOOR = 0.90   # ...or recall at fpr 1% falling more than 10%

# --- NOT measured: judgement calls, deliberately wide --------------------
# There is only one dataset here, so "how much does a healthy week vary" has
# never been observed. These are set loose enough to ignore normal drift and
# only trip on gross breakage. Re-derive them once real weekly runs exist.
MIN_ROWS_RATIO = 0.80
FRAUD_RATE_RANGE = (0.5, 2.0)
MAX_MISSING = 0.90


def sidecar(model_path: Path, suffix: str) -> Path:
    return model_path.with_suffix(suffix)


def check_hygiene(train: pd.DataFrame, cand_meta: dict | None,
                  prod_meta: dict | None) -> list[tuple]:
    """
    Checks that have nothing to do with model quality.

    A pipeline that half-loaded its data can still produce a model with a fine
    AUC - it just only knows half the world. No performance metric catches
    that, so these look at the data itself.
    """
    results = []

    # (3) constant or absent features. Not a statistical call: a feature with
    # one distinct value is broken, not unlucky, so there is no threshold to
    # tune and no risk of rejecting a good model by accident.
    broken = [f for f in FEATURE_NAMES if train[f].nunique() <= 1]
    missing = [f for f in FEATURE_NAMES if train[f].isna().mean() > MAX_MISSING]
    results.append((
        "features usable",
        not broken and not missing,
        f"constant={broken or 'none'} mostly-null={missing or 'none'}",
    ))

    if prod_meta is None or cand_meta is None:
        results.append(("training data vs previous", True,
                        "no previous run recorded - nothing to compare"))
        return results

    # Both figures come from the models' own metadata. Reading the current
    # table instead would compare this script against production and pass no
    # matter what the candidate was actually fitted on.
    # (1) did we train on a comparable amount of data?
    ratio = cand_meta["train_rows"] / prod_meta["train_rows"]
    results.append((
        "training rows vs previous",
        ratio >= MIN_ROWS_RATIO,
        f"{cand_meta['train_rows']:,} vs {prod_meta['train_rows']:,} "
        f"= {ratio:.2f}x (floor {MIN_ROWS_RATIO})",
    ))

    # (2) did the labels arrive looking like labels?
    rate = cand_meta["train_fraud_rate"]
    prev = prod_meta["train_fraud_rate"]
    factor = rate / prev if prev else float("inf")
    low, high = FRAUD_RATE_RANGE
    results.append((
        "fraud rate vs previous",
        low <= factor <= high,
        f"{100 * rate:.2f}% vs {100 * prev:.2f}% = {factor:.2f}x "
        f"(allowed {low}-{high})",
    ))
    return results


def check_provenance(cand_meta: dict | None, test: pd.DataFrame) -> tuple:
    """
    core/features.py refuses to score a transaction using anything that came
    after it. This is the same rule one level up: a candidate must not have
    been fitted on the rows it is about to be marked on. A date arithmetic slip
    in the retraining job, a late-arriving partition, a mis-aimed backfill -
    each of them produces a model that looks excellent for the worst reason,
    and every performance check below will wave it straight through.

    Like the constant-feature check this is a yes/no, not a measurement:
    overlap is a mistake, never noise, so there is no threshold and no chance
    of turning away a model that deserved to go live.
    """
    if not cand_meta or "train_max_dt" not in cand_meta:
        return ("no train/test overlap", False,
                "candidate has no recorded training window - refusing on principle")

    train_end = cand_meta["train_max_dt"]
    leaked = int((test["TransactionDT"] <= train_end).sum())
    return (
        "no train/test overlap",
        leaked == 0,
        f"{leaked:,} of {len(test):,} test rows fall inside the candidate's "
        f"training window" + ("" if leaked == 0 else "   <-- LEAK"),
    )


def check_performance(prod: dict, cand: dict) -> list[tuple]:
    """
    Both sets of numbers come from the SAME test rows, scored moments apart.
    That is what makes a 0.005 threshold defensible: a shared test set cancels
    most of the sampling noise, so we are not up against the +/-0.009 spread
    that a single AUC carries on its own.
    """
    return [
        ("auc improves",
         cand["auc"] >= prod["auc"] + AUC_MARGIN,
         f"{cand['auc']:.4f} vs {prod['auc']:.4f} "
         f"(need +{AUC_MARGIN}, got {cand['auc'] - prod['auc']:+.4f})"),
        ("pr_auc holds",
         cand["pr_auc"] >= prod["pr_auc"] * PR_AUC_FLOOR,
         f"{cand['pr_auc']:.4f} vs {prod['pr_auc']:.4f} "
         f"= {cand['pr_auc'] / prod['pr_auc']:.3f}x (floor {PR_AUC_FLOOR})"),
        ("recall@fpr0.01 holds",
         cand["recall@fpr0.01"] >= prod["recall@fpr0.01"] * RECALL_FPR_FLOOR,
         f"{cand['recall@fpr0.01']:.4f} vs {prod['recall@fpr0.01']:.4f} "
         f"= {cand['recall@fpr0.01'] / prod['recall@fpr0.01']:.3f}x "
         f"(floor {RECALL_FPR_FLOOR})"),
    ]


def install(candidate: Path, production: Path) -> None:
    """Model, scores and metadata move together or the set stops making sense."""
    for suffix in (candidate.suffix, ".preds.parquet", ".meta.json"):
        src = candidate.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, production.with_suffix(suffix))


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide whether to promote.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--production", default=PRODUCTION)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--training-set", default=TRAINING_SET,
                        help="table to judge against; a path lets us feed the "
                             "gate deliberately broken data and watch it refuse")
    args = parser.parse_args()

    candidate, production = Path(args.candidate), Path(args.production)
    if not candidate.exists():
        raise SystemExit(f"no candidate at {candidate}")

    df = pd.read_parquet(args.training_set)
    train, test = split_by_time(df, args.train_frac, 0)

    prod_meta_file = sidecar(production, ".meta.json")
    prod_meta = json.loads(prod_meta_file.read_text()) if prod_meta_file.exists() else None

    cand_meta_file = sidecar(candidate, ".meta.json")
    cand_meta = json.loads(cand_meta_file.read_text()) if cand_meta_file.exists() else None

    checks = check_hygiene(train, cand_meta, prod_meta)
    # Runs before the metrics on purpose: if the candidate saw the test rows,
    # its scores are fiction and there is nothing worth comparing.
    checks.append(check_provenance(cand_meta, test))

    if production.exists():
        # Score both on the same rows, now, rather than trusting numbers that
        # were written at different times against possibly different splits.
        y = test[LABEL_FIELD]
        X = test[FEATURE_NAMES]
        prod_metrics = evaluate(y, lgb.Booster(model_file=str(production)).predict(X))
        cand_metrics = evaluate(y, lgb.Booster(model_file=str(candidate)).predict(X))
        checks += check_performance(prod_metrics, cand_metrics)
    else:
        checks.append(("performance vs production", True,
                       "nothing in production yet - first model goes live"))

    print(f"candidate  {candidate}")
    print(f"production {production}\n")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<28} {detail}")

    if all(passed for _, passed, _ in checks):
        install(candidate, production)
        print(f"\nPROMOTED -> {production}")
    else:
        failed = [n for n, p, _ in checks if not p]
        print(f"\nREFUSED ({', '.join(failed)}). Production model left alone.")


if __name__ == "__main__":
    main()
