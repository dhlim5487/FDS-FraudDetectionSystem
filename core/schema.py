"""
The contract for one transaction event.

Design rule: an event carries ONLY what a real payment terminal would send.
Vesta's pre-engineered columns (V1-V339, C1-C14, D1-D15, M1-M9) are excluded
on purpose - we compute our own equivalents in core/features.py.

394 columns in the CSV -> 18 fields on the wire.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any


# --- From train_transaction.csv ------------------------------------------
TRANSACTION_FIELDS = [
    "TransactionID",    # unique id for this purchase
    "TransactionDT",    # seconds from a hidden reference point = EVENT TIME
    "TransactionAmt",   # Transaction Amount, how much was spent
    "ProductCD",        # product category: W, C, R, H, S
    "card1",            # &&& our grouping key &&& : stands in for "the account"
    "card2", "card3", "card5",   # other numeric card attributes
    "card4", "card6",   # card network (visa/mastercard), type (debit/credit)
    "addr1", "addr2",   # billing region codes
    "dist1", "dist2",   # distance signals (masked units)
    "P_emaildomain",    # purchaser email domain
    "R_emaildomain",    # recipient email domain
]

# --- From train_identity.csv (missing for ~75% of transactions) ----------
IDENTITY_FIELDS = [
    "DeviceType",       # mobile / desktop
    "DeviceInfo",       # e.g. "SM-G930V Build/NRD90M", "Windows"
]

# The answer key. Deliberately NOT part of the event.
LABEL_FIELD = "isFraud"

EVENT_FIELDS = TRANSACTION_FIELDS + IDENTITY_FIELDS


@dataclass(frozen=True)
class TransactionEvent:
    """One transaction, exactly as it travels through the system."""

    # Always present
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float
    card1: int

    # Often present, sometimes missing
    ProductCD: str | None
    card2: float | None
    card3: float | None
    card4: str | None
    card5: float | None
    card6: str | None
    addr1: float | None
    addr2: float | None
    dist1: float | None
    dist2: float | None
    P_emaildomain: str | None
    R_emaildomain: str | None

    # Usually missing (~75% of the time)
    DeviceType: str | None = None
    DeviceInfo: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """For sending over Kafka."""
        return asdict(self)

"""pandas uses NaN for missing values; the rest of our system wants None."""
def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def from_csv_row(row: dict[str, Any]) -> TransactionEvent | None:
    """
    Turn one raw CSV row into a TransactionEvent.

    Returns None if the row is unusable - no card1 means no grouping key,
    which means we cannot compute any features for it.
    """
    card1 = _clean(row.get("card1"))
    if card1 is None:
        return None

    values = {name: _clean(row.get(name)) for name in EVENT_FIELDS}

    # These four are never missing, so we can safely force their types.
    values["TransactionID"] = int(values["TransactionID"])
    values["TransactionDT"] = int(values["TransactionDT"])
    values["TransactionAmt"] = float(values["TransactionAmt"])
    values["card1"] = int(card1)

    return TransactionEvent(**values)


# --- Chunk 4: self-test ---------------------------------------------------
# Run this file directly to watch a real CSV row become an event:
#     uv run python core/schema.py
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/train_transaction.csv", nrows=3)
    identity = pd.read_csv("data/train_identity.csv", nrows=3)

    print(f"CSV has {df.shape[1]} columns; our event keeps {len(EVENT_FIELDS)}.\n")

    for _, raw in df.iterrows():
        event = from_csv_row(raw.to_dict())
        print("RAW card1 =", raw["card1"], "-> EVENT:")
        print(" ", event)
        print("  as dict:", event.to_dict())
        print()
