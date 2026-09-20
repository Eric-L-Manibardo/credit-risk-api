"""Deterministic feature transform: same columns at train time and at score time.

Does not read CSV or SQLite. Age bins are module constants, not quantiles
of the current batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("credit_amount", "duration", "age")

# pd.cut right=True: (17, 25] → 18-25, (25, 35] → 26-35, etc.
AGE_BIN_EDGES: tuple[float, ...] = (17, 25, 35, 50, 100)
AGE_BIN_LABELS: tuple[str, ...] = ("18-25", "26-35", "36-50", "51+")

DERIVED_COLUMNS = (
    "credit_per_month",
    "log_credit_amount",
    "age_bin",
)

# Passed to CatBoost as cat_features in M3. age_bin is new; originals stay text.
CATEGORICAL_FEATURES = (
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
    "age_bin",
)


def as_catboost_frame(x: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to Python str. CatBoost (and SHAP) need that."""
    missing = [col for col in CATEGORICAL_FEATURES if col not in x.columns]
    if missing:
        raise ValueError(f"as_catboost_frame missing columns: {missing}")
    out = x.copy()
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str)
    return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with derived columns. Does not drop id or target."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"create_features missing columns: {missing}")

    out = df.copy()
    out["credit_per_month"] = out["credit_amount"] / out["duration"]
    out["log_credit_amount"] = np.log1p(out["credit_amount"])
    out["age_bin"] = pd.cut(
        out["age"],
        bins=list(AGE_BIN_EDGES),
        labels=list(AGE_BIN_LABELS),
        right=True,
        include_lowest=True,
    ).astype("string")
    return out
