"""Quality checks on loan data after it is a table (or about to become one).

SQL NOT NULL is not enough: empty strings, out-of-range ages, and unknown
categories still get through. These rules are the business contract.
"""

from __future__ import annotations

import pandas as pd

from src.data.schema import CREDIT_CLASSES, INGEST_COLUMNS

# Domain bounds, not min/max of this extract. A 76-year-old must not fail
# just because the German Credit sample stopped at 75.
NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "duration": (1, 120),
    "credit_amount": (1, 100_000),
    "installment_commitment": (1, 4),
    "residence_since": (1, 4),
    "age": (18, 100),
    "existing_credits": (1, 10),
    "num_dependents": (1, 20),
}

CATEGORICAL_VALUES: dict[str, frozenset[str]] = {
    "checking_status": frozenset({"<0", "0<=X<200", ">=200", "no checking"}),
    "credit_history": frozenset(
        {
            "no credits/all paid",
            "all paid",
            "existing paid",
            "delayed previously",
            "critical/other existing credit",
        }
    ),
    "purpose": frozenset(
        {
            "new car",
            "used car",
            "furniture/equipment",
            "radio/tv",
            "domestic appliance",
            "repairs",
            "education",
            "retraining",
            "business",
            "other",
        }
    ),
    "savings_status": frozenset(
        {"<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"}
    ),
    "employment": frozenset({"unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"}),
    "personal_status": frozenset(
        {"male single", "male mar/wid", "male div/sep", "female div/dep/mar"}
    ),
    "other_parties": frozenset({"none", "co applicant", "guarantor"}),
    "property_magnitude": frozenset({"real estate", "life insurance", "car", "no known property"}),
    "other_payment_plans": frozenset({"none", "bank", "stores"}),
    "housing": frozenset({"own", "rent", "for free"}),
    "job": frozenset(
        {
            "unemp/unskilled non res",
            "unskilled resident",
            "skilled",
            "high qualif/self emp/mgmt",
        }
    ),
    "own_telephone": frozenset({"none", "yes"}),
    "foreign_worker": frozenset({"yes", "no"}),
    "credit_class": CREDIT_CLASSES,
}


class DataValidationError(ValueError):
    """Raised when one or more loan quality checks fail."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        bullet_list = "\n".join(f"- {issue}" for issue in issues)
        super().__init__(f"Loan data failed validation:\n{bullet_list}")


def collect_issues(df: pd.DataFrame) -> list[str]:
    """Return every failed check (empty list means the frame is usable)."""
    issues: list[str] = []

    if df.empty:
        issues.append("frame is empty")
        return issues

    missing = [col for col in INGEST_COLUMNS if col not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
        return issues

    for col in INGEST_COLUMNS:
        n_null = int(df[col].isna().sum())
        if n_null:
            issues.append(f"{col}: {n_null} null value(s)")

    for col, (low, high) in NUMERIC_RANGES.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{col}: expected numeric dtype, got {df[col].dtype}")
            continue
        n_out = int((df[col].dropna() < low).sum() + (df[col].dropna() > high).sum())
        if n_out:
            issues.append(f"{col}: {n_out} value(s) outside [{low}, {high}]")

    for col, allowed in CATEGORICAL_VALUES.items():
        series = df[col].dropna().astype("string").str.strip()
        n_empty = int((series == "").sum())
        if n_empty:
            issues.append(f"{col}: {n_empty} empty string(s)")
        unknown = sorted(set(series[series != ""].unique()) - allowed)
        if unknown:
            issues.append(f"{col}: unknown value(s) {unknown}")

    if "id" in df.columns:
        n_dup = int(df["id"].duplicated().sum())
        if n_dup:
            issues.append(f"id: {n_dup} duplicate(s)")

    return issues


def validate_loans(df: pd.DataFrame) -> None:
    """Raise DataValidationError if the frame would poison training or scoring."""
    issues = collect_issues(df)
    if issues:
        raise DataValidationError(issues)


if __name__ == "__main__":
    from src.data.queries import fetch_loans

    loans_df = fetch_loans()
    validate_loans(loans_df)
    print(f"OK: {len(loans_df)} rows passed validation")
