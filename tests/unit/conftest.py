"""Shared fixtures for the data-layer unit tests.

Uses a single in-memory SQLite (StaticPool) so we never touch data/credit.db.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

VALID_LOAN: dict[str, object] = {
    "checking_status": "no checking",
    "duration": 12,
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "credit_amount": 2000,
    "savings_status": "<100",
    "employment": "1<=X<4",
    "installment_commitment": 2,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "real estate",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "yes",
    "credit_class": "good",
}


@pytest.fixture
def engine() -> Generator[Engine, None, None]:
    """In-memory SQLite shared across connections (SQLAlchemy StaticPool)."""
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    yield eng
    eng.dispose()


@pytest.fixture
def valid_loans_frame() -> pd.DataFrame:
    good = dict(VALID_LOAN)
    bad = dict(VALID_LOAN)
    bad["credit_class"] = "bad"
    bad["age"] = 42
    return pd.DataFrame([good, bad])


def write_source_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a CSV using the OpenML column name ``class`` (pre-ingest)."""
    records = []
    for row in rows:
        record = dict(row)
        record["class"] = record.pop("credit_class")
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)
    return path
