"""Single path: SQL → validate → features.

Training and the API call these helpers. They do not read the CSV or
duplicate validate / create_features.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine

from src.data.queries import fetch_loans
from src.data.validation import validate_loans
from src.features.engineering import create_features


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a loan frame and add derived columns. No database access."""
    validate_loans(df)
    return create_features(df)


def load_featured_loans(
    *,
    engine: Engine | None = None,
    credit_class: str | None = None,
) -> pd.DataFrame:
    """Fetch from SQLite, validate, then create_features. Used for training."""
    return prepare_features(fetch_loans(engine=engine, credit_class=credit_class))
