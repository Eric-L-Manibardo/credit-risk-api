"""Ingest the German Credit CSV into SQLite.

This is the only production-adjacent code that reads the CSV. After this
job, all access is SQL against the loans table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from src.data.database import DEFAULT_DB_PATH, get_engine
from src.data.download import RAW_CSV_PATH, download_dataset
from src.data.schema import INGEST_COLUMNS, loans
from src.data.validation import validate_loans

COLUMN_RENAME = {"class": "credit_class"}


def _strip_wrapping_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove OpenML's extra single quotes from text fields.

    The download keeps a faithful copy of the source. Cleaning belongs here.
    """
    cleaned = df.copy()
    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            continue
        cleaned[col] = cleaned[col].astype("string").str.strip().str.strip("'")
    return cleaned


def prepare_loans_frame(csv_path: Path) -> pd.DataFrame:
    """Load the extract and return a DataFrame that matches the SQL schema."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Run: uv run python -m src.data.download"
        )

    df = pd.read_csv(csv_path)
    df = _strip_wrapping_quotes(df)
    df = df.rename(columns=COLUMN_RENAME)

    missing = set(INGEST_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(INGEST_COLUMNS)
    if missing or extra:
        raise ValueError(f"CSV columns do not match schema. missing={missing} extra={extra}")

    return df[INGEST_COLUMNS]


def reset_loans_table(engine: Engine) -> None:
    """Drop and recreate loans so re-running ingest is idempotent."""
    loans.drop(engine, checkfirst=True)
    loans.create(engine)


def ingest_csv(
    *,
    csv_path: Path | None = None,
    db_path: Path | None = None,
    engine: Engine | None = None,
) -> int:
    """Replace the loans table with the cleaned CSV. Returns the row count.

    Pass ``engine`` in tests (in-memory SQLite). Production uses ``db_path``.
    """
    source = csv_path or RAW_CSV_PATH
    db_engine = engine if engine is not None else get_engine(db_path)
    df = prepare_loans_frame(source)
    validate_loans(df)
    reset_loans_table(db_engine)

    records = df.to_dict(orient="records")
    with db_engine.begin() as conn:
        conn.execute(loans.insert(), records)

    return len(records)


if __name__ == "__main__":
    download_dataset()
    n_rows = ingest_csv()
    print(f"Ingested {n_rows} rows into {DEFAULT_DB_PATH}")
