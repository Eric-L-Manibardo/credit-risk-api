"""Read loans from SQLite via SQLAlchemy Core.

Training, validation, and the API import these functions. They never
read the CSV.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import Engine, func, select

from src.data.database import get_engine
from src.data.schema import CREDIT_CLASSES, loans

LOAN_COLUMNS = [column.name for column in loans.columns]


def _resolve_engine(engine: Engine | None) -> Engine:
    return engine if engine is not None else get_engine()


def fetch_loans(
    *,
    engine: Engine | None = None,
    credit_class: str | None = None,
) -> pd.DataFrame:
    """Return loans as a DataFrame. Optionally filter by credit_class."""
    if credit_class is not None and credit_class not in CREDIT_CLASSES:
        allowed = sorted(CREDIT_CLASSES)
        raise ValueError(f"credit_class must be one of {allowed}, got {credit_class!r}")

    stmt = select(loans)
    if credit_class is not None:
        stmt = stmt.where(loans.c.credit_class == credit_class)

    with _resolve_engine(engine).connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    return pd.DataFrame(rows, columns=LOAN_COLUMNS)


def fetch_loan_by_id(loan_id: int, *, engine: Engine | None = None) -> dict[str, Any] | None:
    """Return one loan as a dict, or None if the id does not exist."""
    stmt = select(loans).where(loans.c.id == loan_id)
    with _resolve_engine(engine).connect() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row is not None else None


def count_by_credit_class(*, engine: Engine | None = None) -> dict[str, int]:
    """Return {'bad': n, 'good': n} from SQL GROUP BY (not a DataFrame value_counts)."""
    stmt = select(loans.c.credit_class, func.count().label("n")).group_by(loans.c.credit_class)
    with _resolve_engine(engine).connect() as conn:
        rows = conn.execute(stmt).all()
    return {str(label): int(n) for label, n in rows}
