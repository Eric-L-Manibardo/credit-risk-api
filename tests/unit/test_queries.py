"""Queries run against the engine fixture, never data/credit.db."""

from __future__ import annotations

import pytest
from sqlalchemy import Engine

from src.data.queries import count_by_credit_class, fetch_loan_by_id, fetch_loans


def test_fetch_loans_returns_all_rows(seeded_engine: Engine) -> None:
    df = fetch_loans(engine=seeded_engine)
    assert len(df) == 2
    assert "id" in df.columns
    assert "credit_class" in df.columns


def test_fetch_loans_filters_in_sql(seeded_engine: Engine) -> None:
    df = fetch_loans(engine=seeded_engine, credit_class="bad")
    assert len(df) == 1
    assert df.iloc[0]["credit_class"] == "bad"
    assert df.iloc[0]["purpose"] == "education"


def test_fetch_loans_rejects_unknown_class(seeded_engine: Engine) -> None:
    with pytest.raises(ValueError, match="credit_class"):
        fetch_loans(engine=seeded_engine, credit_class="maybe")


def test_fetch_loan_by_id(seeded_engine: Engine) -> None:
    row = fetch_loan_by_id(1, engine=seeded_engine)
    assert row is not None
    assert row["id"] == 1
    assert fetch_loan_by_id(999, engine=seeded_engine) is None


def test_count_by_credit_class(seeded_engine: Engine) -> None:
    assert count_by_credit_class(engine=seeded_engine) == {"good": 1, "bad": 1}
