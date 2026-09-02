"""Ingest writes to the engine we pass; production DB is never touched."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from src.data.ingest import ingest_csv, prepare_loans_frame
from src.data.queries import fetch_loans
from src.data.validation import DataValidationError
from tests.unit.conftest import VALID_LOAN, write_source_csv


def test_prepare_strips_openml_quotes(tmp_path: Path) -> None:
    row = dict(VALID_LOAN)
    row["checking_status"] = "'no checking'"
    row["credit_history"] = "'existing paid'"
    csv_path = write_source_csv(tmp_path / "quoted.csv", [row])

    df = prepare_loans_frame(csv_path)

    assert df.loc[0, "checking_status"] == "no checking"
    assert df.loc[0, "credit_history"] == "existing paid"
    assert "class" not in df.columns
    assert df.loc[0, "credit_class"] == "good"


def test_prepare_rejects_unknown_columns(tmp_path: Path) -> None:
    csv_path = write_source_csv(tmp_path / "ok.csv", [dict(VALID_LOAN)])
    raw = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(raw.replace("duration", "loan_months", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="columns do not match schema"):
        prepare_loans_frame(csv_path)


def test_ingest_roundtrip_and_idempotent(engine: Engine, tmp_path: Path) -> None:
    bad = dict(VALID_LOAN)
    bad["credit_class"] = "bad"
    csv_path = write_source_csv(tmp_path / "loans.csv", [dict(VALID_LOAN), bad])

    assert ingest_csv(csv_path=csv_path, engine=engine) == 2
    assert ingest_csv(csv_path=csv_path, engine=engine) == 2
    df = fetch_loans(engine=engine)
    assert len(df) == 2
    assert set(df["credit_class"]) == {"good", "bad"}


def test_failed_validation_does_not_drop_existing_table(engine: Engine, tmp_path: Path) -> None:
    good_csv = write_source_csv(tmp_path / "good.csv", [dict(VALID_LOAN)])
    ingest_csv(csv_path=good_csv, engine=engine)

    dirty = dict(VALID_LOAN)
    dirty["age"] = 12
    bad_csv = write_source_csv(tmp_path / "bad.csv", [dirty])
    with pytest.raises(DataValidationError):
        ingest_csv(csv_path=bad_csv, engine=engine)

    remaining = fetch_loans(engine=engine)
    assert len(remaining) == 1
    assert int(remaining.loc[0, "age"]) == 35
