"""Single path: validate then features; load from SQL for training."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import Engine

from src.data.validation import DataValidationError
from src.features import load_featured_loans, prepare_features
from src.features.engineering import DERIVED_COLUMNS
from tests.unit.conftest import VALID_LOAN


def test_prepare_features_validates_then_transforms() -> None:
    out = prepare_features(pd.DataFrame([dict(VALID_LOAN)]))
    for col in DERIVED_COLUMNS:
        assert col in out.columns
    assert out.loc[0, "credit_class"] == "good"


def test_prepare_features_rejects_invalid_without_transform() -> None:
    raw = pd.DataFrame([{**VALID_LOAN, "age": 12}])
    with pytest.raises(DataValidationError):
        prepare_features(raw)
    assert "credit_per_month" not in raw.columns


def test_load_featured_loans_from_sql(seeded_engine: Engine) -> None:
    out = load_featured_loans(engine=seeded_engine)
    assert len(out) == 2
    for col in DERIVED_COLUMNS:
        assert col in out.columns
    bad_only = load_featured_loans(engine=seeded_engine, credit_class="bad")
    assert len(bad_only) == 1
    assert bad_only.iloc[0]["purpose"] == "education"
    assert bad_only.iloc[0]["age_bin"] == "26-35"
