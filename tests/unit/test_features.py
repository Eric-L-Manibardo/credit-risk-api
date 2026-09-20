"""create_features: pure transform, fixed bins, no one-hot, no target drop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    CATEGORICAL_FEATURES,
    DERIVED_COLUMNS,
    as_catboost_frame,
    create_features,
)
from tests.unit.conftest import VALID_LOAN

AGE_CASES = (
    (18, "18-25"),
    (25, "18-25"),
    (26, "26-35"),
    (35, "26-35"),
    (36, "36-50"),
    (50, "36-50"),
    (51, "51+"),
    (76, "51+"),
)


def _loan_frame(**overrides: object) -> pd.DataFrame:
    row = dict(VALID_LOAN)
    row["id"] = 1
    row.update(overrides)
    return pd.DataFrame([row])


def test_single_row_adds_derived_and_keeps_target() -> None:
    raw = _loan_frame()
    out = create_features(raw)
    assert list(DERIVED_COLUMNS) == ["credit_per_month", "log_credit_amount", "age_bin"]
    for col in DERIVED_COLUMNS:
        assert col in out.columns
    assert "has_checking" not in out.columns
    assert out.loc[0, "credit_class"] == "good"
    assert out.loc[0, "id"] == 1
    assert out.loc[0, "purpose"] == "radio/tv"
    assert "purpose_radio/tv" not in out.columns


def test_does_not_mutate_input() -> None:
    raw = _loan_frame()
    create_features(raw)
    assert "credit_per_month" not in raw.columns


def test_credit_per_month_and_log() -> None:
    out = create_features(_loan_frame(credit_amount=2000, duration=12))
    assert out.loc[0, "credit_per_month"] == pytest.approx(2000 / 12)
    assert out.loc[0, "log_credit_amount"] == pytest.approx(np.log1p(2000))


@pytest.mark.parametrize(("age", "label"), AGE_CASES)
def test_age_bin_edges(age: int, label: str) -> None:
    out = create_features(_loan_frame(age=age))
    assert out.loc[0, "age_bin"] == label


def test_age_bin_stable_on_one_row_vs_batch() -> None:
    """Fixed edges: a 51-year-old is 51+ even if they are the only row."""
    alone = create_features(_loan_frame(age=51))
    batch = create_features(
        pd.concat(
            [_loan_frame(age=22), _loan_frame(id=2, age=51)],
            ignore_index=True,
        )
    )
    assert alone.loc[0, "age_bin"] == "51+"
    assert batch.loc[1, "age_bin"] == "51+"


def test_missing_column_raises() -> None:
    raw = _loan_frame().drop(columns=["duration"])
    with pytest.raises(ValueError, match="missing columns"):
        create_features(raw)


def test_as_catboost_frame_casts_categoricals_to_python_str() -> None:
    out = as_catboost_frame(create_features(_loan_frame()))
    for col in CATEGORICAL_FEATURES:
        assert out[col].map(type).eq(str).all()


def test_as_catboost_frame_missing_column_raises() -> None:
    featured = create_features(_loan_frame()).drop(columns=["age_bin"])
    with pytest.raises(ValueError, match="missing columns"):
        as_catboost_frame(featured)
