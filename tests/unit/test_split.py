"""Split protocol: sort by id, stratify, sealed test, reproducible seed."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import create_features
from src.model.config import RANDOM_STATE, TEST_SIZE
from src.model.split import frame_to_xy, sealed_test_split
from tests.unit.conftest import VALID_LOAN


def _featured_toy(n: int = 200) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n):
        row = dict(VALID_LOAN)
        row["id"] = i + 1
        row["age"] = 20 + (i % 50)
        row["credit_class"] = "bad" if i % 10 < 3 else "good"
        rows.append(row)
    return create_features(pd.DataFrame(rows))


def test_frame_to_xy_drops_id_and_encodes_bad() -> None:
    x, y, ids = frame_to_xy(_featured_toy(20))
    assert "id" not in x.columns
    assert "credit_class" not in x.columns
    assert set(y.unique()) <= {0, 1}
    assert list(ids) == list(range(1, 21))
    assert ids.is_monotonic_increasing


def test_sealed_test_is_stratified_and_reproducible() -> None:
    x, y, ids = frame_to_xy(_featured_toy(200))
    first = sealed_test_split(x, y, ids)
    second = sealed_test_split(x, y, ids)
    _, _, _, y_test, _, ids_test = first
    _, _, _, y_test_2, _, ids_test_2 = second
    assert list(ids_test) == list(ids_test_2)
    assert abs(len(y_test) / len(y) - TEST_SIZE) < 0.02
    rest_ids = set(first[4].tolist())
    test_ids = set(ids_test.tolist())
    assert rest_ids.isdisjoint(test_ids)
    assert rest_ids | test_ids == set(ids.tolist())
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.05)
    assert RANDOM_STATE == 42


def test_shuffled_input_same_split() -> None:
    df = _featured_toy(200)
    shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
    _, _, ids_a = frame_to_xy(df)
    *_, ids_test_a = sealed_test_split(*frame_to_xy(df))
    *_, ids_test_b = sealed_test_split(*frame_to_xy(shuffled))
    assert set(ids_a) == set(df["id"])
    assert set(ids_test_a.tolist()) == set(ids_test_b.tolist())
