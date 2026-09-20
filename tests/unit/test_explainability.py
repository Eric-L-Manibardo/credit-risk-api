"""SHAP helpers on a tiny CatBoost (module-scoped). Does not touch models/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.explainability.explain import save_global_bar
from src.explainability.explainer import (
    explain_row,
    make_explainer,
    mean_abs_shap,
    shap_values,
    to_explanation,
)
from src.features.engineering import as_catboost_frame, create_features
from src.model.split import frame_to_xy
from src.model.train import train_baseline
from tests.unit.conftest import VALID_LOAN


def _toy_featured(n: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n):
        row = dict(VALID_LOAN)
        row["id"] = i + 1
        row["age"] = 20 + (i % 50)
        row["credit_class"] = "bad" if i % 10 < 3 else "good"
        rows.append(row)
    return create_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def trained() -> dict:
    return train_baseline(_toy_featured(), params={"iterations": 40, "depth": 3})


def test_shap_adds_up_to_raw_score(trained: dict) -> None:
    x, _, _ = frame_to_xy(_toy_featured(12))
    x_cb = as_catboost_frame(x)
    explainer = make_explainer(trained["model"])
    values, base = shap_values(explainer, x_cb)
    raw = trained["model"].predict(x_cb, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(base + values.sum(axis=1), raw, rtol=1e-5, atol=1e-5)


def test_explain_row_lists_every_feature(trained: dict) -> None:
    x, _, ids = frame_to_xy(_toy_featured(12))
    x_cb = as_catboost_frame(x)
    explainer = make_explainer(trained["model"])
    values, base = shap_values(explainer, x_cb.iloc[:1])
    proba = float(trained["model"].predict_proba(x_cb.iloc[:1])[0, 1])
    payload = explain_row(
        feature_names=list(x_cb.columns),
        feature_values=x_cb.iloc[0].to_numpy(),
        shap_row=values[0],
        base_value=base,
        p_bad=proba,
        loan_id=int(ids.iloc[0]),
    )
    assert payload["units"] == "log_odds"
    assert payload["id"] == 1
    assert 0.0 <= payload["p_bad"] <= 1.0
    assert {item["feature"] for item in payload["contributions"]} == set(x_cb.columns)
    abs_order = [abs(item["shap"]) for item in payload["contributions"]]
    assert abs_order == sorted(abs_order, reverse=True)


def test_mean_abs_shap_ranks_all_columns(trained: dict) -> None:
    x, _, _ = frame_to_xy(_toy_featured(12))
    values, _ = shap_values(make_explainer(trained["model"]), as_catboost_frame(x))
    ranked = mean_abs_shap(values, list(x.columns))
    assert {item["feature"] for item in ranked} == set(x.columns)
    scores = [item["mean_abs_shap"] for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_save_global_bar_writes_png(trained: dict, tmp_path: Path) -> None:
    x, _, _ = frame_to_xy(_toy_featured(12))
    x_cb = as_catboost_frame(x)
    values, base = shap_values(make_explainer(trained["model"]), x_cb)
    path = save_global_bar(to_explanation(x_cb, values, base), tmp_path / "bar.png")
    assert path.exists() and path.stat().st_size > 0
