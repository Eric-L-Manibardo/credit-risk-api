"""SHAP explanations for the CatBoost credit model.

CatBoost categorical splits only work with TreeExplainer in
``tree_path_dependent`` mode, which emits **log-odds** (raw formula),
not probabilities. Positive values push toward class ``bad``.
``predict_proba`` is still the number the API serves; SHAP explains
the score underneath.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from numpy.typing import ArrayLike

from src.features.engineering import as_catboost_frame

GLOBAL_SAMPLE = 200


def make_explainer(model: CatBoostClassifier) -> Any:
    """Path-dependent TreeExplainer. The only mode CatBoost categoricals allow."""
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def shap_values(explainer: Any, x: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Return (n × features) log-odds contributions and the scalar base value."""
    x_cb = as_catboost_frame(x)
    raw = explainer.shap_values(x_cb)
    values = np.asarray(getattr(raw, "values", raw), dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected 2-d SHAP matrix, got shape {values.shape}")
    base = float(np.ravel(explainer.expected_value)[-1])
    return values, base


def to_explanation(x: pd.DataFrame, values: np.ndarray, base_value: float) -> Any:
    """Bundle a SHAP matrix into the object the plotting helpers expect."""
    x_cb = as_catboost_frame(x)
    n = len(values)
    return shap.Explanation(
        values=values,
        base_values=np.full(n, float(base_value)),
        data=x_cb.to_numpy(),
        feature_names=list(x_cb.columns),
    )


def explain_row(
    *,
    feature_names: list[str],
    feature_values: ArrayLike,
    shap_row: ArrayLike,
    base_value: float,
    p_bad: float,
    loan_id: int | None = None,
) -> dict[str, Any]:
    """Local breakdown, sorted by |SHAP|. Additive in log-odds, not in P(bad)."""
    names = list(feature_names)
    values = list(np.asarray(feature_values, dtype=object))
    contrib = np.asarray(shap_row, dtype=float).ravel()
    if not (len(names) == len(values) == len(contrib)):
        raise ValueError("feature names, values, and SHAP row must have the same length")
    items = [
        {
            "feature": name,
            "value": _json_value(raw),
            "shap": float(shap_i),
        }
        for name, raw, shap_i in zip(names, values, contrib, strict=True)
    ]
    items.sort(key=lambda item: abs(float(item["shap"])), reverse=True)
    payload: dict[str, Any] = {
        "p_bad": float(p_bad),
        "base_value": float(base_value),
        "units": "log_odds",
        "contributions": items,
    }
    if loan_id is not None:
        payload["id"] = int(loan_id)
    return payload


def mean_abs_shap(shap_matrix: ArrayLike, feature_names: list[str]) -> list[dict[str, Any]]:
    """Global ranking: mean |SHAP| per feature, descending."""
    matrix = np.asarray(shap_matrix, dtype=float)
    means = np.abs(matrix).mean(axis=0)
    ranked: list[dict[str, Any]] = [
        {"feature": name, "mean_abs_shap": float(score)}
        for name, score in zip(feature_names, means, strict=True)
    ]
    ranked.sort(key=lambda item: float(item["mean_abs_shap"]), reverse=True)
    return ranked


def _json_value(raw: object) -> str | int | float:
    if isinstance(raw, (np.integer, int)):
        return int(raw)
    if isinstance(raw, (np.floating, float)):
        return float(raw)
    return str(raw)
