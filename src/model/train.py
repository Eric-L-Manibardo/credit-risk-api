"""Baseline CatBoost: sealed test + 5-fold CV, then refit on the 85% pool."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.features.engineering import CATEGORICAL_FEATURES
from src.features.pipeline import load_featured_loans
from src.model.config import CATBOOST_PARAMS
from src.model.metrics import classification_metrics, summarize_cv
from src.model.split import cv_folds, frame_to_xy, sealed_test_split


def _catboost_frame(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str)
    return out


def _make_model() -> CatBoostClassifier:
    return CatBoostClassifier(**CATBOOST_PARAMS)


def train_baseline(df: pd.DataFrame) -> dict[str, Any]:
    """Run the M3 paso 2 protocol. Does not write .cbm (paso 3)."""
    x, y, ids = frame_to_xy(df)
    x_rest, x_test, y_rest, y_test, ids_rest, ids_test = sealed_test_split(x, y, ids)
    x_rest_cb = _catboost_frame(x_rest)
    x_test_cb = _catboost_frame(x_test)

    fold_metrics: list[dict[str, float]] = []
    # Out-of-fold: every row of the 85% pool scored by a model that did not see it.
    oof_proba = np.zeros(len(y_rest), dtype=float)
    splitter = cv_folds(y_rest)
    for train_idx, val_idx in splitter.split(x_rest_cb, y_rest):
        model = _make_model()
        model.fit(
            x_rest_cb.iloc[train_idx],
            y_rest.iloc[train_idx],
            cat_features=list(CATEGORICAL_FEATURES),
        )
        proba = model.predict_proba(x_rest_cb.iloc[val_idx])[:, 1]
        oof_proba[val_idx] = proba
        fold_metrics.append(classification_metrics(y_rest.iloc[val_idx], proba))

    final_model = _make_model()
    final_model.fit(x_rest_cb, y_rest, cat_features=list(CATEGORICAL_FEATURES))
    test_proba = final_model.predict_proba(x_test_cb)[:, 1]
    test_metrics = classification_metrics(y_test, test_proba)

    return {
        "model": final_model,
        "cv": summarize_cv(fold_metrics),
        "test": test_metrics,
        "n_rest": int(len(y_rest)),
        "n_test": int(len(y_test)),
        "test_ids": [int(i) for i in ids_test.tolist()],
        "feature_names": list(x.columns),
        "y_rest": y_rest.to_numpy(),
        "oof_proba": oof_proba,
        "y_test": y_test.to_numpy(),
        "test_proba": test_proba,
    }


def _format_cv(cv: dict[str, dict[str, float]]) -> str:
    lines = []
    for key, stats in cv.items():
        lines.append(f"  {key:16} {stats['mean']:.4f} ± {stats['std']:.4f}")
    return "\n".join(lines)


def _format_test(test: dict[str, float]) -> str:
    skip = {"tn", "fp", "fn", "tp"}
    lines = [f"  {key:16} {value:.4f}" for key, value in test.items() if key not in skip]
    lines.append(
        "  confusion       "
        f"tn={int(test['tn'])} fp={int(test['fp'])} "
        f"fn={int(test['fn'])} tp={int(test['tp'])}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    featured = load_featured_loans()
    result = train_baseline(featured)
    print(f"n_rest={result['n_rest']} n_test={result['n_test']}")
    print("CV (5-fold, mean ± std):")
    print(_format_cv(result["cv"]))
    print("Test (sealed, 15%):")
    print(_format_test(result["test"]))
