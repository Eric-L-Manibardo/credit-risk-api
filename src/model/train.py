"""CatBoost fit protocol: sealed test + 5-fold CV, then refit on the 85% pool."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.features.engineering import CATEGORICAL_FEATURES, as_catboost_frame
from src.features.pipeline import load_featured_loans
from src.model.config import CATBOOST_PARAMS, RANDOM_STATE
from src.model.metrics import classification_metrics, summarize_cv
from src.model.split import cv_folds, frame_to_xy, sealed_test_split

Params = dict[str, Any]


def merge_params(overrides: Mapping[str, Any] | None = None) -> Params:
    """CATBOOST_PARAMS plus overrides. Seed, loss and silence stay fixed."""
    merged: Params = dict(CATBOOST_PARAMS)
    if overrides:
        merged.update({key: value for key, value in overrides.items() if key != "verbose"})
    merged["verbose"] = False
    merged["random_seed"] = RANDOM_STATE
    merged["loss_function"] = "Logloss"
    return merged


def _make_model(params: Mapping[str, Any] | None = None) -> CatBoostClassifier:
    return CatBoostClassifier(**merge_params(params))


def cross_validate(
    x: pd.DataFrame,
    y: pd.Series,
    params: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, float]], np.ndarray]:
    """5-fold on one pool. Returns per-fold metrics and out-of-fold P(bad)."""
    x_cb = as_catboost_frame(x)
    fold_metrics: list[dict[str, float]] = []
    oof_proba = np.zeros(len(y), dtype=float)
    splitter = cv_folds(y)
    for train_idx, val_idx in splitter.split(x_cb, y):
        model = _make_model(params)
        model.fit(
            x_cb.iloc[train_idx],
            y.iloc[train_idx],
            cat_features=list(CATEGORICAL_FEATURES),
        )
        proba = model.predict_proba(x_cb.iloc[val_idx])[:, 1]
        oof_proba[val_idx] = proba
        fold_metrics.append(classification_metrics(y.iloc[val_idx], proba))
    return fold_metrics, oof_proba


def train_baseline(
    df: pd.DataFrame,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Split, CV, refit. Does not write files; the CLI does."""
    used = merge_params(params)
    x, y, ids = frame_to_xy(df)
    x_rest, x_test, y_rest, y_test, ids_rest, ids_test = sealed_test_split(x, y, ids)
    fold_metrics, oof_proba = cross_validate(x_rest, y_rest, used)

    x_rest_cb = as_catboost_frame(x_rest)
    x_test_cb = as_catboost_frame(x_test)
    final_model = _make_model(used)
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
        "params": used,
        "version": "baseline-v1",
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
    from src.model.config import artifact_path
    from src.model.registry import save_model

    featured = load_featured_loans()
    result = train_baseline(featured)

    print(f"n_rest={result['n_rest']} n_test={result['n_test']}")
    print("CV (5-fold, mean ± std):")
    print(_format_cv(result["cv"]))
    print("Test (sealed, 15%):")
    print(_format_test(result["test"]))

    registry_path = save_model(result["model"], result=result)
    print("\nArtifacts written:")
    print(f"  model     {artifact_path(result['version'])}")
    print(f"  registry  {registry_path}")
