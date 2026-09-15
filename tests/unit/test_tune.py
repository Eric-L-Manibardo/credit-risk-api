"""Optuna search uses the 85% pool only. Tiny CatBoost, few trials."""

from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd
import pytest
from optuna.trial import BaseTrial

from src.features.engineering import create_features
from src.model.registry import load_registry, save_model
from src.model.split import frame_to_xy, sealed_test_split
from src.model.tune import (
    run_study,
    save_study_figures,
    suggest_catboost_params,
    trial_pr_auc,
    tune,
)
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


def _tiny_suggest(trial: BaseTrial) -> dict[str, int | float]:
    return {
        "iterations": 30,
        "depth": trial.suggest_int("depth", 3, 4),
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
    }


@pytest.fixture(scope="module")
def tuned() -> dict:
    return tune(_toy_featured(), n_trials=2, suggest=_tiny_suggest)


def test_suggest_params_covers_the_search_keys() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "depth": 4,
            "learning_rate": 0.05,
            "iterations": 300,
            "l2_leaf_reg": 3.0,
        }
    )
    params = suggest_catboost_params(trial)
    assert set(params) == {"depth", "learning_rate", "iterations", "l2_leaf_reg"}
    assert "random_seed" not in params
    assert "auto_class_weights" not in params


def test_run_study_scores_only_the_rest_pool() -> None:
    df = _toy_featured()
    x, y, ids = frame_to_xy(df)
    x_rest, x_test, y_rest, y_test, ids_rest, ids_test = sealed_test_split(x, y, ids)
    study = run_study(x_rest, y_rest, n_trials=2, suggest=_tiny_suggest)
    assert 0.0 <= study.best_value <= 1.0
    assert "depth" in study.best_params
    assert set(ids_test).isdisjoint(set(ids_rest))
    assert len(x_test) > 0
    assert len(y_test) > 0


def test_trial_pr_auc_is_mean_cv_pr_auc() -> None:
    df = _toy_featured()
    x, y, ids = frame_to_xy(df)
    x_rest, _, y_rest, _, _, _ = sealed_test_split(x, y, ids)
    trial = optuna.trial.FixedTrial({"depth": 3})
    score = trial_pr_auc(trial, x_rest, y_rest, suggest=_tiny_suggest)
    assert 0.0 <= score <= 1.0


def test_tune_records_study_and_opens_test_once(tuned: dict) -> None:
    assert tuned["version"] == "tuned-v1"
    assert tuned["tuning"]["n_trials"] == 2
    assert tuned["tuning"]["objective"] == "cv_mean_pr_auc"
    assert tuned["tuning"]["best_params"]["depth"] in {3, 4}
    assert tuned["n_test"] > 0
    assert "pr_auc" in tuned["test"]


def test_save_model_writes_tuning_and_winner_params(tuned: dict, tmp_path: Path) -> None:
    save_model(
        tuned["model"],
        result=tuned,
        model_path=tmp_path / "baseline.cbm",
        registry_path=tmp_path / "registry.json",
    )
    data = load_registry(tmp_path / "registry.json")
    assert data["version"] == "tuned-v1"
    assert data["tuning"]["n_trials"] == 2
    assert data["catboost_params"]["iterations"] == 30
    assert data["catboost_params"]["depth"] in {3, 4}
    assert "verbose" not in data["catboost_params"]


def test_save_study_figures_writes_history(tmp_path: Path) -> None:
    df = _toy_featured()
    x, y, ids = frame_to_xy(df)
    x_rest, _, y_rest, _, _, _ = sealed_test_split(x, y, ids)
    study = run_study(x_rest, y_rest, n_trials=2, suggest=_tiny_suggest)
    written = save_study_figures(study, tmp_path, baseline_pr_auc=0.5)
    names = {path.name for path in written}
    assert "optuna_history.png" in names
    history = tmp_path / "optuna_history.png"
    assert history.stat().st_size > 0
