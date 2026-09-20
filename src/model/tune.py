"""Optuna search for CatBoost, scored on 5-fold CV PR-AUC of the 85% pool.

The sealed 15% test is split first and then ignored until the study is
over. Each trial sees only out-of-fold scores on the remainder. With 1000
rows, a long search mostly memorizes the folds: keep ``n_trials`` small.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import BaseTrial

from src.features.pipeline import load_featured_loans
from src.model.config import OPTUNA_N_TRIALS, RANDOM_STATE, artifact_path
from src.model.metrics import summarize_cv
from src.model.split import frame_to_xy, sealed_test_split
from src.model.train import _format_cv, _format_test, cross_validate, train_baseline

optuna.logging.set_verbosity(optuna.logging.WARNING)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures" / "optuna"
# Published baseline CV PR-AUC (make train). Drawn on the history plot for context.
BASELINE_CV_PR_AUC = 0.609
N_TRIALS = OPTUNA_N_TRIALS
SuggestFn = Callable[[BaseTrial], dict[str, Any]]


def suggest_catboost_params(trial: BaseTrial) -> dict[str, Any]:
    """Search around the baseline defaults. Seed and class weights stay fixed."""
    return {
        "depth": trial.suggest_int("depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "iterations": trial.suggest_int("iterations", 150, 400, step=50),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0, log=True),
    }


def trial_pr_auc(
    trial: BaseTrial,
    x_rest: pd.DataFrame,
    y_rest: pd.Series,
    suggest: SuggestFn = suggest_catboost_params,
) -> float:
    """Mean 5-fold PR-AUC on the CV pool. Does not receive the sealed test."""
    fold_metrics, _ = cross_validate(x_rest, y_rest, suggest(trial))
    return summarize_cv(fold_metrics)["pr_auc"]["mean"]


def run_study(
    x_rest: pd.DataFrame,
    y_rest: pd.Series,
    *,
    n_trials: int = N_TRIALS,
    suggest: SuggestFn = suggest_catboost_params,
) -> optuna.Study:
    """TPE search. ``x_rest`` must already be the 85% pool, not the full frame."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: trial_pr_auc(trial, x_rest, y_rest, suggest=suggest),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study


def tune(
    df: pd.DataFrame,
    *,
    n_trials: int = N_TRIALS,
    suggest: SuggestFn = suggest_catboost_params,
) -> dict[str, Any]:
    """Search on the 85% pool, then ``train_baseline`` once with the winner."""
    x, y, ids = frame_to_xy(df)
    x_rest, _x_test, y_rest, _y_test, _ids_rest, _ids_test = sealed_test_split(x, y, ids)
    study = run_study(x_rest, y_rest, n_trials=n_trials, suggest=suggest)
    winner = suggest(study.best_trial)
    result = train_baseline(df, params=winner)
    result["version"] = "tuned-v1"
    result["tuning"] = {
        "n_trials": n_trials,
        "objective": "cv_mean_pr_auc",
        "best_value": float(study.best_value),
        "best_params": winner,
    }
    result["_study"] = study
    return result


def save_study_figures(
    study: optuna.Study,
    outdir: Path,
    *,
    baseline_pr_auc: float | None = None,
) -> list[Path]:
    """Optimization history and parameter importances. Regenerable; not committed from here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    values: list[float] = []
    for trial in study.trials:
        if trial.value is None:
            continue
        values.append(float(trial.value))
    best_so_far: list[float] = []
    running = float("-inf")
    for value in values:
        running = max(running, value)
        best_so_far.append(running)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(range(1, len(values) + 1), values, color="tab:blue", zorder=3, label="trial")
    ax.plot(range(1, len(values) + 1), best_so_far, color="tab:orange", label="best so far")
    if baseline_pr_auc is not None:
        ax.axhline(
            baseline_pr_auc,
            color="grey",
            ls="--",
            label=f"baseline CV PR-AUC {baseline_pr_auc:.3f}",
        )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean 5-fold PR-AUC")
    ax.set_title("Optuna search (sealed test closed)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    history_path = outdir / "optuna_history.png"
    fig.savefig(history_path, dpi=120)
    plt.close(fig)
    written.append(history_path)

    try:
        from optuna.importance import get_param_importances

        importance = get_param_importances(study)
        items = sorted(importance.items(), key=lambda pair: pair[1])
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.barh([name for name, _ in items], [value for _, value in items], color="tab:blue")
        ax.set_xlabel("Fraction of CV PR-AUC movement")
        ax.set_title("Which hyperparameters the search used")
        fig.tight_layout()
        importance_path = outdir / "optuna_param_importances.png"
        fig.savefig(importance_path, dpi=120)
        plt.close(fig)
        written.append(importance_path)
    except (ValueError, RuntimeError, ImportError):
        pass

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search on CV PR-AUC (sealed test closed).")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    return parser.parse_args()


if __name__ == "__main__":
    from src.model.registry import save_model

    args = _parse_args()
    featured = load_featured_loans()
    result = tune(featured, n_trials=args.n_trials)

    print(f"n_rest={result['n_rest']} n_test={result['n_test']}")
    print(
        f"Optuna: {result['tuning']['n_trials']} trials, objective={result['tuning']['objective']}"
    )
    print(f"  best CV PR-AUC {result['tuning']['best_value']:.4f}")
    print(f"  best params    {result['tuning']['best_params']}")
    print("CV (5-fold, mean ± std) with winner:")
    print(_format_cv(result["cv"]))
    print("Test (sealed, 15%, opened once):")
    print(_format_test(result["test"]))

    registry_path = save_model(result["model"], result=result)
    print("\nArtifacts written:")
    print(f"  model     {artifact_path(result['version'])}")
    print(f"  registry  {registry_path}")
    for path in save_study_figures(
        result["_study"],
        FIGURES_DIR,
        baseline_pr_auc=BASELINE_CV_PR_AUC,
    ):
        print(f"  figure    {path}")
