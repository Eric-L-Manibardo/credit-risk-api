"""Ranking, calibration, and threshold metrics. Positive class is bad=1."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model.config import DECISION_THRESHOLD, FN_COST, FP_COST


def classification_metrics(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Probabilities are P(bad). Threshold metrics use class bad."""
    y = np.asarray(y_true, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "precision_bad": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_bad": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1_bad": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "cost_5fn_1fp": float((FN_COST * fn + FP_COST * fp) / n),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def threshold_sweep(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    thresholds: ArrayLike | None = None,
) -> list[dict[str, float]]:
    """Metrics at several cut-offs. Ranking metrics repeat: they ignore the threshold."""
    default_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    grid = default_grid if thresholds is None else np.asarray(thresholds)
    return [
        {"threshold": float(t), **classification_metrics(y_true, y_proba, threshold=float(t))}
        for t in grid
    ]


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}


def summarize_cv(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = [key for key in fold_metrics[0] if key not in {"tn", "fp", "fn", "tp"}]
    return {key: mean_std([fold[key] for fold in fold_metrics]) for key in keys}
