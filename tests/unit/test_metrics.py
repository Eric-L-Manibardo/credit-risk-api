"""Metric helpers on tiny arrays (no CatBoost)."""

from __future__ import annotations

import numpy as np

from src.model.metrics import classification_metrics, summarize_cv, threshold_sweep


def test_perfect_ranking_and_threshold() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, proba, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["precision_bad"] == 1.0
    assert m["recall_bad"] == 1.0
    assert m["fn"] == 0
    assert m["fp"] == 0
    assert m["cost_5fn_1fp"] == 0.0


def test_cost_counts_fn_five_times_fp() -> None:
    y = np.array([1, 0])
    proba = np.array([0.1, 0.9])  # FN and FP at 0.5
    m = classification_metrics(y, proba, threshold=0.5)
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["cost_5fn_1fp"] == (5 + 1) / 2


def test_threshold_sweep_recall_never_increases_with_threshold() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    proba = np.array([0.1, 0.45, 0.3, 0.6, 0.85, 0.7])
    rows = threshold_sweep(y, proba, thresholds=[0.2, 0.5, 0.8])
    assert [row["threshold"] for row in rows] == [0.2, 0.5, 0.8]
    recalls = [row["recall_bad"] for row in rows]
    assert recalls == sorted(recalls, reverse=True)
    # Ranking metrics do not depend on the cut-off.
    assert len({round(row["roc_auc"], 6) for row in rows}) == 1


def test_summarize_cv_mean_std() -> None:
    folds = [{"roc_auc": 0.7, "pr_auc": 0.5}, {"roc_auc": 0.8, "pr_auc": 0.5}]
    summary = summarize_cv(folds)
    assert summary["roc_auc"]["mean"] == 0.75
    assert summary["pr_auc"]["std"] == 0.0
