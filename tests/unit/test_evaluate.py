"""Figures and text report on synthetic probabilities (no CatBoost, no DB)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from src.model.evaluate import build_figures, report_text, save_report


def _synthetic(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y = (rng.random(n) < 0.3).astype(int)
    # Separable but overlapping, so every plot has both classes in most bins.
    proba = np.clip(rng.normal(loc=np.where(y == 1, 0.65, 0.35), scale=0.15), 0.01, 0.99)
    return y, proba


def test_build_figures_returns_every_panel() -> None:
    y, proba = _synthetic()
    figures = build_figures(y, proba)
    assert set(figures) == {
        "confusion_matrix",
        "roc_pr",
        "calibration",
        "score_distribution",
        "threshold_sweep",
    }
    assert all(isinstance(fig, Figure) for fig in figures.values())


def test_report_text_names_classes_and_adds_ranking_metrics() -> None:
    y, proba = _synthetic()
    text = report_text(y, proba, threshold=0.5)
    assert "good" in text
    assert "bad" in text
    assert "roc_auc" in text
    assert "brier" in text


def test_save_report_writes_pngs_and_txt(tmp_path: Path) -> None:
    y, proba = _synthetic()
    written = save_report(y, proba, label="oof", outdir=tmp_path)
    assert all(path.exists() and path.stat().st_size > 0 for path in written)
    assert sum(path.suffix == ".png" for path in written) == 5
    assert tmp_path / "oof_classification_report.txt" in written
