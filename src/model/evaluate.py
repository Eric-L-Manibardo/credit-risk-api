"""Evaluation figures and classification report.

Re-runs the training protocol with the hyperparameters in
``models/registry.json`` (or ``CATBOOST_PARAMS`` if the registry is missing).
It does not load the ``.cbm``: out-of-fold curves need per-fold scores, and
the binary only stores the final 85% fit.

Two evaluation sets are available, and they answer different questions:

* ``oof``  — out-of-fold scores over the whole 85% pool (~850 rows). Every row
  is scored by a fold model that did not train on it, so curves are far more
  stable than a single 150-row slice. This is the set to read while iterating.
* ``test`` — the sealed 15%. Looked at once, at the end, as a reality check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from src.features.pipeline import load_featured_loans
from src.model.config import DECISION_THRESHOLD, FN_COST, FP_COST
from src.model.metrics import classification_metrics, threshold_sweep
from src.model.train import train_baseline

matplotlib.use("Agg")  # No display under WSL or CI: figures go straight to PNG.

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
CLASS_NAMES = ("good", "bad")
SPLIT_LABELS = ("oof", "test")


def _as_arrays(y_true: ArrayLike, y_proba: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=int), np.asarray(y_proba, dtype=float)


def plot_confusion(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> Figure:
    """Counts of the four outcomes. The bottom-left cell is the expensive one."""
    y, proba = _as_arrays(y_true, y_proba)
    pred = (proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y,
        pred,
        display_labels=list(CLASS_NAMES),
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion matrix at threshold {threshold:.2f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc_pr(y_true: ArrayLike, y_proba: ArrayLike) -> Figure:
    """Ranking quality, threshold-free. PR is the honest one under 30% bad."""
    y, proba = _as_arrays(y_true, y_proba)
    scores = classification_metrics(y, proba)
    fpr, tpr, _ = roc_curve(y, proba)
    precision, recall, _ = precision_recall_curve(y, proba)
    base_rate = float(y.mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].plot(fpr, tpr, color="tab:blue", label=f"ROC-AUC {scores['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", label="coin flip (0.500)")
    axes[0].set_xlabel("False positive rate: good clients flagged as bad")
    axes[0].set_ylabel("True positive rate: bad clients caught")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")

    axes[1].step(
        recall, precision, where="post", color="tab:red", label=f"PR-AUC {scores['pr_auc']:.3f}"
    )
    axes[1].axhline(base_rate, ls="--", color="grey", label=f"base rate {base_rate:.3f}")
    axes[1].set_xlabel("Recall: share of bad clients caught")
    axes[1].set_ylabel("Precision: share of alarms that were real")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Precision-Recall curve (class bad)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_calibration(y_true: ArrayLike, y_proba: ArrayLike, *, n_bins: int = 10) -> Figure:
    """Is a 0.30 score really a 30% default rate? Only this plot answers that."""
    y, proba = _as_arrays(y_true, y_proba)
    scores = classification_metrics(y, proba)
    prob_true, prob_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfectly calibrated")
    ax.plot(
        prob_pred, prob_true, "o-", color="tab:green", label=f"model (Brier {scores['brier']:.3f})"
    )
    ax.set_xlabel("Mean predicted P(bad) in the bin")
    ax.set_ylabel("Observed share of bad in the bin")
    ax.set_title(f"Calibration, {n_bins} equal-sized bins")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_score_distribution(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> Figure:
    """Overlap of the two score distributions: what a threshold actually cuts."""
    y, proba = _as_arrays(y_true, y_proba)
    bins = np.linspace(0.0, 1.0, 21).tolist()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(proba[y == 0], bins=bins, alpha=0.6, color="tab:blue", label="actual good")
    ax.hist(proba[y == 1], bins=bins, alpha=0.6, color="tab:red", label="actual bad")
    ax.axvline(threshold, color="black", ls="--", label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Predicted P(bad)")
    ax.set_ylabel("Clients")
    ax.set_title("Score distribution by actual class")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_threshold_sweep(y_true: ArrayLike, y_proba: ArrayLike) -> Figure:
    """The stakeholder plot: what we gain and what it costs at every cut-off."""
    rows = threshold_sweep(y_true, y_proba)
    grid = [row["threshold"] for row in rows]
    cheapest = min(rows, key=lambda row: row["cost_5fn_1fp"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for key, colour, name in (
        ("precision_bad", "tab:blue", "precision (bad)"),
        ("recall_bad", "tab:red", "recall (bad)"),
        ("f1_bad", "tab:purple", "F1 (bad)"),
    ):
        axes[0].plot(grid, [row[key] for row in rows], color=colour, label=name)
    axes[0].axvline(
        DECISION_THRESHOLD, ls="--", color="grey", label=f"current {DECISION_THRESHOLD:.2f}"
    )
    axes[0].set_xlabel("Decision threshold on P(bad)")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Precision / recall trade-off")
    axes[0].legend(loc="lower center", fontsize=8)

    axes[1].plot(
        grid,
        [row["cost_5fn_1fp"] for row in rows],
        color="tab:orange",
        label=f"cost ({FN_COST}xFN + {FP_COST}xFP) / client",
    )
    axes[1].axvline(
        DECISION_THRESHOLD, ls="--", color="grey", label=f"current {DECISION_THRESHOLD:.2f}"
    )
    axes[1].axvline(
        cheapest["threshold"], ls=":", color="black", label=f"cheapest {cheapest['threshold']:.2f}"
    )
    axes[1].set_xlabel("Decision threshold on P(bad)")
    axes[1].set_ylabel("Cost per client (Hofmann matrix)")
    axes[1].set_title("Expected cost per client")
    axes[1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig


def build_figures(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Figure]:
    """All figures for one evaluation set, keyed by file stem."""
    return {
        "confusion_matrix": plot_confusion(y_true, y_proba, threshold=threshold),
        "roc_pr": plot_roc_pr(y_true, y_proba),
        "calibration": plot_calibration(y_true, y_proba),
        "score_distribution": plot_score_distribution(y_true, y_proba, threshold=threshold),
        "threshold_sweep": plot_threshold_sweep(y_true, y_proba),
    }


def report_text(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> str:
    """sklearn classification_report plus the metrics a threshold cannot show."""
    y, proba = _as_arrays(y_true, y_proba)
    pred = (proba >= threshold).astype(int)
    scores = classification_metrics(y, proba, threshold=threshold)
    table = classification_report(
        y,
        pred,
        target_names=list(CLASS_NAMES),
        digits=3,
        zero_division=0,
    )
    tail = (
        f"threshold-free: roc_auc {scores['roc_auc']:.3f}  "
        f"pr_auc {scores['pr_auc']:.3f}  brier {scores['brier']:.3f}\n"
        f"at threshold {threshold:.2f}: cost/client {scores['cost_5fn_1fp']:.3f}  "
        f"tn={int(scores['tn'])} fp={int(scores['fp'])} "
        f"fn={int(scores['fn'])} tp={int(scores['tp'])}\n"
    )
    return f"{table}\n{tail}"


def save_report(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    label: str,
    outdir: Path = FIGURES_DIR,
    threshold: float = DECISION_THRESHOLD,
) -> list[Path]:
    """Write ``<label>_*.png`` plus the text report. Returns what it wrote."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, fig in build_figures(y_true, y_proba, threshold=threshold).items():
        path = outdir / f"{label}_{stem}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    txt = outdir / f"{label}_classification_report.txt"
    txt.write_text(report_text(y_true, y_proba, threshold=threshold), encoding="utf-8")
    written.append(txt)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation figures for the model in models/registry.json."
    )
    parser.add_argument(
        "--split",
        choices=(*SPLIT_LABELS, "both"),
        default="both",
        help="oof = 5-fold out-of-fold pool, test = sealed 15%%.",
    )
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--outdir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--prefix",
        default="",
        help="Filename prefix, e.g. tuned → tuned_oof_roc_pr.png. Empty keeps oof_/test_.",
    )
    return parser.parse_args()


def _file_label(split_label: str, prefix: str) -> str:
    return f"{prefix}_{split_label}" if prefix else split_label


if __name__ == "__main__":
    args = _parse_args()
    from src.model.config import REGISTRY_PATH
    from src.model.registry import load_registry

    params = None
    version = "defaults (CATBOOST_PARAMS)"
    if REGISTRY_PATH.exists():
        registry = load_registry()
        params = registry.get("catboost_params")
        version = str(registry.get("version", "unknown"))
        print(f"Evaluating {version} from {REGISTRY_PATH}")
    else:
        print(f"No {REGISTRY_PATH}; evaluating CATBOOST_PARAMS defaults")

    result = train_baseline(load_featured_loans(), params=params)
    evaluation_sets = {
        "oof": (result["y_rest"], result["oof_proba"]),
        "test": (result["y_test"], result["test_proba"]),
    }
    labels = list(SPLIT_LABELS) if args.split == "both" else [args.split]

    for split_label in labels:
        y_split, proba_split = evaluation_sets[split_label]
        file_label = _file_label(split_label, args.prefix)
        print(f"\n=== {version} / {split_label} (n={len(y_split)}, bad={int(np.sum(y_split))}) ===")
        print(report_text(y_split, proba_split, threshold=args.threshold))
        for written_path in save_report(
            y_split,
            proba_split,
            label=file_label,
            outdir=args.outdir,
            threshold=args.threshold,
        ):
            print(f"  wrote {written_path}")
