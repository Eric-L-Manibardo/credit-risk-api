"""SHAP helpers. The CLI lives in ``python -m src.explainability.explain``."""

from src.explainability.explainer import (
    explain_row,
    make_explainer,
    mean_abs_shap,
    shap_values,
    to_explanation,
)

__all__ = [
    "explain_row",
    "make_explainer",
    "mean_abs_shap",
    "shap_values",
    "to_explanation",
]
