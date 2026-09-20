"""Score and explain one validated application. Same feature path as training."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from catboost import CatBoostClassifier

from src.api.schemas import ExplainResponse, LoanApplication, PredictResponse, ShapContribution
from src.explainability.explainer import explain_row, shap_values
from src.features.engineering import as_catboost_frame, create_features


def application_frame(application: LoanApplication) -> pd.DataFrame:
    """One-row frame of raw fields. Drops ``id``; never includes the label."""
    return pd.DataFrame([application.model_dump(exclude={"id"})])


def model_frame(application: LoanApplication, registry: dict[str, Any]) -> pd.DataFrame:
    """Feature row aligned to the registry, ready for CatBoost / SHAP."""
    featured = create_features(application_frame(application))
    feature_names = list(registry["features"])
    missing = [col for col in feature_names if col not in featured.columns]
    if missing:
        raise ValueError(f"scored frame missing model features: {missing}")
    return as_catboost_frame(featured[feature_names])


def _decision(p_bad: float, threshold: float) -> Literal["approve", "review"]:
    return "review" if p_bad >= threshold else "approve"


def score_application(
    application: LoanApplication,
    *,
    model: CatBoostClassifier,
    registry: dict[str, Any],
) -> PredictResponse:
    """P(bad) plus approve/review at the registry threshold."""
    x = model_frame(application, registry)
    p_bad = float(model.predict_proba(x)[0, 1])
    threshold = float(registry["threshold"])
    return PredictResponse(
        id=application.id,
        p_bad=p_bad,
        decision=_decision(p_bad, threshold),
        threshold=threshold,
    )


def explain_application(
    application: LoanApplication,
    *,
    model: CatBoostClassifier,
    registry: dict[str, Any],
    explainer: Any,
) -> ExplainResponse:
    """Score plus the local SHAP payload (log-odds)."""
    x = model_frame(application, registry)
    p_bad = float(model.predict_proba(x)[0, 1])
    threshold = float(registry["threshold"])
    values, base = shap_values(explainer, x)
    payload = explain_row(
        feature_names=list(x.columns),
        feature_values=x.iloc[0].to_numpy(),
        shap_row=values[0],
        base_value=base,
        p_bad=p_bad,
        loan_id=application.id,
    )
    return ExplainResponse(
        id=application.id,
        p_bad=p_bad,
        decision=_decision(p_bad, threshold),
        threshold=threshold,
        base_value=float(payload["base_value"]),
        units="log_odds",
        contributions=[ShapContribution(**item) for item in payload["contributions"]],
    )
