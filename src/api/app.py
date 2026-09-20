"""FastAPI app: load the registered model once, then score and explain."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from catboost import CatBoostClassifier
from fastapi import FastAPI, Request

from src.api.schemas import (
    ExplainResponse,
    HealthResponse,
    LoanApplication,
    ModelInfoResponse,
    PredictResponse,
)
from src.api.scoring import explain_application, score_application
from src.explainability.explainer import make_explainer
from src.model.registry import load_model, load_registry


@dataclass
class Runtime:
    """What every request is allowed to read. Built in the lifespan."""

    registry: dict[str, Any]
    model: CatBoostClassifier
    explainer: Any


def _runtime_from_parts(registry: dict[str, Any], model: CatBoostClassifier) -> Runtime:
    return Runtime(registry=registry, model=model, explainer=make_explainer(model))


def _runtime_from_disk() -> Runtime:
    registry = load_registry()
    return _runtime_from_parts(registry, load_model(registry))


def _model_info(registry: dict[str, Any]) -> ModelInfoResponse:
    return ModelInfoResponse(
        version=str(registry["version"]),
        trained_at=str(registry["trained_at"]),
        n_train=int(registry["n_train"]),
        n_test=int(registry["n_test"]),
        threshold=float(registry["threshold"]),
        cost_ratio={key: int(value) for key, value in registry["cost_ratio"].items()},
        features=list(registry["features"]),
        cat_features=list(registry["cat_features"]),
        cv_metrics=dict(registry["cv_metrics"]),
        test_metrics=dict(registry["test_metrics"]),
    )


def create_app(
    *,
    registry: dict[str, Any] | None = None,
    model: CatBoostClassifier | None = None,
) -> FastAPI:
    """Build the app. Tests pass a toy registry+model; uvicorn loads from disk."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if not hasattr(application.state, "runtime"):
            if registry is not None and model is not None:
                application.state.runtime = _runtime_from_parts(registry, model)
            else:
                application.state.runtime = _runtime_from_disk()
        yield

    application = FastAPI(
        title="Credit Risk API",
        description="CatBoost default scoring with per-application SHAP explanations.",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        runtime: Runtime = request.app.state.runtime
        return HealthResponse(status="ok", version=str(runtime.registry["version"]))

    @application.get("/model/info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        runtime: Runtime = request.app.state.runtime
        return _model_info(runtime.registry)

    @application.post("/predict", response_model=PredictResponse)
    def predict(application: LoanApplication, request: Request) -> PredictResponse:
        runtime: Runtime = request.app.state.runtime
        return score_application(application, model=runtime.model, registry=runtime.registry)

    @application.post("/predict/explain", response_model=ExplainResponse)
    def predict_explain(application: LoanApplication, request: Request) -> ExplainResponse:
        runtime: Runtime = request.app.state.runtime
        return explain_application(
            application,
            model=runtime.model,
            registry=runtime.registry,
            explainer=runtime.explainer,
        )

    return application


app = create_app()
