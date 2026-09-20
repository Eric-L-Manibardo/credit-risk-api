"""HTTP contracts. Application fields reuse the table's ranges and categories."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.validation import CATEGORICAL_VALUES, NUMERIC_RANGES

APPLICATION_CATEGORIES = {
    name: values for name, values in CATEGORICAL_VALUES.items() if name != "credit_class"
}


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    version: str


class ModelInfoResponse(BaseModel):
    """Operating point and metrics from the registry. Not the search log."""

    model_config = ConfigDict(extra="forbid")

    version: str
    trained_at: str
    n_train: int
    n_test: int
    threshold: float
    cost_ratio: dict[str, int]
    features: list[str]
    cat_features: list[str]
    cv_metrics: dict[str, Any]
    test_metrics: dict[str, Any] = Field(
        description="Sealed-test metrics recorded when the artifact was saved."
    )


class LoanApplication(BaseModel):
    """One application. No ``credit_class``: that is the label we estimate.

    ``personal_status`` (gendered marital status) and ``foreign_worker``
    are columns of the 1994 German Credit extract, not attributes a live
    EU model should score on.
    """

    model_config = ConfigDict(extra="forbid")

    id: int | None = Field(default=None, ge=1)
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str

    @model_validator(mode="after")
    def _domain_rules(self) -> Self:
        for name, allowed in APPLICATION_CATEGORIES.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name}: must be one of {sorted(allowed)}")
        for name, (low, high) in NUMERIC_RANGES.items():
            value = getattr(self, name)
            if not (low <= value <= high):
                raise ValueError(f"{name}: must be in [{low}, {high}]")
        return self


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    p_bad: float
    decision: Literal["approve", "review"]
    threshold: float


class ShapContribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    value: str | int | float
    shap: float


class ExplainResponse(PredictResponse):
    """Predict payload plus the local SHAP breakdown (log-odds, not P(bad))."""

    base_value: float
    units: Literal["log_odds"]
    contributions: list[ShapContribution]
