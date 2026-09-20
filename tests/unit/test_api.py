"""Health, model/info, predict, and explain on a toy artifact."""

from __future__ import annotations

import pytest
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient

from src.api.schemas import APPLICATION_CATEGORIES, LoanApplication
from src.api.scoring import model_frame, score_application
from src.explainability.explainer import make_explainer, shap_values
from tests.conftest import application_payload


def test_health_reports_ok_and_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "toy-v1"}


def test_model_info_exposes_operating_point(client: TestClient) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "toy-v1"
    assert body["threshold"] == 0.5
    assert body["cost_ratio"] == {"fn": 5, "fp": 1}
    assert body["n_train"] > 0 and body["n_test"] > 0
    assert "checking_status" in body["features"]
    assert "age_bin" in body["cat_features"]
    assert "pr_auc" in body["cv_metrics"]
    assert "pr_auc" in body["test_metrics"]
    assert "catboost_params" not in body
    assert "tuning" not in body


def test_predict_returns_probability_and_decision(
    client: TestClient,
    trained_api: tuple[dict, CatBoostClassifier],
) -> None:
    registry, model = trained_api
    response = client.post("/predict", json=application_payload())
    assert response.status_code == 200
    body = response.json()
    expected = score_application(
        LoanApplication.model_validate(application_payload()),
        model=model,
        registry=registry,
    )
    assert body["id"] == 7
    assert 0.0 <= body["p_bad"] <= 1.0
    assert body["p_bad"] == pytest.approx(expected.p_bad)
    assert body["threshold"] == 0.5
    assert body["decision"] == expected.decision
    assert body["decision"] in {"approve", "review"}


def test_predict_rejects_underage(client: TestClient) -> None:
    response = client.post("/predict", json=application_payload(age=12))
    assert response.status_code == 422


def test_predict_rejects_unknown_category(client: TestClient) -> None:
    response = client.post("/predict", json=application_payload(checking_status="gold"))
    assert response.status_code == 422


def test_predict_rejects_label_in_body(client: TestClient) -> None:
    response = client.post("/predict", json=application_payload(credit_class="bad"))
    assert response.status_code == 422


def test_explain_matches_predict_and_adds_up(
    client: TestClient,
    trained_api: tuple[dict, CatBoostClassifier],
) -> None:
    registry, model = trained_api
    payload = application_payload()
    scored = client.post("/predict", json=payload).json()
    response = client.post("/predict/explain", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == scored["id"]
    assert body["p_bad"] == pytest.approx(scored["p_bad"])
    assert body["decision"] == scored["decision"]
    assert body["threshold"] == scored["threshold"]
    assert body["units"] == "log_odds"
    names = [item["feature"] for item in body["contributions"]]
    assert set(names) == set(registry["features"])
    abs_order = [abs(item["shap"]) for item in body["contributions"]]
    assert abs_order == sorted(abs_order, reverse=True)
    x = model_frame(LoanApplication.model_validate(payload), registry)
    values, base = shap_values(make_explainer(model), x)
    raw = float(model.predict(x, prediction_type="RawFormulaVal")[0])
    shap_sum = sum(float(item["shap"]) for item in body["contributions"])
    assert body["base_value"] == pytest.approx(base)
    assert base + float(values.sum()) == pytest.approx(raw, rel=1e-5, abs=1e-5)
    assert body["base_value"] + shap_sum == pytest.approx(raw, rel=1e-5, abs=1e-5)


def test_explain_rejects_underage(client: TestClient) -> None:
    response = client.post("/predict/explain", json=application_payload(age=12))
    assert response.status_code == 422


def test_loan_schema_covers_application_categories() -> None:
    assert set(LoanApplication.model_fields) >= set(APPLICATION_CATEGORIES)
    assert "credit_class" not in LoanApplication.model_fields
