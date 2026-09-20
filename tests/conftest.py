"""API fixtures shared by unit and integration tests. Does not touch models/."""

from __future__ import annotations

from collections.abc import Generator

import pandas as pd
import pytest
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.features.engineering import create_features
from src.model.registry import load_model, load_registry, save_model
from src.model.train import train_baseline
from tests.unit.conftest import VALID_LOAN


def application_payload(**overrides: object) -> dict[str, object]:
    row = {key: value for key, value in VALID_LOAN.items() if key != "credit_class"}
    row["id"] = 7
    row.update(overrides)
    return row


def _toy_featured(n: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n):
        row = dict(VALID_LOAN)
        row["id"] = i + 1
        row["age"] = 20 + (i % 50)
        row["credit_class"] = "bad" if i % 10 < 3 else "good"
        rows.append(row)
    return create_features(pd.DataFrame(rows))


@pytest.fixture(scope="session")
def trained_api(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, CatBoostClassifier]:
    tmp = tmp_path_factory.mktemp("api")
    result = train_baseline(_toy_featured(), params={"iterations": 40, "depth": 3})
    result["version"] = "toy-v1"
    save_model(
        result["model"],
        result=result,
        model_path=tmp / "toy-v1.cbm",
        registry_path=tmp / "registry.json",
    )
    registry = load_registry(tmp / "registry.json")
    model = load_model(registry, models_dir=tmp)
    return registry, model


@pytest.fixture(scope="session")
def client(trained_api: tuple[dict, CatBoostClassifier]) -> Generator[TestClient, None, None]:
    registry, model = trained_api
    with TestClient(create_app(registry=registry, model=model)) as test_client:
        yield test_client
