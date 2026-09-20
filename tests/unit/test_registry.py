"""Artifact tests: save/load roundtrip, probabilities in [0, 1], registry schema.

Trains one tiny CatBoost on a toy frame (module-scoped). The rest of the
suite does not fit a model; this is the exception because the contract
under test is the binary on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import as_catboost_frame, create_features
from src.model.config import artifact_path
from src.model.registry import REGISTRY_KEYS, load_model, load_registry, save_model
from src.model.split import frame_to_xy
from src.model.train import train_baseline
from tests.unit.conftest import VALID_LOAN


def _toy_featured(n: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n):
        row = dict(VALID_LOAN)
        row["id"] = i + 1
        row["age"] = 20 + (i % 50)
        row["credit_class"] = "bad" if i % 10 < 3 else "good"
        rows.append(row)
    return create_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def trained() -> dict:
    return train_baseline(_toy_featured())


def test_artifact_path_uses_version() -> None:
    assert artifact_path("tuned-v1").name == "tuned-v1.cbm"
    assert artifact_path("baseline-v1").name == "baseline-v1.cbm"


def test_save_creates_cbm_and_json(trained: dict, tmp_path: Path) -> None:
    cbm = tmp_path / f"{trained['version']}.cbm"
    registry_path = tmp_path / "registry.json"
    save_model(trained["model"], result=trained, model_path=cbm, registry_path=registry_path)
    assert cbm.exists() and cbm.stat().st_size > 0
    data = load_registry(registry_path)
    assert set(data.keys()) >= REGISTRY_KEYS
    assert data["version"] == "baseline-v1"
    assert data["model_path"] == "baseline-v1.cbm"
    assert data["n_train"] == trained["n_rest"]
    assert data["n_test"] == trained["n_test"]


def test_registry_has_cv_and_test_metrics(trained: dict, tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    save_model(
        trained["model"],
        result=trained,
        model_path=tmp_path / f"{trained['version']}.cbm",
        registry_path=registry_path,
    )
    data = load_registry(registry_path)
    assert "mean" in data["cv_metrics"]["roc_auc"]
    assert "std" in data["cv_metrics"]["roc_auc"]
    assert isinstance(data["test_metrics"]["roc_auc"], float)
    assert "tn" not in data["test_metrics"]


def test_registry_records_threshold_and_cost(trained: dict, tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    save_model(
        trained["model"],
        result=trained,
        model_path=tmp_path / f"{trained['version']}.cbm",
        registry_path=registry_path,
    )
    data = load_registry(registry_path)
    assert data["threshold"] == 0.5
    assert data["cost_ratio"] == {"fn": 5, "fp": 1}


def test_load_model_roundtrip_predicts(trained: dict, tmp_path: Path) -> None:
    save_model(
        trained["model"],
        result=trained,
        model_path=tmp_path / f"{trained['version']}.cbm",
        registry_path=tmp_path / "registry.json",
    )
    loaded = load_model(load_registry(tmp_path / "registry.json"), models_dir=tmp_path)
    x, _, _ = frame_to_xy(_toy_featured(10))
    proba = loaded.predict_proba(as_catboost_frame(x))[:, 1]
    assert proba.shape == (10,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_load_model_missing_cbm_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make train or make tune"):
        load_model({"model_path": "ghost.cbm"}, models_dir=tmp_path)


def test_load_registry_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make train or make tune"):
        load_registry(tmp_path / "missing.json")
